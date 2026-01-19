import torch
import torch.nn as nn
import pytorchvideo.models.x3d as x3d
import logging
import copy

class FlowHallucinationBlock(nn.Module):
    def __init__(self, in_channels):
        super(FlowHallucinationBlock, self).__init__()
        
        # 1. Space-only Conv (1x3x3)
        self.space_conv = nn.Conv3d(
            in_channels, 
            in_channels, 
            kernel_size=(1, 3, 3), 
            padding=(0, 1, 1), 
            bias=False 
        )
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.act1 = nn.SiLU(inplace=True)

        # 2. Time-only Conv (3x1x1)
        self.time_conv = nn.Conv3d(
            in_channels, 
            in_channels, 
            kernel_size=(3, 1, 1), 
            padding=(1, 0, 0), 
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.act_final = nn.SiLU(inplace=True)

    def forward(self, x):
        # KHÔNG DÙNG RESIDUAL (Theo đúng chiến thuật đã chốt)
        
        # Bước 1: Xử lý không gian
        out = self.space_conv(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        # Bước 2: Xử lý thời gian
        out = self.time_conv(out)
        out = self.bn2(out)
        out = self.act_final(out)
        
        return out

# ==================================================================
# BIO-X3D STUDENT (BASELINE: STANDARD CONV)
# ==================================================================
class BioX3D_Student(nn.Module):
    def __init__(self, clip_len=13, feature_dim=192, num_classes=400):
        super(BioX3D_Student, self).__init__()
        
        print(f"🛠️ Khởi tạo BioX3D Student (BASELINE: Standard Conv3d)...")
        
        # 1. Tạo X3D chuẩn
        full_x3d = x3d.create_x3d(
            input_channel=3, 
            input_clip_length=clip_len, 
            model_num_class=num_classes,
            head_activation=None # Quan trọng: Lấy Logits
        )
        modules = list(full_x3d.blocks.children())
        
        # --- NHÁNH RGB (PRIMARY) ---
        self.blocks = nn.Sequential(*modules[:-1]) # Backbone
        self.head = modules[-1]                    # Head
        
        # Kiểm tra và gỡ bỏ Softmax nếu có
        if hasattr(self.head, 'activation'):
            self.head.activation = None

        del full_x3d

        # --- ABLATION: Adapter dùng Conv3d thường ---
        self.flow_adapter = nn.Sequential(
            # 1. Depthwise Conv3d (3x3x3)
            # Giữ nguyên logic Depthwise (groups=feature_dim) để so sánh công bằng về số tham số
            nn.Conv3d(
                in_channels=feature_dim, 
                out_channels=feature_dim, 
                kernel_size=(3, 3, 3), 
                stride=1, 
                padding=(1, 1, 1), 
                groups=feature_dim, # Depthwise
                bias=False
            ),
            nn.BatchNorm3d(feature_dim),
            nn.ReLU(inplace=True),

            # 2. Pointwise Conv3d (1x1x1)
            nn.Conv3d(
                in_channels=feature_dim, 
                out_channels=feature_dim, 
                kernel_size=(1, 1, 1), 
                stride=1, 
                padding=0, 
                bias=False
            ),
            nn.BatchNorm3d(feature_dim),
            nn.ReLU(inplace=True)
        )

        self.hallucinator = FlowHallucinationBlock(feature_dim)
        self.flow_head = copy.deepcopy(self.head)

    def _extract_embedding(self, feat_map, head_module):
        vec = head_module.pool(feat_map)
        vec = head_module.output_pool(vec)
        vec = vec.flatten(1)
        return vec

    def forward(self, x, return_embeddings=False):
        # 1. RGB Path
        rgb_feat_map = self.blocks(x) 
        rgb_logits = self.head(rgb_feat_map)
        
        # 2. Flow Path
        flow_feat_pre = self.flow_adapter(rgb_feat_map)
        flow_hallucinated = self.hallucinator(flow_feat_pre)
        flow_logits = self.flow_head(flow_hallucinated)
        
        # --- Output Logic ---
        if return_embeddings:
            rgb_embed = self._extract_embedding(rgb_feat_map, self.head)
            flow_embed = self._extract_embedding(flow_hallucinated, self.flow_head)
            return rgb_logits, flow_logits, rgb_feat_map, flow_hallucinated, rgb_embed, flow_embed
        else:
            return rgb_logits, flow_logits, rgb_feat_map, flow_hallucinated
            
    def load_pretrained_weights(self, rgb_path, flow_teacher_path=None):
        # --- 1. LOAD RGB ---
        if rgb_path:
            logging.info(f"📥 Loading RGB weights: {rgb_path}")
            try:
                ckpt = torch.load(rgb_path, map_location='cpu')
                state = ckpt['model_state'] if 'model_state' in ckpt else (ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
                
                rgb_dict = {}
                for k, v in state.items():
                    if k.startswith("blocks.5"):
                        new_key = k.replace("blocks.5", "head")
                        rgb_dict[new_key] = v
                    elif k.startswith("blocks"):
                        rgb_dict[k] = v
                
                msg = self.load_state_dict(rgb_dict, strict=False)
                logging.info(f"✅ RGB Backbone & Head Loaded: {msg}")
            except Exception as e:
                logging.error(f"❌ Failed to load RGB weights: {e}")

        # --- 2. LOAD FLOW HEAD ---
        if flow_teacher_path:
            logging.info(f"📥 Loading FLOW Head from Teacher: {flow_teacher_path}")
            try:
                t_ckpt = torch.load(flow_teacher_path, map_location='cpu')
                t_state = t_ckpt['model_state'] if 'model_state' in t_ckpt else (t_ckpt['state_dict'] if 'state_dict' in t_ckpt else t_ckpt)
                
                flow_head_dict = {}
                count = 0
                for k, v in t_state.items():
                    if "blocks.5" in k or "head" in k: 
                        if "blocks.5" in k:
                            new_key = k.replace("blocks.5", "flow_head")
                        elif "head" in k:
                            new_key = k.replace("head", "flow_head")
                        flow_head_dict[new_key] = v
                        count += 1
                
                if count > 0:
                    self.load_state_dict(flow_head_dict, strict=False)
                    logging.info(f"✅ Flow Head Loaded ({count} params)!")
                else:
                    logging.warning("⚠️ Không tìm thấy layer Head phù hợp.")
                    
            except Exception as e:
                logging.error(f"❌ Failed to load Flow Head: {e}")


# ==================================================================
# 3. TEST
# ==================================================================
# ==================================================================
# 3. ANALYSIS & TEST (UPDATED FOR PAPER MATCHING)
# ==================================================================
if __name__ == "__main__":
    try:
        from thop import profile
    except ImportError:
        print("⚠️ Cần cài đặt thop: pip install thop")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- CẤU HÌNH INPUT ĐỂ KHỚP PAPER X3D-S ---
    # Paper X3D-S dùng size 160x160 cho input 13 frames
    # Nếu bạn dùng 224x224 thì GFLOPS sẽ cao hơn (khoảng gấp 1.9 lần)
    FRAMES = 13
    IMG_SIZE = 182  # Đổi về 160 nếu muốn khớp con số ~1.96 GFLOPS của paper
    #IMG_SIZE = 224 # Dùng 224 nếu project của bạn chạy 224 (GFLOPS sẽ to hơn)
    
    BATCH = 1
    
    print("\n" + "="*60)
    print(f"📊 BIO-X3D EFFICIENCY REPORT (Input: {IMG_SIZE}x{IMG_SIZE}, {FRAMES} frames)")
    print("="*60)

    try:
        # 1. Init Model
        model = BioX3D_Student(clip_len=FRAMES).to(device)
        model.eval()
        dummy_input = torch.randn(BATCH, 3, FRAMES, IMG_SIZE, IMG_SIZE).to(device)

        # ---------------------------------------------------------
        # A. TÍNH PARAMS (Chi tiết từng phần)
        # ---------------------------------------------------------
        total_params = sum(p.numel() for p in model.parameters())
        rgb_head_params = sum(p.numel() for p in model.head.parameters())
        flow_head_params = sum(p.numel() for p in model.flow_head.parameters())
        
        # Params cốt lõi (Bỏ 2 head phân loại)
        backbone_no_head_params = total_params - rgb_head_params - flow_head_params

        # ---------------------------------------------------------
        # B. TÍNH GFLOPS (Theo chuẩn Paper: GFLOPS = G-MACs)
        # ---------------------------------------------------------
        print("🔄 Profiling GFLOPS...")
        
        # 1. Tính toàn bộ Model
        macs_total, _ = profile(model, inputs=(dummy_input, ), verbose=False)
        
        # [QUAN TRỌNG] Paper X3D báo cáo GFLOPS thực chất là G-MACs (Multiply-Adds)
        # Nên ta KHÔNG nhân 2 ở đây.
        gflops_total = macs_total / 1e9  
        
        # 2. Tính riêng Backbone (Phần Feature Extractor)
        # (Chỉ chạy qua blocks, không chạy qua head)
        macs_backbone, _ = profile(model.blocks, inputs=(dummy_input, ), verbose=False)
        gflops_backbone = macs_backbone / 1e9

        # ---------------------------------------------------------
        # C. KẾT QUẢ
        # ---------------------------------------------------------
        print("-" * 60)
        print(f"{'METRIC':<30} | {'VALUE':<20}")
        print("-" * 60)
        
        # 1. Params
        print(f"{'Params (Full Model)':<30} | {total_params / 1e6:.2f} M")
        print(f"{'Params (Backbone Only)':<30} | {backbone_no_head_params / 1e6:.2f} M")
        print(f"{' - RGB Head':<30} | {rgb_head_params / 1e6:.2f} M")
        print(f"{' - Flow Head':<30} | {flow_head_params / 1e6:.2f} M")
        
        print("-" * 60)
        
        # 2. GFLOPS
        print(f"{'GFLOPS (Full Model)':<30} | {gflops_total:.3f} G")
        print(f"{'GFLOPS (Backbone Only)':<30} | {gflops_backbone:.3f} G")
        
        # 3. So sánh với Paper
        if IMG_SIZE == 160:
            print("-" * 60)
            print(f"📝 Note: Paper X3D-S report ~1.96 GFLOPS.")
            print(f"   Model của bạn: {gflops_backbone:.3f} G (Backbone) + Head/Adapter overhead.")
        elif IMG_SIZE == 224:
            print("-" * 60)
            print(f"📝 Note: Bạn đang chạy size 224x224.")
            print(f"   GFLOPS sẽ cao hơn paper (160x160) khoảng {(224/160)**2:.1f} lần.")

        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")