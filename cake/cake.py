import torch
import torch.nn as nn
import pytorchvideo.models.x3d as x3d
import logging
import copy
from odconv3d import ODConv3d

class FlowHallucinationBlock(nn.Module):
    def __init__(self, in_channels):
        super(FlowHallucinationBlock, self).__init__()
        self.time_odconv = ODConv3d(in_planes=in_channels, out_planes=in_channels, 
                                    kernel_size=(3, 1, 1), padding=(1, 0, 0), 
                                    reduction=0.0625, kernel_num=4)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.space_odconv = ODConv3d(in_planes=in_channels, out_planes=in_channels, 
                                     kernel_size=(1, 3, 3), padding=(0, 1, 1), 
                                     reduction=0.0625, kernel_num=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.act_final = nn.SiLU(inplace=True)

    def forward(self, x):
        #residual = x 
        out = self.time_odconv(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.space_odconv(out)
        out = self.bn2(out)
        #out += residual 
        out = self.act_final(out)
        return out

# ==================================================================
# BIO-X3D STUDENT (Updated with Output Flag)
# ==================================================================
class BioX3D_Student(nn.Module):
    def __init__(self, clip_len=13, feature_dim=192, num_classes=400):
        super(BioX3D_Student, self).__init__()
        
        print(f"🛠️ Khởi tạo BioX3D Student...")
        
        # 1. Tạo X3D chuẩn
        full_x3d = x3d.create_x3d(
            input_channel=3, 
            input_clip_length=clip_len, 
            model_num_class=num_classes,
            head_activation=None
        )
        modules = list(full_x3d.blocks.children())
        
        # --- NHÁNH RGB (PRIMARY) ---
        self.blocks = nn.Sequential(*modules[:-1]) # Backbone trả về (B, 192, T, H, W)
        self.head = modules[-1]                    # Head
        del full_x3d


        # self.flow_adapter = nn.Sequential(
        #     nn.Conv3d(feature_dim, feature_dim, kernel_size=1, bias=False),
        #     nn.BatchNorm3d(feature_dim),
        #     nn.ReLU(inplace=True)
        # )

        # self.flow_adapter = nn.Sequential(
        #     nn.Conv3d(feature_dim, feature_dim // 4, kernel_size=1, bias=False),
        #     nn.BatchNorm3d(feature_dim // 4),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(feature_dim // 4, feature_dim, kernel_size=1, bias=False),
        #     nn.BatchNorm3d(feature_dim),
        #     nn.ReLU(inplace=True)
        # )
        
        self.flow_adapter = nn.Sequential(
            # 1. Depthwise ODConv3d: Thu thập thông tin chuyển động (Context)
            # - Kernel (3,3,3): Giúp pixel nhìn được lân cận không gian và thời gian (frame trước/sau).
            # - groups=feature_dim: Chìa khóa để làm nó nhẹ (Depthwise).
            ODConv3d(
                in_planes=feature_dim, 
                out_planes=feature_dim, 
                kernel_size=(3, 3, 3), 
                stride=1, 
                padding=(1, 1, 1), 
                reduction=0.0625, 
                kernel_num=1,
                groups=feature_dim # <--- QUAN TRỌNG: Biến nó thành Depthwise
            ),
            nn.BatchNorm3d(feature_dim),
            nn.ReLU(inplace=True),

            # 2. Pointwise ODConv3d: Trộn thông tin giữa các kênh (Channel Mixing)
            # Sau khi mỗi kênh đã tự nhìn hàng xóm (bước 1), bước này giúp các kênh giao tiếp với nhau.
            ODConv3d(
                in_planes=feature_dim, 
                out_planes=feature_dim, 
                kernel_size=(1, 1, 1), 
                stride=1, 
                padding=0, 
                reduction=0.0625, 
                kernel_num=1
            ),
            nn.BatchNorm3d(feature_dim),
            nn.ReLU(inplace=True)
        )

        self.hallucinator = FlowHallucinationBlock(feature_dim)
        self.flow_head = copy.deepcopy(self.head)

    def _extract_embedding(self, feat_map, head_module):
        """Helper để lấy vector 2048 chiều"""
        vec = head_module.pool(feat_map)
        vec = head_module.output_pool(vec)
        vec = vec.flatten(1)
        return vec

    def forward(self, x, return_embeddings=False):
        """
        Args:
            x: Input tensor (B, 3, T, H, W)
            return_embeddings (bool): 
                - False (Default): Trả về 4 output (Logits, FeatMap)
                - True: Trả về 6 output (Logits, FeatMap, Embeddings 2048)
        """
        # 1. RGB Path
        rgb_feat_map = self.blocks(x) 
        rgb_logits = self.head(rgb_feat_map)
        
        # 2. Flow Path
        flow_feat_pre = self.flow_adapter(rgb_feat_map)
        flow_hallucinated = self.hallucinator(flow_feat_pre)
        flow_logits = self.flow_head(flow_hallucinated)
        
        # --- Logic trả về ---
        if return_embeddings:
            # Tính thêm embeddings 2048 chiều
            rgb_embed = self._extract_embedding(rgb_feat_map, self.head)
            flow_embed = self._extract_embedding(flow_hallucinated, self.flow_head)
            
            # Trả về 6 giá trị
            return rgb_logits, flow_logits, rgb_feat_map, flow_hallucinated, rgb_embed, flow_embed
        else:
            # Mặc định: Trả về 4 giá trị
            return rgb_logits, flow_logits, rgb_feat_map, flow_hallucinated
            
    def load_pretrained_weights(self, rgb_path, flow_teacher_path=None):
        """
        Hàm load weight thông minh:
        1. Load RGB Weights vào self.blocks và self.head
        2. Load Flow Teacher Weights vào self.flow_head (nếu có)
        """
        # --- 1. LOAD RGB (STUDENT PRETRAINED) ---
        if rgb_path:
            logging.info(f"📥 Loading RGB weights: {rgb_path}")
            try:
                ckpt = torch.load(rgb_path, map_location='cpu')
                # Lấy state_dict chuẩn
                if 'model_state' in ckpt: state = ckpt['model_state']
                elif 'state_dict' in ckpt: state = ckpt['state_dict']
                else: state = ckpt
                
                rgb_dict = {}
                for k, v in state.items():
                    # Map Head gốc (blocks.5) -> self.head
                    if k.startswith("blocks.5"):
                        new_key = k.replace("blocks.5", "head")
                        rgb_dict[new_key] = v
                    # Map Backbone (blocks.0-4) -> self.blocks
                    elif k.startswith("blocks"):
                        rgb_dict[k] = v
                    # Bỏ qua các key không liên quan
                
                msg = self.load_state_dict(rgb_dict, strict=False)
                logging.info(f"✅ RGB Backbone & Head Loaded: {msg}")
            except Exception as e:
                logging.error(f"❌ Failed to load RGB weights: {e}")

        # --- 2. LOAD FLOW HEAD (TEACHER WEIGHTS) ---
        if flow_teacher_path:
            logging.info(f"📥 Loading FLOW Head from Teacher: {flow_teacher_path}")
            try:
                t_ckpt = torch.load(flow_teacher_path, map_location='cpu')
                t_state = t_ckpt['model_state'] if 'model_state' in t_ckpt else (t_ckpt['state_dict'] if 'state_dict' in t_ckpt else t_ckpt)
                
                # Lọc lấy weight của Head từ Teacher (blocks.5) và nhét vào flow_head
                flow_head_dict = {}
                count = 0
                for k, v in t_state.items():
                    # Tìm layer thuộc Head trong file teacher
                    if "blocks.5" in k or "head" in k: 
                        # Map sang tên biến của Student: 'flow_head'
                        # Logic replace này cần linh hoạt tùy tên trong checkpoint teacher
                        if "blocks.5" in k:
                            new_key = k.replace("blocks.5", "flow_head")
                        elif "head" in k:
                            new_key = k.replace("head", "flow_head")
                        
                        # Chỉ lấy những layer khớp tên với self.flow_head
                        # (Ví dụ: flow_head.proj.weight)
                        flow_head_dict[new_key] = v
                        count += 1
                
                if count > 0:
                    msg = self.load_state_dict(flow_head_dict, strict=False)
                    logging.info(f"✅ Flow Head Loaded ({count} params)! Student đã kế thừa tri thức phân loại của Teacher.")
                else:
                    logging.warning("⚠️ Không tìm thấy layer Head phù hợp trong Flow Teacher Checkpoint.")
                    
            except Exception as e:
                logging.error(f"❌ Failed to load Flow Head: {e}")


# ==================================================================
# 3. TEST
# ==================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH = 2
    FRAMES = 13
    
    try:
        model = BioX3D_Student(clip_len=FRAMES).to(device)
        print("\n✅ Model created!")
        
        # Test input
        dummy = torch.randn(BATCH, 3, FRAMES, 224, 224).to(device)
        rgb_logits, flow_logits, rgb_feat, flow_feat = model(dummy)
        
        print(f"\nShape Check:")
        print(f"RGB Logits: {rgb_logits.shape} (Expect {BATCH}, 400)")
        print(f"Flow Logits: {flow_logits.shape} (Expect {BATCH}, 400)")
        print(f"Flow Feat: {flow_feat.shape} (Expect {BATCH}, 192, {FRAMES}, 7, 7)")
        
        print(f"\n🧪 Sanity Check (Logits vs Softmax):")
        sample_output = rgb_logits[0] # Lấy mẫu đầu tiên
        print(f"   - Min val: {sample_output.min().item():.4f}")
        print(f"   - Max val: {sample_output.max().item():.4f}")
        print(f"   - Sum val: {sample_output.sum().item():.4f}")
        
        if abs(sample_output.sum().item() - 1.0) > 0.1:
             print("   ✅ Kết luận: Output là LOGITS (Vì tổng != 1)")
        else:
             print("   ⚠️ Kết luận: Output có thể là SOFTMAX (Vì tổng ~ 1)")
        
        if rgb_logits.shape == flow_logits.shape == (BATCH, 400):
            print("\n🎉 Verification Passed!")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Error: {e}")