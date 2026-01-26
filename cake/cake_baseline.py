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
        sample_output = rgb_logits[0]
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
    model.to(device)
    model.eval() # Chuyển sang chế độ eval

    # 3. Tạo dữ liệu giả lập (B, C, T, H, W)
    # X3D yêu cầu input 3 kênh màu (RGB)
    dummy_input = torch.randn(BATCH_SIZE, 3, CLIP_LEN, RESOLUTION, RESOLUTION).to(device)

    print(f"\n🚀 Đang chạy inference với input shape: {dummy_input.shape}")

    with torch.no_grad():
        # 4. Forward pass qua model
        # Sử dụng return_embeddings=True để lấy các feature map trước head
        rgb_logits, flow_logits, rgb_feat_map, flow_feat_map, rgb_embed_internal, flow_embed_internal = model(
            dummy_input, return_embeddings=True
        )

    print("-" * 30)
    print(f"✅ Feature map RGB (trước head): {rgb_feat_map.shape}")
    print(f"✅ Feature map Flow (trước head): {flow_feat_map.shape}")

    # 5. Áp dụng các lớp Pool từ Head vào các feature map vừa lấy được
    # Lưu ý: Head của X3D thường có .pool và .output_pool
    
    def manual_pooling(feat, head_module):
        # Bước A: Spatial-Temporal Pooling (thường là AdaptiveAvgPool3d)
        pooled = head_module.pool(feat)
        # Bước B: Global pooling cuối cùng trước khi flatten
        pooled = head_module.output_pool(pooled)
        # Bước C: Flatten để tạo vector embedding
        embedding = pooled.flatten(1)
        return embedding

    # Thực hiện pooling cho nhánh RGB
    rgb_pooled_manual = manual_pooling(rgb_feat_map, model.head)
    
    # Thực hiện pooling cho nhánh Flow
    flow_pooled_manual = manual_pooling(flow_feat_map, model.flow_head)

    print("-" * 30)
    print(f"📊 Kết quả pooling thủ công:")
    print(f"   -> RGB Pooled shape: {rgb_pooled_manual.shape}")
    print(f"   -> Flow Pooled shape: {flow_pooled_manual.shape}")

    # 6. Kiểm chứng (So sánh với embedding mà model tự tính bên trong)
    diff_rgb = torch.norm(rgb_pooled_manual - rgb_embed_internal)
    diff_flow = torch.norm(flow_pooled_manual - flow_embed_internal)

    print("-" * 30)
    print(f"🔍 Kiểm tra sai số (so với internal embedding):")
    print(f"   -> Sai số RGB: {diff_rgb.item():.6f}")
    print(f"   -> Sai số Flow: {diff_flow.item():.6f}")

    if diff_rgb < 1e-5 and diff_flow < 1e-5:
        print("\n✨ KẾT QUẢ TRÙNG KHỚP! Bạn đã lấy và pooling feature thành công.")
    else:
        print("\n⚠️ Có sự khác biệt nhỏ, hãy kiểm tra lại cấu trúc head.")