import torch
import torch.nn as nn
import logging
import pytorchvideo.models.x3d as x3d
from torchvision.models.optical_flow import raft_large

# --- TỐI ƯU CHO A100/GPU ĐỜI MỚI ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def run_raft_in_chunks(raft_model, img1, img2, chunk_size=64):
    """
    Chia nhỏ batch để chạy RAFT, tránh OOM trên GPU.
    Input: img1, img2 [N, 3, H, W] giá trị 0-255
    Output: Flow [N, 2, H, W]
    """
    total_frames = img1.shape[0]
    flow_list = []
    
    # Tắt gradient hoàn toàn để tiết kiệm VRAM
    with torch.no_grad():
        for i in range(0, total_frames, chunk_size):
            # .contiguous() rất quan trọng để tối ưu bộ nhớ
            i1_chunk = img1[i : i + chunk_size].contiguous()
            i2_chunk = img2[i : i + chunk_size].contiguous()
            
            # Chuẩn hóa về [-1, 1] theo yêu cầu của RAFT
            # Giả định input đầu vào đang là [0, 255]
            i1_norm = (i1_chunk / 255.0) * 2.0 - 1.0
            i2_norm = (i2_chunk / 255.0) * 2.0 - 1.0
            
            # Forward RAFT
            flow_preds = raft_model(i1_norm, i2_norm)
            flow_final = flow_preds[-1] # Lấy kết quả tốt nhất
            
            flow_list.append(flow_final)
            
    return torch.cat(flow_list, dim=0)

class TeacherPipeline(nn.Module):
    def __init__(self, raft_weights_path, x3d_flow_weights_path, device='cuda'):
        super().__init__()
        self.device = device
        logging.info(f"--- Khởi tạo Teacher Pipeline trên {device} ---")
        
        # 1. Load RAFT (Optical Flow Extractor)
        self.raft = raft_large(weights=None).to(device)
        if raft_weights_path:
            state = torch.load(raft_weights_path, map_location=device)
            self.raft.load_state_dict(state)
            logging.info(f"✅ RAFT Loaded: {raft_weights_path}")
        self.raft.eval()
        
        # 2. Load X3D-Flow (Feature Extractor)
        # Input channel = 2 (x, y flow)
        self.x3d_flow = x3d.create_x3d(
            input_channel=2, 
            input_clip_length=13, 
            model_num_class=400
        ).to(device)
        
        if x3d_flow_weights_path:
            # Load weight X3D Flow (Đã train ở bước chuẩn bị)
            state = torch.load(x3d_flow_weights_path, map_location=device)
            # Xử lý key nếu cần (bỏ module. prefix nếu train DDP)
            new_state = {k.replace("module.", ""): v for k, v in state.items()}
            # Load lỏng (strict=False) để tránh lỗi head nếu số class lệch
            self.x3d_flow.load_state_dict(new_state, strict=False)
            logging.info(f"✅ X3D-Flow Teacher Loaded: {x3d_flow_weights_path}")
            
        self.x3d_flow.eval()
        
        # --- [MODIFIED 1] Tách Backbone và Head ---
        # Cấu trúc X3D: blocks[0]..blocks[5]. blocks[5] là head.
        modules = list(self.x3d_flow.blocks.children())
        
        self.backbone = nn.Sequential(*modules[:-1]) # Layer 0->4
        self.head = modules[-1]                      # Layer 5 (Chứa Pool, Dropout, Proj)
        
        # Freeze toàn bộ teacher
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def get_teacher_features(self, rgb_clip_0_255, return_embedding=False):
        """
        Input: 
            rgb_clip [B, 3, T, H, W] - Giá trị pixel [0, 255]
            return_embedding (bool): Nếu True trả về vector [B, C], nếu False trả về map [B, C, T, H, W]
        Output: 
            Tensor feature
        """
        b, c, t, h, w = rgb_clip_0_255.shape
        
        # --- BƯỚC 1: CHUẨN BỊ INPUT CHO RAFT ---
        img1 = rgb_clip_0_255[:, :, :-1, :, :].permute(0, 2, 1, 3, 4).reshape(-1, 3, h, w)
        img2 = rgb_clip_0_255[:, :, 1:, :, :].permute(0, 2, 1, 3, 4).reshape(-1, 3, h, w)
        
        # --- BƯỚC 2: TÍNH OPTICAL FLOW ---
        flow_flat = run_raft_in_chunks(self.raft, img1, img2, chunk_size=64)
        
        # --- BƯỚC 3: CHUẨN BỊ INPUT CHO X3D ---
        flow_clip = flow_flat.view(b, t-1, 2, h, w).permute(0, 2, 1, 3, 4)
        
        # Duplicate frame cuối
        last_flow = flow_clip[:, :, -1:, :, :]
        flow_clip_13 = torch.cat([flow_clip, last_flow], dim=2) # [B, 2, 13, H, W]
        
        # Chuẩn hóa Flow
        flow_clip_13 = torch.clamp(flow_clip_13 / 20.0, -1.0, 1.0)
        
        # --- BƯỚC 4: TRÍCH XUẤT ĐẶC TRƯNG (Backbone) ---
        # Output: [B, 192, 13, 7, 7]
        feat_map = self.backbone(flow_clip_13)
        
        # --- [MODIFIED 2] Xử lý Embedding nếu cần ---
        if return_embedding:
            # Tái hiện logic pooling của X3D Head
            # 1. Spatial Pool
            vec = self.head.pool(feat_map) 
            # 2. Output Pool (Temporal - thường là AdaptiveAvgPool3d(1))
            vec = self.head.output_pool(vec)
            # 3. Flatten [B, C, 1, 1, 1] -> [B, C]
            return vec.flatten(1)
        
        return feat_map

# --- Test nhanh ---
if __name__ == "__main__":
    print("Testing Teacher Pipeline...")
    try:
        teacher = TeacherPipeline(None, None) # Dummy init
        
        # Tạo dummy RGB clip
        dummy_input = torch.rand(2, 3, 13, 224, 224).cuda() * 255
        
        # Test 1: Feature Map
        feat = teacher.get_teacher_features(dummy_input, return_embedding=False)
        print(f"Feature Map Shape: {feat.shape}") # Expect: (2, 192, 13, 7, 7)
        
        # Test 2: Embedding
        embed = teacher.get_teacher_features(dummy_input, return_embedding=True)
        print(f"Embedding Shape: {embed.shape}")   # Expect: (2, 2048) hoặc (2, 192) tuỳ model X3D cụ thể
        
    except Exception as e:
        print(f"Lỗi (có thể do thiếu file weights thật): {e}")