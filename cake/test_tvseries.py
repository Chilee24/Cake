import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import os
import sys
import argparse
import decord
from tqdm import tqdm
import logging
import json
import glob
from torchvision.models.optical_flow import raft_large
import pytorchvideo.models.x3d as x3d

# --- CONFIG ---
decord.bridge.set_bridge('torch')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Tối ưu cho GPU đời mới
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==============================================================================
# 1. TEACHER PIPELINE (RAFT + X3D Flow)
# ==============================================================================

def run_raft_in_chunks(raft_model, img1, img2, chunk_size=32):
    """Chia nhỏ batch để chạy RAFT, tránh OOM trên GPU."""
    total_frames = img1.shape[0]
    flow_list = []
    
    with torch.no_grad():
        for i in range(0, total_frames, chunk_size):
            i1_chunk = img1[i : i + chunk_size].contiguous()
            i2_chunk = img2[i : i + chunk_size].contiguous()
            
            # RAFT cần input [-1, 1]
            i1_norm = (i1_chunk / 255.0) * 2.0 - 1.0
            i2_norm = (i2_chunk / 255.0) * 2.0 - 1.0
            
            flow_preds = raft_model(i1_norm, i2_norm)
            flow_final = flow_preds[-1]
            flow_list.append(flow_final)
            
    return torch.cat(flow_list, dim=0)

class TeacherPipeline(nn.Module):
    def __init__(self, raft_weights_path, x3d_flow_weights_path, device='cuda'):
        super().__init__()
        self.device = device
        logging.info(f"--- Khởi tạo Teacher Pipeline trên {device} ---")
        
        # 1. Load RAFT
        self.raft = raft_large(weights=None).to(device)
        if raft_weights_path and os.path.exists(raft_weights_path):
            state = torch.load(raft_weights_path, map_location=device)
            self.raft.load_state_dict(state)
            logging.info(f"✅ RAFT Loaded: {raft_weights_path}")
        else:
            logging.warning("⚠️ Using RAFT with random weights (Check path!)")
        self.raft.eval()
        
        # 2. Load X3D-Flow
        self.x3d_flow = x3d.create_x3d(
            input_channel=2, 
            input_clip_length=13, 
            model_num_class=400
        ).to(device)
        
        if x3d_flow_weights_path and os.path.exists(x3d_flow_weights_path):
            state = torch.load(x3d_flow_weights_path, map_location=device)
            if 'state_dict' in state: state = state['state_dict']
            elif 'model_state' in state: state = state['model_state']
            
            new_state = {k.replace("module.", ""): v for k, v in state.items()}
            self.x3d_flow.load_state_dict(new_state, strict=False)
            logging.info(f"✅ X3D-Flow Teacher Loaded: {x3d_flow_weights_path}")
        else:
            logging.warning("⚠️ Using X3D-Flow with random weights (Check path!)")
            
        self.x3d_flow.eval()
        
        # Tách Backbone và Head để lấy Embedding
        modules = list(self.x3d_flow.blocks.children())
        self.backbone = nn.Sequential(*modules[:-1]) 
        self.head = modules[-1]                      
        
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def get_teacher_features(self, rgb_clip_0_255, return_embedding=True):
        """
        Input: rgb_clip [B, 3, T, H, W] (0-255)
        Output: Embedding [B, 2048] (nếu return_embedding=True)
        """
        b, c, t, h, w = rgb_clip_0_255.shape
        
        # 1. Prepare Inputs for RAFT
        img1 = rgb_clip_0_255[:, :, :-1, :, :].permute(0, 2, 1, 3, 4).reshape(-1, 3, h, w)
        img2 = rgb_clip_0_255[:, :, 1:, :, :].permute(0, 2, 1, 3, 4).reshape(-1, 3, h, w)
        
        # 2. Compute Flow
        # Chunk size nhỏ (16-32) để RAFT không ăn hết VRAM
        flow_flat = run_raft_in_chunks(self.raft, img1, img2, chunk_size=32) 
        
        # 3. Prepare Inputs for X3D
        flow_clip = flow_flat.view(b, t-1, 2, h, w).permute(0, 2, 1, 3, 4)
        last_flow = flow_clip[:, :, -1:, :, :]
        flow_clip_13 = torch.cat([flow_clip, last_flow], dim=2) # Pad cho đủ 13 frame
        
        # Chuẩn hóa Flow cho X3D (theo Kinetics stats)
        flow_clip_13 = torch.clamp(flow_clip_13 / 20.0, -1.0, 1.0)
        
        # 4. Extract Features
        feat_map = self.backbone(flow_clip_13)
        
        if return_embedding:
            vec = self.head.pool(feat_map) 
            vec = self.head.output_pool(vec)
            return vec.flatten(1) # [B, 2048]
        
        return feat_map

# ==============================================================================
# 2. DATA UTILS (TVSeries)
# ==============================================================================

def find_video_paths(video_dir, video_names):
    mapping = {}
    # Quét đệ quy tìm video
    all_files = glob.glob(os.path.join(video_dir, "**", "*.*"), recursive=True)
    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        if ext in ['.mp4', '.avi', '.mkv', '.webm', '.mov']:
            basename = os.path.basename(f).rsplit('.', 1)[0]
            mapping[basename] = f
            
    found_items = []
    for name in video_names:
        if name in mapping:
            found_items.append({'path': mapping[name], 'name': name})
        else:
            logging.warning(f"Video not found: {name}")
    return found_items

def get_clip_indices(total_frames, orig_fps, target_fps=24, clip_len=13, stride=6):
    # Resample về 24 FPS
    if abs(orig_fps - target_fps) > 0.5:
        scale = target_fps / orig_fps
        new_total = int(total_frames * scale)
        frame_idxs = np.linspace(0, total_frames - 1, new_total).astype(int)
    else:
        frame_idxs = np.arange(total_frames)

    sample_rate = 2
    window_span = (clip_len - 1) * sample_rate + 1

    # Padding frame cuối nếu video ngắn
    if len(frame_idxs) < window_span:
        pad = [frame_idxs[-1]] * (window_span - len(frame_idxs))
        frame_idxs = np.concatenate([frame_idxs, pad])

    all_clips_indices = []
    
    # Sliding window
    for start in range(0, len(frame_idxs) - window_span + 1, stride):
        indices = frame_idxs[start : start + window_span : sample_rate]
        all_clips_indices.append(indices)
        
    return all_clips_indices

def process_video_flow(teacher_model, vid_info, args, device):
    """Xử lý 1 video: Đọc -> Flow -> X3D Feature"""
    try:
        vr = decord.VideoReader(vid_info['path'])
        orig_fps = vr.get_avg_fps()
        total_frames = len(vr)
    except Exception as e:
        logging.error(f"Error reading {vid_info['name']}: {e}")
        return None

    indices = get_clip_indices(total_frames, orig_fps, args.target_fps, args.clip_len, args.stride)
    if not indices: return None

    all_feats = []
    
    # Batch Processing
    for i in range(0, len(indices), args.batch_size):
        batch_idx = indices[i : i + args.batch_size]
        flat_idx = np.concatenate(batch_idx)
        
        # Đọc frames
        batch_frames = vr.get_batch(flat_idx)
        if hasattr(batch_frames, 'asnumpy'):
            frames = batch_frames.asnumpy()
        else:
            frames = batch_frames.cpu().numpy()
        
        # Transform: (N, H, W, 3) -> (N, 3, H, W)
        # QUAN TRỌNG: Giữ nguyên giá trị 0-255, KHÔNG Normalize Mean/Std ở đây
        # TeacherPipeline sẽ tự xử lý việc đó
        batch_t = torch.from_numpy(frames).float() # [N, H, W, 3]
        batch_t = batch_t.permute(0, 3, 1, 2)      # [N, 3, H, W]
        
        # Reshape thành Batch Clips
        B_curr = len(batch_idx)
        batch_t = batch_t.view(B_curr, args.clip_len, 3, frames.shape[1], frames.shape[2])
        batch_t = batch_t.permute(0, 2, 1, 3, 4) # [B, 3, T, H, W]
        
        # Resize & Crop (Vẫn trên 0-255)
        # Gom Batch và Time để resize 1 lần cho nhanh
        B, C, T_dim, H, W = batch_t.shape
        batch_t = batch_t.reshape(B * T_dim, C, H, W)
        batch_t = T.Resize(args.resize_short, antialias=True)(batch_t)
        batch_t = T.CenterCrop(args.crop_size)(batch_t)
        
        # Reshape lại
        _, _, new_H, new_W = batch_t.shape
        batch_t = batch_t.reshape(B, C, T_dim, new_H, new_W) # [B, 3, T, H, W]
        
        batch_t = batch_t.to(device)
        
        # --- FEATURE EXTRACTION ---
        with torch.no_grad():
            # Input vào Teacher là 0-255
            feats = teacher_model.get_teacher_features(batch_t, return_embedding=True)
            all_feats.append(feats.cpu().numpy())

    if len(all_feats) == 0: return None
    return np.concatenate(all_feats)

# ==============================================================================
# 3. MAIN LOOP
# ==============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Output Dir
    out_dir = os.path.join(args.output_dir, 'flow_teacher_features')
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Init Teacher
    teacher = TeacherPipeline(
        raft_weights_path=args.raft_weights, 
        x3d_flow_weights_path=args.x3d_weights, 
        device=device
    )
    
    # 3. Load Video List from JSON
    with open(args.json_file, 'r') as f:
        full_json = json.load(f)
        json_data = full_json["TVSERIES"] # Hoặc THUMOS tùy dataset
    
    # Gom danh sách video cần xử lý
    # (Có thể xử lý cả train và test hoặc lọc theo args)
    target_sessions = json_data["train_session_set"] + json_data["test_session_set"]
    
    # Tìm file
    print(f"🔍 Searching for {len(target_sessions)} videos...")
    vid_infos = find_video_paths(args.video_dir, target_sessions)
    print(f"✅ Found {len(vid_infos)} videos. Starting extraction...")
    
    # 4. Processing Loop
    for v_info in tqdm(vid_infos):
        save_path = os.path.join(out_dir, f"{v_info['name']}.npy")
        
        if os.path.exists(save_path):
            continue # Skip if exists
            
        feats = process_video_flow(teacher, v_info, args, device)
        
        if feats is not None:
            np.save(save_path, feats)
            
    print(f"\n🎉 Completed! Features saved to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--video_dir', type=str, required=True, help='Thư mục chứa video')
    parser.add_argument('--json_file', type=str, required=True, help='File json list video')
    parser.add_argument('--output_dir', type=str, default='./output')
    
    # Weights
    parser.add_argument('--raft_weights', type=str, default=None, help='Path to RAFT weights')
    parser.add_argument('--x3d_weights', type=str, default=None, help='Path to X3D Flow weights')
    
    # Params
    parser.add_argument('--target_fps', type=int, default=24)
    parser.add_argument('--clip_len', type=int, default=13)
    parser.add_argument('--stride', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size cho clip (RAFT nặng nên để thấp)')
    parser.add_argument('--resize_short', type=int, default=224)
    parser.add_argument('--crop_size', type=int, default=224)
    
    args = parser.parse_args()
    main(args)

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# import numpy as np
# import os
# import sys
# import argparse
# import decord
# from tqdm import tqdm
# import logging
# import json
# import glob
# from collections import defaultdict

# # --- CONFIG ---
# decord.bridge.set_bridge('torch')
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# from cake import BioX3D_Student  

# # ==============================================================================
# # 1. UTILS (Xử lý Tensor & Tìm kiếm Video)
# # ==============================================================================
# class X3D_Normalizer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1)
#         self.std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1)

#     def forward(self, x):
#         return (x / 255.0 - self.mean.to(x.device)) / self.std.to(x.device)

# def set_odconv_temperature(model, temperature=1.0):
#     for m in model.modules():
#         if hasattr(m, 'update_temperature'):
#             m.update_temperature(temperature)

# def find_video_paths(video_dir, video_names):
#     """Tìm đường dẫn video trong 1 thư mục duy nhất dựa trên danh sách tên trong JSON"""
#     mapping = {}
#     all_files = glob.glob(os.path.join(video_dir, "**", "*.*"), recursive=True)
#     for f in all_files:
#         ext = os.path.splitext(f)[1].lower()
#         if ext in ['.mp4', '.avi', '.mkv', '.webm', '.mov']:
#             basename = os.path.basename(f).rsplit('.', 1)[0]
#             mapping[basename] = f
            
#     found_items = []
#     for name in video_names:
#         if name in mapping:
#             found_items.append({'path': mapping[name], 'name': name})
#     return found_items

# # ==============================================================================
# # 2. ANNOTATION PARSING (Dành riêng cho TVSeries)
# # ==============================================================================
# def load_class_mapping(json_path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     # class_index[0] là "background", [1..30] là các hành động
#     class_names = data['TVSERIES']['class_index']
#     name_to_id = {name: idx for idx, name in enumerate(class_names)}
#     return name_to_id

# def parse_tvseries_gt(anno_path, class_map):
#     """Đọc file GT-train.txt hoặc GT-test.txt (ngăn cách bởi dấu Tab)"""
#     gt_dict = defaultdict(list)
#     if not os.path.exists(anno_path):
#         return gt_dict
        
#     with open(anno_path, 'r') as f:
#         for line in f:
#             # TVSeries dùng Tab làm dấu phân tách
#             parts = line.strip().split('\t')
#             if len(parts) < 4: continue
            
#             vid_name = parts[0].strip()
#             cls_name = parts[1].strip()
#             t_start  = float(parts[2])
#             t_end    = float(parts[3])
            
#             if cls_name in class_map:
#                 gt_dict[vid_name].append((t_start, t_end, class_map[cls_name]))
#     return gt_dict

# # ==============================================================================
# # 3. SYNCHRONIZED EXTRACTION (24 FPS, Sample Rate 6, Stride 6)
# # ==============================================================================


# def get_clip_indices(total_frames, orig_fps, args):
#     # 1. Resample về 24 FPS
#     if abs(orig_fps - args.target_fps) > 0.5:
#         scale = args.target_fps / orig_fps
#         new_total = int(total_frames * scale)
#         frame_idxs = np.linspace(0, total_frames - 1, new_total).astype(int)
#     else:
#         frame_idxs = np.arange(total_frames)

#     sample_rate = 4
#     window_span = (args.clip_len - 1) * sample_rate + 1 # 73 frames (~3.04s)

#     # Padding
#     if len(frame_idxs) < window_span:
#         pad = [frame_idxs[-1]] * (window_span - len(frame_idxs))
#         frame_idxs = np.concatenate([frame_idxs, pad])

#     all_clips_indices = []
#     clip_center_times = []
    
#     # Sliding window với Stride 6
#     for start in range(0, len(frame_idxs) - window_span + 1, args.stride):
#         # Dilated sampling
#         indices = frame_idxs[start : start + window_span : sample_rate]
#         all_clips_indices.append(indices)
        
#         # Center time (điểm mốc 1.5s trong cửa sổ 3s)
#         center_res_idx = start + (window_span // 2)
#         clip_center_times.append(center_res_idx / args.target_fps)
        
#     return all_clips_indices, clip_center_times

# def generate_labels(center_times, gt_segments):
#     """Tạo matrix nhãn 31 cột cho TVSeries"""
#     num_clips = len(center_times)
#     # 0: background, 1-30: actions
#     labels = np.zeros((num_clips, 31), dtype=np.float32)
#     labels[:, 0] = 1.0 # Mặc định là background
    
#     for k, center_time in enumerate(center_times):
#         for ts, te, cid in gt_segments:
#             if ts <= center_time <= te:
#                 labels[k, cid] = 1.0
#                 labels[k, 0] = 0.0 # Bỏ background nếu có hành động
#                 # TVSeries có thể có các hành động chồng lấn (overlap)
#                 # nên ta không break ở đây để bắt được đa nhãn nếu cần
#     return labels

# def process_video(model, vid_info, gt_data, args, device, normalizer):
#     try:
#         vr = decord.VideoReader(vid_info['path'])
#         orig_fps = vr.get_avg_fps()
#         total_frames = len(vr)
#     except: return None, None, None

#     indices, times = get_clip_indices(total_frames, orig_fps, args)
#     if not indices: return None, None, None

#     labels = generate_labels(times, gt_data.get(vid_info['name'], []))

#     all_rgb, all_flow = [], []
#     for i in range(0, len(indices), args.batch_size):
#         batch_idx = indices[i : i + args.batch_size]
#         flat_idx = np.concatenate(batch_idx)
#         batch_frames = vr.get_batch(flat_idx)
#         if hasattr(batch_frames, 'asnumpy'):
#             frames = batch_frames.asnumpy()
#         else:
#             # Nếu đã là torch.Tensor thì chuyển sang numpy
#             frames = batch_frames.cpu().numpy()
        
#         # (N, T, H, W, C) -> (N, C, T, H, W)
#         batch_t = torch.from_numpy(frames).float()
#         B_curr = len(batch_idx)
#         batch_t = batch_t.view(B_curr, args.clip_len, frames.shape[1], frames.shape[2], 3)
#         batch_t = batch_t.permute(0, 1, 4, 2, 3) # (B, T, C, H, W)
        
#         B, T_dim, C, H, W = batch_t.shape
#         # Xử lý 4D để Resize
#         batch_t = batch_t.reshape(B * T_dim, C, H, W)
#         batch_t = T.Resize(args.resize_short, antialias=True)(batch_t)
#         batch_t = T.CenterCrop(args.crop_size)(batch_t)
        
#         # Trả về 5D (B, C, T, H, W)
#         _, C, new_H, new_W = batch_t.shape
#         batch_t = batch_t.reshape(B, T_dim, C, new_H, new_W).permute(0, 2, 1, 3, 4)
        
#         batch_t = normalizer(batch_t).to(device)
        
#         with torch.no_grad():
#             _, _, _, _, rgb_emb, flow_emb = model(batch_t, return_embeddings=True)
            
#         all_rgb.append(rgb_emb.cpu().numpy())
#         all_flow.append(flow_emb.cpu().numpy())

#     return np.concatenate(all_rgb), np.concatenate(all_flow), labels

# # ==============================================================================
# # 4. MAIN
# # ==============================================================================
# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Chuẩn bị folder
#     for d in ['rgb_kinetics_x3d', 'flow_kinetics_x3d', 'target_perframe']:
#         os.makedirs(os.path.join(args.output_dir, d), exist_ok=True)

#     # Khởi tạo Model
#     print("🏗️ Building BioX3D Student...")
#     model = BioX3D_Student(clip_len=args.clip_len, num_classes=400).to(device)
#     if os.path.exists(args.weights):
#         ckpt = torch.load(args.weights, map_location='cpu')
#         state = ckpt.get('state_dict', ckpt.get('model_state', ckpt))
#         model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
#         set_odconv_temperature(model, 4.56)
#     model.eval()
#     normalizer = X3D_Normalizer().to(device)

#     # Đọc JSON
#     with open(args.json_file, 'r') as f:
#         full_json = json.load(f)
#         json_data = full_json["TVSERIES"]
    
#     class_map = load_class_mapping(args.json_file)
    
#     # Danh sách các task (Train & Test)
#     tasks = [
#         {"name": "TRAIN", "anno": args.anno_train, "sessions": json_data["train_session_set"]},
#         {"name": "TEST",  "anno": args.anno_test,  "sessions": json_data["test_session_set"]}
#     ]

#     for task in tasks:
#         print(f"\n🚀 Processing {task['name']} SET...")
#         vid_infos = find_video_paths(args.video_dir, task['sessions'])
#         gt_data = parse_tvseries_gt(task['anno'], class_map)
        
#         for v_info in tqdm(vid_infos):
#             name = v_info['name']
#             rgb_path = os.path.join(args.output_dir, 'rgb_kinetics_x3d', f"{name}.npy")
#             target_path = os.path.join(args.output_dir, 'target_perframe', f"{name}.npy")
            
#             if os.path.exists(rgb_path) and os.path.exists(target_path): continue
            
#             rgb, flow, lbl = process_video(model, v_info, gt_data, args, device, normalizer)
            
#             if rgb is not None:
#                 np.save(rgb_path, rgb)
#                 np.save(os.path.join(args.output_dir, 'flow_kinetics_x3d', f"{name}.npy"), flow)
#                 np.save(target_path, lbl)

#     print(f"\n✅ Hoàn tất! Kết quả lưu tại: {args.output_dir}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # Paths
#     parser.add_argument('--video_dir', type=str, required=True, help='Folder chứa toàn bộ video TVSeries')
#     parser.add_argument('--anno_train', type=str, required=True, help='Đường dẫn file GT-train.txt')
#     parser.add_argument('--anno_test', type=str, required=True, help='Đường dẫn file GT-test.txt')
#     parser.add_argument('--json_file', type=str, required=True, help='video_list.json')
#     parser.add_argument('--weights', type=str, required=True, help='BioX3D weights (.pth)')
#     parser.add_argument('--output_dir', type=str, default='./tvseries_features')
    
#     # Hyperparams
#     parser.add_argument('--target_fps', type=int, default=24)
#     parser.add_argument('--clip_len', type=int, default=13)
#     parser.add_argument('--stride', type=int, default=6)
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--resize_short', type=int, default=224)
#     parser.add_argument('--crop_size', type=int, default=224)
    
#     args = parser.parse_args()
#     main(args)