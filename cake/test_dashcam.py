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
import glob
from collections import defaultdict

# --- CONFIG ---
decord.bridge.set_bridge('torch')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Giả sử file cake.py nằm cùng thư mục
from cake import BioX3D_Student  

# ==============================================================================
# 1. UTILS (Xử lý Tensor) - GIỮ NGUYÊN
# ==============================================================================
class X3D_Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1)

    def forward(self, x):
        return (x / 255.0 - self.mean.to(x.device)) / self.std.to(x.device)

def set_odconv_temperature(model, temperature=1.0):
    for m in model.modules():
        if hasattr(m, 'update_temperature'):
            m.update_temperature(temperature)

# ==============================================================================
# 2. ANNOTATION PARSING (MỚI - Xử lý định dạng THUMOS từ file .txt)
# ==============================================================================
def build_class_mapping(train_label_dir):
    """Tự động tạo Map Class dựa trên tên các file .txt"""
    txt_files = [f for f in os.listdir(train_label_dir) if f.endswith('.txt')]
    # Sắp xếp theo tên để đảm bảo ID không bị lệch giữa các lần chạy
    class_names = sorted([f.replace('.txt', '') for f in txt_files])
    
    # Background luôn là 0, các class hành động bắt đầu từ 1
    name_to_id = {name: idx + 1 for idx, name in enumerate(class_names)}
    return name_to_id, len(class_names) + 1 # +1 cho background

def parse_thumos_gt(label_dir, class_map):
    """Đọc các file .txt và gom nhóm theo tên Video"""
    gt_dict = defaultdict(list)
    video_set = set() # Danh sách các video có mặt trong tập này
    
    for cls_name, cls_id in class_map.items():
        txt_path = os.path.join(label_dir, f"{cls_name}.txt")
        if not os.path.exists(txt_path): continue
        
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                
                vid_name = parts[0]
                t_start = float(parts[1])
                t_end = float(parts[2])
                
                gt_dict[vid_name].append((t_start, t_end, cls_id))
                video_set.add(vid_name)
                
    return gt_dict, video_set

def find_video_paths_dynamic(video_dir, target_video_names):
    """Tìm đường dẫn vật lý của các video thuộc danh sách Train hoặc Test"""
    mapping = {}
    all_files = glob.glob(os.path.join(video_dir, "**", "*.*"), recursive=True)
    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        if ext in ['.mp4', '.avi', '.mkv', '.webm', '.mov']:
            basename = os.path.basename(f).rsplit('.', 1)[0]
            mapping[basename] = f
            
    found_items = []
    for name in target_video_names:
        if name in mapping:
            found_items.append({'path': mapping[name], 'name': name})
        else:
            logging.warning(f"Không tìm thấy file video cho: {name}")
    return found_items

# ==============================================================================
# 3. SYNCHRONIZED EXTRACTION - ĐÃ CẬP NHẬT KÍCH THƯỚC LABEL
# ==============================================================================
def get_clip_indices(total_frames, orig_fps, args):
    if abs(orig_fps - args.target_fps) > 0.5:
        scale = args.target_fps / orig_fps
        new_total = int(total_frames * scale)
        frame_idxs = np.linspace(0, total_frames - 1, new_total).astype(int)
    else:
        frame_idxs = np.arange(total_frames)

    sample_rate = 4
    window_span = (args.clip_len - 1) * sample_rate + 1 

    if len(frame_idxs) < window_span:
        pad = [frame_idxs[-1]] * (window_span - len(frame_idxs))
        frame_idxs = np.concatenate([frame_idxs, pad])

    all_clips_indices = []
    clip_center_times = []
    
    for start in range(0, len(frame_idxs) - window_span + 1, args.stride):
        indices = frame_idxs[start : start + window_span : sample_rate]
        all_clips_indices.append(indices)
        center_res_idx = start + (window_span // 2)
        clip_center_times.append(center_res_idx / args.target_fps)
        
    return all_clips_indices, clip_center_times

def generate_labels(center_times, gt_segments, total_classes):
    """Tạo ma trận nhãn động theo số lượng class thực tế"""
    num_clips = len(center_times)
    # Cột 0: Background, Cột 1..N: Hành động
    labels = np.zeros((num_clips, total_classes), dtype=np.float32)
    labels[:, 0] = 1.0 # Mặc định là background
    
    for k, center_time in enumerate(center_times):
        for ts, te, cid in gt_segments:
            if ts <= center_time <= te:
                labels[k, cid] = 1.0
                labels[k, 0] = 0.0 # Bỏ background nếu có hành động
    return labels

def process_video(model, vid_info, gt_data, total_classes, args, device, normalizer):
    try:
        vr = decord.VideoReader(vid_info['path'])
        orig_fps = vr.get_avg_fps()
        total_frames = len(vr)
    except: return None, None, None

    indices, times = get_clip_indices(total_frames, orig_fps, args)
    if not indices: return None, None, None

    labels = generate_labels(times, gt_data.get(vid_info['name'], []), total_classes)

    all_rgb, all_flow = [], []
    for i in range(0, len(indices), args.batch_size):
        batch_idx = indices[i : i + args.batch_size]
        flat_idx = np.concatenate(batch_idx)
        batch_frames = vr.get_batch(flat_idx)
        if hasattr(batch_frames, 'asnumpy'):
            frames = batch_frames.asnumpy()
        else:
            frames = batch_frames.cpu().numpy()
        
        batch_t = torch.from_numpy(frames).float()
        B_curr = len(batch_idx)
        batch_t = batch_t.view(B_curr, args.clip_len, frames.shape[1], frames.shape[2], 3)
        batch_t = batch_t.permute(0, 1, 4, 2, 3) # (B, T, C, H, W)
        
        B, T_dim, C, H, W = batch_t.shape
        batch_t = batch_t.reshape(B * T_dim, C, H, W)
        batch_t = T.Resize(args.resize_short, antialias=True)(batch_t)
        batch_t = T.CenterCrop(args.crop_size)(batch_t)
        
        _, C, new_H, new_W = batch_t.shape
        batch_t = batch_t.reshape(B, T_dim, C, new_H, new_W).permute(0, 2, 1, 3, 4)
        batch_t = normalizer(batch_t).to(device)
        
        with torch.no_grad():
            _, _, _, _, rgb_emb, flow_emb = model(batch_t, return_embeddings=True)
            
        all_rgb.append(rgb_emb.cpu().numpy())
        all_flow.append(flow_emb.cpu().numpy())

    return np.concatenate(all_rgb), np.concatenate(all_flow), labels

# ==============================================================================
# 4. MAIN - ĐƯỢC THAY ĐỔI THEO LUỒNG DATA MỚI
# ==============================================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # 1. Chuẩn bị folder
    for d in ['rgb_kinetics_x3d', 'flow_kinetics_x3d', 'target_perframe']:
        os.makedirs(os.path.join(args.output_dir, d), exist_ok=True)

    # 2. Xây dựng Class Map từ thư mục Train
    class_map, total_classes = build_class_mapping(args.label_train_dir)
    print(f"📊 Tìm thấy {total_classes - 1} lớp hành động: {list(class_map.keys())}")

    # 3. Khởi tạo Model
    print("🏗️ Building BioX3D Student...")
    model = BioX3D_Student(clip_len=args.clip_len, num_classes=400).to(device)
    if os.path.exists(args.weights):
        ckpt = torch.load(args.weights, map_location='cpu')
        state = ckpt.get('state_dict', ckpt.get('model_state', ckpt))
        model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
        set_odconv_temperature(model, 2.8)
    model.eval()
    normalizer = X3D_Normalizer().to(device)

    # 4. Phân tích Dữ liệu Train & Test
    gt_train, train_videos = parse_thumos_gt(args.label_train_dir, class_map)
    gt_test, test_videos = parse_thumos_gt(args.label_test_dir, class_map)

    tasks = [
        {"name": "TRAIN", "videos": train_videos, "gt": gt_train},
        {"name": "TEST",  "videos": test_videos,  "gt": gt_test}
    ]

    # 5. Bắt đầu trích xuất
    for task in tasks:
        print(f"\n🚀 Processing {task['name']} SET ({len(task['videos'])} videos)...")
        vid_infos = find_video_paths_dynamic(args.video_dir, task['videos'])
        
        for v_info in tqdm(vid_infos):
            name = v_info['name']
            rgb_path = os.path.join(args.output_dir, 'rgb_kinetics_x3d', f"{name}.npy")
            target_path = os.path.join(args.output_dir, 'target_perframe', f"{name}.npy")
            
            # Bỏ qua nếu đã extract rồi (Resume capability)
            if os.path.exists(rgb_path) and os.path.exists(target_path): continue
            
            rgb, flow, lbl = process_video(model, v_info, task['gt'], total_classes, args, device, normalizer)
            
            if rgb is not None:
                np.save(rgb_path, rgb)
                np.save(os.path.join(args.output_dir, 'flow_kinetics_x3d', f"{name}.npy"), flow)
                np.save(target_path, lbl)

    print(f"\n✅ Hoàn tất! Kết quả lưu tại: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Các đường dẫn mới phù hợp với cấu trúc file của bạn
    parser.add_argument('--video_dir', type=str, required=True, help='Folder chứa toàn bộ file .mp4 gốc')
    parser.add_argument('--label_train_dir', type=str, required=True, help='Folder chứa các file .txt của tập Train')
    parser.add_argument('--label_test_dir', type=str, required=True, help='Folder chứa các file .txt của tập Test')
    parser.add_argument('--weights', type=str, required=True, help='BioX3D weights (.pth)')
    parser.add_argument('--output_dir', type=str, default='./dashcam_features')
    
    # Hyperparams (Giữ nguyên)
    parser.add_argument('--target_fps', type=int, default=24)
    parser.add_argument('--clip_len', type=int, default=13)
    parser.add_argument('--stride', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resize_short', type=int, default=224)
    parser.add_argument('--crop_size', type=int, default=224)
    
    args = parser.parse_args()
    main(args)