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
from collections import defaultdict

# --- CONFIG ---
decord.bridge.set_bridge('torch')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

from cake import BioX3D_Student  

# ==============================================================================
# 1. UTILS (Xử lý Tensor & Tìm kiếm Video)
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

def find_video_paths(video_dir, video_names):
    """Tìm đường dẫn video trong 1 thư mục duy nhất dựa trên danh sách tên trong JSON"""
    mapping = {}
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
    return found_items

# ==============================================================================
# 2. ANNOTATION PARSING (Dành riêng cho TVSeries)
# ==============================================================================
def load_class_mapping(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # class_index[0] là "background", [1..30] là các hành động
    class_names = data['TVSERIES']['class_index']
    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    return name_to_id

def parse_tvseries_gt(anno_path, class_map):
    """Đọc file GT-train.txt hoặc GT-test.txt (ngăn cách bởi dấu Tab)"""
    gt_dict = defaultdict(list)
    if not os.path.exists(anno_path):
        return gt_dict
        
    with open(anno_path, 'r') as f:
        for line in f:
            # TVSeries dùng Tab làm dấu phân tách
            parts = line.strip().split('\t')
            if len(parts) < 4: continue
            
            vid_name = parts[0].strip()
            cls_name = parts[1].strip()
            t_start  = float(parts[2])
            t_end    = float(parts[3])
            
            if cls_name in class_map:
                gt_dict[vid_name].append((t_start, t_end, class_map[cls_name]))
    return gt_dict

# ==============================================================================
# 3. SYNCHRONIZED EXTRACTION (24 FPS, Sample Rate 6, Stride 6)
# ==============================================================================


def get_clip_indices(total_frames, orig_fps, args):
    # 1. Resample về 24 FPS
    if abs(orig_fps - args.target_fps) > 0.5:
        scale = args.target_fps / orig_fps
        new_total = int(total_frames * scale)
        frame_idxs = np.linspace(0, total_frames - 1, new_total).astype(int)
    else:
        frame_idxs = np.arange(total_frames)

    sample_rate = 4
    window_span = (args.clip_len - 1) * sample_rate + 1 # 73 frames (~3.04s)

    # Padding
    if len(frame_idxs) < window_span:
        pad = [frame_idxs[-1]] * (window_span - len(frame_idxs))
        frame_idxs = np.concatenate([frame_idxs, pad])

    all_clips_indices = []
    clip_center_times = []
    
    # Sliding window với Stride 6
    for start in range(0, len(frame_idxs) - window_span + 1, args.stride):
        # Dilated sampling
        indices = frame_idxs[start : start + window_span : sample_rate]
        all_clips_indices.append(indices)
        
        # Center time (điểm mốc 1.5s trong cửa sổ 3s)
        center_res_idx = start + (window_span // 2)
        clip_center_times.append(center_res_idx / args.target_fps)
        
    return all_clips_indices, clip_center_times

def generate_labels(center_times, gt_segments):
    """Tạo matrix nhãn 31 cột cho TVSeries"""
    num_clips = len(center_times)
    # 0: background, 1-30: actions
    labels = np.zeros((num_clips, 31), dtype=np.float32)
    labels[:, 0] = 1.0 # Mặc định là background
    
    for k, center_time in enumerate(center_times):
        for ts, te, cid in gt_segments:
            if ts <= center_time <= te:
                labels[k, cid] = 1.0
                labels[k, 0] = 0.0 # Bỏ background nếu có hành động
                # TVSeries có thể có các hành động chồng lấn (overlap)
                # nên ta không break ở đây để bắt được đa nhãn nếu cần
    return labels

def process_video(model, vid_info, gt_data, args, device, normalizer):
    try:
        vr = decord.VideoReader(vid_info['path'])
        orig_fps = vr.get_avg_fps()
        total_frames = len(vr)
    except: return None, None, None

    indices, times = get_clip_indices(total_frames, orig_fps, args)
    if not indices: return None, None, None

    labels = generate_labels(times, gt_data.get(vid_info['name'], []))

    all_rgb, all_flow = [], []
    for i in range(0, len(indices), args.batch_size):
        batch_idx = indices[i : i + args.batch_size]
        flat_idx = np.concatenate(batch_idx)
        batch_frames = vr.get_batch(flat_idx)
        if hasattr(batch_frames, 'asnumpy'):
            frames = batch_frames.asnumpy()
        else:
            # Nếu đã là torch.Tensor thì chuyển sang numpy
            frames = batch_frames.cpu().numpy()
        
        # (N, T, H, W, C) -> (N, C, T, H, W)
        batch_t = torch.from_numpy(frames).float()
        B_curr = len(batch_idx)
        batch_t = batch_t.view(B_curr, args.clip_len, frames.shape[1], frames.shape[2], 3)
        batch_t = batch_t.permute(0, 1, 4, 2, 3) # (B, T, C, H, W)
        
        B, T_dim, C, H, W = batch_t.shape
        # Xử lý 4D để Resize
        batch_t = batch_t.reshape(B * T_dim, C, H, W)
        batch_t = T.Resize(args.resize_short, antialias=True)(batch_t)
        batch_t = T.CenterCrop(args.crop_size)(batch_t)
        
        # Trả về 5D (B, C, T, H, W)
        _, C, new_H, new_W = batch_t.shape
        batch_t = batch_t.reshape(B, T_dim, C, new_H, new_W).permute(0, 2, 1, 3, 4)
        
        batch_t = normalizer(batch_t).to(device)
        
        with torch.no_grad():
            _, _, _, _, rgb_emb, flow_emb = model(batch_t, return_embeddings=True)
            
        all_rgb.append(rgb_emb.cpu().numpy())
        all_flow.append(flow_emb.cpu().numpy())

    return np.concatenate(all_rgb), np.concatenate(all_flow), labels

# ==============================================================================
# 4. MAIN
# ==============================================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Chuẩn bị folder
    for d in ['rgb_kinetics_x3d', 'flow_kinetics_x3d', 'target_perframe']:
        os.makedirs(os.path.join(args.output_dir, d), exist_ok=True)

    # Khởi tạo Model
    print("🏗️ Building BioX3D Student...")
    model = BioX3D_Student(clip_len=args.clip_len, num_classes=400).to(device)
    if os.path.exists(args.weights):
        ckpt = torch.load(args.weights, map_location='cpu')
        state = ckpt.get('state_dict', ckpt.get('model_state', ckpt))
        model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
        set_odconv_temperature(model, 4.56)
    model.eval()
    normalizer = X3D_Normalizer().to(device)

    # Đọc JSON
    with open(args.json_file, 'r') as f:
        full_json = json.load(f)
        json_data = full_json["TVSERIES"]
    
    class_map = load_class_mapping(args.json_file)
    
    # Danh sách các task (Train & Test)
    tasks = [
        {"name": "TRAIN", "anno": args.anno_train, "sessions": json_data["train_session_set"]},
        {"name": "TEST",  "anno": args.anno_test,  "sessions": json_data["test_session_set"]}
    ]

    for task in tasks:
        print(f"\n🚀 Processing {task['name']} SET...")
        vid_infos = find_video_paths(args.video_dir, task['sessions'])
        gt_data = parse_tvseries_gt(task['anno'], class_map)
        
        for v_info in tqdm(vid_infos):
            name = v_info['name']
            rgb_path = os.path.join(args.output_dir, 'rgb_kinetics_x3d', f"{name}.npy")
            target_path = os.path.join(args.output_dir, 'target_perframe', f"{name}.npy")
            
            if os.path.exists(rgb_path) and os.path.exists(target_path): continue
            
            rgb, flow, lbl = process_video(model, v_info, gt_data, args, device, normalizer)
            
            if rgb is not None:
                np.save(rgb_path, rgb)
                np.save(os.path.join(args.output_dir, 'flow_kinetics_x3d', f"{name}.npy"), flow)
                np.save(target_path, lbl)

    print(f"\n✅ Hoàn tất! Kết quả lưu tại: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--video_dir', type=str, required=True, help='Folder chứa toàn bộ video TVSeries')
    parser.add_argument('--anno_train', type=str, required=True, help='Đường dẫn file GT-train.txt')
    parser.add_argument('--anno_test', type=str, required=True, help='Đường dẫn file GT-test.txt')
    parser.add_argument('--json_file', type=str, required=True, help='video_list.json')
    parser.add_argument('--weights', type=str, required=True, help='BioX3D weights (.pth)')
    parser.add_argument('--output_dir', type=str, default='./tvseries_features')
    
    # Hyperparams
    parser.add_argument('--target_fps', type=int, default=24)
    parser.add_argument('--clip_len', type=int, default=13)
    parser.add_argument('--stride', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resize_short', type=int, default=224)
    parser.add_argument('--crop_size', type=int, default=224)
    
    args = parser.parse_args()
    main(args)