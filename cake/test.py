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

try:
    from cake import BioX3D_Student  
except ImportError:
    print("❌ Lỗi Import: Không tìm thấy 'cake.py'.")
    sys.exit(1)

# ==============================================================================
# 1. UTILS
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

def find_video_paths_in_dir(video_dir, video_names):
    """Tìm đường dẫn video trong một folder cụ thể dựa trên danh sách tên"""
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
# 2. ANNOTATION PARSING
# ==============================================================================
def load_class_mapping(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    class_names = data['THUMOS']['class_index']
    # Background thường là index 0
    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    return name_to_id

def parse_ambiguous(anno_dir, mode="val"):
    """Thường là Ambiguous_val.txt hoặc Ambiguous_test.txt"""
    amb_dict = defaultdict(list)
    filename = "Ambiguous_val.txt" if "train" in mode.lower() or "val" in mode.lower() else "Ambiguous_test.txt"
    amb_path = os.path.join(anno_dir, filename)
    
    if os.path.exists(amb_path):
        with open(amb_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                amb_dict[parts[0]].append((float(parts[1]), float(parts[2])))
    else:
        logging.warning(f"⚠️ Không tìm thấy file Ambiguous tại: {amb_path}")
    return amb_dict

def parse_annotations(anno_dir, class_map):
    gt_dict = defaultdict(list)
    txt_files = glob.glob(os.path.join(anno_dir, "*.txt"))
    for txt_path in txt_files:
        raw_name = os.path.basename(txt_path).replace('.txt', '')
        if "Ambiguous" in raw_name: continue
        
        # Clean tên class (ví dụ: BaseballPitch_val -> BaseballPitch)
        class_name = raw_name.replace('_val', '').replace('_test', '')
        if class_name not in class_map: continue
        class_id = class_map[class_name]
        
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                # format: video_name start_time end_time
                gt_dict[parts[0]].append((float(parts[1]), float(parts[2]), class_id))
    return gt_dict

# ==============================================================================
# 3. SYNCHRONIZED EXTRACTION LOGIC
# ==============================================================================


def get_clip_indices(total_frames, orig_fps, args):
    if abs(orig_fps - args.target_fps) > 1.0:
        scale = args.target_fps / orig_fps
        new_total = int(total_frames * scale)
        frame_idxs = np.linspace(0, total_frames - 1, new_total).astype(int)
    else:
        frame_idxs = np.arange(total_frames)

    sample_rate = 3
    window_span = (args.clip_len - 1) * sample_rate + 1 # 73 frames cho 3 giây

    if len(frame_idxs) < window_span:
        pad = [frame_idxs[-1]] * (window_span - len(frame_idxs))
        frame_idxs = np.concatenate([frame_idxs, pad])

    all_clips_indices = []
    clip_center_times = []
    
    for start in range(0, len(frame_idxs) - window_span + 1, args.stride):
        # Sampling thưa: [0, 6, 12...72]
        indices = frame_idxs[start : start + window_span : sample_rate]
        all_clips_indices.append(indices)
        # Center frame: start + 36
        center_time = (start + (window_span // 2)) / args.target_fps
        clip_center_times.append(center_time)
        
    return all_clips_indices, clip_center_times

def generate_labels(center_times, gt_segments, amb_segments):
    num_clips = len(center_times)
    # Khởi tạo matrix label toàn số 0
    labels = np.zeros((num_clips, 22), dtype=np.float32)
    
    # Mặc định gán toàn bộ là Background (cột 0)
    labels[:, 0] = 1.0 
    
    for k, center_time in enumerate(center_times):
        # 1. Check Ambiguous (Ưu tiên cao nhất - Exclusive)
        # Nếu là Ambiguous thì thường ta coi như không xác định được hành động nào khác
        is_amb = False
        for ts, te in amb_segments:
            if ts <= center_time <= te:
                labels[k, :] = 0.0    # Reset hết
                labels[k, 21] = 1.0   # Gán Ambiguous
                labels[k, 0] = 0.0    # Xóa Background
                is_amb = True
                break # Ambiguous là trạng thái độc quyền, nên break ở đây là ĐÚNG
        
        if is_amb: continue
        
        # 2. Check Actions (Hỗ trợ Multi-hot)
        found_action = False
        for ts, te, cid in gt_segments:
            if ts <= center_time <= te:
                labels[k, cid] = 1.0  # Đánh dấu hành động này
                found_action = True
                # [FIX] XÓA dòng 'break' ở đây để tìm tiếp các hành động chồng lấn
        
        # Nếu tìm thấy ít nhất 1 hành động, xóa Background
        if found_action:
            labels[k, 0] = 0.0

    return labels

def process_single_video(model, video_info, gt_data, amb_data, args, device, normalizer):
    video_path = video_info['path']
    video_name = video_info['name']
    
    try:
        vr = decord.VideoReader(video_path)
        orig_fps = vr.get_avg_fps()
        total_frames = len(vr)
    except: return None, None, None

    indices, times = get_clip_indices(total_frames, orig_fps, args)
    if not indices: return None, None, None

    labels = generate_labels(times, gt_data.get(video_name, []), amb_data.get(video_name, []))

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
        
        # Chuyển sang Tensor (N, H, W, C) -> (N, C, H, W)
        batch_t = torch.from_numpy(frames).float()
        
        # Reshape về (B, T, H, W, C)
        B_curr = len(batch_idx)
        batch_t = batch_t.view(B_curr, args.clip_len, frames.shape[1], frames.shape[2], 3)
        
        # Chuyển trục để gộp xử lý: (B, T, H, W, C) -> (B, T, C, H, W)
        batch_t = batch_t.permute(0, 1, 4, 2, 3)
        B, T_dim, C, H, W = batch_t.shape
        
        # Gộp B*T để Resize 4D
        batch_t = batch_t.reshape(B * T_dim, C, H, W)
        batch_t = T.Resize(args.resize_short, antialias=True)(batch_t)
        batch_t = T.CenterCrop(args.crop_size)(batch_t)
        
        # Trả về 5D: (B, C, T, H, W)
        _, C, new_H, new_W = batch_t.shape
        batch_t = batch_t.reshape(B, T_dim, C, new_H, new_W).permute(0, 2, 1, 3, 4)
        
        batch_t = normalizer(batch_t).to(device)
        
        with torch.no_grad():
            _, _, _, _, rgb_emb, flow_emb = model(batch_t, return_embeddings=True)
            
        all_rgb.append(rgb_emb.cpu().numpy())
        all_flow.append(flow_emb.cpu().numpy())

    return np.concatenate(all_rgb), np.concatenate(all_flow), labels

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tạo thư mục đầu ra
    for d in ['rgb', 'flow', 'target']:
        os.makedirs(os.path.join(args.output_dir, d), exist_ok=True)

    # 1. Khởi tạo Model
    print("🏗️ Loading BioX3D Model...")
    model = BioX3D_Student(clip_len=args.clip_len, num_classes=400).to(device)
    if os.path.exists(args.weights):
        ckpt = torch.load(args.weights, map_location='cpu')
        state = ckpt.get('state_dict', ckpt.get('model_state', ckpt))
        model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
        set_odconv_temperature(model, 4.56)
    model.eval()
    normalizer = X3D_Normalizer().to(device)

    # 2. Đọc JSON và phân loại video
    with open(args.json_file, 'r') as f:
        json_data = json.load(f)["THUMOS"]
    
    class_map = load_class_mapping(args.json_file)
    
    # Cấu hình danh sách cần xử lý
    tasks = [
        {
            "name": "TRAIN SET",
            "video_dir": args.video_train,
            "anno_dir": args.anno_train,
            "session_list": json_data["train_session_set"],
            "mode": "train"
        },
        {
            "name": "TEST SET",
            "video_dir": args.video_test,
            "anno_dir": args.anno_test,
            "session_list": json_data["test_session_set"],
            "mode": "test"
        }
    ]

    for task in tasks:
        print(f"\n🚀 Processing {task['name']}...")
        video_list = find_video_paths_in_dir(task['video_dir'], task['session_list'])
        gt_data = parse_annotations(task['anno_dir'], class_map)
        amb_data = parse_ambiguous(task['anno_dir'], task['mode'])
        
        for vid_info in tqdm(video_list, desc=f"Extracting {task['mode']}"):
            v_name = vid_info['name']
            rgb_p = os.path.join(args.output_dir, 'rgb', f"{v_name}.npy")
            lbl_p = os.path.join(args.output_dir, 'target', f"{v_name}.npy")
            
            if os.path.exists(rgb_p) and os.path.exists(lbl_p): continue
            
            rgb, flow, lbl = process_single_video(model, vid_info, gt_data, amb_data, args, device, normalizer)
            
            if rgb is not None:
                np.save(rgb_p, rgb)
                np.save(os.path.join(args.output_dir, 'flow', f"{v_name}.npy"), flow)
                np.save(lbl_p, lbl)

    print(f"\n✅ Hoàn tất! Dữ liệu tại: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths cho Train
    parser.add_argument('--video_train', type=str, required=True, help='Folder video tập Validation (Train)')
    parser.add_argument('--anno_train', type=str, required=True, help='Folder annotation tập Validation')
    # Paths cho Test
    parser.add_argument('--video_test', type=str, required=True, help='Folder video tập Test')
    parser.add_argument('--anno_test', type=str, required=True, help='Folder annotation tập Test')
    
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./thumos_synced')
    
    parser.add_argument('--target_fps', type=int, default=24)
    parser.add_argument('--clip_len', type=int, default=13)
    parser.add_argument('--stride', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resize_short', type=int, default=224)
    parser.add_argument('--crop_size', type=int, default=224)
    
    args = parser.parse_args()
    main(args)