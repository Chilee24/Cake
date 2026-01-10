import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import numpy as np
import os
import sys
import argparse
import decord
from tqdm import tqdm
import logging

# --- Cấu hình môi trường ---
decord.bridge.set_bridge('torch')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

try:
    from cake import BioX3D_Student  
except ImportError:
    print("❌ Lỗi Import: Không tìm thấy 'cake'.")
    sys.exit(1)

# ==============================================================================
# 0. UTILS
# ==============================================================================
# [FIX] Hàm set nhiệt độ cho ODConv
def set_odconv_temperature(model, temperature=3.0):
    count = 0
    for m in model.modules():
        if hasattr(m, 'update_temperature'):
            m.update_temperature(temperature)
            count += 1
    print(f"🌡️ Đã set ODConv Temperature = {temperature} cho {count} modules.")

# ==============================================================================
# 1. DATASET & TRANSFORMS
# ==============================================================================

class X3D_Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1)

    def forward(self, x):
        return (x / 255.0 - self.mean.to(x.device)) / self.std.to(x.device)

class Kinetics30ViewsDataset(Dataset):
    def __init__(self, root_dir, list_file, clip_len=13, sampling_rate=6, resize_short=182, crop_size=182):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.sampling_rate = sampling_rate
        self.resize_short = resize_short
        self.crop_size = crop_size
        
        with open(list_file, 'r') as f:
            self.samples = [line.strip().split() for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_str = " ".join(self.samples[idx][:-1])
        label = int(self.samples[idx][-1])
        full_path = os.path.join(self.root_dir, path_str)

        try:
            vr = decord.VideoReader(full_path)
            total_frames = len(vr)
        except Exception as e:
            return None, None

        clip_span = (self.clip_len - 1) * self.sampling_rate + 1
        
        if total_frames <= clip_span:
            start_indices = [0] * 10
        else:
            tick = (total_frames - clip_span) / 10.0
            start_indices = [int(tick * i) for i in range(10)]

        input_clips = []
        for start_idx in start_indices:
            frame_indices = []
            for i in range(self.clip_len):
                idx = min(start_idx + i * self.sampling_rate, total_frames - 1)
                frame_indices.append(idx)
            
            buffer = vr.get_batch(frame_indices) # (T, H, W, C)
            if not isinstance(buffer, torch.Tensor):
                buffer = torch.from_numpy(buffer.asnumpy())
            
            buffer = buffer.permute(0, 3, 1, 2).float() # (T, C, H, W)
            buffer = T.Resize(self.resize_short, antialias=True)(buffer)

            # 3 Spatial Crops
            _, _, h, w = buffer.shape
            c1 = TF.crop(buffer, 0, 0, self.crop_size, self.crop_size)
            c2 = TF.center_crop(buffer, (self.crop_size, self.crop_size))
            if w >= h:
                c3 = TF.crop(buffer, 0, w - self.crop_size, self.crop_size, self.crop_size)
            else:
                c3 = TF.crop(buffer, h - self.crop_size, 0, self.crop_size, self.crop_size)

            input_clips.extend([c1, c2, c3])

        # Kết quả: (30, C, T, H, W)
        views = torch.stack(input_clips).permute(0, 2, 1, 3, 4)
        return views, label

def collate_fn_safe(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0: return None, None
    return torch.utils.data.dataloader.default_collate(batch)

# ==============================================================================
# 2. MAIN EVALUATION
# ==============================================================================

def main(args):
    device = torch.device("cuda")
    
    # 1. Model
    print(f"🏗️ Building Model (Batch Size: {args.batch_size})...")
    model = BioX3D_Student(clip_len=args.clip_len, num_classes=400, feature_dim=192)
    model = model.to(device)
    model.eval()
    
    # Load Checkpoint
    if os.path.exists(args.weights):
        ckpt = torch.load(args.weights, map_location='cpu')
        state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        new_state = {k.replace('module.', ''): v for k, v in state.items()}
        
        # Load strict=True để đảm bảo load hết
        try:
            model.load_state_dict(new_state, strict=True)
            print("✅ Weights Loaded (Strict).")
        except:
            print("⚠️ Load strict failed, trying loose load...")
            model.load_state_dict(new_state, strict=False)

        # [QUAN TRỌNG] FIX LỖI 5% ACCURACY
        # Set nhiệt độ về 1.0 (giá trị cuối cùng của quá trình training)
        set_odconv_temperature(model, temperature=4.54)

    else:
        print(f"❌ Weights not found: {args.weights}")
        return

    normalizer = X3D_Normalizer().to(device)

    # 2. DataLoader
    dataset = Kinetics30ViewsDataset(
        root_dir=args.root,
        list_file=args.test_list,
        clip_len=args.clip_len,
        sampling_rate=args.sampling_rate,
        resize_short=args.resize_short,
        crop_size=args.crop_size
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        pin_memory=True,
        collate_fn=collate_fn_safe 
    )

    print(f"🚀 Testing on {len(dataset)} videos | Workers: {args.workers}")

    # Accumulators
    rgb_correct_1 = 0
    rgb_correct_5 = 0
    flow_correct_1 = 0
    flow_correct_5 = 0
    total_processed = 0
    
    pbar = tqdm(loader, dynamic_ncols=True)

    with torch.no_grad():
        for batch_views, batch_labels in pbar:
            if batch_views is None: continue 

            b, n_views, c, t, h, w = batch_views.shape
            
            # Input model: (Batch_Size * 30, C, T, H, W)
            flat_inputs = batch_views.view(b * n_views, c, t, h, w).to(device)
            batch_labels = batch_labels.to(device)

            flat_inputs = normalizer(flat_inputs)

            # --- INFERENCE ---
            outputs = model(flat_inputs)
            
            logits_rgb = outputs[0]  
            logits_flow = outputs[1] 

            # --- HÀM TÍNH ĐIỂM ---
            def evaluate_branch(logits, labels, batch_size, num_views):
                probs = F.softmax(logits, dim=1) 
                probs = probs.view(batch_size, num_views, -1)
                avg_probs = torch.mean(probs, dim=1) 
                pred_1 = avg_probs.argmax(dim=1)
                c1 = (pred_1 == labels).sum().item()
                _, pred_5 = avg_probs.topk(5, dim=1)
                c5 = labels.view(-1, 1).eq(pred_5).sum().item()
                return c1, c5

            # Tính điểm
            r_c1, r_c5 = evaluate_branch(logits_rgb, batch_labels, b, n_views)
            rgb_correct_1 += r_c1
            rgb_correct_5 += r_c5
            
            f_c1, f_c5 = evaluate_branch(logits_flow, batch_labels, b, n_views)
            flow_correct_1 += f_c1
            flow_correct_5 += f_c5

            total_processed += b
            
            rgb_acc = rgb_correct_1 / total_processed * 100
            flow_acc = flow_correct_1 / total_processed * 100
            pbar.set_description(f"RGB_Top1: {rgb_acc:.2f}% | Flow_Top1: {flow_acc:.2f}%")

    # In kết quả
    print(f"\n🏆 FINAL RESULT (Kinetics-400 30-Views)")
    print("-" * 50)
    print(f"📸 RGB BRANCH:")
    print(f"   - Top-1: {rgb_correct_1 / total_processed * 100:.2f}%")
    print(f"   - Top-5: {rgb_correct_5 / total_processed * 100:.2f}%")
    print("-" * 50)
    print(f"🌊 HALLUCINATED FLOW BRANCH:")
    print(f"   - Top-1: {flow_correct_1 / total_processed * 100:.2f}%")
    print(f"   - Top-5: {flow_correct_5 / total_processed * 100:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--test_list', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    
    # Defaults cho K400 
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--clip_len', type=int, default=13)
    parser.add_argument('--sampling_rate', type=int, default=6)
    parser.add_argument('--resize_short', type=int, default=224) # Lưu ý: Train 224 thì Eval cũng nên 224
    parser.add_argument('--crop_size', type=int, default=224)    # Lưu ý: Train 224 thì Eval cũng nên 224
    
    args = parser.parse_args()
    main(args)