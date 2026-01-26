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
def set_odconv_temperature(model, temperature=4.6):
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
        except Exception:
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
    
    # 1. Model Setup
    print(f"🏗️ Building Model (Batch Size: {args.batch_size})...")
    model = BioX3D_Student(clip_len=args.clip_len, num_classes=400, feature_dim=192)
    model = model.to(device)
    
    if os.path.exists(args.weights):
        print(f"📥 Loading weights from: {args.weights}")
        ckpt = torch.load(args.weights, map_location='cpu')
        state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        new_state = {k.replace('module.', ''): v for k, v in state.items()}
        
        try:
            model.load_state_dict(new_state, strict=True)
            print("✅ Weights Loaded (Strict Mode).")
        except:
            print("⚠️ Load strict failed, trying loose load...")
            model.load_state_dict(new_state, strict=False)

        # Set nhiệt độ tối ưu cho ODConv (4.6 cho epoch 17)
        set_odconv_temperature(model, temperature=2.8)
    else:
        print(f"❌ Weights not found: {args.weights}")
        return

    model.eval()
    normalizer = X3D_Normalizer().to(device)

    # 2. DataLoader Setup
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
    print(f"🔥 Strategy: Conditional Confidence Fusion (Thresh: {args.conf_thresh} | RGB_W: {args.rgb_weight})")

    # Accumulators
    rgb_c1, rgb_c5 = 0, 0
    flow_c1, flow_c5 = 0, 0
    fused_c1, fused_c5 = 0, 0
    total_processed = 0
    
    pbar = tqdm(loader, dynamic_ncols=True)

    with torch.no_grad():
        for batch_views, batch_labels in pbar:
            if batch_views is None: continue 

            b, n_views, c, t, h, w = batch_views.shape
            
            # Flatten inputs for batch processing
            flat_inputs = batch_views.view(b * n_views, c, t, h, w).to(device)
            batch_labels = batch_labels.to(device)
            flat_inputs = normalizer(flat_inputs)

            # --- INFERENCE ---
            logits_rgb, logits_flow, _, _ = model(flat_inputs)
            
            # --- VIEW AGGREGATION (Softmax per view then Average) ---
            # RGB
            probs_rgb_views = F.softmax(logits_rgb, dim=1).view(b, n_views, -1)
            avg_probs_rgb = torch.mean(probs_rgb_views, dim=1) # (B, 400)
            
            # Flow
            probs_flow_views = F.softmax(logits_flow, dim=1).view(b, n_views, -1)
            avg_probs_flow = torch.mean(probs_flow_views, dim=1) # (B, 400)

            # --- 1. Evaluate RGB & Flow independently ---
            # RGB Metrics
            pred_rgb_1 = avg_probs_rgb.argmax(dim=1)
            rgb_c1 += (pred_rgb_1 == batch_labels).sum().item()
            _, pred_rgb_5 = avg_probs_rgb.topk(5, dim=1)
            rgb_c5 += batch_labels.view(-1, 1).eq(pred_rgb_5).sum().item()

            # Flow Metrics
            pred_flow_1 = avg_probs_flow.argmax(dim=1)
            flow_c1 += (pred_flow_1 == batch_labels).sum().item()
            _, pred_flow_5 = avg_probs_flow.topk(5, dim=1)
            flow_c5 += batch_labels.view(-1, 1).eq(pred_flow_5).sum().item()

            # --- 2. ADVANCED FUSION LOGIC (Conditional Confidence) ---
            # Lấy max score để kiểm tra độ tự tin
            conf_rgb, _ = avg_probs_rgb.max(dim=1)
            conf_flow, _ = avg_probs_flow.max(dim=1)
            
            final_preds_1 = []
            final_preds_5_list = []

            for i in range(b):
                # CASE 1: RGB rất tự tin -> Tin RGB
                if conf_rgb[i] > args.conf_thresh:
                    final_preds_1.append(pred_rgb_1[i])
                    final_preds_5_list.append(pred_rgb_5[i])
                
                # CASE 2: Flow rất tự tin -> Tin Flow
                elif conf_flow[i] > args.conf_thresh:
                    final_preds_1.append(pred_flow_1[i])
                    final_preds_5_list.append(pred_flow_5[i])
                
                # CASE 3: Cả 2 đều không chắc -> Weighted Average
                else:
                    # Hợp nhất xác suất
                    mixed_prob = (args.rgb_weight * avg_probs_rgb[i]) + ((1.0 - args.rgb_weight) * avg_probs_flow[i])
                    
                    # Top-1 Mixed
                    final_preds_1.append(mixed_prob.argmax())
                    
                    # Top-5 Mixed
                    _, p5 = mixed_prob.topk(5)
                    final_preds_5_list.append(p5)

            # Stack lại thành Tensor để tính toán
            final_preds_1 = torch.stack(final_preds_1)
            final_preds_5 = torch.stack(final_preds_5_list)

            fused_c1 += (final_preds_1 == batch_labels).sum().item()
            fused_c5 += batch_labels.view(-1, 1).eq(final_preds_5).sum().item()

            total_processed += b
            
            # --- Update Progress Bar ---
            pbar.set_description(
                f"RGB:{rgb_c1/total_processed*100:.1f}% | "
                f"Flow:{flow_c1/total_processed*100:.1f}% | "
                f"FUSED:{fused_c1/total_processed*100:.1f}%"
            )

    # --- FINAL REPORT ---
    def print_acc(name, c1, c5, total):
        acc1 = c1 / total * 100
        acc5 = c5 / total * 100
        print(f"{name:<15} | Top-1: {acc1:.2f}% | Top-5: {acc5:.2f}%")

    print(f"\n🏆 FINAL RESULT (Kinetics-400 30-Views)")
    print("=" * 60)
    print_acc("📸 RGB Only", rgb_c1, rgb_c5, total_processed)
    print_acc("🌊 Flow Only", flow_c1, flow_c5, total_processed)
    print("-" * 60)
    print_acc("🔥 FUSION", fused_c1, fused_c5, total_processed)
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--test_list', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    
    # Eval settings
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--clip_len', type=int, default=13)
    parser.add_argument('--sampling_rate', type=int, default=6)
    parser.add_argument('--resize_short', type=int, default=224)
    parser.add_argument('--crop_size', type=int, default=224)
    
    # Fusion Hyperparameters
    parser.add_argument('--conf_thresh', type=float, default=0.8, help='Ngưỡng tự tin để chọn nhánh (0.0-1.0)')
    parser.add_argument('--rgb_weight', type=float, default=0.5, help='Trọng số cho RGB khi mix (0.0-1.0)')
    
    args = parser.parse_args()
    main(args)