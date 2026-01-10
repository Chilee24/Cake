import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import decord
import math

# --- CONFIG ---
decord.bridge.set_bridge('torch')

# --- IMPORTS ---
try:
    from cake import BioX3D_Student 
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    sys.exit(1)

# ==============================================================================
# 1. HELPER CLASSES
# ==============================================================================

class X3D_Normalizer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mean = torch.tensor([0.45, 0.45, 0.45], device=device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 3, 1, 1, 1)
    
    def forward(self, x):
        # x: (B, C, T, H, W) -> range [0, 255]
        return (x / 255.0 - self.mean) / self.std

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# ==============================================================================
# 2. DATASET 30-VIEWS (10 Temporal x 3 Spatial)
# ==============================================================================
class Kinetics30ViewDataset(Dataset):
    def __init__(self, list_file, root_dir, clip_len=13, crop_size=224, short_side_size=256, num_temporal_views=10):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.num_temporal_views = num_temporal_views
        self.samples = []
        
        # Load list file (format: path label_idx)
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.samples.append((" ".join(parts[:-1]), int(parts[-1])))
    
    def __len__(self):
        return len(self.samples)

    def _spatial_three_crop(self, clip):
        """
        Input: (C, T, H, W)
        Output: Stacked Tensor (3, C, T, H, W) - Left, Center, Right crops
        """
        c, t, h, w = clip.shape
        scale = self.short_side_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 1. Resize (Bilinear)
        clip = clip.permute(1, 0, 2, 3) # (T, C, H, W)
        clip = F.interpolate(clip, size=(new_h, new_w), mode='bilinear', align_corners=False)
        clip = clip.permute(1, 0, 2, 3) # (C, T, H, W)
        
        # 2. Three Crops Logic (ĐÃ SỬA)
        # Nguyên tắc: Trục dài cắt 3 phần, Trục ngắn cắt giữa (Center Crop)
        
        crops = []
        if new_w >= new_h: # Video Ngang (Landscape)
            # a. Tính tọa độ Center Crop cho chiều cao (H)
            start_h = (new_h - self.crop_size) // 2
            end_h = start_h + self.crop_size
            
            # b. Tính bước nhảy cho chiều rộng (W)
            step_w = (new_w - self.crop_size) // 2
            
            # Crop 1: Left + Center H
            crops.append(clip[:, :, start_h:end_h, 0:self.crop_size])
            
            # Crop 2: Center W + Center H
            start_w_center = (new_w - self.crop_size) // 2
            crops.append(clip[:, :, start_h:end_h, start_w_center:start_w_center+self.crop_size])
            
            # Crop 3: Right + Center H
            crops.append(clip[:, :, start_h:end_h, new_w-self.crop_size:])
            
        else: # Video Dọc (Portrait)
            # a. Tính tọa độ Center Crop cho chiều rộng (W)
            start_w = (new_w - self.crop_size) // 2
            end_w = start_w + self.crop_size
            
            # b. Tính bước nhảy cho chiều cao (H)
            step_h = (new_h - self.crop_size) // 2
            
            # Crop 1: Top + Center W
            crops.append(clip[:, :, 0:self.crop_size, start_w:end_w])
            
            # Crop 2: Center H + Center W
            start_h_center = (new_h - self.crop_size) // 2
            crops.append(clip[:, :, start_h_center:start_h_center+self.crop_size, start_w:end_w])
            
            # Crop 3: Bottom + Center W
            crops.append(clip[:, :, new_h-self.crop_size:, start_w:end_w])

        return torch.stack(crops) # (3, C, T, 224, 224)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        full_path = os.path.join(self.root_dir, rel_path)
        
        try:
            vr = decord.VideoReader(full_path)
            total_frames = len(vr)
            
            # --- 1. Temporal Uniform Sampling (10 clips) ---
            tick = (total_frames - self.clip_len + 1) / float(self.num_temporal_views)
            temporal_indices = []
            
            for i in range(self.num_temporal_views):
                start = int(tick / 2.0 + tick * i)
                start = max(0, min(start, total_frames - self.clip_len))
                indices = range(start, start + self.clip_len)
                temporal_indices.extend(indices)
            
            buffer = vr.get_batch(temporal_indices) # (10*T, H, W, C)
            buffer = buffer.permute(0, 3, 1, 2).float() # (10*T, C, H, W)
            
            # Reshape lại
            views_temporal = buffer.view(self.num_temporal_views, self.clip_len, 3, buffer.size(2), buffer.size(3))
            views_temporal = views_temporal.permute(0, 2, 1, 3, 4) # (10, C, T, H, W)
            
            # --- 2. Spatial Three Crop ---
            final_views = []
            for i in range(self.num_temporal_views):
                clip = views_temporal[i] 
                three_crops = self._spatial_three_crop(clip) 
                final_views.append(three_crops)
            
            data = torch.cat(final_views, dim=0) # (30, C, T, 224, 224)
            return data, label, rel_path

        except Exception as e:
            # print(f"⚠️ Error {rel_path}: {e}")
            # Trả về dummy kích thước chuẩn
            return torch.zeros(30, 3, self.clip_len, self.crop_size, self.crop_size), label, rel_path
# ==============================================================================
# 3. MAIN EVALUATION
# ==============================================================================
def main(args):
    device = torch.device("cuda")
    print(f"🚀 Multi-view Inference (30 views) on {device}")
    
    # 1. Model
    print("📦 Loading Model...")
    model = BioX3D_Student(clip_len=13, num_classes=400, feature_dim=192).to(device)
    
    if os.path.isfile(args.checkpoint):
        print(f"📥 Loading weights: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Loaded: {msg}")
    else:
        print("⚠️ No checkpoint found!")
        return

    model.eval()
    normalizer = X3D_Normalizer(device)

    # 2. Dataset
    val_ds = Kinetics30ViewDataset(args.val_list, args.val_root, clip_len=13)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=True)

    # 3. Loop
    top1_avg_correct = 0      # Cách chuẩn (Trung bình cộng)
    at_least_one_correct = 0  # Cách mới (Chỉ cần 1 view đúng)
    total_samples = 0
    
    print(f"Starting inference on {len(val_ds)} videos...")
    
    with torch.no_grad():
        for i, (views, labels, _) in enumerate(tqdm(val_loader)):
            # views shape: (B, 30, C, T, H, W)
            B, V, C, T, H, W = views.shape
            
            # Input shape: (B*30, C, T, H, W)
            views = views.view(B * V, C, T, H, W).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            views = normalizer(views)
            
            with torch.amp.autocast('cuda'):
                rgb_logits, flow_logits, _, _ = model(views)
            
            # --- FUSION (RGB + 0.5 Flow) ---
            combined_logits_all = rgb_logits + 0.5 * flow_logits
            
            # Tách lại về từng video: (B, 30, 400)
            combined_logits_video = combined_logits_all.view(B, V, -1)
            
            # Tính Softmax cho từng view
            probs_video = F.softmax(combined_logits_video, dim=2)
            
            # =========================================================
            # CÁCH 1: STANDARD (AVERAGE VOTING) - CHUẨN SOTA
            # =========================================================
            # Lấy trung bình cộng (Mean) của 30 views -> (B, 400)
            final_probs_avg = torch.mean(probs_video, dim=1)
            pred_avg = final_probs_avg.argmax(dim=1) # (B)
            top1_avg_correct += (pred_avg == labels).sum().item()

            # =========================================================
            # CÁCH 2: AT LEAST ONE (ORACLE) - CÁCH BẠN MUỐN
            # =========================================================
            # 1. Lấy class dự đoán cho TỪNG view riêng lẻ
            # (B, 30, 400) -> (B, 30) chứa index class max
            _, view_preds = torch.max(probs_video, dim=2) 
            
            # 2. Mở rộng label để so sánh: (B) -> (B, 30)
            labels_expanded = labels.view(B, 1).expand_as(view_preds)
            
            # 3. Kiểm tra view nào đúng (Boolean Mask)
            # hits shape: (B, 30) (True nếu view đó đúng, False nếu sai)
            hits = (view_preds == labels_expanded)
            
            # 4. Kiểm tra xem video có ÍT NHẤT 1 view đúng không (.any())
            # video_hit shape: (B)
            video_hit = hits.any(dim=1)
            
            # 5. Cộng dồn
            at_least_one_correct += video_hit.sum().item()
            
            # =========================================================

            total_samples += B
            
            if i % 100 == 0:
                print(f"Step {i}: Standard Acc: {top1_avg_correct/total_samples*100:.2f}% | At-Least-One Acc: {at_least_one_correct/total_samples*100:.2f}%")

    # --- KẾT QUẢ ---
    acc_std = top1_avg_correct / total_samples * 100
    acc_oracle = at_least_one_correct / total_samples * 100
    
    print(f"\n🏆 FINAL RESULTS (30 Views):")
    print("-" * 50)
    print(f"1. Standard Accuracy (Average Voting) : {acc_std:.2f}%")
    print(f"   (Đây là chỉ số dùng để so sánh với Paper)")
    print("-" * 50)
    print(f"2. At-Least-One Accuracy (Best View)  : {acc_oracle:.2f}%")
    print(f"   (Đây là tiềm năng tối đa của model)")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_list', type=str, required=True, help="Path to val.txt (format: path label_idx)")
    parser.add_argument('--val_root', type=str, required=True, help="Root folder of validation videos")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1, help="Số video xử lý cùng lúc (mỗi video = 30 clips)")
    parser.add_argument('--workers', type=int, default=8)
    
    args = parser.parse_args()
    main(args)