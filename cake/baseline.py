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

# --- CONFIG ---
decord.bridge.set_bridge('torch')

# --- IMPORT LOCAL (Thay vì torch.hub) ---
# Yêu cầu đã cài đặt: pip install pytorchvideo
try:
    import pytorchvideo.models.x3d as x3d
except ImportError:
    print("❌ Lỗi: Chưa cài đặt thư viện pytorchvideo. Hãy chạy: pip install pytorchvideo")
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
        return (x / 255.0 - self.mean) / self.std

def accuracy(output, target, topk=(1,)):
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
        
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.samples.append((" ".join(parts[:-1]), int(parts[-1])))
    
    def __len__(self):
        return len(self.samples)

    def _spatial_three_crop(self, clip):
        """Input: (C, T, H, W) -> Output: (3, C, T, H, W)"""
        c, t, h, w = clip.shape
        scale = self.short_side_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        clip = clip.permute(1, 0, 2, 3) 
        clip = F.interpolate(clip, size=(new_h, new_w), mode='bilinear', align_corners=False)
        clip = clip.permute(1, 0, 2, 3) 
        
        crops = []
        if new_w >= new_h: # Landscape
            start_h = (new_h - self.crop_size) // 2
            end_h = start_h + self.crop_size
            step_w = (new_w - self.crop_size) // 2
            
            crops.append(clip[:, :, start_h:end_h, 0:self.crop_size]) # Left
            start_w_center = (new_w - self.crop_size) // 2
            crops.append(clip[:, :, start_h:end_h, start_w_center:start_w_center+self.crop_size]) # Center
            crops.append(clip[:, :, start_h:end_h, new_w-self.crop_size:]) # Right
        else: # Portrait
            start_w = (new_w - self.crop_size) // 2
            end_w = start_w + self.crop_size
            step_h = (new_h - self.crop_size) // 2
            
            crops.append(clip[:, :, 0:self.crop_size, start_w:end_w]) # Top
            start_h_center = (new_h - self.crop_size) // 2
            crops.append(clip[:, :, start_h_center:start_h_center+self.crop_size, start_w:end_w]) # Center
            crops.append(clip[:, :, new_h-self.crop_size:, start_w:end_w]) # Bottom

        return torch.stack(crops)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        full_path = os.path.join(self.root_dir, rel_path)
        
        try:
            vr = decord.VideoReader(full_path)
            total_frames = len(vr)
            
            tick = (total_frames - self.clip_len + 1) / float(self.num_temporal_views)
            temporal_indices = []
            
            for i in range(self.num_temporal_views):
                start = int(tick / 2.0 + tick * i)
                start = max(0, min(start, total_frames - self.clip_len))
                indices = range(start, start + self.clip_len)
                temporal_indices.extend(indices)
            
            buffer = vr.get_batch(temporal_indices)
            buffer = buffer.permute(0, 3, 1, 2).float()
            
            views_temporal = buffer.view(self.num_temporal_views, self.clip_len, 3, buffer.size(2), buffer.size(3))
            views_temporal = views_temporal.permute(0, 2, 1, 3, 4)
            
            final_views = []
            for i in range(self.num_temporal_views):
                clip = views_temporal[i] 
                three_crops = self._spatial_three_crop(clip) 
                final_views.append(three_crops)
            
            data = torch.cat(final_views, dim=0) 
            return data, label, rel_path

        except Exception as e:
            return torch.zeros(30, 3, self.clip_len, self.crop_size, self.crop_size), label, rel_path

# ==============================================================================
# 3. MAIN EVALUATION
# ==============================================================================
def main(args):
    device = torch.device("cuda")
    print(f"🚀 X3D-S Local Baseline Inference (30 views) on {device}")
    
    # --- 1. DEFINE MODEL LOCALLY ---
    print("📦 Creating X3D-S model from local pytorchvideo...")
    
    model = x3d.create_x3d(
        input_clip_length=13,
        input_crop_size=224, 
        model_num_class=400
    )
    model = model.to(device)
    
    # --- 2. LOAD WEIGHTS (FIX LỖI CẤU TRÚC DICT) ---
    if os.path.isfile(args.checkpoint):
        print(f"📥 Loading weights: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # --- BƯỚC QUAN TRỌNG: BÓC TÁCH WEIGHTS ---
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
            print("   -> Phát hiện key 'model_state', đang trích xuất weights...")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("   -> Phát hiện key 'state_dict', đang trích xuất weights...")
        else:
            state_dict = checkpoint # Trường hợp file chỉ chứa weight thuần
        
        # --- MAPPING KEY TỪ BioX3D -> X3D Gốc ---
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '') 
            
            # Bỏ qua nhánh Flow/Hallucination
            if 'flow' in k or 'hallucinator' in k: 
                continue
            
            # Mapping: BioX3D (head) -> X3D (blocks.5)
            if k.startswith('head.'):
                # Map 'head.xxx' -> 'blocks.5.xxx'
                k_new = k.replace('head.', 'blocks.5.')
                new_state_dict[k_new] = v
            elif k.startswith('blocks.'):
                # Giữ nguyên backbone (blocks.0 -> blocks.4)
                new_state_dict[k] = v
            else:
                # Các key khác
                new_state_dict[k] = v
            
        # Load weight (strict=True để đảm bảo khớp 100%)
        msg = model.load_state_dict(new_state_dict, strict=True)
        print(f"✅ Loaded: {msg}")
    else:
        print("⚠️ No checkpoint found! Exiting.")
        return

    model.eval()
    normalizer = X3D_Normalizer(device)

    # 2. Dataset
    val_ds = Kinetics30ViewDataset(args.val_list, args.val_root, clip_len=13)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=True)

    # 3. Loop
    top1_avg_correct = 0
    at_least_one_correct = 0
    total_samples = 0
    
    print(f"Starting inference on {len(val_ds)} videos...")
    
    with torch.no_grad():
        for i, (views, labels, _) in enumerate(tqdm(val_loader)):
            B, V, C, T, H, W = views.shape
            views = views.view(B * V, C, T, H, W).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            views = normalizer(views)
            
            with torch.amp.autocast('cuda'):
                logits = model(views) 
            
            logits_video = logits.view(B, V, -1)
            probs_video = F.softmax(logits_video, dim=2)
            
            # 1. Standard Acc
            final_probs_avg = torch.mean(probs_video, dim=1) 
            pred_avg = final_probs_avg.argmax(dim=1) 
            top1_avg_correct += (pred_avg == labels).sum().item()

            # 2. At Least One Acc
            _, view_preds = torch.max(probs_video, dim=2) 
            labels_expanded = labels.view(B, 1).expand_as(view_preds)
            hits = (view_preds == labels_expanded)
            video_hit = hits.any(dim=1)
            at_least_one_correct += video_hit.sum().item()
            
            total_samples += B
            
            if i % 100 == 0:
                print(f"Step {i}: Standard Acc: {top1_avg_correct/total_samples*100:.2f}% | At-Least-One Acc: {at_least_one_correct/total_samples*100:.2f}%")

    print(f"\n🏆 FINAL RESULTS (X3D-S Local Baseline - 30 Views):")
    print("-" * 50)
    print(f"1. Standard Accuracy : {top1_avg_correct/total_samples*100:.2f}%")
    print(f"2. At-Least-One Acc  : {at_least_one_correct/total_samples*100:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_list', type=str, required=True)
    parser.add_argument('--val_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)
    
    args = parser.parse_args()
    main(args)