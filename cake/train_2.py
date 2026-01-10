import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

import torchvision.transforms as T
import numpy as np
import os
import logging
import argparse
from tqdm import tqdm
import decord
import random
import shutil
import sys
import time
import json

# --- CONFIG ---
decord.bridge.set_bridge('torch')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# --- IMPORTS ---
try:
    # [THAY ĐỔI 1] Import từ cake_baseline
    from cake_baseline import BioX3D_Student 
    from mgd import MGDLoss3D
    from teacher_utils import TeacherPipeline
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    print("⚠️ Hãy chắc chắn bạn đã lưu file model Conv3d thường thành 'cake_baseline.py'")
    sys.exit(1)

# ==============================================================================
# 1. UTILS
# ==============================================================================
def setup_logger(save_dir, resume=False):
    log_file = os.path.join(save_dir, 'train_log.txt')
    if not resume and os.path.exists(log_file):
        os.remove(log_file)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )

def save_config(args, save_dir):
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logging.info(f"💾 Config saved to: {config_path}")

class AverageMeter(object):
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    if is_best: shutil.copyfile(path, os.path.join(save_dir, 'model_best.pth'))

def update_odconv_temperature(model, epoch, total_epochs):
    # Hàm này vẫn giữ lại được. 
    # Với Conv3d thường, hasattr(m, 'update_temperature') sẽ trả về False -> Không làm gì cả. An toàn.
    start_temp, end_temp = 5.0, 1.0
    current_temp = start_temp - (start_temp - end_temp) * (epoch / total_epochs)
    for m in model.modules():
        if hasattr(m, 'update_temperature'): m.update_temperature(current_temp)
    return current_temp

class X3D_Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1).cuda()
        self.std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1).cuda()
    def forward(self, x): return (x / 255.0 - self.mean) / self.std

# ==============================================================================
# 2. DATASET
# ==============================================================================
class RGBVideoDataset(Dataset):
    def __init__(self, video_list_file, root_dir, clip_len=13, crop_size=224, stride=6, mode='train'):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.stride = stride
        self.mode = mode
        self.samples = []
        if not os.path.exists(video_list_file): raise FileNotFoundError(f"Missing: {video_list_file}")
        with open(video_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) >= 2:
                    self.samples.append((" ".join(parts[:-1]), int(parts[-1])))
                elif len(parts) == 1:
                    self.samples.append((parts[0], -1))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        full_path = os.path.join(self.root_dir, path)
        try:
            vr = decord.VideoReader(full_path)
            total = len(vr)
            
            clip_span = (self.clip_len - 1) * self.stride + 1
            
            if self.mode == 'train': 
                start = random.randint(0, max(0, total - clip_span))
            else: 
                start = max(0, (total - clip_span) // 2)
            
            indices = [min(start + i * self.stride, total - 1) for i in range(self.clip_len)]
            
            buffer = vr.get_batch(indices) 
            if not isinstance(buffer, torch.Tensor): buffer = torch.from_numpy(buffer.asnumpy())
            buffer = buffer.permute(0, 3, 1, 2).float() 
            buffer = F.interpolate(buffer, size=(256, 256), mode='bilinear', align_corners=False)
            
            if self.mode == 'train':
                i, j, h, w = T.RandomCrop.get_params(buffer, output_size=(self.crop_size, self.crop_size))
                buffer = T.functional.crop(buffer, i, j, h, w)
                if random.random() < 0.5: buffer = T.functional.hflip(buffer)
            else:
                buffer = T.CenterCrop(self.crop_size)(buffer)
            return buffer.permute(1, 0, 2, 3), label
        except Exception as e:
            return torch.zeros(3, self.clip_len, self.crop_size, self.crop_size), -1

# ==============================================================================
# 3. VALIDATION
# ==============================================================================
def validate(val_loader, student, teacher, criterion_cls, criterion_distill, device, normalizer, args):
    losses = AverageMeter()
    acc_rgb = AverageMeter()
    acc_flow = AverageMeter()
    distill_meter = AverageMeter()
    student.eval()
    
    use_embed = (args.distill_type == 'embed')

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Val", dynamic_ncols=True)
        for i, (raw_clip, labels) in enumerate(pbar):
            raw_clip = raw_clip.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            teacher_feat = teacher.get_teacher_features(raw_clip, return_embedding= use_embed)
            norm_clip = normalizer(raw_clip)
            
            if use_embed:
                rgb_logits, flow_logits, _, _, _, flow_embed = student(norm_clip, return_embeddings=True)
                
                if teacher_feat.dim() == 5:
                    t_target = F.adaptive_avg_pool3d(teacher_feat, 1).flatten(1)
                else:
                    t_target = teacher_feat 

                ones_target = torch.ones(raw_clip.size(0)).to(device)
                l_distill = criterion_distill(flow_embed, t_target, ones_target)
                
            else:
                rgb_logits, flow_logits, _, flow_hallucinated = student(norm_clip)
                l_distill = criterion_distill(flow_hallucinated, teacher_feat)
            
            l_rgb = criterion_cls(rgb_logits, labels)
            l_flow = criterion_cls(flow_logits, labels)
            loss = l_rgb + 0.5 * l_flow + l_distill
            
            a1_rgb, _ = accuracy(rgb_logits, labels, topk=(1, 5))
            a1_flow, _ = accuracy(flow_logits, labels, topk=(1, 5))
            
            losses.update(loss.item(), raw_clip.size(0))
            acc_rgb.update(a1_rgb[0].item(), raw_clip.size(0))
            acc_flow.update(a1_flow[0].item(), raw_clip.size(0))
            distill_meter.update(l_distill.item(), raw_clip.size(0))
            
            pbar.set_postfix(AR=f"{acc_rgb.avg:.2f}", AF=f"{acc_flow.avg:.2f}")
            
    logging.info(f" * Val: RGB_Acc {acc_rgb.avg:.2f}% | Flow_Acc {acc_flow.avg:.2f}% | Distill {distill_meter.avg:.4f}")
    return acc_rgb.avg

# ==============================================================================
# 4. MAIN
# ==============================================================================
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    setup_logger(args.save_dir, resume=args.resume is not None)
    
    if not args.resume: save_config(args, args.save_dir)
    
    device = torch.device("cuda")
    logging.info(f"Config: {vars(args)}")

    # 1. Models
    logging.info(">>> Init Models (BASELINE: STANDARD CONV)...")
    
    # [THAY ĐỔI 2] Bỏ clone_idx=args.clone_idx vì Baseline Model không dùng
    student = BioX3D_Student(clip_len=args.clip_len, num_classes=400, feature_dim=192).to(device)
    
    if args.student_pretrained and os.path.exists(args.student_pretrained):
        student.load_pretrained_weights(rgb_path=args.student_pretrained, flow_teacher_path=args.flow_teacher_weights)

    if args.freeze_rgb:
        logging.info("❄️ FREEZING RGB BACKBONE & HEAD (Only Training Flow Branch) ❄️")
        for param in student.blocks.parameters(): param.requires_grad = False
        for param in student.head.parameters(): param.requires_grad = False
    
    teacher = TeacherPipeline(args.raft_weights, args.flow_teacher_weights, device=device)

    # 2. Losses
    criterion_cls = nn.CrossEntropyLoss().to(device)
    
    if args.distill_type == 'mgd':
        logging.info(f"👉 Using MGD Loss with alpha={args.alpha_distill}")
        criterion_distill = MGDLoss3D(192, 192, alpha_mgd=args.alpha_distill, lambda_mgd=0.65).to(device)
    
    elif args.distill_type == 'embed':
        logging.info(f"👉 Using Cosine Embedding Loss with alpha={args.alpha_distill}")
        class CosineWrapper(nn.Module):
            def __init__(self, alpha): 
                super().__init__()
                self.alpha = alpha
                self.cosine = nn.CosineEmbeddingLoss()
            def forward(self, s, t, target): 
                return self.cosine(s, t, target) * self.alpha
        criterion_distill = CosineWrapper(args.alpha_distill).to(device)
        
    else: 
        logging.info(f"👉 Using MSE Loss with alpha={args.alpha_distill}")
        class MSEWrapper(nn.Module):
            def __init__(self, alpha): super().__init__(); self.alpha = alpha; self.mse = nn.MSELoss()
            def forward(self, s, t): return self.mse(s, t) * self.alpha
        criterion_distill = MSEWrapper(args.alpha_distill).to(device)

    normalizer = X3D_Normalizer()
    scaler = GradScaler()

    # 3. Optimizer
    backbone_params = []
    new_layers_params = []
    
    for name, param in student.named_parameters():
        if not param.requires_grad: continue 
        if 'flow_adapter' in name or 'hallucinator' in name:
            new_layers_params.append(param)
        else:
            backbone_params.append(param)   

    generator_params = []
    if args.distill_type == 'mgd':
        for name, param in criterion_distill.named_parameters():
            if param.requires_grad: generator_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr_backbone}, 
        {'params': new_layers_params, 'lr': args.lr_new_layers},
        {'params': generator_params, 'lr': args.lr_new_layers}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    # 4. Resume
    start_epoch = 0
    best_acc1 = 0.0
    if args.resume and os.path.isfile(args.resume):
        logging.info(f"🔄 Resuming: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        student.load_state_dict(ckpt['state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        if args.distill_type == 'mgd' and 'mgd_state_dict' in ckpt:
            criterion_distill.load_state_dict(ckpt['mgd_state_dict'])
        start_epoch = ckpt.get('epoch', 0) 
        best_acc1 = ckpt.get('best_acc1', 0.0)

    # 5. Loaders
    logging.info(">>> Creating Loaders...")
    train_ds = RGBVideoDataset(args.train_list, args.train_root, clip_len=args.clip_len, mode='train', stride=args.stride)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True)
    
    val_loader = None
    if args.val_list:
        val_root = args.val_root if args.val_root else args.train_root
        val_ds = RGBVideoDataset(args.val_list, val_root, clip_len=args.clip_len, mode='val', stride=args.stride)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 6. Loop
    logging.info(f">>> START TRAINING (Method: {args.distill_type.upper()})...")
    
    use_embed = (args.distill_type == 'embed')

    for epoch in range(start_epoch, args.epochs):
        student.train()
        if args.freeze_rgb:
            student.blocks.eval()
            student.head.eval()
            
        curr_temp = update_odconv_temperature(student, epoch, args.epochs)
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", dynamic_ncols=True)
        loss_m = AverageMeter(); distill_m = AverageMeter(); cos_m = AverageMeter()
        acc_r_m = AverageMeter(); acc_f_m = AverageMeter()
        loss_r_m = AverageMeter(); loss_f_m = AverageMeter()
        
        for i, (raw_clip, labels) in enumerate(pbar):
            raw_clip = raw_clip.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.no_grad():
                teacher_feat = teacher.get_teacher_features(raw_clip, return_embedding= use_embed)
            
            with autocast(device_type='cuda', enabled=True):
                norm_clip = normalizer(raw_clip)
                
                if use_embed:
                    rgb_logits, flow_logits, _, _, _, flow_embed = student(norm_clip, return_embeddings=True)
                    
                    if teacher_feat.dim() == 5:
                        print("eacher trả về 5D (B,C,T,H,W)")
                    else:
                        t_target = teacher_feat
                    
                    ones_target = torch.ones(raw_clip.size(0)).to(device)
                    l_distill = criterion_distill(flow_embed, t_target, ones_target)
                    
                    with torch.no_grad():
                         cos_sim = F.cosine_similarity(flow_embed, t_target).mean()
                         cos_m.update(cos_sim.item())
                else:
                    rgb_logits, flow_logits, _, flow_hallucinated = student(norm_clip)
                    l_distill = criterion_distill(flow_hallucinated, teacher_feat)
                    
                    with torch.no_grad():
                        flat_s = flow_hallucinated.view(flow_hallucinated.size(0), -1)
                        flat_t = teacher_feat.view(teacher_feat.size(0), -1)
                        cos_sim = F.cosine_similarity(flat_s, flat_t).mean()
                        cos_m.update(cos_sim.item())

                valid_idx = labels > -1
                if valid_idx.sum() > 0:
                    # 1. Loss RGB (Vẫn giữ hoặc tắt tùy bạn, hiện tại đang freeze nên nó = 0)
                    if not args.freeze_rgb:
                        l_rgb = criterion_cls(rgb_logits[valid_idx], labels[valid_idx])
                    else:
                        l_rgb = torch.tensor(0.0, device=device)
                        
                    # 2. Loss Flow Classification -> CẦN TẮT ĐỂ TRÁNH GIAN LẬN RGB
                    # l_flow = criterion_cls(flow_logits[valid_idx], labels[valid_idx]) # <--- COMMENT DÒNG NÀY
                    l_flow = torch.tensor(0.0, device=device) # <--- GÁN VỀ 0
                    
                    # Vẫn tính Accuracy để theo dõi (nhưng không backprop)
                    a_flow = accuracy(flow_logits[valid_idx], labels[valid_idx], topk=(1,))
                    acc_f_m.update(a_flow[0].item(), valid_idx.sum().item())
                    # loss_f_m.update(l_flow.item()) # Comment dòng này
                else:
                    l_rgb = l_flow = torch.tensor(0.0, device=device)
                
                # --- TỔNG LOSS MỚI ---
                # Chỉ còn Distill Loss chịu trách nhiệm hướng dẫn Hallucinator
                l_distill = criterion_distill(flow_hallucinated, teacher_feat)
                distill_m.update(l_distill.item())
                loss = l_rgb + l_distill

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=20.0)
            scaler.step(optimizer)
            scaler.update()
            
            loss_m.update(loss.item())
            distill_m.update(l_distill.item())
            
            pbar.set_postfix(
                Lr=f"{loss_r_m.val:.2f}", Lf=f"{loss_f_m.val:.2f}",
                Ar=f"{acc_r_m.val:.1f}", Af=f"{acc_f_m.val:.1f}",
                Dis=f"{distill_m.val:.2f}", Cos=f"{cos_m.val:.2f}"
            )
        
        scheduler.step()
        logging.info(f"Ep {epoch+1}: RGB_Acc={acc_r_m.avg:.2f}% | Flow_Acc={acc_f_m.avg:.2f}% | Cos={cos_m.avg:.3f}")

        acc1 = 0.0
        if val_loader:
            acc1 = validate(val_loader, student, teacher, criterion_cls, criterion_distill, device, normalizer, args)
            
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        
        save_dict = {
            'epoch': epoch + 1,
            'state_dict': student.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
        }
        if args.distill_type == 'mgd': save_dict['mgd_state_dict'] = criterion_distill.state_dict()
            
        save_checkpoint(save_dict, is_best, args.save_dir, filename='last.pth')
        if (epoch + 1) % 5 == 0:
            save_checkpoint(save_dict, False, args.save_dir, filename=f'checkpoint_ep{epoch+1}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths & Weights
    parser.add_argument('--train_list', type=str, required=True)
    parser.add_argument('--train_root', type=str, required=True)
    parser.add_argument('--val_list', type=str, default=None)
    parser.add_argument('--val_root', type=str, default=None)
    parser.add_argument('--raft_weights', type=str, required=True)
    parser.add_argument('--flow_teacher_weights', type=str, required=True)
    parser.add_argument('--student_pretrained', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    
    # Params
    parser.add_argument('--save_dir', type=str, default='./checkpoints_k400')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--clip_len', type=int, default=13)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr_backbone', type=float, default=1e-4)
    parser.add_argument('--lr_new_layers', type=float, default=1e-3)
    
    # [NEW] Config Stride
    parser.add_argument('--stride', type=int, default=6, help="Temporal sampling stride (default: 6)")
    
    # Config Distill
    parser.add_argument('--distill_type', type=str, default='mgd', choices=['mgd', 'mse', 'embed'])
    parser.add_argument('--alpha_distill', type=float, default=1.0)
    
    # [NEW] Config Freeze
    parser.add_argument('--freeze_rgb', action='store_true', help="Nếu bật, sẽ đóng băng toàn bộ nhánh RGB (Backbone + Head)")

    args = parser.parse_args()
    main(args)