import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import tqdm
import yaml
import os
import os.path as osp
import numpy as np
import random
from utils import get_logger, create_outdir
from model import build_model
from datasets import build_data_loader
from criterions import build_criterion
from trainer import build_trainer, build_eval 

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    
def fill_memory_bank(loader, model, device):
    """
    Chạy 1 epoch (không backprop) để lấp đầy Memory Bank bằng feature thật.
    """
    print(f"--> [Warm-up] Starting to fill Memory Bank (Size per class: {model.K})...")
    
    model.eval() # Chuyển sang eval để tắt Dropout/BatchNorm update (hoặc để train tùy chiến lược)
    # Tuy nhiên, với MoCo, thường ta để teacher chạy ở mode eval để feature ổn định.
    
    with torch.no_grad():
        # Dùng tqdm để xem tiến độ
        pbar = tqdm(loader, desc="Filling Queue")
        
        for batch_data in pbar:
            # 1. Move data to GPU
            rgb_anchor = batch_data['rgb_anchor'].to(device, non_blocking=True)
            flow_anchor = batch_data['flow_anchor'].to(device, non_blocking=True)
            labels = batch_data['labels'].to(device, non_blocking=True)
            
            # 2. Forward qua Teacher (Encoder K)
            # Chúng ta cần truy cập trực tiếp encoder_k để lấy feature sạch
            # Hoặc gọi hàm forward của model nhưng chỉ lấy k_cls
            
            # Cách an toàn nhất: Gọi forward của model (đã có logic tách rgb/flow)
            # Vì ta đang no_grad nên không tốn bộ nhớ cho graph
            # Truyền dummy cho shuff vì không dùng đến
            out_dict = model(rgb_anchor, flow_anchor, rgb_anchor, flow_anchor, labels)
            
            k_cls = out_dict['k_cls']
            
            # 3. Update Queue
            if hasattr(model, 'module'):
                model.module.update_queue(k_cls, labels)
            else:
                model.update_queue(k_cls, labels)
                
    print("--> [Warm-up] Memory Bank filled with real features!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/contrastive_oad.yaml') 
    parser.add_argument('--eval', type=str, default=None)
    parser.add_argument('--amp', action='store_true', help='Use Mixed Precision Training')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--lr_scheduler', action='store_true', default=True) 
    parser.add_argument('--no_rgb', action='store_true')
    parser.add_argument('--no_flow', action='store_true')
    args = parser.parse_args()

    # --- 1. CONFIG SETUP ---
    opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(vars(args))
    cfg = opt

    set_seed(20)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Tạo identifier riêng cho bài toán Contrastive
    identifier = f'Contrastive_{cfg["model"]}_{cfg["data_name"]}'
    # Nếu có config feature type thì thêm vào tên
    if 'rgb_type' in cfg:
        identifier += f'_{cfg["rgb_type"]}'
        
    result_path = create_outdir(osp.join(cfg['output_path'], identifier))
    
    # Init Logger & Tensorboard
    logger = get_logger(result_path)
    logger.info(f"Running Experiment: {identifier}")
    logger.info(cfg)
    
    writer = SummaryWriter(osp.join(result_path, 'runs')) if args.tensorboard else None

    # --- 2. DATASETS ---
    # mode='train' trả về ContrastiveOADDataset (có shuffle_indices)
    trainloader = build_data_loader(cfg, mode='train')
    # mode='test' trả về ContrastiveOADDataset (sequential)
    testloader = build_data_loader(cfg, mode='test')

    # --- 3. MODEL & CRITERION ---
    # Trả về: ContrastiveMROADMultiQueue
    model = build_model(cfg, device) 
    evaluate = build_eval(cfg)
    train_one_epoch = build_trainer(cfg)
    criterion = build_criterion(cfg, device) 

    # --- 4. OPTIMIZER ---
    optim_cls = torch.optim.AdamW if cfg.get('optimizer', 'Adam') == 'AdamW' else torch.optim.Adam
    optimizer = optim_cls([
        {'params': model.parameters(), 'initial_lr': cfg['lr']}
    ], lr=cfg['lr'], weight_decay=cfg.get("weight_decay", 1e-4))

    # --- 5. SCHEDULER (COSINE ANNEALING) ---
    scheduler = None
    if args.lr_scheduler:
        # CosineAnnealingLR update theo EPOCH
        # T_max: Tổng số epoch train
        # eta_min: Learning rate tối thiểu (cuối chu kỳ)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg['num_epoch'], 
            eta_min=cfg.get('min_lr', 1e-6)
        )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f'Dataset: {cfg["data_name"]} | Model: {cfg["model"]}')    
    logger.info(f'LR:{cfg["lr"]} | Decay:{cfg["weight_decay"]} | Window:{cfg["window_size"]} | Batch:{cfg["batch_size"]}') 
    logger.info(f'Epochs:{cfg["num_epoch"]} | Params:{total_params/1e6:.1f}M | Optimizer: {cfg.get("optimizer", "Adam")}')
    logger.info(f'Output Path:{result_path}')

    # --- 6. EVALUATION MODE ---
    if args.eval is not None:
        logger.info(f"Loading checkpoint from {args.eval}...")
        model.load_state_dict(torch.load(args.eval))
        # Valid 1 epoch để lấy chỉ số CAC
        cac_score = evaluate(testloader, model, criterion, epoch=0, writer=None)
        logger.info(f'{cfg["task"]} Evaluation -> CAC Score (Action): {cac_score:.2f}%')
        exit()

    # --- 7. TRAINING LOOP ---
    
    # Lấp đầy Memory Bank trước khi train
    fill_memory_bank(trainloader, model, device)

    best_score, best_epoch = 0.0, 0
    
    for epoch in range(1, cfg['num_epoch'] + 1):
        # A. TRAIN ONE EPOCH
        # Lưu ý: Pass scheduler=None vào train_one_epoch vì ta step ở ngoài vòng lặp (theo epoch)
        train_loss = train_one_epoch(
            trainloader, model, criterion, optimizer, scaler, epoch, writer, scheduler=None
        )
        
        # B. SHUFFLE DATASET (Quan trọng!)
        # Tạo lại lưới sliding window ngẫu nhiên cho epoch sau
        if hasattr(trainloader.dataset, 'shuffle_indices'):
            trainloader.dataset.shuffle_indices()
        
        # C. VALIDATION
        # Sử dụng CAC Score làm thước đo chính thay vì mAP
        cac_score = evaluate(testloader, model, criterion, epoch, writer)
        
        # D. SCHEDULER STEP (Per Epoch)
        if scheduler is not None:
            scheduler.step()
            
        # E. SAVE BEST CHECKPOINT
        if cac_score > best_score:
            best_score = cac_score
            best_epoch = epoch
            save_path = osp.join(result_path, 'ckpts', 'best.pth')
            # Tạo folder ckpts nếu chưa có
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"--> New Best CAC: {best_score:.2f}%! Model saved.")

        # F. LOGGING
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f'Epoch {epoch}/{cfg["num_epoch"]} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val CAC: {cac_score:.2f}% (Best: {best_score:.2f}% @ Ep{best_epoch}) | '
            f'LR: {current_lr:.7f}'
        )
        
    # Rename best model với điểm số cuối cùng
    final_best_path = osp.join(result_path, 'ckpts', f'best_CAC_{best_score:.2f}.pth')
    if osp.exists(osp.join(result_path, 'ckpts', 'best.pth')):
        os.rename(osp.join(result_path, 'ckpts', 'best.pth'), final_best_path)
        
    logger.info(f"Training Finished. Best Model: {final_best_path}")

if __name__ == '__main__':
    main()