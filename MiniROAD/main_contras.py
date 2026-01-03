import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
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
    print(f"--> [Warm-up] Starting to fill Memory Bank (Size per class: {model.K})...")
    model.eval() 
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Filling Queue")
        
        for batch_data in pbar:
            rgb_anchor = batch_data['rgb_anchor'].to(device, non_blocking=True)
            flow_anchor = batch_data['flow_anchor'].to(device, non_blocking=True)
            labels = batch_data['labels'].to(device, non_blocking=True)

            # --- SỬA ĐOẠN NÀY (Thêm labels_per_frame) ---
            labels_per_frame = batch_data.get('labels_per_frame', None)
            if labels_per_frame is not None:
                labels_per_frame = labels_per_frame.to(device, non_blocking=True)
            
            # Truyền đủ 4 tham số: rgb, flow, labels, labels_per_frame
            out_dict = model(rgb_anchor, flow_anchor, labels, labels_per_frame=labels_per_frame)
            # ----------------------------------------------
            
            k_cls = out_dict['k_cls']
            
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

    # Tạo identifier riêng
    identifier = f'Contrastive_{cfg["model"]}_{cfg["data_name"]}'
    if 'rgb_type' in cfg:
        identifier += f'_{cfg["rgb_type"]}'
        
    # result_path là đường dẫn thư mục output thực tế (VD: ./output/Contrastive_..._timestamp)
    result_path = create_outdir(osp.join(cfg['output_path'], identifier))
    
    # Init Logger & Tensorboard
    logger = get_logger(result_path)
    logger.info(f"Running Experiment: {identifier}")
    logger.info(cfg)
    
    writer = SummaryWriter(osp.join(result_path, 'runs')) if args.tensorboard else None

    # --- 2. DATASETS ---
    trainloader = build_data_loader(cfg, mode='train')
    testloader = build_data_loader(cfg, mode='test')

    # --- 3. MODEL & CRITERION ---
    model = build_model(cfg, device) 
    evaluate = build_eval(cfg)
    train_one_epoch = build_trainer(cfg)
    criterion = build_criterion(cfg, device) 

    # --- 4. OPTIMIZER ---
    optim_cls = torch.optim.AdamW if cfg.get('optimizer', 'Adam') == 'AdamW' else torch.optim.Adam
    optimizer = optim_cls([
        {'params': model.parameters(), 'initial_lr': cfg['lr']}
    ], lr=cfg['lr'], weight_decay=cfg.get("weight_decay", 1e-4))

    # --- 5. SCHEDULER ---
    scheduler = None
    if args.lr_scheduler:
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
        checkpoint = torch.load(args.eval, map_location=device)
        # Handle state dict keys
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        
        cac_score = evaluate(testloader, model, criterion, epoch=0, writer=None)
        logger.info(f'{cfg["task"]} Evaluation -> CAC Score (Action): {cac_score:.2f}%')
        exit()

    # --- 7. TRAINING LOOP ---
    
    # Lấp đầy Memory Bank trước khi train
    fill_memory_bank(trainloader, model, device)

    best_score, best_epoch = 0.0, 0
    
    for epoch in range(1, cfg['num_epoch'] + 1):
        # A. TRAIN ONE EPOCH
        train_loss = train_one_epoch(
            trainloader, model, criterion, optimizer, scaler, epoch, writer, scheduler=None
        )
        
        # B. SHUFFLE DATASET
        if hasattr(trainloader.dataset, 'shuffle_indices'):
            trainloader.dataset.shuffle_indices()
        
        # C. VALIDATION
        cac_score = evaluate(testloader, model, criterion, epoch, writer)
        
        # D. SCHEDULER STEP
        if scheduler is not None:
            scheduler.step()
        
        # E. SAVE CHECKPOINT (MỌI EPOCH)
        checkpoint = {
            'epoch': epoch, # Lưu epoch hiện tại
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'best_cac': best_score
        }
        
        # 1. Lưu latest (đè lên nhau để resume)
        torch.save(checkpoint, osp.join(result_path, 'latest_model.pth'))
        
        # 2. Lưu từng epoch (để visualize sau này) - Lưu vào subfolder 'ckpts' cho gọn
        ckpt_dir = osp.join(result_path, 'ckpts')
        os.makedirs(ckpt_dir, exist_ok=True)
        
        epoch_save_path = osp.join(ckpt_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_save_path)
        # print(f"--> Saved checkpoint: {epoch_save_path}") # Comment bớt cho đỡ rác log
            
        # 3. Lưu Best Model
        if cac_score > best_score:
            best_score = cac_score
            best_epoch = epoch
            best_save_path = osp.join(result_path, 'best_model.pth')
            torch.save(checkpoint, best_save_path) # Lưu cả checkpoint thay vì chỉ state_dict
            logger.info(f"--> New Best CAC: {best_score:.2f}%! Saved best_model.pth")

        # F. LOGGING
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f'Epoch {epoch}/{cfg["num_epoch"]} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val CAC: {cac_score:.2f}% (Best: {best_score:.2f}% @ Ep{best_epoch}) | '
            f'LR: {current_lr:.7f}'
        )
        
    logger.info(f"Training Finished. Best CAC: {best_score:.2f}% at Epoch {best_epoch}")

if __name__ == '__main__':
    main()