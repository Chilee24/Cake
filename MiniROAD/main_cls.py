import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import os
import os.path as osp
from utils import get_logger
from model import build_model
from datasets import build_data_loader
from criterions import build_criterion
from trainer import build_trainer, build_eval
from utils import *

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/miniroad_thumos.yaml')
    parser.add_argument('--eval', type=str, default=None)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--lr_scheduler', action='store_true', default=True)
    parser.add_argument('--no_rgb', action='store_true')
    parser.add_argument('--no_flow', action='store_true')
    args = parser.parse_args()

    # --- 1. SETUP ---
    opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(vars(args))
    cfg = opt

    set_seed(20)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Tạo identifier (Tên thư mục output)
    identifier = f'{cfg["model"]}_{cfg["data_name"]}'
    if cfg.get('freeze_backbone', False):
        identifier += '_FrozenBackbone'
    
    result_path = create_outdir(osp.join(cfg['output_path'], identifier))
    logger = get_logger(result_path)
    logger.info(f"Running Experiment: {identifier}")
    logger.info(cfg)

    writer = SummaryWriter(osp.join(result_path, 'runs')) if args.tensorboard else None
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # --- 2. DATA & MODEL ---
    testloader = build_data_loader(cfg, mode='test')
    trainloader = build_data_loader(cfg, mode='train')
    
    # Model MROAD sẽ tự load contrastive weights nếu có config 'contrastive_path'
    model = build_model(cfg, device)
    
    # Tính params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model: {cfg["model"]} | Total Params: {total_params/1e6:.2f}M | Trainable: {trainable_params/1e6:.2f}M')

    evaluate = build_eval(cfg)
    criterion = build_criterion(cfg, device)
    train_one_epoch = build_trainer(cfg)

    # --- 3. EVALUATION MODE ---
    if args.eval is not None:
        logger.info(f"Loading checkpoint for eval: {args.eval}")
        checkpoint = torch.load(args.eval, map_location=device)
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
        mAP = evaluate(model, testloader, logger)
        logger.info(f'{cfg["task"]} result: {mAP*100:.2f} mAP')
        exit()

    # --- 4. OPTIMIZER ---
    # Tách tham số thành 2 nhóm: Backbone và Head
    backbone_params = []
    head_params = []
    
    # Duyệt qua từng tham số của model
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue # Bỏ qua nếu tham số bị đóng băng cứng
            
        if 'f_classification' in name:
            # Đây là Head -> Dùng LR to
            head_params.append(param)
        else:
            # Đây là Backbone -> Dùng LR nhỏ
            backbone_params.append(param)

    # Lấy LR từ config
    head_lr = cfg['lr']
    # Nếu không config backbone_lr thì mặc định lấy 1/10 head_lr
    backbone_lr = cfg.get('backbone_lr', head_lr * 0.1)

    logger.info(f"--> Optimizer Setup: Head LR={head_lr} | Backbone LR={backbone_lr}")

    # Tạo list các group cho Optimizer
    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr}
    ]
    
    optim_cls = torch.optim.AdamW if cfg.get('optimizer', 'Adam') == 'AdamW' else torch.optim.Adam
    
    # Truyền param_groups vào optimizer
    optimizer = optim_cls(param_groups, weight_decay=cfg.get("weight_decay", 1e-4))

    # --- 5. SCHEDULER ---
    # Sử dụng CosineAnnealingLR (Step theo Epoch)
    scheduler = None
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg['num_epoch'], 
            eta_min=cfg.get('min_lr', 1e-6)
        )

    logger.info(f'Dataset: {cfg["data_name"]}')    
    logger.info(f'LR:{cfg["lr"]} | Window:{cfg["window_size"]} | Batch:{cfg["batch_size"]}') 
    logger.info(f'Output Path:{result_path}')

    # --- 6. TRAINING LOOP ---
    best_mAP, best_epoch = 0.0, 0
    
    for epoch in range(1, cfg['num_epoch'] + 1):
        
        # A. TRAIN
        # Pass scheduler=None vào trong vì ta step ở ngoài vòng lặp epoch
        epoch_loss = train_one_epoch(trainloader, model, criterion, optimizer, scaler, epoch, writer, scheduler=None)
        trainloader.dataset._init_features()

        # C. SCHEDULER STEP
        if scheduler is not None:
            scheduler.step()
        
        # D. EVALUATE
        # Chạy eval mỗi epoch (hoặc chỉnh freq nếu muốn nhanh)
        mAP = evaluate(model, testloader, logger)
        
        # E. SAVE CHECKPOINT
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_mAP': best_mAP
        }
        
        # 1. Save Latest (để resume)
        torch.save(checkpoint, osp.join(result_path, 'latest_model.pth'))
        
        # 2. Save Best
        if mAP > best_mAP:
            best_mAP = mAP
            best_epoch = epoch
            
            # Tạo folder ckpts nếu chưa có
            ckpt_dir = osp.join(result_path, 'ckpts')
            if not osp.exists(ckpt_dir): os.makedirs(ckpt_dir)
                
            torch.save(model.state_dict(), osp.join(ckpt_dir, 'best.pth'))
            
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f'Epoch {epoch}/{cfg["num_epoch"]} | Loss: {epoch_loss:.4f} | mAP: {mAP*100:.2f}% (Best: {best_mAP*100:.2f}% @ Ep{best_epoch}) | LR: {current_lr:.7f}')
        
    # Rename best file at the end
    if osp.exists(osp.join(result_path, 'ckpts', 'best.pth')):
        os.rename(
            osp.join(result_path, 'ckpts', 'best.pth'), 
            osp.join(result_path, 'ckpts', f'best_{best_mAP*100:.2f}.pth')
        )
    logger.info("Training Finished.")

if __name__ == '__main__':
    main()