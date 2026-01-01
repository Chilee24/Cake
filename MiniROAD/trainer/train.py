from tqdm import tqdm
import torch
from trainer.train_builder import TRAINER

@TRAINER.register("CONTRASTIVE")
def train_one_epoch(trainloader, model, criterion, optimizer, scaler, epoch, writer=None, scheduler=None):
    model.train()
    epoch_loss = 0.0
    
    # Thanh tiến trình
    pbar = tqdm(trainloader, desc=f'Epoch:{epoch} [Contrastive Train]', leave=True)
    
    for it, batch_data in enumerate(pbar):
    # # --- DEBUG ---
    #     print(type(batch_data))
    #     print(batch_data)
    #     # -------------
        rgb_anchor = batch_data['rgb_anchor'].cuda(non_blocking=True)
        flow_anchor = batch_data['flow_anchor'].cuda(non_blocking=True)
        rgb_shuff = batch_data['rgb_shuff'].cuda(non_blocking=True)
        flow_shuff = batch_data['flow_shuff'].cuda(non_blocking=True)
        labels = batch_data['labels'].cuda(non_blocking=True)
        
        # 2. FORWARD PASS
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # Model nhận 5 tham số đầu vào
                out_dict = model(rgb_anchor, flow_anchor, rgb_shuff, flow_shuff, labels)
                
                # Loss tính dựa trên output dictionary và labels
                loss = criterion(out_dict, labels)
            
            # 3. BACKWARD & OPTIMIZER (Mixed Precision)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard Precision
            out_dict = model(rgb_anchor, flow_anchor, rgb_shuff, flow_shuff, labels)
            loss = criterion(out_dict, labels)
            loss.backward()
            optimizer.step()

        # 4. MEMORY BANK UPDATE (QUAN TRỌNG NHẤT)
        # Cần đẩy key mới vào queue sau khi update trọng số
        with torch.no_grad():
            # Lấy key teacher (k_cls) từ output
            k_cls = out_dict['k_cls'].detach()
            
            # Xử lý trường hợp chạy nhiều GPU (DataParallel)
            if hasattr(model, 'module'):
                model.module.update_queue(k_cls, labels)
            else:
                model.update_queue(k_cls, labels)

        # 5. LR SCHEDULER STEP (Per Iteration)
        # Contrastive Learning cần Warmup từng step để không vỡ gradient lúc đầu
        if scheduler is not None:
            # Chỉ step nếu scheduler là loại update theo batch (vd: OneCycleLR, CosineWithWarmup)
            # Nếu dùng StepLR (theo epoch) thì bỏ dòng này, gọi ở ngoài vòng for
            scheduler.step()

        # 6. LOGGING
        loss_val = loss.item()
        epoch_loss += loss_val
        current_lr = optimizer.param_groups[0]["lr"]

        # Update tqdm bar
        pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{current_lr:.7f}")

        if writer is not None:
            global_step = it + epoch * len(trainloader)
            writer.add_scalar("Train/Loss", loss_val, global_step)
            writer.add_scalar("Train/LR", current_lr, global_step)
            
            # Log thêm accuracy của task phụ (VD: tỉ lệ chọn đúng positive) nếu cần
            # writer.add_scalar("Train/Gen_Score_Mean", out_dict['gen_scores'].mean().item(), global_step)

    return epoch_loss / len(trainloader)

@TRAINER.register("OAD")
def train_one_epoch(trainloader, model, criterion, optimizer, scaler, epoch, writer=None, scheduler=None):
    epoch_loss = 0
    for it, (rgb_input, flow_input, target) in enumerate(tqdm(trainloader, desc=f'Epoch:{epoch} Training', postfix=f'lr: {optimizer.param_groups[0]["lr"]:.7f}')):
        rgb_input, flow_input, target = rgb_input.cuda(), flow_input.cuda(), target.cuda()
        model.train()
        if scaler != None:
            with torch.cuda.amp.autocast():    
                out_dict = model(rgb_input, flow_input) 
                loss = criterion(out_dict, target)   
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out_dict = model(rgb_input, flow_input) 
            loss = criterion(out_dict, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        if writer != None:
            writer.add_scalar("Train Loss", loss.item(), it+epoch*len(trainloader))
    return epoch_loss

@TRAINER.register("ANTICIPATION")
def ant_train_one_epoch(trainloader, model, criterion, optimizer, scaler, epoch, writer=None, scheduler=None):
    epoch_loss = 0
    for it, (rgb_input, flow_input, target, ant_target) in enumerate(tqdm(trainloader, desc=f'Epoch:{epoch} Training', postfix=f'lr: {optimizer.param_groups[0]["lr"]:.7f}')):
        rgb_input, flow_input, target, ant_target = rgb_input.cuda(), flow_input.cuda(), target.cuda(), ant_target.cuda()
        model.train()
        if scaler != None:
            with torch.cuda.amp.autocast():    
                out_dict = model(rgb_input, flow_input) 
                loss = criterion(out_dict, target, ant_target)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: 
            out_dict = model(rgb_input, flow_input) 
            loss = criterion(out_dict, target, ant_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
        if writer != None:
            writer.add_scalar("Train Loss", loss.item(), it+epoch*len(trainloader))
    return epoch_loss