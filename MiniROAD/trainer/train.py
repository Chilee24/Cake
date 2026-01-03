from tqdm import tqdm
import torch
from trainer.train_builder import TRAINER

import torch
from tqdm import tqdm

@TRAINER.register("CONTRASTIVE")
def train_one_epoch(trainloader, model, criterion, optimizer, scaler, epoch, writer=None, scheduler=None):
    model.train()
    epoch_loss = 0.0
    
    # Thanh tiến trình
    pbar = tqdm(trainloader, desc=f'Epoch:{epoch} [Contrastive Train]', leave=True)
    
    for it, batch_data in enumerate(pbar):
        rgb_anchor = batch_data['rgb_anchor'].cuda(non_blocking=True)
        flow_anchor = batch_data['flow_anchor'].cuda(non_blocking=True)
        rgb_shuff = batch_data['rgb_shuff'].cuda(non_blocking=True)
        flow_shuff = batch_data['flow_shuff'].cuda(non_blocking=True)
        labels = batch_data['labels'].cuda(non_blocking=True)
        
        # [MỚI] Lấy labels_per_frame để phục vụ Conditional Shuffling
        labels_per_frame = batch_data.get('labels_per_frame', None)
        if labels_per_frame is not None:
            labels_per_frame = labels_per_frame.cuda(non_blocking=True)
        
        # 2. FORWARD PASS
        optimizer.zero_grad(set_to_none=True)
        
        # Hàm model forward đã được cập nhật ở bước trước để nhận labels_per_frame
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out_dict = model(rgb_anchor, flow_anchor, rgb_shuff, flow_shuff, labels, labels_per_frame=labels_per_frame)
                loss = criterion(out_dict, labels)
            
            # 3. BACKWARD & OPTIMIZER (Mixed Precision)
            scaler.scale(loss).backward()
            
            # [QUAN TRỌNG] Unscale trước khi Clip Gradient
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0) # Chống NaN
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard Precision
            out_dict = model(rgb_anchor, flow_anchor, rgb_shuff, flow_shuff, labels, labels_per_frame=labels_per_frame)
            loss = criterion(out_dict, labels)
            loss.backward()
            
            # [QUAN TRỌNG] Clip Gradient chống NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()

        # 4. MEMORY BANK UPDATE
        with torch.no_grad():
            k_cls = out_dict['k_cls'].detach()
            if hasattr(model, 'module'):
                model.module.update_queue(k_cls, labels)
            else:
                model.update_queue(k_cls, labels)

        # 5. LR SCHEDULER STEP
        if scheduler is not None:
            scheduler.step()

        # 6. LOGGING & MONITORING
        loss_val = loss.item()
        epoch_loss += loss_val
        current_lr = optimizer.param_groups[0]["lr"]

        # Monitor nhanh (để kiểm tra xem model có tách được Shuff/Context không)
        # Chỉ hiển thị mỗi 50 steps để đỡ lag
        postfix_dict = {'loss': f"{loss_val:.4f}", 'lr': f"{current_lr:.7f}"}
        
        if it % 50 == 0:
             with torch.no_grad():
                # Lấy Sim Positive vs Sim Shuffle
                q, k = out_dict['q_cls'], out_dict['k_cls']
                q_shuff = out_dict['q_shuff']
                valid_mask = out_dict.get('valid_shuffle_mask', None)
                
                pos_sim = torch.einsum('nc,nc->n', [q, k]).mean().item()
                # Tính Sim giữa Teacher và Shuffle (Logic mới)
                shuff_sim_all = torch.einsum('nc,nc->n', [k, q_shuff])
                
                if valid_mask is not None and valid_mask.sum() > 0:
                    shuff_sim = (shuff_sim_all * valid_mask.float()).sum() / valid_mask.sum()
                    postfix_dict['Pos'] = f"{pos_sim:.2f}"
                    postfix_dict['Shuf'] = f"{shuff_sim.item():.2f}"

        pbar.set_postfix(postfix_dict)

        if writer is not None:
            global_step = it + epoch * len(trainloader)
            writer.add_scalar("Train/Loss", loss_val, global_step)
            writer.add_scalar("Train/LR", current_lr, global_step)

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