from tqdm import tqdm
import torch
from trainer.train_builder import TRAINER

@TRAINER.register("CONTRASTIVE")
def train_one_epoch(trainloader, model, criterion, optimizer, scaler, epoch, writer=None, scheduler=None):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(trainloader, desc=f'Epoch:{epoch} [Contrastive Train]', leave=True)
    
    for it, batch_data in enumerate(pbar):
        rgb_anchor = batch_data['rgb_anchor'].cuda(non_blocking=True)
        flow_anchor = batch_data['flow_anchor'].cuda(non_blocking=True)
        labels = batch_data['labels'].cuda(non_blocking=True) # Vẫn giữ để Generator (nếu cần)
        
        # [MỚI] Lấy targets_multihot
        targets_multihot = batch_data['targets_multihot'].cuda(non_blocking=True)
        
        labels_per_frame = batch_data.get('labels_per_frame', None)
        if labels_per_frame is not None:
            labels_per_frame = labels_per_frame.cuda(non_blocking=True)
        
        # 2. FORWARD PASS
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out_dict = model(rgb_anchor, flow_anchor, labels, labels_per_frame=labels_per_frame)
                # [SỬA] Truyền targets_multihot vào Loss
                loss = criterion(out_dict, targets_multihot)
            
            # 3. BACKWARD
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out_dict = model(rgb_anchor, flow_anchor, labels, labels_per_frame=labels_per_frame)
            # [SỬA] Truyền targets_multihot vào Loss
            loss = criterion(out_dict, targets_multihot)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        # 4. MEMORY BANK UPDATE
        with torch.no_grad():
            k_cls = out_dict['k_cls'].detach()
            # [SỬA] Update Queue với Multi-hot
            model.update_queue(k_cls, targets_multihot)

        # 5. LOGGING
        loss_val = loss.item()
        epoch_loss += loss_val
        current_lr = optimizer.param_groups[0]["lr"]

        postfix_dict = {'loss': f"{loss_val:.4f}", 'lr': f"{current_lr:.7f}"}
        
        if it % 50 == 0:
             with torch.no_grad():
                # [SỬA LOGGING] Model mới chỉ trả về q_cls và k_cls
                q_cls = out_dict['q_cls'] 
                k_cls = out_dict['k_cls']      

                # Tính độ tương đồng Student-Teacher
                sim_cls = torch.einsum('nc,nc->n', [q_cls, k_cls]).mean().item()
                
                postfix_dict['Sim'] = f"{sim_cls:.2f}"

        pbar.set_postfix(postfix_dict)

        if writer is not None:
            global_step = it + (epoch - 1) * len(trainloader)
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