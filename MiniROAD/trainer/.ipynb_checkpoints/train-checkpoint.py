from tqdm import tqdm
import torch
from trainer.train_builder import TRAINER

@TRAINER.register("OAD_PHASE2")
def train_one_epoch_phase2(trainloader, model, criterion, optimizer, scaler, epoch, writer=None, scheduler=None):
    
    # Biến tích lũy loss để theo dõi từng thành phần
    epoch_losses = {
        'total': 0.0,
        # Nếu hàm Loss trả về dict chi tiết thì tốt, 
        # nhưng hiện tại StarRankingLoss trả về 1 cục total.
        # Tuy nhiên, ta vẫn có thể log total loss.
    }
    
    model.train()
    
    # Progress Bar
    pbar = tqdm(trainloader, desc=f'Epoch:{epoch} [P2-Calibration]', postfix=f'lr: {optimizer.param_groups[0]["lr"]:.7f}')

    for it, batch_data in enumerate(pbar):
        
        # --- 1. DATA PREPARATION ---
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.cuda(non_blocking=True)
                
        rgb_input = batch_data['rgb']
        flow_input = batch_data['flow']
        
        # --- 2. FORWARD PASS ---
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed Precision
        if scaler is not None:
            amp_context = torch.cuda.amp.autocast()
        else:
            from contextlib import nullcontext
            amp_context = nullcontext()

        with amp_context:
            # A. Model Forward
            # MROAD_P2 trả về: z_anchor, z_aug, z_shuf, z_shuf_k, queues
            out_dict = model(rgb_input, flow_input)
            
            # B. Loss Calculation (Star Ranking Loss)
            # Loss này tự động tính Consist + Rank + Sep bên trong
            loss = criterion(out_dict, batch_data)

        # --- 3. BACKPROPAGATION ---
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        # --- 4. SELECTIVE QUEUE UPDATE (SỰ KHÁC BIỆT LỚN NHẤT) ---
        # Ta không update mù quáng nữa.
        # Ta gọi hàm update thông minh của Model Phase 2
        with torch.no_grad():
            # Cần truyền cả out_dict (chứa feature teacher & feature shuffle teacher)
            # và batch_data (chứa overlap info để lọc)
            
            # Lưu ý: Cần chắc chắn model là MROAD_P2 (hoặc được wrap bởi DDP)
            if hasattr(model, 'module'):
                model.module.update_queues_phase2(out_dict, batch_data)
            else:
                model.update_queues_phase2(out_dict, batch_data)

        # --- 5. LOGGING ---
        loss_val = loss.item()
        epoch_losses['total'] += loss_val
        
        if writer is not None:
            global_step = it + epoch * len(trainloader)
            writer.add_scalar("Train/Total Loss", loss_val, global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            
        # Update progress bar realtime
        pbar.set_postfix({'loss': f"{loss_val:.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.7f}"})

    # Trả về trung bình loss của epoch
    avg_loss = epoch_losses['total'] / len(trainloader)
    return {'total': avg_loss}

@TRAINER.register("OAD_PHASE1")
def train_one_epoch_phase1(trainloader, model, criterion, optimizer, scaler, epoch, writer=None, scheduler=None):
    
    # Biến tích lũy loss để theo dõi
    epoch_losses = {
        'total': 0.0
    }
    
    model.train()
    
    # Progress Bar với thông tin rõ ràng
    pbar = tqdm(trainloader, desc=f'Epoch:{epoch} [P1-Clustering]', postfix=f'lr: {optimizer.param_groups[0]["lr"]:.7f}')

    for it, batch_data in enumerate(pbar):
        
        # --- 1. DATA PREPARATION ---
        # Move to GPU: Dataloader trả về dict, ta loop qua để đẩy lên GPU
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.cuda(non_blocking=True)
                
        rgb_input = batch_data['rgb']
        flow_input = batch_data['flow']
        
        # --- 2. FORWARD PASS ---
        optimizer.zero_grad(set_to_none=True)
        
        # AMP Context (Mixed Precision)
        if scaler is not None:
            amp_context = torch.cuda.amp.autocast()
        else:
            from contextlib import nullcontext
            amp_context = nullcontext()

        with amp_context:
            # A. Model Forward
            # MROAD_P1 trả về: z_anchor, z_aug, z_shuf, queues
            out_dict = model(rgb_input, flow_input)
            
            # B. Loss Calculation (SupCon)
            # Truyền batch_data vào loss để lấy queue_id (quan trọng!)
            loss = criterion(out_dict, batch_data)

        # --- 3. BACKPROPAGATION ---
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        # --- 4. MOMENTUM QUEUE UPDATE (BẮT BUỘC) ---
        # Sau khi update weights, ta update memory bank
        with torch.no_grad():
            # Lấy feature từ Teacher (z_aug) để đẩy vào Queue
            # queue_ids từ dataset giúp model biết đẩy vào ngăn nào (Action A, Action B hay BG)
            z_teacher = out_dict['z_aug'].detach()
            queue_ids = batch_data['queue_id'].detach()
            
            # Gọi hàm update của Model
            model.update_queues(z_teacher, queue_ids)

        # --- 5. LOGGING ---
        loss_val = loss.item()
        epoch_losses['total'] += loss_val
        
        if writer is not None:
            global_step = it + epoch * len(trainloader)
            writer.add_scalar("Train/SupCon Loss", loss_val, global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            
        # Update progress bar realtime
        pbar.set_postfix({'loss': f"{loss_val:.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.7f}"})

    # Trả về trung bình loss của epoch
    avg_loss = epoch_losses['total'] / len(trainloader)
    return {'total': avg_loss}

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