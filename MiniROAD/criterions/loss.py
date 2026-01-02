import torch
import torch.nn as nn
import torch.nn.functional as F
from criterions.loss_builder import CRITERIONS

@CRITERIONS.register('CONTRASTIVE')
class SupConMROADLoss(nn.Module):
    def __init__(self, cfg):
        super(SupConMROADLoss, self).__init__()
        self.T = cfg.get("temperature", 0.07)
        self.bg_class_idx = cfg.get("bg_class_idx", 0)

    def forward(self, output_dict, labels):
        q = output_dict['q_cls']      # [B, D]
        k = output_dict['k_cls']      # [B, D]
        queues = output_dict['queues'] # [C, D, K]
        
        batch_size = q.shape[0]
        num_classes, dim, K = queues.shape
        device = q.device

        # 1. Flatten Queue
        queue_flat = queues.permute(0, 2, 1).reshape(-1, dim).T.clone().detach() # [D, N_All]
        
        # Tạo nhãn cho Queue: [0, 0...0, 1, 1...1, ...]
        queue_labels = torch.arange(num_classes, device=device).unsqueeze(1).repeat(1, K).view(-1)

        # 2. Tính Logits
        # Positive (q . k) -> [B, 1]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # Queue (q . queue) -> [B, N_All]
        l_queue = torch.mm(q, queue_flat)
        
        # Gom lại: [B, 1 + N_All]
        logits = torch.cat([l_pos, l_queue], dim=1)
        logits /= self.T

        # --- 3. TẠO MASK POSITIVE (ĐỂ TÍNH TỬ SỐ) ---
        mask = torch.zeros_like(logits, device=device)
        mask[:, 0] = 1.0 # Cột 0 (Key) luôn là Positive

        # Logic cho Action: Kéo queue cùng loại
        is_same_class = labels.unsqueeze(1) == queue_labels.unsqueeze(0) # [B, N_All]
        mask[:, 1:] = is_same_class.float()

        # Logic cho Background: Chỉ kéo k (Cột 0), Queue cùng loại KHÔNG PHẢI Positive
        is_bg_sample = (labels == self.bg_class_idx) # [B] -> Index dòng là BG
        mask[is_bg_sample, 1:] = 0.0

        # --- 4. TẠO MASK IGNORE (ĐỂ LOẠI BỎ KHỎI MẪU SỐ - Ý TƯỞNG CỦA BẠN) ---
        # Mục tiêu: Với dòng là BG, các cột Queue cũng là BG sẽ bị loại bỏ.
        
        # Tìm các phần tử trong Queue là Background
        is_bg_in_queue = (queue_labels == self.bg_class_idx) # [N_All]
        
        # Tạo ma trận [B, N_All] xác định vị trí (Dòng BG, Cột BG)
        # is_bg_sample[:, None]: [B, 1]
        # is_bg_in_queue[None, :]: [1, N_All]
        ignore_mask_queue = is_bg_sample.unsqueeze(1) & is_bg_in_queue.unsqueeze(0)
        
        # Mở rộng thêm cột đầu tiên (Key) -> Key không bao giờ bị ignore
        # Shape [B, 1 + N_All]
        ignore_mask = torch.cat([
            torch.zeros(batch_size, 1, device=device, dtype=torch.bool), # Cột k giữ nguyên
            ignore_mask_queue
        ], dim=1)

        # Gán logits tại các vị trí ignore thành âm vô cùng
        # exp(-inf) = 0 -> Biến mất khỏi mẫu số
        logits[ignore_mask] = -1e9

        # --- 5. TÍNH LOSS ---
        # log_prob = logits - log(sum(exp(logits)))
        # Lúc này sum(exp) sẽ không còn chứa các mẫu BG-BG nữa
        log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True))
        
        # Tính trung bình trên các mẫu Positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        loss = - mean_log_prob_pos.mean()
        
        return loss

@CRITERIONS.register('FOCAL')
class FocalOadLoss(nn.Module):
    """
    Focal Loss for Online Action Detection.
    Công thức: L = - alpha * (1 - p)^gamma * target * log(p)
    
    Mục tiêu:
    - Giảm trọng số của các mẫu dễ đoán (Background).
    - Tập trung Gradient vào các mẫu khó (Action).
    """
    
    def __init__(self, cfg, reduction='mean'):
        super(FocalOadLoss, self).__init__()
        self.reduction = reduction
        self.gamma = cfg.get('focal_gamma', 2.0) # Mặc định gamma = 2.0
        self.alpha = cfg.get('focal_alpha', 0.25) # Mặc định alpha = 0.25 (cân bằng Positive/Negative)
        
        # Nếu muốn alpha riêng cho từng class (ví dụ giảm BG mạnh hơn)
        # cfg['class_weights'] nên là list [1.0, 1.0, ..., 0.1]
        self.class_weights = cfg.get('class_weights', None)
        if self.class_weights is not None:
            self.class_weights = torch.tensor(self.class_weights).float()

    def forward(self, out_dict, target):
        """
        out_dict['logits']: (B, Seq, Num_Classes)
        target: (B, Seq, Num_Classes) - Dạng One-hot hoặc Multi-hot
        """
        logits = out_dict['logits']
        
        # 1. OAD Logic: Chỉ lấy frame cuối cùng
        # (B, Seq, C) -> (B, C)
        logits = logits[:, -1, :].contiguous()
        target = target[:, -1, :].contiguous()
        
        return self.focal_loss(logits, target)

    def focal_loss(self, logits, target):
        # 2. Tính Log Softmax & Probabilities
        # log_softmax ổn định hơn log(softmax)
        log_p = F.log_softmax(logits, dim=-1) 
        p = torch.exp(log_p) # (B, C)
        
        # 3. Tính Focal Term: (1 - p)^gamma
        # Nếu p gần 1 (dễ đoán) -> (1-p) gần 0 -> Loss bị triệt tiêu
        focal_term = (1 - p) ** self.gamma
        
        # 4. Tính Cross Entropy cơ bản: - target * log(p)
        ce_loss = -target * log_p
        
        # 5. Kết hợp Focal Term
        loss = focal_term * ce_loss
        
        # 6. Áp dụng Alpha (Balancing)
        # Cách 1: Alpha scalar chung (như paper gốc dùng cho binary)
        if self.class_weights is None:
            # Trong Multi-class, alpha thường được áp dụng cho class Positive
            # Ở đây ta áp dụng alpha=1 cho mọi class, hoặc dùng giá trị từ config
            # Để đơn giản, ta nhân trực tiếp nếu user set alpha != 1
            if self.alpha != 1.0:
                 loss = self.alpha * loss
                 
        # Cách 2: Alpha riêng cho từng class (Advanced)
        else:
            if self.class_weights.device != loss.device:
                self.class_weights = self.class_weights.to(loss.device)
            loss = loss * self.class_weights.view(1, -1)
            
        # 7. Reduction
        if self.reduction == 'mean':
            # Mean trên cả Batch và Class
            return loss.sum() / logits.shape[0] 
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

@CRITERIONS.register('NONUNIFORM')
class OadLoss(nn.Module):
    
    def __init__(self, cfg, reduction='mean'):
        super(OadLoss, self).__init__()
        self.reduction = reduction
        self.num_classes = cfg['num_classes']
        self.loss = self.end_loss

    def end_loss(self, out_dict, target):
        # logits: (B, seq, K) target: (B, seq, K)
        logits = out_dict['logits']
        logits = logits[:,-1,:].contiguous()
        target = target[:,-1,:].contiguous()
        ce_loss = self.mlce_loss(logits, target)
        return ce_loss

    def mlce_loss(self, logits, target):
        '''
        multi label cross entropy loss. 
        logits: (B, K) target: (B, K) 
        '''
        logsoftmax = nn.LogSoftmax(dim=-1).to(logits.device)
        output = torch.sum(-F.normalize(target) * logsoftmax(logits), dim=1) # B
        if self.reduction == 'mean':
            loss = torch.mean(output)
        elif self.reduction == 'sum':
            loss = torch.sum(output)
        return loss

    def forward(self, out_dict, target): 
        return self.loss(out_dict, target)
    

@CRITERIONS.register('ANTICIPATION')
class OadAntLoss(nn.Module):
    
    def __init__(self, cfg, reduction='sum'):
        super(OadAntLoss, self).__init__()
        self.reduction = reduction
        self.loss = self.anticipation_loss
        self.num_classes = cfg['num_classes']

    def anticipation_loss(self, out_dict, target, ant_target):
        anticipation_logits = out_dict['anticipation_logits']
        pred_anticipation_logits = anticipation_logits[:,-1,:,:].contiguous().view(-1, self.num_classes)
        anticipation_logit_targets = ant_target.view(-1, self.num_classes)
        ant_loss = self.mlce_loss(pred_anticipation_logits, anticipation_logit_targets)
        return ant_loss 

    def ce_loss(self, out_dict, target):
        # logits: (B, seq, K) target: (B, seq, K)
        logits = out_dict['logits']
        logits = logits[:,-1,:].contiguous()
        target = target[:,-1,:].contiguous()
        ce_loss = self.mlce_loss(logits, target)
        return ce_loss

    def mlce_loss(self, logits, target):
        '''
        multi label cross entropy loss. 
        logits: (B, K) target: (B, K) 
        '''
        logsoftmax = nn.LogSoftmax(dim=-1).to(logits.device)
        output = torch.sum(-F.normalize(target) * logsoftmax(logits), dim=1) # B
        if self.reduction == 'mean':
            loss = torch.mean(output)
        elif self.reduction == 'sum':
            loss = torch.sum(output)

        return loss

    def forward(self, out_dict, target, ant_target): 
        return self.loss(out_dict, target, ant_target)
