import torch
import torch.nn as nn
import torch.nn.functional as F
from criterions.loss_builder import CRITERIONS

@CRITERIONS.register('CONTRASTIVE')
class SupConMROADLoss(nn.Module):
    def __init__(self, cfg):
        super(SupConMROADLoss, self).__init__()
        self.T = cfg["temperature"]
        self.bg_class_idx = cfg["bg_class_idx"]

    def forward(self, output_dict, labels):
        """
        Args:
            output_dict: Output từ model, chứa:
                - 'q_cls': [Batch, Dim] (Query)
                - 'k_cls': [Batch, Dim] (Key)
                - 'queues': [Num_Classes, Dim, K] (Memory Bank)
            labels: [Batch] chứa index của class thật.
        """
        q = output_dict['q_cls']      # [B, D]
        k = output_dict['k_cls']      # [B, D]
        queues = output_dict['queues'] # [C, D, K]
        
        batch_size = q.shape[0]
        num_classes, dim, K = queues.shape
        device = q.device

        # --- BƯỚC 1: CHUẨN BỊ MEMORY BANK PHẲNG (FLATTEN) ---
        # Ta cần biến đổi Queues thành dạng [Dim, Total_Samples] để nhân ma trận
        # Queues gốc: [Num_Classes, Dim, K]
        # 1. Permute -> [Num_Classes, K, Dim]
        # 2. Reshape -> [Num_Classes * K, Dim]
        # 3. Transpose -> [Dim, Num_Classes * K]
        queue_flat = queues.permute(0, 2, 1).reshape(-1, dim).T.clone().detach() # [D, N_All]
        
        # Tạo nhãn cho từng phần tử trong Queue phẳng
        # Ex: [0, 0, ..., 0, 1, 1, ..., 1, ...]
        queue_labels = torch.arange(num_classes, device=device).unsqueeze(1).repeat(1, K).view(-1) # [N_All]

        # --- BƯỚC 2: TÍNH LOGITS (SIMILARITY) ---
        # 1. Positive Logits (Giữa q và k tương ứng): [B, 1]
        # bmm: (B, 1, D) x (B, D, 1) -> (B, 1, 1) -> squeeze -> (B, 1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # 2. Negative/Queue Logits (Giữa q và toàn bộ Memory Bank): [B, N_All]
        l_queue = torch.mm(q, queue_flat)
        
        # 3. Gom lại: Cột 0 là k (Positive), Cột 1..N là Queue
        # Shape: [B, 1 + N_All]
        logits = torch.cat([l_pos, l_queue], dim=1)
        
        # Scale by Temperature
        logits /= self.T

        # --- BƯỚC 3: TẠO MASK (CHIẾN LƯỢC CỐT LÕI) ---
        # Mask shape: [B, 1 + N_All]. Giá trị 1 là Positive, 0 là Negative.
        
        # Khởi tạo mask
        mask = torch.zeros_like(logits, device=device)
        
        # A. Cột đầu tiên luôn là k (Teacher) -> Luôn là Positive
        mask[:, 0] = 1.0

        # B. Xử lý các cột Queue (SupCon Logic)
        # So sánh nhãn của Batch với nhãn của Queue
        # batch_labels: [B, 1] vs queue_labels: [1, N_All]
        # is_same_class: [B, N_All] (True nếu cùng class)
        is_same_class = labels.unsqueeze(1) == queue_labels.unsqueeze(0)
        
        # Gán vào mask (bắt đầu từ cột 1)
        mask[:, 1:] = is_same_class.float()

        # C. Xử lý Background (NT-Xent Logic)
        # Nếu mẫu i là Background -> Chỉ có k là Positive, Queue cùng loại cũng là Negative.
        # Tìm các dòng trong batch là Background
        is_bg_sample = (labels == self.bg_class_idx) # [B] -> True/False
        
        # Tại các dòng là BG, set phần mask của Queue về 0 hết
        # (Giữ lại cột 0 vì k luôn là positive)
        mask[is_bg_sample, 1:] = 0.0

        # --- BƯỚC 4: TÍNH LOSS ---
        # Sử dụng công thức SupCon ổn định số học
        
        # Log-Softmax trên toàn bộ logits (Mẫu số trong công thức)
        # log_prob shape: [B, 1 + N_All]
        log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True))
        
        # Chỉ lấy log_prob tại các vị trí Positive (theo mask)
        # sum(1) để cộng dồn các positive của 1 mẫu lại
        # mask.sum(1) là số lượng positive của mẫu đó (|P(i)|)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss là trung bình âm của batch
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
