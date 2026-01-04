import torch
import torch.nn as nn
import torch.nn.functional as F
from criterions.loss_builder import CRITERIONS

@CRITERIONS.register('CONTRASTIVE_STRICT')
class StrictInfoNCELoss(nn.Module):
    def __init__(self, cfg):
        super(StrictInfoNCELoss, self).__init__()
        self.T = cfg.get("temperature", 0.07)
        self.bg_class_idx = cfg.get("bg_class_idx", 0)
        self.bg_separation_bias = 0.0 # Hệ số phạt Nền

    def forward(self, output_dict, labels):
        # 1. UNPACK & STACK (Gộp Core và Aug lại)
        # q_core: [B, D], q_aug: [B, D] -> q_all: [2B, D]
        q_all = torch.cat([output_dict['q_core'], output_dict['q_aug']], dim=0)
        
        # Teacher k cũng phải nhân đôi để khớp với q_all
        # k: [B, D] -> k_all: [2B, D]
        k = output_dict['k_cls']
        k_all = torch.cat([k, k], dim=0)
        
        # Labels cũng nhân đôi
        labels_all = torch.cat([labels, labels], dim=0)
        
        # Lấy Queue
        queues = output_dict['queues'] # [C, D, K]
        
        # Dimensions
        total_batch_size = q_all.shape[0] # 2B
        num_classes, dim, K = queues.shape
        device = q_all.device

        # 2. FLATTEN QUEUE
        # [D, N_All]
        queue_flat = queues.permute(0, 2, 1).reshape(-1, dim).T.clone().detach()
        # [N_All]
        queue_labels = torch.arange(num_classes, device=device).unsqueeze(1).repeat(1, K).view(-1)

        # --- 3. TÍNH LOGITS (Unified) ---
        
        # A. Positive: q_all giống k_all
        # [2B, 1]
        sim_pos = torch.einsum('nc,nc->n', [q_all, k_all]).unsqueeze(-1) / self.T
        
        # B. Queue Negatives: q_all giống Queue
        # [2B, N_All]
        sim_queue = torch.mm(q_all, queue_flat) / self.T
        
        # --- 4. ÁP DỤNG BIAS PHẠT NỀN (Vectorized) ---
        
        # Tạo các mask logic trên batch lớn (2B)
        is_bg_in_queue = (queue_labels == self.bg_class_idx).unsqueeze(0)    # [1, N_All]
        is_action_in_queue = (~is_bg_in_queue)                               # [1, N_All]
        
        is_bg_anchor = (labels_all == self.bg_class_idx).unsqueeze(1)        # [2B, 1]
        is_action_anchor = (~is_bg_anchor)                                   # [2B, 1]

        # Bias 1: Anchor Action đẩy Queue BG
        bias_action_vs_bg = (is_action_anchor & is_bg_in_queue).float() * self.bg_separation_bias
        
        # Bias 2: Anchor BG đẩy Queue Action
        bias_bg_vs_action = (is_bg_anchor & is_action_in_queue).float() * self.bg_separation_bias
        
        # Cộng Bias vào Logits Queue
        sim_queue = sim_queue + bias_action_vs_bg + bias_bg_vs_action

        # --- 5. TỔ CHỨC LOGITS ---
        # [2B, 1 + N_All] -> Cột 0 là Pos, Cột 1+ là Queue
        logits = torch.cat([sim_pos, sim_queue], dim=1)
        
        # --- 6. MASKING (Logic Gates) ---
        
        # A. Mask Ignore (Floating BG)
        # Nếu Anchor là BG VÀ Queue Item là BG -> Ignore
        is_bg_queue_row = (queue_labels == self.bg_class_idx).unsqueeze(0)
        mask_ignore_queue = is_bg_anchor & is_bg_queue_row # [2B, N_All]
        
        mask_ignore_pos = torch.zeros(total_batch_size, 1, device=device, dtype=torch.bool)
        mask_ignore = torch.cat([mask_ignore_pos, mask_ignore_queue], dim=1)
        
        # B. Mask Positive (Class Matching)
        mask_pos = torch.zeros_like(logits, dtype=torch.bool)
        mask_pos[:, 0] = True # Teacher luôn là Pos
        
        # Check Queue items cùng class với Anchor
        is_same_class = labels_all.unsqueeze(1) == queue_labels.unsqueeze(0)
        mask_pos[:, 1:] = is_same_class
        
        # TRỪ KHI: Cả hai đều là BG (Floating BG không kéo nhau)
        # Đoạn này thực ra đã bị Mask Ignore chặn rồi, nhưng set False cho chắc
        mask_pos[is_bg_anchor.squeeze(1), 1:] = False

        # --- 7. TÍNH LOSS FINAL ---
        # Ổn định số học
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # Apply Ignore
        logits = logits.masked_fill(mask_ignore, -1e9)
        exp_logits = torch.exp(logits)
        
        # Mẫu số = Pos + Neg (Neg là những thằng không phải Pos và không bị Ignore)
        mask_neg = (~mask_pos) & (~mask_ignore)
        sum_exp_neg = (exp_logits * mask_neg.float()).sum(1, keepdim=True)
        denominator = exp_logits + sum_exp_neg
        
        # Log Prob
        log_prob = logits - torch.log(denominator + 1e-8)
        
        # Chỉ lấy loss tại các vị trí Positive
        # Chia cho tổng số lượng Positive để normalize
        loss = -(log_prob * mask_pos.float()).sum() / (mask_pos.sum() + 1e-8)
        
        return loss

@CRITERIONS.register('CONTRASTIVE')
class SupConMROADLoss(nn.Module):
    def __init__(self, cfg):
        super(SupConMROADLoss, self).__init__()
        self.T = cfg.get("temperature", 0.07)
        self.bg_class_idx = cfg.get("bg_class_idx", 0)

    def forward(self, output_dict, labels):
        q = output_dict['q_cls']          # [B, D] (Anchor)
        k = output_dict['k_cls']          # [B, D] (Positive)
        q_shuff = output_dict['q_shuff']  # [B, D] (Negative 1: Temporal Chaos)
        q_context = output_dict['q_context'] # [B, D] (Negative 2: Context Only)
        queues = output_dict['queues']    # [C, D, K] (Queue)
        
        batch_size = q.shape[0]
        num_classes, dim, K = queues.shape
        device = q.device

        # --- 1. CHUẨN BỊ QUEUE PHẲNG ---
        queue_flat = queues.permute(0, 2, 1).reshape(-1, dim).T.clone().detach() # [D, N_All]
        queue_labels = torch.arange(num_classes, device=device).unsqueeze(1).repeat(1, K).view(-1)

        # --- 2. TÍNH LOGITS (MỞ RỘNG 4 PHẦN) ---
        
        # A. Positive (q . k) -> [B, 1]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # B. Hard Negative 1: Shuffle (q . q_shuff) -> [B, 1]
        l_neg_shuff = torch.einsum('nc,nc->n', [q, q_shuff]).unsqueeze(-1)

        # C. Hard Negative 2: Context Only (q . q_context) -> [B, 1]
        l_neg_ctx = torch.einsum('nc,nc->n', [q, q_context]).unsqueeze(-1)

        # D. Queue (q . queue) -> [B, N_All]
        l_queue = torch.mm(q, queue_flat)
        
        # GỘP LẠI: [Positive, Shuff, Context, Queue...]
        # Shape: [B, 1 + 1 + 1 + N_All] = [B, 3 + N_All]
        logits = torch.cat([l_pos, l_neg_shuff, l_neg_ctx, l_queue], dim=1)
        logits /= self.T

        # --- 3. TẠO MASK POSITIVE (ĐỂ XÁC ĐỊNH TỬ SỐ) ---
        mask = torch.zeros_like(logits, device=device)
        
        # Cột 0 (Key) LUÔN LÀ POSITIVE
        mask[:, 0] = 1.0 
        
        # Cột 1 (Shuffle) & Cột 2 (Context) LUÔN LÀ NEGATIVE (Mặc định là 0, không cần gán)
        # mask[:, 1] = 0.0
        # mask[:, 2] = 0.0

        # Cột 3 trở đi (Queue): Xử lý logic SupCon
        # Dịch index đi 3 đơn vị
        is_same_class = labels.unsqueeze(1) == queue_labels.unsqueeze(0) # [B, N_All]
        mask[:, 3:] = is_same_class.float()

        # Logic Background: Nếu Anchor là BG -> Chỉ kéo k, KHÔNG kéo Queue BG
        is_bg_sample = (labels == self.bg_class_idx) 
        mask[is_bg_sample, 3:] = 0.0

        # --- 4. TẠO MASK IGNORE (CHO LỚP NỀN TRÔI NỔI) ---
        # Mục tiêu: Loại bỏ sự tranh chấp giữa các mẫu Nền trong Queue
        
        is_bg_in_queue = (queue_labels == self.bg_class_idx) # [N_All]
        ignore_mask_queue = is_bg_sample.unsqueeze(1) & is_bg_in_queue.unsqueeze(0) # [B, N_All]
        
        # Mở rộng cho 3 cột đầu: 
        # Cột k (0): Không ignore
        # Cột Shuff (1): Không ignore (Đẩy mạnh ra)
        # Cột Ctx (2): Không ignore (Đẩy mạnh ra)
        ignore_mask = torch.cat([
            torch.zeros(batch_size, 3, device=device, dtype=torch.bool),
            ignore_mask_queue
        ], dim=1)

        # Gán logits ignore = -inf
        logits[ignore_mask] = -1e9

        # --- 5. TÍNH LOSS ---
        # Ổn định số học
        log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True))
        
        # Tính loss trên tập Positive
        # Tránh chia cho 0 (dù k luôn là positive nên sum >= 1, nhưng an toàn vẫn hơn)
        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
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
