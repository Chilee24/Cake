import torch
import torch.nn as nn
import torch.nn.functional as F
from criterions.loss_builder import CRITERIONS

@CRITERIONS.register('CONTRASTIVE_STRICT')
class OADSupervisedMinorityLoss(nn.Module):
    def __init__(self, cfg):
        super(OADSupervisedMinorityLoss, self).__init__()
        self.T = cfg.get("temperature", 0.07)
        self.bg_class_idx = cfg.get("bg_class_idx", 0)
        # Bias này giúp việc "đẩy Action" của Background mạnh hơn
        self.bg_separation_bias = cfg.get("bg_separation_bias", 0) 

    def forward(self, output_dict, targets_multihot):
        """
        Logic: 
        - Action anchors: Kéo Action Queue (SupCon), Đẩy Background Queue.
        - BG anchors: Chỉ kéo chính Teacher của nó (SimCLR), IGNORE Background Queue, Đẩy Action Queue.
        """
        q = output_dict['q_cls'] # [B, D]
        k = output_dict['k_cls'] # [B, D] (Teacher - Positive mặc định)
        queues = output_dict['queues'] # [C, D, K]
        
        B, D = q.shape
        num_classes, _, K = queues.shape
        device = q.device

        # --- 1. PREPARE DATA ---
        # Flatten Queue: [D, C*K]
        queue_flat = queues.permute(0, 2, 1).reshape(-1, D).T.clone().detach()
        
        # ID của từng bucket trong queue: [C*K]
        queue_bucket_ids = torch.arange(num_classes, device=device).unsqueeze(1).repeat(1, K).view(-1)

        # Xác định ai là BG, ai là Action (Dựa trên Anchor)
        # targets_multihot: [B, C]
        is_bg_anchor = targets_multihot[:, self.bg_class_idx].bool() # [B]
        is_action_anchor = ~is_bg_anchor

        # Xác định ai là BG, ai là Action (Dựa trên Queue)
        is_bg_in_queue = (queue_bucket_ids == self.bg_class_idx).unsqueeze(0) # [1, C*K]
        
        # --- 2. COMPUTE LOGITS ---
        # Similarity với Teacher (Luôn là Positive cho cả BG và Action)
        sim_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) / self.T # [B, 1]

        # Similarity với Queue
        sim_queue = torch.mm(q, queue_flat) / self.T # [B, C*K]

        # --- 3. ÁP DỤNG CHIẾN LƯỢC SUPERVISED MINORITY ---

        # A. Xử lý Positive Mask (Tử số)
        # Lấy nhãn từ target mapping sang queue
        mask_queue_pos = targets_multihot[:, queue_bucket_ids] # [B, C*K]
        mask_queue_pos = (mask_queue_pos > 0.5).float()

        # [QUAN TRỌNG]: Background Anchor KHÔNG được kéo bất kỳ ai trong Queue
        # Lý do: Nếu để BG kéo BG trong queue, chúng sẽ tụ lại thành 1 cụm lớn (Collapse)
        # BG chỉ được phép kéo sim_pos (Teacher) của chính nó.
        mask_queue_pos[is_bg_anchor, :] = 0.0

        # B. Xử lý Negative Mask (Mẫu số - Ignore Logic)
        
        # Tạo mask ignore mặc định (không ignore gì cả)
        mask_ignore = torch.zeros_like(sim_queue, dtype=torch.bool) # [B, C*K]

        # [QUAN TRỌNG]: Background Anchor sẽ IGNORE các Background trong Queue
        # Logic: "Sẽ không làm gì nhau cả". BG Anchor coi BG Queue như người vô hình.
        # Nó chỉ nhìn thấy Action trong Queue (để đẩy ra - Negative).
        mask_ignore_bg_bg = is_bg_anchor.unsqueeze(1) & is_bg_in_queue
        mask_ignore = mask_ignore | mask_ignore_bg_bg

        # --- 4. OPTIONAL: HARD SEPARATION BIAS ---
        # Giúp việc "đẩy" mạnh mẽ hơn.
        # Action Anchor vs BG Queue -> Đẩy mạnh
        # BG Anchor vs Action Queue -> Đẩy mạnh
        if self.bg_separation_bias != 0:
            is_action_in_queue = ~is_bg_in_queue
            
            # Action Anchor gặp BG Queue
            bias_mask_1 = is_action_anchor.unsqueeze(1) & is_bg_in_queue
            # BG Anchor gặp Action Queue
            bias_mask_2 = is_bg_anchor.unsqueeze(1) & is_action_in_queue
            
            final_bias_mask = bias_mask_1 | bias_mask_2
            sim_queue = sim_queue + (final_bias_mask.float() * self.bg_separation_bias)

        # --- 5. FINALIZE LOSS ---
        
        # Nối Teacher và Queue
        # Logits: [B, 1 + C*K]
        logits = torch.cat([sim_pos, sim_queue], dim=1)
        
        # Mask Positive: Cột đầu (Teacher) luôn là 1 (Positive), phần sau tùy thuộc logic trên
        mask_pos_final = torch.cat([torch.ones(B, 1, device=device), mask_queue_pos], dim=1)
        
        # Mask Ignore: Cột đầu không ignore, phần sau theo logic BG-BG
        mask_ignore_final = torch.cat([torch.zeros(B, 1, device=device, dtype=torch.bool), mask_ignore], dim=1)

        # Tính Log Softmax
        # Trừ max để ổn định số học
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Che đi các phần tử Ignore (gán giá trị cực nhỏ trước khi exp)
        logits = logits.masked_fill(mask_ignore_final, -1e9)

        exp_logits = torch.exp(logits)
        
        # Mẫu số: Tổng exp của tất cả (trừ những cái đã bị ignore)
        denominator = exp_logits.sum(1, keepdim=True)
        
        log_prob = logits - torch.log(denominator + 1e-8)

        # Chỉ lấy loss trên các cặp Positive
        loss = -(log_prob * mask_pos_final).sum(1) / (mask_pos_final.sum(1) + 1e-8)

        return loss.mean()

@CRITERIONS.register('CONTRASTIVE')
class StandardMultiLabelQueueInfoNCELoss(nn.Module):
    def __init__(self, cfg):
        super(StandardMultiLabelQueueInfoNCELoss, self).__init__()
        self.T = cfg.get("temperature", 0.07)

    def forward(self, output_dict, targets_multihot):
        """
        Gom tất cả các class (bao gồm Nền) theo logic SupCon chuẩn:
        - Cùng nhãn => Kéo gần.
        - Khác nhãn => Đẩy xa.
        """
        q = output_dict['q_cls'] # [B, D]
        k = output_dict['k_cls'] # [B, D]
        queues = output_dict['queues'] # [C, D, K]
        
        B, D = q.shape
        num_classes, _, K = queues.shape
        device = q.device

        # 1. FLATTEN QUEUE
        # [C, D, K] -> [D, C*K]
        queue_flat = queues.permute(0, 2, 1).reshape(-1, D).T.clone().detach()
        
        # Tạo ID cho queue: [0,0..., 1,1..., ..., 20,20...]
        queue_bucket_ids = torch.arange(num_classes, device=device).unsqueeze(1).repeat(1, K).view(-1)

        # 2. TÍNH LOGITS
        # A. Positive Teacher (q vs k)
        sim_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) / self.T
        
        # B. Queue Contrast (q vs All Queue)
        sim_queue = torch.mm(q, queue_flat) / self.T
        
        # 3. TẠO MASK POSITIVE (Logic SupCon Chuẩn)
        # Bất kỳ ai trong Queue có cùng nhãn với Anchor đều là Positive
        # Kể cả đó là nhãn Nền (Background)
        
        # Lấy nhãn từ target dựa trên ID của bucket
        mask_queue_pos = targets_multihot[:, queue_bucket_ids]
        mask_queue_pos = (mask_queue_pos > 0.5).float()

        # --- ĐÃ XÓA: Logic phạt Nền (Bias) và logic Ignore Nền ---
        # Bây giờ Background cư xử y hệt Action:
        # Background Anchor sẽ coi Background trong Queue là bạn (Positive).

        # 4. TỔNG HỢP LOGITS
        # [B, 1 + C*K]
        logits = torch.cat([sim_pos, sim_queue], dim=1)
        
        # Mask Positive (Cột 0 là Teacher luôn = 1)
        mask_teacher = torch.ones(B, 1, device=device)
        mask_pos_final = torch.cat([mask_teacher, mask_queue_pos], dim=1)

        # 5. TÍNH LOSS (SupCon)
        # Numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        exp_logits = torch.exp(logits)
        
        # Mẫu số: Tổng exp của TẤT CẢ (Pos + Neg)
        denominator = exp_logits.sum(1, keepdim=True)
        
        log_prob = logits - torch.log(denominator + 1e-8)
        
        # Loss = - Mean(LogProb của các cặp Positive)
        loss = -(log_prob * mask_pos_final).sum(1) / (mask_pos_final.sum(1) + 1e-8)
        
        return loss.mean()

@CRITERIONS.register('FOCAL')
class FocalOadLoss(nn.Module):
    """
    Binary Focal Loss for ActionFormer-style Head (Sigmoid).
    
    Công thức cho mỗi class k (độc lập):
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Trong đó:
        p_t = p      nếu y=1 (Action)
        p_t = 1 - p  nếu y=0 (Background/Other Action)
    """
    
    def __init__(self, cfg, reduction='mean'):
        super(FocalOadLoss, self).__init__()
        self.reduction = reduction
        self.gamma = cfg.get('focal_gamma', 2.0)
        self.alpha = cfg.get('focal_alpha', 0.5) 

    def forward(self, out_dict, target):
        """
        out_dict['logits']: (B, Seq, K) - Logits thô chưa qua Sigmoid
        target: (B, Seq, K) - One-hot (hoặc Multi-hot), Cột BG đã bị xóa.
        """
        logits = out_dict['logits']
        
        # 1. OAD Logic: Chỉ tính Loss cho frame cuối cùng
        # (B, Seq, K) -> (B, K)
        logits = logits[:, -1, :].contiguous()
        target = target[:, -1, :].contiguous()
        
        return self.sigmoid_focal_loss(logits, target)

    def sigmoid_focal_loss(self, logits, target):
        # 2. Tính BCE Loss cơ bản (chưa nhân Focal term)
        # Sử dụng binary_cross_entropy_with_logits để ổn định số học (tự động sigmoid bên trong)
        # reduction='none' để giữ nguyên shape (B, K) để nhân với trọng số sau
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        
        # 3. Tính xác suất p (để tính Focal term)
        p = torch.sigmoid(logits)
        
        # 4. Tính p_t (Xác suất dự đoán đúng theo target)
        # Nếu target=1 -> p_t = p
        # Nếu target=0 -> p_t = 1 - p
        p_t = p * target + (1 - p) * (1 - target)
        
        # 5. Tính Focal Term: (1 - p_t)^gamma
        # - Nếu đoán đúng (p_t gần 1) -> Term gần 0 -> Loss giảm (Mẫu dễ)
        # - Nếu đoán sai (p_t gần 0) -> Term lớn -> Loss tăng (Mẫu khó)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        # 6. Áp dụng Alpha Balancing
        if self.alpha >= 0:
            # alpha_t = alpha nếu target=1, ngược lại (1-alpha)
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
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
