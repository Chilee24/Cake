import torch
import torch.nn as nn
import torch.nn.functional as F
from criterions.loss_builder import CRITERIONS

@CRITERIONS.register('CONTRASTIVE_STRICT')
class MultiLabelQueueInfoNCELoss(nn.Module):
    def __init__(self, cfg):
        super(MultiLabelQueueInfoNCELoss, self).__init__()
        self.T = cfg.get("temperature", 0.07)
        self.bg_class_idx = cfg.get("bg_class_idx", 0)
        self.bg_separation_bias = -2.0 # Hệ số phạt Nền (đẩy xa)

    def forward(self, output_dict, targets_multihot):
        """
        output_dict: chứa q_cls, k_cls, queues
        targets_multihot: [B, C] - Vector nhãn gốc
        """
        q = output_dict['q_cls'] # [B, D]
        k = output_dict['k_cls'] # [B, D]
        
        # Queues: [C, D, K] - Class-Specific Queues
        queues = output_dict['queues'] 
        
        # Dimensions
        B, D = q.shape
        num_classes, _, K = queues.shape
        device = q.device

        # 2. FLATTEN QUEUE (Trải phẳng các ngăn tủ)
        # Biến đổi [C, D, K] -> [D, C*K] để nhân ma trận
        # Thứ tự sẽ là: [Toàn bộ Queue 0, Toàn bộ Queue 1, ...]
        queue_flat = queues.permute(0, 2, 1).reshape(-1, D).T.clone().detach()
        
        # Tạo ID cho từng phần tử trong Queue đã flatten
        # [C*K] -> [0,0...0, 1,1...1, ... , 30,30...30]
        queue_bucket_ids = torch.arange(num_classes, device=device).unsqueeze(1).repeat(1, K).view(-1)

        # 3. TÍNH LOGITS
        
        # A. Positive Teacher: q vs k [B, 1]
        sim_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) / self.T
        
        # B. Queue Contrast: q vs Toàn bộ Queue [B, C*K]
        sim_queue = torch.mm(q, queue_flat) / self.T
        
        # 4. TẠO MASK POSITIVE (Logic Label-wise Cloning)
        
        # Mục tiêu: Tạo mask [B, C*K]
        # Nếu mẫu b có nhãn c (targets_multihot[b, c] == 1)
        # -> Tất cả phần tử thuộc Queue c (queue_bucket_ids == c) đều là Positive
        
        # Cách làm nhanh: Dùng queue_bucket_ids làm index để lấy giá trị từ targets_multihot
        # targets_multihot: [B, C]
        # queue_bucket_ids: [C*K]
        # Kết quả: [B, C*K]
        mask_queue_pos = targets_multihot[:, queue_bucket_ids]
        
        # Đảm bảo mask là 0/1 (float)
        mask_queue_pos = (mask_queue_pos > 0.5).float()

        # 5. XỬ LÝ BACKGROUND (Nền)
        
        # Xác định Anchor là Nền hay Action
        # [B, 1]
        is_bg_anchor = targets_multihot[:, self.bg_class_idx].unsqueeze(1).bool()
        is_action_anchor = ~is_bg_anchor
        
        # Xác định Bucket trong Queue là Nền hay Action
        # [1, C*K]
        is_bg_bucket = (queue_bucket_ids == self.bg_class_idx).unsqueeze(0).bool()
        is_action_bucket = ~is_bg_bucket

        # LUẬT 1: Anchor BG không kéo ai cả (trừ Teacher)
        # Xóa toàn bộ Positive trong Queue của Anchor BG
        mask_queue_pos = mask_queue_pos.masked_fill(is_bg_anchor, 0.0)
        
        # LUẬT 2: Bias Phạt (Đẩy xa Action và Nền)
        # Action Anchor vs BG Bucket -> Đẩy
        # BG Anchor vs Action Bucket -> Đẩy
        bias_mask = (is_action_anchor & is_bg_bucket) | (is_bg_anchor & is_action_bucket)
        sim_queue = sim_queue + (bias_mask.float() * self.bg_separation_bias)
        
        # LUẬT 3: Ignore cặp BG-BG (Nền không phạm Nền)
        mask_ignore_queue = is_bg_anchor & is_bg_bucket

        # 6. TỔNG HỢP LOGITS
        
        # [B, 1 + C*K]
        logits = torch.cat([sim_pos, sim_queue], dim=1)
        
        # Mask Positive Final (Cột 0 là Teacher luôn True)
        mask_teacher = torch.ones(B, 1, device=device)
        mask_pos_final = torch.cat([mask_teacher, mask_queue_pos], dim=1)
        
        # Mask Ignore Final (Cột 0 không bao giờ Ignore)
        mask_ignore_teacher = torch.zeros(B, 1, device=device, dtype=torch.bool)
        mask_ignore_final = torch.cat([mask_ignore_teacher, mask_ignore_queue], dim=1)

        # 7. TÍNH LOSS (SupCon)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # Apply Ignore
        logits = logits.masked_fill(mask_ignore_final, -1e9)
        exp_logits = torch.exp(logits)
        
        # Mẫu số: Sum Exp (Pos + Neg)
        denominator = exp_logits.sum(1, keepdim=True)
        
        log_prob = logits - torch.log(denominator + 1e-8)
        
        # Mean Log Prob của các Positive
        loss = -(log_prob * mask_pos_final).sum(1) / (mask_pos_final.sum(1) + 1e-8)
        
        return loss.mean()

@CRITERIONS.register('CONTRASTIVE')
class SupConUnifiedClassLoss(nn.Module):
    def __init__(self, cfg):
        super(SupConUnifiedClassLoss, self).__init__()
        self.T = cfg.get("temperature", 0.07)
        # Các tham số BG không còn dùng đến trong logic này nữa
        # nhưng vẫn giữ để tránh lỗi khi load config
        self.bg_class_idx = cfg.get("bg_class_idx", 0)

    def forward(self, output_dict, targets_multihot):
        """
        ABLATION VERSION: Gom cụm tất cả (Unified Clustering)
        - Coi BG là 1 class bình thường.
        - BG kéo BG.
        - Không có Bias đẩy BG và Action ra xa.
        """
        # 1. UNPACK
        if 'q_cls' in output_dict:
            q = output_dict['q_cls']
        else:
            q = output_dict['q_core']
            
        k = output_dict['k_cls']       # [B, D]
        queues = output_dict['queues'] # [C, D, K]
        
        B, D = q.shape
        num_classes, _, K = queues.shape
        device = q.device

        # 2. FLATTEN QUEUE
        # [C, D, K] -> [D, C*K]
        queue_flat = queues.permute(0, 2, 1).reshape(-1, D).T.clone().detach()
        
        # [C*K] -> [0,0...0, 1,1...1, ..., 30,30...30]
        queue_bucket_ids = torch.arange(num_classes, device=device).unsqueeze(1).repeat(1, K).view(-1)

        # 3. TÍNH LOGITS
        
        # A. Positive Teacher (q . k) -> [B, 1]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # B. Queue Contrast (q . queue) -> [B, C*K]
        l_queue = torch.mm(q, queue_flat)
        
        # GỘP LOGITS: [B, 1 + C*K]
        logits = torch.cat([l_pos, l_queue], dim=1)
        logits /= self.T

        # 4. TẠO MASK POSITIVE (Logic Unified)
        
        # A. Cột 0 (Teacher): Luôn là Positive
        mask_pos_teacher = torch.ones(B, 1, device=device)
        
        # B. Cột Queue:
        # Chiếu vector Target lên Bucket IDs
        # Bất kể là Action hay BG, cứ trùng nhãn là KÉO.
        # Ví dụ: Anchor là BG -> Kéo tất cả BG trong Queue lại gần.
        mask_queue_pos = targets_multihot[:, queue_bucket_ids]
        mask_queue_pos = (mask_queue_pos > 0.5).float()

        # [REMOVED] Rule 1: Anchor BG không kéo ai -> ĐÃ BỎ
        # [REMOVED] Rule 2: Bias Phạt -> ĐÃ BỎ
        # [REMOVED] Rule 3: Ignore cặp BG-BG -> ĐÃ BỎ

        # 5. TỔNG HỢP MASK
        mask_pos_final = torch.cat([mask_pos_teacher, mask_queue_pos], dim=1)
        
        # Không còn Ignore (trừ khi bạn muốn ignore chính bản thân nó trong queue, 
        # nhưng ở đây ta dùng Teacher làm Positive chính nên queue coi như negative sample set mở rộng)
        mask_ignore_final = torch.zeros_like(logits, dtype=torch.bool)

        # 6. TÍNH SUPCON LOSS
        
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # Mask Ignore (nếu có, ở đây là toàn False)
        logits = logits.masked_fill(mask_ignore_final, -1e9)
        exp_logits = torch.exp(logits)
        
        # Mẫu số: Tổng exp của tất cả
        denominator = exp_logits.sum(1, keepdim=True)
        
        log_prob = logits - torch.log(denominator + 1e-8)
        
        # Tính Mean Loss trên các cặp Positive
        mask_sum = mask_pos_final.sum(1)
        mean_log_prob_pos = (mask_pos_final * log_prob).sum(1) / (mask_sum + 1e-8)
        
        loss = - mean_log_prob_pos.mean()
        
        return loss

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
        self.alpha = cfg.get('focal_alpha', 0.25) 

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
