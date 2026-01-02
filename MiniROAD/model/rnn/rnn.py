import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_builder import META_ARCHITECTURES

FEATURE_SIZES = {
    'rgb_anet_resnet50': 2048,
    'flow_anet_resnet50': 2048,
    'rgb_kinetics_bninception': 1024,
    'flow_kinetics_bninception': 1024,
    'rgb_kinetics_resnet50': 2048,
    'flow_kinetics_resnet50': 2048,
    'flow_nv_kinetics_bninception': 1024,
    'rgb_kinetics_x3d': 2048,
    'flow_kinetics_x3d': 2048,
    'cake_kinetics': 4096
}

class SemanticMaskGenerator(nn.Module):
    def __init__(self, bg_class_idx=0, mask_ratio=0.5):
        super(SemanticMaskGenerator, self).__init__()
        self.bg_class_idx = bg_class_idx
        self.mask_ratio = mask_ratio

    def forward(self, x, labels_per_frame, final_labels):
        """
        Input:
            x: [B, T, D]
            labels_per_frame: [B, T] (Nhãn từng frame)
            final_labels: [B] (Nhãn của cả window)
        Output:
            x_core: Anchor (Action frames nếu là Action, Random mask nếu là BG)
            x_context: Negative (BG frames của Action window)
        """
        B, T, D = x.shape
        device = x.device

        # --- 1. TẠO MASK CƠ BẢN (RANDOM) CHO CLASS NỀN ---
        # Dùng cho các window là Background thuần túy
        rand_tensor = torch.rand(B, T - 1, 1, device=device)
        mask_random_past = (rand_tensor > self.mask_ratio).float()
        mask_random_last = torch.ones(B, 1, 1, device=device)
        mask_random = torch.cat([mask_random_past, mask_random_last], dim=1) # [B, T, 1]

        # --- 2. TẠO MASK NGỮ NGHĨA (SEMANTIC) CHO CLASS ACTION ---
        # Xác định frame nào là nền trong window [B, T]
        is_bg_frame = (labels_per_frame == self.bg_class_idx).unsqueeze(-1).float() # [B, T, 1]
        
        # Mask Core: Giữ Action (Not BG), Che BG
        mask_semantic_core = 1.0 - is_bg_frame
        
        # Mask Context: Giữ BG, Che Action
        mask_semantic_ctx = is_bg_frame

        # --- 3. KẾT HỢP (SWITCHING LOGIC) ---
        # Xác định mẫu nào là Action, mẫu nào là BG
        is_bg_sample = (final_labels == self.bg_class_idx).view(B, 1, 1) # [B, 1, 1]

        # A. Xử lý x_core (Anchor)
        # Nếu là BG sample -> Dùng Random Mask
        # Nếu là Action sample -> Dùng Semantic Core Mask
        mask_final_core = torch.where(is_bg_sample, mask_random, mask_semantic_core)
        
        # *Safety Check*: Nếu Semantic Core bị đen thùi (lỡ Action sample mà toàn frame nền)
        # Thì fallback về Random Mask để tránh Anchor bị O (Zero)
        has_action_frames = mask_final_core.sum(dim=1, keepdim=True) > 0
        mask_final_core = torch.where(has_action_frames, mask_final_core, mask_random)
        
        x_core = x * mask_final_core

        # B. Xử lý x_context (Negative đặc biệt)
        # Chỉ tạo cho Action sample. BG sample thì gán 0 (sẽ bị mask ignore trong loss)
        mask_final_ctx = torch.where(is_bg_sample, torch.zeros_like(mask_semantic_ctx), mask_semantic_ctx)
        
        x_context = x * mask_final_ctx

        return x_core, x_context

@META_ARCHITECTURES.register("CONTRASTIVE_MROAD")
class ContrastiveMROADMultiQueue(nn.Module):
    def __init__(self, cfg):
        super(ContrastiveMROADMultiQueue, self).__init__()
        self.contrastive_dim = cfg.get('contrastive_dim', 128)
        self.hidden_dim = cfg.get('hidden_dim', 2048)
        self.num_classes = cfg['num_classes']
        self.bg_class_idx = cfg.get('bg_class_idx', 0) # Cần lấy từ config
        
        # Dùng Generator Mới
        # mask_ratio=0.5 để tăng độ khó cho các mẫu nền
        self.generator = SemanticMaskGenerator(bg_class_idx=self.bg_class_idx, mask_ratio=0)
        
        self.K = cfg.get('queue_size_per_class', 1024) 
        self.m = cfg.get('momentum', 0.999)
        self.T = cfg.get('temperature', 0.07)

        # Input Dim Setup (Giữ nguyên)
        self.input_dim = 0
        if not cfg.get('no_rgb', False): 
            self.input_dim += FEATURE_SIZES.get(cfg['rgb_type'])
        if not cfg.get('no_flow', False): 
            self.input_dim += FEATURE_SIZES.get(cfg['flow_type'])

        # Encoder & Heads (Giữ nguyên)
        self.encoder_q = MROAD(cfg)
        self.encoder_k = MROAD(cfg)

        if cfg.get('pretrained_backbone_path'):
            self._load_pretrained_backbone(cfg['pretrained_backbone_path'])

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False 

        self.head_queue = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.contrastive_dim)
        )

        # Memory Bank (Khởi tạo Random như đã bàn)
        self.register_buffer("queues", torch.randn(self.num_classes, self.contrastive_dim, self.K))
        self.queues = F.normalize(self.queues, dim=1) 
        self.register_buffer("queue_ptrs", torch.zeros(self.num_classes, dtype=torch.long))

    def _load_pretrained_backbone(self, path):
        # (Code load backbone giữ nguyên)
        print(f"--> Loading Pre-trained Backbone from: {path}")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        backbone_dict = {}
        for k, v in state_dict.items():
            if 'f_classification' not in k: 
                new_k = k.replace('module.', '')
                backbone_dict[new_k] = v
        self.encoder_q.load_state_dict(backbone_dict, strict=False)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # (Code update queue giữ nguyên)
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = (labels == label)
            keys_subset = keys[mask] 
            lbl_idx = label.item()
            ptr = int(self.queue_ptrs[lbl_idx])
            batch_size_subset = keys_subset.shape[0]
            if ptr + batch_size_subset <= self.K:
                self.queues[lbl_idx, :, ptr : ptr + batch_size_subset] = keys_subset.T
                ptr = (ptr + batch_size_subset) % self.K
            else:
                remaining = self.K - ptr
                self.queues[lbl_idx, :, ptr:] = keys_subset[:remaining].T
                overflow = batch_size_subset - remaining
                self.queues[lbl_idx, :, :overflow] = keys_subset[remaining:].T
                ptr = overflow
            self.queue_ptrs[lbl_idx] = ptr

    def forward(self, rgb_anchor, flow_anchor, rgb_shuff, flow_shuff, labels, labels_per_frame=None):
        """
        Modified Forward:
        - Nhận thêm labels_per_frame để tạo Semantic Mask.
        - Trả về q_cls (Anchor), k_cls (Positive), q_shuff (Negative 1), q_context (Negative 2)
        """
        # Input gộp
        x_raw = torch.cat((rgb_anchor, flow_anchor), dim=2) 
        
        # --- A. GENERATOR STEP (Chiến thuật 3 nhánh) ---
        # x_core: Dùng làm Anchor (q_cls)
        # x_context: Dùng làm Negative (q_context)
        if labels_per_frame is None: 
            # Fallback nếu quên sửa dataset (dù không nên xảy ra)
            x_core = x_raw 
            x_context = torch.zeros_like(x_raw)
        else:
            x_core, x_context = self.generator(x_raw, labels_per_frame, labels)
        
        # Tách RGB/Flow cho x_core
        rgb_dim = rgb_anchor.shape[2]
        rgb_core = x_core[:, :, :rgb_dim]
        flow_core = x_core[:, :, rgb_dim:]

        # Tách RGB/Flow cho x_context (Fake BG)
        rgb_ctx = x_context[:, :, :rgb_dim]
        flow_ctx = x_context[:, :, rgb_dim:]

        # --- B. ENCODER STUDENT ---
        
        # 1. Nhánh Anchor (Core Action)
        feat_q = self.encoder_q(rgb_core, flow_core, return_embedding=True)
        q_cls = F.normalize(self.head_queue(feat_q), dim=1) 

        # 2. Nhánh Context (Fake Background - Negative)
        # Chỉ tính toán, kết quả sẽ được đẩy vào loss để ĐẨY RA XA
        feat_ctx = self.encoder_q(rgb_ctx, flow_ctx, return_embedding=True)
        q_context = F.normalize(self.head_queue(feat_ctx), dim=1)

        # 3. Nhánh Shuffle (Temporal Chaos - Negative)
        # Đã được bật lại!
        feat_shuff = self.encoder_q(rgb_shuff, flow_shuff, return_embedding=True)
        q_shuff = F.normalize(self.head_queue(feat_shuff), dim=1)

        # --- C. ENCODER TEACHER (KEY) ---
        with torch.no_grad():
            self._momentum_update_key_encoder() 
            # Teacher nhìn thấy toàn bộ input sạch (Full Context + Action)
            feat_k = self.encoder_k(rgb_anchor, flow_anchor, return_embedding=True)
            k_cls = F.normalize(self.head_queue(feat_k), dim=1)

        # --- D. RETURN ---
        return {
            'q_cls': q_cls,        # Anchor (Core Action)
            'k_cls': k_cls,        # Positive (Full Info)
            'q_shuff': q_shuff,    # Hard Negative 1 (Sai thời gian)
            'q_context': q_context,# Hard Negative 2 (Chỉ có nền, thiếu action)
            'queues': self.queues, 
            'queue_ptrs': self.queue_ptrs,
        }
    
    def update_queue(self, k_cls, labels):
        self._dequeue_and_enqueue(k_cls, labels)

@META_ARCHITECTURES.register("MiniROAD")
class MROAD(nn.Module):
    
    def __init__(self, cfg):
        super(MROAD, self).__init__()
        self.use_flow = not cfg['no_flow']
        self.use_rgb = not cfg['no_rgb']
        self.input_dim = 0
        if self.use_rgb:
            self.input_dim += FEATURE_SIZES[cfg['rgb_type']]
        if self.use_flow:
            self.input_dim += FEATURE_SIZES[cfg['flow_type']]

        self.hidden_dim = cfg['hidden_dim']
        self.num_layers = cfg['num_layers']
        self.out_dim = cfg['num_classes']
        self.window_size = cfg['window_size']

        self.relu = nn.ReLU()
        self.embedding_dim = cfg['embedding_dim']
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg['dropout']),
        )
        self.f_classification = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim)
        )
        # self.h0 = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.h0 = torch.zeros(self.num_layers, 1, self.hidden_dim)

    def forward(self, rgb_input, flow_input, return_embedding=False):
        if self.use_rgb and self.use_flow:
            x = torch.cat((rgb_input, flow_input), 2)
        elif self.use_rgb:
            x = rgb_input
        elif self.use_flow:
            x = flow_input
        x = self.layer1(x)
        B, _, _ = x.shape
        h0 = self.h0.expand(-1, B, -1).to(x.device)
        ht, _ = self.gru(x, h0) 
        ht = self.relu(ht)
        if return_embedding:
            return ht[:, -1, :]
        # ht = self.relu(ht + x)
        logits = self.f_classification(ht)
        out_dict = {}
        if self.training:
            out_dict['logits'] = logits
        else:
            pred_scores = F.softmax(logits, dim=-1)
            out_dict['logits'] = pred_scores
        return out_dict

@META_ARCHITECTURES.register("MiniROADA")
class MROADA(nn.Module):
    
    def __init__(self, cfg):
        super(MROADA, self).__init__()
        self.use_flow = not cfg['no_flow']
        self.use_rgb = not cfg['no_rgb']
        self.input_dim = 0
        if self.use_rgb:
            self.input_dim += FEATURE_SIZES[cfg['rgb_type']]
        if self.use_flow:
            self.input_dim += FEATURE_SIZES[cfg['flow_type']]

        self.embedding_dim = cfg['embedding_dim']
        self.hidden_dim = cfg['hidden_dim']
        self.num_layers = cfg['num_layers']
        self.anticipation_length = cfg["anticipation_length"]
        self.out_dim = cfg['num_classes']

        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg['dropout'])
        )
        self.actionness = cfg['actionness']
        if self.actionness:
            self.f_actionness = nn.Sequential(
                nn.Linear(self.hidden_dim, 1),
            )
        self.relu = nn.ReLU()
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.f_classification = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim)
        )
        self.anticipation_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.anticipation_length*self.hidden_dim),
        )


    def forward(self, rgb_input, flow_input):
        if self.use_rgb and self.use_flow:
            x = torch.cat((rgb_input, flow_input), 2)
        elif self.use_rgb:
            x = rgb_input
        elif self.use_flow:
            x = flow_input
        B, S, _ = x.shape
        x = self.layer1(x)
        h0 = torch.zeros(1, B, self.hidden_dim).to(x.device)
        ht, _ = self.gru(x, h0)
        logits = self.f_classification(self.relu(ht))
        anticipation_ht = self.anticipation_layer(self.relu(ht)).view(B, S, self.anticipation_length, self.hidden_dim)
        anticipation_logits = self.f_classification(self.relu(anticipation_ht))
        out_dict = {}
        if self.training:
            out_dict['logits'] = logits
            out_dict['anticipation_logits'] = anticipation_logits
        else:
            pred_scores = F.softmax(logits, dim=-1)
            pred_anticipation_scores = F.softmax(anticipation_logits, dim=-1)
            out_dict['logits'] = pred_scores
            out_dict['anticipation_logits'] = pred_anticipation_scores

        return out_dict