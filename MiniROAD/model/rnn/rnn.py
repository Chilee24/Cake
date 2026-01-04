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
        Output:
            x_core: 
                - Action Sample -> Giữ Action, che Nền.
                - BG Sample -> Giữ Nền, che Action (Pure BG).
            x_aug: Random Mask (Robustness).
        """
        B, T, D = x.shape
        device = x.device

        # --- 1. TẠO X_AUG (Random Mask - Robustness) ---
        # Mask ngẫu nhiên T-1 frame đầu, GIỮ FRAME CUỐI
        rand_tensor = torch.rand(B, T - 1, 1, device=device)
        mask_aug_past = (rand_tensor > self.mask_ratio).float()
        mask_aug_last = torch.ones(B, 1, 1, device=device)
        mask_aug = torch.cat([mask_aug_past, mask_aug_last], dim=1) # [B, T, 1]
        
        x_aug = x * mask_aug

        # --- 2. TẠO X_CORE (Semantic Focus) ---
        
        # Tạo mask cơ sở
        is_bg_frame = (labels_per_frame == self.bg_class_idx).unsqueeze(-1).float() # [B, T, 1]
        
        mask_action_focus = 1.0 - is_bg_frame # Giữ Action
        mask_bg_focus = is_bg_frame           # Giữ Nền
        
        # Phân loại mẫu (Action hay BG?)
        is_bg_sample = (final_labels == self.bg_class_idx).view(B, 1, 1) # [B, 1, 1]
        
        # LOGIC CHÍNH:
        # Nếu là BG Sample -> Dùng mask_bg_focus (để tạo Pure BG, loại bỏ action quá khứ)
        # Nếu là Action Sample -> Dùng mask_action_focus (để tạo Pure Action)
        mask_core = torch.where(is_bg_sample, mask_bg_focus, mask_action_focus)
        
        # --- 3. SAFETY CHECK (Chống đen xì) ---
        # Nếu mask toàn số 0 (VD: Action sample nhưng toàn nền, hoặc BG sample nhưng toàn action)
        # -> Fallback về mask_aug để tránh lãng phí
        has_content = mask_core.sum(dim=1, keepdim=True) > 0 
        mask_core = torch.where(has_content, mask_core, mask_aug)
        
        x_core = x * mask_core

        # Ta BỎ x_context vì không còn dùng nữa
        return x_core, x_aug

@META_ARCHITECTURES.register("CONTRASTIVE_MROAD")
class ContrastiveMROADMultiQueue(nn.Module):
    def __init__(self, cfg):
        super(ContrastiveMROADMultiQueue, self).__init__()
        self.contrastive_dim = cfg.get('contrastive_dim', 128)
        self.hidden_dim = cfg.get('hidden_dim', 2048)
        self.num_classes = cfg['num_classes']
        self.bg_class_idx = cfg.get('bg_class_idx', 0)
        
        # Config mask_ratio
        mr = cfg.get('mask_ratio')
        self.mask_ratio = mr if mr is not None else 0.5
        
        # Generator
        self.generator = SemanticMaskGenerator(bg_class_idx=self.bg_class_idx, mask_ratio=self.mask_ratio)
        
        self.K = cfg.get('queue_size_per_class', 1024) 
        self.m = cfg.get('momentum', 0.999)

        # Init Encoders
        self.encoder_q = MROAD(cfg)
        self.encoder_k = MROAD(cfg)

        if cfg.get('pretrained_backbone_path'):
            self._load_pretrained_backbone(cfg['pretrained_backbone_path'])

        # Copy weight k <- q
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False 

        # Projection Head
        self.head_queue = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.contrastive_dim)
        )

        # Memory Bank
        self.register_buffer("queues", torch.randn(self.num_classes, self.contrastive_dim, self.K))
        self.queues = F.normalize(self.queues, dim=1) 
        self.register_buffer("queue_ptrs", torch.zeros(self.num_classes, dtype=torch.long))

    def _load_pretrained_backbone(self, path):
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

    def forward(self, rgb_anchor, flow_anchor, labels, labels_per_frame=None):
        """
        Output: q_core, q_aug, k_cls (Đã bỏ q_context)
        """
        # Gộp input
        x_raw = torch.cat((rgb_anchor, flow_anchor), dim=2) 
        
        # --- A. GENERATOR STEP ---
        x_core, x_aug = self.generator(x_raw, labels_per_frame, labels)
        
        def split_feat(x_in):
            rgb_d = rgb_anchor.shape[2]
            return x_in[:, :, :rgb_d], x_in[:, :, rgb_d:]

        # --- B. ENCODER STUDENT ---
        
        # 1. Nhánh Core (Semantic - Action Focus hoặc BG Pure)
        r_core, f_core = split_feat(x_core)
        feat_core = self.encoder_q(r_core, f_core, return_embedding=True)
        q_core = F.normalize(self.head_queue(feat_core), dim=1) 

        # 2. Nhánh Aug (Robustness - Random Mask)
        r_aug, f_aug = split_feat(x_aug)
        feat_aug = self.encoder_q(r_aug, f_aug, return_embedding=True)
        q_aug = F.normalize(self.head_queue(feat_aug), dim=1)

        # --- C. ENCODER TEACHER ---
        with torch.no_grad():
            self._momentum_update_key_encoder() 
            feat_k = self.encoder_k(rgb_anchor, flow_anchor, return_embedding=True)
            k_cls = F.normalize(self.head_queue(feat_k), dim=1)
        
        # --- D. RETURN ---
        return {
            'q_core': q_core,      # Positive 1
            'q_aug': q_aug,        # Positive 2
            'k_cls': k_cls,        # Teacher Target
            'queues': self.queues, 
            'queue_ptrs': self.queue_ptrs
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
        # --- BACKBONE ---
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg['dropout']),
        )
        # --- HEAD ---
        self.f_classification = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim)
        )
        
        self.h0 = torch.zeros(self.num_layers, 1, self.hidden_dim)

        contrastive_path = cfg.get('contrastive_path', None)
        if contrastive_path:
            self._load_from_contrastive(contrastive_path)

        if cfg.get('freeze_backbone', False):
            self._freeze_backbone()

    def _load_from_contrastive(self, path):
        print(f"--> [Model] Loading Contrastive weights from: {path}")
        
        # --- SỬA DÒNG NÀY: Thêm weights_only=False ---
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        # ---------------------------------------------

        # Lấy state_dict (xử lý trường hợp lưu cả optimizer/epoch)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder_q.'):
                new_key = k.replace('encoder_q.', '')
                if 'f_classification' not in new_key:
                    new_state_dict[new_key] = v
            
        msg = self.load_state_dict(new_state_dict, strict=False)
        print(f"--> [Model] Weights loaded. Missing keys (expected for head): {msg.missing_keys}")

    def _freeze_backbone(self):
        print("--> [Model] FREEZING Backbone (Layer1 & GRU). Only training Head.")
        for name, param in self.named_parameters():
            if 'f_classification' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

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