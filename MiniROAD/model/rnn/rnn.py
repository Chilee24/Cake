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

@META_ARCHITECTURES.register("CONTRASTIVE_MROAD")
class ContrastiveMROADMultiQueue(nn.Module):
    def __init__(self, cfg):
        super(ContrastiveMROADMultiQueue, self).__init__()
        self.contrastive_dim = cfg.get('contrastive_dim', 128)
        self.hidden_dim = cfg.get('hidden_dim', 2048)
        self.num_classes = cfg['num_classes']
        self.bg_class_idx = cfg.get('bg_class_idx', 0)
        self.mask_ratio = cfg.get('mask_ratio', 0.25) 
        self.K = cfg.get('queue_size_per_class', 1024) 
        self.m = cfg.get('momentum', 0.999)
        
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
            nn.Linear(self.hidden_dim, self.contrastive_dim)
        )
        # Memory Bank: [Classes, Dim, K]
        self.register_buffer("queues", torch.randn(self.num_classes, self.contrastive_dim, self.K))
        self.queues = F.normalize(self.queues, dim=1) 
        self.register_buffer("queue_ptrs", torch.zeros(self.num_classes, dtype=torch.long))

    def _load_pretrained_backbone(self, path):
        print(f"--> Loading Pre-trained Backbone from: {path}")
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        backbone_dict = {}
        for k, v in state_dict.items():
            if 'f_classification' not in k and 'fc' not in k: 
                new_k = k.replace('module.', '')
                backbone_dict[new_k] = v
        self.encoder_q.load_state_dict(backbone_dict, strict=False)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, targets_multihot):
        for c in range(self.num_classes):
            idxs = torch.nonzero(targets_multihot[:, c] > 0.5).squeeze(1)
            if idxs.numel() == 0: continue
            
            keys_subset = keys[idxs]
            n_subset = keys_subset.shape[0]
            ptr = int(self.queue_ptrs[c])
            
            if ptr + n_subset <= self.K:
                self.queues[c, :, ptr : ptr + n_subset] = keys_subset.T
                ptr = (ptr + n_subset) % self.K
            else:
                remaining = self.K - ptr
                self.queues[c, :, ptr:] = keys_subset[:remaining].T
                overflow = n_subset - remaining
                self.queues[c, :, :overflow] = keys_subset[remaining:].T
                ptr = overflow
            self.queue_ptrs[c] = ptr

    def forward(self, rgb_anchor, flow_anchor, labels, targets_multihot=None, labels_per_frame=None):
        if self.training and self.mask_ratio > 0:
            B, T, D = rgb_anchor.shape
            device = rgb_anchor.device
            rand_tensor = torch.rand(B, T - 1, 1, device=device)
            mask_past = (rand_tensor > self.mask_ratio).float()
            mask_last = torch.ones(B, 1, 1, device=device)
            mask = torch.cat([mask_past, mask_last], dim=1)
            rgb_student = rgb_anchor * mask
            flow_student = flow_anchor * mask
        else:
            rgb_student = rgb_anchor
            flow_student = flow_anchor

        # --- STUDENT FORWARD---
        feat_student = self.encoder_q(rgb_student, flow_student, return_embedding=True)
        q_student = F.normalize(self.head_queue(feat_student), dim=1) 

        # --- TEACHER FORWARD---
        with torch.no_grad():
            self._momentum_update_key_encoder() 
            feat_k = self.encoder_k(rgb_anchor, flow_anchor, return_embedding=True)
            k_cls = F.normalize(self.head_queue(feat_k), dim=1)
        
        return {
            'q_cls': q_student,      
            'k_cls': k_cls,        
            'queues': self.queues,
            'queue_ptrs': self.queue_ptrs
        }
    
    def update_queue(self, k_cls, targets_multihot):
        self._dequeue_and_enqueue(k_cls, targets_multihot)

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

        # --- BACKBONE (Giữ nguyên MiniROAD) ---
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg['dropout']),
        )

        # --- NEW HEAD (ACTIONFORMER STYLE) ---
        self.f_classification = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim)
        )
        
        self.h0 = torch.zeros(self.num_layers, 1, self.hidden_dim)

        # --- LOAD WEIGHTS TỪ CONTRASTIVE ---
        contrastive_path = cfg.get('contrastive_path', None)
        if contrastive_path:
            self._load_from_contrastive(contrastive_path)

        if cfg.get('freeze_backbone', False):
            self._freeze_backbone()

    def _load_from_contrastive(self, path):
        print(f"--> [Model] Loading Contrastive weights from: {path}")
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder_q.'):
                new_key = k.replace('encoder_q.', '')
                if 'f_classification' not in new_key:
                    new_state_dict[new_key] = v
            
        msg = self.load_state_dict(new_state_dict, strict=False)
        print(f"--> [Model] Weights loaded. Missing keys (expected for new head): {msg.missing_keys}")

    def _freeze_backbone(self):
        print("--> [Model] FREEZING Backbone (Layer1 & GRU). Only training Head.")
        for name, param in self.named_parameters():
            if 'f_classification' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, rgb_input, flow_input, return_embedding=False):
        # 1. Feature Aggregation
        if self.use_rgb and self.use_flow:
            x = torch.cat((rgb_input, flow_input), 2)
        elif self.use_rgb:
            x = rgb_input
        elif self.use_flow:
            x = flow_input
            
        # 2. Backbone Forward
        x = self.layer1(x)
        B, _, _ = x.shape
        h0 = self.h0.expand(-1, B, -1).to(x.device)
        ht, _ = self.gru(x, h0) 
        ht = self.relu(ht)

        # Trả về embedding nếu cần (cho contrastive learning hoặc visualization)
        if return_embedding:
            return ht[:, -1, :]

        # 3. New Classification Head
        # Input: Hidden state sequence (B, T, Hidden_Dim)
        # Output: Logits (B, T, K)
        logits = self.f_classification(ht) 

        out_dict = {}
        if self.training:
            out_dict['logits'] = logits
        else:
            pred_scores = torch.sigmoid(logits)
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