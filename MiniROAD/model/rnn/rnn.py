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

class RandomMaskGenerator(nn.Module):
    def __init__(self, mask_ratio=0.25):
        """
        Args:
            mask_ratio (float): Tỷ lệ frame bị che (VD: 0.25 = 25%).
        """
        super(RandomMaskGenerator, self).__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        """
        Input:
            x: Feature tensor [Batch, Window_Size, Dim] (Đã nối RGB + Flow)
        Output:
            x_masked: Tensor đã bị che [Batch, Window_Size, Dim]
            mask: Binary mask [Batch, Window_Size, 1] (0 là bị che, 1 là giữ)
        """
        B, T, D = x.shape
        device = x.device
        rand_tensor = torch.rand(B, T - 1, 1, device=device)
        mask_past = (rand_tensor > self.mask_ratio).float()
        mask_last = torch.ones(B, 1, 1, device=device)
        mask = torch.cat([mask_past, mask_last], dim=1)
        x_masked = x * mask
        return x_masked
    
@META_ARCHITECTURES.register("CONTRASTIVE_MROAD")
class ContrastiveMROADMultiQueue(nn.Module):
    def __init__(self, cfg):
        """
        MoCo v2 + Multi-Queue Architecture (Simplified Mapping)
        """
        super(ContrastiveMROADMultiQueue, self).__init__()
        self.contrastive_dim = cfg.get('contrastive_dim', 128)
        self.hidden_dim = cfg.get('hidden_dim', 2048)
        self.num_classes = cfg['num_classes']
        self.generator = RandomMaskGenerator(mask_ratio=0.25)
        
        # K: Queue Size per Class
        self.K = cfg.get('queue_size_per_class', 1024) 
        
        self.m = cfg.get('momentum', 0.999)
        self.T = cfg.get('temperature', 0.07)

        # Input Dim Setup
        self.input_dim = 0
        if not cfg.get('no_rgb', False): 
            self.input_dim += FEATURE_SIZES.get(cfg['rgb_type'])
        if not cfg.get('no_flow', False): 
            self.input_dim += FEATURE_SIZES.get(cfg['flow_type'])

        # --- 1. NETWORK INIT ---
        self.encoder_q = MROAD(cfg)
        self.encoder_k = MROAD(cfg)

        # Load Pretrained
        if cfg.get('pretrained_backbone_path'):
            self._load_pretrained_backbone(cfg['pretrained_backbone_path'])

        # Copy Weights Student -> Teacher
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False 

        # --- 2. PROJECTION HEADS ---
        self.head_queue = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.contrastive_dim)
        )

        # --- 3. MEMORY BANK (MULTI-QUEUE) ---
        self.register_buffer("queues", torch.zeros(self.num_classes, self.contrastive_dim, self.K))
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
        msg = self.encoder_q.load_state_dict(backbone_dict, strict=False)
        print(f"--> Load status: {msg}")

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        """
        Cập nhật Queue dựa trên Labels trực tiếp.
        Logic cũ: Cần map queue_id.
        Logic mới: Label chính là Queue ID.
        """
        unique_labels = torch.unique(labels)

        for label in unique_labels:
            mask = (labels == label)
            keys_subset = keys[mask] 
            
            # Lấy thẳng label làm index cho queue (Không cần mapping phức tạp)
            lbl_idx = label.item()
            
            # --- Logic cập nhật Circular Buffer (Giữ nguyên) ---
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

    def forward(self, rgb_anchor, flow_anchor, rgb_shuff, flow_shuff, labels):
        """
        Baseline Mode:
        - Bỏ qua tính toán nhánh Shuffle để tiết kiệm GPU (vì input giống hệt Anchor).
        """
        # --- A. GENERATOR STEP ---
        x_anchor = torch.cat((rgb_anchor, flow_anchor), dim=2) 
        
        # Masking 25% (Trừ frame cuối)
        x_gen = self.generator(x_anchor) 
        
        # Tách RGB/Flow
        rgb_dim = rgb_anchor.shape[2]
        rgb_gen = x_gen[:, :, :rgb_dim]
        flow_gen = x_gen[:, :, rgb_dim:]

        # --- B. ENCODER STUDENT (QUERY) ---
        # 1. Tính toán feature cho Anchor (đã bị Mask)
        feat_q = self.encoder_q(rgb_gen, flow_gen, return_embedding=True) # [B, Hidden]
        
        # Head 1: MoCo Projection
        q_cls = F.normalize(self.head_queue(feat_q), dim=1) 

        # feat_shuff = self.encoder_q(rgb_shuff, flow_shuff, return_embedding=True) # <--- SKIP
        # q_shuff = F.normalize(self.head_queue(feat_shuff), dim=1)                 # <--- SKIP
        q_shuff = q_cls 

        # --- C. ENCODER TEACHER (KEY) ---
        with torch.no_grad():
            self._momentum_update_key_encoder() 
            feat_k = self.encoder_k(rgb_anchor, flow_anchor, return_embedding=True)
            k_cls = F.normalize(self.head_queue(feat_k), dim=1)

        # --- D. RETURN ---
        return {
            'q_cls': q_cls,        # Query (Masked)
            'k_cls': k_cls,        # Key (Clean)
            'q_shuff': q_shuff,    # Shuffle (Baseline = Query)           
            'queues': self.queues, # Memory Bank
            'queue_ptrs': self.queue_ptrs,
            'feat_q': feat_q,      
            'feat_k': feat_k  
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