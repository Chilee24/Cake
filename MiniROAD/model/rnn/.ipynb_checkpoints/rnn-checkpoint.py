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
    'rgb_kinetics_i3d': 2048,
    'flow_kinetics_i3d': 2048,
    'cake_kinetics': 4096
}

# 1. Base Encoder: Phần cốt lõi (Linear + GRU)
class GRUEncoder(nn.Module):
    def __init__(self, cfg, input_dim):
        super(GRUEncoder, self).__init__()
        self.hidden_dim = cfg['hidden_dim']
        self.num_layers = cfg['num_layers']
        self.embedding_dim = cfg['embedding_dim']
        self.dropout = cfg['dropout']
        
        # Projection Input -> Embedding
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )
        
        # Temporal Aggregation
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, 
                          self.num_layers, batch_first=True)
        
        # Init Hidden State
        self.register_buffer("h0", torch.zeros(self.num_layers, 1, self.hidden_dim))

    def forward(self, x):
        # x: (B, Seq, Input_Dim)
        x = self.layer1(x)
        
        B = x.size(0)
        h0 = self.h0.expand(-1, B, -1).contiguous()
        
        # GRU Forward
        # out: (B, Seq, Hidden), hn: (Layers, B, Hidden)
        out, _ = self.gru(x, h0)
        
        # Chỉ lấy Frame cuối cùng làm đại diện (cho OAD)
        # out_last: (B, Hidden)
        out_last = out[:, -1, :]
        return out_last

# 2. Main Model: MROAD Phase 1 (MoCo Wrapper)
@META_ARCHITECTURES.register("MROAD_P1")
class MROAD_P1(nn.Module):
    
    def __init__(self, cfg):
        super(MROAD_P1, self).__init__()
        
        # --- Configs ---
        self.use_flow = not cfg['no_flow']
        self.use_rgb = not cfg['no_rgb']
        
        input_dim = 0
        if self.use_rgb: input_dim += FEATURE_SIZES[cfg['rgb_type']]
        if self.use_flow: input_dim += FEATURE_SIZES[cfg['flow_type']]
        
        # MoCo Configs
        self.dim = cfg.get('proj_dim', 128)   # Output Dimension cho Contrastive
        self.K = cfg.get('queue_size', 2048)  # Kích thước mỗi hàng đợi
        self.m = cfg.get('momentum', 0.999)   # Momentum update
        self.T = cfg.get('tau', 0.07)
        
        # Số lượng hàng đợi = Số class Action + 1 Background
        # THUMOS: 20 Action + 1 BG = 21 Queues
        # Nếu dataset trả về class_id từ 0-20, thì ta cần 21 queues.
        self.num_queues = cfg['num_classes'] 
        if cfg.get('bg_in_queues', True): 
            # Đảm bảo có chỗ cho BG nếu num_classes chưa bao gồm
             pass 

        # --- A. Encoders (Student & Teacher) ---
        self.encoder_q = GRUEncoder(cfg, input_dim)
        self.encoder_k = GRUEncoder(cfg, input_dim)
        
        # Projectors (Hidden -> Contrastive Dim)
        self.proj_q = nn.Sequential(
            nn.Linear(cfg['hidden_dim'], cfg['hidden_dim']),
            nn.ReLU(),
            nn.Linear(cfg['hidden_dim'], self.dim)
        )
        self.proj_k = nn.Sequential(
            nn.Linear(cfg['hidden_dim'], cfg['hidden_dim']),
            nn.ReLU(),
            nn.Linear(cfg['hidden_dim'], self.dim)
        )

        # Init Weights cho Teacher = Student
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # Teacher không update bằng Gradient
            
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # --- B. Multi-Queue Memory Bank ---
        # Shape: (Num_Queues, K, Dim)
        self.register_buffer("queues", torch.randn(self.num_queues, self.K, self.dim))
        self.queues = F.normalize(self.queues, dim=2) # Normalize ngay từ đầu
        self.register_buffer("queue_ptr", torch.zeros(self.num_queues, dtype=torch.long))
        
        # Load Pretrained (Nếu có)
        # if cfg.get('pretrained_path'):
        self._load_from_pretrained(r"D:\project\DashCam\model\MiniROAD\THUMOS_Kinetics.pth")

    def _load_from_pretrained(self, path):
        print(f"--> Loading pretrained weights from {path}")
        checkpoint = torch.load(path, map_location='cpu')
        
        # Lọc key để chỉ load vào encoder_q và encoder_k
        # Nếu checkpoint cũ là model MROAD thường (chỉ có 'gru', 'layer1'...)
        # Ta cần map nó vào 'encoder_q.gru', 'encoder_q.layer1'...
        state_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint['state_dict']
        new_dict = {}
        for k, v in state_dict.items():
            # Map weights cũ vào Student Encoder
            if not k.startswith('encoder'):
                 new_dict[f'encoder_q.{k}'] = v
                 new_dict[f'encoder_k.{k}'] = v # Load cả cho Teacher
            else:
                 new_dict[k] = v
                 
        msg = self.load_state_dict(new_dict, strict=False)
        print(f"--> Load result: {msg}")

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """ Update Teacher weights: k = m*k + (1-m)*q """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_ids):
        """
        keys: (B, Dim) - Feature từ Teacher
        queue_ids: (B,) - ID của hàng đợi tương ứng (từ Dataset)
        """
        # Gather keys theo từng Queue ID
        unique_ids = torch.unique(queue_ids)
        
        for q_id in unique_ids:
            # Lấy các mẫu thuộc queue này
            mask = (queue_ids == q_id)
            feats = keys[mask]
            
            if feats.shape[0] == 0: continue
            
            ptr = int(self.queue_ptr[q_id])
            num_feats = feats.shape[0]
            
            # Xử lý trường hợp số lượng mẫu > sức chứa còn lại (tràn vòng)
            # Để đơn giản, ta replace tuần tự
            if ptr + num_feats <= self.K:
                self.queues[q_id, ptr : ptr + num_feats, :] = feats
                self.queue_ptr[q_id] = (ptr + num_feats) % self.K
            else:
                # Nếu tràn, điền phần còn lại rồi quay về 0 điền tiếp
                rem = self.K - ptr
                self.queues[q_id, ptr : self.K, :] = feats[:rem]
                self.queues[q_id, 0 : num_feats - rem, :] = feats[rem:]
                self.queue_ptr[q_id] = num_feats - rem

    def forward(self, rgb_input, flow_input):
        # 1. Input Preparation
        if self.use_rgb and self.use_flow:
            x = torch.cat((rgb_input, flow_input), 2)
        elif self.use_rgb:
            x = rgb_input
        elif self.use_flow:
            x = flow_input
            
        # 2. Student Forward (Query)
        # q_last: (B, Hidden)
        h_q = self.encoder_q(x) 
        z_q = F.normalize(self.proj_q(h_q), dim=1)
        
        # 3. Teacher Forward (Key) & Shuffle Logic
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            # a. Teacher Clean (Positive)
            h_k = self.encoder_k(x)
            z_k = F.normalize(self.proj_k(h_k), dim=1)
            
            # b. Student Shuffled (Hard Negative)
            # Shuffle chiều thời gian (Dim 1)
            # Lưu ý: Mỗi mẫu trong batch shuffle ngẫu nhiên khác nhau
            idx = torch.randperm(x.size(1), device=x.device)
            x_shuf = x[:, idx, :] 
            
            # Forward qua STUDENT (vì ta muốn phạt Student nếu nó ko phân biệt đc)
            # Hoặc forward qua Teacher cũng được, nhưng thường phạt Student trực tiếp sẽ tốt hơn
            h_shuf = self.encoder_q(x_shuf) 
            z_shuf = F.normalize(self.proj_q(h_shuf), dim=1)

        # 4. Return Output Dictionary
        out_dict = {
            'z_anchor': z_q,    # Student Clean
            'z_aug': z_k,       # Teacher Clean
            'z_shuf': z_shuf,   # Student Shuffled
            'queues': self.queues.clone().detach(), # Toàn bộ bộ nhớ (Num_Q, K, D)
            # 'logits': None    # Phase 1 không có Classifier
        }
        
        return out_dict

    def update_queues(self, z_teacher, queue_ids):
        """ Helper function để gọi từ Trainer """
        self._dequeue_and_enqueue(z_teacher, queue_ids)

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
        if cfg.get('phase1_ckpt'):
            self.load_phase1_weights(cfg['phase1_ckpt'])
            
        # [OPTIONAL] Freeze Backbone?
        if cfg.get('freeze_backbone', False):
            self._freeze_backbone()

    def load_phase1_weights(self, ckpt_path):
        print(f"--> [Linear Probe] Loading Phase 1 weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint['state_dict']
        
        new_dict = {}
        for k, v in state_dict.items():
            # Chỉ lấy trọng số của Student (encoder_q)
            if k.startswith('encoder_q.'):
                # Xóa tiền tố 'encoder_q.' để khớp với MROAD chuẩn
                new_key = k.replace('encoder_q.', '')
                new_dict[new_key] = v
                
        # Load vào model hiện tại (strict=False vì Phase 1 không có f_classification)
        msg = self.load_state_dict(new_dict, strict=False)
        print(f"--> Load status: {msg}")
        print("--> Backbone loaded. Classifier head initialized randomly.")

    def _freeze_backbone(self):
        print("--> FREEZING BACKBONE (Linear Probing Mode)")
        for name, param in self.named_parameters():
            # Freeze tất cả trừ lớp phân loại cuối cùng
            if "f_classification" not in name:
                param.requires_grad = False

    def forward(self, rgb_input, flow_input):
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