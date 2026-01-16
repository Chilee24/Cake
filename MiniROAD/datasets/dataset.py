import torch
import torch.utils.data as data
import numpy as np
import json
import os.path as osp
import gc
from datasets.dataset_builder import DATA_LAYERS
import random

FEATURE_SIZES = {
    'rgb_anet_resnet50': 2048,
    'flow_anet_resnet50': 2048,
    'rgb_kinetics_bninception': 1024,
    'flow_kinetics_bninception': 1024,
    'rgb_kinetics_resnet50': 2048,
    'flow_kinetics_resnet50': 2048,
    'flow_nv_kinetics_bninception': 1024,
    'rgb_kinetics_x3d': 2048,
    'flow_kinetics_x3d': 2048
}

@DATA_LAYERS.register("THUMOS")
class ContrastiveOADDataset(data.Dataset):
    
    def __init__(self, cfg, mode='train'):
        self.root_path = cfg['root_path']
        self.mode = mode
        self.training = mode == 'train'
        self.window_size = cfg['window_size']
        self.stride = cfg['stride']
        self.num_classes = cfg['num_classes']
        self.bg_class_idx = cfg.get('bg_class_idx', 0)
        
        self.inputs = [] 
        
        data_name = cfg['data_name']
        # Load list video
        with open(cfg['video_list_path'], 'r') as f:
            self.vids = json.load(f)[data_name][mode + '_session_set']
        
        # 1. LOAD DATA & PADDING
        self._load_features(cfg)
        
        # 2. KHỞI TẠO GRID
        if self.training:
            self.shuffle_indices()
        else:
            self._init_test_indices()
            
    def _load_features(self, cfg):
        self.annotation_type = cfg['annotation_type']
        self.rgb_type = cfg['rgb_type']
        self.flow_type = cfg['flow_type']
        
        self.target_all = {}
        self.rgb_inputs = {}
        self.flow_inputs = {}
        
        # Padding length
        pad_len = self.window_size - 1
        
        # Dummy Target: Toàn bộ là Background
        dummy_target = np.zeros((pad_len, self.num_classes))
        dummy_target[:, self.bg_class_idx] = 1.0 

        # Lấy feature dim từ dict hoặc cfg
        rgb_dim = FEATURE_SIZES.get(self.rgb_type)
        flow_dim = FEATURE_SIZES.get(self.flow_type)

        dummy_rgb = np.zeros((pad_len, rgb_dim))
        dummy_flow = np.zeros((pad_len, flow_dim))

        print(f"--> [Dataset] Pre-loading features into RAM ({self.mode})...")
        for vid in self.vids:
            # Load npy files
            target = np.load(osp.join(self.root_path, self.annotation_type, vid + '.npy'))
            rgb = np.load(osp.join(self.root_path, self.rgb_type, vid + '.npy'))
            flow = np.load(osp.join(self.root_path, self.flow_type, vid + '.npy'))
            
            # Padding
            self.target_all[vid] = np.concatenate((dummy_target, target), axis=0)
            self.rgb_inputs[vid] = np.concatenate((dummy_rgb, rgb), axis=0)
            self.flow_inputs[vid] = np.concatenate((dummy_flow, flow), axis=0)

    def shuffle_indices(self):
        self.inputs = []
        print(f"--> [Dataset] Reshuffling grid with stride={self.stride}...")
        
        for vid in self.vids:
            target = self.target_all[vid]
            total_frames = target.shape[0]
            
            # Random offset cho stride
            seed = np.random.randint(self.stride)
            
            # Sliding Window
            for start in range(seed, total_frames - self.window_size + 1, self.stride):
                end = start + self.window_size
                
                # Label argmax dùng cho Generator (để tạo mask semantic)
                last_frame_vec = target[end - 1] 
                label_idx = np.argmax(last_frame_vec)
                
                self.inputs.append([vid, start, end, label_idx])
                
        random.shuffle(self.inputs)

    def _init_test_indices(self):
        self.inputs = []
        for vid in self.vids:
            target = self.target_all[vid]
            total_frames = target.shape[0]
            
            # Test stride = 1
            for start in range(0, total_frames - self.window_size + 1, 1):
                 end = start + self.window_size
                 last_frame_vec = target[end - 1]
                 label_idx = np.argmax(last_frame_vec)
                 self.inputs.append([vid, start, end, label_idx])

    def __getitem__(self, index):
        vid, start, end, label_idx = self.inputs[index]
        
        # 1. GET DATA
        rgb_input = self.rgb_inputs[vid][start:end]
        flow_input = self.flow_inputs[vid][start:end]
        target_window = self.target_all[vid][start:end] 
        
        # [QUAN TRỌNG] Lấy vector Multi-hot của frame cuối
        # Model sẽ dùng vector này để biết cần push sample vào những queue nào
        last_frame_vec = target_window[-1]

        # 2. CONVERT TENSOR
        rgb_tensor = torch.tensor(rgb_input.astype(np.float32))
        flow_tensor = torch.tensor(flow_input.astype(np.float32))
        
        # targets_multihot: FloatTensor [Num_Classes] (VD: [0, 1, 0, 1...])
        targets_multihot = torch.tensor(last_frame_vec, dtype=torch.float32)

        # 3. LẤY NHÃN LỊCH SỬ
        labels_per_frame = torch.tensor(np.argmax(target_window, axis=1), dtype=torch.long)

        return {
            'rgb_anchor': rgb_tensor,
            'flow_anchor': flow_tensor,
            'labels': torch.tensor(label_idx, dtype=torch.long),
            'targets_multihot': targets_multihot,
            'labels_per_frame': labels_per_frame
        }

    def __len__(self):
        return len(self.inputs)

# @DATA_LAYERS.register("THUMOS")
# @DATA_LAYERS.register("TVSERIES")
class THUMOSDataset(data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.root_path = cfg['root_path']
        self.mode = mode
        self.training = mode == 'train'
        self.window_size = cfg['window_size']
        self.stride = cfg['stride']
        data_name = cfg['data_name']
        self.vids = json.load(open(cfg['video_list_path']))[data_name][mode + '_session_set']
        self.num_classes = cfg['num_classes'] 
        self.bg_idx = cfg.get('bg_idx', 0)
        self.inputs = []
        
        self._load_targets_only(cfg)
        self._init_features()
        print(f"--> [Lazy Mode] Mode: {mode} | Samples: {len(self.inputs)}")
        
    def _load_targets_only(self, cfg):
        self.annotation_type = cfg['annotation_type']
        self.rgb_type = cfg['rgb_type']
        self.flow_type = cfg['flow_type']
        self.target_all = {}
        # Biến để lưu dummy target (Padding), khởi tạo là None
        dummy_target = None

        for vid in self.vids:
            # Load nhãn gốc từ file (Vẫn chứa cột Background)
            # Shape: [Time, K + 1]
            raw_target = np.load(osp.join(self.root_path, self.annotation_type, vid + '.npy'))
            
            # --- SỬA ĐỔI CHÍNH: XÓA CỘT BACKGROUND ---
            # Dùng np.delete để xóa cột tại vị trí self.bg_idx
            # Kết quả: target chỉ còn K cột (chỉ chứa các Action)
            # Nếu raw là Background [1, 0, ...] -> target thành [0, ...] (Vector 0)
            target = np.delete(raw_target, self.bg_idx, axis=1)
            
            # --- TẠO DUMMY TARGET (Chỉ cần làm 1 lần) ---
            if dummy_target is None and self.training:
                # Lấy số chiều thực tế của Action (K)
                actual_num_classes = target.shape[1]
                
                # Tạo vector toàn số 0. 
                # Ý nghĩa: Padding (màn hình đen) = Không có hành động = Vector 0
                dummy_target = np.zeros((self.window_size - 1, actual_num_classes))
            
            # --- PADDING & LƯU ---
            if self.training:
                # Nối phần padding (toàn 0) vào trước video
                self.target_all[vid] = np.concatenate((dummy_target, target), axis=0)
            else:
                # Eval: Chỉ lưu target đã cắt bỏ cột nền
                self.target_all[vid] = target

    def _init_features(self):
        del self.inputs
        gc.collect()
        self.inputs = []
        for vid in self.vids:
            target = self.target_all[vid]
            if self.training:
                seed = np.random.randint(self.stride)
                for start, end in zip(range(seed, target.shape[0], self.stride), 
                                      range(seed + self.window_size, target.shape[0] + 1, self.stride)):
                    self.inputs.append([vid, start, end, target[start:end]])
            else:
                # Eval: Trả về tọa độ thực
                start = 0
                end = target.shape[0]
                self.inputs.append([vid, start, end, target[start:end]])

    def __getitem__(self, index):
        vid, start, end, target = self.inputs[index]
        
        rgb_path = osp.join(self.root_path, self.rgb_type, vid + '.npy')
        flow_path = osp.join(self.root_path, self.flow_type, vid + '.npy')
        
        rgb_mmap = np.load(rgb_path, mmap_mode='r')
        flow_mmap = np.load(flow_path, mmap_mode='r')
        
        # --- LOGIC PADDING INPUT GIỮ NGUYÊN ---
        if self.training:
            pad_len = self.window_size - 1
            real_start = start - pad_len
            real_end = end - pad_len
            
            if real_start >= 0:
                rgb_val = rgb_mmap[real_start:real_end]
                flow_val = flow_mmap[real_start:real_end]
            else:
                pad_needed = abs(real_start)
                rgb_part = rgb_mmap[0:max(0, real_end)]
                flow_part = flow_mmap[0:max(0, real_end)]
                
                rgb_zeros = np.zeros((pad_needed, rgb_mmap.shape[1]), dtype=np.float32)
                flow_zeros = np.zeros((pad_needed, flow_mmap.shape[1]), dtype=np.float32)
                
                rgb_val = np.concatenate((rgb_zeros, rgb_part), axis=0)
                flow_val = np.concatenate((flow_zeros, flow_part), axis=0)
            
            rgb_val = rgb_val[:self.window_size]
            flow_val = flow_val[:self.window_size]
        else:
            rgb_val = rgb_mmap[start:end]
            flow_val = flow_mmap[start:end]

        rgb_input = torch.tensor(rgb_val.astype(np.float32))
        flow_input = torch.tensor(flow_val.astype(np.float32))
        target_tensor = torch.tensor(target.astype(np.float32))
        
        return rgb_input.clone(), flow_input.clone(), target_tensor

    def __len__(self):
        return len(self.inputs)
    

@DATA_LAYERS.register("THUMOS_ANTICIPATION")
@DATA_LAYERS.register("TVSERIES_ANTICIPATION")
class THUMOSDataset(data.Dataset):
    
    def __init__(self, cfg, mode='train'):
        self.root_path = cfg['root_path']
        self.mode = mode
        self.training = mode == 'train'
        self.window_size = cfg['window_size']
        self.stride = cfg['stride']
        self.anticipation_length = cfg['anticipation_length']
        data_name = cfg["data_name"].split('_')[0]
        self.vids = json.load(open(cfg['video_list_path']))[data_name][mode + '_session_set'] # list of video names
        self.num_classes = cfg['num_classes']
        self.inputs = []
        self._load_features(cfg)
        self._init_features()
        
    def _load_features(self, cfg):
        self.annotation_type = cfg['annotation_type']
        self.rgb_type = cfg['rgb_type']
        self.flow_type = cfg['flow_type']
        self.target_all = {}
        self.rgb_inputs = {}
        self.flow_inputs = {}
        dummy_target = np.zeros((self.window_size-1, self.num_classes))
        dummy_rgb = np.zeros((self.window_size-1, FEATURE_SIZES[cfg['rgb_type']]))
        dummy_flow = np.zeros((self.window_size-1, FEATURE_SIZES[cfg['flow_type']]))
        for vid in self.vids:
            target = np.load(osp.join(self.root_path, self.annotation_type, vid + '.npy'))
            rgb = np.load(osp.join(self.root_path, self.rgb_type, vid + '.npy'))
            flow = np.load(osp.join(self.root_path, self.flow_type, vid + '.npy'))
            if self.training:
                self.target_all[vid] = np.concatenate((dummy_target, target), axis=0)
                self.rgb_inputs[vid] = np.concatenate((dummy_rgb, rgb), axis=0)
                self.flow_inputs[vid] = np.concatenate((dummy_flow, flow), axis=0)
            else:
                self.target_all[vid] = target
                self.rgb_inputs[vid] = rgb
                self.flow_inputs[vid] = flow
        
    def _init_features(self):
        del self.inputs
        gc.collect()
        self.inputs = []

        for vid in self.vids:
            target = self.target_all[vid]
            if self.training:
                seed = np.random.randint(self.stride)
                for start, end in zip(range(seed, target.shape[0], self.stride), 
                    range(seed + self.window_size, target.shape[0]-self.anticipation_length, self.stride)):
                    self.inputs.append([
                        vid, start, end, target[start:end], target[end:end+self.anticipation_length]
                    ])
            else:
                start = 0
                end = target.shape[0] - self.anticipation_length
                ant_target = []
                for s in range(0, target.shape[0]-self.anticipation_length):
                    ant_target.append(target[s:s+self.anticipation_length])

                self.inputs.append([
                    vid, start, end, target[start:end], np.array(ant_target)
                ])
    
    def __getitem__(self, index):
        vid, start, end, target, ant_target = self.inputs[index]
        rgb_input = self.rgb_inputs[vid][start:end]
        flow_input = self.flow_inputs[vid][start:end]
        rgb_input = torch.tensor(rgb_input.astype(np.float32))
        flow_input = torch.tensor(flow_input.astype(np.float32))
        target = torch.tensor(target.astype(np.float32))
        ant_target = torch.tensor(ant_target.astype(np.float32))
        return rgb_input, flow_input, target, ant_target

    def __len__(self):
        return len(self.inputs)

    
@DATA_LAYERS.register("FINEACTION")
class FINEACTIONDataset(data.Dataset):
    
    def __init__(self, cfg, mode='train'):
        self.root_path = cfg['root_path']
        self.mode = mode
        self.training = mode == 'train'
        self.window_size = cfg['window_size']
        self.stride = cfg['stride']
        data_name = cfg['data_name']
        self.vids = json.load(open(cfg['video_list_path']))[data_name][mode + '_session_set'] # list of video names
        self.num_classes = cfg['num_classes']
        self.inputs = []
        self._load_features(cfg)
        self._init_features()

    def _load_features(self, cfg):
        self.annotation_type = cfg['annotation_type']
        self.rgb_type = cfg['rgb_type']
        self.flow_type = cfg['flow_type']
        
    def _init_features(self, seed=0):
        # self.inputs = []
        del self.inputs
        gc.collect()
        self.inputs = []
        for vid in self.vids:
            target = np.load(osp.join(self.root_path, self.annotation_type, vid + '.npy'))
            if self.training:
                seed = np.random.randint(self.stride)
                for start, end in zip(range(seed, target.shape[0], self.stride), 
                    range(seed + self.window_size, target.shape[0]+1, self.stride)):
                    self.inputs.append([
                        vid, start, end
                    ])
            else:
                start = 0
                end = target.shape[0]
                self.inputs.append([
                    vid, start, end
                ])

    def __getitem__(self, index):
        vid, start, end = self.inputs[index]
        rgb_input = np.load(osp.join(self.root_path, self.rgb_type, vid + '.npy'), mmap_mode='r')[start:end]
        flow_input = np.load(osp.join(self.root_path, self.flow_type, vid + '.npy'), mmap_mode='r')[start:end]
        target = np.load(osp.join(self.root_path, self.annotation_type, vid + '.npy'), mmap_mode='r')[start:end]
        rgb_input = torch.tensor(rgb_input.astype(np.float32))
        flow_input = torch.tensor(flow_input.astype(np.float32))
        target = torch.tensor(target.astype(np.float32))
        return rgb_input, flow_input, target

    def __len__(self):
        return len(self.inputs)    