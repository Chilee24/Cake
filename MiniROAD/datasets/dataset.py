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


import torch.utils.data as data
import torch
import numpy as np
import random
import json
import os.path as osp
from datasets import DATA_LAYERS

# Đảm bảo FEATURE_SIZES đã được import
# from config import FEATURE_SIZES 

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
        self.vids = json.load(open(cfg['video_list_path']))[data_name][mode + '_session_set']
        
        # 1. LOAD DATA
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
        
        # Lấy feature dim (Giả sử bạn đã có biến global FEATURE_SIZES hoặc hardcode)
        rgb_dim = 2048 
        flow_dim = 1024
        
        dummy_target = np.zeros((self.window_size - 1, self.num_classes))
        dummy_rgb = np.zeros((self.window_size - 1, rgb_dim))
        dummy_flow = np.zeros((self.window_size - 1, flow_dim))
        
        print(f"--> [Dataset] Pre-loading features into RAM ({self.mode})...")
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

    def shuffle_indices(self):
        self.inputs = []
        print(f"--> [Dataset] Reshuffling grid with stride={self.stride}...")
        
        for vid in self.vids:
            target = self.target_all[vid]
            seed = np.random.randint(self.stride)
            
            # Sliding Window
            for start in range(seed, target.shape[0] - self.window_size + 1, self.stride):
                end = start + self.window_size
                
                # Lấy vector nhãn frame cuối để xác định label chính
                last_frame_vec = target[end - 1] 
                label_idx = np.argmax(last_frame_vec)
                
                self.inputs.append([vid, start, end, label_idx])
                
        random.shuffle(self.inputs)

    def _init_test_indices(self):
        self.inputs = []
        for vid in self.vids:
            target = self.target_all[vid]
            for w_start in range(0, target.shape[0] - self.window_size + 1, 1):
                 w_end = w_start + self.window_size
                 if self.training:
                     last_frame_vec = target[w_end - 1]
                 else:
                     last_frame_vec = target[w_end - 1]
                 label_idx = np.argmax(last_frame_vec)
                 self.inputs.append([vid, w_start, w_end, label_idx])

    def __getitem__(self, index):
        vid, start, end, label_idx = self.inputs[index]
        
        # 1. GET DATA
        rgb_input = self.rgb_inputs[vid][start:end]
        flow_input = self.flow_inputs[vid][start:end]
        target_window = self.target_all[vid][start:end] 
        
        # 2. CONVERT TENSOR
        rgb_tensor = torch.tensor(rgb_input.astype(np.float32))
        flow_tensor = torch.tensor(flow_input.astype(np.float32))

        # 3. LẤY NHÃN TỪNG FRAME
        labels_per_frame = torch.tensor(np.argmax(target_window, axis=1), dtype=torch.long)

        # 4. TẠO SHUFFLE CÓ ĐIỀU KIỆN (Conditional Shuffling)
        rgb_shuff = rgb_tensor.clone()
        flow_shuff = flow_tensor.clone()
        
        if self.training:
            T = self.window_size
            
            # Kiểm tra 2 frame cuối cùng
            # labels_per_frame có shape [T]. Index cuối là -1, kế cuối là -2.
            last_frame_is_action = (labels_per_frame[-1] != self.bg_class_idx)
            second_last_is_action = (labels_per_frame[-2] != self.bg_class_idx)
            
            # ĐIỀU KIỆN: Chỉ shuffle khi CẢ 2 frame cuối đều là Action
            if last_frame_is_action and second_last_is_action:
                
                # Tạo hoán vị cho phần lịch sử (0 đến T-2)
                # T-1 (Frame cuối) GIỮ NGUYÊN
                perm_indices = torch.randperm(T - 1) 
                
                rgb_shuff[:-1] = rgb_tensor[perm_indices]
                flow_shuff[:-1] = flow_tensor[perm_indices]
            
            # Nếu không thỏa mãn điều kiện -> Giữ nguyên rgb_shuff y hệt bản gốc
            # (Lúc này q_shuff sẽ giống q_masked một phần, nhưng model sẽ tự học cách ignore)

        return {
            'rgb_anchor': rgb_tensor,
            'flow_anchor': flow_tensor,
            'rgb_shuff': rgb_shuff,
            'flow_shuff': flow_shuff,
            'labels': torch.tensor(label_idx, dtype=torch.long),
            'labels_per_frame': labels_per_frame
        }

    def __len__(self):
        return len(self.inputs)

#@DATA_LAYERS.register("THUMOS")
@DATA_LAYERS.register("TVSERIES")
class THUMOSDataset(data.Dataset):
    
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
            # concatting dummy target at the front 
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
                    range(seed + self.window_size, target.shape[0]+1, self.stride)):
                    self.inputs.append([
                        vid, start, end, target[start:end]
                    ])
            else:
                start = 0
                end = target.shape[0]
                self.inputs.append([
                    vid, start, end, target[start:end]
                ])

    def __getitem__(self, index):
        vid, start, end, target = self.inputs[index]
        rgb_input = self.rgb_inputs[vid][start:end]
        flow_input = self.flow_inputs[vid][start:end]
        rgb_input = torch.tensor(rgb_input.astype(np.float32))
        flow_input = torch.tensor(flow_input.astype(np.float32))
        target = torch.tensor(target.astype(np.float32))
        return rgb_input, flow_input, target

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