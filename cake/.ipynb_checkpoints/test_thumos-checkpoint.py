import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm
import json
import gc
from cake_baseline import BioX3D_Student

# =============================================================================
# 1. CẤU HÌNH HỆ THỐNG (USER CONFIG)
# =============================================================================

# --- A. INPUT PATHS (CHỈNH SỬA TẠI ĐÂY CHO TỪNG LẦN CHẠY) ---

# Ví dụ: Cấu hình để chạy tập TEST
# Khi muốn chạy tập TRAIN, hãy đổi đường dẫn ở 2 dòng dưới này
VIDEO_DIR = "/workspace/data_nas01/cuongnd36/thumos14/train" 
ANNO_DIR  = "/workspace/data_nas01/cuongnd36/thumos14/annotation_train"

# --- B. MODEL & OUTPUT ---
CHECKPOINT_PATH = "/workspace/data_nas06/cuongnd36/dashcam/cake/base_mse_freeze_no_cls/checkpoint_ep70.pth"
OUTPUT_ROOT     = "/workspace/raid/os_callbot/kiennh/oad/data/cake_5"

# Tên 3 folder output
DIR_RGB    = os.path.join(OUTPUT_ROOT, "rgb_kinetics_x3d")
DIR_FLOW   = os.path.join(OUTPUT_ROOT, "flow_kinetics_x3d")
DIR_TARGET = os.path.join(OUTPUT_ROOT, "target_perframe")

# 3. JSON List Video Path (Dùng để lọc Whitelist)
JSON_PATH = "/workspace/raid/os_callbot/kiennh/oad/MiniROAD/data_info/video_list.json"

# --- C. PARAMETERS ---
TARGET_FPS = 24.0
PROCESS_BATCH_SIZE = 128
CLIP_LEN = 13
STRIDE = 6
CROP_SIZE = 224
SIDE_SIZE = 224

THUMOS_CLASSES = [
    "BaseballPitch", "BasketballDunk", "Billiards", "CleanAndJerk", "CliffDiving",
    "CricketBowling", "CricketShot", "Diving", "FrisbeeCatch", "GolfSwing",
    "HammerThrow", "HighJump", "JavelinThrow", "LongJump", "PoleVault",
    "Shotput", "SoccerPenalty", "TennisSwing", "ThrowDiscus", "VolleyballSpiking", 
    "Ambiguous" 
]
NUM_CLASSES = 22

# =============================================================================
# 2. CHUẨN BỊ (SETUP)
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f">> Device: {device}")

os.makedirs(DIR_RGB, exist_ok=True)
os.makedirs(DIR_FLOW, exist_ok=True)
os.makedirs(DIR_TARGET, exist_ok=True)

# --- Load Whitelist from JSON ---
def load_whitelist(json_path):
    print(f">> Loading whitelist from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'THUMOS' not in data:
        raise ValueError("JSON file does not contain 'THUMOS' key.")
        
    train_list = data['THUMOS'].get('train_session_set', [])
    test_list = data['THUMOS'].get('test_session_set', [])
    
    # Gộp cả train và test vào whitelist để script linh hoạt chạy folder nào cũng được
    whitelist = set(train_list + test_list)
    print(f">> Whitelist loaded: {len(whitelist)} videos allowed (Train + Test combined).")
    return whitelist

def set_odconv_temperature(model, temperature=1.0):
    for m in model.modules():
        if hasattr(m, 'update_temperature'):
            m.update_temperature(temperature)

# --- Load Model ---
def load_model():
    print(f">> Initializing BioX3D_Student...")
    model = BioX3D_Student(clip_len=CLIP_LEN, num_classes=400)
    print(f">> Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
    state = ckpt.get('state_dict', ckpt.get('model_state', ckpt))
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
    set_odconv_temperature(model, 4.56)
    model = model.to(device).eval()
    return model

model = load_model()

# --- Transform ---
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
transform = T.Compose([
    T.Resize(SIDE_SIZE),
    T.CenterCrop(CROP_SIZE),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])

# =============================================================================
# 3. XỬ LÝ GROUND TRUTH (SINGLE FOLDER)
# =============================================================================

def load_ground_truth(anno_dir):
    print(f">> Loading annotations from: {anno_dir}")
    class_to_id = {name: i + 1 for i, name in enumerate(THUMOS_CLASSES)}
    gt_db = {} 
    
    if not os.path.exists(anno_dir):
        print(f"!! Warning: Annotation dir not found: {anno_dir}")
        return gt_db

    txt_files = list(Path(anno_dir).glob("*.txt"))
    for txt in txt_files:
        cname = txt.stem.replace("_val", "").replace("_test", "")
        if cname not in class_to_id: continue
        cid = class_to_id[cname]
        with open(txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                vid, t_s, t_e = parts[0], float(parts[1]), float(parts[2])
                if vid not in gt_db: gt_db[vid] = []
                gt_db[vid].append({'start': t_s, 'end': t_e, 'label': cid})
    
    print(f">> Loaded Annotations for {len(gt_db)} videos.")
    return gt_db

# Load GT cho folder hiện tại
gt_database = load_ground_truth(ANNO_DIR)

# =============================================================================
# 4. ENGINE XỬ LÝ (TRÍCH XUẤT 2 NHÁNH + LABEL)
# =============================================================================

@torch.no_grad()
def process_single_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None, None, None

    target_interval = 1000.0 / TARGET_FPS
    next_target_time = 0.0
    frames_buf = []
    clips_buf = []
    rgb_feats = []
    flow_feats = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        curr_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        if curr_time < next_target_time: continue
        next_target_time += target_interval
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(Image.fromarray(img))
        frames_buf.append(img_tensor)
        
        if len(frames_buf) == CLIP_LEN:
            clip = torch.stack(frames_buf, dim=1)
            clips_buf.append(clip)
            frames_buf = frames_buf[STRIDE:]
            
            if len(clips_buf) == PROCESS_BATCH_SIZE:
                batch = torch.stack(clips_buf).to(device)
                outputs = model(batch, return_embeddings=True)
                rgb_feats.append(outputs[4].cpu())
                flow_feats.append(outputs[5].cpu())
                clips_buf = []

    cap.release()
    
    if len(clips_buf) > 0:
        batch = torch.stack(clips_buf).to(device)
        outputs = model(batch, return_embeddings=True)
        rgb_feats.append(outputs[4].cpu())
        flow_feats.append(outputs[5].cpu())

    if not rgb_feats: return None, None, None

    final_rgb_features  = torch.cat(rgb_feats, dim=0).numpy()
    final_flow_features = torch.cat(flow_feats, dim=0).numpy()
    
    # --- Multi-hot Label Generation ---
    num_clips = final_rgb_features.shape[0]
    final_targets = np.zeros((num_clips, NUM_CLASSES), dtype=np.float32)
    final_targets[:, 0] = 1.0 # Init Background
    
    vid_name = video_path.stem
    anns = gt_database.get(vid_name, [])
    
    for i in range(num_clips):
        center_time = ((i * STRIDE) + (CLIP_LEN // 2)) / TARGET_FPS
        has_action = False
        
        for ann in anns:
            if ann['start'] <= center_time <= ann['end']:
                label_id = ann['label']
                final_targets[i, label_id] = 1.0
                has_action = True
                # No break -> Multi-hot
        
        if has_action:
            final_targets[i, 0] = 0.0
                
    return final_rgb_features, final_flow_features, final_targets

# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

def process_dataset_folder(video_dir, whitelist):
    if not os.path.exists(video_dir):
        print(f"!! Error: Video path not found: {video_dir}")
        return

    # Lấy tất cả video trong folder
    all_files = []
    for ext in ["*.mp4", "*.avi", "*.mkv", "*.webm"]:
        all_files.extend(Path(video_dir).glob(ext))
        
    # --- [FILTER] LỌC VIDEO THEO WHITELIST ---
    # Whitelist chứa cả train và test, nên nó sẽ pass đúng các video thuộc folder hiện tại
    target_files = []
    for f in all_files:
        if f.stem in whitelist:
            target_files.append(f)
            
    print(f"\n>> Processing Folder: {video_dir}")
    print(f">> Found {len(all_files)} files. Matching Whitelist: {len(target_files)} videos.")
    
    for vid_path in tqdm(target_files, desc="Processing"):
        try:
            vname = vid_path.stem
            path_rgb = Path(DIR_RGB) / f"{vname}.npy"
            path_flow = Path(DIR_FLOW) / f"{vname}.npy"
            path_targ = Path(DIR_TARGET) / f"{vname}.npy"
            
            # Resume Check
            if path_rgb.exists() and path_flow.exists() and path_targ.exists():
                continue
            
            rgb_np, flow_np, targ_np = process_single_video(vid_path)
            
            if rgb_np is not None:
                np.save(path_rgb, rgb_np)
                np.save(path_flow, flow_np)
                np.save(path_targ, targ_np)
                
        except Exception as e:
            print(f"Error {vid_path.name}: {e}")
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                gc.collect()

def main():
    # 1. Load Whitelist (Chứa danh sách cả train và test để lọc nhiễu)
    whitelist = load_whitelist(JSON_PATH)
    
    print("==================================================")
    print("   BIO-X3D PROCESSING (SINGLE FOLDER)")
    print("==================================================")
    print(f"Video Input: {VIDEO_DIR}")
    print(f"Label Input: {ANNO_DIR}")
    print("==================================================")
    
    # 2. Process
    process_dataset_folder(VIDEO_DIR, whitelist)
    
    print("\n==================================================")
    print("COMPLETED!")

if __name__ == "__main__":
    main()

# import torch
# import torch.nn as nn
# import cv2
# import numpy as np
# import torchvision.transforms as T
# from PIL import Image
# from pathlib import Path
# import os
# from tqdm import tqdm
# import json
# import gc
# from cake_baseline import BioX3D_Student

# # =============================================================================
# # 1. CẤU HÌNH HỆ THỐNG (USER CONFIG)
# # =============================================================================

# # --- A. INPUT PATHS (TRAIN & TEST) ---
# # 1. Video Dirs
# TRAIN_VIDEO_DIR = "/workspace/data_nas01/cuongnd36/thumos14/train"
# TEST_VIDEO_DIR  = "/workspace/data_nas01/cuongnd36/thumos14/test"

# # 2. Annotation Dirs
# TRAIN_ANNO_DIR = "/workspace/data_nas01/cuongnd36/thumos14/annotation_train"
# TEST_ANNO_DIR  = "/workspace/data_nas01/cuongnd36/thumos14/annotation_test"

# # --- B. MODEL & OUTPUT ---
# CHECKPOINT_PATH = "/workspace/data_nas06/cuongnd36/dashcam/cake/base_mse_freeze_no_cls/checkpoint_ep70.pth"
# OUTPUT_ROOT     = "/workspace/raid/os_callbot/kiennh/oad/data/cake_5"

# # Tên 3 folder output theo yêu cầu
# DIR_RGB    = os.path.join(OUTPUT_ROOT, "rgb_kinetics_x3d")
# DIR_FLOW   = os.path.join(OUTPUT_ROOT, "flow_kinetics_x3d")
# DIR_TARGET = os.path.join(OUTPUT_ROOT, "target_perframe")

# # 3. [NEW] JSON List Video Path
# JSON_PATH = "/workspace/raid/os_callbot/kiennh/oad/MiniROAD/data_info/video_list.json"  # <--- CẬP NHẬT ĐƯỜNG DẪN NÀY

# # --- C. PARAMETERS ---
# TARGET_FPS = 24.0
# PROCESS_BATCH_SIZE = 128
# CLIP_LEN = 13
# STRIDE = 6
# CROP_SIZE = 224
# SIDE_SIZE = 224

# THUMOS_CLASSES = [
#     "BaseballPitch", "BasketballDunk", "Billiards", "CleanAndJerk", "CliffDiving",
#     "CricketBowling", "CricketShot", "Diving", "FrisbeeCatch", "GolfSwing",
#     "HammerThrow", "HighJump", "JavelinThrow", "LongJump", "PoleVault",
#     "Shotput", "SoccerPenalty", "TennisSwing", "ThrowDiscus", "VolleyballSpiking", 
#     "Ambiguous" 
# ]
# NUM_CLASSES = 22

# # =============================================================================
# # 2. CHUẨN BỊ (SETUP)
# # =============================================================================

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f">> Device: {device}")

# os.makedirs(DIR_RGB, exist_ok=True)
# os.makedirs(DIR_FLOW, exist_ok=True)
# os.makedirs(DIR_TARGET, exist_ok=True)

# # --- [NEW] Load Whitelist from JSON ---
# def load_whitelist(json_path):
#     print(f">> Loading whitelist from {json_path}...")
#     with open(json_path, 'r') as f:
#         data = json.load(f)
    
#     # Lấy list từ key THUMOS
#     if 'THUMOS' not in data:
#         raise ValueError("JSON file does not contain 'THUMOS' key.")
        
#     train_list = data['THUMOS'].get('train_session_set', [])
#     test_list = data['THUMOS'].get('test_session_set', [])
    
#     # Gộp lại thành 1 set để tra cứu nhanh
#     whitelist = set(train_list + test_list)
#     print(f">> Whitelist loaded: {len(whitelist)} videos allowed.")
#     return whitelist

# def set_odconv_temperature(model, temperature=1.0):
#     for m in model.modules():
#         if hasattr(m, 'update_temperature'):
#             m.update_temperature(temperature)

# # --- Load Model ---
# def load_model():
#     print(f">> Initializing BioX3D_Student...")
#     model = BioX3D_Student(clip_len=CLIP_LEN, num_classes=400)
#     ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
#     state = ckpt.get('state_dict', ckpt.get('model_state', ckpt))
#     model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
#     set_odconv_temperature(model, 4.56)
#     model = model.to(device).eval()
#     return model

# model = load_model()

# # --- Transform ---
# mean = [0.45, 0.45, 0.45]
# std = [0.225, 0.225, 0.225]
# transform = T.Compose([
#     T.Resize(SIDE_SIZE),
#     T.CenterCrop(CROP_SIZE),
#     T.ToTensor(),
#     T.Normalize(mean=mean, std=std),
# ])

# # =============================================================================
# # 3. XỬ LÝ GROUND TRUTH
# # =============================================================================

# def load_all_ground_truth(train_anno_dir, test_anno_dir):
#     print(f">> Loading annotations...")
#     class_to_id = {name: i + 1 for i, name in enumerate(THUMOS_CLASSES)}
#     gt_db = {} 
#     dirs = [train_anno_dir, test_anno_dir]
    
#     for d in dirs:
#         if not os.path.exists(d): continue
#         txt_files = list(Path(d).glob("*.txt"))
#         for txt in txt_files:
#             cname = txt.stem.replace("_val", "").replace("_test", "")
#             if cname not in class_to_id: continue
#             cid = class_to_id[cname]
#             with open(txt, 'r') as f:
#                 for line in f:
#                     parts = line.strip().split()
#                     if len(parts) < 3: continue
#                     vid, t_s, t_e = parts[0], float(parts[1]), float(parts[2])
#                     if vid not in gt_db: gt_db[vid] = []
#                     gt_db[vid].append({'start': t_s, 'end': t_e, 'label': cid})
#     return gt_db

# gt_database = load_all_ground_truth(TRAIN_ANNO_DIR, TEST_ANNO_DIR)

# # =============================================================================
# # 4. ENGINE XỬ LÝ (TRÍCH XUẤT 2 NHÁNH + LABEL)
# # =============================================================================

# @torch.no_grad()
# def process_single_video(video_path):
#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened(): return None, None, None

#     target_interval = 1000.0 / TARGET_FPS
#     next_target_time = 0.0
#     frames_buf = []
#     clips_buf = []
#     rgb_feats = []
#     flow_feats = []
    
#     while True:
#         ret, frame = cap.read()
#         if not ret: break
#         curr_time = cap.get(cv2.CAP_PROP_POS_MSEC)
#         if curr_time < next_target_time: continue
#         next_target_time += target_interval
        
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img_tensor = transform(Image.fromarray(img))
#         frames_buf.append(img_tensor)
        
#         if len(frames_buf) == CLIP_LEN:
#             clip = torch.stack(frames_buf, dim=1)
#             clips_buf.append(clip)
#             frames_buf = frames_buf[STRIDE:]
            
#             if len(clips_buf) == PROCESS_BATCH_SIZE:
#                 batch = torch.stack(clips_buf).to(device)
#                 outputs = model(batch, return_embeddings=True)
#                 rgb_feats.append(outputs[4].cpu())
#                 flow_feats.append(outputs[5].cpu())
#                 clips_buf = []

#     cap.release()
    
#     if len(clips_buf) > 0:
#         batch = torch.stack(clips_buf).to(device)
#         outputs = model(batch, return_embeddings=True)
#         rgb_feats.append(outputs[4].cpu())
#         flow_feats.append(outputs[5].cpu())

#     if not rgb_feats: return None, None, None

#     final_rgb_features  = torch.cat(rgb_feats, dim=0).numpy()
#     final_flow_features = torch.cat(flow_feats, dim=0).numpy()
    
#     # --- Multi-hot Label Generation ---
#     num_clips = final_rgb_features.shape[0]
#     final_targets = np.zeros((num_clips, NUM_CLASSES), dtype=np.float32)
#     final_targets[:, 0] = 1.0 # Init Background
    
#     vid_name = video_path.stem
#     anns = gt_database.get(vid_name, [])
    
#     for i in range(num_clips):
#         center_time = ((i * STRIDE) + (CLIP_LEN // 2)) / TARGET_FPS
#         has_action = False
        
#         for ann in anns:
#             if ann['start'] <= center_time <= ann['end']:
#                 label_id = ann['label']
#                 final_targets[i, label_id] = 1.0
#                 has_action = True
#                 # No break -> Multi-hot
        
#         if has_action:
#             final_targets[i, 0] = 0.0
                
#     return final_rgb_features, final_flow_features, final_targets

# # =============================================================================
# # 5. MAIN EXECUTION
# # =============================================================================

# def process_dataset_folder(video_dir, whitelist, desc_name):
#     if not os.path.exists(video_dir):
#         print(f"Skipping {desc_name} (Path not found): {video_dir}")
#         return

#     # Lấy tất cả video trong folder
#     all_files = []
#     for ext in ["*.mp4", "*.avi", "*.mkv", "*.webm"]:
#         all_files.extend(Path(video_dir).glob(ext))
        
#     # --- [FILTER] LỌC VIDEO THEO WHITELIST ---
#     target_files = []
#     for f in all_files:
#         if f.stem in whitelist:
#             target_files.append(f)
            
#     print(f"\n>> Processing {desc_name}: {len(target_files)} videos (Filtered from {len(all_files)})")
    
#     for vid_path in tqdm(target_files, desc=desc_name):
#         try:
#             vname = vid_path.stem
#             path_rgb = Path(DIR_RGB) / f"{vname}.npy"
#             path_flow = Path(DIR_FLOW) / f"{vname}.npy"
#             path_targ = Path(DIR_TARGET) / f"{vname}.npy"
            
#             if path_rgb.exists() and path_flow.exists() and path_targ.exists():
#                 continue
            
#             rgb_np, flow_np, targ_np = process_single_video(vid_path)
            
#             if rgb_np is not None:
#                 np.save(path_rgb, rgb_np)
#                 np.save(path_flow, flow_np)
#                 np.save(path_targ, targ_np)
                
#         except Exception as e:
#             print(f"Error {vid_path.name}: {e}")
#             if "out of memory" in str(e):
#                 torch.cuda.empty_cache()
#                 gc.collect()

# def main():
#     # 1. Load Whitelist
#     whitelist = load_whitelist(JSON_PATH)
    
#     print("==================================================")
#     print("   BIO-X3D DUAL-BRANCH (FILTERED BY JSON)")
#     print("==================================================")
    
#     # 2. Process with Filter
#     process_dataset_folder(TRAIN_VIDEO_DIR, whitelist, "TRAIN SET")
#     process_dataset_folder(TEST_VIDEO_DIR, whitelist, "TEST SET")
    
#     print("\n==================================================")
#     print("COMPLETED!")

# if __name__ == "__main__":
#     main()