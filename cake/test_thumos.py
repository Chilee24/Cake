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

# --- NEW IMPORTS FOR RAFT & X3D ---
from torchvision.models.optical_flow import raft_large
import pytorchvideo.models.x3d as x3d

# =============================================================================
# 1. CẤU HÌNH HỆ THỐNG (USER CONFIG)
# =============================================================================

# --- A. INPUT PATHS ---
TRAIN_VIDEO_DIR = "/workspace/data_nas01/cuongnd36/thumos14/train"
TEST_VIDEO_DIR  = "/workspace/data_nas01/cuongnd36/thumos14/test"

TRAIN_ANNO_DIR = "/workspace/data_nas01/cuongnd36/thumos14/annotation_train"
TEST_ANNO_DIR  = "/workspace/data_nas01/cuongnd36/thumos14/annotation_test"

# --- B. MODEL WEIGHTS ---
# Đường dẫn file weight RAFT (thường có sẵn trong torchvision hoặc tải về)
RAFT_WEIGHTS_PATH = "/workspace/data_nas06/cuongnd36/dashcam/train_teacher/checkpoint/raft_large_C_T_SKHT_V2-ff5fadd5.pth" 
# Đường dẫn file weight X3D được train trên Optical Flow
X3D_FLOW_WEIGHTS_PATH = "/workspace/data_nas06/cuongnd36/dashcam/checkpoints_x3d_of/best_x3d_flow.pth" 

# --- C. OUTPUT ---
OUTPUT_ROOT = "/workspace/raid/os_callbot/kiennh/oad/data/cake_5/real_flow_features"

DIR_FLOW   = os.path.join(OUTPUT_ROOT, "flow_kinetics_x3d")
DIR_TARGET = os.path.join(OUTPUT_ROOT, "target_perframe")

# JSON List Path
JSON_PATH = "/workspace/raid/os_callbot/kiennh/oad/MiniROAD/data_info/video_list.json"

# --- D. PARAMETERS ---
TARGET_FPS = 24.0
# Lưu ý: RAFT rất nặng VRAM, cần giảm Batch Size xuống thấp (ví dụ 1-4 clip)
PROCESS_BATCH_SIZE = 16
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
# 2. CHUẨN BỊ MODEL (RAFT + X3D)
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f">> Device: {device}")

# Tạo thư mục
os.makedirs(DIR_FLOW, exist_ok=True)
#os.makedirs(DIR_TARGET, exist_ok=True)

# --- Helper: Chạy RAFT theo chunks để tránh OOM ---
def run_raft_in_chunks(raft_model, img1, img2, chunk_size=16):
    total_frames = img1.shape[0]
    flow_list = []
    with torch.no_grad():
        for i in range(0, total_frames, chunk_size):
            i1 = img1[i : i + chunk_size]
            i2 = img2[i : i + chunk_size]
            
            # RAFT expects [-1, 1] input
            i1 = (i1 / 255.0) * 2.0 - 1.0
            i2 = (i2 / 255.0) * 2.0 - 1.0
            
            flow_preds = raft_model(i1, i2)
            flow_list.append(flow_preds[-1]) # Lấy kết quả cuối cùng
    return torch.cat(flow_list, dim=0)

# --- Class Pipeline Trích xuất Flow ---
class RealFlowExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        print(">> Initializing RAFT + X3D Flow Pipeline...")
        
        # 1. Load RAFT
        self.raft = raft_large(weights=None).to(device)
        if os.path.exists(RAFT_WEIGHTS_PATH):
            print(f"   Loading RAFT weights: {RAFT_WEIGHTS_PATH}")
            self.raft.load_state_dict(torch.load(RAFT_WEIGHTS_PATH, map_location=device))
        else:
            print("!! WARNING: RAFT weights not found. Using random init.")
        self.raft.eval()
        
        # 2. Load X3D (Input Channel = 2)
        self.x3d = x3d.create_x3d(
            input_channel=2, 
            input_clip_length=CLIP_LEN, 
            model_num_class=400
        ).to(device)
        
        if os.path.exists(X3D_FLOW_WEIGHTS_PATH):
            print(f"   Loading X3D Flow weights: {X3D_FLOW_WEIGHTS_PATH}")
            ckpt = torch.load(X3D_FLOW_WEIGHTS_PATH, map_location=device)
            state = ckpt.get('state_dict', ckpt.get('model_state', ckpt))
            clean_state = {k.replace('module.', ''): v for k, v in state.items()}
            self.x3d.load_state_dict(clean_state, strict=False)
        else:
            print("!! WARNING: X3D Flow weights not found.")
            
        self.x3d.eval()
        
        # Tách Backbone và Head để lấy embedding
        modules = list(self.x3d.blocks.children())
        self.backbone = nn.Sequential(*modules[:-1])
        self.head = modules[-1]

    @torch.no_grad()
    def forward(self, rgb_batch_0_255):
        """
        Input: [B, 3, T, H, W] - Giá trị pixel 0-255 (chưa normalize mean/std)
        Output: [B, 2048] - Feature vector
        """
        B, C, T, H, W = rgb_batch_0_255.shape
        
        # A. Chuẩn bị input cho RAFT (Flatten Batch và Time)
        # Lấy cặp frame liên tiếp: (t) và (t+1)
        img1 = rgb_batch_0_255[:, :, :-1, :, :].permute(0, 2, 1, 3, 4).reshape(-1, 3, H, W)
        img2 = rgb_batch_0_255[:, :, 1:, :, :].permute(0, 2, 1, 3, 4).reshape(-1, 3, H, W)
        
        # B. Tính Flow (ra shape [-1, 2, H, W])
        flow_flat = run_raft_in_chunks(self.raft, img1, img2, chunk_size=16)
        
        # C. Reshape lại thành Video Clip [B, 2, T-1, H, W]
        flow_clip = flow_flat.view(B, T-1, 2, H, W).permute(0, 2, 1, 3, 4)
        
        # D. Padding frame cuối (để đủ độ dài T=13)
        last_flow = flow_clip[:, :, -1:, :, :]
        flow_input = torch.cat([flow_clip, last_flow], dim=2) # [B, 2, 13, H, W]
        
        # E. Chuẩn hóa Flow cho X3D (Thường kẹp trong khoảng [-20, 20] rồi về [-1, 1])
        flow_input = torch.clamp(flow_input / 20.0, -1.0, 1.0)
        
        # F. X3D Feature Extraction
        feat_map = self.backbone(flow_input) # [B, 192, T, 7, 7]
        vec = self.head.pool(feat_map)
        vec = self.head.output_pool(vec)
        vec = vec.flatten(1) # [B, 2048]
        
        return vec

# Transform: Resize only (Không Normalize Mean/Std vì RAFT tự xử lý)
transform = T.Compose([
    T.Resize(SIDE_SIZE),
    T.CenterCrop(CROP_SIZE),
])

# =============================================================================
# 3. UTILS (Whitelist & Ground Truth)
# =============================================================================

def load_whitelist(json_path):
    print(f">> Loading whitelist from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    if 'THUMOS' not in data:
        raise ValueError("JSON file missing 'THUMOS' key.")
    whitelist = set(data['THUMOS'].get('train_session_set', []) + 
                    data['THUMOS'].get('test_session_set', []))
    print(f">> Whitelist: {len(whitelist)} videos.")
    return whitelist

def load_all_ground_truth(train_dir, test_dir):
    print(f">> Loading annotations...")
    class_to_id = {name: i + 1 for i, name in enumerate(THUMOS_CLASSES)}
    gt_db = {} 
    for d in [train_dir, test_dir]:
        if not os.path.exists(d): continue
        for txt in list(Path(d).glob("*.txt")):
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
    return gt_db

# Init Global Objects
model = RealFlowExtractor()
gt_database = load_all_ground_truth(TRAIN_ANNO_DIR, TEST_ANNO_DIR)

# =============================================================================
# 4. ENGINE XỬ LÝ (SINGLE VIDEO)
# =============================================================================

@torch.no_grad()
def process_single_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None, None

    target_interval = 1000.0 / TARGET_FPS
    next_target_time = 0.0
    
    frames_buf = []
    clips_buf = []
    flow_feats = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        curr_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        if curr_time < next_target_time: continue
        next_target_time += target_interval
        
        # Đọc RGB (giữ nguyên giá trị pixel để đưa vào pipeline)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Transform: Resize/Crop, convert to Tensor but KEEP RANGE 0-255 (float)
        pil_img = Image.fromarray(img)
        # Resize/Crop
        img_cropped = transform(pil_img) 
        # To Tensor (0-1) sau đó nhân 255 để về lại range pixel cho RAFT xử lý logic
        img_tensor = T.ToTensor()(img_cropped) * 255.0 
        
        frames_buf.append(img_tensor)
        
        if len(frames_buf) == CLIP_LEN:
            clip = torch.stack(frames_buf, dim=1) # [3, T, H, W]
            clips_buf.append(clip)
            frames_buf = frames_buf[STRIDE:]
            
            if len(clips_buf) == PROCESS_BATCH_SIZE:
                batch = torch.stack(clips_buf).to(device) # [B, 3, T, H, W]
                # Extract Flow Feature
                feats = model(batch)
                flow_feats.append(feats.cpu())
                clips_buf = []

    cap.release()
    
    if len(clips_buf) > 0:
        batch = torch.stack(clips_buf).to(device)
        feats = model(batch)
        flow_feats.append(feats.cpu())

    if not flow_feats: return None, None

    final_flow_features = torch.cat(flow_feats, dim=0).numpy()
    
    # --- Label Generation ---
    num_clips = final_flow_features.shape[0]
    final_targets = np.zeros((num_clips, NUM_CLASSES), dtype=np.float32)
    final_targets[:, 0] = 1.0 # Init Background
    
    vid_name = video_path.stem
    anns = gt_database.get(vid_name, [])
    
    for i in range(num_clips):
        center_time = ((i * STRIDE) + (CLIP_LEN // 2)) / TARGET_FPS
        has_action = False
        for ann in anns:
            if ann['start'] <= center_time <= ann['end']:
                final_targets[i, ann['label']] = 1.0
                has_action = True
        if has_action: final_targets[i, 0] = 0.0
                
    return final_flow_features, final_targets

# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

def process_dataset_folder(video_dir, whitelist, desc_name):
    if not os.path.exists(video_dir):
        print(f"Skipping {desc_name} (Path not found): {video_dir}")
        return

    all_files = []
    for ext in ["*.mp4", "*.avi", "*.mkv", "*.webm"]:
        all_files.extend(Path(video_dir).glob(ext))
        
    target_files = [f for f in all_files if f.stem in whitelist]
    print(f"\n>> Processing {desc_name}: {len(target_files)} videos")
    
    skipped_count = 0
    processed_count = 0
    error_count = 0
    
    for vid_path in tqdm(target_files, desc=desc_name):
        try:
            vname = vid_path.stem
            path_flow = Path(DIR_FLOW) / f"{vname}.npy"
            #path_targ = Path(DIR_TARGET) / f"{vname}.npy"
            
            # --- SKIP IF EXISTS (Chỉ cần có đủ Flow và Target) ---
            if path_flow.exists(): #and path_targ.exists():
                skipped_count += 1
                continue
            
            flow_np, targ_np = process_single_video(vid_path)
            
            if flow_np is not None:
                np.save(path_flow, flow_np)
                #np.save(path_targ, targ_np)
                processed_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            print(f"Error {vid_path.name}: {e}")
            error_count += 1
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                gc.collect()

    print(f"   [Report] Processed: {processed_count} | Skipped: {skipped_count} | Errors: {error_count}")

def main():
    whitelist = load_whitelist(JSON_PATH)
    
    print("==================================================")
    print("   REAL FLOW EXTRACTION (RAFT + X3D)")
    print("==================================================")
    
    process_dataset_folder(TRAIN_VIDEO_DIR, whitelist, "TRAIN SET")
    process_dataset_folder(TEST_VIDEO_DIR, whitelist, "TEST SET")
    
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

# # Import model của bạn
# from cake_baseline import BioX3D_Student

# # =============================================================================
# # 1. CẤU HÌNH HỆ THỐNG (USER CONFIG)
# # =============================================================================

# # --- A. INPUT PATHS ---
# TRAIN_VIDEO_DIR = "/workspace/data_nas01/cuongnd36/thumos14/train"
# TEST_VIDEO_DIR  = "/workspace/data_nas01/cuongnd36/thumos14/test"

# TRAIN_ANNO_DIR = "/workspace/data_nas01/cuongnd36/thumos14/annotation_train"
# TEST_ANNO_DIR  = "/workspace/data_nas01/cuongnd36/thumos14/annotation_test"

# # --- B. MODEL & OUTPUT ---
# CHECKPOINT_PATH = "/workspace/data_nas06/cuongnd36/dashcam/cake/base_mse_freeze_no_cls/checkpoint_ep70.pth"
# OUTPUT_ROOT     = "/workspace/raid/os_callbot/kiennh/oad/data/cake_5"

# # Tên 3 folder output
# DIR_RGB    = os.path.join(OUTPUT_ROOT, "rgb_kinetics_x3d")
# DIR_FLOW   = os.path.join(OUTPUT_ROOT, "flow_kinetics_x3d")
# DIR_TARGET = os.path.join(OUTPUT_ROOT, "target_perframe")

# # JSON List Path
# JSON_PATH = "/workspace/raid/os_callbot/kiennh/oad/MiniROAD/data_info/video_list.json"

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

# # Tạo thư mục nếu chưa có
# os.makedirs(DIR_RGB, exist_ok=True)
# os.makedirs(DIR_FLOW, exist_ok=True)
# os.makedirs(DIR_TARGET, exist_ok=True)

# def load_whitelist(json_path):
#     print(f">> Loading whitelist from {json_path}...")
#     with open(json_path, 'r') as f:
#         data = json.load(f)
    
#     if 'THUMOS' not in data:
#         raise ValueError("JSON file does not contain 'THUMOS' key.")
        
#     train_list = data['THUMOS'].get('train_session_set', [])
#     test_list = data['THUMOS'].get('test_session_set', [])
    
#     whitelist = set(train_list + test_list)
#     print(f">> Whitelist loaded: {len(whitelist)} videos allowed.")
#     return whitelist

# def set_odconv_temperature(model, temperature=1.0):
#     for m in model.modules():
#         if hasattr(m, 'update_temperature'):
#             m.update_temperature(temperature)

# def load_model():
#     print(f">> Initializing BioX3D_Student...")
#     model = BioX3D_Student(clip_len=CLIP_LEN, num_classes=400)
    
#     print(f">> Loading checkpoint: {CHECKPOINT_PATH}")
#     ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
    
#     # Xử lý state dict linh hoạt
#     state = ckpt.get('state_dict', ckpt.get('model_state', ckpt))
#     # Loại bỏ prefix 'module.' nếu có (do DataParallel)
#     clean_state = {k.replace('module.', ''): v for k, v in state.items()}
    
#     model.load_state_dict(clean_state, strict=False)
    
#     set_odconv_temperature(model, 4.56)
#     model = model.to(device).eval()
#     return model

# model = load_model()

# transform = T.Compose([
#     T.Resize(SIDE_SIZE),
#     T.CenterCrop(CROP_SIZE),
#     T.ToTensor(),
#     T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
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
# # 4. ENGINE XỬ LÝ
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
#                 # Output flag: return_embeddings=True
#                 outputs = model(batch, return_embeddings=True)
#                 # Lấy output index 4 (rgb) và 5 (flow)
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
#                 final_targets[i, ann['label']] = 1.0
#                 has_action = True
        
#         if has_action: final_targets[i, 0] = 0.0
                
#     return final_rgb_features, final_flow_features, final_targets

# # =============================================================================
# # 5. MAIN EXECUTION
# # =============================================================================

# def process_dataset_folder(video_dir, whitelist, desc_name):
#     if not os.path.exists(video_dir):
#         print(f"Skipping {desc_name} (Path not found): {video_dir}")
#         return

#     all_files = []
#     for ext in ["*.mp4", "*.avi", "*.mkv", "*.webm"]:
#         all_files.extend(Path(video_dir).glob(ext))
        
#     # Filter Whitelist
#     target_files = [f for f in all_files if f.stem in whitelist]
            
#     print(f"\n>> Processing {desc_name}: {len(target_files)} videos (Filtered from {len(all_files)})")
    
#     # Counter
#     skipped_count = 0
#     processed_count = 0
#     error_count = 0
    
#     for vid_path in tqdm(target_files, desc=desc_name):
#         try:
#             vname = vid_path.stem
#             path_rgb = Path(DIR_RGB) / f"{vname}.npy"
#             path_flow = Path(DIR_FLOW) / f"{vname}.npy"
#             path_targ = Path(DIR_TARGET) / f"{vname}.npy"
            
#             # --- [CHECK LOGIC] SKIP IF EXISTS ---
#             # Chỉ skip nếu CẢ 3 file đều đã tồn tại
#             if path_rgb.exists() and path_flow.exists() and path_targ.exists():
#                 skipped_count += 1
#                 continue
            
#             # Nếu chưa đủ file -> Chạy lại
#             rgb_np, flow_np, targ_np = process_single_video(vid_path)
            
#             if rgb_np is not None:
#                 np.save(path_rgb, rgb_np)
#                 np.save(path_flow, flow_np)
#                 np.save(path_targ, targ_np)
#                 processed_count += 1
#             else:
#                 error_count += 1
                
#         except Exception as e:
#             print(f"Error {vid_path.name}: {e}")
#             error_count += 1
#             if "out of memory" in str(e):
#                 torch.cuda.empty_cache()
#                 gc.collect()

#     print(f"   [Report {desc_name}] Processed: {processed_count} | Skipped (Exist): {skipped_count} | Errors: {error_count}")

# def main():
#     whitelist = load_whitelist(JSON_PATH)
    
#     print("==================================================")
#     print("   BIO-X3D DUAL-BRANCH (RESUME MODE)")
#     print("==================================================")
    
#     process_dataset_folder(TRAIN_VIDEO_DIR, whitelist, "TRAIN SET")
#     process_dataset_folder(TEST_VIDEO_DIR, whitelist, "TEST SET")
    
#     print("\n==================================================")
#     print("COMPLETED!")

# if __name__ == "__main__":
#     main()