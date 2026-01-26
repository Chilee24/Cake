import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
import time
import yaml
from collections import deque
from tqdm import tqdm
import threading
from queue import Queue
from torch.amp import autocast # Dùng cho Tối ưu FP16

# --- IMPORT TỪ PROJECT CỦA BẠN ---
import sys
sys.path.append(r'D:\project\DashCam\model\cake')
from cake.cake import BioX3D_Student
from MiniROAD.model import build_model

# ===============================
# TIỀN XỬ LÝ ẢNH TRÊN GPU
# ===============================
class X3D_Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1)

    def forward(self, x):
        return (x / 255.0 - self.mean.to(x.device)) / self.std.to(x.device)

# ===============================
# HÀM LOAD MODEL TỪ YAML CONFIG
# ===============================
def load_pytorch_models(biox3d_path, miniroad_path, config_path, device):
    print(f"🔧 Đang load cấu hình từ: {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # 1. Load BioX3D
    biox3d = BioX3D_Student(clip_len=cfg.get('clip_len', 13), num_classes=400).to(device)
    biox_state = torch.load(biox3d_path, map_location=device, weights_only=False).get('state_dict', {})
    biox3d.load_state_dict({k.replace('module.', ''): v for k, v in biox_state.items()}, strict=False)
    biox3d.eval()

    # 2. Load MiniROAD
    miniroad = build_model(cfg, device)
    miniroad.load_state_dict(torch.load(miniroad_path, map_location=device, weights_only=True))
    miniroad.eval()

    return biox3d, miniroad, cfg

# ==========================================
# CÁC LUỒNG ĐỌC/GHI VIDEO (CPU)
# ==========================================
def frame_reader_thread(video_path, input_queue):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        input_queue.put(frame)
    input_queue.put(None)
    cap.release()

def frame_writer_thread(output_path, fps, width, height, output_queue):
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # --- MỚI THÊM: Cấu hình cửa sổ hiển thị ---
    window_name = "DashCam Real-time Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Cho phép kéo giãn kích thước cửa sổ
    cv2.resizeWindow(window_name, 1280, 720) # Đặt kích thước mặc định vừa mắt
    
    show_video = True # Cờ kiểm soát việc hiển thị

    while True:
        frame = output_queue.get()
        if frame is None: break
        
        # 1. Ghi vào file MP4
        out.write(frame)

        # 2. Hiển thị lên màn hình (nếu chưa bị tắt)
        if show_video:
            cv2.imshow(window_name, frame)
            
            # Đợi 1ms để load frame. Nếu nhấn 'q', cửa sổ sẽ tắt nhưng code vẫn chạy ngầm
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️ Đã tắt cửa sổ hiển thị. Quá trình xử lý vẫn đang tiếp tục ngầm...")
                cv2.destroyWindow(window_name)
                show_video = False 

    out.release()
    cv2.destroyAllWindows()

# ===============================
# PIPELINE XỬ LÝ CHÍNH (GPU)
# ===============================
def run_e2e_gpu_multithread(video_path, output_path, biox3d, miniroad, cfg, device):
    CLIP_LEN = cfg.get('window_size', 13)
    SAMPLE_RATE = cfg.get('sample_rate', 4)
    STRIDE = 6
    SEQ_LEN = 128
    IMG_SIZE = 224
    WINDOW_SPAN = (CLIP_LEN - 1) * SAMPLE_RATE + 1 
    
    CLASS_NAMES = ["Background", "Distracted", "Hands_off_wheel", "Head_check_left",
                   "Head_check_right", "Head_turn_back", "Not_wearing_seatbelt",
                   "One_hand_off_wheel", "Smoking", "Turn_left", "Turn_right", "Using_phone"]

    # --- KHỞI TẠO VIDEO ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ret, first_frame = cap.read()
    cap.release()

    input_queue = Queue(maxsize=128)
    output_queue = Queue(maxsize=128)

    reader = threading.Thread(target=frame_reader_thread, args=(video_path, input_queue))
    writer = threading.Thread(target=frame_writer_thread, args=(output_path, fps, width, height, output_queue))
    reader.start()
    writer.start()

    normalizer = X3D_Normalizer().to(device)

    # Lấp đầy buffer bằng frame đầu tiên
    frame_buffer = deque([first_frame.copy() for _ in range(WINDOW_SPAN)], maxlen=WINDOW_SPAN)
    
    # Warm-up (Khởi động luồng CUDA)
    dummy_in = torch.zeros(1, 3, CLIP_LEN, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
        _, _, _, _, dummy_rgb, dummy_flow = biox3d(dummy_in, return_embeddings=True)
        feat_dim = dummy_rgb.shape[-1]

    # Pre-fill lịch sử
    rgb_history = deque([torch.zeros((1, feat_dim), device=device) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
    flow_history = deque([torch.zeros((1, feat_dim), device=device) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)

    current_predictions = []
    decisions = 0
    t = 0
    start_time = time.perf_counter()

    pbar = tqdm(total=total_frames, desc="Processing")

    # --- LUỒNG CHÍNH GPU ---
    with torch.no_grad():
        while True:
            frame = input_queue.get()
            if frame is None: break
            
            frame_buffer.append(frame)
            t += 1

            if t % STRIDE == 0:
                sampled_frames = list(frame_buffer)[::SAMPLE_RATE]
                clip_tensor = torch.from_numpy(np.array(sampled_frames)).to(device, non_blocking=True).float()
                clip_tensor = clip_tensor.permute(0, 3, 1, 2)

                clip_tensor = T.Resize(IMG_SIZE, antialias=True)(clip_tensor)
                clip_tensor = T.CenterCrop(IMG_SIZE)(clip_tensor)
                clip_tensor = normalizer(clip_tensor.permute(1, 0, 2, 3).unsqueeze(0))

                with autocast(device_type='cuda', dtype=torch.float16):
                    # BioX3D
                    _, _, _, _, rgb_feat, flow_feat = biox3d(clip_tensor, return_embeddings=True)

                    rgb_history.append(rgb_feat)
                    flow_history.append(flow_feat)
                    rgb_seq = torch.stack(list(rgb_history), dim=1)
                    flow_seq = torch.stack(list(flow_history), dim=1)

                    # MiniROAD
                    out_dict = miniroad(rgb_seq, flow_seq)
                    probs = out_dict['logits'].squeeze(0)[-1].cpu().numpy()

                current_predictions = sorted(
                    [(CLASS_NAMES[i], float(p)) for i, p in enumerate(probs) if i > 0],
                    key=lambda x: x[1], reverse=True
                )[:5]
                decisions += 1

            # Vẽ Dashboard mượt mà lên Video
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 40 + len(current_predictions) * 30), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            y_offset = 40
            for class_name, prob in current_predictions:
                color = (0, 0, 255) if prob > 0.5 else (0, 255, 255) if prob > 0.1 else (150, 150, 150)
                cv2.putText(frame, f"{class_name}: {prob*100:.1f}%", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                y_offset += 30

            current_fps = t / (time.perf_counter() - start_time)
            cv2.putText(frame, f"Stream FPS: {current_fps:.1f}", (width - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            output_queue.put(frame)
            pbar.update(1)

    pbar.close()
    output_queue.put(None)
    reader.join()
    writer.join()

    total_time = time.perf_counter() - start_time
    
    total_processed_frames = decisions * CLIP_LEN 
    fps_model_effective = total_processed_frames / total_time
    fps_video_io = total_frames / total_time

    print("-" * 50)
    print(f"Tổng thời gian chạy: {total_time:.2f} giây")
    print("-" * 50)
    print(f"🚀 TỐC ĐỘ MODEL (EFFECTIVE FPS): {fps_model_effective:.2f} FPS")
    print(f"   (Xử lý {decisions} clips x 13 frames)")
    print(f"🎥 Tốc độ luồng Video (I/O)    : {fps_video_io:.2f} FPS")
    print("-" * 50)

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    CONFIG_YAML = "D:\\project\\DashCam\\model\\cake\\MiniROAD\\configs\\dashcam_cls.yaml"
    BIOX3D_PTH = "D:\\project\\DashCam\\model\\cake\\cake\\new_mse_freeze_no_cls\\cake_41.03_e110.pth"
    MINIROAD_PTH = "D:\\project\\DashCam\\model\\cake\\MiniROAD\\output_dashcam\\MiniROAD_DASHCAM_kinetics_flowTrue_7\\ckpts\\best_57.59.pth" 
    VIDEO_IN = "D:\\project\\DashCam\\data\\dashcam\\raw\\0032002C37_2025-09-15_Cam1_Segment2.mp4"
    VIDEO_OUT = "D:\\project\\DashCam\\data\\dashcam\\test_gpu_out.mp4"

    biox3d, miniroad, cfg = load_pytorch_models(BIOX3D_PTH, MINIROAD_PTH, CONFIG_YAML, DEVICE)
    run_e2e_gpu_multithread(VIDEO_IN, VIDEO_OUT, biox3d, miniroad, cfg, DEVICE)

# import torch
# import torch.nn as nn
# import torchvision.transforms as T
# import numpy as np
# import cv2
# import time
# import yaml
# from collections import deque
# from tqdm import tqdm
# import threading
# from queue import Queue

# # --- IMPORT TỪ PROJECT CỦA BẠN ---
# import sys
# sys.path.append(r'D:\project\DashCam\model\cake')
# from cake.cake import BioX3D_Student
# from MiniROAD.model import build_model

# # ===============================
# # TIỀN XỬ LÝ ẢNH TRÊN CPU
# # ===============================
# class X3D_Normalizer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1)
#         self.std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1)

#     def forward(self, x):
#         return (x / 255.0 - self.mean.to(x.device)) / self.std.to(x.device)

# # ===============================
# # HÀM LOAD MODEL
# # ===============================
# def load_pytorch_models(biox3d_path, miniroad_path, config_path, device):
#     print(f"🔧 Đang load cấu hình từ: {config_path}")
#     with open(config_path, 'r') as f:
#         cfg = yaml.load(f, Loader=yaml.FullLoader)
    
#     # Load lên CPU
#     biox3d = BioX3D_Student(clip_len=cfg.get('clip_len', 13), num_classes=400).to(device)
#     biox_state = torch.load(biox3d_path, map_location=device, weights_only=False).get('state_dict', {})
#     biox3d.load_state_dict({k.replace('module.', ''): v for k, v in biox_state.items()}, strict=False)
#     biox3d.eval()

#     miniroad = build_model(cfg, device)
#     miniroad.load_state_dict(torch.load(miniroad_path, map_location=device, weights_only=True))
#     miniroad.eval()

#     return biox3d, miniroad, cfg

# # ==========================================
# # CÁC LUỒNG ĐỌC/GHI VIDEO (CPU)
# # ==========================================
# def frame_reader_thread(video_path, input_queue):
#     cap = cv2.VideoCapture(video_path)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
#         input_queue.put(frame)
#     input_queue.put(None)
#     cap.release()

# def frame_writer_thread(output_path, fps, width, height, output_queue):
#     out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
#     # --- ĐÃ XÓA TÍNH NĂNG HIỂN THỊ (GUI) ĐỂ TIẾT KIỆM TÀI NGUYÊN CPU ---
    
#     while True:
#         frame = output_queue.get()
#         if frame is None: break
#         out.write(frame) # Chỉ ghi ra file
        
#     out.release()

# # ===============================
# # PIPELINE XỬ LÝ CHÍNH (CPU)
# # ===============================
# def run_e2e_cpu_multithread(video_path, output_path, biox3d, miniroad, cfg, device):
#     CLIP_LEN = cfg.get('window_size', 13)
#     SAMPLE_RATE = cfg.get('sample_rate', 4)
#     STRIDE = 6
#     SEQ_LEN = 128
#     IMG_SIZE = 224
#     WINDOW_SPAN = (CLIP_LEN - 1) * SAMPLE_RATE + 1 
    
#     CLASS_NAMES = ["Background", "Distracted", "Hands_off_wheel", "Head_check_left",
#                    "Head_check_right", "Head_turn_back", "Not_wearing_seatbelt",
#                    "One_hand_off_wheel", "Smoking", "Turn_left", "Turn_right", "Using_phone"]

#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     ret, first_frame = cap.read()
#     cap.release()

#     input_queue = Queue(maxsize=128)
#     output_queue = Queue(maxsize=128)

#     reader = threading.Thread(target=frame_reader_thread, args=(video_path, input_queue))
#     writer = threading.Thread(target=frame_writer_thread, args=(output_path, fps, width, height, output_queue))
#     reader.start()
#     writer.start()

#     normalizer = X3D_Normalizer().to(device)

#     frame_buffer = deque([first_frame.copy() for _ in range(WINDOW_SPAN)], maxlen=WINDOW_SPAN)
    
#     dummy_in = torch.zeros(1, 3, CLIP_LEN, IMG_SIZE, IMG_SIZE).to(device)
#     with torch.no_grad(): # KHÔNG DÙNG AUTOCAST
#         _, _, _, _, dummy_rgb, dummy_flow = biox3d(dummy_in, return_embeddings=True)
#         feat_dim = dummy_rgb.shape[-1]

#     rgb_history = deque([torch.zeros((1, feat_dim), device=device) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
#     flow_history = deque([torch.zeros((1, feat_dim), device=device) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)

#     current_predictions = []
#     decisions = 0
#     t = 0
#     start_time = time.perf_counter()

#     pbar = tqdm(total=total_frames, desc="CPU Processing")

#     # --- LUỒNG CHÍNH TÍNH TOÁN BẰNG CPU ---
#     with torch.no_grad():
#         while True:
#             frame = input_queue.get()
#             if frame is None: break
            
#             frame_buffer.append(frame)
#             t += 1

#             if t % STRIDE == 0:
#                 sampled_frames = list(frame_buffer)[::SAMPLE_RATE]
#                 # Bỏ tham số non_blocking=True của GPU
#                 clip_tensor = torch.from_numpy(np.array(sampled_frames)).to(device).float()
#                 clip_tensor = clip_tensor.permute(0, 3, 1, 2)

#                 clip_tensor = T.Resize(IMG_SIZE, antialias=True)(clip_tensor)
#                 clip_tensor = T.CenterCrop(IMG_SIZE)(clip_tensor)
#                 clip_tensor = normalizer(clip_tensor.permute(1, 0, 2, 3).unsqueeze(0))

#                 # --- CHẠY THUẦN FP32 TRÊN CPU ---
#                 _, _, _, _, rgb_feat, flow_feat = biox3d(clip_tensor, return_embeddings=True)

#                 rgb_history.append(rgb_feat)
#                 flow_history.append(flow_feat)
#                 rgb_seq = torch.stack(list(rgb_history), dim=1)
#                 flow_seq = torch.stack(list(flow_history), dim=1)

#                 out_dict = miniroad(rgb_seq, flow_seq)
#                 probs = out_dict['logits'].squeeze(0)[-1].cpu().numpy()

#                 current_predictions = sorted(
#                     [(CLASS_NAMES[i], float(p)) for i, p in enumerate(probs) if i > 0],
#                     key=lambda x: x[1], reverse=True
#                 )[:5]
#                 decisions += 1

#             overlay = frame.copy()
#             cv2.rectangle(overlay, (10, 10), (350, 40 + len(current_predictions) * 30), (0, 0, 0), -1)
#             cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

#             y_offset = 40
#             for class_name, prob in current_predictions:
#                 color = (0, 0, 255) if prob > 0.5 else (0, 255, 255) if prob > 0.1 else (150, 150, 150)
#                 cv2.putText(frame, f"{class_name}: {prob*100:.1f}%", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
#                 y_offset += 30

#             current_fps = t / (time.perf_counter() - start_time)
#             cv2.putText(frame, f"CPU Stream FPS: {current_fps:.1f}", (width - 320, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#             output_queue.put(frame)
#             pbar.update(1)

#     pbar.close()
#     output_queue.put(None)
#     reader.join()
#     writer.join()

#     total_time = time.perf_counter() - start_time
#     total_processed_frames = decisions * CLIP_LEN 
#     fps_model_effective = total_processed_frames / total_time
#     fps_video_io = total_frames / total_time

#     print("-" * 50)
#     print(f"Tổng thời gian chạy: {total_time:.2f} giây")
#     print("-" * 50)
#     print(f"🚀 TỐC ĐỘ MODEL (EFFECTIVE FPS): {fps_model_effective:.2f} FPS (CPU Only)")
#     print(f"   (Xử lý {decisions} clips x 13 frames)")
#     print(f"🎥 Tốc độ luồng Video (I/O)    : {fps_video_io:.2f} FPS")
#     print("-" * 50)

# # ===============================
# # MAIN
# # ===============================
# if __name__ == "__main__":
#     # Ép buộc chạy trên CPU 100%
#     DEVICE = torch.device("cpu")
#     #print(f"⚠️ HỆ THỐNG ĐANG CHẠY CHẾ ĐỘ BENCHMARK TRÊN: {DEVICE.upper()}")
    
#     CONFIG_YAML = "D:\\project\\DashCam\\model\\cake\\MiniROAD\\configs\\dashcam_cls.yaml"
#     BIOX3D_PTH = "D:\\project\\DashCam\\model\\cake\\cake\\new_mse_freeze_no_cls\\cake_41.03_e110.pth"
#     MINIROAD_PTH = "D:\\project\\DashCam\\model\\cake\\MiniROAD\\output_dashcam\\MiniROAD_DASHCAM_kinetics_flowTrue_7\\ckpts\\best_57.59.pth" 
#     VIDEO_IN = "D:\\project\\DashCam\\data\\dashcam\\raw\\0032002C37_2025-09-15_Cam1_Segment0.mp4"
    
#     # Đổi tên file output để không ghi đè lên file kết quả của GPU
#     VIDEO_OUT = "D:\\project\\DashCam\\data\\dashcam\\test_cpu_out.mp4"

#     biox3d, miniroad, cfg = load_pytorch_models(BIOX3D_PTH, MINIROAD_PTH, CONFIG_YAML, DEVICE)
#     run_e2e_cpu_multithread(VIDEO_IN, VIDEO_OUT, biox3d, miniroad, cfg, DEVICE)