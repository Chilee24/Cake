import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import decord
import cv2
import random
import argparse
from torchvision.utils import flow_to_image 
sys.path.append("..")

# --- CONFIG & IMPORTS ---
try:
    from cake import BioX3D_Student
    from teacher_utils import TeacherPipeline
    # Đảm bảo model load được nếu dùng ODConv trong cake.py
    # from odconv3d import ODConv3d 
except ImportError:
    print("❌ Hãy đặt file này cùng thư mục với cake.py và teacher_utils.py")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# UTILS
# ==============================================================================
def set_odconv_temperature(model, temperature=4.6):
    count = 0
    for m in model.modules():
        if hasattr(m, 'update_temperature'):
            m.update_temperature(temperature)
            count += 1
    print(f"🌡️ Đã set ODConv Temperature = {temperature} cho {count} modules.")

class X3D_Normalizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1).to(device)
        self.std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1).to(device)
    def forward(self, x): return (x / 255.0 - self.mean) / self.std

def load_video_with_stride(path, clip_len=13, stride=1, model_input_size=224):
    if not os.path.exists(path): raise FileNotFoundError(f"Not found: {path}")
    vr = decord.VideoReader(path)
    total = len(vr)
    required_frames = (clip_len - 1) * stride + 1
    if total > required_frames:
        start = random.randint(0, total - required_frames)
    else:
        start = 0
    
    indices = [min(start + i * stride, total - 1) for i in range(clip_len)]
    print(f"-> Sampling Indices (Stride={stride}): {indices}")
    
    buffer_full_res = vr.get_batch(indices).asnumpy() 
    full_res_imgs = list(buffer_full_res)
    
    tensor = torch.from_numpy(buffer_full_res).permute(0, 3, 1, 2).float() 
    tensor = F.interpolate(tensor, size=(model_input_size, model_input_size), mode='bilinear', align_corners=False)
    tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0) 
    
    return full_res_imgs, tensor.to(device)

def get_heatmap_sequence_full_res(feature_map, target_H, target_W):
    """Tạo heatmap và resize lên kích thước gốc. Trả về List ảnh BGR."""
    heatmap_t = feature_map.mean(dim=1).squeeze(0) # (T, h, w)
    heatmaps_bgr = []
    for t in range(heatmap_t.shape[0]):
        hm = heatmap_t[t].detach().cpu().numpy()
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        hm_full = cv2.resize(hm, (target_W, target_H))
        hm_bgr = cv2.applyColorMap(np.uint8(255 * hm_full), cv2.COLORMAP_JET)
        heatmaps_bgr.append(hm_bgr)
    return heatmaps_bgr

def get_raft_flow_sequence_full_res(teacher_pipeline, inputs_tensor, target_H, target_W):
    raft_model = teacher_pipeline.raft
    inputs_flat = inputs_tensor.squeeze(0).permute(1, 0, 2, 3) 
    img1 = inputs_flat[:-1] 
    img2 = inputs_flat[1:]  
    
    # Normalize [-1, 1] cho RAFT
    img1 = 2 * (img1 / 255.0) - 1.0
    img2 = 2 * (img2 / 255.0) - 1.0
    
    with torch.no_grad():
        list_of_flows = raft_model(img1, img2)
        predicted_flows = list_of_flows[-1] 
        
    flow_imgs_tensor = flow_to_image(predicted_flows) 
    
    flow_imgs_bgr = []
    for i in range(flow_imgs_tensor.shape[0]):
        flow_img = flow_imgs_tensor[i].permute(1, 2, 0).cpu().numpy()
        flow_img = cv2.resize(flow_img, (target_W, target_H))
        flow_img = cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR)
        flow_imgs_bgr.append(flow_img)
        
    flow_imgs_bgr.append(flow_imgs_bgr[-1].copy())
    return flow_imgs_bgr

def create_video(frames_bgr, output_path, fps=5):
    """Lưu video từ list frames"""
    if len(frames_bgr) == 0: return
    H, W, C = frames_bgr[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for frame in frames_bgr:
        out.write(frame)
    out.release()
    print(f"🎥 Video saved: {output_path}")

# ==============================================================================
# NEW SAVING FUNCTION
# ==============================================================================
def save_separate_hstack_sequences(full_imgs_rgb, hm_rgb_bgr, raft_flows_bgr, hm_teacher_bgr, hm_student_bgr, output_dir, num_frames_to_save=6):
    """
    Lưu 6 file ảnh riêng biệt. Mỗi file là một dải ảnh ghép ngang (hstack) 
    của 6 frame đại diện theo thời gian cho MỘT loại dữ liệu.
    Không viết chữ lên ảnh.
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    T = len(full_imgs_rgb)
    # Chọn index đều nhau. Ví dụ T=13 -> [0, 2, 4, 7, 9, 12]
    indices = np.linspace(0, T-1, num_frames_to_save, dtype=int)
    alpha = 0.5
    
    print(f"📸 Saving separate sequences for {num_frames_to_save} key frames to: {output_dir}")

    # Khởi tạo các list để chứa chuỗi ảnh cho từng loại
    seq_rgb = []
    seq_flow = []
    seq_stu_rgb_ov = []
    seq_tea_flow_ov = []
    seq_stu_hal_ov = []
    seq_stu_hal_raw = []

    # Vòng lặp thu thập các frame tại các thời điểm đã chọn
    for idx in indices:
        # 1. RGB Gốc
        rgb = cv2.cvtColor(full_imgs_rgb[idx], cv2.COLOR_RGB2BGR)
        seq_rgb.append(rgb)
        
        # 2. Optical Flow (RAFT)
        flow = raft_flows_bgr[idx]
        seq_flow.append(flow)
        
        # 3. Heatmap Student RGB (Overlay)
        stu_rgb_ov = cv2.addWeighted(rgb, 1-alpha, hm_rgb_bgr[idx], alpha, 0)
        seq_stu_rgb_ov.append(stu_rgb_ov)
        
        # 4. Heatmap Teacher Flow (Overlay)
        tea_flow_ov = cv2.addWeighted(rgb, 1-alpha, hm_teacher_bgr[idx], alpha, 0)
        seq_tea_flow_ov.append(tea_flow_ov)
        
        # 5. Hallucination (Overlay)
        stu_hal_ov = cv2.addWeighted(rgb, 1-alpha, hm_student_bgr[idx], alpha, 0)
        seq_stu_hal_ov.append(stu_hal_ov)
        
        # 6. Heatmap Raw (Raw Jet)
        stu_hal_raw = hm_student_bgr[idx]
        seq_stu_hal_raw.append(stu_hal_raw)

    # Định nghĩa các file đầu ra và danh sách ảnh tương ứng
    outputs_to_save = {
        "01_sequence_RGB_Original.jpg": seq_rgb,
        "02_sequence_Optical_Flow_RAFT.jpg": seq_flow,
        "03_sequence_Heatmap_Student_RGB_Overlay.jpg": seq_stu_rgb_ov,
        "04_sequence_Heatmap_Teacher_Flow_Overlay.jpg": seq_tea_flow_ov,
        "05_sequence_Heatmap_Student_Hallucination_Overlay.jpg": seq_stu_hal_ov,
        "06_sequence_Heatmap_Student_Hallucination_Raw.jpg": seq_stu_hal_raw,
    }

    # Thực hiện ghép ngang (hstack) và lưu từng file
    for filename, sequence_list in outputs_to_save.items():
        # Ghép các frame trong list lại theo chiều ngang
        stacked_img = np.hstack(sequence_list)
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, stacked_img)
        print(f"   -> Saved: {filename}")

# ==============================================================================
# MAIN
# ==============================================================================
def main(args):
    print(f"🚀 Loading Student: {args.checkpoint}")
    student = BioX3D_Student(clip_len=args.clip_len, feature_dim=192).to(device)
    
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        student.load_state_dict(state, strict=False)
    student.eval()
    
    # Set nhiệt độ (Quan trọng nếu dùng ODConv)
    # Hãy điều chỉnh giá trị này khớp với lúc train xong (ví dụ 1.0 hoặc 2.8)
    set_odconv_temperature(student, temperature=2.8) 
    
    print("🚀 Loading Teacher...")
    teacher = TeacherPipeline(args.raft_weights, args.flow_teacher_weights, device=device)
    normalizer = X3D_Normalizer()

    # --- VIDEO PATH ---
    video_path = args.video_path
    if not video_path:
        if args.val_list:
            with open(args.val_list, 'r') as f:
                lines = f.readlines()
                line = random.choice(lines).strip()
                v_name = " ".join(line.split()[:-1])
                video_path = os.path.join(args.val_root, v_name)
        else:
            print("❌ Cần --video_path")
            sys.exit(1)
            
    print(f"🎬 Processing: {video_path}")
    
    # --- LOAD ---
    full_res_imgs, inputs = load_video_with_stride(video_path, args.clip_len, stride=args.stride)
    orig_H, orig_W = full_res_imgs[0].shape[:2]
    
    # --- FORWARD ---
    with torch.no_grad():
        teacher_feat = teacher.get_teacher_features(inputs)
        inputs_norm = normalizer(inputs)
        # Lấy 4 outputs từ Student
        _, _, rgb_features, student_flow_hallucinated = student(inputs_norm)

    # --- GENERATE VISUALS ---
    print("-> Generating Heatmaps...")
    hms_teacher_bgr = get_heatmap_sequence_full_res(teacher_feat, orig_H, orig_W)
    hms_student_flow_bgr = get_heatmap_sequence_full_res(student_flow_hallucinated, orig_H, orig_W)
    hms_student_rgb_bgr = get_heatmap_sequence_full_res(rgb_features, orig_H, orig_W)
    
    print("-> Generating RAFT Flow...")
    raft_flows = get_raft_flow_sequence_full_res(teacher, inputs, orig_H, orig_W)
    
    # --- OUTPUT SETUP ---
    # Tạo tên thư mục dựa trên tên video
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = f"vis_output_{video_basename}"
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # 1. LƯU CÁC DẢI ẢNH RIÊNG BIỆT (Mới)
    save_separate_hstack_sequences(
        full_res_imgs,
        hms_student_rgb_bgr,
        raft_flows,
        hms_teacher_bgr,
        hms_student_flow_bgr,
        out_dir,
        num_frames_to_save=6 # Số lượng frame muốn ghép trong 1 ảnh
    )

    # 2. LƯU VIDEO (Vẫn giữ lại để xem chuyển động nếu cần)
    video_frames = []
    alpha = 0.5
    for t in range(len(full_res_imgs)):
        img = cv2.cvtColor(full_res_imgs[t], cv2.COLOR_RGB2BGR)
        ov_hal = cv2.addWeighted(img, 1-alpha, hms_student_flow_bgr[t], alpha, 0)
        ov_tea = cv2.addWeighted(img, 1-alpha, hms_teacher_bgr[t], alpha, 0)
        flow = raft_flows[t]
        # Grid 4 cột cho video (RGB, Flow, Teacher, Student Hal)
        row = np.hstack([img, flow, ov_tea, ov_hal])
        video_frames.append(row)
    
    create_video(video_frames, os.path.join(out_dir, "video_preview_4col.mp4"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="./checkpoints_k400_distill_mgd/model_best.pth")
    parser.add_argument('--raft_weights', type=str, required=True)
    parser.add_argument('--flow_teacher_weights', type=str, required=True)
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--val_list', type=str, default=None)
    parser.add_argument('--val_root', type=str, default=None)
    parser.add_argument('--clip_len', type=int, default=13)
    parser.add_argument('--stride', type=int, default=6)
    
    args = parser.parse_args()
    main(args)