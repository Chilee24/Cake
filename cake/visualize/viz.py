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
from cake import BioX3D_Student
from teacher_utils import TeacherPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# UTILS
# ==============================================================================
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
    """
    Tạo heatmap và resize lên kích thước gốc. Trả về List ảnh BGR.
    """
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

def create_grid_video_5col(full_imgs_rgb, hm_rgb_bgr, raft_flows_bgr, hm_teacher_bgr, hm_student_bgr, output_path, fps=5):
    """
    Tạo video grid 5 cột: RGB | Student RGB Feat | RAFT Flow | Teacher Flow Feat | Student Flow Feat
    """
    T = len(full_imgs_rgb)
    H_orig, W_orig, C = full_imgs_rgb[0].shape
    
    # Grid size: 5 hình ngang
    grid_W = W_orig * 5
    grid_H = H_orig
    print(f"-> Creating output video (5 cols): {grid_W}x{grid_H}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (grid_W, grid_H))
    
    alpha = 0.5
    
    for t in range(T):
        # 1. RGB Orig
        img_bgr = cv2.cvtColor(full_imgs_rgb[t], cv2.COLOR_RGB2BGR)
        
        # 2. Student RGB Feature (Overlay)
        ov_rgb = cv2.addWeighted(img_bgr, 1-alpha, hm_rgb_bgr[t], alpha, 0)

        # 3. RAFT Flow
        raft_bgr = raft_flows_bgr[t]
        
        # 4. Teacher Flow Feature (Overlay)
        ov_tea = cv2.addWeighted(img_bgr, 1-alpha, hm_teacher_bgr[t], alpha, 0)
        
        # 5. Student Flow Hallucination (Overlay)
        ov_stu = cv2.addWeighted(img_bgr, 1-alpha, hm_student_bgr[t], alpha, 0)
        
        # Text Annotation
        font_scale = max(0.8, H_orig / 600.0) 
        thickness = max(2, int(H_orig / 350.0))
        color = (255, 255, 255)
        
        cv2.putText(img_bgr, f"Raw RGB (Frame {t})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        cv2.putText(ov_rgb,  "Student RGB Features", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        cv2.putText(raft_bgr,"RAFT Optical Flow", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        cv2.putText(ov_tea,  "Teacher Flow Features", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        cv2.putText(ov_stu,  "Student Flow Hallucinated", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # Ghép 5 hình
        grid_frame = np.hstack([img_bgr, ov_rgb, raft_bgr, ov_tea, ov_stu])
        out.write(grid_frame)
        
    out.release()
    print(f"✅ Saved to: {output_path}")

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
    print(f"-> Resolution: {orig_W}x{orig_H}")
    
    # --- FORWARD ---
    with torch.no_grad():
        teacher_feat = teacher.get_teacher_features(inputs)
        inputs_norm = normalizer(inputs)
        # [QUAN TRỌNG] Lấy cả rgb_features
        _, _, rgb_features, student_flow_hallucinated = student(inputs_norm)

    # --- GENERATE VISUALS ---
    print("-> Generating Heatmaps...")
    hms_teacher_bgr = get_heatmap_sequence_full_res(teacher_feat, orig_H, orig_W)
    hms_student_flow_bgr = get_heatmap_sequence_full_res(student_flow_hallucinated, orig_H, orig_W)
    
    # [MỚI] Tạo heatmap cho nhánh RGB
    hms_student_rgb_bgr = get_heatmap_sequence_full_res(rgb_features, orig_H, orig_W)
    
    print("-> Generating RAFT Flow...")
    raft_flows = get_raft_flow_sequence_full_res(teacher, inputs, orig_H, orig_W)
    
    # --- SAVE ---
    output_name = f"vis_5col_{os.path.basename(video_path)}"
    create_grid_video_5col(
        full_res_imgs, 
        hms_student_rgb_bgr, 
        raft_flows, 
        hms_teacher_bgr, 
        hms_student_flow_bgr, 
        output_name, 
        fps=5
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="./checkpoints_k400_distill_mgd/model_best.pth")
    parser.add_argument('--raft_weights', type=str, required=True)
    parser.add_argument('--flow_teacher_weights', type=str, required=True)
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--val_list', type=str, default=None)
    parser.add_argument('--val_root', type=str, default=None)
    parser.add_argument('--clip_len', type=int, default=13)
    parser.add_argument('--stride', type=int, default=4)
    
    args = parser.parse_args()
    main(args)