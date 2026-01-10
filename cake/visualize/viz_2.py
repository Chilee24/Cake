import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import decord
import cv2
import argparse

# --- IMPORTS ---
import sys
sys.path.append("..")
try:
    from cake import BioX3D_Student
    from teacher_utils import TeacherPipeline
except ImportError:
    print("❌ Thiếu file cake.py / teacher_utils.py")
    sys.exit(1)

device = torch.device("cuda")

# ==============================================================================
# UTILS
# ==============================================================================
class X3D_Normalizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1).to(device)
        self.std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1).to(device)
    def forward(self, x): return (x / 255.0 - self.mean) / self.std

def get_temporal_variance_map(feature_map):
    """
    Tính độ biến thiên của feature theo thời gian.
    Input: (1, C, T, H, W)
    Output: Heatmap (H, W) thể hiện chỗ nào thay đổi nhiều nhất
    """
    # 1. Tính Std theo trục T: (1, C, H, W)
    # std càng cao -> Feature tại đó thay đổi càng mạnh qua các frame
    feat_std = feature_map.std(dim=2) 
    
    # 2. Mean theo trục C: (1, H, W)
    heatmap = feat_std.mean(dim=1).squeeze(0)
    
    # 3. Normalize & Colorize
    hm = heatmap.detach().cpu().numpy()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hm = cv2.resize(hm, (224, 224))
    hm_color = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_INFERNO) # Dùng Inferno cho ngầu
    return hm_color

def load_video_center(path, clip_len=13):
    vr = decord.VideoReader(path)
    total = len(vr)
    start = max(0, (total - clip_len) // 2)
    indices = [min(start + i, total - 1) for i in range(clip_len)]
    buffer = vr.get_batch(indices).asnumpy()
    img_mid = cv2.resize(buffer[clip_len//2], (224, 224)) # Frame giữa
    
    tensor = torch.from_numpy(buffer).permute(0, 3, 1, 2).float()
    tensor = torch.nn.functional.interpolate(tensor, size=(224, 224), mode='bilinear')
    tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)
    return img_mid, tensor.to(device)

# ==============================================================================
# MAIN
# ==============================================================================
def main(args):
    print(f"🚀 Loading Student: {args.checkpoint}")
    student = BioX3D_Student(clip_len=13, feature_dim=192).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    student.load_state_dict(ckpt['state_dict'], strict=False)
    student.eval()
    
    normalizer = X3D_Normalizer()
    
    # Lấy video
    with open(args.val_list, 'r') as f:
        line = f.readline().strip() # Lấy video đầu tiên hoặc random
        v_name = " ".join(line.split()[:-1])
        path = os.path.join(args.val_root, v_name)
    print(f"🎬 Checking Variance on: {v_name}")

    # Forward
    img_mid, inputs = load_video_center(path)
    with torch.no_grad():
        inputs_norm = normalizer(inputs)
        _, _, rgb_feat, flow_feat = student(inputs_norm)

    # Tính Variance Map
    var_rgb = get_temporal_variance_map(rgb_feat)
    var_flow = get_temporal_variance_map(flow_feat)
    
    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_mid)
    axs[0].set_title("Input (Middle Frame)")
    
    axs[1].imshow(cv2.cvtColor(var_rgb, cv2.COLOR_BGR2RGB))
    axs[1].set_title("RGB Temporal Variance\n(Nên thấp/ổn định)")
    
    axs[2].imshow(cv2.cvtColor(var_flow, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Flow Temporal Variance\n(Nên cao/nhấp nháy)")
    
    plt.savefig("vis_variance.png")
    print("✅ Saved to vis_variance.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--val_list', type=str, required=True)
    parser.add_argument('--val_root', type=str, required=True)
    args = parser.parse_args()
    main(args)