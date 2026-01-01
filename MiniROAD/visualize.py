import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import yaml
import os
import os.path as osp
from tqdm import tqdm

# Import các module dự án của bạn
from utils import create_outdir
from model import build_model
from datasets import build_data_loader

def extract_features(loader, model, device):
    """
    Trích xuất đặc trưng từ Encoder Student (Q) với dữ liệu sạch.
    """
    model.eval()
    features_list = []
    labels_list = []
    
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Features"):
            # Unpack Dictionary
            rgb = batch['rgb_anchor'].to(device, non_blocking=True)
            flow = batch['flow_anchor'].to(device, non_blocking=True)
            
            # Lấy nhãn class (hoặc queue_id nếu muốn visualize theo cụm queue)
            # Ở đây ta lấy labels gốc để visualize ngữ nghĩa
            labels = batch['labels'] 
            
            # --- FORWARD PASS (Manual) ---
            # Ta muốn xem feature của Encoder Student trên dữ liệu sạch
            # 1. Backbone Feature
            feat = model.encoder_q(rgb, flow, return_embedding=True)
            
            # 2. Projection Head (Optional: Thường T-SNE trên Projection space sẽ đẹp hơn)
            proj = model.head_queue(feat)
            proj = F.normalize(proj, dim=1) # Normalize là bắt buộc cho Cosine Similarity
            
            features_list.append(proj.cpu().numpy())
            labels_list.append(labels.numpy())
            
            total_samples += rgb.size(0)
                
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels

def plot_tsne(features, labels, save_path, title='T-SNE'):
    """
    Chạy thuật toán T-SNE và vẽ biểu đồ
    """
    print(f"--> Running T-SNE on {features.shape[0]} samples...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(features)
    
    # Setup plot
    plt.figure(figsize=(16, 10))
    
    # Lấy danh sách class unique
    unique_labels = np.unique(labels)
    
    # Sử dụng colormap 'jet' hoặc 'tab20' để có nhiều màu
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        # Lọc các điểm thuộc class này
        indices = labels == label
        plt.scatter(
            tsne_results[indices, 0], 
            tsne_results[indices, 1], 
            c=[cmap(i)], 
            label=f'Class {label}', 
            alpha=0.6, 
            s=20
        )
    
    plt.title(title, fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    print(f"--> Saving figure to {save_path}")
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=r'D:\project\DashCam\MiniROAD\configs\miniroad_thumos_kinetics_phase1.yaml')
    parser.add_argument('--checkpoint', type=str, default=r'C:\Users\hieuh\Downloads\best_encoder.pth')
    args = parser.parse_args()

    # Load Config
    cfg = yaml.load(open(args.config), Loader=yaml.FullLoader)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Output Dir
    vis_dir = osp.join(osp.dirname(args.checkpoint), 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Build Model
    print("--> Building Model...")
    model = build_model(cfg).to(device)
    
    # 2. Load Checkpoint
    print(f"--> Loading Checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Xử lý key 'state_dict'
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
        
    model.load_state_dict(state_dict, strict=False) # strict=False để bỏ qua queue buffer nếu lệch size
    
    # 3. Process Datasets
    modes = ['train', 'test']
    
    for mode in modes:
        print(f"\n=== Processing {mode.upper()} Set ===")
        # Build Loader
        loader = build_data_loader(cfg, mode=mode)
        
        # Extract
        feats, lbls = extract_features(loader, model, device)
        
        # Plot
        save_name = osp.join(vis_dir, f'tsne_{mode}_epoch{ckpt.get("epoch", "unknown")}.png')
        plot_tsne(feats, lbls, save_name, title=f'T-SNE Visualization ({mode.upper()})')

if __name__ == '__main__':
    main()