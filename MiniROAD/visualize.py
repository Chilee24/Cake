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
import matplotlib

# Import các module dự án
from utils import create_outdir
from model import build_model
from datasets import build_data_loader

def extract_features(loader, model, device, max_samples=None):
    """
    Trích xuất đặc trưng từ Encoder Student (Q).
    Args:
        max_samples: Giới hạn số lượng mẫu để visualize (tránh tràn RAM với tập Train)
    """
    model.eval()
    features_list = []
    labels_list = []
    
    total_samples = 0
    
    # Shuffle loader nếu cần lấy mẫu ngẫu nhiên cho tập Train
    # Tuy nhiên, loader test thường không shuffle. 
    # Nếu muốn random subsample chính xác, cần code thêm, nhưng lấy n mẫu đầu tiên của loader đã shuffle (train) là ổn.
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Extracting Features")):
            # Unpack Dictionary
            rgb = batch['rgb_anchor'].to(device, non_blocking=True)
            flow = batch['flow_anchor'].to(device, non_blocking=True)
            labels = batch['labels']
            
            # --- FORWARD PASS ---
            # 1. Backbone Feature
            # Lưu ý: encoder_q trong ContrastiveMROADMultiQueue nhận (rgb, flow)
            feat = model.encoder_q(rgb, flow, return_embedding=True)
            
            # 2. Projection Head (Quan trọng: Visualize không gian Contrastive)
            proj = model.head_queue(feat)
            proj = F.normalize(proj, dim=1) 
            
            features_list.append(proj.cpu().numpy())
            labels_list.append(labels.numpy())
            
            current_batch_size = rgb.size(0)
            total_samples += current_batch_size
            
            # Dừng nếu vượt quá giới hạn mẫu
            if max_samples is not None and total_samples >= max_samples:
                print(f"--> Reached limit of {max_samples} samples. Stopping extraction.")
                break
                
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    # Nếu vượt quá, cắt bớt cho gọn
    if max_samples is not None and features.shape[0] > max_samples:
        features = features[:max_samples]
        labels = labels[:max_samples]
        
    return features, labels

def plot_tsne(features, labels, save_path, title='T-SNE', bg_class_idx=0):
    """
    Vẽ T-SNE với màu sắc phân biệt.
    """
    print(f"--> Running T-SNE on {features.shape[0]} samples...")
    
    # Init T-SNE (perplexity=30-50 thường tốt cho data dày đặc)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(features)
    
    # Setup plot
    plt.figure(figsize=(16, 12))
    
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # Chọn colormap hỗ trợ nhiều class (THUMOS có 22 class)
    # 'nipy_spectral' hoặc 'jet' có dải màu rộng hơn 'tab20'
    cmap = matplotlib.colormaps['nipy_spectral']
    
    print("--> Plotting...")
    for i, label in enumerate(unique_labels):
        indices = labels == label
        
        # Xử lý màu sắc & Label
        color = cmap(i / num_classes)
        label_text = f'Class {label}'
        
        # Làm mờ Background để Action nổi bật hơn
        alpha = 0.1 if label == bg_class_idx else 0.8
        size = 10 if label == bg_class_idx else 25
        zorder = 1 if label == bg_class_idx else 10 # Vẽ Action đè lên Background
        
        if label == bg_class_idx:
            label_text = "Background"
            color = 'lightgray' # Màu xám cho nền
        
        plt.scatter(
            tsne_results[indices, 0], 
            tsne_results[indices, 1], 
            c=[color], 
            label=label_text, 
            alpha=alpha, 
            s=size,
            edgecolors='none' if label == bg_class_idx else 'k', # Viền đen cho Action
            linewidth=0.1,
            zorder=zorder
        )
    
    plt.title(title, fontsize=20)
    # Legend bên ngoài
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, markerscale=2)
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    print(f"--> Saving figure to {save_path}")
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    # Sửa đường dẫn mặc định cho khớp môi trường Kaggle/Colab của bạn
    parser.add_argument('--config', type=str, default='/kaggle/working/Cake/MiniROAD/configs/thumos_contrastive.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint')
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
    
    # Handle state dict
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
        
    # Lọc bỏ tiền tố 'module.' nếu có (do DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    print("--> Model loaded!")
    
    # 3. Process Datasets
    # Chỉ chạy Test trước để kiểm tra nhanh (Train chạy sau vì nặng)
    modes = ['test', 'train'] 
    
    bg_class_idx = cfg.get('bg_class_idx', 0)

    for mode in modes:
        print(f"\n=== Processing {mode.upper()} Set ===")
        
        # Build Loader
        # Lưu ý: Train loader mặc định shuffle=True trong dataset.py, rất tốt cho việc lấy mẫu ngẫu nhiên
        loader = build_data_loader(cfg, mode=mode)
        
        # Extract
        feats, lbls = extract_features(loader, model, device)
        
        # Plot
        epoch_info = ckpt.get("epoch", "unknown")
        save_name = osp.join(vis_dir, f'tsne_{mode}_ep{epoch_info}.png')
        plot_tsne(feats, lbls, save_name, title=f'T-SNE {mode.upper()} (Epoch {epoch_info})', bg_class_idx=bg_class_idx)

if __name__ == '__main__':
    main()