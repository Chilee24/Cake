import torch
import torch.nn as nn
from tqdm import tqdm
import time
from utils import thumos_postprocessing
from utils import *
import json
from trainer.eval_builder import EVAL
from utils import thumos_postprocessing, perframe_average_precision
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

@EVAL.register("CONTRASTIVE")
class ContrastiveEvaluator:
    def __init__(self, cfg):
        """
        Khởi tạo Evaluator với các tham số từ Config.
        """
        self.num_classes = cfg['num_classes']
        # Lấy background index, mặc định là 0
        self.bg_class_idx = cfg.get('bg_class_idx', 0) 
        # Cấu hình tính CAC (5% hàng xóm)
        self.k_neighbors_ratio = 0.05 

    def __call__(self, val_loader, model, criterion, epoch, writer=None):
        """
        Phương thức chính được gọi khi chạy eval.
        Signature giống hệt hàm val_one_epoch cũ để tương thích với main.py.
        """
        model.eval()
        total_loss = 0.0
        
        # Containers để gom dữ liệu
        all_features = []
        all_labels = []

        with torch.no_grad():
            # --- BƯỚC 1: LOOP QUA TẬP VAL ---
            # Dùng tqdm để hiển thị tiến độ
            pbar = tqdm(val_loader, desc=f"Epoch:{epoch} Validating")
            
            for batch_data in pbar:
                # Move data to GPU
                rgb_anchor = batch_data['rgb_anchor'].cuda(non_blocking=True)
                flow_anchor = batch_data['flow_anchor'].cuda(non_blocking=True)
                rgb_shuff = batch_data['rgb_shuff'].cuda(non_blocking=True)
                flow_shuff = batch_data['flow_shuff'].cuda(non_blocking=True)
                labels = batch_data['labels'].cuda(non_blocking=True)

                # Forward Pass
                out_dict = model(rgb_anchor, flow_anchor, rgb_shuff, flow_shuff, labels)

                # Tính Loss (để tham khảo)
                loss = criterion(out_dict, labels)
                total_loss += loss.item()

                # Thu thập Feature từ Teacher (k_cls)
                # k_cls đã được normalize trong model (L2 norm)
                k_cls = out_dict['k_cls']
                
                all_features.append(k_cls)
                all_labels.append(labels)

        # --- BƯỚC 2: CHUẨN BỊ DỮ LIỆU TÍNH METRIC ---
        # Nối các batch lại thành 1 Tensor lớn
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        avg_loss = total_loss / len(val_loader)
        n_samples = all_features.shape[0]

        print(f"--> [Val] Calculating Distance Matrix for {n_samples} samples...")

        # Tính Ma trận khoảng cách Euclidean (Distance Matrix)
        # Feature đã normalize -> Euclidean distance tương đương Cosine distance về thứ tự
        # Dùng GPU để tính cho nhanh, sau đó đẩy về CPU
        dist_matrix = torch.cdist(all_features, all_features, p=2)
        
        # Chuyển về CPU numpy để xử lý logic index (tránh OOM GPU nếu data lớn)
        dist_matrix_np = dist_matrix.cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()

        # --- BƯỚC 3: TÍNH CAC & CAD ---
        cac_score = self._calculate_cac_multiclass(dist_matrix_np, all_labels_np)
        cad_score = self._calculate_cad_multiclass(dist_matrix_np, all_labels_np)

        print(f"--> [Val Epoch {epoch}] Loss: {avg_loss:.4f} | CAC (Action): {cac_score:.2f}% | CAD (Action): {cad_score:.4f}")

        # --- BƯỚC 4: LOGGING ---
        if writer is not None:
            writer.add_scalar("Val/Loss", avg_loss, epoch)
            writer.add_scalar("Val/CAC_Action", cac_score, epoch)
            writer.add_scalar("Val/CAD_Action", cad_score, epoch)

        return cac_score

    def _calculate_cac_multiclass(self, dist_matrix, labels):
        """
        Helper: Tính Class Alignment Consistency (CAC).
        """
        n_samples = dist_matrix.shape[0]
        # Số lượng hàng xóm cần xét (ví dụ 5%)
        k_neighbors = max(1, int(n_samples * self.k_neighbors_ratio))

        # Tìm index của k hàng xóm gần nhất (bỏ cột 0 là chính nó)
        sorted_indices = np.argsort(dist_matrix, axis=1)[:, 1:k_neighbors+1]

        # Lấy nhãn của các hàng xóm
        neighbor_labels = labels[sorted_indices]

        cac_per_class = []

        for c in range(self.num_classes):
            if c == self.bg_class_idx: continue  # Bỏ qua Background

            # Lấy indices của các mẫu thuộc class c
            indices_c = np.where(labels == c)[0]
            if len(indices_c) == 0: continue

            # Lấy bảng nhãn hàng xóm tương ứng
            neighbors_of_c = neighbor_labels[indices_c]

            # Tính độ nhất quán (Consistency): Tỷ lệ hàng xóm đúng class
            consistency = (neighbors_of_c == c).mean(axis=1)
            
            # Trung bình consistency của class này
            cac_per_class.append(consistency.mean())

        if len(cac_per_class) == 0: return 0.0
        return np.mean(cac_per_class) * 100.0

    def _calculate_cad_multiclass(self, dist_matrix, labels):
        """
        Helper: Tính Class Alignment Distance (CAD).
        """
        cad_per_class = []

        for c in range(self.num_classes):
            if c == self.bg_class_idx: continue  # Bỏ qua Background

            indices_c = np.where(labels == c)[0]
            # Cần ít nhất 2 mẫu để tính khoảng cách
            if len(indices_c) < 2: continue

            # Lấy sub-matrix khoảng cách giữa các điểm trong cùng class
            sub_dist = dist_matrix[np.ix_(indices_c, indices_c)]

            # Tính trung bình (tổng khoảng cách / số cặp)
            # sub_dist bao gồm cả đường chéo chính (0).
            # Số phần tử ma trận là n*n. Số cặp (trừ chính nó) là n*(n-1).
            n = len(indices_c)
            avg_dist = sub_dist.sum() / (n * (n - 1))

            cad_per_class.append(avg_dist)

        if len(cad_per_class) == 0: return 0.0
        return np.mean(cad_per_class)

@EVAL.register("OAD")
class Evaluate(nn.Module):
    
    def __init__(self, cfg):
        super(Evaluate, self).__init__()
        self.data_processing = thumos_postprocessing if 'THUMOS' in cfg['data_name'] else None
        self.metric = cfg['metric']
        self.eval_method = perframe_average_precision
        self.all_class_names = json.load(open(cfg['video_list_path']))[cfg["data_name"].split('_')[0]]['class_index']
    
    def eval(self, model, dataloader, logger):
        device = "cuda:0"
        model.eval()   
        with torch.no_grad():
            pred_scores, gt_targets = [], []
            start = time.time()
            for rgb_input, flow_input, target in tqdm(dataloader, desc='Evaluation:', leave=False):
                rgb_input, flow_input, target = rgb_input.to(device), flow_input.to(device), target.to(device)
                out_dict = model(rgb_input, flow_input)
                pred_logit = out_dict['logits']
                prob_val = pred_logit.squeeze().cpu().numpy()
                target_batch = target.squeeze().cpu().numpy()
                pred_scores += list(prob_val) 
                gt_targets += list(target_batch)
            end = time.time()
            num_frames = len(gt_targets)
            result = self.eval_method(pred_scores, gt_targets, self.all_class_names, self.data_processing, self.metric)
            time_taken = end - start
            logger.info(f'Processed {num_frames} frames in {time_taken:.1f} seconds ({num_frames / time_taken :.1f} FPS)')
        return result['mean_AP']
    
    def forward(self, model, dataloader, logger):
        return self.eval(model, dataloader, logger)

@EVAL.register("ANTICIPATION")
class ANT_Evaluate(nn.Module):
    
    def __init__(self, cfg):
        super(ANT_Evaluate, self).__init__()
        data_name = cfg["data_name"].split('_')[0]
        self.data_processing = thumos_postprocessing if data_name == 'THUMOS' else None
        self.metric = cfg['metric']
        self.eval_method = perframe_average_precision
        self.all_class_names = json.load(open(cfg['video_list_path']))[data_name]['class_index']
    
    def eval(self, model, dataloader, logger):
        device = "cuda:0"
        model.eval()   
        with torch.no_grad():
            pred_scores, gt_targets, ant_pred_scores, ant_gt_targets = [], [], [], []
            start = time.time()
            anticipation_mAPs = []
            for rgb_input, flow_input, target, ant_target in tqdm(dataloader, desc='Evaluation:', leave=False):
                rgb_input, flow_input, target, ant_target = rgb_input.to(device), flow_input.to(device), target.to(device), ant_target.to(device)
                out_dict = model(rgb_input, flow_input)
                pred_logit = out_dict['logits']
                ant_pred_logit = out_dict['anticipation_logits']
                prob_val = pred_logit.squeeze().cpu().numpy()
                target_batch = target.squeeze().cpu().numpy()
                ant_prob_val = ant_pred_logit.squeeze().cpu().numpy()
                ant_target_batch = ant_target.squeeze().cpu().numpy()
                pred_scores += list(prob_val)  
                gt_targets += list(target_batch)
                ant_pred_scores += list(ant_prob_val)
                ant_gt_targets += list(ant_target_batch)      
            end = time.time()
            num_frames = len(gt_targets)
            result = self.eval_method(pred_scores, gt_targets, self.all_class_names, self.data_processing, self.metric)
            ant_pred_scores = np.array(ant_pred_scores)
            ant_gt_targets = np.array(ant_gt_targets)
            logger.info(f'OAD mAP: {result["mean_AP"]*100:.2f}')
            for step in range(ant_gt_targets.shape[1]):
                result[f'anticipation_{step+1}'] = self.eval_method(ant_pred_scores[:,step,:], ant_gt_targets[:,step,:], self.all_class_names, self.data_processing, self.metric)
                anticipation_mAPs.append(result[f'anticipation_{step+1}']['mean_AP'])
                logger.info(f"Anticipation at step {step+1}: {result[f'anticipation_{step+1}']['mean_AP']*100:.2f}")
            logger.info(f'Mean Anticipation mAP: {np.mean(anticipation_mAPs)*100:.2f}')
            
            time_taken = end - start
            logger.info(f'Processed {num_frames} frames in {time_taken:.1f} seconds ({num_frames / time_taken :.1f} FPS)')
            
        return np.mean(anticipation_mAPs)
    
    def forward(self, model, dataloader, logger):
        return self.eval(model, dataloader, logger)
    