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
        self.num_classes = cfg['num_classes']
        self.bg_class_idx = cfg.get('bg_class_idx', 0)
        self.k_neighbors = 50 
        self.calc_batch_size = 1024 
        
        # [MỚI] Tỉ lệ mẫu muốn đo CAC (5%)
        self.sample_ratio = 0.05

    def __call__(self, val_loader, model, criterion, epoch, writer=None):
        model.eval()
        total_loss = 0.0
        
        all_features = []
        all_targets = [] # Lưu vector multi-hot

        with torch.no_grad():
            # --- BƯỚC 1: THU THẬP FEATURE (Chạy hết để tính Loss chính xác) ---
            pbar = tqdm(val_loader, desc=f"Epoch:{epoch} Validating")
            for batch_data in pbar:
                rgb_anchor = batch_data['rgb_anchor'].cuda(non_blocking=True)
                flow_anchor = batch_data['flow_anchor'].cuda(non_blocking=True)
                labels = batch_data['labels'].cuda(non_blocking=True)
                
                # [MỚI] Lấy targets_multihot
                targets_multihot = batch_data['targets_multihot'].cuda(non_blocking=True)
                
                labels_per_frame = batch_data.get('labels_per_frame', None)
                if labels_per_frame is not None:
                    labels_per_frame = labels_per_frame.cuda(non_blocking=True)

                out_dict = model(rgb_anchor, flow_anchor, labels, labels_per_frame=labels_per_frame)
                
                # Tính Loss chuẩn trên toàn bộ tập Valid
                loss = criterion(out_dict, targets_multihot)
                total_loss += loss.item()

                # Lưu lại Feature Teacher và Target Multi-hot
                k_cls = out_dict['k_cls'].detach().cpu() 
                targets_cpu = targets_multihot.detach().cpu()
                
                all_features.append(k_cls)
                all_targets.append(targets_cpu)

        # Gộp thành Tensor lớn
        all_features = torch.cat(all_features, dim=0) # [N, D]
        all_targets = torch.cat(all_targets, dim=0)   # [N, C]
        
        n_total = all_features.shape[0]
        avg_loss = total_loss / len(val_loader)

        # --- BƯỚC 2: SAMPLING 5% DATASET ---
        n_subset = int(n_total * self.sample_ratio)
        if n_subset < 100: n_subset = n_total # Safety check cho dataset quá nhỏ
            
        print(f"--> [Val] Sampling {self.sample_ratio*100}% data ({n_subset}/{n_total}) for CAC...")
        
        # Lấy index ngẫu nhiên
        perm_indices = torch.randperm(n_total)[:n_subset]
        
        feat_subset = all_features[perm_indices]
        target_subset = all_targets[perm_indices]

        # --- BƯỚC 3: TÍNH CAC TRÊN SUBSET ---
        cac_score = self._calculate_cac_multilabel(feat_subset, target_subset)

        print(f"--> [Val Epoch {epoch}] Loss: {avg_loss:.4f} | CAC (Action, 5% Sample): {cac_score:.2f}%")

        if writer is not None:
            writer.add_scalar("Val/Loss", avg_loss, epoch)
            writer.add_scalar("Val/CAC_Action", cac_score, epoch)

        return cac_score

    def _calculate_cac_multilabel(self, features, targets):
        """
        Tính CAC Multi-label. (Logic y hệt bài trước)
        """
        # 1. Lọc bỏ Background
        is_bg = targets[:, self.bg_class_idx] > 0.5
        is_action = ~is_bg
        action_indices = torch.where(is_action)[0]
        
        if len(action_indices) == 0: return 0.0

        act_feats = features[action_indices]
        act_targets = targets[action_indices]
        
        num_actions = act_feats.shape[0]
        correct_counts = 0
        total_neighbors = 0
        
        # Chuyển lên GPU để tính toán
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        act_feats = act_feats.to(device)
        act_targets = act_targets.to(device)

        # Chunking loop
        for i in range(0, num_actions, self.calc_batch_size):
            end = min(i + self.calc_batch_size, num_actions)
            query_feat = act_feats[i:end]
            query_target = act_targets[i:end]
            
            # Cosine Sim
            sim_matrix = torch.mm(query_feat, act_feats.T)
            
            # Top-K
            _, topk_indices = torch.topk(sim_matrix, k=self.k_neighbors + 1, dim=1)
            neighbor_indices = topk_indices[:, 1:] 
            
            # Check Overlap
            neighbor_targets = act_targets[neighbor_indices]
            q_target_expanded = query_target.unsqueeze(1)
            overlap = (q_target_expanded * neighbor_targets).sum(dim=-1)
            
            is_correct = (overlap > 0).float()
            correct_counts += is_correct.sum().item()
            total_neighbors += (is_correct.numel())
            
        if device == 'cuda':
            del act_feats, act_targets
            torch.cuda.empty_cache()
            
        return (correct_counts / total_neighbors) * 100.0

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
    