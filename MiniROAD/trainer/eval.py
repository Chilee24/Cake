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
        self.num_classes = cfg.get('num_classes', 20)
        self.bg_class_idx = cfg.get('bg_class_idx', 0)
        
        # Top 5% số lượng mẫu
        self.top_k_ratio = 0.05 
        self.calc_batch_size = 1024 

        # [CONFIG MỚI]
        # True: Chỉ dùng Action để làm Query (hỏi).
        # False: Dùng cả Nền để làm Query.
        # Lưu ý: Dù True hay False thì "Hàng xóm" luôn tìm trong TOÀN BỘ tập dữ liệu.
        self.eval_action_only = cfg.get('eval_action_only', True)

    def __call__(self, val_loader, model, criterion, epoch, writer=None):
        model.eval()
        total_loss = 0.0
        
        all_features = []
        all_targets = [] 

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch:{epoch} Validating CAC")
            for batch_data in pbar:
                # 1. Move data
                rgb_anchor = batch_data['rgb_anchor'].cuda(non_blocking=True)
                flow_anchor = batch_data['flow_anchor'].cuda(non_blocking=True)
                labels = batch_data['labels'].cuda(non_blocking=True)
                targets_multihot = batch_data['targets_multihot'].cuda(non_blocking=True)
                
                labels_per_frame = batch_data.get('labels_per_frame', None)
                if labels_per_frame is not None:
                    labels_per_frame = labels_per_frame.cuda(non_blocking=True)

                # 2. Forward
                out_dict = model(rgb_anchor, flow_anchor, labels, labels_per_frame=labels_per_frame)
                
                # 3. Loss
                loss = criterion(out_dict, targets_multihot)
                total_loss += loss.item()

                # 4. Collect
                k_cls = out_dict['k_cls'].detach().cpu() 
                targets_cpu = targets_multihot.detach().cpu()
                
                all_features.append(k_cls)
                all_targets.append(targets_cpu)

        # Gộp Tensor (Đây là GALLERY - Kho tìm kiếm)
        gallery_features = torch.cat(all_features, dim=0) # [Total, D]
        gallery_targets = torch.cat(all_targets, dim=0)   # [Total, C]
        
        avg_loss = total_loss / len(val_loader)
        
        # --- BƯỚC CHUẨN BỊ QUERY ---
        if self.eval_action_only:
            # Lọc: Chỉ lấy Action làm Query
            is_bg = gallery_targets[:, self.bg_class_idx] > 0.5
            query_features = gallery_features[~is_bg]
            query_targets = gallery_targets[~is_bg]
            mode_str = "Action Only"
        else:
            # Lấy tất cả làm Query
            query_features = gallery_features
            query_targets = gallery_targets
            mode_str = "All Samples"

        num_queries = query_features.shape[0]
        num_gallery = gallery_features.shape[0]

        print(f"\n--> [Val Info] Queries: {num_queries} ({mode_str}) | Search Space: {num_gallery} (All)")

        # --- TÍNH CAC ---
        cac_score = 0.0
        if num_queries > 0:
            # Truyền cả Query và Gallery riêng biệt
            cac_score = self._calculate_cac_matrix(
                query_features, query_targets, 
                gallery_features, gallery_targets
            )

        print(f"--> [Val Epoch {epoch}] Loss: {avg_loss:.4f} | CAC ({mode_str} -> Global): {cac_score:.2f}%")

        if writer is not None:
            writer.add_scalar("Val/Loss", avg_loss, epoch)
            writer.add_scalar(f"Val/CAC_{mode_str.replace(' ', '_')}", cac_score, epoch)

        return cac_score

    def _calculate_cac_matrix(self, q_feats, q_targets, g_feats, g_targets):
        """
        Tính CAC bất đối xứng: 
        - Query: Tập con (ví dụ chỉ Action)
        - Gallery: Tập toàn bộ (bao gồm cả Nền)
        """
        num_queries = q_feats.shape[0]
        num_gallery = g_feats.shape[0]
        
        # Tính K dựa trên GALLERY size (Không gian tìm kiếm)
        # Vì chúng ta muốn tìm Top 5% trong đám đông hỗn loạn
        k_dynamic = int(num_gallery * self.top_k_ratio)
        k_dynamic = max(1, k_dynamic)
        k_dynamic = min(k_dynamic, num_gallery - 1)
        
        print(f"--> [CAC Logic] Finding Top-{k_dynamic} neighbors in Global Space ({num_gallery} samples).")

        correct_counts = 0
        total_neighbors_checked = 0
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Normalize cả 2
        q_feats = F.normalize(q_feats, p=2, dim=1).to(device)
        g_feats = F.normalize(g_feats, p=2, dim=1).to(device) # Gallery lên GPU
        
        q_targets = q_targets.to(device)
        g_targets = g_targets.to(device)

        # Chunking loop cho QUERY
        for i in range(0, num_queries, self.calc_batch_size):
            end = min(i + self.calc_batch_size, num_queries)
            
            # Batch Query hiện tại
            curr_q_feat = q_feats[i:end]      # [Batch, D]
            curr_q_target = q_targets[i:end]  # [Batch, C]
            
            # 1. Cosine Sim: Query (Batch) vs Gallery (ALL)
            # [Batch, N_Gallery]
            sim_matrix = torch.mm(curr_q_feat, g_feats.T)
            
            # 2. Tìm K+1 người gần nhất
            # Lưu ý: Vì Query là tập con của Gallery, nên mẫu gần nhất (Top-1)
            # CHẮC CHẮN là chính nó (Sim ~ 1.0). Ta vẫn lấy K+1 và bỏ thằng đầu tiên.
            _, topk_indices = torch.topk(sim_matrix, k=k_dynamic + 1, dim=1)
            
            # Bỏ cột đầu tiên (Self-match)
            neighbor_indices = topk_indices[:, 1:] # [Batch, K_dynamic]
            
            # 3. Kiểm tra nhãn hàng xóm
            # Lấy nhãn từ GALLERY
            neighbor_targets = g_targets[neighbor_indices] # [Batch, K_dynamic, C]
            
            curr_q_target_expanded = curr_q_target.unsqueeze(1) # [Batch, 1, C]
            
            # Overlap Logic
            overlap = (curr_q_target_expanded * neighbor_targets).sum(dim=-1)
            is_correct = (overlap > 0).float()
            
            correct_counts += is_correct.sum().item()
            total_neighbors_checked += is_correct.numel()
            
        if device == 'cuda':
            del q_feats, g_feats, q_targets, g_targets
            torch.cuda.empty_cache()
            
        if total_neighbors_checked == 0: return 0.0
        return (correct_counts / total_neighbors_checked) * 100.0

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
    