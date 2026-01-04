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
        self.k_neighbors_ratio = 0.05 
        self.calc_batch_size = 512 

    def __call__(self, val_loader, model, criterion, epoch, writer=None):
        model.eval()
        total_loss = 0.0
        
        # Chỉ lưu Feature và Label trên CPU (nhẹ hơn nhiều so với ma trận hàng xóm)
        all_features = []
        all_labels = []

        with torch.no_grad():
            # --- BƯỚC 1: THU THẬP FEATURE ---
            pbar = tqdm(val_loader, desc=f"Epoch:{epoch} Validating")
            for batch_data in pbar:
                # 1. Move inputs to GPU
                rgb_anchor = batch_data['rgb_anchor'].cuda(non_blocking=True)
                flow_anchor = batch_data['flow_anchor'].cuda(non_blocking=True)
                labels = batch_data['labels'].cuda(non_blocking=True)
                
                # [QUAN TRỌNG] Lấy thêm labels_per_frame để khớp signature model
                labels_per_frame = batch_data.get('labels_per_frame', None)
                if labels_per_frame is not None:
                    labels_per_frame = labels_per_frame.cuda(non_blocking=True)

                # 2. Forward
                # Model hiện tại chỉ nhận 4 tham số này
                out_dict = model(rgb_anchor, flow_anchor, labels, labels_per_frame=labels_per_frame)
                
                # Tính val loss (để tham khảo)
                loss = criterion(out_dict, labels)
                total_loss += loss.item()

                # 3. Lấy Feature Teacher (k_cls) làm chuẩn đánh giá
                # Đưa feature về CPU ngay lập tức để tiết kiệm GPU cho bước tính toán sau
                k_cls = out_dict['k_cls'].detach().cpu() 
                labels_cpu = labels.detach().cpu()
                
                all_features.append(k_cls)
                all_labels.append(labels_cpu)

        # Gộp thành Tensor lớn [N, Dim]
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        n_samples = all_features.shape[0]
        avg_loss = total_loss / len(val_loader)

        print(f"--> [Val] Calculating CAC Streaming for {n_samples} samples...")

        # --- BƯỚC 2: TÍNH CAC STREAMING ---
        cac_score = self._calculate_cac_streaming(all_features, all_labels)

        print(f"--> [Val Epoch {epoch}] Loss: {avg_loss:.4f} | CAC (Action): {cac_score:.2f}%")

        if writer is not None:
            writer.add_scalar("Val/Loss", avg_loss, epoch)
            writer.add_scalar("Val/CAC_Action", cac_score, epoch)

        return cac_score

    def _calculate_cac_streaming(self, all_features, all_labels):
        """
        Tính CAC theo kiểu 'Streaming': Tính đến đâu cộng dồn đến đó, 
        không bao giờ lưu toàn bộ ma trận khoảng cách N*N.
        """
        n_samples = all_features.shape[0]
        k = max(1, int(n_samples * self.k_neighbors_ratio))
        k_search = k + 1 # +1 vì topk bao gồm chính nó (khoảng cách = 0)
        
        # Dictionary để cộng dồn kết quả cho từng class
        class_stats = {c: {'sum': 0.0, 'count': 0} for c in range(self.num_classes)}
        
        # Move Reference lên GPU (Toàn bộ dataset làm mốc so sánh)
        # 157k vector 128-dim ~ 80MB VRAM -> Rất nhẹ, để trên GPU tính cho nhanh
        ref_features = all_features.cuda()
        ref_labels = all_labels.cuda()

        # Chia nhỏ Query để xử lý (Chunking) để tránh OOM khi tạo ma trận dist [Batch, N]
        num_chunks = (n_samples + self.calc_batch_size - 1) // self.calc_batch_size
        
        for i in tqdm(range(0, n_samples, self.calc_batch_size), total=num_chunks, desc="CAC Streaming"):
            # 1. Lấy Batch Query
            end = min(i + self.calc_batch_size, n_samples)
            
            # Query Features [Batch, Dim]
            query_chunk = all_features[i:end].cuda()
            # Query Labels [Batch]
            query_labels_chunk = all_labels[i:end].cuda()
            
            # 2. Tính khoảng cách & Top-K
            # [Batch, N] -> [Batch, k_search]
            # torch.cdist rất tối ưu trên GPU
            dists = torch.cdist(query_chunk, ref_features, p=2)
            _, topk_indices = torch.topk(dists, k=k_search, dim=1, largest=False)
            
            # 3. Lấy nhãn hàng xóm
            # [Batch, k_search]
            neighbor_labels = ref_labels[topk_indices]
            
            # Bỏ cột đầu tiên (chính là bản thân nó, dist=0)
            neighbor_labels = neighbor_labels[:, 1:] # [Batch, k]
            
            # 4. TÍNH CONSISTENCY (Vectorized)
            # Kiểm tra xem hàng xóm có cùng nhãn với Query không
            matches = (neighbor_labels == query_labels_chunk.unsqueeze(1))
            
            # Tính trung bình consistency cho từng mẫu trong batch -> [Batch]
            consistency_scores = matches.float().mean(dim=1)
            
            # 5. CỘNG DỒN VÀO CLASS STATS (Về CPU để xử lý logic python)
            consistency_scores = consistency_scores.cpu().numpy()
            query_labels_np = query_labels_chunk.cpu().numpy()
            
            unique_classes = np.unique(query_labels_np)
            for c in unique_classes:
                if c == self.bg_class_idx: continue # Bỏ qua BG
                
                # Mask lọc ra các mẫu thuộc class c trong batch này
                mask = (query_labels_np == c)
                
                class_stats[c]['sum'] += consistency_scores[mask].sum()
                class_stats[c]['count'] += mask.sum()
            
            # Clean up GPU memory cho batch này
            del dists, topk_indices, query_chunk, query_labels_chunk, neighbor_labels, matches

        # --- TÍNH TỔNG KẾT ---
        cac_per_class = []
        for c in range(self.num_classes):
            if c == self.bg_class_idx: continue
            
            if class_stats[c]['count'] > 0:
                avg_cac = class_stats[c]['sum'] / class_stats[c]['count']
                cac_per_class.append(avg_cac)
        
        if len(cac_per_class) == 0: return 0.0
        return np.mean(cac_per_class) * 100.0

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
    