import torch.nn as nn
import torch.nn.functional as F
import torch

class MGDLoss3D(nn.Module):
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 alpha_mgd=1.0,
                 lambda_mgd=0.30, # [FIX 1] Giảm xuống 0.30 (chỉ che 30%) để Student dễ học hơn
                 ):
        super(MGDLoss3D, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
     
        # [FIX 2] LUÔN LUÔN tạo align layer để chuyển đổi miền (Domain Adaptation)
        # Dù kênh bằng nhau (192), ta vẫn cần lớp này để map RGB Space -> Flow Space
        self.align = nn.Conv3d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)

        # Generator: R(2+1)D Block
        self.generation = nn.Sequential(
            nn.Conv3d(teacher_channels, teacher_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(teacher_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(teacher_channels, teacher_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(teacher_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(teacher_channels, teacher_channels, kernel_size=1, bias=True)
        )

    def forward(self, preds_S, preds_T):
        """
        Args:
            preds_S (Tensor): Student Feature (B, C, T, H, W)
            preds_T (Tensor): Teacher Feature (B, C, T, H, W)
        """
        assert preds_S.shape[-3:] == preds_T.shape[-3:], \
            f"Shape mismatch: Student {preds_S.shape} vs Teacher {preds_T.shape}"

        # [QUAN TRỌNG] Luôn Align trước khi Mask
        preds_S = self.align(preds_S)
     
        # Tính Loss
        loss = self.get_dis_loss(preds_S, preds_T) * self.alpha_mgd
        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='mean')
        N, C, T, H, W = preds_T.shape
        device = preds_S.device

        # Teacher luôn cần detach để không tính gradient ngược lại nó
        preds_T = preds_T.detach()

        # --- Mask Generation ---
        # Giữ nguyên Time, giảm không gian H, W để tạo mask dạng khối (Blocky Mask)
        grid_T = T  
        grid_H = max(1, H // 2)
        grid_W = max(1, W // 2)
        
        mask_small = torch.rand((N, 1, grid_T, grid_H, grid_W)).to(device)
        mask = F.interpolate(mask_small, size=(T, H, W), mode='nearest')
        
        # Masking Logic:
        # mask < lambda -> 0 (Bị che)
        # mask >= lambda -> 1 (Giữ lại)
        # Với lambda=0.3, ta chỉ che 30%, giữ lại 70% -> Student dễ thở hơn nhiều
        mask = torch.where(mask < self.lambda_mgd, 0.0, 1.0)

        # Che Feature
        masked_fea = preds_S * mask
        
        # Tái tạo
        new_fea = self.generation(masked_fea)

        # Loss: So sánh bản tái tạo với Teacher gốc
        dis_loss = loss_mse(new_fea, preds_T)

        return dis_loss