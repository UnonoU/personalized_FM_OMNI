import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from datetime import datetime
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import json
import os
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from src.utils import *

from src.cbramod import CBraMod
import torch
import torch.nn as nn
import torch.nn.functional as F
# from BIOT.utils import BIOT_DataProjector

class ReconstructionHead_channel(nn.Module):
    """
    Enhanced Reconstruction Head with:
    - Larger capacity (hidden_dim=1024)
    - Deeper network (4 layers)
    - Residual connections
    - Layer normalization
    - No dropout (or minimal dropout=0.0)
    """

    def __init__(self, embed_dim=200, patch_size=200, hidden_dim=1024, dropout=0.0):
        super().__init__()

        # Larger capacity
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, patch_size)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        # Minimal or no dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        Forward pass:
        x: Input tensor of shape (B, C, A, D)
        """
        B, C, A, D = x.shape

        # Reshape input from (B, C, A, D) to (B * C * A, D) for processing
        x_reshaped = x.view(B * C * A, D)  # Flatten the first three dimensions
        
        # Apply the layers
        h1 = self.dropout(F.gelu(self.ln1(self.fc1(x_reshaped))))
        h2 = self.dropout(F.gelu(self.ln2(self.fc2(h1))))
        h3 = self.dropout(F.gelu(self.ln3(self.fc3(h2))))

        # Output layer
        out = self.fc_out(h3)

        # Reshape the output back to (B, C, A, D)
        out = out.view(B, C, A, D)

        return out

class PersonalizedReconstructionHeadSimple(nn.Module):
    """
    简化版的个体化适配解码器，适应 LoRA 微调
    通过简单的线性层减少重建头的参数量
    """
    def __init__(self, input_dim=200, num_channels=61, hidden_dim=16, time_steps=200):
        super().__init__()

        self.num_channels = num_channels
        self.hidden_dim = hidden_dim

        # 通道级别编码器：每个通道通过独立的线性层进行编码
        self.channel_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ) for _ in range(num_channels)
        ])

        # 使用共享解码器来重建每个通道的信号
        self.channel_decoder = nn.Linear(hidden_dim, time_steps)

    def forward(self, x):
        """
        输入：嵌入表示（b, c, a, d），输出：增强的通道特征
        """
        batch_size, channels, num_patches, feat_dim = x.shape
        
        # 1. 每个通道的独立编码
        channel_features = []
        for c in range(channels):
            channel_feat = x[:, c, :, :]  # (b, a, d)
            encoded = self.channel_encoders[c](channel_feat.reshape(-1, feat_dim))  # (b*a, h)
            channel_features.append(encoded.view(batch_size, num_patches, -1))  # (b, a, h)
        
        # 2. 通过共享解码器重建每个通道的信号
        outputs = []
        for c in range(channels):
            time_seq = self.channel_decoder(channel_features[c].reshape(-1, channel_features[c].shape[-1]))  # (b*a, t)
            outputs.append(time_seq.view(batch_size, 1, num_patches, -1))  # (b, 1, a, t)
        
        # 合并所有通道的输出
        output = torch.cat(outputs, dim=1)  # (b, c, a, t)

        return output , x


class ReconstructionHead(nn.Module):
    """
    Enhanced Reconstruction Head with:
    - Larger capacity (hidden_dim=1024)
    - Deeper network (4 layers)
    - Residual connections
    - Layer normalization
    - No dropout (or minimal dropout=0.0)
    """
    def __init__(self, embed_dim=200, patch_size=200, hidden_dim=1024, dropout=0.0):
        super().__init__()

        # Larger capacity
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, patch_size)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        # Minimal or no dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h1 = self.dropout(F.gelu(self.ln1(self.fc1(x))))
        h2 = self.dropout(F.gelu(self.ln2(self.fc2(h1))))
        h3 = self.dropout(F.gelu(self.ln3(self.fc3(h2))))

        out = self.fc_out(h3)
        return out
    
    
class Embedding_MaskingStrategy:
    """
    Implements various masking strategies for EEG data (batch, channel, patch, feature).

    Supports:
    - Temporal patch masking: Mask contiguous patches in the patch dimension
    - Channel masking: Mask entire channels
    - Random masking: Random point-wise masking
    - Hybrid masking: Combination of temporal and channel masking
    """

    def __init__(self,
                 mask_ratio: float = 0.4,
                 mask_type: str = 'patch'):
        """
        Args:
            mask_ratio: Proportion of data to mask (0.0 to 1.0)
            mask_type: Type of masking ('temporal_patch', 'channel', 'random', 'hybrid')
            patch_size: Size of temporal patches for patch-based masking
            min_mask_len: Minimum length of masked segments
            max_mask_len: Maximum length of masked segments
        """
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type

    def __call__(self, x: torch.Tensor) -> tuple:
        """
        Apply masking to input EEG data.

        Args:
            x: Input tensor of shape (batch, channels, patches, features)

        Returns:
            masked_x: Masked input (masked regions set to 0)
            mask: Binary mask (1 = masked, 0 = visible)
            original_x: Original unmasked input
        """
        if self.mask_type == 'patch':
            return self._temporal_patch_mask(x)
        elif self.mask_type == 'channel':
            return self._channel_mask(x)
        elif self.mask_type == 'random':
            return self._random_mask(x)
        elif self.mask_type == 'hybrid':
            return self._hybrid_mask(x)
        elif self.mask_type == 'non_mask':
            return x, 0
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")

    def _temporal_patch_mask(self, x: torch.Tensor) -> tuple:
        """Mask contiguous temporal patches (in the patch dimension)."""
        batch, channels, patches, features = x.shape
        mask = torch.zeros_like(x)

        # Calculate number of patches
        n_patches = patches
        n_mask = int(n_patches * self.mask_ratio)

        for b in range(batch):
            # Randomly select patches to mask (mask entire patches, not individual time points)
            mask_indices = torch.randperm(n_patches)[:n_mask]
            for idx in mask_indices:
                start = idx
                end = min(start + 1, patches)  # mask whole patch (a patch is one unit)
                mask[b, :, start:end, :] = 1  # Mask the whole patch

        masked_x = x * (1 - mask)
        return masked_x, mask

    def _channel_mask(self, x: torch.Tensor) -> tuple:
        """Mask entire channels."""
        batch, channels, patches, features = x.shape
        mask = torch.zeros_like(x)

        n_mask_channels = int(channels * self.mask_ratio)

        for b in range(batch):
            mask_channels = torch.randperm(channels)[:n_mask_channels]
            mask[b, mask_channels, :, :] = 1

        masked_x = x * (1 - mask)
        return masked_x, mask

    def _random_mask(self, x: torch.Tensor) -> tuple:
        """Random point-wise masking."""
        batch, channels, patches, features = x.shape
        mask = (torch.rand_like(x) < self.mask_ratio).float()
        masked_x = x * (1 - mask)
        return masked_x, mask
    
    def _hybrid_mask(self, x: torch.Tensor) -> tuple:
        """Combine temporal and channel masking."""
        batch, channels, patches, features = x.shape
        mask = torch.zeros_like(x)

        # First apply channel masking (half the mask ratio)
        n_mask_channels = int(channels * self.mask_ratio / 2)
        for b in range(batch):
            mask_channels = torch.randperm(channels)[:n_mask_channels]
            mask[b, mask_channels, :, :] = 1

        # Then apply temporal masking on remaining channels
        n_mask_patches = int(patches * self.mask_ratio / 2)

        for b in range(batch):
            mask_indices = torch.randperm(patches)[:n_mask_patches]
            for idx in mask_indices:
                start = idx
                end = min(start + 1, patches)  # mask entire patch
                # Only mask features that aren't already fully masked
                visible_channels = (mask[b, :, 0, 0] == 0).nonzero(as_tuple=True)[0]
                if len(visible_channels) > 0:
                    mask[b, visible_channels, start:end, :] = 1

        masked_x = x * (1 - mask)
        return masked_x, mask


class CausalConv1d(nn.Module):
    """Causal 1D Convolution for temporal feature extraction"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
            groups=groups
        )

    def forward(self, x):
        x = self.conv(x)
        # Remove future padding
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x
    
class ResEEGNet(nn.Module):
    """
    Residual EEG Network
    """
    def __init__(self, num_channels=61, num_patches=10, patch_size=200,
                 embed_dim=200, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.temporal_encoder = nn.Sequential(
            CausalConv1d(num_channels, num_channels, kernel_size=3, dilation=1, groups=num_channels),
            nn.BatchNorm1d(num_channels),
            nn.GELU()
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(patch_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        """
        Input: (B, C, A, T) -> (Batch, 61, 10, 200)
        """
        B, C, A, T = x.shape

        x_reshaped = rearrange(x, 'b c a t -> (b a) c t')

        x_temporal = self.temporal_encoder(x_reshaped)  

        x_mlp = self.mlp_head(x_temporal)  # (B*A, Hidden, embed_dim)

        residual_features = rearrange(x_mlp, '(b a) c d -> b c a d', b=B, a=A)
        return residual_features
    
class EncoderWithResidual(nn.Module):
    """
    新的 Encoder 只包含 Encoder 和 ResNet 部分
    """
    def __init__(self, encoder, residual_net, in_chans=None,embed_dim=200,return_patch_tokens=True,return_all_tokens=False):
        super().__init__()
        self.encoder = encoder
        self.residual_net = residual_net
        self.in_chans = in_chans
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.return_patch_tokens=return_patch_tokens
        self.return_all_tokens=return_all_tokens
        
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: (B, C, A, T)
        Returns:
            fused_features: (B*C, A, D)
        """
        B, C, A, T = x.shape
        x_reshaped = x

        # Encoder forward
        features = self.encoder.forward_features(x_reshaped, input_chans=self.in_chans, 
                                                 return_patch_tokens=self.return_patch_tokens, return_all_tokens=self.return_all_tokens)  # (B, C*A, D)

        if features.ndim == 3 and features.shape[1] == A + 1:
            features = features[:, 1:, :]  # (B*C, A, D)
        features = rearrange(features, 'b (c a) t -> b c a t', c=C, a=A, t=T)

        # Residual network forward
        residual_features = self.residual_net(x)

        # Feature fusion
        fused_features = features + residual_features  # (B*C, A, D)
        fused_features = self.fusion_norm(fused_features)
        
        return fused_features
    

def calculate_fc(signal, epsilon=1e-8):
    """
    计算输入信号的功能连接矩阵。
    输入：signal (b, c, a, t)
    输出：功能连接矩阵，大小为 (b, c, c)
    """
    batch_size, channels, num_patches, time_steps = signal.shape
    
    # 对每个batch进行计算
    fc_matrix = []
    for b in range(batch_size):
        # 对每个通道计算皮尔逊相关系数
        # 转换为numpy数组并重新调整维度 (c, t)
        signal_b = signal[b].cpu().numpy().reshape(channels, -1)
        
        # 计算每个通道的皮尔逊相关系数
        corr_matrix = np.corrcoef(signal_b)
        
        # 处理0标准差的情况：如果信号的标准差为0，设相关系数为0
        # 如果信号没有变化（标准差为零），那么相关性为0
        corr_matrix[np.isnan(corr_matrix)] = 0  # 替换NaN为0（标准差为0时产生NaN）
        corr_matrix[np.isinf(corr_matrix)] = 0  # 替换无穷大为0（防止出现极大值）

        fc_matrix.append(corr_matrix)
    
    return torch.tensor(np.array(fc_matrix), dtype=torch.float32).to(signal.device)

def evaluate_reconstruction_enhanced_val(model, dataloader, criterion, device, use_amp=False, keep_channel=False):
    """
    Enhanced evaluation with std_ratio and high_freq_ratio monitoring

    Returns:
        avg_loss: Average loss
        r2: R² score (clipped at -1.0)
        pcc: Pearson Correlation Coefficient
        std_ratio: Std deviation ratio (target: >0.8)
        high_freq_ratio: High-frequency preservation ratio (target: >0.7)
        loss_components: Dict with individual loss components
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    # all_loss_components = {'mse': [], 'variance': [], 'spectral': []}
    all_loss_components = {'mse': [], 'correlation': []}
    all_fc_similarity = []

    with torch.no_grad():
        for batch in dataloader:
            eeg = batch['eeg'].to(device)
            if keep_channel:
                B, C, A, T = eeg.shape
                outputs, mask, embedding  = model(eeg)
                target = eeg
            else:
                outputs, mask = model(eeg)
                B, C, A, T = eeg.shape
                target = eeg.reshape(B * C, A, T)
                
            mask = mask.to(device)    
            loss, loss_dict = criterion(outputs, target, mask, embedding)
            total_loss += loss.item()

            # Track loss components
            for key in all_loss_components.keys():
                all_loss_components[key].append(loss_dict[key])

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            
            # 计算 FC 相似性
            predicted_fc = calculate_fc(outputs)  # (b, c, c) - predicted FC
            target_fc = calculate_fc(target)  # (b, c, c) - target FC
            
            # 计算 FC 的相关性（例如：平均皮尔逊相关系数）
            fc_similarity = torch.cosine_similarity(predicted_fc.view(-1, C * C), target_fc.view(-1, C * C), dim=-1)
            all_fc_similarity.append(fc_similarity.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)  # (N*C, A, T)
    targets = np.concatenate(all_targets, axis=0)  # (N*C, A, T)
    
    
    # Calculate global metrics
    # predictions_flat = predictions.flatten()
    # targets_flat = targets.flatten()

    # r2 = r2_score(targets_flat, predictions_flat)
    # r2 = max(r2, -1.0)
    # pcc, _ = pearsonr(targets_flat, predictions_flat)

    predictions_flat = predictions[mask.bool().cpu()].flatten()
    targets_flat = targets[mask.bool().cpu()].flatten()

    r2 = r2_score(targets_flat, predictions_flat)
    pcc, _ = pearsonr(targets_flat, predictions_flat)

    # V2-Fixed: Additional monitoring metrics
    pred_std = predictions.std(axis=-1).mean()
    target_std = targets.std(axis=-1).mean()
    std_ratio = pred_std / (target_std + 1e-8)

    # High-frequency preservation (Handle diff length)
    pred_diff = np.diff(predictions, axis=-1)
    target_diff = np.diff(targets, axis=-1)
    high_freq_ratio = pred_diff.std() / (target_diff.std() + 1e-8)

    avg_loss = total_loss / len(dataloader)

    # Average loss components
    avg_loss_components = {k: np.mean(v) for k, v in all_loss_components.items()}
    
    avg_fc_similarity = np.mean(all_fc_similarity)

    return avg_loss, r2, pcc, std_ratio, high_freq_ratio, avg_loss_components,avg_fc_similarity


def evaluate_reconstruction_enhanced(model, dataloader, criterion, device, use_amp=False, keep_channel=False):
    """
    Enhanced evaluation with std_ratio and high_freq_ratio monitoring

    Returns:
        avg_loss: Average loss
        r2: R² score (clipped at -1.0)
        pcc: Pearson Correlation Coefficient
        std_ratio: Std deviation ratio (target: >0.8)
        high_freq_ratio: High-frequency preservation ratio (target: >0.7)
        loss_components: Dict with individual loss components
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    # all_loss_components = {'mse': [], 'variance': [], 'spectral': []}
    all_loss_components = {'mse': [], 'correlation': []}


    with torch.no_grad():
        for batch in dataloader:
            eeg = batch['eeg'].to(device)
            if keep_channel:
                B, C, A, T = eeg.shape
                outputs, mask,embedding  = model(eeg)
                target = eeg
            else:
                outputs, mask = model(eeg)
                B, C, A, T = eeg.shape
                target = eeg.reshape(B * C, A, T)
                
            mask = mask.to(device)    
            loss, loss_dict = criterion(outputs, target, mask)
            total_loss += loss.item()

            # Track loss components
            for key in all_loss_components.keys():
                all_loss_components[key].append(loss_dict[key])

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)  # (N*C, A, T)
    targets = np.concatenate(all_targets, axis=0)  # (N*C, A, T)
    
    
    # Calculate global metrics
    # predictions_flat = predictions.flatten()
    # targets_flat = targets.flatten()

    # r2 = r2_score(targets_flat, predictions_flat)
    # r2 = max(r2, -1.0)
    # pcc, _ = pearsonr(targets_flat, predictions_flat)

    predictions_flat = predictions[mask.bool().cpu()].flatten()
    targets_flat = targets[mask.bool().cpu()].flatten()

    r2 = r2_score(targets_flat, predictions_flat)
    pcc, _ = pearsonr(targets_flat, predictions_flat)

    # V2-Fixed: Additional monitoring metrics
    pred_std = predictions.std(axis=-1).mean()
    target_std = targets.std(axis=-1).mean()
    std_ratio = pred_std / (target_std + 1e-8)

    # High-frequency preservation (Handle diff length)
    pred_diff = np.diff(predictions, axis=-1)
    target_diff = np.diff(targets, axis=-1)
    high_freq_ratio = pred_diff.std() / (target_diff.std() + 1e-8)

    avg_loss = total_loss / len(dataloader)

    # Average loss components
    avg_loss_components = {k: np.mean(v) for k, v in all_loss_components.items()}
    

    return avg_loss, r2, pcc, std_ratio, high_freq_ratio, avg_loss_components

class ReconstructionLoss(nn.Module):
    """
    Mixed Loss Function to Fix Over-smoothing

    Components:
    1. MSE Loss: Basic reconstruction error
    2. Variance Loss: Penalizes std deviation mismatch (CRITICAL!)
    3. Spectral Loss: Preserves high-frequency details

    This fixes the "mean bias" problem of pure MSE loss.
    """
    def __init__(self, mse_weight=1.0, var_weight=0.5, spectral_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.var_weight = var_weight
        self.spectral_weight = spectral_weight

    def forward(self, pred, target):
        """
        Args:
            pred: (B*C, A, T) predicted signals
            target: (B*C, A, T) target signals

        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Individual loss components for logging
        """
        # 1. MSE Loss (basic reconstruction)
        mse_loss = F.mse_loss(pred, target)

        # 2. Variance Loss (CRITICAL for fixing over-smoothing!)
        # Forces std(pred) ≈ std(target)
        pred_std = pred.std(dim=-1, keepdim=True)  # (B*C, A, 1)
        target_std = target.std(dim=-1, keepdim=True)
        var_loss = torch.mean((pred_std - target_std) ** 2)

        # 3. Spectral Loss (preserves high-frequency components)
        # Uses first-order difference as a proxy for high-freq content
        pred_diff = pred[..., 1:] - pred[..., :-1]
        target_diff = target[..., 1:] - target[..., :-1]
        spectral_loss = F.mse_loss(pred_diff, target_diff)

        # Total loss
        total_loss = (self.mse_weight * mse_loss +
                     self.var_weight * var_loss +
                     self.spectral_weight * spectral_loss)

        # Return individual components for logging
        loss_dict = {
            'mse': mse_loss.item(),
            'variance': var_loss.item(),
            'spectral': spectral_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict
    
    

# class ReconstructionLoss(nn.Module):
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss_channel(nn.Module):
    def __init__(self, mse_weight=1.0, spectral_weight=0.3, correlation_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.spectral_weight = spectral_weight
        self.correlation_weight = correlation_weight

    # def compute_spectral_loss(self, x, x_hat):
    #     x = x.float()  # 确保x是float32类型
    #     X = torch.fft.fft(x, dim=-1)
    #     X_hat = torch.fft.fft(x_hat, dim=-1)
    #     spectral_loss = torch.norm(X - X_hat, p=2)
    #     return spectral_loss
    def compute_spectral_loss(self, x, x_hat):
        """
        计算频谱一致性损失，确保在进行FFT时使用float32类型。
        """
        x = x.float()  # 确保x是float32类型
        x_hat = x_hat.float()  # 确保x_hat是float32类型
        
        # 计算FFT
        X = torch.fft.fft(x, dim=-1)
        X_hat = torch.fft.fft(x_hat, dim=-1)
        
        # 计算频谱损失（L2范数）
        spectral_loss = torch.norm(X - X_hat, p=2)
        
        return spectral_loss
    
    def compute_channel_correlation_loss(self, x, x_hat,epsilon=1e-8):
        """
        计算通道相关性损失（Channel Correlation Loss），
        通过计算每个时间步的通道之间的相关性差异。
        加入稳定性的改进。
        """
        # 去均值化：计算每个时间步的均值，并进行去均值化
        x_mean = x.mean(dim=-1, keepdim=True)  # (B, C, A, 1)
        x_hat_mean = x_hat.mean(dim=-1, keepdim=True)
        
        x_centered = x - x_mean
        x_hat_centered = x_hat - x_hat_mean

        # 计算去均值后的信号的L2范数（避免除零错误）
        norm_x = torch.norm(x_centered, dim=-1, keepdim=True) + epsilon  # 防止除零
        norm_x_hat = torch.norm(x_hat_centered, dim=-1, keepdim=True) + epsilon

        # 计算余弦相似度而不是直接内积，避免溢出
        corr_matrix = torch.einsum('bcad,bcad->ab', x_centered, x_hat_centered) / (norm_x * norm_x_hat).squeeze(-1)
        
        # 计算通道相关性损失
        # 我们取 1 - 平均相关性
        channel_correlation_loss = 1 - torch.mean(corr_matrix)

        return channel_correlation_loss

    def forward(self, pred, target):
        """
        计算总损失，包括 MSE、频谱损失和通道相关性损失
        """
        # 1. MSE Loss (basic reconstruction)
        mse_loss = F.mse_loss(pred, target)

        # 2. Spectral Loss
        # spectral_loss = self.compute_spectral_loss(pred, target)

        # 3. Channel Correlation Loss
        channel_correlation_loss = self.compute_channel_correlation_loss(pred, target)

        # 计算总损失
        total_loss = (self.mse_weight * mse_loss + self.correlation_weight * channel_correlation_loss)
                      

        # 返回总损失以及各个损失的组件
        loss_dict = {
            'mse': mse_loss.item(),
            'correlation': channel_correlation_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict

    
class ReconstructionLoss_channel_MAE(nn.Module):
    def __init__(self, mse_weight=1.0, spectral_weight=0.3, correlation_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.spectral_weight = spectral_weight
        self.correlation_weight = correlation_weight

    
    def compute_channel_correlation_loss(self, x, x_hat,epsilon=1e-8):
        """
        计算通道相关性损失（Channel Correlation Loss），
        通过计算每个时间步的通道之间的相关性差异。
        加入稳定性的改进。
        """
        # 去均值化：计算每个时间步的均值，并进行去均值化
        x_mean = x.mean(dim=-1, keepdim=True)  # (B, C, A, 1)
        x_hat_mean = x_hat.mean(dim=-1, keepdim=True)
        
        x_centered = x - x_mean
        x_hat_centered = x_hat - x_hat_mean
        corr_matrix = F.cosine_similarity(x_centered, x_hat_centered, dim=-1, eps=epsilon)
        channel_correlation_loss = 1 - torch.mean(corr_matrix)

        return channel_correlation_loss
    
    def forward(self, pred, target, mask):
        """
        计算总损失，包括 MSE、频谱损失和通道相关性损失
        """
        # 1. MSE Loss (basic reconstruction)
        # mse_loss = F.mse_loss(pred * mask, target * mask, reduction='sum') / mask.sum()
        # mse_loss = F.mse_loss(pred , target ) 
        masked_mse_loss = F.mse_loss(pred * mask, target * mask, reduction='sum') / mask.sum()
        non_masked_mse_loss = F.mse_loss(pred * (1 - mask), target * (1 - mask), reduction='sum') / (1 - mask).sum()
        mse_loss = 0.8*masked_mse_loss + 0.2*non_masked_mse_loss

        
        # 3. Channel Correlation Loss
        channel_correlation_loss = self.compute_channel_correlation_loss(pred, target)

        # 计算总损失
        total_loss = (self.mse_weight * mse_loss + self.correlation_weight * channel_correlation_loss)
                    #   self.spectral_weight * spectral_loss )
        # 返回总损失以及各个损失的组件
        loss_dict = {
            'mse': mse_loss.item(),
            'correlation': channel_correlation_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict   
    
class ReconstructionLoss_channel_MAE_test(nn.Module):
    def __init__(self, mse_weight=1.0, spectral_weight=0.3, correlation_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.spectral_weight = spectral_weight
        self.correlation_weight = correlation_weight
    def compute_fc(self, signal, epsilon=1e-8):
        """
        计算输入信号的功能连接（FC）矩阵
        输入：signal (b, c, a, t)
        输出：功能连接矩阵，大小为 (b, c, c)
        """
        batch_size, channels, num_patches, time_steps = signal.shape
        
        # 对每个batch计算功能连接矩阵
        fc_matrix = []
        for b in range(batch_size):
            # 获取信号并去均值化
            signal_b = signal[b].detach().cpu().numpy().reshape(channels, -1)  # (c, t)
            signal_b_centered = signal_b - signal_b.mean(axis=-1, keepdims=True)  # 去均值

            # 计算皮尔逊相关系数
            corr_matrix = np.corrcoef(signal_b_centered)  # (c, c)
            fc_matrix.append(corr_matrix)
        
        return torch.tensor(np.array(fc_matrix), dtype=torch.float32).to(signal.device)

    def compute_FC_loss(self, x, x_hat, epsilon=1e-8):
        """
        计算通道相关性损失（Channel Correlation Loss），
        通过计算每个时间步的通道之间的相关性差异。
        这里使用皮尔逊相关系数作为功能连接（FC）的度量。
        """
        # 计算预测信号和目标信号的功能连接（FC）
        fc_pred = self.compute_fc(x_hat)
        fc_target = self.compute_fc(x)

        # 计算FC矩阵之间的损失（可以使用均方误差或余弦相似度）
        fc_loss = F.mse_loss(fc_pred, fc_target)
        
        return fc_loss
    def compute_channel_correlation_loss(self, x, x_hat,epsilon=1e-8):
        """
        计算通道相关性损失（Channel Correlation Loss），
        通过计算每个时间步的通道之间的相关性差异。
        加入稳定性的改进。
        """
        # 去均值化：计算每个时间步的均值，并进行去均值化
        x_mean = x.mean(dim=-1, keepdim=True)  # (B, C, A, 1)
        x_hat_mean = x_hat.mean(dim=-1, keepdim=True)
        
        x_centered = x - x_mean
        x_hat_centered = x_hat - x_hat_mean
        corr_matrix = F.cosine_similarity(x_centered, x_hat_centered, dim=-1, eps=epsilon)
        channel_correlation_loss = 1 - torch.mean(corr_matrix)

        return channel_correlation_loss    

    
    def forward(self, pred, target, mask, embedding):
        """
        计算总损失，包括 MSE、频谱损失和通道相关性损失
        """
        # 1. MSE Loss (basic reconstruction)
        # mse_loss = F.mse_loss(pred * mask, target * mask, reduction='sum') / mask.sum()
        # mse_loss = F.mse_loss(pred , target ) 
        masked_mse_loss = F.mse_loss(pred * mask, target * mask, reduction='sum') / mask.sum()
        non_masked_mse_loss = F.mse_loss(pred * (1 - mask), target * (1 - mask), reduction='sum') / (1 - mask).sum()
        mse_loss = 0.8*masked_mse_loss + 0.2*non_masked_mse_loss

        
        # # 3. Channel Correlation Loss
        # channel_correlation_loss_dynamics = self.compute_channel_correlation_loss(pred, target)
        # channel_correlation_loss_embedding = self.compute_channel_correlation_loss(embedding, target)
        # channel_correlation_loss = 0.5*channel_correlation_loss_dynamics+0.5*channel_correlation_loss_embedding
        
        channel_correlation_loss_dynamics = self.compute_FC_loss(pred, target)  # 计算 FC 损失
        channel_correlation_loss = channel_correlation_loss_dynamics
        # channel_correlation_loss_embedding = self.compute_FC_loss(embedding, target)
        # channel_correlation_loss = 0.5*channel_correlation_loss_dynamics+0.5*channel_correlation_loss_embedding # 将 FC 损失加权

        # 计算总损失
        total_loss = (self.mse_weight * mse_loss + self.correlation_weight * channel_correlation_loss)
                    #   self.spectral_weight * spectral_loss )
        # 返回总损失以及各个损失的组件
        loss_dict = {
            'mse': mse_loss.item(),
            'correlation': channel_correlation_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict    



class ReconstructionLoss_channel_MLP(nn.Module):
    def __init__(self, mse_weight=1.0, spectral_weight=0.3, correlation_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.spectral_weight = spectral_weight
        self.correlation_weight = correlation_weight

    
    def compute_channel_correlation_loss(self, x, x_hat,epsilon=1e-8):
        """
        计算通道相关性损失（Channel Correlation Loss），
        通过计算每个时间步的通道之间的相关性差异。
        加入稳定性的改进。
        """
        # 去均值化：计算每个时间步的均值，并进行去均值化
        x_mean = x.mean(dim=-1, keepdim=True)  # (B, C, A, 1)
        x_hat_mean = x_hat.mean(dim=-1, keepdim=True)
        
        x_centered = x - x_mean
        x_hat_centered = x_hat - x_hat_mean

        corr_matrix = F.cosine_similarity(x_centered, x_hat_centered, dim=-1, eps=epsilon)

        # 对相关性矩阵进行平滑，确保相关性矩阵不超过 [0, 1] 范围
        # corr_matrix = torch.clamp(corr_matrix, min=0, max=1)

        # 计算通道相关性损失
        # 我们取 1 - 平均相关性
        channel_correlation_loss = 1 - torch.mean(corr_matrix)

        return channel_correlation_loss
    
    def forward(self, pred, target, mask):
        """
        计算总损失，包括 MSE、频谱损失和通道相关性损失
        """
        # 1. MSE Loss (basic reconstruction)
        mse_loss = F.mse_loss(pred, target)
        
        # 3. Channel Correlation Loss
        channel_correlation_loss = self.compute_channel_correlation_loss(pred, target)

        # 计算总损失
        total_loss = (self.mse_weight * mse_loss + self.correlation_weight * channel_correlation_loss)
                    #   self.spectral_weight * spectral_loss )
        # 返回总损失以及各个损失的组件
        loss_dict = {
            'mse': mse_loss.item(),
            'correlation': channel_correlation_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict    

class LinearClassificationHead(nn.Module):
    """Linear classification head: Simple linear layer"""
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        # Linear layer for classification
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # Directly apply the linear transformation
        x = self.fc(x)
        return x
 

class SimpleFinetuneClassifier_lora(nn.Module):
    """
    Simple Fine-tuning Classifier (with direct classification after Encoder)

    Architecture:
    1. Encoder (all parameters trainable)
    2. Global Average Pooling (GAP)
    3. Linear Classification Head (1-layer MLP)
    """
    def __init__(self, encoder, in_chans, num_channels=61, num_classes=3, embed_dim=200):
        super().__init__()
        self.encoder = encoder
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.in_chans = in_chans

        # Unfreeze all encoder parameters for full fine-tuning
        # for param in self.encoder.parameters():
        #     param.requires_grad = encoder_free  # Ensure all parameters are trainable

        # Global Average Pooling (GAP) after encoder
        # self.gap = nn.AdaptiveAvgPool1d(1)  # GAP layer to reduce temporal dimension

        # Linear Classification Head (input: embed_dim after GAP)
        self.classifier = nn.Linear(embed_dim, num_classes)  # Adjusted to embed_dim

    def forward(self, x):
        """
        Args: x: (B, C, A, T)
        Returns: logits: (B, num_classes)
        """
        B, C, A, T = x.shape
        # Reshape input for the encoder: (B, C, A, T) -> (B*C, A, T)
        # x_reshaped = rearrange(x, 'b c a t -> (b c) 1 a t')
        x_reshaped = x
        # print(x_reshaped.shape)

        # Encoder to extract features
        features = self.encoder.forward_features(x_reshaped, input_chans = self.in_chans)
        # print('features shape',features.shape)

        # If features have extra CLS token, remove it
        if features.ndim == 3 and features.shape[1] == A + 1:
            features = features[:, 1:, :]

        # # Flatten features: (B, A, D) -> (B, A * D)
        gap_features = features.view(B, -1)
        # print('gap_features',gap_features.shape)

        # Classification
        logits = self.classifier(gap_features)
        return logits
       
class SimpleFinetuneClassifier(nn.Module):
    """
    Simple Fine-tuning Classifier (with direct classification after Encoder)

    Architecture:
    1. Encoder (all parameters trainable)
    2. Global Average Pooling (GAP)
    3. Linear Classification Head (1-layer MLP)
    """
    def __init__(self, encoder, encoder_free, in_chans, num_channels=61, num_classes=3, embed_dim=200):
        super().__init__()
        self.encoder = encoder
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.in_chans = in_chans

        
        # # Unfreeze all encoder parameters for full fine-tuning
        for param in self.encoder.parameters():
            param.requires_grad = encoder_free  # Ensure all parameters are trainable

        # Global Average Pooling (GAP) after encoder
        # self.gap = nn.AdaptiveAvgPool1d(1)  # GAP layer to reduce temporal dimension

        # Linear Classification Head (input: embed_dim after GAP)
        self.classifier = nn.Linear(embed_dim, num_classes)  # Adjusted to embed_dim

    def forward(self, x):
        """
        Args: x: (B, C, A, T)
        Returns: logits: (B, num_classes)
        """
        B, C, A, T = x.shape
        # Reshape input for the encoder: (B, C, A, T) -> (B*C, A, T)
        # x_reshaped = rearrange(x, 'b c a t -> (b c) 1 a t')
        x_reshaped = x
        # print(x_reshaped.shape)

        # Encoder to extract features
        features = self.encoder.forward_features(x_reshaped, input_chans = self.in_chans)
        # print('features shape',features.shape)

        # If features have extra CLS token, remove it
        if features.ndim == 3 and features.shape[1] == A + 1:
            features = features[:, 1:, :]

        # # Apply Global Average Pooling (GAP) to the features (B, C, A, D) -> (B, D)
        # gap_features = self.gap(features.mean(dim=2))  # (B, C, D) -> (B, D)

        # # Flatten features: (B, A, D) -> (B, A * D)
        gap_features = features.view(B, -1)
        # print('gap_features',gap_features.shape)

        # Classification
        logits = self.classifier(gap_features)
        return logits
    
    
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from einops import rearrange    
def evaluate(model, dataloader, criterion, device, use_amp=False):
    """Evaluate model on dataloader"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                eeg, labels = batch
            else:
                eeg = batch['eeg']
                labels = batch['label']

            eeg = eeg.to(device)
            labels = labels.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(eeg)
                    loss = criterion(logits, labels)
            else:
                logits = model(eeg)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(dataloader)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, f1, cm


def evaluate_multitask(model, dataloader, criterion, device, label_mapping, use_amp=False):

    """Evaluate model on dataloader"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                eeg, labels = batch
            else:
                eeg = batch['eeg']
                labels = batch['label']
                mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])

            eeg = eeg.to(device)
            mapped_labels = mapped_labels.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(eeg)
                    loss = criterion(logits, mapped_labels)
            else:
                logits = model(eeg)
                loss = criterion(logits, mapped_labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(mapped_labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(dataloader)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, f1, cm


def evaluate_multitask_cog(model, dataloader, criterion, device, use_amp=False):
    label_mapping = {
        10: 0,
        11: 1,
        12: 2,
        20: 3,
        21: 4,
        22: 5,
    }
    """Evaluate model on dataloader"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                eeg, labels = batch
            else:
                eeg = batch['eeg']
                labels = batch['label']
                mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])

            eeg = eeg.to(device)
            mapped_labels = mapped_labels.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(eeg)
                    loss = criterion(logits, mapped_labels)
            else:
                logits = model(eeg)
                loss = criterion(logits, mapped_labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(mapped_labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(dataloader)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, f1, cm


from einops import rearrange

class SimpleClassifier_multi_model(nn.Module):
    """
    Simple Fine-tuning Classifier (with direct classification after Encoder)

    Architecture:
    1. Encoder (all parameters trainable)
    2. Global Average Pooling (GAP)
    3. Linear Classification Head (1-layer MLP)
    """
    def __init__(self, encoder, in_chans,args, num_channels=61, num_classes=3, embed_dim=256):
        super().__init__()
        self.encoder = encoder
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.model = args.model
        self.classifier_CBM = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                # nn.Linear(200, n_class),
                nn.Linear(embed_dim, num_classes),
            )
        self.classifier = nn.Linear(embed_dim, num_classes)  # Adjusted to embed_dim

    def forward(self, x):
        """
        Args: x: (B, C, A, T)
        Returns: logits: (B, num_classes)
        """
        if self.model == 'CBraMod':
            features = self.encoder(x)
            features=rearrange(features,'b c s d -> b d c s')
             # print(features.shape)
            logits = self.classifier_CBM(features)
        # elif self.model == 'Brainomni':
        #     features = self.encoder.encode(x)
        #     features=rearrange(features,'b c s d -> b d c s')
        #     logits = self.classifier_CBM(features)
        else:
            features = self.encoder.forward_features(x, input_chans = self.in_chans)
            logits = self.classifier(features)
        return logits
    
    
class SimpleClassifier_brainomni(nn.Module):
    """
    Simple Fine-tuning Classifier (with direct classification after Encoder)

    Architecture:
    1. Encoder (all parameters trainable)
    2. Global Average Pooling (GAP)
    3. Linear Classification Head (1-layer MLP)
    """
    def __init__(self, encoder,position,sensortype, args, num_channels=61, num_classes=3, embed_dim=200):
        super().__init__()
        self.encoder = encoder
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.position = position
        self.sensortype=sensortype
        self.model = args.model
        self.classifier_CBM = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                # nn.Linear(200, n_class),
                nn.Linear(embed_dim, num_classes),
            )
        self.classifier = nn.Linear(embed_dim, num_classes)  # Adjusted to embed_dim

    def forward(self, x):
        """
        Args: x: (B, C, A, T)
        Returns: logits: (B, num_classes)
        """
        B,C,A,T = x.shape
        x = x.reshape(B, C, A * T)
        positions = self.position.unsqueeze(0).repeat(B, 1, 1).to(x.device)
        sensor_types = self.sensortype.unsqueeze(0).repeat(B, 1).to(x.device)
        # print('have position')
        features = self.encoder.encode(x,positions, sensor_types)
        features=rearrange(features,'b c s d -> b d c s')
        logits = self.classifier_CBM(features)

        return logits

class SimpleClassifier_BIOT(nn.Module):
    """
    Simple Fine-tuning Classifier (with direct classification after Encoder)

    Architecture:
    1. Encoder (all parameters trainable)
    2. Global Average Pooling (GAP)
    3. Linear Classification Head (1-layer MLP)
    """
    def __init__(self, encoder, args, ch_names, num_classes=3, embed_dim=256):
        super().__init__()
        self.encoder = encoder
        self.eeg_projector = BIOT_DataProjector()
        self.embed_dim = embed_dim
        self.model = args.model
        self.ch_names =ch_names
        self.classifier = nn.Linear(embed_dim, num_classes)  # Adjusted to embed_dim

    def forward(self, x):
        x = self.eeg_projector.channel_project(x, self.ch_names)
        features = self.encoder(x)
        logits = self.classifier(features)

        return logits
    
class SimpleClassifier(nn.Module):
    """
    Simple Fine-tuning Classifier (with direct classification after Encoder)

    Architecture:
    1. Encoder (all parameters trainable)
    2. Global Average Pooling (GAP)
    3. Linear Classification Head (1-layer MLP)
    """
    def __init__(self, encoder, in_chans,args, num_channels=61, num_classes=3, embed_dim=200):
        super().__init__()
        self.encoder = encoder
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.model = args.model
        self.classifier_CBM = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                # nn.Linear(200, n_class),
                nn.Linear(embed_dim, num_classes),
            )
        self.classifier = nn.Linear(embed_dim, num_classes)  # Adjusted to embed_dim

    def forward(self, x):
        """
        Args: x: (B, C, A, T)
        Returns: logits: (B, num_classes)
        """
        if self.model == 'CBraMod':
            features = self.encoder(x)
            features = rearrange(features, 'b c s d -> b d c s')
            logits = self.classifier_CBM(features)
        else:
            features = self.encoder.forward_features(x, input_chans = self.in_chans)
            logits = self.classifier(features)
        return logits
  

import math
def adjust_alpha(epoch, initial_alpha=0.5, max_alpha=1.0, min_alpha=0.1, alpha_decay_factor=0.01,warmup_epochs=5):
    """
    根据epoch调整alpha，避免过早增加alpha
    """
    # 动态调整alpha，避免过大或过小
    if epoch < warmup_epochs:  # 训练初期可以保持较小的 alpha
        alpha = initial_alpha
    else:
        alpha = min(initial_alpha + alpha_decay_factor * (epoch-warmup_epochs), max_alpha)
    return alpha

def cosine_annealing_temperature(epoch, initial_temp=0.1, min_temp=0.005, warmup_epochs=5):
    """
    余弦退火温度调度：逐渐降低温度，避免过早训练过拟合
    """
    if epoch < warmup_epochs:
        return initial_temp  # 使用初始温度进行预热

    progress = (epoch - warmup_epochs) / (300 - warmup_epochs)  # 计算当前进度
    return min_temp + 0.5 * (initial_temp - min_temp) * (1 + math.cos(math.pi * progress))

def label_aware_contrastive_dynamics(mask_emb, truth_emb, labels, epoch, warmup_epoch,
                            temperature=0.07, alpha=0, lambda_mse=0.1, main_weight=1):
    """
    优化版：减少循环，使用tensor操作计算目标概率分布，并引入自适应温度控制
    """
    B, D = mask_emb.shape
    device = mask_emb.device
    
    # 归一化
    mask_norm = F.normalize(mask_emb, dim=-1)
    truth_norm = F.normalize(truth_emb, dim=-1)
    
    # 计算调整后的温度（余弦退火）
    current_temperature = cosine_annealing_temperature(epoch,  initial_temp = temperature, min_temp=0.005,warmup_epochs=warmup_epoch )
    alpha = adjust_alpha(epoch, initial_alpha=alpha, max_alpha=0.8, 
                         alpha_decay_factor=0.005, warmup_epochs=warmup_epoch)
    # 相似度矩阵
    sim = torch.matmul(mask_norm, truth_norm.T)  # (B, B)
    logits = sim / current_temperature  # 使用当前的温度调整相似度
    
    # 创建标签矩阵
    label_mat = (labels.unsqueeze(0) == labels.unsqueeze(1))  # (B, B)
    
    # 创建目标概率分布
    targets = torch.zeros((B, B), device=device)
    
    # 1. 正样本：truth_i
    targets[torch.arange(B), torch.arange(B)] = 1.0  # 最高权重
    
    # 2. 同label但不是配对样本：中等概率
    same_label_mask = label_mat & ~torch.eye(B, device=device, dtype=torch.bool)  # 同label且不为自己
    same_label_idx = same_label_mask.nonzero(as_tuple=False)
    targets[same_label_idx[:, 0], same_label_idx[:, 1]] = alpha  # 较小的正权重
    
    # 3. 不同label：默认概率为0，无需额外操作
    
    # 关键：归一化为概率分布
    targets = targets / targets.sum(dim=1, keepdim=True)
    
    # 验证：确保是有效的概率分布
    assert torch.all(targets >= 0), "targets包含负数"
    assert torch.allclose(targets.sum(dim=1), torch.ones(B, device=device)), "行和不等于1"
    
    # 计算损失
    loss = F.kl_div(F.log_softmax(logits, dim=-1), targets.detach(), reduction='batchmean')
    
    # 添加MSE损失，确保嵌入空间的平滑性
    loss_mse = F.mse_loss(mask_norm, truth_norm)
    total_loss = main_weight * loss + lambda_mse * loss_mse
    
    return total_loss, loss_mse, loss

def label_aware_contrastive_4d_channel_dynamics(subject_embeddings, group_embeddings, labels, args,
                                                epoch, warmup_epoch=5, temperature=0.07, alpha=0,lambda_mse=0.1, 
                                       batch_size=None, channel=None, patch=None, dimension=None):
    B = batch_size
    C= channel
    A = patch
    D = dimension
    if args.model == 'labram_base_patch200_200':
        subject_embeddings = rearrange(subject_embeddings, 'b (c a) t -> b c a t', c=C, a=A, t=D)
        group_embeddings = rearrange(group_embeddings, 'b (c a) t -> b c a t', c=C, a=A, t=D)
        B, C, A, D = subject_embeddings.shape
        subject_flat = subject_embeddings.view(B, C, -1)  # 将patch维度和embedding维度展平 (B, C, A * D)
        group_flat = group_embeddings.view(B, C, -1)  # 将patch维度和embedding维度展平 (B, C, A * D)
    elif args.model == 'CBraMod':
        B, C, A, D = subject_embeddings.shape
        subject_flat = subject_embeddings.view(B, C, -1)
        group_flat = group_embeddings.view(B, C, -1)
    elif args.model =='Brainomni':
        B,C,A,D = subject_embeddings.shape
        subject_flat = subject_embeddings.view(B, C, -1)
        group_flat = group_embeddings.view(B, C, -1)
        
    device = subject_embeddings.device
    
    # 存储每个通道的损失
    channel_losses = []
    channel_intra_losses = []
    channel_cross_losses = []
    
    for c in range(C):
        # 提取当前通道
        subject_c = group_flat[:, c, :]  
        group_c = subject_flat[:, c, :]  

        # 计算当前通道的损失
        loss_c, intra_c, cross_c = label_aware_contrastive_dynamics(
            subject_c, group_c, labels, epoch, warmup_epoch, temperature, alpha, lambda_mse
        )
        
        channel_losses.append(loss_c)
        channel_intra_losses.append(intra_c)
        channel_cross_losses.append(cross_c)
    
    # 平均所有通道
    total_loss = torch.stack(channel_losses).mean()
    avg_intra = torch.stack([l for l in channel_intra_losses if torch.is_tensor(l)]).mean() if channel_intra_losses else torch.tensor(0.0, device=device)
    avg_cross = torch.stack(channel_cross_losses).mean()
    
    return total_loss, avg_intra, avg_cross


###############################################################################################################################

def contrastive_dynamics(mask_emb, truth_emb, labels, epoch, warmup_epoch,
                            temperature=0.07, alpha=0, lambda_mse=0.1,main_weight=1):
    """
    优化版：减少循环，使用tensor操作计算目标概率分布，并引入自适应温度控制
    """
    B, D = mask_emb.shape
    device = mask_emb.device
    
    # 归一化
    mask_norm = F.normalize(mask_emb, dim=-1)
    truth_norm = F.normalize(truth_emb, dim=-1)
    
    # 计算调整后的温度（余弦退火）
    current_temperature = cosine_annealing_temperature(epoch,  initial_temp = temperature, min_temp=0.005,warmup_epochs=warmup_epoch )
    alpha =0 
    #adjust_alpha(epoch, initial_alpha=alpha, max_alpha=0.8, alpha_decay_factor=0.005, warmup_epochs=warmup_epoch)
    
    # 相似度矩阵
    sim = torch.matmul(mask_norm, truth_norm.T)  # (B, B)
    logits = sim / current_temperature  # 使用当前的温度调整相似度
    
    # 创建标签矩阵
    label_mat = (labels.unsqueeze(0) == labels.unsqueeze(1))  # (B, B)
    
    # 创建目标概率分布
    targets = torch.zeros((B, B), device=device)
    
    # 1. 正样本：truth_i
    targets[torch.arange(B), torch.arange(B)] = 1.0  # 最高权重
    
    # 2. 同label但不是配对样本：中等概率
    same_label_mask = label_mat & ~torch.eye(B, device=device, dtype=torch.bool)  # 同label且不为自己
    same_label_idx = same_label_mask.nonzero(as_tuple=False)
    targets[same_label_idx[:, 0], same_label_idx[:, 1]] = alpha  # 较小的正权重
    
    # 3. 不同label：默认概率为0，无需额外操作
    
    # 关键：归一化为概率分布
    targets = targets / targets.sum(dim=1, keepdim=True)
    
    # 验证：确保是有效的概率分布
    assert torch.all(targets >= 0), "targets包含负数"
    assert torch.allclose(targets.sum(dim=1), torch.ones(B, device=device)), "行和不等于1"
    
    # 计算损失
    loss = F.kl_div(F.log_softmax(logits, dim=-1), targets.detach(), reduction='batchmean')
    
    # 添加MSE损失，确保嵌入空间的平滑性
    loss_mse = F.mse_loss(mask_norm, truth_norm)
    total_loss = main_weight * loss + lambda_mse * loss_mse
    
    return total_loss, loss_mse, loss

def contrastive_4d_channel_dynamics(subject_embeddings, group_embeddings, labels, args,
                                    epoch, warmup_epoch=5, temperature=0.07, alpha=0, lambda_mse=0.1, main_weight=1,
                                    batch_size=None, channel=None, patch=None, dimension=None):
    B = batch_size
    C= channel
    A = patch
    D = dimension
    if args.model == 'labram_base_patch200_200':
        subject_embeddings = rearrange(subject_embeddings, 'b (c a) t -> b c a t', c=C, a=A, t=D)
        group_embeddings = rearrange(group_embeddings, 'b (c a) t -> b c a t', c=C, a=A, t=D)
        B, C, A, D = subject_embeddings.shape
        subject_flat = subject_embeddings.view(B, C, -1)  # 将patch维度和embedding维度展平 (B, C, A * D)
        group_flat = group_embeddings.view(B, C, -1)  # 将patch维度和embedding维度展平 (B, C, A * D)
    elif args.model == 'CBraMod':
        B, C, A, D = subject_embeddings.shape
        subject_flat = subject_embeddings.view(B, C, -1)
        group_flat = group_embeddings.view(B, C, -1)
    elif args.model =='Brainomni':
        B,C,A,D = subject_embeddings.shape
        subject_flat = subject_embeddings.view(B, C, -1)
        group_flat = group_embeddings.view(B, C, -1)
        
    device = subject_embeddings.device
    
    # 存储每个通道的损失
    channel_losses = []
    channel_intra_losses = []
    channel_cross_losses = []
    
    for c in range(C):
        # 提取当前通道
        subject_c = group_flat[:, c, :]  
        group_c = subject_flat[:, c, :]  

        # 计算当前通道的损失
        loss_c, intra_c, cross_c = label_aware_contrastive_dynamics(
            subject_c, group_c, labels, epoch, warmup_epoch, temperature, alpha, lambda_mse, main_weight
        )
        
        channel_losses.append(loss_c)
        channel_intra_losses.append(intra_c)
        channel_cross_losses.append(cross_c)
    
    # 平均所有通道
    total_loss = torch.stack(channel_losses).mean()
    avg_intra = torch.stack([l for l in channel_intra_losses if torch.is_tensor(l)]).mean() if channel_intra_losses else torch.tensor(0.0, device=device)
    avg_cross = torch.stack(channel_cross_losses).mean()
    
    return total_loss, avg_intra, avg_cross

########################################################################################################################
def label_aware_contrastive(mask_emb, truth_emb, labels, temperature=0.07, alpha=0.5, lambda_mse=0.1, main_weight=1):
    """
    优化版：减少循环，使用tensor操作计算目标概率分布
    """
    B, D = mask_emb.shape
    device = mask_emb.device
    
    # 归一化
    mask_norm = F.normalize(mask_emb, dim=-1)
    truth_norm = F.normalize(truth_emb, dim=-1)
    
    # 相似度矩阵
    sim = torch.matmul(mask_norm, truth_norm.T)  # (B, B)
    logits = sim / temperature
    
    # 创建标签矩阵
    label_mat = (labels.unsqueeze(0) == labels.unsqueeze(1))  # (B, B)
    
    # 创建目标概率分布
    targets = torch.zeros((B, B), device=device)
    
    # 1. 正样本：truth_i
    targets[torch.arange(B), torch.arange(B)] = 1.0  # 最高权重
    
    # 2. 同label但不是配对样本：中等概率
    same_label_mask = label_mat & ~torch.eye(B, device=device, dtype=torch.bool)  # 同label且不为自己
    same_label_idx = same_label_mask.nonzero(as_tuple=False)
    targets[same_label_idx[:, 0], same_label_idx[:, 1]] = alpha  # 较小的正权重
    
    # 3. 不同label：默认概率为0，无需额外操作
    
    # 关键：归一化为概率分布
    targets = targets / targets.sum(dim=1, keepdim=True)
    
    # 验证：确保是有效的概率分布
    assert torch.all(targets >= 0), "targets包含负数"
    assert torch.allclose(targets.sum(dim=1), torch.ones(B, device=device)), "行和不等于1"
    
    # 计算损失
    loss = F.kl_div(F.log_softmax(logits, dim=-1), targets.detach(), reduction='batchmean')
    
    loss_mse = F.mse_loss(mask_norm, truth_norm)
    total_loss = main_weight * loss + lambda_mse * loss_mse
    
    return total_loss, loss_mse, loss




def label_aware_contrastive_4d_channel(subject_embeddings, group_embeddings, labels, args, temperature=0.07, alpha=0.5, 
                                       lambda_intra=0.1, eps=1e-8, batch_size=None, channel=None, patch=None, dimension=None):
    B = batch_size
    C= channel
    A = patch
    D = dimension
    if args.model == 'labram_base_patch200_200':
        subject_embeddings = rearrange(subject_embeddings, 'b (c a) t -> b c a t', c=C, a=A, t=D)
        group_embeddings = rearrange(group_embeddings, 'b (c a) t -> b c a t', c=C, a=A, t=D)
        B, C, A, D = subject_embeddings.shape
        subject_flat = subject_embeddings.view(B, C, -1)  # 将patch维度和embedding维度展平 (B, C, A * D)
        group_flat = group_embeddings.view(B, C, -1)  # 将patch维度和embedding维度展平 (B, C, A * D)
    elif args.model == 'CBraMod':
        B, C, A, D = subject_embeddings.shape
        subject_flat = subject_embeddings.view(B, C, -1)
        group_flat = group_embeddings.view(B, C, -1)
    elif args.model =='Brainomni':
        B,C,A,D = subject_embeddings.shape
        subject_flat = subject_embeddings.view(B, C, -1)
        group_flat = group_embeddings.view(B, C, -1)
        
    device = subject_embeddings.device
    
    # 存储每个通道的损失
    channel_losses = []
    channel_intra_losses = []
    channel_cross_losses = []
    
    for c in range(C):
        # 提取当前通道
        subject_c = group_flat[:, c, :]  
        group_c = subject_flat[:, c, :]  
        
        # 计算当前通道的损失
        loss_c, intra_c, cross_c = label_aware_contrastive(
            subject_c, group_c, labels, temperature,alpha
        )
        
        channel_losses.append(loss_c)
        channel_intra_losses.append(intra_c)
        channel_cross_losses.append(cross_c)
    
    # 平均所有通道
    total_loss = torch.stack(channel_losses).mean()
    avg_intra = torch.stack([l for l in channel_intra_losses if torch.is_tensor(l)]).mean() if channel_intra_losses else torch.tensor(0.0, device=device)
    avg_cross = torch.stack(channel_cross_losses).mean()
    
    return total_loss, avg_intra, avg_cross

##################################################################################################################################


def label_aware_contrastive_mask(mask_emb, truth_emb, labels, temperature=0.07, alpha=0.5, lambda_mse=0.1):
    """
    优化版：减少循环，使用tensor操作计算目标概率分布
    """
    B, D = mask_emb.shape
    device = mask_emb.device
    
    # 归一化
    mask_norm = F.normalize(mask_emb, dim=-1)
    truth_norm = F.normalize(truth_emb, dim=-1)
    
    # 相似度矩阵
    sim = torch.matmul(mask_norm, truth_norm.T)  # (B, B)
    logits = sim / temperature
    
    # 创建标签矩阵
    label_mat = (labels.unsqueeze(0) == labels.unsqueeze(1))  # (B, B)
    
    # 创建目标概率分布
    targets = torch.zeros((B, B), device=device)
    
    # 1. 正样本：truth_i
    targets[torch.arange(B), torch.arange(B)] = 1.0  # 最高权重
    
    # 2. 同label但不是配对样本：中等概率
    same_label_mask = label_mat & ~torch.eye(B, device=device, dtype=torch.bool)  # 同label且不为自己
    same_label_idx = same_label_mask.nonzero(as_tuple=False)
    targets[same_label_idx[:, 0], same_label_idx[:, 1]] = alpha  # 较小的正权重
    
    # 3. 不同label：默认概率为0，无需额外操作
    
    # 关键：归一化为概率分布
    targets = targets / targets.sum(dim=1, keepdim=True)
    
    # 验证：确保是有效的概率分布
    assert torch.all(targets >= 0), "targets包含负数"
    assert torch.allclose(targets.sum(dim=1), torch.ones(B, device=device)), "行和不等于1"
    
    # 计算损失
    loss = F.kl_div(F.log_softmax(logits, dim=-1), targets.detach(), reduction='batchmean')
    
    # loss_mse = F.mse_loss(mask_norm, truth_norm)
    # total_loss = loss + lambda_mse * loss_mse
    
    return loss


def label_aware_contrastive_4d_channel_mask(subject_embeddings, group_embeddings, mask, labels, args, temperature=0.07, alpha=0, batch_size=None, channel=None, patch=None, dimension=None):
    B = batch_size
    C= channel
    A = patch
    D = dimension
    
    if args.model == 'labram_base_patch200_200':
        subject_embeddings = rearrange(subject_embeddings, 'b (c a) t -> b c a t', c=C, a=A, t=D)
        group_embeddings = rearrange(group_embeddings, 'b (c a) t -> b c a t', c=C, a=A, t=D)
        B, C, A, D = subject_embeddings.shape
        subject_flat = subject_embeddings.view(B, C, -1)  # 将patch维度和embedding维度展平 (B, C, A * D)
        group_flat = group_embeddings.view(B, C, -1)  # 将patch维度和embedding维度展平 (B, C, A * D)
        
    elif args.model == 'CBraMod':
        B, C, A, D = subject_embeddings.shape
        subject_flat = subject_embeddings.view(B, C, -1)
        group_flat = group_embeddings.view(B, C, -1)
        
    device = subject_embeddings.device
    mask_ratio = mask.sum(dim=[2, 3]) / (A * D)  # (B, C), 每个通道在每个样本上的mask比例
    
    # 存储每个通道的损失
    channel_losses = []
    
    for c in range(C):
        # 提取当前通道
        subject_c = group_flat[:, c, :]  
        group_c = subject_flat[:, c, :]  
        
        # 计算当前通道的损失
        loss_c = label_aware_contrastive_mask(
            subject_c, group_c, labels, temperature,alpha)
        
        channel_losses.append(loss_c)

    weighted_losses = []
    for i, loss_c in enumerate(channel_losses):
        weighted_loss = loss_c * mask_ratio[:, i]  # 通过mask比例加权
        weighted_losses.append(weighted_loss)
        
    # 平均所有通道
    total_loss = torch.stack(weighted_losses).mean()
    
    return total_loss






def info_embedding_loss(mask_embeddings, truth_embeddings, temperature=0.1, lambda_mse=0.5,eps=1e-8):
    # 1. 输入检查
    if torch.isnan(mask_embeddings).any() or torch.isinf(mask_embeddings).any():
        print("警告: mask_embeddings包含nan/inf")
        mask_embeddings = torch.nan_to_num(mask_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
    
    if torch.isnan(truth_embeddings).any() or torch.isinf( truth_embeddings).any():
        print("警告:  truth_embeddings包含nan/inf")
        truth_embeddings = torch.nan_to_num( truth_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 2. 获取形状
    B, D = mask_embeddings.shape
    device = mask_embeddings.device
    
    # 调试信息
    # print(f"Loss输入: B={B}, D={D}")
    # print(f"subject范数: {subject_embeddings.norm().item():.4f}")
    # print(f"group范数: {group_embeddings.norm().item():.4f}")
    
    # 3. 安全归一化（避免除以0）
    def safe_normalize(x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=eps)  # 避免除以0
        return x / norm
    
    mask_norm = safe_normalize(mask_embeddings)
    truth_norm = safe_normalize(truth_embeddings)
    
    # 4. 跨模态对比损失
    # 计算相似度矩阵
    sim_cross = torch.matmul(mask_norm, truth_norm.T)  # (B, B)
    sim_cross = torch.clamp(sim_cross, -1.0 + eps, 1.0 - eps)
    logits_cross = sim_cross / temperature  
   
    # 检查logits是否为nan
    if torch.isnan(logits_cross).any():
        print(f"警告: logits_cross包含nan, sim_cross范围: [{sim_cross.min().item():.4f}, {sim_cross.max().item():.4f}]")
        logits_cross = torch.nan_to_num(logits_cross, nan=0.0)
    
    # 创建标签
    labels = torch.arange(B, device=device)
    
    # 计算交叉熵损失
    loss_cross = F.cross_entropy(logits_cross, labels)
    
    loss_mse = F.mse_loss(mask_embeddings, truth_embeddings)
    
    # 6. 总损失
    total_loss = loss_cross + lambda_mse * loss_mse
    
    # 7. 最终检查
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"严重警告: 总损失为nan/inf")
        print(f"  loss_cross: {loss_cross.item() if not torch.isnan(loss_cross) else 'nan'}")
        print(f"  loss_intra: {loss_mse.item() if not torch.isnan(loss_mse) else 'nan'}")
        print(f"  sim_cross范围: [{sim_cross.min().item():.4f}, {sim_cross.max().item():.4f}]")
        
        # 返回一个安全的损失值
        return torch.tensor(0.1, device=device, requires_grad=True)
    
    # 8. 打印调试信息
    # if B > 1:  # 只在batch_size>1时打印
    #     print(f"损失详情: cross={loss_cross.item():.4f}, intra={loss_intra.item():.4f}, total={total_loss.item():.4f}")
    #     print(f"相似度范围: cross[{sim_cross.min().item():.4f}, {sim_cross.max().item():.4f}], "
    #           f"intra[{sim_intra.min().item():.4f}, {sim_intra.max().item():.4f}]")
    
    return total_loss, lambda_mse * loss_mse, loss_cross

def info_embedding_loss_4d_channel(subject_embeddings, group_embeddings, temperature=0.7, lambda_intra=10, eps=1e-8, batch_size=None, channel=None, patch=None, dimension=None):
    B = batch_size
    C= channel
    A = patch
    D = dimension
    subject_embeddings = rearrange(subject_embeddings, 'b (c a) t -> b c a t', c=C, a=A, t=D)
    group_embeddings = rearrange(group_embeddings, 'b (c a) t -> b c a t', c=C, a=A, t=D)
    B, C, A, D = subject_embeddings.shape
    device = subject_embeddings.device
    
    # 存储每个通道的损失
    channel_losses = []
    channel_intra_losses = []
    channel_cross_losses = []
    
    for c in range(C):
        # 提取当前通道
        subject_c = subject_embeddings[:, c, :, :]  # [B, A, D]
        group_c = group_embeddings[:, c, :, :]      # [B, A, D]
        
        # Patch平均
        subject_c_pooled = torch.mean(subject_c, dim=1)  # [B, D]
        group_c_pooled = torch.mean(group_c, dim=1)      # [B, D]
        
        # 计算当前通道的损失
        loss_c, intra_c, cross_c = info_embedding_loss(
            subject_c_pooled, group_c_pooled, temperature, lambda_intra, eps
        )
        
        channel_losses.append(loss_c)
        channel_intra_losses.append(intra_c)
        channel_cross_losses.append(cross_c)
    
    # 平均所有通道
    total_loss = torch.stack(channel_losses).mean()
    avg_intra = torch.stack([l for l in channel_intra_losses if torch.is_tensor(l)]).mean() if channel_intra_losses else torch.tensor(0.0, device=device)
    avg_cross = torch.stack(channel_cross_losses).mean()
    
    return total_loss, avg_intra, avg_cross





def info_nce_loss_loop(subject_embeddings, group_embeddings, temperature=0.7, lambda_intra=10, eps=1e-8):
    """
    数值稳定的双向对比学习损失
    """
    # 1. 输入检查
    if torch.isnan(subject_embeddings).any() or torch.isinf(subject_embeddings).any():
        print("警告: subject_embeddings包含nan/inf")
        subject_embeddings = torch.nan_to_num(subject_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
    
    if torch.isnan(group_embeddings).any() or torch.isinf(group_embeddings).any():
        print("警告: group_embeddings包含nan/inf")
        group_embeddings = torch.nan_to_num(group_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 2. 获取形状
    B, D = subject_embeddings.shape
    device = subject_embeddings.device
    
    # 调试信息
    # print(f"Loss输入: B={B}, D={D}")
    # print(f"subject范数: {subject_embeddings.norm().item():.4f}")
    # print(f"group范数: {group_embeddings.norm().item():.4f}")
    
    # 3. 安全归一化（避免除以0）
    def safe_normalize(x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=eps)  # 避免除以0
        return x / norm
    
    subj_norm = safe_normalize(subject_embeddings)
    group_norm = safe_normalize(group_embeddings)
    # subj_norm=subject_embeddings
    # group_norm=group_embeddings
    
    # 4. 跨模态对比损失
    # 计算相似度矩阵
    sim_cross = torch.matmul(subj_norm, group_norm.T)  # (B, B)
    sim_cross = torch.clamp(sim_cross, -1.0 + eps, 1.0 - eps)
    logits_cross = - sim_cross / temperature  
   
    # 检查logits是否为nan
    if torch.isnan(logits_cross).any():
        print(f"警告: logits_cross包含nan, sim_cross范围: [{sim_cross.min().item():.4f}, {sim_cross.max().item():.4f}]")
        logits_cross = torch.nan_to_num(logits_cross, nan=0.0)
    
    # 创建标签
    labels = torch.arange(B, device=device)
    
    # 计算交叉熵损失
    loss_cross = F.cross_entropy(logits_cross, labels)
    
    # 5. 模态内对比损失
    # 计算subject内部的相似度
    sim_intra = torch.matmul(subj_norm, subj_norm.T)  # (B, B)
    sim_intra = torch.clamp(sim_intra, -1.0 + eps, 1.0 - eps)
    
    # 创建正样本掩码（排除对角线）
    pos_mask = torch.ones(B, B, device=device, dtype=torch.bool)
    pos_mask.fill_diagonal_(0)  # 排除自己
    
    # 计算正样本平均相似度
    if pos_mask.sum() > 0:
        pos_sim = sim_intra[pos_mask].mean()
        loss_intra = 1 - pos_sim
    else:
        loss_intra = torch.tensor(0.0, device=device)
        print("警告: 没有有效的模态内正样本")
    
    # 6. 总损失
    total_loss = loss_cross + lambda_intra * loss_intra
    
    # 7. 最终检查
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"严重警告: 总损失为nan/inf")
        print(f"  loss_cross: {loss_cross.item() if not torch.isnan(loss_cross) else 'nan'}")
        print(f"  loss_intra: {loss_intra.item() if not torch.isnan(loss_intra) else 'nan'}")
        print(f"  sim_cross范围: [{sim_cross.min().item():.4f}, {sim_cross.max().item():.4f}]")
        print(f"  sim_intra范围: [{sim_intra.min().item():.4f}, {sim_intra.max().item():.4f}]")
        
        # 返回一个安全的损失值
        return torch.tensor(0.1, device=device, requires_grad=True)
    
    # 8. 打印调试信息
    # if B > 1:  # 只在batch_size>1时打印
    #     print(f"损失详情: cross={loss_cross.item():.4f}, intra={loss_intra.item():.4f}, total={total_loss.item():.4f}")
    #     print(f"相似度范围: cross[{sim_cross.min().item():.4f}, {sim_cross.max().item():.4f}], "
    #           f"intra[{sim_intra.min().item():.4f}, {sim_intra.max().item():.4f}]")
    
    return total_loss, lambda_intra * loss_intra, loss_cross


def info_nce_loss_4d_channel(subject_embeddings, group_embeddings, temperature=0.7, lambda_intra=10, eps=1e-8, batch_size=None, channel=None, patch=None, dimension=None):
    """
    通道级对比损失
    对每个通道单独计算损失，然后平均
    """
    B = batch_size
    C= channel
    A = patch
    D = dimension
    subject_embeddings = rearrange(subject_embeddings, 'b (c a) t -> b c a t', c=C, a=A, t=D)
    group_embeddings = rearrange(group_embeddings, 'b (c a) t -> b c a t', c=C, a=A, t=D)
    B, C, A, D = subject_embeddings.shape
    device = subject_embeddings.device
    
    # 存储每个通道的损失
    channel_losses = []
    channel_intra_losses = []
    channel_cross_losses = []
    
    for c in range(C):
        # 提取当前通道
        subject_c = subject_embeddings[:, c, :, :]  # [B, A, D]
        group_c = group_embeddings[:, c, :, :]      # [B, A, D]
        
        # Patch平均
        subject_c_pooled = torch.mean(subject_c, dim=1)  # [B, D]
        group_c_pooled = torch.mean(group_c, dim=1)      # [B, D]
        
        # 计算当前通道的损失
        loss_c, intra_c, cross_c = info_nce_loss_loop(
            subject_c_pooled, group_c_pooled, temperature, lambda_intra, eps
        )
        
        channel_losses.append(loss_c)
        channel_intra_losses.append(intra_c)
        channel_cross_losses.append(cross_c)
    
    # 平均所有通道
    total_loss = torch.stack(channel_losses).mean()
    avg_intra = torch.stack([l for l in channel_intra_losses if torch.is_tensor(l)]).mean() if channel_intra_losses else torch.tensor(0.0, device=device)
    avg_cross = torch.stack(channel_cross_losses).mean()
    
    return total_loss, avg_intra, avg_cross


# def encode_femba(model: FEMBA, X: torch.Tensor, mask: torch.Tensor | None = None,
#                  return_tokens: bool = False):
#     if mask is None:
#         mask = torch.zeros_like(X, dtype=torch.bool)
#     X_masked = X.clone()
#     X_masked[mask] = 0
#     tokens = model.patch_embed(X_masked) # [B, grid_w, D_model]
#     tokens = tokens + model.pos_embed
#     for blk, ln in zip(model.mamba_blocks, model.norm_layers):
#         tokens = ln(tokens+blk(tokens))
#     emb = tokens.mean(dim=1) # [B, D_model]
#     return (emb, tokens) if return_tokens else emb


class Conv1dWithConstraint(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)