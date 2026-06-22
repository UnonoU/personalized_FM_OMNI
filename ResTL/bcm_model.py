"""
BCM (Baseline Correction Module) Implementation
Based on: Kwak et al. 2023 - Subject-Invariant Deep Neural Networks Based on Baseline Correction for EEG Motor Imagery BCI

This implementation integrates BCM with ResTL's existing EEGNet/Conformer backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaselineCorrectionModule(nn.Module):
    """
    Baseline Correction Module (BCM)
    
    Estimates subject-specific gain matrix and background features
    to extract subject-invariant representations.
    
    Args:
        feature_dim: Dimension of input features (default: 64)
        m: Feature magnitude constraint (BCIC IV 2a: m=5, OpenBMI: m=1)
    """
    def __init__(self, feature_dim=64, m=5.0):
        super(BaselineCorrectionModule, self).__init__()
        self.feature_dim = feature_dim
        self.m = m
        
        # H_η_gain: Estimates pseudo-inverse gain matrix (G^i)^+
        # Input: feature_dim, Output: feature_dim x feature_dim (flattened)
        self.gain_estimator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, feature_dim * feature_dim)  # 64*64=4096
        )
        
        # H_η_back: Estimates background features E[s_back^i]
        # Input: feature_dim, Output: feature_dim
        self.back_estimator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, feature_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, f_task, f_baseline, phase='train'):
        """
        Args:
            f_task: Task EEG features [batch_size, feature_dim] or [1, feature_dim] for test
            f_baseline: Baseline EEG features
                - Train: [batch_size, feature_dim] (randomly paired)
                - Test: [num_baseline_segments, feature_dim] (averaged from 1-min baseline)
            phase: 'train' or 'test'
        
        Returns:
            s_target: Subject-invariant features
        """
        if phase == 'train':
            return self.forward_train(f_task, f_baseline)
        else:
            return self.forward_test(f_task, f_baseline)
    
    def forward_train(self, f_task, f_baseline):
        """
        Training phase (Equation 14 in paper)
        Single-sample pairing with random baseline
        """
        batch_size = f_task.size(0)
        
        # (G^i)^+ ≈ H_η_gain(f_baseline)
        gain_flat = self.gain_estimator(f_baseline)  # [batch, feature_dim^2]
        G_inv = gain_flat.view(batch_size, self.feature_dim, self.feature_dim)
        
        # E[s_back^i] ≈ H_η_back(f_baseline)
        s_back = self.back_estimator(f_baseline)  # [batch, feature_dim]
        
        # Apply gain matrix: (G^i)^+ * f_task
        # f_task: [batch, feature_dim], G_inv: [batch, feature_dim, feature_dim]
        f_task_expanded = f_task.unsqueeze(-1)  # [batch, feature_dim, 1]
        corrected = torch.bmm(G_inv, f_task_expanded).squeeze(-1)  # [batch, feature_dim]
        
        # Subtract background: s_target^i = (G^i)^+ * x^i - E[s_back^i]
        s_target = corrected - s_back
        
        # Unit vector normalization with magnitude constraint (U(x))
        s_target = F.normalize(s_target, p=2, dim=1) * self.m
        
        return s_target
    
    def forward_test(self, f_task, f_baseline):
        """
        Test phase (Equation 15 in paper)
        Average over multiple baseline segments from 1-minute resting state
        
        Args:
            f_task: [batch_size, feature_dim] - test samples (single or batch)
            f_baseline: [num_segments, feature_dim] - multiple baseline segments
        """
        num_base = f_baseline.size(0)
        batch_size = f_task.size(0)
        
        # Estimate gain matrices for all baseline segments
        gain_flat = self.gain_estimator(f_baseline)  # [N_base, feature_dim^2]
        G_inv_all = gain_flat.view(num_base, self.feature_dim, self.feature_dim)
        G_inv_mean = torch.mean(G_inv_all, dim=0, keepdim=True)  # [1, feature_dim, feature_dim]
        
        # Estimate background features for all baseline segments
        s_back_all = self.back_estimator(f_baseline)  # [N_base, feature_dim]
        s_back_mean = torch.mean(s_back_all, dim=0, keepdim=True)  # [1, feature_dim]
        
        # Apply to test samples (handle both single sample and batch)
        f_task_expanded = f_task.unsqueeze(-1)  # [batch_size, feature_dim, 1]
        # Expand G_inv_mean to match batch size for bmm
        G_inv_mean_expanded = G_inv_mean.expand(batch_size, self.feature_dim, self.feature_dim)
        corrected = torch.bmm(G_inv_mean_expanded, f_task_expanded).squeeze(-1)  # [batch_size, feature_dim]
        # Expand s_back_mean to match batch size
        s_back_mean_expanded = s_back_mean.expand(batch_size, self.feature_dim)
        s_target = corrected - s_back_mean_expanded
        s_target = F.normalize(s_target, p=2, dim=1) * self.m
        
        return s_target


class SubjectInvariantLoss(nn.Module):
    """
    Subject-Invariant Loss (Triplet Loss with Cosine Distance)
    
    Based on Equation 16 in the paper.
    
    Args:
        alpha: Weight for invariant loss (EEGNet: 0.1, MIN2Net: 1.0)
    """
    def __init__(self, alpha=0.1):
        super(SubjectInvariantLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, features, labels):
        """
        Args:
            features: [batch, feature_dim] - BCM normalized output
            labels: [batch] - Class labels
        
        Returns:
            loss: Triplet loss value
        """
        batch_size = features.size(0)
        if batch_size < 3:
            return torch.tensor(0.0).to(features.device)
        
        # Compute cosine distance matrix
        # dist[i,j] = 1 - cos(theta) = 1 - (f_i · f_j) / (||f_i|| ||f_j||)
        features_norm = F.normalize(features, p=2, dim=1)
        similarity = torch.mm(features_norm, features_norm.t())  # [batch, batch]
        dist = 1 - similarity  # Cosine distance in range [0, 2]
        
        losses = []
        
        # For each anchor
        for i in range(batch_size):
            anchor_label = labels[i]
            
            # Positive mask: same class, excluding self
            pos_mask = (labels == anchor_label) & (torch.arange(batch_size, device=features.device) != i)
            # Negative mask: different class
            neg_mask = labels != anchor_label
            
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            
            # Hardest positive: furthest positive sample
            pos_dist = dist[i][pos_mask]
            hardest_pos = pos_dist.max()
            
            # Hardest negative: closest negative sample
            neg_dist = dist[i][neg_mask]
            hardest_neg = neg_dist.min()
            
            # Triplet loss: max(0, d(a,p) - d(a,n))
            # Paper uses margin=0 implicitly
            loss_i = torch.clamp(hardest_pos - hardest_neg, min=0.0)
            losses.append(loss_i)
        
        if len(losses) == 0:
            return torch.tensor(0.0).to(features.device)
        
        return torch.stack(losses).mean()


class BCMLoss(nn.Module):
    """
    Total loss for BCM training
    L_total = L_cls + alpha * L_inv
    """
    def __init__(self, alpha=0.1):
        super(BCMLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.inv_loss = SubjectInvariantLoss(alpha)
        self.alpha = alpha
    
    def forward(self, outputs, features, labels):
        """
        Args:
            outputs: [batch, num_classes] - Classifier output
            features: [batch, feature_dim] - BCM output
            labels: [batch] - Ground truth labels
        
        Returns:
            total_loss, cls_loss, inv_loss
        """
        loss_cls = self.ce_loss(outputs, labels)
        loss_inv = self.inv_loss(features, labels)
        total_loss = loss_cls + self.alpha * loss_inv
        return total_loss, loss_cls, loss_inv


class BCMWithEEGNet(nn.Module):
    """
    BCM integrated with EEGNet backbone from ResTL
    
    Architecture:
        EEGNet (encoder) -> BCM -> Classifier
    """
    def __init__(self, args, feature_dim=96, m=5.0, alpha=0.1):
        """
        Args:
            args: Configuration arguments
            feature_dim: Feature dimension from EEGNet encoder
                         For size=200: (200//4//8)*16 = 96
                         For size=750: (750//4//8)*16 = 368
            m: Feature magnitude constraint
            alpha: Loss weight for invariant loss
        """
        super(BCMWithEEGNet, self).__init__()
        self.args = args
        self.feature_dim = feature_dim
        
        # Import EEGNet components from model.py
        from model import EEGNet, Conv2dWithConstraint
        
        # EEGNet encoder (without final classifier)
        eegnet = EEGNet(args, global_f=feature_dim, feature_size=64)
        self.encoder1 = eegnet.encoder1  # First conv block
        self.encoder2 = eegnet.encoder2  # Second conv block (Depthwise separable)
        self.flatten = nn.Flatten()
        
        # Project to BCM feature dimension if needed
        if feature_dim != 64:
            self.feature_projector = nn.Linear(feature_dim, 64)
            bcm_input_dim = 64
        else:
            self.feature_projector = None
            bcm_input_dim = feature_dim
        
        # BCM module
        self.bcm = BaselineCorrectionModule(feature_dim=bcm_input_dim, m=m)
        
        # Classifier
        self.classifier = nn.Linear(bcm_input_dim, args.datanum)
        
        # Loss function
        self.criterion = BCMLoss(alpha=alpha)
    
    def extract_features(self, x):
        """
        Extract features using EEGNet encoder
        
        Args:
            x: [batch, 1, time, channels]
        
        Returns:
            features: [batch, feature_dim]
        """
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.flatten(x)
        
        if self.feature_projector is not None:
            x = self.feature_projector(x)
        
        return x
    
    def forward(self, x_task, x_baseline=None, phase='train', labels=None):
        """
        Forward pass
        
        Args:
            x_task: Task EEG [batch, 1, time, channels]
            x_baseline: Baseline EEG [batch, 1, time, channels] or [N_base, 1, time, channels]
            phase: 'train' or 'test'
            labels: Ground truth labels for training (optional)
        
        Returns:
            During training: (output, invariant_feat, total_loss, cls_loss, inv_loss) if labels provided
                           (output, invariant_feat) otherwise
            During testing: (output, invariant_feat)
        """
        # Extract features from task EEG
        f_task = self.extract_features(x_task)  # [batch, feature_dim]
        
        # Handle baseline
        if x_baseline is None:
            # No baseline provided, use identity (no correction)
            invariant_feat = f_task
        else:
            # Extract baseline features
            if phase == 'train':
                # Training: same batch size
                f_baseline = self.extract_features(x_baseline)
            else:
                # Testing: variable number of baseline segments
                if x_baseline.dim() == 4:
                    batch_size = x_baseline.size(0)
                    # Flatten batch dimension if needed
                    x_baseline_flat = x_baseline.view(-1, x_baseline.size(-2), x_baseline.size(-1))
                    f_baseline = self.extract_features(x_baseline_flat.unsqueeze(1))
                else:
                    f_baseline = self.extract_features(x_baseline)
            
            # Apply BCM
            invariant_feat = self.bcm(f_task, f_baseline, phase)
        
        # Classification
        output = self.classifier(invariant_feat)
        
        # Compute loss if labels provided
        if labels is not None and phase == 'train':
            total_loss, cls_loss, inv_loss = self.criterion(output, invariant_feat, labels)
            return output, invariant_feat, total_loss, cls_loss, inv_loss
        
        return output, invariant_feat


class BCMWithConformer(nn.Module):
    """
    BCM integrated with Conformer backbone from ResTL
    
    Architecture:
        Conformer (encoder) -> BCM -> Classifier
    """
    def __init__(self, args, feature_dim=40, m=5.0, alpha=0.1):
        """
        Args:
            args: Configuration arguments
            feature_dim: Feature dimension from Conformer (default: 40 from emb_size)
            m: Feature magnitude constraint
            alpha: Loss weight for invariant loss
        """
        super(BCMWithConformer, self).__init__()
        self.args = args
        self.feature_dim = feature_dim
        
        # Import Conformer components
        from conformer import PatchEmbedding, TransformerEncoder
        from einops.layers.torch import Rearrange
        
        # Conformer encoder components
        self.patch = PatchEmbedding(emb_size=40, ch=args.chnum)
        self.encoder = TransformerEncoder(depth=6, emb_size=40)
        self.rearrange1 = Rearrange('b hw e -> b e hw 1')
        self.rearrange2 = Rearrange('b e (h) (w) -> b (h w) e')
        
        # Project to BCM feature dimension
        self.feature_projector = nn.Linear(40 * 44, 64)  # 40 channels * 44 patches
        
        # BCM module
        self.bcm = BaselineCorrectionModule(feature_dim=64, m=m)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, args.datanum)
        )
        
        # Loss function
        self.criterion = BCMLoss(alpha=alpha)
    
    def extract_features(self, x):
        """
        Extract features using Conformer encoder
        
        Args:
            x: [batch, 1, time, channels]
        
        Returns:
            features: [batch, feature_dim]
        """
        # Patch embedding
        x = self.patch(x.permute(0, 1, 3, 2))  # [batch, 40, 44]
        x = self.rearrange1(x)  # [batch, 40, 44, 1]
        x = self.rearrange2(x)  # [batch, 44*1, 40] = [batch, 44, 40]
        
        # Transformer encoding
        x = self.encoder(x)  # [batch, 44, 40]
        x = x.permute(0, 2, 1).unsqueeze(-1)  # [batch, 40, 44, 1]
        x = x.reshape(x.size(0), -1)  # [batch, 40*44]
        
        # Project
        x = self.feature_projector(x)  # [batch, 64]
        
        return x
    
    def forward(self, x_task, x_baseline=None, phase='train', labels=None):
        """
        Forward pass (same interface as BCMWithEEGNet)
        """
        # Extract features
        f_task = self.extract_features(x_task)
        
        # Handle baseline
        if x_baseline is None:
            invariant_feat = f_task
        else:
            if phase == 'train':
                f_baseline = self.extract_features(x_baseline)
            else:
                if x_baseline.dim() == 4:
                    x_baseline_flat = x_baseline.view(-1, x_baseline.size(-2), x_baseline.size(-1))
                    f_baseline = self.extract_features(x_baseline_flat.unsqueeze(1))
                else:
                    f_baseline = self.extract_features(x_baseline)
            
            invariant_feat = self.bcm(f_task, f_baseline, phase)
        
        # Classification
        output = self.classifier(invariant_feat)
        
        # Compute loss
        if labels is not None and phase == 'train':
            total_loss, cls_loss, inv_loss = self.criterion(output, invariant_feat, labels)
            return output, invariant_feat, total_loss, cls_loss, inv_loss
        
        return output, invariant_feat


def create_bcm_model(args, m=5.0, alpha=0.1):
    """
    Factory function to create BCM model with appropriate backbone
    
    Args:
        args: Configuration with args.model ('EEGNet' or 'Conformer') and args.size
        m: Feature magnitude constraint
        alpha: Loss weight for invariant loss
    
    Returns:
        model: BCM model instance
    """
    if args.model == 'EEGNet':
        # Calculate feature dimension based on time size
        time_after_pool = ((args.size // 4) // 8)
        feature_dim = 16 * time_after_pool  # 16 channels * time
        return BCMWithEEGNet(args, feature_dim=feature_dim, m=m, alpha=alpha)
    
    elif args.model == 'Conformer':
        # Conformer uses 40-dim embeddings
        return BCMWithConformer(args, feature_dim=40, m=m, alpha=alpha)
    
    else:
        raise ValueError(f"Unknown model: {args.model}")


# ============================================================================
# Training and Testing Functions for BCM
# ============================================================================

def train_bcm_one_epoch(model, dataloader, optimizer, device, alpha=0.1):
    """
    Train BCM for one epoch
    
    Args:
        model: BCM model
        dataloader: Training dataloader with (task_eeg, baseline_eeg, labels)
        optimizer: Optimizer
        device: Device
        alpha: Loss weight
    
    Returns:
        avg_loss, avg_cls_loss, avg_inv_loss, accuracy
    """
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_inv_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        # Unpack batch
        if len(batch) == 3:
            x_task, x_baseline, labels = batch
        else:
            # Handle different batch formats
            x_task = batch['x_anc'].to(device)
            x_baseline = batch.get('x_anc_rest', x_task).to(device)
            labels = batch['label'].argmax(dim=1).to(device)
        
        x_task = x_task.to(device)
        x_baseline = x_baseline.to(device)
        labels = labels.to(device)
        
        # Ensure correct shape: [batch, 1, time, channels]
        if x_task.dim() == 3:
            x_task = x_task.unsqueeze(1)
        if x_baseline.dim() == 3:
            x_baseline = x_baseline.unsqueeze(1)
        
        # Forward
        optimizer.zero_grad()
        output, features, loss, cls_loss, inv_loss = model(
            x_task, x_baseline, phase='train', labels=labels
        )
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_inv_loss += inv_loss.item()
        
        _, predicted = output.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_inv_loss = total_inv_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, avg_cls_loss, avg_inv_loss, accuracy


def test_bcm(model, dataloader, baseline_data, device):
    """
    Test BCM with resting state baseline
    
    Args:
        model: BCM model
        dataloader: Test dataloader with task EEG
        baseline_data: Resting state EEG for baseline [N, 1, time, channels]
        device: Device
    
    Returns:
        accuracy, predictions, labels
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Pre-compute baseline features (average over all baseline segments)
    with torch.no_grad():
        if baseline_data.dim() == 3:
            baseline_data = baseline_data.unsqueeze(1)  # [N, 1, time, channels]
        
        # BCM handles baseline averaging internally in test mode
        baseline_batch = baseline_data.to(device)
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                x_task = batch['x_anc'].to(device)
                labels = batch['label'].to(device)
                # Convert one-hot to indices if needed
                if labels.dim() > 1:
                    labels = labels.argmax(dim=1)
            else:
                x_task, labels = batch
                x_task = x_task.to(device)
                labels = labels.to(device)
                # Convert one-hot to indices if needed
                if labels.dim() > 1:
                    labels = labels.argmax(dim=1)
            
            if x_task.dim() == 3:
                x_task = x_task.unsqueeze(1)
            
            # Forward with baseline
            output, _ = model(x_task, baseline_batch, phase='test')
            
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    return accuracy, all_preds, all_labels


if __name__ == '__main__':
    # Test the implementation
    import argparse
    
    print("=" * 70)
    print("BCM (Baseline Correction Module) Implementation Test")
    print("=" * 70)
    
    # Mock arguments
    args = argparse.Namespace()
    args.model = 'EEGNet'
    args.size = 200
    args.chnum = 62
    args.datanum = 6
    
    # Create model
    print("\n[1] Testing BCM with EEGNet...")
    model = create_bcm_model(args, m=5.0, alpha=0.1)
    
    # Test forward pass
    batch_size = 4
    x_task = torch.randn(batch_size, 1, 200, 62)
    x_baseline = torch.randn(batch_size, 1, 200, 62)
    labels = torch.randint(0, 6, (batch_size,))
    
    output, features, loss, cls_loss, inv_loss = model(
        x_task, x_baseline, phase='train', labels=labels
    )
    
    print(f"  Output shape: {output.shape}")
    print(f"  Features shape: {features.shape}")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Cls loss: {cls_loss.item():.4f}")
    print(f"  Inv loss: {inv_loss.item():.4f}")
    
    # Test inference
    print("\n[2] Testing BCM inference...")
    model.eval()
    with torch.no_grad():
        x_test = torch.randn(1, 1, 200, 62)
        x_baseline_test = torch.randn(10, 1, 200, 62)  # 10 baseline segments
        output, features = model(x_test, x_baseline_test, phase='test')
        print(f"  Test output shape: {output.shape}")
        print(f"  Test features shape: {features.shape}")
    
    # Test BCM module directly
    print("\n[3] Testing BCM module directly...")
    bcm = BaselineCorrectionModule(feature_dim=64, m=5.0)
    f_task = torch.randn(4, 64)
    f_baseline = torch.randn(4, 64)
    output = bcm(f_task, f_baseline, phase='train')
    print(f"  BCM output shape: {output.shape}")
    print(f"  Output magnitude (should be ~{bcm.m}): {output.norm(dim=1).mean().item():.4f}")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
