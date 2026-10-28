#!/usr/bin/env python3
"""Local mock test for CBraMod continue pretraining logic."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, 'W:/yinuo/Rest_tuning')

from src.cbramod import CBraMod


# ------------------------------------------------------------------
# 1. Replicate PatchMasking from run_cbramod_pretraining_resume.py
# ------------------------------------------------------------------
class PatchMasking:
    def __init__(self, mask_ratio=0.4):
        self.mask_ratio = mask_ratio

    def __call__(self, x):
        B, C, A, T = x.shape
        device = x.device

        if A == 1:
            mask = torch.zeros(B, C, A, device=device)
            n_mask = max(1, int(C * self.mask_ratio))
            for b in range(B):
                mask_ch = torch.randperm(C)[:n_mask]
                mask[b, mask_ch, :] = 1
        else:
            mask = torch.zeros(B, C, A, device=device)
            n_mask = max(1, int(A * self.mask_ratio))
            for b in range(B):
                mask_patches = torch.randperm(A)[:n_mask]
                mask[b, :, mask_patches] = 1
        return mask


# ------------------------------------------------------------------
# 2. Replicate masked_mse_loss
# ------------------------------------------------------------------
def masked_mse_loss(pred, target, mask):
    mask_4d = mask.unsqueeze(-1)  # (B, C, A, 1)
    diff = (pred - target) * mask_4d
    loss = (diff ** 2).sum() / (mask.sum() * pred.size(-1) + 1e-8)
    return loss


# ------------------------------------------------------------------
# 3. Run one fake epoch
# ------------------------------------------------------------------
def main():
    device = torch.device('cpu')
    B, C, A, T = 4, 62, 1, 200

    # Fake data (same shape as CogBCI)
    eeg = torch.randn(B, C, A, T).to(device)

    # Model
    model = CBraMod(
        in_dim=200, out_dim=200, d_model=200,
        dim_feedforward=800, seq_len=30, n_layer=12, nhead=8,
    ).to(device)

    masking = PatchMasking(mask_ratio=0.4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    model.train()
    for step in range(5):
        mask = masking(eeg)

        out = model(eeg, mask=mask)
        loss = masked_mse_loss(out, eeg, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Visible MSE
        with torch.no_grad():
            visible_mask = 1 - mask
            visible_mse = masked_mse_loss(out, eeg, visible_mask)

        print(f"Step {step}:  mask.sum={mask.sum().item():.0f}  "
              f"masked_loss={loss.item():.4f}  visible_mse={visible_mse.item():.4f}")

        # Debug: check mask per sample
        for b in range(min(2, B)):
            n_masked = mask[b].sum().item()
            n_visible = (1 - mask[b]).sum().item()
            print(f"  Sample {b}: masked_channels={n_masked:.0f}  visible_channels={n_visible:.0f}")


if __name__ == '__main__':
    main()
