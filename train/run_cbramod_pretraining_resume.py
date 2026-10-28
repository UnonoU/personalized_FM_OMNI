#!/usr/bin/env python3
"""
CBraMod Continue Pretraining on CogBCI Multitask Dataset
Reference: run_labram_pretraining_resume.py

Uses CogBCI MULTITASK data (NOT resting-state) for continue pretraining:
  /mnt/dataset4/fuxy/m3cv/processed_cogbci_multitask_session1_ChangeFilter/cogbci_multitask_session1_temporal.h5

Pretraining objective: Masked Autoencoding (MAE-style)
  1. Randomly mask a ratio of patches in the input EEG
  2. Forward through CBraMod encoder with masking
  3. Reconstruct the original signal at masked positions
  4. Loss = MSE on masked positions only

Usage (single subject):
    python train/run_cbramod_pretraining_resume.py \
        --data_path /mnt/dataset4/fuxy/m3cv/processed_cogbci_multitask_session1_ChangeFilter/cogbci_multitask_session1_temporal.h5 \
        --output_dir /path/to/output/cbramod_resume_sub_1 \
        --exclude_subjects "1" \
        --epochs 50 \
        --batch_size 64 \
        --lr 5e-4 \
        --mask_ratio 0.4 \
        --device cuda:0

Usage (LOO batch, see run_sh/CBraMod/COG_continue_pretrain.sh)
"""

import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import json
import os
import sys
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Subset
from src.cbramod import CBraMod
from src.data_loader import EEGDataset


# ============================================================================
# Dataset Wrapper
# ============================================================================
class ExcludeSubjectDataset(Dataset):
    """
    Wrapper around EEGDataset that excludes specified subjects.
    Used for Leave-One-Out (LOO) pretraining.
    """
    def __init__(self, base_dataset, exclude_subjects=None):
        self.base = base_dataset
        self.exclude = set(int(s) for s in (exclude_subjects or []))
        self.indices = [
            i for i in range(len(self.base))
            if int(self.base.subject_ids[i]) not in self.exclude
        ]
        print(f"  [Dataset] Total samples: {len(self.base)}, "
              f"Excluded subjects: {sorted(self.exclude)}, "
              f"Kept samples: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


# ============================================================================
# Masking Strategy
# ============================================================================
class PatchMasking:
    """
    Random channel-level masking for single-patch CBraMod inputs.
    Input x shape: (B, C, A, T)
    Mask shape: (B, C, A) -- 1 = masked, 0 = visible

    CogBCI samples contain exactly one temporal patch (A == 1), so masking is
    performed independently across channels for every sample. At least one
    channel is masked and one channel is kept visible.
    """
    def __init__(self, mask_ratio=0.4):
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(
                f"mask_ratio must be strictly between 0 and 1, got {mask_ratio}"
            )
        self.mask_ratio = mask_ratio

    def __call__(self, x):
        """
        Args:
            x: Tensor of shape (B, C, A, T)
        Returns:
            mask: Tensor of shape (B, C, A) with values 0 or 1
        """
        B, C, A, T = x.shape
        device = x.device

        if A != 1:
            raise ValueError(
                f"PatchMasking expects single-patch input (A=1), got A={A}"
            )
        if C < 2:
            raise ValueError(
                f"Channel masking requires at least two channels, got C={C}"
            )

        mask = torch.zeros(B, C, A, dtype=torch.long, device=device)
        n_mask = min(C - 1, max(1, round(C * self.mask_ratio)))
        for b in range(B):
            mask_ch = torch.randperm(C, device=device)[:n_mask]
            mask[b, mask_ch, 0] = 1

        return mask


class CBraModPretrain(nn.Module):
    """
    Wrapper around CBraMod for masked reconstruction pretraining.
    """
    def __init__(self, cbramod):
        super().__init__()
        self.encoder = cbramod

    def forward(self, x, mask=None):
        return self.encoder(x, mask=mask)


# ============================================================================
# Argument Parser
# ============================================================================
def get_args():
    parser = argparse.ArgumentParser('CBraMod continue pre-training on CogBCI', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='cbramod', type=str)
    parser.add_argument('--in_dim', default=200, type=int,
                        help='Input patch dimension (should match patch_size)')
    parser.add_argument('--out_dim', default=200, type=int,
                        help='Output patch dimension (should match in_dim for reconstruction)')
    parser.add_argument('--d_model', default=200, type=int,
                        help='Model dimension')
    parser.add_argument('--dim_feedforward', default=800, type=int)
    parser.add_argument('--seq_len', default=30, type=int,
                        help='Maximum sequence length (not critical for CogBCI with A=1)')
    parser.add_argument('--n_layer', default=12, type=int,
                        help='Number of transformer layers')
    parser.add_argument('--nhead', default=8, type=int,
                        help='Number of attention heads')

    # Training parameters
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)
    parser.add_argument('--eval_freq', default=1, type=int)

    # Masking parameters
    parser.add_argument('--mask_ratio', default=0.4, type=float,
                        help='Ratio of patches to mask (0.0 ~ 1.0)')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--clip_grad', type=float, default=3.0)

    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CogBCI multitask H5 file')
    parser.add_argument('--exclude_subjects', type=str, default='',
                        help='Comma-separated subject IDs to exclude (LOO mode)')
    parser.add_argument('--patch_size', default=200, type=int,
                        help='Patch size for EEG reshaping')
    parser.add_argument('--normalize', action='store_true', default=True)
    parser.add_argument('--normalize_per_channel', action='store_true', default=True)
    parser.add_argument('--train_split', default=0.9, type=float)

    # Resume parameters
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint path')
    parser.add_argument('--output_dir', default='', required=True)

    # System parameters
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--pin_mem', action='store_true', default=True)

    return parser.parse_args()


# ============================================================================
# Checkpoint Loading (with key remapping for CBraMod pretrained weights)
# ============================================================================
def load_checkpoint(model, checkpoint_path, optimizer=None):
    """
    Load checkpoint with robust key remapping.

    Handles:
      - Raw CBraMod weights (keys without 'encoder.' prefix)
      - Our continue-pretrain checkpoints (keys with 'encoder.' prefix)
      - Standard checkpoint dicts with 'model' key

    If `optimizer` is provided and the checkpoint contains an 'optimizer'
    state dict, the optimizer state is restored as well.
    """
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract state dict from various checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"Checkpoint format is unrecognized: expected a state-dict mapping, "
            f"got {type(state_dict).__name__}"
        )

    model_dict = model.state_dict()

    # Canonicalize common wrapper prefixes, then resolve each key against both
    # the wrapped model and a raw CBraMod state dict. Resolving key-by-key also
    # handles raw keys such as "encoder.layers.*", whose "encoder." prefix is
    # part of CBraMod itself rather than CBraModPretrain.
    filtered = {}
    skipped = []
    unexpected = []
    duplicates = []
    removable_prefixes = ('module.', '_orig_mod.')

    for checkpoint_key, value in state_dict.items():
        if not isinstance(checkpoint_key, str) or not torch.is_tensor(value):
            unexpected.append(str(checkpoint_key))
            continue

        key = checkpoint_key
        prefix_removed = True
        while prefix_removed:
            prefix_removed = False
            for prefix in removable_prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    prefix_removed = True

        candidates = (key, f'encoder.{key}')
        target_key = next((candidate for candidate in candidates if candidate in model_dict), None)
        if target_key is None:
            unexpected.append(checkpoint_key)
            continue
        if value.shape != model_dict[target_key].shape:
            skipped.append(
                (checkpoint_key, tuple(value.shape), tuple(model_dict[target_key].shape))
            )
            continue
        if target_key in filtered:
            duplicates.append((checkpoint_key, target_key))
            continue
        filtered[target_key] = value

    loaded = len(filtered)
    total = len(model_dict)
    missing = [k for k in model_dict if k not in filtered]

    # Continue pretraining must never silently fall back to a partially random
    # model. Extra checkpoint keys are tolerated only when every model tensor
    # has still been resolved; missing, mismatched, or duplicate tensors abort.
    if missing or skipped or duplicates:
        details = [
            f"Checkpoint is incompatible with the current model: loaded {loaded}/{total} tensors."
        ]
        if missing:
            details.append(f"Missing keys ({len(missing)}): {missing[:10]}")
        if skipped:
            details.append(f"Shape mismatches ({len(skipped)}): {skipped[:10]}")
        if duplicates:
            details.append(f"Duplicate mappings ({len(duplicates)}): {duplicates[:10]}")
        if unexpected:
            details.append(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}")
        raise RuntimeError("\n  ".join(details))

    model.load_state_dict(filtered, strict=True)
    print(f"  Loaded {loaded}/{total} tensors successfully.")
    if unexpected:
        print(f"  Ignored {len(unexpected)} extra checkpoint keys: {unexpected[:5]}")
    # Restore optimizer state if available
    if optimizer is not None:
        if isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("  Optimizer state restored.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to restore optimizer state from {checkpoint_path}"
                ) from e
        else:
            print("  No optimizer state in checkpoint; using fresh optimizer.")

    # Return epoch if available
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        return checkpoint['epoch']
    return None


# ============================================================================
# Model
# ============================================================================
def get_model(args):
    print(f"Creating CBraMod model: in_dim={args.in_dim}, out_dim={args.out_dim}, "
          f"d_model={args.d_model}, n_layer={args.n_layer}, nhead={args.nhead}")
    cbramod = CBraMod(
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        d_model=args.d_model,
        dim_feedforward=args.dim_feedforward,
        seq_len=args.seq_len,
        n_layer=args.n_layer,
        nhead=args.nhead,
    )
    model = CBraModPretrain(cbramod)
    return model


# ============================================================================
# Scheduler
# ============================================================================
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    if len(iters) == 0:
        schedule = warmup_schedule
    else:
        schedule = np.array([
            final_value + 0.5 * (base_value - final_value) *
            (1 + np.cos(np.pi * i / len(iters)))
            for i in iters
        ])
        schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep, \
        f"Schedule length {len(schedule)} != expected {epochs * niter_per_ep}"
    return schedule


# ============================================================================
# Masked MSE Loss
# ============================================================================
def masked_mse_loss(pred, target, mask):
    """
    Compute MSE loss only on masked positions.

    Args:
        pred:   (B, C, A, P)
        target: (B, C, A, P)
        mask:   (B, C, A) -- 1 = compute loss, 0 = ignore
    Returns:
        loss: scalar
    """
    mask_4d = mask.unsqueeze(-1)  # (B, C, A, 1)
    diff = (pred - target) * mask_4d
    loss = (diff ** 2).sum() / (mask.sum() * pred.size(-1) + 1e-8)
    return loss


# ============================================================================
# Training
# ============================================================================
def train_one_epoch(model, masking, dataloader, optimizer, device, epoch,
                    lr_schedule, wd_schedule, start_steps, clip_grad, use_amp=False):
    """
    Train for one epoch using masked reconstruction objective.
    """
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    num_steps = len(dataloader)

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for step, batch in enumerate(dataloader):
        global_step = start_steps + step

        # Update LR and weight decay
        if global_step < len(lr_schedule):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[global_step]
                if wd_schedule is not None and "weight_decay" in param_group:
                    param_group["weight_decay"] = wd_schedule[global_step]

        # Data: (B, C, A, T)
        eeg = batch['eeg'].to(device, non_blocking=True)

        # Generate mask: (B, C, A)
        mask = masking(eeg)

        # Forward with masking
        if use_amp:
            with torch.cuda.amp.autocast():
                out = model(eeg, mask=mask)  # (B, C, A, out_dim)
                loss = masked_mse_loss(out, eeg, mask)
        else:
            out = model(eeg, mask=mask)
            loss = masked_mse_loss(out, eeg, mask)

        # Backward
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        total_loss += loss.item()

        # Also compute MSE on visible positions for logging
        with torch.no_grad():
            visible_mask = 1 - mask
            visible_mse = masked_mse_loss(out, eeg, visible_mask)
            total_mse += visible_mse.item()

        if step % 50 == 0 or step == num_steps - 1:
            print(f"  Epoch [{epoch}][{step}/{num_steps}] "
                  f"Masked Loss: {loss.item():.4f}  "
                  f"Visible MSE: {visible_mse.item():.4f}  "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    return {
        'loss': total_loss / num_steps,
        'visible_mse': total_mse / num_steps,
    }


@torch.no_grad()
def evaluate(model, masking, dataloader, device, use_amp=False):
    """
    Evaluate on validation set.
    """
    model.eval()
    total_loss = 0.0
    total_visible_mse = 0.0

    for batch in dataloader:
        eeg = batch['eeg'].to(device, non_blocking=True)
        mask = masking(eeg)

        if use_amp:
            with torch.cuda.amp.autocast():
                out = model(eeg, mask=mask)
                loss = masked_mse_loss(out, eeg, mask)
        else:
            out = model(eeg, mask=mask)
            loss = masked_mse_loss(out, eeg, mask)

        total_loss += loss.item()

        # Visible MSE
        visible_mask = 1 - mask
        visible_mse = masked_mse_loss(out, eeg, visible_mask)
        total_visible_mse += visible_mse.item()

    return {
        'loss': total_loss / len(dataloader),
        'visible_mse': total_visible_mse / len(dataloader),
    }


# ============================================================================
# Checkpoint
# ============================================================================
def save_checkpoint(args, epoch, model, optimizer, is_best=False):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': vars(args),
    }

    torch.save(checkpoint, output_dir / 'checkpoint.pth')

    if (epoch + 1) % args.save_ckpt_freq == 0:
        torch.save(checkpoint, output_dir / f'checkpoint-{epoch}.pth')

    if is_best:
        torch.save(checkpoint, output_dir / 'checkpoint-best.pth')
        print(f"  -> Best checkpoint saved!")

    print(f"  -> Checkpoint saved to {output_dir / 'checkpoint.pth'}")


# ============================================================================
# Main
# ============================================================================
def main(args):
    print("=" * 80)
    print("CBraMod Continue Pretraining on CogBCI Multitask Data (Masked Reconstruction)")
    print("=" * 80)
    print(args)
    print("=" * 80)

    device = torch.device(args.device)

    # Fix seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # Create model
    model = get_model(args).to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nNumber of trainable params: {n_parameters}")

    # Verify out_dim matches patch_size for reconstruction
    if args.out_dim != args.patch_size:
        print(f"\nWARNING: out_dim ({args.out_dim}) != patch_size ({args.patch_size}). "
              f"Reconstruction MSE may fail. Consider setting --out_dim={args.patch_size}")

    # Optimizer (created BEFORE checkpoint loading so its state can be restored)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            ckpt_epoch = load_checkpoint(model, args.resume, optimizer=optimizer)
            if ckpt_epoch is not None:
                start_epoch = ckpt_epoch + 1
                print(f"Resuming from epoch {start_epoch}")
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

    # Load dataset
    print(f"\nLoading dataset from: {args.data_path}")
    base_dataset = EEGDataset(
        h5_path=args.data_path,
        patch_size=args.patch_size,
        normalize=args.normalize,
        normalize_per_channel=args.normalize_per_channel,
    )
    print(f"  Base dataset size: {len(base_dataset)}")

    # Exclude subjects
    exclude_list = []
    if args.exclude_subjects:
        exclude_list = [int(s.strip()) for s in args.exclude_subjects.split(',')]
        print(f"  Excluding subjects: {exclude_list}")

    dataset = ExcludeSubjectDataset(base_dataset, exclude_subjects=exclude_list)

    # Train / Val split
    n_total = len(dataset)
    n_val = max(1, int(n_total * (1 - args.train_split)))
    n_train = n_total - n_val

    indices = list(range(n_total))
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    # Masking
    masking = PatchMasking(mask_ratio=args.mask_ratio)
    print(f"\nMasking config: ratio={args.mask_ratio}")

    # Schedulers
    num_training_steps_per_epoch = len(train_loader)
    total_steps = args.epochs * num_training_steps_per_epoch

    lr_schedule = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay, args.weight_decay, args.epochs, num_training_steps_per_epoch,
    )

    print(f"\nTraining config:")
    print(f"  Total epochs: {args.epochs}")
    print(f"  Steps per epoch: {num_training_steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  LR range: [{args.min_lr:.2e}, {args.lr:.2e}]")

    # Training loop
    print(f"\n{'='*80}")
    print(f"Start training")
    print(f"{'='*80}")

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'#'*60}")
        print(f"# Epoch {epoch}/{args.epochs}")
        print(f"{'#'*60}")

        train_stats = train_one_epoch(
            model, masking, train_loader, optimizer, device, epoch,
            lr_schedule, wd_schedule,
            start_steps=epoch * num_training_steps_per_epoch,
            clip_grad=args.clip_grad,
            use_amp=args.use_amp,
        )

        # Validation
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_stats = evaluate(model, masking, val_loader, device, use_amp=args.use_amp)
            print(f"\n  Epoch {epoch} Summary:")
            print(f"    Train Masked Loss: {train_stats['loss']:.4f}")
            print(f"    Train Visible MSE: {train_stats['visible_mse']:.4f}")
            print(f"    Val Masked Loss:   {val_stats['loss']:.4f}")
            print(f"    Val Visible MSE:   {val_stats['visible_mse']:.4f}")

            is_best = val_stats['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_stats['loss']
                print(f"    *** New best val masked loss: {best_val_loss:.4f} ***")
        else:
            val_stats = {'loss': -1, 'visible_mse': -1}
            is_best = False
            print(f"\n  Epoch {epoch} Summary:")
            print(f"    Train Masked Loss: {train_stats['loss']:.4f}")
            print(f"    Train Visible MSE: {train_stats['visible_mse']:.4f}")

        # Save checkpoint
        if args.output_dir:
            save_checkpoint(args, epoch, model, optimizer, is_best=is_best)

        # Log
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters,
        }
        log_path = os.path.join(args.output_dir, "log.txt")
        with open(log_path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"\n{'='*80}")
    print(f"Training completed! Total time: {total_time_str}")
    print(f"Best val masked loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
