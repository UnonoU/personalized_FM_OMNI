#!/usr/bin/env python3
"""
BIOT Continue Pretraining on CogBCI Multitask Dataset
Reference: official BIOT run_unsupervised_pretrain.py
           (https://github.com/ycq091044/BIOT)

Uses CogBCI MULTITASK data (NOT resting-state) for continue pretraining:
  /mnt/dataset4/fuxy/m3cv/processed_cogbci_multitask_session1_ChangeFilter/cogbci_multitask_session1_temporal.h5

Pretraining objective: official BIOT InfoNCE contrastive loss
  - L2-normalize perturbed & clean embeddings, in-batch cross-entropy (T=0.2)
  - Perturbation: original BIOT drops random STFT time frames; CogBCI samples
    are 1s (200 pts) -> a single STFT frame, so we fall back to time-domain
    segment masking (BIOTEncoderSafePerturb)

Channel projection: 62 CogBCI channels -> 16 bipolar channels
  - Per-sample montage: the H5 stores two channel layouts (montage_type 0=V1,
    1=V2); each sample is projected with its own layout's channel names
  - The 16 standard bipolar channels are used; "C3-A2"/"C4-A1" are dropped
    because CogBCI has no A1/A2 mastoid electrodes (they were always zero)
  - Model n_channels stays 18 so the pretrained channel_tokens fully load;
    only tokens 0..15 are used at input time

Normalization: OFF by default (the CogBCI H5 is already globally z-scored);
  enable with --normalize if needed

Checkpoint handling:
  - Supports original BIOT pretrained encoder weights (e.g., EEG-six-datasets-18-channels.ckpt)
  - Auto-remaps raw encoder keys -> biot.xxx keys

Embedding extraction (after pretraining):
    from train.run_biot_pretraining_resume import extract_biot_embeddings
    emb = extract_biot_embeddings(
        "/path/to/biot_resume_1/checkpoint.pth",
        eeg,               # (N, 62, T) raw CogBCI signals
        montage_types,     # (N,) 0=V1, 1=V2 (from H5 'montage_type')
        device="cuda:0",
    )  # -> (N, 256)

Usage (LOO batch, see run_sh/BIOT/COG_continue_pretrain.sh)
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
from src.biot import BIOTEncoder
from src.data_loader import EEGDataset
from src.utils import BIOT_DataProjector, get_cogbci_dataset


# ============================================================================
# Constants
# ============================================================================
# 16 standard bipolar channels of the BIOT six-datasets montage.
# NOTE: the old default also included "C3-A2" / "C4-A1", but CogBCI has no
# A1/A2 mastoid electrodes, so those two channels were always zero.
BIPOLAR_TARGET_CHANS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]


def get_montage_channel_names():
    """
    Return {montage_type: channel_names} for CogBCI.
    The H5 'montage_type' flag is 0 for the V1 layout and 1 for the V2 layout.
    """
    dummy_args = argparse.Namespace(dataset='cogbci')
    _, ch_v1, _ = get_cogbci_dataset(dummy_args, subject_id=1)   # V1 layout
    _, ch_v2, _ = get_cogbci_dataset(dummy_args, subject_id=17)  # V2 layout
    return {0: ch_v1, 1: ch_v2}


# ============================================================================

# ============================================================================
# Safe Perturbation Encoder (fixes crash on short sequences)
# ============================================================================
class BIOTEncoderSafePerturb(BIOTEncoder):
    """
    BIOTEncoder with safe perturbation for short sequences.

    Original BIOT perturbation randomly drops STFT time frames.
    For CogBCI's 200 timepoints, STFT produces only 1 frame,
    causing ``cannot reshape tensor of 0 elements`` when all frames are dropped.

    Fix: when STFT yields <= 1 frame, fall back to time-domain segment masking.
    """
    def __init__(self, *args, time_mask_ratio=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_mask_ratio = time_mask_ratio

    def _time_mask(self, x):
        """Random contiguous time segment masking on raw signal."""
        B, C, T = x.shape
        x_masked = x.clone()
        mask_len = max(1, int(T * self.time_mask_ratio))
        for b in range(B):
            start = np.random.randint(0, T - mask_len + 1)
            x_masked[b, :, start:start + mask_len] = 0
        return x_masked

    def forward(self, x, n_channel_offset=0, perturb=False, return_channel_embeddings=False):
        if perturb:
            B, C, T = x.shape
            # Number of STFT frames: floor((T - n_fft) / hop_length) + 1
            ts = (T - self.n_fft) // self.hop_length + 1
            if ts <= 1:
                # Short sequence: use time-domain masking, disable STFT perturbation
                x = self._time_mask(x)
                perturb = False
        return super().forward(x, n_channel_offset, perturb, return_channel_embeddings)


# ============================================================================
# Unsupervised Pretraining Module (matches original BIOT)
# ============================================================================
class UnsupervisedPretrainSafe(nn.Module):
    """
    Original BIOT UnsupervisedPretrain architecture with safe encoder.

    Structure matches the original so pretrained checkpoints load correctly:
      - self.biot: BIOTEncoder (or our safe subclass)
      - self.prediction: MLP (emb_size -> emb_size)
    """
    def __init__(self, emb_size=256, heads=8, depth=4, n_channels=18, **kwargs):
        super().__init__()
        self.biot = BIOTEncoderSafePerturb(emb_size, heads, depth, n_channels, **kwargs)
        self.prediction = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Linear(emb_size, emb_size),
        )

    def forward(self, x, n_channel_offset=0):
        emb = self.biot(x, n_channel_offset, perturb=True)
        emb = self.prediction(emb)
        pred_emb = self.biot(x, n_channel_offset)
        return emb, pred_emb




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


class MontageProjectDataset(Dataset):
    """
    Per-sample bipolar projection (62 CogBCI channels -> 16 bipolar channels),
    using the channel layout that matches each sample's montage_type.

    The H5 file stores samples in two different channel orders (montage_type
    0=V1, 1=V2). Projecting everything with a single layout would silently
    compute bipolar differences between the wrong electrodes (~14% of samples).
    Samples without montage info fall back to the V1 layout.
    """
    def __init__(self, base_dataset, target_chans=BIPOLAR_TARGET_CHANS,
                 montage_ch_names=None):
        self.base = base_dataset
        self.target_chans = target_chans
        self.montage_ch_names = montage_ch_names or get_montage_channel_names()
        self.projector = BIOT_DataProjector(target_chans=target_chans)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = dict(self.base[idx])
        eeg = item['eeg']  # (62, num_patches, patch_size)
        montage = int(item['montage_type'])
        if montage not in self.montage_ch_names:
            montage = 0  # fallback: V1 layout
        x = self.projector.channel_project(
            eeg.unsqueeze(0), self.montage_ch_names[montage]
        )  # (1, 16, T)
        item['eeg'] = x.squeeze(0)  # (16, T)
        return item


# ============================================================================
# Argument Parser
# ============================================================================
def get_args():
    parser = argparse.ArgumentParser('BIOT continue pre-training on CogBCI', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='biot', type=str)
    parser.add_argument('--emb_size', default=256, type=int,
                        help='Embedding dimension for BIOT')
    parser.add_argument('--heads', default=8, type=int,
                        help='Number of attention heads')
    parser.add_argument('--depth', default=4, type=int,
                        help='Number of transformer layers')
    parser.add_argument('--n_channels', default=18, type=int,
                        help='Number of EEG channels fed to BIOT after projection (18 for BIOT bipolar)')
    parser.add_argument('--n_fft', default=200, type=int,
                        help='FFT window size for STFT')
    parser.add_argument('--hop_length', default=100, type=int,
                        help='Hop length for STFT')
    parser.add_argument('--time_mask_ratio', default=0.3, type=float,
                        help='Time-domain mask ratio for short sequences (0.0 ~ 1.0)')
    parser.add_argument('--cogbci_version', default='auto', type=str,
                        choices=['auto', 'v1', 'v2'],
                        help='Fallback channel layout (v1/v2) only used when the H5 '
                             'file has no montage_type field. Normally per-sample '
                             'montage_type drives the projection.')


    # Training parameters
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--eval_freq', default=1, type=int,
                        help='Evaluate every N epochs')

    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=3.0,
                        help='Gradient clipping norm')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='InfoNCE temperature (official BIOT pretraining uses 0.2)')

    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CogBCI multitask H5 file')
    parser.add_argument('--exclude_subjects', type=str, default='',
                        help='Comma-separated subject IDs to exclude (LOO mode)')
    parser.add_argument('--patch_size', default=200, type=int,
                        help='Patch size for EEG reshaping')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='Apply extra Z-score normalization (default OFF: '
                             'the CogBCI H5 is already globally z-scored)')
    parser.add_argument('--normalize_per_channel', action='store_true', default=True,
                        help='Normalize each channel independently (only with --normalize)')
    parser.add_argument('--train_split', default=0.9, type=float,
                        help='Training data split ratio')

    # Resume parameters
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint path')
    parser.add_argument('--output_dir', default='', required=True,
                        help='Output directory for checkpoints and logs')

    # System parameters
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Use automatic mixed precision')
    parser.add_argument('--pin_mem', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader')

    return parser.parse_args()


# ============================================================================
# Checkpoint Loading (robust remapping for original BIOT checkpoints)
# ============================================================================
def load_checkpoint(model, checkpoint_path):
    """
    Load checkpoint with robust key remapping.

    Handles:
      - Raw BIOTEncoder weights (keys without 'biot.' prefix)
      - Full UnsupervisedPretrain weights (keys with 'biot.' and 'prediction.' prefix)
      - Standard checkpoint dicts with 'model' key
      - Channel count mismatch (e.g., 18-ch pretrained -> 62-ch CogBCI)
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
        print(f"  WARNING: Checkpoint format unrecognized, skipping load.")
        return 0

    model_dict = model.state_dict()

    # Detect if checkpoint has raw encoder keys (no biot. prefix)
    has_raw_encoder = any(
        not k.startswith('biot.') and not k.startswith('prediction.')
        for k in state_dict.keys()
    )

    # Remap keys if needed
    if has_raw_encoder:
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith('biot.') or k.startswith('prediction.'):
                remapped[k] = v
            else:
                remapped[f'biot.{k}'] = v
        state_dict = remapped

    # Filter by shape compatibility
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                filtered[k] = v
            else:
                skipped.append((k, tuple(v.shape), tuple(model_dict[k].shape)))

    model.load_state_dict(filtered, strict=False)

    loaded = len(filtered)
    total = len(model_dict)
    print(f"  Loaded {loaded}/{total} parameters")

    if skipped:
        print(f"  Skipped {len(skipped)} keys due to shape mismatch:")
        for k, ckpt_shape, model_shape in skipped[:5]:
            print(f"    {k}: checkpoint {ckpt_shape} vs model {model_shape}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")

    # Report missing keys
    missing = [k for k in model_dict if k not in filtered]
    if missing:
        print(f"  Missing {len(missing)} keys (randomly initialized):")
        for k in missing[:5]:
            print(f"    {k}: {tuple(model_dict[k].shape)}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")

    # Return epoch if available
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        return checkpoint['epoch']
    return 0


# ============================================================================
# Model
# ============================================================================
def get_model(args):
    print(f"Creating BIOT UnsupervisedPretrain: emb_size={args.emb_size}, heads={args.heads}, "
          f"depth={args.depth}, n_channels={args.n_channels}, "
          f"time_mask_ratio={args.time_mask_ratio}")
    model = UnsupervisedPretrainSafe(
        emb_size=args.emb_size,
        heads=args.heads,
        depth=args.depth,
        n_channels=args.n_channels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        time_mask_ratio=args.time_mask_ratio,
    )
    return model


# ============================================================================
# Scheduler
# ============================================================================
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0):
    """
    Cosine learning rate scheduler with warmup.
    """
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
# Training
# ============================================================================
def info_nce_loss(emb_masked, emb_clean, temperature=0.2):
    """
    Official BIOT unsupervised pretraining loss (run_unsupervised_pretrain.py):
    L2-normalize both embeddings, then in-batch contrastive cross-entropy.
    """
    emb_masked = F.normalize(emb_masked, dim=1, p=2)
    emb_clean = F.normalize(emb_clean, dim=1, p=2)
    logits = torch.mm(emb_clean, emb_masked.t()) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels, reduction="mean")


def train_one_epoch(model, dataloader, optimizer, device, epoch,
                    lr_schedule, wd_schedule, start_steps, clip_grad,
                    use_amp=False, temperature=0.2):
    """
    Train for one epoch using the official BIOT InfoNCE pretraining objective.
    """
    model.train()
    total_loss = 0.0
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

        # MontageProjectDataset already returns projected bipolar data (B, 16, T)
        x = batch['eeg'].to(device, non_blocking=True)

        # Forward: UnsupervisedPretrain returns (emb_perturbed, emb_clean)
        if use_amp:
            with torch.cuda.amp.autocast():
                emb_pred, emb_clean = model(x)
                loss = info_nce_loss(emb_pred, emb_clean, temperature)
        else:
            emb_pred, emb_clean = model(x)
            loss = info_nce_loss(emb_pred, emb_clean, temperature)

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

        if step % 50 == 0 or step == num_steps - 1:
            print(f"  Epoch [{epoch}][{step}/{num_steps}] "
                  f"Loss: {loss.item():.4f}  LR: {optimizer.param_groups[0]['lr']:.6f}")

    return {'loss': total_loss / num_steps}


@torch.no_grad()
def evaluate(model, dataloader, device, use_amp=False, temperature=0.2):
    """
    Evaluate on validation set.
    """
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        # MontageProjectDataset already returns projected bipolar data (B, 16, T)
        x = batch['eeg'].to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                emb_pred, emb_clean = model(x)
                loss = info_nce_loss(emb_pred, emb_clean, temperature)
        else:
            emb_pred, emb_clean = model(x)
            loss = info_nce_loss(emb_pred, emb_clean, temperature)

        total_loss += loss.item()

    return {'loss': total_loss / len(dataloader)}


# ============================================================================
# Checkpoint
# ============================================================================
def save_checkpoint(args, epoch, model, optimizer, is_best=False):
    """
    Save training checkpoint.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': vars(args),
    }

    # Always save latest
    torch.save(checkpoint, output_dir / 'checkpoint.pth')

    # Periodic save
    if (epoch + 1) % args.save_ckpt_freq == 0:
        torch.save(checkpoint, output_dir / f'checkpoint-{epoch}.pth')

    # Best model
    if is_best:
        torch.save(checkpoint, output_dir / 'checkpoint-best.pth')
        print(f"  -> Best checkpoint saved!")

    print(f"  -> Checkpoint saved to {output_dir / 'checkpoint.pth'}")


# ============================================================================
# Embedding extraction (use a continue-pretrained checkpoint downstream)
# ============================================================================
def load_biot_encoder(checkpoint_path, device='cuda',
                      emb_size=256, heads=8, depth=4, n_channels=18,
                      n_fft=200, hop_length=100):
    """
    Load the BIOT encoder from a checkpoint.

    Accepts both this script's checkpoints (keys with 'biot.' prefix, saved
    under the 'model' key) and the original raw BIOT checkpoints (keys are
    auto-remapped).

    Returns:
        BIOTEncoder in eval mode on `device`.
        Input: (B, 16, T) bipolar EEG. Output: (B, emb_size) embedding.
    """
    model = UnsupervisedPretrainSafe(
        emb_size=emb_size, heads=heads, depth=depth, n_channels=n_channels,
        n_fft=n_fft, hop_length=hop_length,
    )
    load_checkpoint(model, checkpoint_path)
    encoder = model.biot.to(device)
    encoder.eval()
    return encoder


@torch.no_grad()
def extract_biot_embeddings(checkpoint_path, eeg, montage_types=None,
                            batch_size=256, device='cuda', **model_kwargs):
    """
    Extract embeddings from a (continue-)pretrained BIOT checkpoint.

    Args:
        checkpoint_path: checkpoint.pth produced by this script
            (or the original BIOT pretrained checkpoint).
        eeg: np.ndarray / torch.Tensor of shape (N, 62, T) — raw 62-channel
            CogBCI signals (before bipolar projection). (N, 62, 1, T) is also
            accepted.
        montage_types: np.ndarray of shape (N,); 0 = V1 layout, 1 = V2 layout
            (the H5 'montage_type' field). If None, all samples are assumed
            to use the V1 layout.
        batch_size: inference batch size.
        device: 'cuda', 'cuda:0', 'cpu', ...
        model_kwargs: overrides for load_biot_encoder (emb_size, heads, ...).

    Returns:
        np.ndarray of shape (N, emb_size) — one embedding per sample.
    """
    encoder = load_biot_encoder(checkpoint_path, device=device, **model_kwargs)

    eeg = torch.as_tensor(eeg, dtype=torch.float32)
    if eeg.dim() == 4 and eeg.shape[2] == 1:
        eeg = eeg.squeeze(2)  # (N, 62, 1, T) -> (N, 62, T)
    assert eeg.dim() == 3, f"expected (N, C, T), got {tuple(eeg.shape)}"

    if montage_types is None:
        montage_types = np.zeros(len(eeg), dtype=np.int64)
    montage_types = np.asarray(montage_types).astype(np.int64)

    montage_ch_names = get_montage_channel_names()
    projector = BIOT_DataProjector(target_chans=BIPOLAR_TARGET_CHANS)

    # Project 62 raw channels -> 16 bipolar channels, per-sample montage
    projected = torch.zeros(len(eeg), len(BIPOLAR_TARGET_CHANS), eeg.shape[-1])
    for m, ch_names in montage_ch_names.items():
        idx = np.where(montage_types == m)[0]
        if len(idx) > 0:
            projected[idx] = projector.channel_project(eeg[idx], ch_names)
    # Samples with unknown montage type fall back to the V1 layout
    unknown = ~np.isin(montage_types, list(montage_ch_names))
    if unknown.any():
        projected[unknown] = projector.channel_project(
            eeg[unknown], montage_ch_names[0])

    embeddings = []
    for i in range(0, len(projected), batch_size):
        x = projected[i:i + batch_size].to(device)
        embeddings.append(encoder(x).cpu())
    return torch.cat(embeddings, dim=0).numpy()


# ============================================================================
# Main
# ============================================================================
def main(args):
    print("=" * 80)
    print("BIOT Continue Pretraining on CogBCI Multitask Data")
    print("=" * 80)
    print(args)
    print("=" * 80)

    device = torch.device(args.device)

    # Fix seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # Create model (16 bipolar channels fed; n_channels=18 keeps the
    # pretrained channel_tokens fully loadable, tokens 16-17 unused)
    model = get_model(args).to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nNumber of trainable params: {n_parameters}")

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            ckpt_epoch = load_checkpoint(model, args.resume)
            if ckpt_epoch > 0:
                start_epoch = ckpt_epoch + 1
                print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"WARNING: Resume path not found: {args.resume}")

    # Exclude subjects if specified (LOO)
    exclude_list = []
    if args.exclude_subjects:
        exclude_list = [int(s.strip()) for s in args.exclude_subjects.split(',')]
        print(f"  Excluding subjects: {exclude_list}")

    # Load dataset
    print(f"\nLoading dataset from: {args.data_path}")
    base_dataset = EEGDataset(
        h5_path=args.data_path,
        patch_size=args.patch_size,
        normalize=args.normalize,
        normalize_per_channel=args.normalize_per_channel,
    )
    print(f"  Base dataset size: {len(base_dataset)}")

    dataset = ExcludeSubjectDataset(base_dataset, exclude_subjects=exclude_list)

    # Per-sample montage-aware projection: 62 CogBCI -> 16 bipolar channels
    montage_ch_names = get_montage_channel_names()
    if base_dataset.montage_types is None:
        print(f"  WARNING: no 'montage_type' in H5; using a single layout "
              f"(--cogbci_version={args.cogbci_version}) for all samples")
        if args.cogbci_version == 'v2':
            montage_ch_names = {0: montage_ch_names[1]}
    else:
        n_v2 = int((base_dataset.montage_types == 1).sum())
        print(f"  Montage types: {len(base_dataset) - n_v2} V1 samples, "
              f"{n_v2} V2 samples")
    dataset = MontageProjectDataset(dataset, montage_ch_names=montage_ch_names)
    print(f"  Bipolar target channels ({len(BIPOLAR_TARGET_CHANS)}): "
          f"{BIPOLAR_TARGET_CHANS}")

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

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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
            model, train_loader, optimizer, device, epoch,
            lr_schedule, wd_schedule,
            start_steps=epoch * num_training_steps_per_epoch,
            clip_grad=args.clip_grad,
            use_amp=args.use_amp,
            temperature=args.temperature,
        )


        # Validation
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_stats = evaluate(model, val_loader, device, use_amp=args.use_amp,
                                 temperature=args.temperature)
            print(f"\n  Epoch {epoch} Summary:")
            print(f"    Train Loss: {train_stats['loss']:.4f}")
            print(f"    Val Loss:   {val_stats['loss']:.4f}")

            is_best = val_stats['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_stats['loss']
                print(f"    *** New best val loss: {best_val_loss:.4f} ***")
        else:
            val_stats = {'loss': -1}
            is_best = False
            print(f"\n  Epoch {epoch} Summary:")
            print(f"    Train Loss: {train_stats['loss']:.4f}")

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
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
