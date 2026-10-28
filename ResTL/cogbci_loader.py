"""
CogBCI Dataset Loader for ResTL Framework

This module provides data loading functionality for CogBCI dataset
compatible with the ResTL (Resting-state Transfer Learning) method.

Data Structure:
- Multitask data: (trials, 62 channels, 200 timepoints) @ 200Hz
- Resting data: (trials, 62 channels, 200 timepoints) @ 200Hz
- Labels: 11 classes (10-42, mapped to 0-10)
- Subjects: Multiple subjects with session information
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from scipy import signal
from typing import Tuple, Dict, List, Optional


class CogBCIDataset(Dataset):
    """
    CogBCI Dataset for ResTL framework with triplet sampling.
    
    Returns dictionary with keys:
        - x_anc: Anchor sample (1, time_points, channels)
        - x_pos: Positive sample (1, time_points, channels)  
        - x_neg: Negative sample (1, time_points, channels)
        - label: One-hot encoded label (n_classes,)
        - x_anc_rest: Resting state for anchor (1, time_points, channels)
        - x_pos_rest: Resting state for positive (1, time_points, channels)
        - x_neg_rest: Resting state for negative (1, time_points, channels)
    
    Args:
        task_h5_path: Path to multitask H5 file
        rest_h5_path: Path to resting state H5 file
        target_subject_id: Target subject ID (None for all subjects)
        target_task_id: List of task IDs to include (None for all tasks)
        n_classes: Number of classes (default: 11)
        normalize: Whether to apply Z-score normalization
        normalize_per_channel: If True, normalize each channel independently
    """
    
    # Label mapping from original labels to 0-5
    # Data contains labels: 0, 1, 2, 10, 11, 12
    # We remap them to contiguous indices 0-5
    LABEL_MAPPING = {
        0: 0, 1: 1, 2: 2,      # First task
        10: 3, 11: 4, 12: 5    # Second task (e.g., Flanker)
    }
    
    def __init__(
        self,
        task_h5_path: str,
        rest_h5_path: str,
        target_subject_id: Optional[int] = None,
        target_task_id: Optional[List[int]] = None,
        n_classes: int = 6,  # CogBCI has 6 classes
        normalize: bool = True,
        normalize_per_channel: bool = True,
        is_training: str = 'train'
    ):
        self.task_h5_path = task_h5_path
        self.rest_h5_path = rest_h5_path
        self.target_subject_id = target_subject_id
        self.target_task_id = target_task_id
        self.n_classes = n_classes
        self.normalize = normalize
        self.normalize_per_channel = normalize_per_channel
        self.is_training = is_training
        self.finetuning = False
        
        # Load data
        self._load_task_data()
        self._load_rest_data()
        self._build_indices()
        
    def _load_task_data(self):
        """Load multitask EEG data from H5 file."""
        print(f"Loading CogBCI task data from: {self.task_h5_path}")
        
        with h5py.File(self.task_h5_path, 'r') as f:
            # Load EEG data: (trials, channels, timepoints)
            # self.data = f['data'][:].astype(np.float32)  # (N, 62, 200)
            if 'data' in f:
                self.data = f['data'][:].astype(np.float32)
            elif 'eeg' in f:
                self.data = f['eeg'][:].astype(np.float32)
            else:
                raise KeyError("No 'data' key found in H5 file. Available keys: " + str(list(f.keys())))
            # self.labels = f['labels'][:].astype(np.int64)
            if 'labels' in f:
                self.labels = f['labels'][:]
                # self.labels = torch.tensor(self.labels).long()
            else:
                raise KeyError("No 'labels' key found in H5 file")
            # self.subject_ids = f['subject_ids'][:].astype(np.int64)
            # Load subject IDs
            if 'subject_ids' in f:
                self.subject_ids = f['subject_ids'][:]
            elif 'subject' in f:
                self.subject_ids = f['subject'][:]
            elif 'subject_id' in f:
                self.subject_ids = f['subject_id'][:]
            else:
                raise KeyError("No 'subject_ids' key found in H5 file")
            # self.tasks = f['task'][:].astype(np.int64)
            if 'task' in f:
                self.tasks = f['task'][:]
                print(f"    Tasks: {self.tasks.shape} (Loaded)")
            # self.sessions = f['sessions'][:].astype(np.int64)
            # in _load_task_data
            if 'sessions' in f:
                self.sessions = f['sessions'][:].astype(np.int64)
            else:
                self.sessions = np.zeros(len(self.data), dtype=np.int64)
            
        print(f"  Raw data shape: {self.data.shape}")
        print(f"  Unique subjects: {np.unique(self.subject_ids)}")
        print(f"  Unique labels: {np.unique(self.labels)}")
        print(f"  Unique tasks: {np.unique(self.tasks)}")
        
        # Apply label mapping
        self._remap_labels()
        
        # Filter by subject if specified
        if self.target_subject_id is not None:
            mask = (self.subject_ids == self.target_subject_id)
            self._apply_mask(mask)
            print(f"  Filtered to subject {self.target_subject_id}: {len(self.data)} trials")
            
        # Filter by task if specified
        if self.target_task_id is not None:
            mask = np.isin(self.tasks, self.target_task_id)
            self._apply_mask(mask)
            print(f"  Filtered to tasks {self.target_task_id}: {len(self.data)} trials")
        
        # Apply normalization
        if self.normalize:
            self._normalize_data()
            
        # Transpose to (trials, timepoints, channels) for ResTL compatibility
        # ResTL expects: (batch, 1, time_points, channels)
        self.data = np.transpose(self.data, (0, 2, 1))  # (N, 200, 62)
        print(f"  Final task data shape: {self.data.shape} (trials, time, channels)")
        
    def _load_rest_data(self):
        """Load resting state EEG data from H5 file."""
        print(f"Loading CogBCI resting data from: {self.rest_h5_path}")
        
        with h5py.File(self.rest_h5_path, 'r') as f:
            # Load resting EEG: (trials, channels, timepoints)
            rest_eeg = f['eeg'][:].astype(np.float32)  # (N, 62, 200)
            rest_subjects = f['subject'][:].astype(np.int64)
            rest_sessions = f['session'][:].astype(np.int64)
            
        print(f"  Raw resting data shape: {rest_eeg.shape}")
        
        # Apply same normalization
        if self.normalize:
            if self.normalize_per_channel:
                mean = rest_eeg.mean(axis=(0, 2), keepdims=True)
                std = rest_eeg.std(axis=(0, 2), keepdims=True)
                std = np.where(std == 0, 1.0, std)
                rest_eeg = (rest_eeg - mean) / std
            else:
                mean = rest_eeg.mean()
                std = rest_eeg.std()
                if std == 0:
                    std = 1.0
                rest_eeg = (rest_eeg - mean) / std
        
        # Transpose to (trials, timepoints, channels)
        rest_eeg = np.transpose(rest_eeg, (0, 2, 1))  # (N, 200, 62)
        
        # Organize resting data by subject
        self.rest_data_by_subject = {}
        for subj_id in np.unique(rest_subjects):
            mask = (rest_subjects == subj_id)
            self.rest_data_by_subject[subj_id] = rest_eeg[mask]
            
        print(f"  Resting data organized for subjects: {list(self.rest_data_by_subject.keys())}")
        
    def _remap_labels(self):
        """Remap original labels (10-42) to 0-10."""
        remapped = np.array([self.LABEL_MAPPING.get(l, l) for l in self.labels])
        self.labels = remapped
        
    def _apply_mask(self, mask: np.ndarray):
        """Apply boolean mask to all data arrays."""
        self.data = self.data[mask]
        self.labels = self.labels[mask]
        self.subject_ids = self.subject_ids[mask]
        self.tasks = self.tasks[mask]
        self.sessions = self.sessions[mask]
        
    def _normalize_data(self):
        """Apply Z-score normalization."""
        print(f"  Applying Z-score normalization (per_channel={self.normalize_per_channel})...")
        
        if self.normalize_per_channel:
            # Per-channel normalization
            mean = self.data.mean(axis=(0, 2), keepdims=True)  # (1, 62, 1)
            std = self.data.std(axis=(0, 2), keepdims=True)
            std = np.where(std == 0, 1.0, std)
            self.data = (self.data - mean) / std
        else:
            # Global normalization
            mean = self.data.mean()
            std = self.data.std()
            if std == 0:
                std = 1.0
            self.data = (self.data - mean) / std
            
        print(f"  Normalized data range: [{self.data.min():.4f}, {self.data.max():.4f}]")
        
    def _build_indices(self):
        """Build index mapping for triplet sampling."""
        # Create mapping: (subject, label) -> list of indices
        self.indices_by_subject_label = {}
        for subj in np.unique(self.subject_ids):
            for label in range(self.n_classes):
                mask = (self.subject_ids == subj) & (self.labels == label)
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    self.indices_by_subject_label[(subj, label)] = indices
                    
        print(f"  Built index mapping for {len(self.indices_by_subject_label)} (subject, label) pairs")
        
    def _get_rest_sample(self, subject_id: int) -> np.ndarray:
        """Get a random resting state sample for given subject."""
        if subject_id in self.rest_data_by_subject:
            rest_samples = self.rest_data_by_subject[subject_id]
            idx = np.random.randint(len(rest_samples))
            return rest_samples[idx]
        else:
            # If no resting data for this subject, return zeros
            return np.zeros_like(self.data[0])
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a triplet sample for ResTL training.
        
        Returns dictionary with:
            - x_anc, x_pos, x_neg: Task EEG samples
            - label: One-hot encoded label
            - x_anc_rest, x_pos_rest, x_neg_rest: Corresponding resting EEG
        """
        # Anchor sample
        anchor = self.data[idx]
        anchor_label = self.labels[idx]
        anchor_subject = self.subject_ids[idx]
        
        # Create one-hot label
        label_onehot = np.zeros(self.n_classes)
        label_onehot[anchor_label] = 1.0
        
        # In test mode, only return anchor sample (no triplet needed)
        if self.is_training == 'test':
            anchor_rest = self._get_rest_sample(anchor_subject)
            return {
                'x_anc': torch.from_numpy(anchor).unsqueeze(0).float(),      # (1, T, C)
                'x_pos': torch.from_numpy(anchor).unsqueeze(0).float(),      # Same as anc for test
                'x_neg': torch.from_numpy(anchor).unsqueeze(0).float(),      # Same as anc for test
                'label': torch.from_numpy(label_onehot).float(),              # (n_classes,)
                'x_anc_rest': torch.from_numpy(anchor_rest).unsqueeze(0).float(),
                'x_pos_rest': torch.from_numpy(anchor_rest).unsqueeze(0).float(),
                'x_neg_rest': torch.from_numpy(anchor_rest).unsqueeze(0).float(),
            }
        
        # Sample positive (same subject, same label)
        pos_indices = self.indices_by_subject_label.get(
            (anchor_subject, anchor_label), np.array([idx])
        )
        # Exclude anchor itself if possible
        pos_indices = pos_indices[pos_indices != idx]
        if len(pos_indices) == 0:
            pos_idx = idx
        else:
            pos_idx = np.random.choice(pos_indices)
        positive = self.data[pos_idx]
        
        # Sample negative (same subject, different label)
        other_labels = [l for l in range(self.n_classes) if l != anchor_label]
        neg_label = np.random.choice(other_labels)
        neg_indices = self.indices_by_subject_label.get(
            (anchor_subject, neg_label), np.array([idx])
        )
        if len(neg_indices) == 0:
            # If no samples for this label, pick random
            neg_idx = np.random.randint(len(self.data))
        else:
            neg_idx = np.random.choice(neg_indices)
        negative = self.data[neg_idx]
        
        # Get corresponding resting state samples
        anchor_rest = self._get_rest_sample(anchor_subject)
        positive_rest = self._get_rest_sample(anchor_subject)
        negative_rest = self._get_rest_sample(anchor_subject)
        
        return {
            'x_anc': torch.from_numpy(anchor).unsqueeze(0).float(),      # (1, T, C)
            'x_pos': torch.from_numpy(positive).unsqueeze(0).float(),    # (1, T, C)
            'x_neg': torch.from_numpy(negative).unsqueeze(0).float(),    # (1, T, C)
            'label': torch.from_numpy(label_onehot).float(),              # (n_classes,)
            'x_anc_rest': torch.from_numpy(anchor_rest).unsqueeze(0).float(),
            'x_pos_rest': torch.from_numpy(positive_rest).unsqueeze(0).float(),
            'x_neg_rest': torch.from_numpy(negative_rest).unsqueeze(0).float(),
        }


class CogBCIRestDataset(Dataset):
    """
    CogBCI Resting State Dataset for ResTL test-time adaptation.
    
    Supports two modes:
    1. Raw mode (for generate_signal_from_RS): returns tensor
    2. Dict mode (for one_epoch_distribution with train='RS'): returns dict
    
    Args:
        rest_h5_path: Path to resting state H5 file
        target_subject_id: Target subject ID
        normalize: Whether to apply Z-score normalization
        normalize_per_channel: If True, normalize each channel independently
        return_dict: If True, return dict format for one_epoch_distribution
        n_classes: Number of classes (default: 6 for CogBCI)
        max_samples: Maximum number of resting state samples to use (None for all)
    """
    
    def __init__(
        self,
        rest_h5_path: str,
        target_subject_id: Optional[int] = None,
        normalize: bool = True,
        normalize_per_channel: bool = True,
        return_dict: bool = False,
        n_classes: int = 6,
        max_samples: Optional[int] = None
    ):
        self.rest_h5_path = rest_h5_path
        self.target_subject_id = target_subject_id
        self.normalize = normalize
        self.normalize_per_channel = normalize_per_channel
        self.return_dict = return_dict
        self.n_classes = n_classes
        self.max_samples = max_samples
        
        self._load_data()
        self._organize_by_session()
        self._build_index_mapping()
        
    def _load_data(self):
        """Load resting state data."""
        print(f"Loading CogBCI resting data from: {self.rest_h5_path}")
        
        with h5py.File(self.rest_h5_path, 'r') as f:
            rest_eeg = f['eeg'][:].astype(np.float32)  # (N, 62, 200)
            self.subject_ids = f['subject'][:].astype(np.int64)
            self.sessions = f['session'][:].astype(np.int64)
            
        # Apply normalization
        if self.normalize:
            if self.normalize_per_channel:
                mean = rest_eeg.mean(axis=(0, 2), keepdims=True)
                std = rest_eeg.std(axis=(0, 2), keepdims=True)
                std = np.where(std == 0, 1.0, std)
                rest_eeg = (rest_eeg - mean) / std
            else:
                mean = rest_eeg.mean()
                std = rest_eeg.std()
                if std == 0:
                    std = 1.0
                rest_eeg = (rest_eeg - mean) / std
        
        # Transpose to (trials, timepoints, channels)
        self.data = np.transpose(rest_eeg, (0, 2, 1))  # (N, 200, 62)
        
        # Filter by subject
        if self.target_subject_id is not None:
            mask = (self.subject_ids == self.target_subject_id)
            self.data = self.data[mask]
            self.subject_ids = self.subject_ids[mask]
            self.sessions = self.sessions[mask]
            print(f"  Filtered to subject {self.target_subject_id}: {len(self.data)} samples")
            
        print(f"  Final resting data shape: {self.data.shape}")
        
        # Limit samples if max_samples is specified
        if self.max_samples is not None and len(self.data) > self.max_samples:
            print(f"  Limiting to {self.max_samples} samples (from {len(self.data)})")
            indices = np.random.choice(len(self.data), self.max_samples, replace=False)
            self.data = self.data[indices]
            self.subject_ids = self.subject_ids[indices]
            self.sessions = self.sessions[indices]
        
    def _organize_by_session(self):
        """Organize data by session for multi-session testing."""
        self.testset = {}
        unique_sessions = np.unique(self.sessions)
        
        # Use 0-based index to match range(args.sessionnum) in main script
        for idx, sess in enumerate(unique_sessions):
            mask = (self.sessions == sess)
            indices = np.where(mask)[0]
            # Store as 4D tensor (N, 1, time, channels) to match model input
            data_4d = torch.from_numpy(self.data[indices]).float().unsqueeze(1)  # Add channel dim
            self.testset[idx] = {
                'x': data_4d,  # (N, 1, time, channels)
                'y': torch.zeros(len(indices), self.n_classes)  # Placeholder labels
            }
        print(f"  Organized into {len(unique_sessions)} sessions: {list(range(len(unique_sessions)))} (mapped from {list(unique_sessions)})")
        
        # When in dict mode, use testset data for __getitem__
        # if self.return_dict:
        #     self._build_index_mapping()
    
    def _build_index_mapping(self):
        """Build index mapping for dict mode iteration over testset."""
        self._index_map = []
        for session_id in sorted(self.testset.keys()):
            x_data = self.testset[session_id]['x']
            for i in range(len(x_data)):
                self._index_map.append((session_id, i))
    
    def set_return_dict(self, return_dict: bool):
        """Switch between tensor mode and dict mode."""
        self.return_dict = return_dict
        if return_dict:
            # Always rebuild index mapping in case testset was updated
            self._build_index_mapping()
        
    def __len__(self):
        if self.return_dict:
            return len(getattr(self, '_index_map', []))
        return len(self.data)
        
    def __getitem__(self, idx: int):
        """
        Return data based on mode:
        - Tensor mode: return resting state sample (1, timepoints, channels) for generate_signal_from_RS
        - Dict mode: return dict with x_anc, label, etc. for one_epoch_distribution
        """
        # if self.return_dict:
        # Dict mode: return format expected by one_epoch_distribution
        session_id, sample_idx = self._index_map[idx]
        x = self.testset[session_id]['x'][sample_idx]  # (1, time, channels) - 4D after update
        y = self.testset[session_id]['y'][sample_idx]  # (n_classes,)
        
        # x is already 4D (1, time, channels) or should be squeezed
        if x.dim() == 4:
            x = x.squeeze(0)  # Remove batch dim if present
        
        # For RS mode, all samples are the same (resting state)
        # Return dict with keys expected by one_epoch_distribution
        return {
            'x_anc': x,
            'x_pos': x,
            'x_neg': x,
            'label': y,
            'x_anc_rest': x,
            'x_pos_rest': x,
            'x_neg_rest': x,
        }
        # else:
        #     # Tensor mode: return 3D tensor (1, time, channels) for generate_signal_from_RS
        #     # DataLoader will stack to (batch, 1, time, channels) which matches model input
        #     x = torch.from_numpy(self.data[idx]).float()  # (time, channels)
        #     return x.unsqueeze(0)  # (1, time, channels) - DataLoader -> (batch, 1, time, channels)


def create_cogbci_dataloaders(
    task_h5_path: str,
    rest_h5_path: str,
    target_subject_id: int,
    target_task_id: Optional[List[int]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    random_state: int = 42,
    normalize: bool = True,
    normalize_per_channel: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, CogBCIRestDataset]:
    """
    Create train/val/test DataLoaders for CogBCI dataset.
    
    Args:
        task_h5_path: Path to multitask H5 file
        rest_h5_path: Path to resting H5 file
        target_subject_id: Target subject ID for splitting
        target_task_id: List of task IDs to include
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        batch_size: Batch size
        num_workers: Number of data loading workers
        random_state: Random seed
        normalize: Whether to normalize
        normalize_per_channel: Per-channel normalization
        
    Returns:
        train_loader, val_loader, test_loader, rest_dataset
    """
    # Create full dataset
    full_dataset = CogBCIDataset(
        task_h5_path=task_h5_path,
        rest_h5_path=rest_h5_path,
        target_subject_id=target_subject_id,
        target_task_id=target_task_id,
        normalize=normalize,
        normalize_per_channel=normalize_per_channel,
        is_training='train'
    )
    
    # Split by class to ensure balanced distribution
    np.random.seed(random_state)
    train_indices = []
    val_indices = []
    test_indices = []
    
    for label in range(full_dataset.n_classes):
        label_mask = (full_dataset.labels == label)
        label_indices = np.where(label_mask)[0]
        
        if len(label_indices) == 0:
            continue
            
        # Shuffle indices
        np.random.shuffle(label_indices)
        
        n_total = len(label_indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_indices.extend(label_indices[:n_train])
        val_indices.extend(label_indices[n_train:n_train + n_val])
        test_indices.extend(label_indices[n_train + n_val:])
        
        print(f"Class {label}: Total={n_total}, Train={n_train}, Val={n_val}, Test={n_total - n_train - n_val}")
    
    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Update is_training attribute for test dataset
    for idx in test_indices:
        full_dataset.is_training = 'test'
        break
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # ResTL requires batch_size=1 for testing
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create resting state dataset for test-time adaptation
    rest_dataset = CogBCIRestDataset(
        rest_h5_path=rest_h5_path,
        target_subject_id=target_subject_id,
        normalize=normalize,
        normalize_per_channel=normalize_per_channel
    )
    
    print(f"\nFinal Split Summary:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val:   {len(val_indices)} samples")
    print(f"  Test:  {len(test_indices)} samples")
    
    return train_loader, val_loader, test_loader, rest_dataset


# ===== CogBCI Dataset Configuration =====
# Note: Only Session 1 data is used

COGBCI_CONFIG = {
    'data_path': '/mnt/dataset4/fuxy/m3cv',
    'multitask_file': '/mnt/dataset4/fuxy/m3cv/processed_cogbci_multitask/cogbci_multitask_session1.h5',
    'resting_file': '/mnt/dataset4/fuxy/m3cv/processed_cogbci_session1/cogbci_resting.h5',
    'n_classes': 6,
    'n_channels': 62,
    'n_timepoints': 200,
    'sfreq': 200,  # Hz
    'task_mapping': {
            0: 0, 1: 1, 2: 2,
            10: 3, 11: 4, 12: 5
        }
}


if __name__ == '__main__':
    # Test the data loader
    print("=" * 60)
    print("Testing CogBCI Data Loader")
    print("=" * 60)
    
    # Load single subject data (Session 1 only)
    train_loader, val_loader, test_loader, rest_dataset = create_cogbci_dataloaders(
        task_h5_path=COGBCI_CONFIG['multitask_file'],
        rest_h5_path=COGBCI_CONFIG['resting_file'],
        target_subject_id=1,
        target_task_id=[0,1],  # Flanker + MATB tasks
        batch_size=8,
        num_workers=0
    )
    
    # Test loading a batch
    print("\n" + "=" * 60)
    print("Testing batch loading")
    print("=" * 60)
    
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  x_anc shape: {batch['x_anc'].shape}")
        print(f"  x_pos shape: {batch['x_pos'].shape}")
        print(f"  x_neg shape: {batch['x_neg'].shape}")
        print(f"  label shape: {batch['label'].shape}")
        print(f"  x_anc_rest shape: {batch['x_anc_rest'].shape}")
        break
    
    # Test resting dataset
    print("\n" + "=" * 60)
    print("Testing Rest Dataset")
    print("=" * 60)
    print(f"Rest dataset size: {len(rest_dataset)}")
    print(f"Sessions: {list(rest_dataset.testset.keys())}")
    
    rest_sample = rest_dataset[0]
    print(f"Rest sample shape: {rest_sample.shape}")
    
    print("\n✓ CogBCI data loader test completed!")
