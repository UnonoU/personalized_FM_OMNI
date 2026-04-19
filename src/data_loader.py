
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import h5py
from scipy import signal
from sklearn.model_selection import train_test_split, GroupKFold
from typing import Tuple, List


class EEGDataset(Dataset):
    """
    Dataset for EEG data with automatic resampling, Z-score normalization and patch reshaping.

    Features:
    - Automatic resampling: 250Hz -> 200Hz for LaBraM compatibility
    - Z-score normalization (per-channel or global)
    - Patch reshaping for transformer models

    Args:
        h5_path: Path to H5 file containing EEG data
        patch_size: Size of each patch for reshaping (default: 200)
        normalize: Whether to apply Z-score normalization (default: True)
        normalize_per_channel: If True, normalize each channel independently;
                               if False, use global normalization (default: True)
        transform: Optional transform to apply to EEG data
    """

    def __init__(
        self,
        h5_path: str,
        patch_size: int = 200,
        normalize: bool = True,
        normalize_per_channel: bool = True,
        transform=None
    ):
        self.h5_path = h5_path
        self.patch_size = patch_size
        self.normalize = normalize
        self.normalize_per_channel = normalize_per_channel
        self.transform = transform

        # Load and preprocess data
        self._load_data()

    def _load_data(self):
        """
        Load EEG data, labels, and subject IDs from H5 file.

        Processing steps:
        1. Load data from H5 file
        2. Resample if needed (250Hz -> 200Hz)
        3. Apply Z-score normalization
        4. Verify normalization
        """
        print(f"Loading EEG data from: {self.h5_path}")

        with h5py.File(self.h5_path, 'r') as f:
            # Print available keys for debugging
            print(f"  Available keys in H5 file: {list(f.keys())}")

            # Load EEG data - shape should be (n_trials, n_channels, n_timepoints)
            if 'data' in f:
                self.data = f['data'][:].astype(np.float32)
            elif 'eeg' in f:
                self.data = f['eeg'][:].astype(np.float32)
            else:
                raise KeyError("No 'data' key found in H5 file. Available keys: " + str(list(f.keys())))

            # Load labels
            if 'labels' in f:
                self.labels = f['labels'][:]
                self.labels = torch.tensor(self.labels).long()
            else:
                raise KeyError("No 'labels' key found in H5 file")
            
            if 'task' in f:
                self.tasks = f['task'][:]
                print(f"    Tasks: {self.tasks.shape} (Loaded)")

            # Load subject IDs
            if 'subject_ids' in f:
                self.subject_ids = f['subject_ids'][:]
            elif 'subject' in f:
                self.subject_ids = f['subject'][:]
            elif 'subject_id' in f:
                self.subject_ids = f['subject_id'][:]
            else:
                raise KeyError("No 'subject_ids' key found in H5 file")

            # Load metadata if available
            if hasattr(f, 'attrs'):
                print(f"  Metadata:")
                for key, value in f.attrs.items():
                    print(f"    {key}: {value}")

        print(f"\n  Data shapes:")
        print(f"    EEG data: {self.data.shape} (Trials, Channels, Time)")
        print(f"    Labels: {self.labels.shape}")
        print(f"    Subject IDs: {self.subject_ids.shape}")
        print(f"\n  Statistics (Before Resampling):")
        print(f"    Number of trials: {len(self.data)}")
        print(f"    Unique subjects: {len(np.unique(self.subject_ids))}")
        print(f"    Label distribution: {dict(zip(*np.unique(self.labels, return_counts=True)))}")
        print(f"    Data range: [{self.data.min():.4f}, {self.data.max():.4f}]")
        print(f"    Global mean: {self.data.mean():.4f}, Global std: {self.data.std():.4f}")

        # Resample if time dimension is 250 (250Hz -> 200Hz)
        n_trials, n_channels, n_timepoints = self.data.shape
        if n_timepoints == 250:
            print(f"\n  Resampling detected: Time dimension is 250")
            print(f"    Resampling from 250Hz to 200Hz using scipy.signal.resample...")

            # Resample each trial from 250 to 200 timepoints
            resampled_data = np.zeros((n_trials, n_channels, 200), dtype=np.float32)

            for i in range(n_trials):
                for ch in range(n_channels):
                    # Resample from 250 to 200 timepoints
                    resampled_data[i, ch, :] = signal.resample(self.data[i, ch, :], 200)

            self.data = resampled_data

            print(f"    ✓ Resampling completed!")
            print(f"    New shape: {self.data.shape} (Trials, Channels, Time)")
            print(f"    Data range after resampling: [{self.data.min():.4f}, {self.data.max():.4f}]")
        elif n_timepoints == 200:
            print(f"\n  No resampling needed: Time dimension is already 200")
        elif n_timepoints % self.patch_size != 0:
            print(f"\n  ⚠ Warning: Time dimension ({n_timepoints}) is not 250 or divisible by patch_size ({self.patch_size})")
            print(f"    Attempting to resample to nearest multiple of patch_size...")

            # Calculate target length as the nearest multiple of patch_size
            target_length = (n_timepoints // self.patch_size) * self.patch_size
            if target_length == 0:
                target_length = self.patch_size

            print(f"    Resampling from {n_timepoints} to {target_length} timepoints...")

            resampled_data = np.zeros((n_trials, n_channels, target_length), dtype=np.float32)
            for i in range(n_trials):
                for ch in range(n_channels):
                    resampled_data[i, ch, :] = signal.resample(self.data[i, ch, :], target_length)

            self.data = resampled_data
            print(f"    ✓ Resampling completed!")
            print(f"    New shape: {self.data.shape}")

        # Apply Z-score normalization
        if self.normalize:
            print(f"\n  Applying Z-score normalization...")
            if self.normalize_per_channel:
                # Normalize each channel independently
                print(f"    Method: Per-channel normalization")
                # Calculate mean and std for each channel across all trials and timepoints
                # Shape: (n_channels,)
                mean = self.data.mean(axis=(0, 2), keepdims=True)  # Shape: (1, n_channels, 1)
                std = self.data.std(axis=(0, 2), keepdims=True)    # Shape: (1, n_channels, 1)

                # Avoid division by zero
                std = np.where(std == 0, 1.0, std)

                self.data = (self.data - mean) / std

                print(f"    Per-channel mean shape: {mean.shape}")
                print(f"    Per-channel std shape: {std.shape}")
            else:
                # Global normalization
                print(f"    Method: Global normalization")
                mean = self.data.mean()
                std = self.data.std()

                if std == 0:
                    std = 1.0

                self.data = (self.data - mean) / std

                print(f"    Global mean: {mean:.4f}")
                print(f"    Global std: {std:.4f}")

            # Verification: Check normalization results
            print(f"\n  Normalization Verification:")
            normalized_mean = self.data.mean()
            normalized_std = self.data.std()
            print(f"    Normalized data mean: {normalized_mean:.6f}")
            print(f"    Normalized data std: {normalized_std:.6f}")
            print(f"    Normalized data range: [{self.data.min():.4f}, {self.data.max():.4f}]")

            # Check if normalization is correct
            mean_threshold = 1e-5
            std_threshold = 0.05

            if abs(normalized_mean) < mean_threshold:
                print(f"    ✓ Mean check passed: |{normalized_mean:.6f}| < {mean_threshold}")
            else:
                print(f"    ⚠ Warning: Mean check failed: |{normalized_mean:.6f}| >= {mean_threshold}")

            if abs(normalized_std - 1.0) < std_threshold:
                print(f"    ✓ Std check passed: |{normalized_std:.6f} - 1.0| < {std_threshold}")
            else:
                print(f"    ⚠ Warning: Std check failed: |{normalized_std:.6f} - 1.0| >= {std_threshold}")

            if self.normalize_per_channel:
                print(f"\n    Per-channel statistics:")
                for ch_idx in range(min(5, self.data.shape[1])):  # Show first 5 channels
                    ch_mean = self.data[:, ch_idx, :].mean()
                    ch_std = self.data[:, ch_idx, :].std()
                    print(f"      Channel {ch_idx}: mean={ch_mean:.6f}, std={ch_std:.6f}")
                if self.data.shape[1] > 5:
                    print(f"      ... (showing first 5 of {self.data.shape[1]} channels)")

        # Validate patch size
        n_channels, n_timepoints = self.data.shape[1], self.data.shape[2]
        if n_timepoints % self.patch_size != 0:
            raise ValueError(
                f"Time dimension ({n_timepoints}) must be divisible by patch_size ({self.patch_size})"
            )
        self.num_patches = n_timepoints // self.patch_size
        print(f"\n  Patch configuration:")
        print(f"    Channels: {n_channels}")
        print(f"    Num patches: {self.num_patches}")
        print(f"    Patch size: {self.patch_size}")
        print(f"    Output shape per sample: ({n_channels}, {self.num_patches}, {self.patch_size})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            A dictionary containing:
                eeg: Tensor of shape (Channels, Num_Patches, Patch_Size)
                label: Label (0 or 1)
                subject_id: Subject identifier
        """
        eeg = self.data[idx]  # Already float32 and normalized
        label = self.labels[idx]
        subject_id = self.subject_ids[idx]

        eeg = np.clip(eeg, -6.0, 6.0)

        if self.transform:
            eeg = self.transform(eeg)

        # Reshape: (C, T) -> (C, Num_Patches, Patch_Size)
        n_channels = eeg.shape[0]
        eeg_reshaped = eeg.reshape(n_channels, self.num_patches, self.patch_size)

        return {
            'eeg': torch.from_numpy(eeg_reshaped),
            'label': label,
            'subject_id': subject_id
        }

    def get_subject_indices(self):
        """Get trial indices for each subject."""
        subject_to_indices = {}
        for idx, subj in enumerate(self.subject_ids):
            if subj not in subject_to_indices:
                subject_to_indices[subj] = []
            subject_to_indices[subj].append(idx)
        return subject_to_indices



def create_multitask_single_subject_dataloaders(
    dataset: EEGDataset,
    target_subject_id: int,
    target_task_id: list,
    train_size: float = 0.7,
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 32,
    random_state: int = 42,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    针对单个被试执行【类内随机切分】：确保每个集合都包含所有类别。
    """
    # 1. 获取该被试的所有索引
    mask = (dataset.subject_ids == target_subject_id)
    
    if target_task_id is not None:
        # mask &= (dataset.tasks == target_task_id)
        mask &= np.isin(dataset.tasks, target_task_id)
    
    all_indices = np.where(mask)[0]

    # 2. 准备容器
    train_idx_list, val_idx_list, test_idx_list = [], [], []
    
    # 3. 获取该被试包含的所有类别标签
    unique_labels = np.unique(dataset.labels[all_indices])
    
    print(f"\n[Random Split] Processing Subject: {target_subject_id}")
    
    # 4. 对每个类别独立进行随机切分
    for label in unique_labels:
        # 提取该被试下属于当前类别的所有索引
        label_indices = all_indices[dataset.labels[all_indices] == label]
        
        # 随机打乱该类别的索引
        np.random.seed(random_state)
        np.random.shuffle(label_indices)
        
        n_total = len(label_indices)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = int(n_total * train_size)
        
        # 随机切分
        t_idx = label_indices[:n_train]
        v_idx = label_indices[n_train : n_train + n_val]
        if n_test+n_val+n_train >= n_total:
            s_idx = label_indices[n_train + n_val:]
        else:
            s_idx = label_indices[n_train + n_val:n_test+n_val+n_train]
        
        train_idx_list.extend(t_idx)
        val_idx_list.extend(v_idx)
        test_idx_list.extend(s_idx)
        
        print(f"  Class {label}: Total={n_total}, Train={len(t_idx)}, Val={len(v_idx)}, Test={len(s_idx)}")

    # 5. 转换为 numpy 数组
    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list)
    test_idx = np.array(test_idx_list)

    # 6. 辅助打印：检查最终分布
    def get_dist(indices):
        # print(indices.dtype)
        # indices = indices.astype(int)
        labels = dataset.labels[indices]
        return dict(zip(*np.unique(labels, return_counts=True)))

    print(f"\nFinal Split Summary for Subject {target_subject_id}:")
    print(f"  Train Set: {len(train_idx)} trials, Dist: {get_dist(train_idx)}")
    print(f"  Val Set:   {len(val_idx)} trials, Dist: {get_dist(val_idx)}")
    print(f"  Test Set:  {len(test_idx)} trials, Dist: {get_dist(test_idx)}")

    # 7. 构建 DataLoader
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader




def create_single_subject_dataloaders_order(
    dataset: EEGDataset,
    target_subject_id: int,
    train_size: float = 0.7,
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 32,
    random_state: int = 42,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    针对单个被试执行【类内顺序切分】：确保每个集合都包含所有类别，且时间上完全隔离。
    """
    # 1. 获取该被试的所有索引
    all_indices = np.where(dataset.subject_ids == target_subject_id)[0]
    if len(all_indices) == 0:
        raise ValueError(f"Subject ID {target_subject_id} not found in dataset.")

    # 2. 准备容器
    train_idx_list, val_idx_list, test_idx_list = [], [], []
    
    # 3. 获取该被试包含的所有类别标签
    unique_labels = np.unique(dataset.labels[all_indices])
    
    print(f"\n[Within-Class Chronological Split] Processing Subject: {target_subject_id}")
    
    # 4. 对每个类别独立进行顺序切分
    for label in unique_labels:
        # 提取该被试下属于当前类别的所有索引，并确保按时间排序
        label_indices = all_indices[dataset.labels[all_indices] == label]
        label_indices = np.sort(label_indices)
        
        n_total = len(label_indices)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        # n_train = n_total - n_test - n_val
        n_train = int(n_total * train_size)
        
        # 顺序切分
        t_idx = label_indices[:n_train]
        v_idx = label_indices[n_train : n_train + n_val]
        if n_test+n_val+n_train >= n_total:
            s_idx = label_indices[n_train + n_val:]
        else:
            s_idx = label_indices[n_train + n_val:n_test+n_val+n_train]
        
        train_idx_list.extend(t_idx)
        val_idx_list.extend(v_idx)
        test_idx_list.extend(s_idx)
        
        print(f"  Class {label}: Total={n_total}, Train={len(t_idx)}, Val={len(v_idx)}, Test={len(s_idx)}")

    # 5. 转换为 numpy 数组
    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list)
    test_idx = np.array(test_idx_list)

    # 6. 辅助打印：检查最终分布
    def get_dist(indices):
        labels = dataset.labels[indices]
        return dict(zip(*np.unique(labels, return_counts=True)))

    print(f"\nFinal Split Summary for Subject {target_subject_id}:")
    print(f"  Train Set: {len(train_idx)} trials, Dist: {get_dist(train_idx)}")
    print(f"  Val Set:   {len(val_idx)} trials, Dist: {get_dist(val_idx)}")
    print(f"  Test Set:  {len(test_idx)} trials, Dist: {get_dist(test_idx)}")

    # 7. 构建 DataLoader
    # 训练集开启 shuffle=True 是为了让 Batch 内的梯度更多样，不代表时间泄露
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np

class ContrastiveDataset(Dataset):
    """
    自定义对比学习数据集，用于从目标被试数据和其他被试数据中生成对比学习数据对。
    """
    def __init__(self, dataset, target_indices, other_indices, transform=None):
        """
        初始化对比数据集。
        
        Args:
            dataset: 原始数据集对象
            target_indices: 目标被试的索引
            other_indices: 其他被试的索引，用于对比样本
            transform: 可选的数据变换操作（如数据增强）
        """
        self.dataset = dataset
        self.target_indices = target_indices
        self.other_indices = other_indices
        self.transform = transform

    def __len__(self):
        """
        返回目标被试的样本数
        """
        return len(self.target_indices)

    def __getitem__(self, idx):
        """
        获取目标样本和对比样本
        
        Args:
            idx: 当前目标样本的索引
        
        Returns:
            data: 目标样本
            contrastive_data: 对比样本
            label: 目标样本的标签
            contrastive_label: 对比样本的标签
        """
        # 获取目标样本和标签
        sample = self.dataset[self.target_indices[idx]]
        data = sample['eeg']
        label = sample['label']

        # 如果需要，应用变换
        if self.transform:
            data = self.transform(data)

        # 从其他被试中随机选取一个对比样本
        contrastive_idx = np.random.choice(self.other_indices)
        contrastive_sample = self.dataset[contrastive_idx]
        contrastive_data = contrastive_sample['eeg']
        contrastive_label = contrastive_sample['label']

        # 对对比样本应用变换（如果需要）
        if self.transform:
            contrastive_data = self.transform(contrastive_data)

        return data, contrastive_data, label, contrastive_label



from torch.utils.data import DataLoader

def create_contrastive_dataloaders(
    dataset: Dataset,
    target_subject_id: int,
    train_size: float = 0.7,
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 32,
    random_state: int = 42,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    针对单个被试执行数据切分并为对比学习生成数据加载器。

    Args:
        dataset: EEG 数据集对象
        target_subject_id: 当前目标被试的 ID
        train_size: 训练集大小
        test_size: 测试集大小
        val_size: 验证集大小
        batch_size: 批次大小
        random_state: 随机种子
        num_workers: 数据加载时的工作线程数

    Returns:
        train_loader, val_loader, test_loader: 训练集、验证集和测试集的 DataLoader
    """
    # 获取目标被试的所有样本索引
    all_indices = np.where(dataset.subject_ids == target_subject_id)[0]
    if len(all_indices) == 0:
        raise ValueError(f"Subject ID {target_subject_id} not found in dataset.")

    # 获取其他被试的样本索引
    other_indices = np.where(dataset.subject_ids != target_subject_id)[0]

    # 初始化索引列表
    train_idx_list, val_idx_list, test_idx_list = [], [], []

    # 获取目标被试的所有类别标签
    unique_labels = np.unique(dataset.labels[all_indices])

    # 对每个类别独立进行切分
    for label in unique_labels:
        # 获取当前类别的样本索引，并随机打乱
        label_indices = all_indices[dataset.labels[all_indices] == label]
        np.random.seed(random_state)
        np.random.shuffle(label_indices)

        n_total = len(label_indices)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = int(n_total * train_size)

        # 切分数据
        t_idx = label_indices[:n_train]
        v_idx = label_indices[n_train:n_train + n_val]
        s_idx = label_indices[n_train + n_val:]

        # 将索引添加到对应的列表
        train_idx_list.extend(t_idx)
        val_idx_list.extend(v_idx)
        test_idx_list.extend(s_idx)

    # 将索引转换为 numpy 数组
    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list)
    test_idx = np.array(test_idx_list)

    # 输出切分结果
    def get_dist(indices):
        labels = dataset.labels[indices]
        return dict(zip(*np.unique(labels, return_counts=True)))

    print(f"\nFinal Split Summary for Subject {target_subject_id}:")
    print(f"  Train Set: {len(train_idx)} trials, Dist: {get_dist(train_idx)}")
    print(f"  Val Set:   {len(val_idx)} trials, Dist: {get_dist(val_idx)}")
    print(f"  Test Set:  {len(test_idx)} trials, Dist: {get_dist(test_idx)}")

    # 创建数据集
    train_dataset = ContrastiveDataset(dataset, train_idx, other_indices)
    val_dataset = ContrastiveDataset(dataset, val_idx, other_indices)
    test_dataset = ContrastiveDataset(dataset, test_idx, other_indices)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def create_single_subject_dataloaders(
    dataset: EEGDataset,
    target_subject_id: int,
    train_size: float = 0.7,
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 32,
    random_state: int = 42,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    针对单个被试执行【类内随机切分】：确保每个集合都包含所有类别。
    """
    # 1. 获取该被试的所有索引
    all_indices = np.where(dataset.subject_ids == target_subject_id)[0]
    if len(all_indices) == 0:
        raise ValueError(f"Subject ID {target_subject_id} not found in dataset.")

    # 2. 准备容器
    train_idx_list, val_idx_list, test_idx_list = [], [], []
    
    # 3. 获取该被试包含的所有类别标签
    unique_labels = np.unique(dataset.labels[all_indices])
    
    print(f"\n[Random Split] Processing Subject: {target_subject_id}")
    
    # 4. 对每个类别独立进行随机切分
    for label in unique_labels:
        # 提取该被试下属于当前类别的所有索引
        label_indices = all_indices[dataset.labels[all_indices] == label]
        
        # 随机打乱该类别的索引
        np.random.seed(random_state)
        np.random.shuffle(label_indices)
        
        n_total = len(label_indices)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = int(n_total * train_size)
        
        # 随机切分
        t_idx = label_indices[:n_train]
        v_idx = label_indices[n_train : n_train + n_val]
        if n_test+n_val+n_train >= n_total:
            s_idx = label_indices[n_train + n_val:]
        else:
            s_idx = label_indices[n_train + n_val:n_test+n_val+n_train]
        
        train_idx_list.extend(t_idx)
        val_idx_list.extend(v_idx)
        test_idx_list.extend(s_idx)
        
        print(f"  Class {label}: Total={n_total}, Train={len(t_idx)}, Val={len(v_idx)}, Test={len(s_idx)}")

    # 5. 转换为 numpy 数组
    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list)
    test_idx = np.array(test_idx_list)

    # 6. 辅助打印：检查最终分布
    def get_dist(indices):
        labels = dataset.labels[indices]
        return dict(zip(*np.unique(labels, return_counts=True)))

    print(f"\nFinal Split Summary for Subject {target_subject_id}:")
    print(f"  Train Set: {len(train_idx)} trials, Dist: {get_dist(train_idx)}")
    print(f"  Val Set:   {len(val_idx)} trials, Dist: {get_dist(val_idx)}")
    print(f"  Test Set:  {len(test_idx)} trials, Dist: {get_dist(test_idx)}")

    # 7. 构建 DataLoader
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def create_single_subject_few_shot_dataloaders(
    dataset: EEGDataset,
    target_subject_id: int,
    n_shot: int = 5,  # 每个类别的支持集样本数
    batch_size: int = 32,
    random_state: int = 42,
    num_workers: int = 4,
    val_size: float = 0,  # 保留验证集，或者可以设置为0
    oversample: bool = False  # 添加参数控制是否使用过采样
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    针对单个被试执行【类内随机切分】以适应few-shot学习：
    1. 支持集是平衡的（每个类别 n_shot 个样本）
    2. 查询集样本是剩余样本（即去掉支持集和验证集的样本）
    3. 验证集保持（如果需要）
    4. 可以使用过采样（oversample）来增强少数类样本
    """
    # 1. 获取该被试的所有索引
    all_indices = np.where(dataset.subject_ids == target_subject_id)[0]
    if len(all_indices) == 0:
        raise ValueError(f"Subject ID {target_subject_id} not found in dataset.")

    # 获取每个类别的样本索引
    label_indices = {label: all_indices[dataset.labels[all_indices] == label] for label in np.unique(dataset.labels[all_indices])}

    # 2. 如果开启过采样，针对少数类别进行过采样
    if oversample:
        # 获取每个类别的样本数
        class_counts = {label: len(indices) for label, indices in label_indices.items()}
        max_count = max(class_counts.values())  # 获取最多样本的类别数

        # 对每个类别进行过采样，直到每个类别的样本数达到 max_count
        for label, indices in label_indices.items():
            current_count = len(indices)
            if current_count < max_count:
                # 随机重复样本，直到样本数达到 max_count
                num_samples_to_add = max_count - current_count
                sampled_indices = np.random.choice(indices, num_samples_to_add, replace=True)
                label_indices[label] = np.concatenate([indices, sampled_indices])

    # 3. 准备数据容器
    train_idx_list, val_idx_list, query_idx_list = [], [], []

    print(f"\n[Random Split] Processing Subject: {target_subject_id}")
    
    # 4. 对每个类别独立进行随机切分
    for label, indices in label_indices.items():
        # 随机打乱该类别的索引
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        # 选择 n_shot 个样本作为支持集
        n_shot_samples = indices[:n_shot]
        
        # 选择 val_size 比例的样本作为验证集（如果 val_size > 0）
        n_val_samples = int(len(indices) * val_size)
        val_samples = indices[n_shot:n_shot + n_val_samples] if n_val_samples > 0 else []
        
        # 查询集是剩余样本（去掉支持集和验证集的样本）
        remaining_samples = indices[n_shot + n_val_samples:]
        
        # 将支持集样本添加到训练集列表
        train_idx_list.extend(n_shot_samples)
        
        # 将验证集样本添加到验证集列表
        val_idx_list.extend(val_samples)
        
        # 将剩余的样本作为查询集
        query_idx_list.extend(remaining_samples)
        
        print(f"  Class {label}: Total={len(indices)}, Support Set={n_shot}, Val Set={len(val_samples)}, Query Set={len(remaining_samples)}")

    # 5. 转换为 numpy 数组
    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list) if val_size > 0 else None
    query_idx = np.array(query_idx_list)

    # 6. 辅助打印：检查最终分布
    def get_dist(indices):
        labels = dataset.labels[indices]
        return dict(zip(*np.unique(labels, return_counts=True)))

    print(f"\nFinal Split Summary for Subject {target_subject_id}:")
    print(f"  Train Set: {len(train_idx)} trials, Dist: {get_dist(train_idx)}")
    print(f"  Val Set:   {len(val_idx)} trials, Dist: {get_dist(val_idx)}" if val_idx is not None else "  Val Set:   None")
    print(f"  Query Set: {len(query_idx)} trials, Dist: {get_dist(query_idx)}")

    # 7. 构建 DataLoader
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    # Only create val_loader if val_size > 0
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    ) if val_size > 0 else None
    
    query_loader = DataLoader(
        Subset(dataset, query_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, query_loader


def create_multitask_single_subject_few_shot_dataloaders(
    dataset: EEGDataset,
    target_subject_id: int,
    target_task_id: list,
    n_shot: int = 5,  # 每个类别的支持集样本数
    batch_size: int = 32,
    random_state: int = 42,
    num_workers: int = 4,
    val_size: float = 0,  # 保留验证集，或者可以设置为0
    oversample: bool = False  # 添加参数控制是否使用过采样c
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    针对单个被试执行【类内随机切分】以适应few-shot学习：
    1. 支持集是平衡的（每个类别 n_shot 个样本）
    2. 查询集样本是剩余样本（即去掉支持集和验证集的样本）
    3. 验证集保持（如果需要）
    4. 可以使用过采样（oversample）来增强少数类样本
    """
    # 1. 获取该被试的所有索引
    mask = (dataset.subject_ids == target_subject_id)
    
    if target_task_id is not None:
        mask &= np.isin(dataset.tasks, target_task_id)
    
    all_indices = np.where(mask)[0]

    if len(all_indices) == 0:
        raise ValueError(f"Subject ID {target_subject_id} not found in dataset.")

    # 获取每个类别的样本索引
    label_indices = {label: all_indices[dataset.labels[all_indices] == label] for label in np.unique(dataset.labels[all_indices])}

    # 2. 如果开启过采样，针对少数类别进行过采样
    if oversample:
        # 获取每个类别的样本数
        class_counts = {label: len(indices) for label, indices in label_indices.items()}
        max_count = max(class_counts.values())  # 获取最多样本的类别数

        # 对每个类别进行过采样，直到每个类别的样本数达到 max_count
        for label, indices in label_indices.items():
            current_count = len(indices)
            if current_count < max_count:
                # 随机重复样本，直到样本数达到 max_count
                num_samples_to_add = max_count - current_count
                sampled_indices = np.random.choice(indices, num_samples_to_add, replace=True)
                label_indices[label] = np.concatenate([indices, sampled_indices])

    # 3. 准备数据容器
    train_idx_list, val_idx_list, query_idx_list = [], [], []

    print(f"\n[Random Split] Processing Subject: {target_subject_id}")
    
    # 4. 对每个类别独立进行随机切分
    for label, indices in label_indices.items():
        # 随机打乱该类别的索引
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        # 选择 n_shot 个样本作为支持集
        n_shot_samples = indices[:n_shot]
        
        # 选择 val_size 比例的样本作为验证集（如果 val_size > 0）
        n_val_samples = int(len(indices) * val_size)
        val_samples = indices[n_shot:n_shot + n_val_samples] if n_val_samples > 0 else []
        
        # 查询集是剩余样本（去掉支持集和验证集的样本）
        remaining_samples = indices[n_shot + n_val_samples:]
        
        # 将支持集样本添加到训练集列表
        train_idx_list.extend(n_shot_samples)
        
        # 将验证集样本添加到验证集列表
        val_idx_list.extend(val_samples)
        
        # 将剩余的样本作为查询集
        query_idx_list.extend(remaining_samples)
        
        print(f"  Class {label}: Total={len(indices)}, Support Set={n_shot}, Val Set={len(val_samples)}, Query Set={len(remaining_samples)}")

    # 5. 转换为 numpy 数组
    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list) if val_size > 0 else None
    query_idx = np.array(query_idx_list)

    # 6. 辅助打印：检查最终分布
    def get_dist(indices):
        labels = dataset.labels[indices]
        return dict(zip(*np.unique(labels, return_counts=True)))

    print(f"\nFinal Split Summary for Subject {target_subject_id}:")
    print(f"  Train Set: {len(train_idx)} trials, Dist: {get_dist(train_idx)}")
    print(f"  Val Set:   {len(val_idx)} trials, Dist: {get_dist(val_idx)}" if val_idx is not None else "  Val Set:   None")
    print(f"  Query Set: {len(query_idx)} trials, Dist: {get_dist(query_idx)}")

    # 7. 构建 DataLoader
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    # Only create val_loader if val_size > 0
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    ) if val_size > 0 else None
    
    query_loader = DataLoader(
        Subset(dataset, query_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, query_loader




def create_single_subject_dataloaders_control(
    dataset: EEGDataset,
    target_subject_id: int,
    train_size: float = 0.7,
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 32,
    random_state: int = 42,
    num_workers: int = 4,
    train_count: int = None  # 新增参数：指定训练集的样本数
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    针对单个被试执行【类内随机切分】：确保每个集合都包含所有类别。
    如果指定了train_count，将会使用该数值来设置训练集的数量，覆盖train_size。
    测试集的大小保证不为零，但可能会占用剩余样本的部分。
    """
    # 1. 获取该被试的所有索引
    all_indices = np.where(dataset.subject_ids == target_subject_id)[0]
    if len(all_indices) == 0:
        raise ValueError(f"Subject ID {target_subject_id} not found in dataset.")
    
    # 2. 准备容器
    train_idx_list, val_idx_list, test_idx_list = [], [], []
    
    # 3. 获取该被试包含的所有类别标签
    unique_labels = np.unique(dataset.labels[all_indices])
    
    print(f"\n[Random Split] Processing Subject: {target_subject_id}")
    
    # 4. 对每个类别独立进行随机切分
    for label in unique_labels:
        # 提取该被试下属于当前类别的所有索引
        label_indices = all_indices[dataset.labels[all_indices] == label]
        
        # 随机打乱该类别的索引
        np.random.seed(random_state)
        np.random.shuffle(label_indices)
        
        n_total = len(label_indices)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        
        # 如果指定了训练集数量，使用该数量，否则使用train_size
        if train_count is not None:
            n_train = train_count
        else:
            n_train = int(n_total * train_size)
        
        # 保证切分不会越界，测试集至少为1个样本
        n_train = min(n_train, n_total - n_val - max(1, n_test))
        
        # 随机切分
        t_idx = label_indices[:n_train]
        v_idx = label_indices[n_train : n_train + n_val]
        # 测试集为剩余样本
        s_idx = label_indices[n_train + n_val:]
        
        train_idx_list.extend(t_idx)
        val_idx_list.extend(v_idx)
        test_idx_list.extend(s_idx)
        
        print(f"  Class {label}: Total={n_total}, Train={len(t_idx)}, Val={len(v_idx)}, Test={len(s_idx)}")

    # 5. 转换为 numpy 数组
    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list)
    test_idx = np.array(test_idx_list)

    # 6. 辅助打印：检查最终分布
    def get_dist(indices):
        labels = dataset.labels[indices]
        return dict(zip(*np.unique(labels, return_counts=True)))

    print(f"\nFinal Split Summary for Subject {target_subject_id}:")
    print(f"  Train Set: {len(train_idx)} trials, Dist: {get_dist(train_idx)}")
    print(f"  Val Set:   {len(val_idx)} trials, Dist: {get_dist(val_idx)}")
    print(f"  Test Set:  {len(test_idx)} trials, Dist: {get_dist(test_idx)}")

    # 7. 构建 DataLoader
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader



class RestNormalizedDataset(Dataset):
    """
    使用 Resting data 的统计信息对 Task data 进行标准化
    
    标准化公式: (x - rest_mean) / rest_std
    消除 resting state 的偏移，保留 task-related 的信号变化
    """
    def __init__(
        self, 
        base_dataset: EEGDataset,
        rest_dataset: EEGDataset,
        target_subject_id: int,
        normalize_per_channel: bool = True,
        use_global_rest: bool = False  # 如果为True，使用所有被试的rest数据计算统计信息
    ):
        """
        Args:
            base_dataset: Task dataset (将被标准化的数据)
            rest_dataset: Resting dataset (用于计算统计信息)
            target_subject_id: 目标被试ID
            normalize_per_channel: 是否每个通道独立标准化
            use_global_rest: 是否使用全局rest统计信息（跨所有被试）
        """
        self.base_dataset = base_dataset
        self.normalize_per_channel = normalize_per_channel
        
        # 确定用于计算统计信息的 rest 数据
        if use_global_rest:
            rest_indices = np.arange(len(rest_dataset))
            print(f"\n[Rest Normalization] Using GLOBAL rest statistics")
        else:
            rest_indices = np.where(rest_dataset.subject_ids == target_subject_id)[0]
            print(f"\n[Rest Normalization] Using Subject {target_subject_id} rest statistics")
        
        if len(rest_indices) == 0:
            raise ValueError(f"No rest data found for subject {target_subject_id}")
        
        # 获取 resting data (shape: n_trials, n_channels, n_timepoints)
        rest_data = rest_dataset.data[rest_indices]
        
        if normalize_per_channel:
            # 每个通道独立计算: mean/std shape (n_channels,)
            self.rest_mean = rest_data.mean(axis=(0, 2), keepdims=True)  # (1, n_channels, 1)
            self.rest_std = rest_data.std(axis=(0, 2), keepdims=True)    # (1, n_channels, 1)
            # 转为 (n_channels, 1) 方便后续广播
            self.rest_mean = self.rest_mean.squeeze()  # (n_channels,)
            self.rest_std = self.rest_std.squeeze()    # (n_channels,)
        else:
            # 全局计算
            self.rest_mean = rest_data.mean()
            self.rest_std = rest_data.std()
        
        # 避免除零
        self.rest_std = np.where(self.rest_std == 0, 1.0, self.rest_std)
        
        print(f"  Rest samples: {len(rest_indices)}")
        if normalize_per_channel:
            print(f"  Per-channel mean range: [{self.rest_mean.min():.4f}, {self.rest_mean.max():.4f}]")
            print(f"  Per-channel std range: [{self.rest_std.min():.4f}, {self.rest_std.max():.4f}]")
        else:
            print(f"  Global mean: {self.rest_mean:.4f}, std: {self.rest_std:.4f}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        eeg = sample['eeg']  # shape: (Channels, Num_Patches, Patch_Size)
        
        # 应用标准化: (x - rest_mean) / rest_std
        # eeg shape: (C, P, S)
        if self.normalize_per_channel:
            # rest_mean/rest_std shape: (C,)
            eeg_normalized = (eeg - self.rest_mean.reshape(-1, 1, 1)) / self.rest_std.reshape(-1, 1, 1)
        else:
            eeg_normalized = (eeg - self.rest_mean) / self.rest_std
        
        sample['eeg'] = eeg_normalized.float()
        
        return sample


def create_rest_normalized_dataloaders(
    task_dataset: EEGDataset,
    rest_dataset: EEGDataset,
    target_subject_id: int,
    target_task_id: list = None,
    train_size: float = 0.7,
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 32,
    random_state: int = 42,
    num_workers: int = 4,
    normalize_per_channel: bool = True,
    use_global_rest: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建使用 Resting data 标准化后的 multitask dataloaders
    
    与 create_multitask_single_subject_dataloaders 类似，但返回的 data 经过了 rest 标准化
    """
    # 1. 获取该被试的所有索引
    mask = (task_dataset.subject_ids == target_subject_id)
    
    if target_task_id is not None:
        mask &= np.isin(task_dataset.tasks, target_task_id)
    
    all_indices = np.where(mask)[0]

    # 2. 准备容器
    train_idx_list, val_idx_list, test_idx_list = [], [], []
    
    # 3. 获取该被试包含的所有类别标签
    unique_labels = np.unique(task_dataset.labels[all_indices])
    
    print(f"\n[Rest-Normalized Random Split] Subject: {target_subject_id}")
    
    # 4. 对每个类别独立进行随机切分
    for label in unique_labels:
        label_indices = all_indices[task_dataset.labels[all_indices] == label]
        
        np.random.seed(random_state)
        np.random.shuffle(label_indices)
        
        n_total = len(label_indices)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = int(n_total * train_size)
        
        t_idx = label_indices[:n_train]
        v_idx = label_indices[n_train : n_train + n_val]
        if n_test+n_val+n_train >= n_total:
            s_idx = label_indices[n_train + n_val:]
        else:
            s_idx = label_indices[n_train + n_val:n_test+n_val+n_train]
        
        train_idx_list.extend(t_idx)
        val_idx_list.extend(v_idx)
        test_idx_list.extend(s_idx)
        
        print(f"  Class {label}: Total={n_total}, Train={len(t_idx)}, Val={len(v_idx)}, Test={len(s_idx)}")

    # 5. 转换为 numpy 数组
    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list)
    test_idx = np.array(test_idx_list)

    # 6. 创建 RestNormalizedDataset
    train_dataset = RestNormalizedDataset(
        Subset(task_dataset, train_idx), rest_dataset, target_subject_id, 
        normalize_per_channel, use_global_rest
    )
    val_dataset = RestNormalizedDataset(
        Subset(task_dataset, val_idx), rest_dataset, target_subject_id,
        normalize_per_channel, use_global_rest
    )
    test_dataset = RestNormalizedDataset(
        Subset(task_dataset, test_idx), rest_dataset, target_subject_id,
        normalize_per_channel, use_global_rest
    )

    print(f"\nFinal Split Summary:")
    print(f"  Train: {len(train_idx)} trials")
    print(f"  Val:   {len(val_idx)} trials")
    print(f"  Test:  {len(test_idx)} trials")

    # 7. 构建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader