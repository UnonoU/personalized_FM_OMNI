

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
    
    
task_dataset = EEGDataset(
        h5_path="/mnt/dataset4/fuxy/m3cv/processed_cogbci_multitask/cogbci_multitask_session1.h5",
        patch_size=200,
            normalize=True,
            normalize_per_channel=False
    )
    
rest_dataset = EEGDataset(
    h5_path="/mnt/dataset4/fuxy/m3cv/processed_cogbci_session1/cogbci_resting.h5",
    patch_size=200,
    normalize=True,
    normalize_per_channel=False
)



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


label_mapping = {
            10: 0, 11: 1, 12: 2,
            20: 3, 21: 4,
            30: 5, 31: 6, 32: 7,
            40: 8, 41: 9, 42: 10
        }

train_loader, val_loader, test_loader = create_multitask_single_subject_dataloaders(
                dataset=task_dataset,
                target_subject_id=subject_id,
                target_task_id=[1,2],
                train_size=args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
            # 创建 rest 数据集的 dataloaders
train_rest, val_rest, test_rest = create_single_subject_dataloaders(
    dataset=rest_dataset,
    target_subject_id=subject_id,
    train_size=args.train_size,
    test_size=args.test_size,
    val_size=args.val_size,
    batch_size=args.batch_size,
    random_state=args.seed,
    num_workers=args.num_workers
)