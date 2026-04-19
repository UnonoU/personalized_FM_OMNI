
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import json
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from peft import get_peft_model, LoraConfig
import io
import os
import math
import time
import json
import glob
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict

from pathlib import Path
import argparse

import torch
import torch.distributed as dist
from torch import inf
import h5py

from tensorboardX import SummaryWriter
# from data_processor.dataset import ShockDataset
import pickle
from scipy.signal import resample
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


def get_input_chans(ch_names):
    input_chans = [0] # for cls token
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name) + 1)
    return input_chans

standard_1020 = ['FP1', 'FPZ', 'FP2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]


def get_cogbci_dataset(args, subject_id):
    """
    根据 subject_id 动态返回 CogBCI 的电极配置 (V1 或 V2)
    V2 Subjects: 17, 25, 26, 28, 29
    """
    v2_subjects = [17, 25, 26, 28, 29]
    
    if args.dataset == 'cogbci':
        if subject_id in v2_subjects:
            ch_names = [
                'Fp1', 'Fz', 'F3', 'F7', 'F9', 'FC5', 'FC1', 'C3', 'T7', 'CP1', 
                'Pz', 'P3', 'P7', 'P9', 'O1', 'Oz', 'O2', 'P10', 'P8', 'P4', 
                'CP2', 'CP6', 'T8', 'C4', 'FC2', 'FC6', 'F10', 'F8', 'F4', 'Fp2', 
                'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 
                'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 
                'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 
                'F2', 'FCz'
            ]
            print(f"Subject {subject_id} detected as V2")
        else:
            ch_names = [
                'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 
                'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 
                'CP6', 'CP2', 'FCz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 
                'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 
                'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 
                'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 
                'AF4', 'F2'
            ]
            print(f"  [Config] Subject {subject_id} using V1")
            
        ch_names_upper = [ch.upper() for ch in ch_names]
        in_chans = get_input_chans(ch_names_upper)
        chan_number = 62
        return in_chans, ch_names, chan_number
        
    else:
        return get_dataset_name(args)

def get_dataset_name(args):
    if args.dataset == 'swu':
        ch_names = ['Fp1', 'AF3', 'AF7', 'Fz', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3',\
                    'FC5', 'FT7', 'Cz', 'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3', 'CP5',\
                    'TP7', 'TP9', 'Pz', 'P1', 'P3', 'P5', 'P7', 'PO3', 'PO7', 'Oz',\
                    'O1', 'Fpz', 'Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2',\
                    'FC4', 'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CPz', 'CP2', 'CP4',\
                    'CP6', 'TP8', 'TP10', 'P2', 'P4', 'P6', 'P8', 'POz', 'PO4', 'PO8','O2']
        ch_names_upper = [ch_name.upper() for ch_name in ch_names]
        in_chans=get_input_chans(ch_names_upper)
        # print(in_chans)
        chan_number = 61
        
    elif args.dataset == 'm3cv':
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', \
                    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'FC1', \
                    'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', \
                    'TP9', 'TP10', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', \
                    'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', \
                    'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', \
                    'Fpz', 'CPz', 'POz', 'Oz', 'FCz']
        ch_names_upper = [ch_name.upper() for ch_name in ch_names]
        in_chans=get_input_chans(ch_names_upper)
        chan_number = 64

    return in_chans, ch_names, chan_number



def get_cogbci_dataset_for_embedding(dataset_name, subject_id):
    """
    根据 subject_id 动态返回 CogBCI 的电极配置 (V1 或 V2)
    V2 Subjects: 17, 25, 26, 28, 29
    """
    v2_subjects = [17, 25, 26, 28, 29]
    
    if dataset_name == 'cogbci':
        if subject_id in v2_subjects:
            ch_names = [
                'Fp1', 'Fz', 'F3', 'F7', 'F9', 'FC5', 'FC1', 'C3', 'T7', 'CP1', 
                'Pz', 'P3', 'P7', 'P9', 'O1', 'Oz', 'O2', 'P10', 'P8', 'P4', 
                'CP2', 'CP6', 'T8', 'C4', 'FC2', 'FC6', 'F10', 'F8', 'F4', 'Fp2', 
                'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 
                'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 
                'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 
                'F2', 'FCz'
            ]
            print(f"Subject {subject_id} detected as V2")
        else:
            ch_names = [
                'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 
                'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 
                'CP6', 'CP2', 'FCz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 
                'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 
                'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 
                'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 
                'AF4', 'F2'
            ]
            print(f"  [Config] Subject {subject_id} using V1")
            
        ch_names_upper = [ch.upper() for ch in ch_names]
        in_chans = get_input_chans(ch_names_upper)
        chan_number = 62
        return in_chans, ch_names, chan_number
        
    else:
        return get_dataset_name_for_embedding(dataset_name)


def get_dataset_name_for_embedding(dataset_name):
    if dataset_name == 'swu':
        ch_names = ['Fp1', 'AF3', 'AF7', 'Fz', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3',\
                    'FC5', 'FT7', 'Cz', 'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3', 'CP5',\
                    'TP7', 'TP9', 'Pz', 'P1', 'P3', 'P5', 'P7', 'PO3', 'PO7', 'Oz',\
                    'O1', 'Fpz', 'Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2',\
                    'FC4', 'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CPz', 'CP2', 'CP4',\
                    'CP6', 'TP8', 'TP10', 'P2', 'P4', 'P6', 'P8', 'POz', 'PO4', 'PO8','O2']
        ch_names_upper = [ch_name.upper() for ch_name in ch_names]
        in_chans=get_input_chans(ch_names_upper)
        # print(in_chans)
        chan_number = 61
        
    elif dataset_name == 'm3cv':
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', \
                    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'FC1', \
                    'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', \
                    'TP9', 'TP10', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', \
                    'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', \
                    'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', \
                    'Fpz', 'CPz', 'POz', 'Oz', 'FCz']
        ch_names_upper = [ch_name.upper() for ch_name in ch_names]
        in_chans=get_input_chans(ch_names_upper)
        chan_number = 64

    return in_chans, ch_names, chan_number


import torch

class BIOT_DataProjector:
    def __init__(self, target_chans=None):
        """
        初始化数据投影器
        :param target_chans: 目标通道名称列表 (List[str])
        """
        # 1. 定义目标通道 (target_chans)
        if target_chans is None:
            # 默认使用一个示例目标通道列表
            self.target_chans = [
                "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
                "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
                "C3-A2", "C4-A1"
            ]
        else:
            self.target_chans = target_chans
        
    def channel_project(self, data, use_channel_names):
        """
        将输入数据映射到目标通道空间
        并展平多余的维度
        
        Args:
            data (torch.Tensor): 输入数据，形状为 (batch_size, n_channel_source, patch, time)
            use_channel_names (list): 输入数据的通道名称列表
            
        Returns:
            torch.Tensor: 映射后的数据，形状为 (batch_size, target_chans_number, time)
        """
        device = data.device
        
        # 1. 处理输入数据的维度
        if len(data.shape) == 4 and data.shape[2] == 1:
            # 如果形状是 [B, C, 1, T]，先压缩
            data = data.squeeze(2)  # 变成 [B, C, T]
        elif len(data.shape) == 4:
            # 如果 shape[2] 不是 1，展平 patch 维度
            batch_size, n_channel_source, patch_size, n_timepoints = data.shape
            data = data.reshape(batch_size, n_channel_source * patch_size, n_timepoints)
        
        # 2. 现在处理实际的维度
        if len(data.shape) == 3:
            batch_size, n_channel_source, n_timepoints = data.shape
            patch_size = 1
        else:
            batch_size, n_channel_source, patch_size, n_timepoints = data.shape
        
        target_chans_number = len(self.target_chans)
        
        # 3. 创建输入通道名称映射表
        source_ch_map = {name.upper(): idx for idx, name in enumerate(use_channel_names)}
        
        # 4. 初始化结果张量
        if patch_size == 1:
            # 如果没有 patch 维度
            result = torch.zeros(
                (batch_size, target_chans_number, n_timepoints),
                device=device,
                dtype=data.dtype
            )
        else:
            # 如果有 patch 维度
            result = torch.zeros(
                (batch_size, target_chans_number, patch_size, n_timepoints),
                device=device,
                dtype=data.dtype
            )
        
        # 5. 遍历目标通道列表
        for target_idx, target_name in enumerate(self.target_chans):
            target_name_upper = target_name.upper()
            
            if '-' in target_name_upper:
                try:
                    ch1_name, ch2_name = target_name_upper.split('-')
                    if ch1_name in source_ch_map and ch2_name in source_ch_map:
                        idx1 = source_ch_map[ch1_name]
                        idx2 = source_ch_map[ch2_name]
                        
                        if patch_size == 1:
                            result[:, target_idx, :] = data[:, idx1, :] - data[:, idx2, :]
                        else:
                            result[:, target_idx, :, :] = data[:, idx1, :, :] - data[:, idx2, :, :]
                        continue
                except ValueError:
                    pass
            
            if target_name_upper in source_ch_map:
                src_idx = source_ch_map[target_name_upper]
                if patch_size == 1:
                    result[:, target_idx, :] = data[:, src_idx, :]
                else:
                    result[:, target_idx, :, :] = data[:, src_idx, :, :]
                continue
            
            # 无法匹配，保持0
        
        # 6. 如果结果有 patch 维度，展平它
        if len(result.shape) == 4 and result.shape[2] == 1:
            result = result.squeeze(2)  # [B, C, 1, T] -> [B, C, T]
        
        return result

import numpy as np
import h5py
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from collections import OrderedDict


def get_unique_subject_ids(h5_path):
    """
    Extract unique subject IDs from HDF5 file

    Compatible with both 'subject_ids' and 'subject_id' key names
    """
    with h5py.File(h5_path, 'r') as f:
        # Try both possible key names
        if 'subject_ids' in f:
            subject_ids = f['subject_ids'][:]
        elif 'subject_id' in f:
            subject_ids = f['subject_id'][:]
        elif 'subject' in f:
            subject_ids = f['subject'][:]
        else:
            raise KeyError("Neither 'subject_ids' nor 'subject_id' found in HDF5 file")

        unique_ids = sorted(np.unique(subject_ids))
    return unique_ids


def load_vanilla_labram_weights(encoder, pretrained_path, model_key='model|module', model_prefix=''):
    """
    Load ORIGINAL LaBraM Base pretrained weights

    Handles:
    - model_key extraction (model|module)
    - student. prefix filtering
    - head mismatch
    - relative_position_index removal
    - time_embed truncation (16 -> 10)
    """
    print(f"  Loading LaBraM Base weights from: {pretrained_path}")

    if pretrained_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

    # Extract model weights from checkpoint
    checkpoint_model = None
    for key in model_key.split('|'):
        if key in checkpoint:
            checkpoint_model = checkpoint[key]
            print(f"  ✓ Load state_dict by model_key = {key}")
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    # Filter 'student.' prefix (for self-supervised pretrained models)
    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('student.'):
            new_dict[key[8:]] = checkpoint_model[key]
    if len(new_dict) > 0:
        checkpoint_model = new_dict
        print(f"  ✓ Filtered 'student.' prefix from {len(new_dict)} keys")

    # Handle head mismatch (classification head vs reconstruction head)
    state_dict = encoder.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"  ⚠ Removing key {k} from checkpoint (shape mismatch)")
            del checkpoint_model[k]

    # Remove relative_position_index (will be regenerated)
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

    # ★ Handle time_embed truncation (16 -> 10 patches for Cognitive task)
    if 'time_embed' in checkpoint_model:
        current_shape = encoder.time_embed.shape
        ckpt_shape = checkpoint_model['time_embed'].shape
        if ckpt_shape != current_shape:
            print(f"  ⚠ time_embed shape mismatch: checkpoint {ckpt_shape} vs current {current_shape}")
            print(f"  ✓ Truncating time_embed: {ckpt_shape} -> {current_shape}")
            checkpoint_model['time_embed'] = checkpoint_model['time_embed'][:, :current_shape[1], :]

    # Load weights
    # from utils import load_state_dict
    load_state_dict(encoder, checkpoint_model, prefix=model_prefix)
    print("  ✓ LaBraM Base weights loaded successfully!")

import torch
from collections import OrderedDict

def load_pretrained_model(model,pretrained_path, model_keys='model|module', model_prefix='', ignore_missing="relative_position_index"):
    # Load checkpoint
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

    print("Load ckpt from %s" % pretrained_path)

    # Extract model from checkpoint
    checkpoint_model = None
    for model_key in model_keys.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    
    # Filter the checkpoint model if necessary
    if (checkpoint_model is not None) and (model_prefix != ''):
        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('student.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                pass
        checkpoint_model = new_dict

    # Remove mismatched keys
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # Remove any keys related to relative_position_index
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)
    load_state_dict(model, checkpoint_model, prefix=model_prefix)


def check_and_load_existing(output_root, subject_id):
    """
    检查该被试是否已完成训练。
    如果完成，加载并返回其 metrics 数据；否则返回 None。
    """
    subject_dir = os.path.join(output_root, f'subject_{subject_id:02d}')
    metrics_path = os.path.join(subject_dir, 'metrics.json')
    
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  ⚠ Found metrics.json for Subject {subject_id} but failed to load: {e}")
            return None
    return None



def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, layer_names=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters, layer_names=layer_names)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict): 
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0, layer_names=None) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = [p for p in parameters if p.grad is not None]
        
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        layer_norm = torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters])
        total_norm = torch.norm(layer_norm, norm_type)
        # print(layer_norm.max(dim=0))
        
        if layer_names is not None:
            if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1.0:
                value_top, name_top = torch.topk(layer_norm, k=5)
                print(f"Top norm value: {value_top}")
                print(f"Top norm name: {[layer_names[i][7:] for i in name_top.tolist()]}")
        
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None, save_ckpt_freq=1):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    if not getattr(args, 'enable_deepspeed', False):
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        if epoch == 'best':
            checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name),]
        elif (epoch + 1) % save_ckpt_freq == 0:
            checkpoint_paths.append(output_dir / ('checkpoint-%s.pth' % epoch_name))

        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                # 'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            if loss_scaler is not None:
                to_save['scaler'] = loss_scaler.state_dict()

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)
                
            if optimizer_disc is not None:
                to_save['optimizer_disc'] = optimizer_disc.state_dict()

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)           

def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None):
    output_dir = Path(args.output_dir)
    
    if not getattr(args, 'enable_deepspeed', False):
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint.pth'))
            if len(all_checkpoints) > 0:
                args.resume = os.path.join(output_dir, 'checkpoint.pth')
            else:
                all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('-')[-1].split('.')[0]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model']) # strict: bool=True, , strict=False
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"Resume checkpoint at epoch {checkpoint['epoch']}")
                args.start_epoch = 1#checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
            if 'optimizer_disc' in checkpoint:
                optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])

def create_ds_config(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "latest"), mode="w") as f:
        pass

    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))


# def build_pretraining_dataset(datasets: list, time_window: list, stride_size=200, start_percentage=0, end_percentage=1):
#     shock_dataset_list = []
#     ch_names_list = []
#     for dataset_list, window_size in zip(datasets, time_window):
#         dataset = ShockDataset([Path(file_path) for file_path in dataset_list], window_size * 200, stride_size, start_percentage, end_percentage)
#         shock_dataset_list.append(dataset)
#         ch_names_list.append(dataset.get_ch_names())
#     return shock_dataset_list, ch_names_list


# def get_input_chans(ch_names):
#     input_chans = [0] # for cls token
#     for ch_name in ch_names:
#         input_chans.append(standard_1020.index(ch_name) + 1)
#     return input_chans

def save_args_to_json(args, filepath):
    """将args参数保存为JSON文件"""
    args_dict = vars(args).copy()  # 创建参数的副本
    
    # 处理不可JSON序列化的对象
    for key, value in args_dict.items():
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            args_dict[key] = str(value)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存为JSON
    with open(filepath, 'w') as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 参数已保存到: {filepath}")
    return filepath


def get_checkpoint_path(method, subject_id, recon_results_dir,dataset):
    """
    Map method to checkpoint path

    Args:
        method: 'full', 'sgd', 'lora', or 'resnet'
        subject_id: Subject ID
        recon_results_dir: Root directory for reconstruction results

    Returns:
        Path to best_model.pth for this method and subject
    """
    method_to_subdir = {
        'full': 'full',
        'sgd': 'sgd',
        'lora': 'lora',
        'resnet': 'resnet'
    }

    if method not in method_to_subdir:
        raise ValueError(f"Unknown method: {method}. Must be one of {list(method_to_subdir.keys())}")

    subdir = method_to_subdir[method]
    checkpoint_path = os.path.join(
        recon_results_dir,
        subdir,
        f'subject_{subject_id:02d}',
        'best_model.pth'
    )

    return checkpoint_path


def count_parameters_with_names(model):
    """
    计算模型中的总参数数和可训练参数数，并显示可训练模块的名称
    """
    total_params = 0
    trainable_params = 0
    trainable_modules = []

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_modules.append(name)  # 收集可训练模块的名称

    return total_params, trainable_params, trainable_modules

def calculate_parameter_percentage_with_names(model):
    """
    计算可训练参数占总参数的比例，并返回可训练模块的名称
    """
    total_params, trainable_params, trainable_modules = count_parameters_with_names(model)
    trainable_percentage = (trainable_params / total_params) * 100
    return total_params, trainable_params, trainable_percentage, trainable_modules


import torch
from peft import get_peft_model, LoraConfig

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')

    return list(lora_module_names)

# def create_lora(model, args):
#     lora_config = LoraConfig(
#         r=args.lora_r,  # Rank of LoRA
#         lora_alpha=args.lora_alpha,  # Scaling factor
#         target_modules=find_all_linear_names(model),  # Target modules, can vary depending on architecture
#         lora_dropout=args.lora_dropout  # Dropout for LoRA layers
#     )
#     model = get_peft_model(model, lora_config)
#     print("LoRA modules added to the model.")
#     return model

def create_lora_cls(model, args):
    lora_config = LoraConfig(
        r=args.cls_lora_r,  # Rank of LoRA
        lora_alpha=args.cls_lora_alpha,  # Scaling factor
        target_modules=find_all_linear_fullnames(model),  # Target modules, can vary depending on architecture
        lora_dropout=args.cls_lora_dropout  # Dropout for LoRA layers
    )
    model = get_peft_model(model, lora_config)
    print("LoRA modules added to the model.")
    return model
    
def find_all_linear_fullnames(model):
    lora_module_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "lm_head" not in name:
                lora_module_names.append(name)  
    print(lora_module_names)
    return list(lora_module_names)

def create_lora(model, args):
    lora_config = LoraConfig(
        r=args.lora_r, # Rank of LoRA
        lora_alpha=args.lora_alpha, # Scaling factor
        target_modules=find_all_linear_fullnames(model), # Target modules, can vary depending on architecture
        lora_dropout=args.lora_dropout,
        bias="all"
        )
    model = get_peft_model(model, lora_config)
    print("LoRA modules added to the model.")
    return model


import torch

def labels_to_task_id(labels, device=None):
    # 定义一个映射规则，将标签范围映射到任务ID
    label_to_task = {
        (0, 1): 0,   # 0, 1 转换为 0
        (10, 11, 12): 1,  # 10, 11, 12 转换为 1
        (20, 21): 2,  # 20, 21 转换为 2
        (30, 31, 32): 3,  # 30, 31, 32 转换为 3
        (40, 41, 42): 4,  # 40, 41, 42 转换为 4
    }

    # 为了简化查找，创建一个反向字典
    task_mapping = {label: task_id for task_range, task_id in label_to_task.items() for label in task_range}

    # 对 labels 列表进行转换
    task_ids = [task_mapping.get(label, -1) for label in labels]  # -1 作为默认值处理没有匹配的标签

    # 如果传入了设备参数，则将任务ID转换为tensor并移动到该设备
    task_tensor = torch.tensor(task_ids, dtype=torch.long)
    if device:
        task_tensor = task_tensor.to(device)

    return task_tensor


# def extract_1020_electrode_positions(ch_names):
#     """
#     基于BrainOmni论文2.1节：提取10-20系统电极的6维位置向量
#     返回格式：3D坐标(x,y,z) + 3D方向向量(dx,dy,dz)
#     """
#     # 获取标准10-20系统蒙太奇
#     montage = mne.channels.make_standard_montage("standard_1020")
#     ch_positions = montage.get_positions()['ch_pos']
    
#     positions_6d = []
#     valid_electrodes = []
#     sensor_types = []
    
#     for ch_name in ch_names:
#         # 电极名称标准化映射
#         standardized_name = ch_name
        
#         if standardized_name in ch_positions:
#             # 获取3D坐标 (x, y, z)
#             coord_3d = ch_positions[standardized_name]
            
#             # 计算方向向量（指向球心，符合论文中EEG电极的物理特性）
#             direction = -coord_3d / np.linalg.norm(coord_3d)
            
#             # 组合6维向量 [x, y, z, dx, dy, dz]
#             pos_6d = np.concatenate([coord_3d, direction])
#             positions_6d.append(pos_6d)
#             valid_electrodes.append(ch_name)
            
#             # 传感器类型：EEG=0（根据论文2.1节）
#             sensor_types.append(0)
#         else:
#             print(f"警告: 电极 {ch_name} 不在标准10-20系统中")
#             # 使用默认位置（0,0,0）和默认方向
#             default_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
#             positions_6d.append(default_pos)
#             valid_electrodes.append(ch_name)
#             sensor_types.append(0)
    
#     return (np.array(positions_6d), 
#             np.array(sensor_types), 
#             valid_electrodes)
    
    
def extract_1020_electrode_positions(ch_names, device):
    import mne
    """
    基于BrainOmni论文2.1节：提取10-20系统电极的6维位置向量
    返回PyTorch张量格式，符合BrainOmni模型输入要求
    """
    # 获取标准10-20系统蒙太奇
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_positions = montage.get_positions()['ch_pos']
    
    positions_6d = []
    valid_electrodes = []
    sensor_types = []
    
    for ch_name in ch_names:
        # 电极名称标准化映射
        standardized_name = ch_name
        
        if standardized_name in ch_positions:
            # 获取3D坐标 (x, y, z)
            coord_3d = ch_positions[standardized_name]
            
            # 计算方向向量（指向球心）
            direction = -coord_3d / np.linalg.norm(coord_3d)
            
            # 组合6维向量 [x, y, z, dx, dy, dz]
            pos_6d = np.concatenate([coord_3d, direction])
            positions_6d.append(pos_6d)
            valid_electrodes.append(ch_name)
            
            # 传感器类型：EEG=0（根据论文2.1节）
            sensor_types.append(0)
        else:
            print(f"警告: 电极 {ch_name} 不在标准10-20系统中")
            # 使用默认位置（0,0,0）和默认方向
            default_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            positions_6d.append(default_pos)
            valid_electrodes.append(ch_name)
            sensor_types.append(0)
    
    # 转换为PyTorch张量
    positions_tensor = torch.from_numpy(np.array(positions_6d)).float().to(device)
    sensor_types_tensor = torch.from_numpy(np.array(sensor_types)).long().to(device)
    
    return positions_tensor, sensor_types_tensor, valid_electrodes