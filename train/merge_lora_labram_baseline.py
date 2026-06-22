import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from timm.models import create_model
from torch.utils.data import DataLoader
import os
from collections import OrderedDict
from tqdm import tqdm
from datetime import datetime
from peft import get_peft_model, LoraConfig

import sys
sys.path.append('/mnt/dataset4/yinuo/Rest_tuning')

import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# import utils
# import modeling_finetune
from src.utils import (
    get_unique_subject_ids,
    load_vanilla_labram_weights
)

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from openTSNE import TSNE
from src.data_loader import *
from src.train import load_pretrained_encoder


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
TASK_1_NAME='SSS'#'Motor'#'Transient_Sensory'#
TASK_DATA="/mnt/dataset4/fuxy/m3cv/processed_M3CV_unitask_globalZscore/m3cv_SSS_Sesssion1_temporal.h5"
# "/mnt/dataset0/UnonoU/personalized_FM/dataset_process/m3cv_Transient_Sensory.h5"
# "/mnt/dataset4/fuxy/m3cv/processed_M3CV_unitask_globalZscore/m3cv_Motor_Session1_temporal.h5"
# "/mnt/dataset4/fuxy/m3cv/processed_M3CV_unitask_globalZscore/m3cv_TransientState_Session1_temporal.h5"
# "/mnt/dataset4/fuxy/m3cv/processed_M3CV_unitask_globalZscore/m3cv_SSS_Sesssion1_temporal.h5"

TASK_2_NAME='TS_old'#'Transient_Sensory'#'Steady_State_Sensory'#'Motor'#
# MASK_RATIO=0.7
# TASK_3_NAME='Motor'


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
        return get_dataset_name(dataset_name)


def get_dataset_name(dataset_name):

    if dataset_name == 'swu':
        ch_names = [
            'Fp1', 'AF3', 'AF7', 'Fz', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3',
            'FC5', 'FT7', 'Cz', 'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3', 'CP5',
            'TP7', 'TP9', 'Pz', 'P1', 'P3', 'P5', 'P7', 'PO3', 'PO7', 'Oz',
            'O1', 'Fpz', 'Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2',
            'FC4', 'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CPz', 'CP2', 'CP4',
            'CP6', 'TP8', 'TP10', 'P2', 'P4', 'P6', 'P8', 'POz', 'PO4', 'PO8', 'O2'
        ]
        ch_names_upper = [ch_name.upper() for ch_name in ch_names]
        in_chans = get_input_chans(ch_names_upper)
        chan_number = 61

    elif dataset_name == 'm3cv':
        ch_names = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'FC1',
            'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10',
            'TP9', 'TP10', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4',
            'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6',
            'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8',
            'Fpz', 'CPz', 'POz', 'Oz', 'FCz'
        ]
        ch_names_upper = [ch_name.upper() for ch_name in ch_names]
        in_chans = get_input_chans(ch_names_upper)
        chan_number = 64
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return in_chans, ch_names, chan_number

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

from einops import rearrange

def extract_labram_embeddings(train_loader, val_loader, test_loader, model, in_chans, device='cuda'):
    model.eval()  # 设置模型为评估模式
    embeddings = []
    labels = []
    
    # 提取训练集数据
    for batch in train_loader:
        # 根据批次的数据类型提取数据和标签
        if isinstance(batch, (tuple, list)):
            eeg, target = batch  # 元组或列表格式
        else:
            eeg = batch['eeg']  # 字典格式
            target = batch['label']
        
        # 数据维度重排列：'b c s d' -> 'b c d s'
        eeg = rearrange(eeg, 'b c s d -> b c d s')
        
        eeg = eeg.to(device)  # 将数据移到设备（例如GPU）
        output = model.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)  # 获取模型的输出，即embedding
        
        # 从计算图中分离张量，然后将其转为NumPy数组
        embeddings.append(output.detach().cpu().numpy())  # 将嵌入从GPU移回CPU并转换为numpy
        labels.append(target.cpu().numpy())  # 获取标签并从GPU移回CPU

    # 提取验证集数据
    for batch in val_loader:
        if isinstance(batch, (tuple, list)):
            eeg, target = batch
        else:
            eeg = batch['eeg']
            target = batch['label']
        
        eeg = rearrange(eeg, 'b c s d -> b c d s')
        eeg = eeg.to(device)
        output = model.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
        
        # 从计算图中分离张量，然后将其转为NumPy数组
        embeddings.append(output.detach().cpu().numpy())
        labels.append(target.cpu().numpy())

    # 提取测试集数据
    for batch in test_loader:
        if isinstance(batch, (tuple, list)):
            eeg, target = batch
        else:
            eeg = batch['eeg']
            target = batch['label']
        
        eeg = rearrange(eeg, 'b c s d -> b c d s')
        eeg = eeg.to(device)
        output = model.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
        
        # 从计算图中分离张量，然后将其转为NumPy数组
        embeddings.append(output.detach().cpu().numpy())
        labels.append(target.cpu().numpy())
    
    # 合并所有批次的嵌入和标签
    embeddings = np.concatenate(embeddings, axis=0)  # 将所有batch的embedding连接起来
    labels = np.concatenate(labels, axis=0)  # 将所有标签连接起来
    
    return embeddings, labels



dataset = EEGDataset(
        h5_path=TASK_DATA,
        patch_size=1,
        normalize=False,
        normalize_per_channel=False
    )
sample = dataset[0]
num_patches = sample['eeg'].shape[1]


# dataset_2 = EEGDataset(
#         h5_path="/mnt/dataset0/UnonoU/personalized_FM/dataset_process/m3cv_Transient_Sensory.h5",
#         patch_size=1,
#         normalize=True,
#         normalize_per_channel=False
#     )
# sample = dataset[0]
# num_patches = sample['eeg'].shape[1]

# dataset_3 = EEGDataset(
#         h5_path="/mnt/dataset0/UnonoU/personalized_FM/dataset_process/m3cv_Motor.h5",
#         patch_size=1,
#         normalize=True,
#         normalize_per_channel=False
#     )
# sample = dataset[0]
# num_patches = sample['eeg'].shape[1]

def find_all_linear_fullnames(model):
    lora_module_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "lm_head" not in name:
                lora_module_names.append(name)  
    print(lora_module_names)
    return list(lora_module_names)
def load_finetuned_weights(encoder, checkpoint_path):
    """
    Load fully fine-tuned model weights with robust prefix stripping
    """

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    checkpoint_model = None
    for key in ['encoder_state_dict', '']:
        if key and key in checkpoint:
            checkpoint_model = checkpoint[key]
            print(f"  ✓ Load state_dict by key = '{key}'")
            break

    if checkpoint_model is None:
        checkpoint_model = checkpoint
        print(f"  ✓ Using checkpoint directly as state_dict")

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    module_stripped = False

    for key in all_keys:
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.'
            new_dict[new_key] = checkpoint_model[key]
            module_stripped = True
        else:
            new_dict[key] = checkpoint_model[key]

    if module_stripped:
        checkpoint_model = new_dict
        print(f"  ✓ Stripped 'module.' prefix from state_dict")

    # Second pass: strip 'encoder.' prefix if present
    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    encoder_stripped = False

    for key in all_keys:
        if key.startswith('encoder.'):
            new_key = key[8:]  # Remove 'encoder.'
            new_dict[new_key] = checkpoint_model[key]
            encoder_stripped = True
        else:
            new_dict[key] = checkpoint_model[key]

    if encoder_stripped:
        checkpoint_model = new_dict
        print(f"  ✓ Stripped 'encoder.' prefix from state_dict")

    # Handle time_embed truncation (16 -> 10 patches for M3CV)
    if 'time_embed' in checkpoint_model:
        current_shape = encoder.time_embed.shape
        ckpt_shape = checkpoint_model['time_embed'].shape
        if ckpt_shape != current_shape:
            print(f"  ⚠ time_embed shape mismatch: checkpoint {ckpt_shape} vs current {current_shape}")
            print(f"  ✓ Truncating time_embed: {ckpt_shape} -> {current_shape}")
            checkpoint_model['time_embed'] = checkpoint_model['time_embed'][:, :current_shape[1], :]

    # Handle head mismatch
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

    # Load weights using utils
    from utils import load_state_dict
    load_state_dict(encoder, checkpoint_model, prefix='')
    print("  ✓ Fine-tuned weights loaded successfully!")
    
# baseline model
labram_baseline_encoder = create_model(
        'labram_base_patch200_200',
        pretrained=False,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=True,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=True,
        num_channels_input=128,
        num_patches_per_channel_input=num_patches,
        reconstruction_output_dim=200,
        force_patch_embed=True
).to(device)

baseline_pretrained_path = '/mnt/dataset4/yinuo/personlized_FM/LaBraM/checkpoints/labram-base.pth'

load_vanilla_labram_weights(labram_baseline_encoder, baseline_pretrained_path, 'model|module', '')
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=find_all_linear_fullnames(labram_baseline_encoder),  # Targets Q/K/V/Projection layers
)


    
from einops import rearrange    
class SimpleClassifier_multi_model(nn.Module):
    """
    Simple Fine-tuning Classifier (with direct classification after Encoder)

    Architecture:
    1. Encoder (all parameters trainable)
    2. Global Average Pooling (GAP)
    3. Linear Classification Head (1-layer MLP)
    """
    def __init__(self, encoder, in_chans, model_name='labram', num_channels=61, num_classes=3, embed_dim=200):
        super().__init__()
        self.encoder = encoder
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.model = model_name
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
    
    
criterion = nn.CrossEntropyLoss()    
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
            eeg=rearrange(eeg,'b c s d -> b c d s')

            eeg = eeg.to(device)
            labels = labels.to(device)
            # print(f"eeg shape: {eeg.shape}")
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

def merge_lora_weights(model_1, model_2, alpha=0.9, beta=0.1):
    merged_model = model_1  # 我们将使用model_1的架构

    for name, param in model_1.named_parameters():
        # 检查LoRA参数的名称
        if 'lora_A.default.weight' in name or 'lora_B.default.weight' in name:
            # 从两个模型中分别获取LoRA权重
            param_1 = model_1.state_dict()[name]
            param_2 = model_2.state_dict()[name]

            # 加权合并
            merged_param = alpha * param_1 + beta * param_2

            # 将合并后的权重赋值回merged_model
            merged_model.state_dict()[name].copy_(merged_param)

    return merged_model


import numpy as np
import matplotlib.pyplot as plt


subject_test_accs = {} 

# 保存每个 alpha 值下的平均 test_acc
average_test_acc_per_alpha = []

# 循环 sub_id 从 1 到 95
for sub_id in range(1, 96):  # sub_id 从 1 到 95
    print(f"Processing subject {sub_id}...")
    if TASK_1_NAME=='TS_old':
        # checkpoint_path_1 = rf'/mnt/dataset4/yinuo/personlized_FM/for_personalized_FM_0107/contrastive/mask_embedding_label_high_flatten/baseline/TS/train_0.7_b32_0127/m3cv/lora/subject_{sub_id:02d}/best_model.pth'
        checkpoint_path_1 =rf'/mnt/dataset4/yinuo/Rest_tuning/baseline/labram/TS_old_data/train_0.7_old_data/m3cv/lora/subject_{sub_id:02d}/best_model.pth'
    elif TASK_1_NAME=='SSS':
        # checkpoint_path_1 = rf'/mnt/dataset4/yinuo/personlized_FM/for_personalized_FM_0107/contrastive/mask_embedding_label_high_flatten/baseline/SSS/train_0.7_b32_0127/m3cv/lora/subject_{sub_id:02d}/best_model.pth'
        checkpoint_path_1=rf'/mnt/dataset4/yinuo/Rest_tuning/baseline/labram/sss/model/train_0.7/m3cv/lora/subject_{sub_id:02d}/best_model.pth'
    elif TASK_1_NAME=='Motor':
        # checkpoint_path_1 = rf'/mnt/dataset4/yinuo/personlized_FM/for_personalized_FM_0107/contrastive/mask_embedding_label_high_flatten/baseline/Motor/train_0.7_b32_0127/m3cv/lora/subject_{sub_id:02d}/best_model.pth'
        checkpoint_path_1= rf'/mnt/dataset4/yinuo/Rest_tuning/baseline/labram/motor/model/train_0.7/m3cv/lora/subject_{sub_id:02d}/best_model.pth'
    
    if TASK_2_NAME=='TS_old':
        # checkpoint_path_2 = rf'/mnt/dataset4/yinuo/personlized_FM/for_personalized_FM_0107/contrastive/mask_embedding_label_high_flatten/baseline/TS/train_0.7_b32_0127/m3cv/lora/subject_{sub_id:02d}/best_model.pth'
        checkpoint_path_2= rf'/mnt/dataset4/yinuo/Rest_tuning/baseline/labram/TS_old_data/train_0.7_old_data/m3cv/lora/subject_{sub_id:02d}/best_model.pth'
    elif TASK_2_NAME=='SSS':
        # checkpoint_path_2 = rf'/mnt/dataset4/yinuo/personlized_FM/for_personalized_FM_0107/contrastive/mask_embedding_label_high_flatten/baseline/SSS/train_0.7_b32_0127/m3cv/lora/subject_{sub_id:02d}/best_model.pth'
        checkpoint_path_2=rf'/mnt/dataset4/yinuo/Rest_tuning/baseline/labram/sss/model/train_0.7/m3cv/lora/subject_{sub_id:02d}/best_model.pth'
    elif TASK_2_NAME=='Motor':
        # checkpoint_path_2 = rf'/mnt/dataset4/yinuo/personlized_FM/for_personalized_FM_0107/contrastive/mask_embedding_label_high_flatten/baseline/Motor/train_0.7_b32_0127/m3cv/lora/subject_{sub_id:02d}/best_model.pth'
        checkpoint_path_2= rf'/mnt/dataset4/yinuo/Rest_tuning/baseline/labram/motor/model/train_0.7/m3cv/lora/subject_{sub_id:02d}/best_model.pth'
    # checkpoint_path_2 = rf'/mnt/dataset4/yinuo/personlized_FM/for_personalized_FM_0107/contrastive/mask_embedding_label_high_flatten/lora_mask_{MASK_RATIO}/classification/{TASK_2_NAME}/train_0.7/m3cv/lora/subject_{sub_id:02d}/best_model.pth'

    
    # 获取数据集和模型参数
    DATASET_NAME = 'm3cv'
    if DATASET_NAME == 'cogbci':
        in_chans, ch_names, chan_number = get_cogbci_dataset_for_embedding(DATASET_NAME, sub_id)
        cls_number = 6
    else:
        in_chans, ch_names, chan_number = get_dataset_name(dataset_name=DATASET_NAME)
        cls_number = 11
    
    # 加载数据
    train_loader, val_loader, test_loader = create_single_subject_dataloaders(
        dataset=dataset,
        target_subject_id=sub_id,
        train_size=0.7,
        test_size=0.2,
        val_size=0.1,
        batch_size=32,
        random_state=42,
        num_workers=4
    )
    
    # 加载模型
    model_task_1 = SimpleClassifier_multi_model(
        encoder=get_peft_model(create_model(
            'labram_base_patch200_200',
            pretrained=False,
            use_mean_pooling=True,
            init_scale=0.001,
            use_rel_pos_bias=True,
            use_abs_pos_emb=True,
            init_values=0.1,
            qkv_bias=False,
            num_patches_per_channel_input=1,
        ).to(device), lora_config),
        num_channels=chan_number,
        in_chans=in_chans,
        num_classes=3,
        embed_dim=200
    ).to(device)
    
    checkpoint_1 = torch.load(checkpoint_path_1, weights_only=False)
    model_task_1.load_state_dict(checkpoint_1['model_state_dict'])

    model_task_2 = SimpleClassifier_multi_model(
        encoder=get_peft_model(create_model(
            'labram_base_patch200_200',
            pretrained=False,
            use_mean_pooling=True,
            init_scale=0.001,
            use_rel_pos_bias=True,
            use_abs_pos_emb=True,
            init_values=0.1,
            qkv_bias=False,
            num_patches_per_channel_input=1,
        ).to(device), lora_config),
        num_channels=chan_number,
        in_chans=in_chans,
        num_classes=3,
        embed_dim=200
    ).to(device)

    # checkpoint_path_2 = f'/mnt/dataset4/yinuo/personlized_FM/for_personalized_FM_0107/contrastive/mask_embedding_label_high_flatten/lora_mask_0.7/classification/Transient_Sensory/train_0.7/m3cv/lora/subject_{sub_id:02d}/best_model.pth'
    checkpoint_2 = torch.load(checkpoint_path_2, weights_only=False)
    model_task_2.load_state_dict(checkpoint_2['model_state_dict'])

    # 保存每个 alpha 下的 test_acc
    test_acc_per_alpha = []

    alpha_values = np.linspace(1, 0, num=11)  # 从 1 到 0，共 11 个值
    for alpha in alpha_values:
        # 根据当前的 alpha 值合并 LoRA 权重
        merged_model = merge_lora_weights(model_task_1, model_task_2, alpha=alpha, beta=1-alpha)

        # 对合并后的模型进行评估
        test_loss, test_acc, test_f1, test_cm = evaluate(
            merged_model, test_loader, criterion, device
        )
        
        # 保存当前 alpha 对应的 test_acc
        test_acc_per_alpha.append(test_acc)
    
    # 将当前 subject 的 test_acc 平均值保存到 average_test_acc_per_alpha 中
    average_test_acc_per_alpha.append(test_acc_per_alpha)

# 计算每个 alpha 值下所有 subject 的平均 test_acc
average_test_acc = np.mean(average_test_acc_per_alpha, axis=0)

# 绘制 alpha 与平均 test_acc 的曲线
plt.plot(alpha_values, average_test_acc, marker='o', linestyle='-', color='b')
plt.title('Average Test Accuracy vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Average Test Accuracy')
plt.grid(True)
plt.savefig(rf'/mnt/dataset4/yinuo/Rest_tuning/result/LaBraM/analysis/baseline_train_0.7_{TASK_1_NAME}_{TASK_2_NAME}_average_test_acc_vs_alpha_{TASK_1_NAME}.png')  # 保存图像
plt.show()
np.save(rf'/mnt/dataset4/yinuo/Rest_tuning/result/LaBraM/analysis/baseline_train_0.7_{TASK_1_NAME}_{TASK_2_NAME}_subject_test_accs_{TASK_1_NAME}.npy', subject_test_accs)
np.save(rf'/mnt/dataset4/yinuo/Rest_tuning/result/LaBraM/analysis/baseline_train_0.7_{TASK_1_NAME}_{TASK_2_NAME}_average_test_acc_per_alpha_{TASK_1_NAME}.npy', average_test_acc_per_alpha)

