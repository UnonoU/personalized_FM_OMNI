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
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from collections import OrderedDict
from src.model import *
from src.utils import *
from timm.models import create_model
from src.data_loader import *
from src.train import *
from src.cbramod import CBraMod

def target_modules_list(value):
    """将逗号分隔的字符串转换为列表"""
    if isinstance(value, str):
        return [m.strip() for m in value.split(",")]
    return value

def get_args():
    parser = argparse.ArgumentParser('Full Fine-tuning Reconstruction V2-Fixed (All Subjects)',
                                     add_help=False)

    # Model
    parser.add_argument('--model', default='labram_base_patch200_200', type=str)
    parser.add_argument('--pretrained_path',
                        default='/mnt/dataset4/fuxy/LaBraM/labram_test/checkpoints/labram-base.pth',
                        type=str,
                        help='Path to ORIGINAL LaBraM Base pretrained weights')
    parser.add_argument('--embed_dim', default=200, type=int)
    parser.add_argument('--patch_size', default=200, type=int)

    # V2-Fixed: Enhanced decoder
    parser.add_argument('--hidden_dim', default=1024, type=int,
                        help='V2-Fixed: Increased from 512 to 1024 for better capacity')
    parser.add_argument('--dropout', default=0.0, type=float,
                        help='V2-Fixed: Dropout rate (0.0 recommended for reconstruction)')

    # LaBraM specific
    parser.add_argument('--use_mean_pooling', action='store_true', default=True)
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true', default=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float)
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)

    # Data
    parser.add_argument('--h5_path',
                        default='/mnt/dataset4/fuxy/m3cv/processed_M3CV_data/m3cv_Resting.h5',
                        type=str,
                        help='Path to HDF5 file containing all subjects')
    parser.add_argument('--test_size', default=0.2, type=float)
    parser.add_argument('--val_size', default=0.1, type=float)
    parser.add_argument('--train_size', default=0.7, type=float)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--normalize_per_channel', action='store_true', default=True,
                        help='V2-Fixed: Set to False for per-trial normalization')
    parser.add_argument('--dataset',default='m3cv',type=str)

    # Training
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--patience', default=30, type=int)
    parser.add_argument('--sgd_lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='SGD momentum')
    parser.add_argument('--min_sgd_lr', default=0.08, type=float)

    # Loss function weights
    parser.add_argument('--mse_weight', default=1.0, type=float,
                        help='V2-Fixed: Weight for MSE loss')
    parser.add_argument('--var_weight', default=0.5, type=float,
                        help='V2-Fixed: Weight for variance loss (critical!)')
    parser.add_argument('--spectral_weight', default=0.3, type=float,
                        help='V2-Fixed: Weight for spectral loss')

    # Paths
    parser.add_argument('--output_root',
                        default='./Subject_to_Subject/m3cv_test',
                        type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--use_amp', action='store_true', default=True)

    # Automation
    parser.add_argument('--min_samples_per_class', default=4, type=int)
    
    parser.add_argument('--fitune_type',default='full',type=str)
    
    # ResEEGNet specific
    parser.add_argument('--residual_hidden_dim', default=512, type=int,
                        help='Hidden dimension for ResEEGNet')
    parser.add_argument('--residual_dropout', default=0.1, type=float,
                        help='V2-Fixed: Dropout rate for ResEEGNet (reduced from 0.2 to 0.1)')

    # V2-Fixed: Enhanced decoder
    parser.add_argument('--head_hidden_dim', default=1024, type=int,
                        help='V2-Fixed: Hidden dimension for reconstruction head (increased from 512 to 1024)')
    parser.add_argument('--head_dropout', default=0.1, type=float,
                        help='V2-Fixed: Dropout rate for reconstruction head (0.0 recommended)')
    
    parser.add_argument('--mask_ratio', default=0.4, type=float)
    parser.add_argument('--mask_type', type=str, choices=['random', 'channel', 'hybrid','non_mask','patch'], help="The type of mask to apply")
    
    # lora
    parser.add_argument('--lora_r', default=16, type=int)
    parser.add_argument('--lora_alpha', default=32, type=int)
    parser.add_argument('--lora_dropout', default=0.1, type=float)
    # parser.add_argument('--target_modules', default="blocks.7.attn.qkv,blocks.8.attn.qkv,blocks.9.attn.qkv,blocks.10.attn.qkv", type=str)
    parser.add_argument('--target_modules', 
                   default="blocks.4.attn.qkv, blocks.5.attn.qkv, blocks.6.attn.qkv", 
                   type=target_modules_list)  # 使用自定义类型转换
    # parser.add_argument('--use_lora', action='store_true', help='use_lora')
    # parser.add_argument('--lora_targets', type=list, default=["q_k_v",], help='lora_targets')

    parser.add_argument('--recon_type', default='MAE', type=str)
    parser.add_argument('--dataset_suffle', action='store_true', help="shuffle") # 在命令行中出现,则值为True
    parser.add_argument('--keep_channel', action='store_true', help="keep_channel") # 在命令行中出现,则值为True
    parser.add_argument('--mask_EEG', action='store_true', help="mask_EEG")
    parser.add_argument('--contrastive_temperature', default=0.1, type=float)
    parser.add_argument('--embedding_high_level', action='store_true', help="Use high-level embeddings for contrastive loss")
    parser.add_argument('--contrastive_task_alpha',default=0.1, type=float)
    parser.add_argument('--contrastive_task_temperature', default=0.07, type=float)
    parser.add_argument('--multi_task',action='store_true')
    parser.add_argument('--multi_task_ids', type=int, nargs='+', default=[0, 1, 2, 3, 4])  # nargs='+' 允许传递多个值
    parser.add_argument('--train_count', type=int, default=None)  # nargs='+' 允许传递多个值
    parser.add_argument('--subject_id', default=None, type=int,
                        help='Specify a single subject ID to train. If not provided, all subjects will be processed.')
    parser.add_argument('--lambda_mse', default=0.1, type=float)
    parser.add_argument('--main_weight', default=1.0, type=float)
        
    return parser.parse_args()



def main(args):
    # args = parse_args()
    # print(args)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    os.makedirs(args.output_root, exist_ok=True)

    print("="*80)
    print(f"Reconstruction with {args.recon_type} (All Subjects)")
    print("="*80)
    # print(f"Data: {args.h5_path}")
    print(f"Pretrained: {args.pretrained_path}")
    print(f"Output: {args.output_root}")
    print(f"Hidden Dim: {args.hidden_dim}, Dropout: {args.dropout}")
    print(f"Loss Weights: MSE={args.mse_weight}, Var={args.var_weight}, Spec={args.spectral_weight}")
    print("="*80)

    # Get all subject IDs
    print("\nScanning subjects in dataset...")
    unique_subject_ids = get_unique_subject_ids(args.h5_path)
    print(f"Found {len(unique_subject_ids)} unique subjects: {unique_subject_ids}")
    if args.subject_id is not None:
        if args.subject_id in unique_subject_ids:
            subject_ids_to_process = [args.subject_id]
            print(f"\n>>> Mode: Single Subject Training (Subject {args.subject_id:02d})")
        else:
            raise ValueError(f"Specified subject_id {args.subject_id} not found in dataset. Available subjects: {unique_subject_ids}")
    else:
        subject_ids_to_process = unique_subject_ids
        print(f"\n>>> Mode: All Subjects Training ({len(unique_subject_ids)} subjects)")

    output_path = os.path.join(args.output_root, args.dataset)
    os.makedirs(output_path, exist_ok=True)
    
    # 保存args到JSON文件
    args_filepath = os.path.join(output_path, 'experiment_args.json')
    save_args_to_json(args, args_filepath)
    # Load dataset once (memory efficient)
    temp_dataset = EEGDataset(
        h5_path=args.h5_path,
        patch_size=args.patch_size,
        normalize=args.normalize,
        normalize_per_channel=args.normalize_per_channel
    )
    sample = temp_dataset[0]
    C, A, T = sample['eeg'].shape
    print(f"Data shape: C={C}, A={A}, T={T}")

    # Initialize results tracking
    all_results = []
    all_train_losses = []
    all_val_losses = []
    error_log = []
    
    error_log_path = os.path.join(output_path, 'error_log.txt')

    from datetime import datetime

    start_time = datetime.now()
    
    
    if args.model == 'labram_base_patch200_200':
        encoder_group = create_model(
        args.model,
        pretrained=False,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
        num_patches_per_channel_input=A).to(device)
        load_vanilla_labram_weights(encoder_group, args.pretrained_path, args.model_key, args.model_prefix)
    elif args.model =='CBraMod':
        encoder_group = CBraMod().to(device)
        cbramod_ckpt = torch.load(args.pretrained_path, map_location=device, weights_only=False)
        # 续训保存的是完整 checkpoint（含 model/optimizer/epoch/args），取其中的模型权重
        if isinstance(cbramod_ckpt, dict) and 'model' in cbramod_ckpt and isinstance(cbramod_ckpt['model'], dict):
            cbramod_state = cbramod_ckpt['model']
        else:
            cbramod_state = cbramod_ckpt
        # 按 key 智能解析：原样匹配优先，匹配不上再剥 module./encoder. 包装前缀。
        # 注意 CBraMod 自身 key 就以 encoder. 开头，无脑剥前缀会破坏原始纯 state_dict 权重。
        model_state = encoder_group.state_dict()
        def _resolve_key(k):
            if k in model_state:
                return k
            prefixes = ('module.', 'encoder.')
            for p1 in prefixes:
                if k.startswith(p1):
                    k1 = k[len(p1):]
                    if k1 in model_state:
                        return k1
                    for p2 in prefixes:
                        if k1.startswith(p2) and k1[len(p2):] in model_state:
                            return k1[len(p2):]
            return None
        # 只保留当前模型中存在且形状一致的 key，过滤掉包装器的 decoder 等多余权重
        matched = {}
        for k, v in cbramod_state.items():
            rk = _resolve_key(k)
            if rk is not None and rk not in matched and model_state[rk].shape == v.shape:
                matched[rk] = v
        skipped = [k for k in cbramod_state if _resolve_key(k) not in matched]
        missing = [k for k in model_state if k not in matched]
        encoder_group.load_state_dict(matched, strict=False)
        print(f"CBraMod pretrained: loaded {len(matched)}/{len(model_state)} keys, "
              f"skipped {len(skipped)} unexpected, {len(missing)} missing (random init)")
        if missing:
            print(f"  Missing keys (first 5): {missing[:5]}")
    elif args.model =='Brainomni':
        from brainomni.model import BrainOmni
        def get_brainomni_model(ckpt_path):
            with open(os.path.join(ckpt_path, "model_cfg.json")) as f:
                cfg = json.load(f)
            model = BrainOmni(**cfg)
            # 加载权重
            pt_path = os.path.join(ckpt_path, "BrainOmni.pt")
            if os.path.exists(pt_path):
                print("Loading weights...")
                model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True), strict=False)
            return model
        encoder_group=get_brainomni_model(args.pretrained_path).to(device)
    elif args.model =='BIOT':
        from src.biot import BIOTEncoder
        encoder_group = BIOTEncoder(emb_size=256, heads=8, depth=4, n_channels=18).to(device)
        
        # Load continue-pretrain checkpoint with proper unpacking
        print(f"Loading BIOT continue-pretrain checkpoint from: {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=device, weights_only=False)
        
        # Handle checkpoint dict format (model/optimizer/epoch/args)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"  Extracted state_dict from checkpoint['model']")
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'biot.' prefix (continue-pretrain uses UnsupervisedPretrainSafe wrapper)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('biot.'):
                new_state_dict[k[5:]] = v  # Remove 'biot.' prefix
            elif k.startswith('prediction.'):
                # Skip prediction MLP (only used during pretraining)
                continue
            else:
                new_state_dict[k] = v
        
        # Load with strict=False to handle any remaining mismatches
        missing, unexpected = encoder_group.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"  Missing keys (random init): {missing[:3]}{'...' if len(missing)>3 else ''}")
        if unexpected:
            print(f"  Unexpected keys (skipped): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")
        print(f"  BIOT continue-pretrain weights loaded successfully!")

    # Main loop
    for idx, subject_id in enumerate(subject_ids_to_process, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Processing Subject {subject_id:02d} ({idx}/{len(subject_ids_to_process)})")
        print(f"{'#'*80}")

        existing_metrics = check_and_load_existing(output_path, subject_id)
        if existing_metrics:
            print(f"  >>>> Subject {subject_id:02d} already processed. Loading existing results and skipping training.")
            all_results.append(existing_metrics)
            continue 
        
        metrics,losses = train_for_subject_embedding_with_dynamics(args, subject_id, device, A, temp_dataset,encoder_group)

        if losses:
            all_train_losses.extend(losses['train_losses'])
            all_val_losses.extend(losses['val_losses'])

            # Store the result for the subject
            all_results.append(losses)


        if metrics is not None:
            all_results.append(metrics)
        else:
            error_msg = f"Subject {subject_id:02d}: Training failed"
            error_log.append(error_msg)
            with open(error_log_path, 'a') as f:
                f.write(f"{datetime.now()}: {error_msg}\n")

    end_time = datetime.now()
    duration = end_time - start_time

    # Generate Summary
    print("\n\n" + "="*80)
    print("Generating Summary Results")
    print("="*80)
    
    import matplotlib.pyplot as plt
    import pandas as pd    
    plt.figure(figsize=(10, 6))
    plt.plot(all_train_losses, label='Train Loss', color='blue')
    plt.plot(all_val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'loss_curve.png'))
    plt.show()

    if len(all_results) > 0:

        # Save overall statistics
        overall_stats = {
            'total_subjects': len(all_results),
            'failed_subjects': len(error_log),
            'duration_seconds': duration.total_seconds()
            }

        with open(os.path.join(args.output_root, f'overall_statistics.json'), 'w') as f:
            json.dump(overall_stats, f, indent=2)

    else:
        print("✗ No successful results to summarize")

    if len(error_log) > 0:
        print("\n" + "="*80)
        print("Error Summary")
        print("="*80)
        for error in error_log:
            print(f"  {error}")

    print("\n" + "="*80)
    print(f"Total Duration: {duration}")
    print("✓ Automated Full Fine-tuning Reconstruction V2-Fixed Completed!")
    print("="*80)
    print("V2-Fixed Applied Fixes:")
    print("  1. ✅ Mixed Loss (MSE + Variance + Spectral)")
    print("  2. ✅ Enhanced decoder (1024 hidden, residual connections)")
    print("  3. ✅ No dropout (0.0)")
    print("  4. ✅ Monitoring metrics (Std Ratio, High-Freq Ratio)")
    print("="*80)


if __name__ == '__main__':
    args = get_args()
    main(args)

