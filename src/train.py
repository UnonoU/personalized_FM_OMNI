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
from src.model import *
from src.utils import *
from timm.models import create_model
from src.data_loader import *
from torch.cuda.amp import GradScaler, autocast
import src.modeling_finetune
# from CBraMod.models.cbramod import CBraMod
from src.cbramod import CBraMod
from src.finetune_trainer import freeze_non_lora_parameters, apply_lora_to_model
# from FEMBA import FEMBA

# def train_single_subject_MAE(args, subject_id, device, num_patches, dataset):
#     """
#     Train full fine-tuning reconstruction model for a single subject (V2-Fixed)

#     Args:
#         args: Command line arguments
#         subject_id: Target subject ID
#         device: CUDA device
#         num_patches: Number of patches (A) for current dataset
#         dataset: Pre-loaded EEGDataset instance

#     Returns:
#         dict: Results dictionary with metrics, or None if failed
#     """
#     try:
#         # Create subject-specific output directory
#         subject_dir = os.path.join(args.output_root, args.dataset, args.recon_type,args.mask_type, args.fitune_type, f'subject_{subject_id:02d}')
#         os.makedirs(subject_dir, exist_ok=True)

#         print(f"\n{'='*80}")
#         print(f"Subject {subject_id:02d} - Loading Data")
#         print(f"{'='*80}")

#         # Create dataloaders with within-class chronological split
#         if args.dataset_suffle:
#             train_loader, val_loader, test_loader = create_single_subject_dataloaders(
#                 dataset=dataset,
#                 target_subject_id=subject_id,
#                 train_size =args.train_size,
#                 test_size=args.test_size,
#                 val_size=args.val_size,
#                 batch_size=args.batch_size,
#                 random_state=args.seed,
#                 num_workers=args.num_workers
#             )
#         else:
#             # Create dataloaders for cognitive task
#             train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
#                 dataset=dataset,
#                 target_subject_id=subject_id,
#                 train_size =args.train_size,
#                 test_size=args.test_size,
#                 val_size=args.val_size,
#                 batch_size=args.batch_size,
#                 random_state=args.seed,
#                 num_workers=args.num_workers
#             )


#         train_size = len(train_loader.dataset)
#         val_size = len(val_loader.dataset)
#         test_size = len(test_loader.dataset)

#         print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

#         # Check minimum samples
#         if train_size < args.min_samples_per_class:
#             raise ValueError(f"Insufficient training samples: {train_size}")

#         in_chans, ch_names, chan_number = get_dataset_name(args)
#         print('in_chans',in_chans)
#         # sample = dataset[0]
#         # C = sample['eeg'].shape[0]
        
#         # Create encoder (fresh initialization for each subject)
#         encoder = create_model(
#             args.model,
#             pretrained=False,
#             use_mean_pooling=args.use_mean_pooling,
#             init_scale=args.init_scale,
#             use_rel_pos_bias=args.rel_pos_bias,
#             use_abs_pos_emb=args.abs_pos_emb,
#             init_values=args.layer_scale_init_value,
#             qkv_bias=args.qkv_bias,
#             num_patches_per_channel_input=num_patches,
#             # force_patch_embed=True
#         )
        
#         # model = create_model(
#         #     args.model,
#         #     pretrained=False,
#         #     drop_rate=args.drop,
#         #     drop_path_rate=args.drop_path,
#         #     attn_drop_rate=args.attn_drop_rate,
#         #     drop_block_rate=None,
#         #     use_mean_pooling=args.use_mean_pooling,
#         #     init_scale=args.init_scale,
#         #     use_rel_pos_bias=args.rel_pos_bias,
#         #     use_abs_pos_emb=args.abs_pos_emb,
#         #     init_values=args.layer_scale_init_value,
#         #     qkv_bias=args.qkv_bias,
#         #     num_patches_per_channel_input=num_patches,
#         # )


#         # Load LaBraM weights
#         load_vanilla_labram_weights(
#             encoder,
#             args.pretrained_path,
#             args.model_key,
#             args.model_prefix
#         )
#         residual_net = ResEEGNet(
#             num_channels=chan_number,
#             num_patches=num_patches,
#             patch_size=args.patch_size,
#             embed_dim=args.embed_dim,
#             hidden_dim=args.residual_hidden_dim,
#             dropout=args.residual_dropout
#         )
#         Encoderwith_Residual=EncoderWithResidual(
#             encoder = encoder,
#             residual_net = residual_net,
#             in_chans = in_chans,
#             embed_dim=args.embed_dim
#         )

#         # Create model with enhanced head
#         if args.fitune_type=='full':
#             model = LaBraM_FullFinetune_MAE(
#                 encoder=encoder,
#                 patch_size=args.patch_size,
#                 embed_dim=args.embed_dim,
#                 hidden_dim=args.hidden_dim,
#                 dropout=args.dropout,
#                 in_chans = in_chans,
#                 mask_ratio = args.mask_ratio,
#                 mask_type = args.mask_type,
#                 keep_channel = args.keep_channel,
#                 mask_EEG = args.mask_EEG,
#                 num_channels = chan_number,
#             ).to(device)
#             trainable_params = [p for p in model.parameters() if p.requires_grad]
#             optimizer = torch.optim.AdamW(
#                 trainable_params,
#                 lr=args.lr,
#                 weight_decay=args.weight_decay
#             )
            
#             # optimizer = torch.optim.AdamW(
#             # model.parameters(),
#             # lr=args.lr,
#             # weight_decay=args.weight_decay
#             # )
            
#         if args.fitune_type=='sgd':
#             model = LaBraM_FullFinetune_MAE(
#                 encoder=encoder,
#                 patch_size=args.patch_size,
#                 embed_dim=args.embed_dim,
#                 hidden_dim=args.hidden_dim,
#                 dropout=args.dropout,
#                 in_chans = in_chans,
#                 mask_ratio = args.mask_ratio,
#                 mask_type = args.mask_type,
#                 keep_channel = args.keep_channel,
#                 mask_EEG = args.mask_EEG,
#                 num_channels = chan_number,
#             ).to(device)
            
#             args.lr=args.sgd_lr
#             args.min_lr =args.min_sgd_lr
#             trainable_params = [p for p in model.parameters() if p.requires_grad]
#             optimizer = torch.optim.SGD( # model.parameters(),
#                 trainable_params,
#                 lr=args.lr,
#                 momentum=args.momentum,
#                 weight_decay=args.weight_decay)
            
#         elif args.fitune_type=='lora':
#             model = LaBraM_LoRA_MAE(
#                 encoder=encoder,
#                 patch_size=args.patch_size,
#                 embed_dim = args.embed_dim,
#                 hidden_dim=args.hidden_dim,
#                 dropout=args.dropout,
#                 in_chans = in_chans,
#                 mask_ratio = args.mask_ratio,
#                 mask_type = args.mask_type,
#                 keep_channel = args.keep_channel,
#                 mask_EEG = args.mask_EEG,
#                 num_channels = chan_number,
#             ).to(device)
#             trainable_params = [p for p in model.parameters() if p.requires_grad]
#             optimizer = torch.optim.AdamW(
#                 trainable_params,
#                 lr=args.lr,
#                 weight_decay=args.weight_decay
#             )            
#             # optimizer = torch.optim.AdamW(
#             # model.parameters(),
#             # lr=args.lr,
#             # weight_decay=args.weight_decay
#             # )
            
#         elif args.fitune_type=='resnet':
#             model = LaBraM_Residual_MAE(
#                 encoder=encoder,
#                 encoder_with_residual=Encoderwith_Residual,
#                 residual_net=residual_net,
#                 patch_size=args.patch_size,
#                 embed_dim=args.embed_dim,
#                 hidden_dim=args.head_hidden_dim,
#                 dropout=args.head_dropout,
#                 in_chans = in_chans,
#                 mask_ratio = args.mask_ratio,
#                 mask_type = args.mask_type,
#                 keep_channel = args.keep_channel,
#                 mask_EEG = args.mask_EEG,
#                 num_channels = chan_number,
#             ).to(device) 
#             trainable_params = [p for p in model.parameters() if p.requires_grad]
#             optimizer = torch.optim.AdamW(
#                 trainable_params,
#                 lr=args.lr,
#                 weight_decay=args.weight_decay
#             )    
#             # optimizer = torch.optim.AdamW(
#             # model.parameters(),
#             # lr=args.lr,
#             # weight_decay=args.weight_decay
#             # )   

#         if args.keep_channel:
#             # criterion = ReconstructionLoss_channel(
#             #     mse_weight=args.mse_weight, 
#             #     spectral_weight=args.spectral_weight, 
#             #     correlation_weight=args.var_weight
#             # )
#             criterion = ReconstructionLoss_channel(
#                 mse_weight=args.mse_weight, 
#                 correlation_weight=args.var_weight
#             )
#         else:
#             criterion = ReconstructionLoss(
#                 mse_weight=args.mse_weight,
#                 var_weight=args.var_weight,
#                 spectral_weight=args.spectral_weight
#             )

#         scaler = GradScaler() if args.use_amp else None

#         best_val_r2 = -float('inf')
#         best_val_std_ratio = 0.0
#         patience_counter = 0

#         print(f"\n{'='*80}")
#         print(f"Subject {subject_id:02d} - Training ({args.fitune_type})")
#         print(f"{'='*80}")
#         print(f"Mixed Loss: MSE={args.mse_weight}, Var={args.var_weight}, Spec={args.spectral_weight}")

#         # Training loop
#         for epoch in range(args.epochs):
#             if epoch < args.warmup_epochs:
#                 lr_mult = (epoch + 1) / args.warmup_epochs
#                 current_lr = args.lr * lr_mult
#             else:
#                 progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
#                 current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = current_lr

#             # Train
#             model.train()
#             train_loss = 0
#             # train_loss_components = {'mse': 0, 'variance': 0, 'spectral': 0}
#             train_loss_components = {'mse': 0, 'correlation': 0}

#             for batch in train_loader:
#                 eeg = batch['eeg'].to(device)
#                 optimizer.zero_grad()
#                 if args.keep_channel:
#                     if args.use_amp:
#                         with torch.amp.autocast('cuda'):
#                             outputs = model(eeg)
#                             target = eeg
#                             loss, loss_dict = criterion(outputs, target)
#                         scaler.scale(loss).backward()
#                         scaler.step(optimizer)
#                         scaler.update()
#                     else:
#                         outputs = model(eeg)
#                         # B, C, A, T = eeg.shape
#                         target = eeg
#                         loss, loss_dict = criterion(outputs, target)
#                         loss.backward()
#                         optimizer.step()
#                 else:
#                     if args.use_amp:
#                         with torch.amp.autocast('cuda'):
#                             outputs = model(eeg)
#                             B, C, A, T = eeg.shape
#                             target = eeg.reshape(B * C, A, T)
#                             loss, loss_dict = criterion(outputs, target)
#                         scaler.scale(loss).backward()
#                         scaler.step(optimizer)
#                         scaler.update()
#                     else:
#                         outputs = model(eeg)
#                         B, C, A, T = eeg.shape
#                         target = eeg.reshape(B * C, A, T)
#                         loss, loss_dict = criterion(outputs, target)
#                         loss.backward()
#                         optimizer.step()

#                 train_loss += loss.item()
#                 for key in train_loss_components.keys():
#                     train_loss_components[key] += loss_dict[key]

#             avg_train_loss = train_loss / len(train_loader)
#             avg_train_components = {k: v / len(train_loader) for k, v in train_loss_components.items()}

#             # Validation
#             val_loss, val_r2, val_pcc, val_std_ratio, val_hf_ratio, val_loss_components = \
#                 evaluate_reconstruction_enhanced(model, val_loader, criterion, device, use_amp=args.use_amp, keep_channel = args.keep_channel)

#             # Print progress
#             if (epoch + 1) % 10 == 0 or epoch == 0:
#                 print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
#                       f"TrainLoss={avg_train_loss:.4f} "
#                       f"(MSE:{avg_train_components['mse']:.4f}, "
#                       f"Corr:{avg_train_components['correlation']:.4f}, "
#                       f"ValR2={val_r2:.4f}, ValPCC={val_pcc:.4f}, "
#                       f"StdRatio={val_std_ratio:.4f} "
#                       f"LR={current_lr:.6f}")

#             # Early stopping
#             if val_r2 > best_val_r2:
#                 best_val_r2 = val_r2
#                 best_val_std_ratio = val_std_ratio
#                 patience_counter = 0
                
#                 if args.fitune_type=='resnet':
#                     torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'encoder_state_dict': model.encoder.state_dict(),
#                     'EncoderWithResidual_state_dict':model.encoder_with_residual.state_dict(),
#                     'val_r2': val_r2,
#                     'val_pcc': val_pcc,
#                     'val_std_ratio': val_std_ratio,
#                     'val_hf_ratio': val_hf_ratio,
#                     'hidden_dim': args.hidden_dim,
#                     'dropout': args.dropout,
#                     'loss_weights': {
#                         'mse': args.mse_weight,
#                         'var': args.var_weight,
#                         'spectral': args.spectral_weight
#                     }
#                 }, os.path.join(subject_dir, 'best_model.pth'))
                              
#                 else:
#                     torch.save({
#                         'epoch': epoch,
#                         'model_state_dict': model.state_dict(),
#                         'encoder_state_dict': model.encoder.state_dict(),
#                         'val_r2': val_r2,
#                         'val_pcc': val_pcc,
#                         'val_std_ratio': val_std_ratio,
#                         'val_hf_ratio': val_hf_ratio,
#                         'hidden_dim': args.hidden_dim,
#                         'dropout': args.dropout,
#                         'loss_weights': {
#                             'mse': args.mse_weight,
#                             'var': args.var_weight,
#                             'spectral': args.spectral_weight
#                         },
#                         'model_type':args.fitune_type
#                     }, os.path.join(subject_dir, 'best_model.pth'))
#             else:
#                 patience_counter += 1
#                 if patience_counter >= args.patience:
#                     print(f"  Early stopping at epoch {epoch+1}")
#                     break

#         # Final test evaluation
#         print(f"\n{'='*80}")
#         print(f"Subject {subject_id:02d} - Final Test Evaluation")
#         print(f"{'='*80}")

#         # checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'))
#         # model.load_state_dict(checkpoint['model_state_dict'])
#         checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
#         model.load_state_dict(checkpoint['model_state_dict'])

#         test_loss, test_r2, test_pcc, test_std_ratio, test_hf_ratio, test_loss_components = \
#             evaluate_reconstruction_enhanced(
#                 model, test_loader, criterion, device, args.use_amp, keep_channel = args.keep_channel)

#         print(f"  Test R²: {test_r2:.4f}, Test PCC: {test_pcc:.4f}")
#         print(f"  Std Ratio: {test_std_ratio:.4f} {'✓ GOOD' if test_std_ratio > 0.7 else '✗ SMOOTHED'}")
#         print(f"  High-Freq Ratio: {test_hf_ratio:.4f}")
#         # print(f"  Loss Components: MSE={test_loss_components['mse']:.4f}, "
#         #       f"Var={test_loss_components['variance']:.4f}, "
#         #       f"Spec={test_loss_components['spectral']:.4f}")
#         print(f"  Loss Components: MSE={test_loss_components['mse']:.4f}, "
#               f"Var={test_loss_components['correlation']:.4f}")

#         # Save metrics
#         metrics = {
#             'subject_id': int(subject_id),
#             'train_size': train_size,
#             'val_size': val_size,
#             'test_size': test_size,
#             'best_val_r2': float(best_val_r2),
#             'best_val_std_ratio': float(best_val_std_ratio),
#             'test_r2': float(test_r2),
#             'test_pcc': float(test_pcc),
#             'test_std_ratio': float(test_std_ratio),
#             'test_hf_ratio': float(test_hf_ratio),
#             'test_loss_components': {k: float(v) for k, v in test_loss_components.items()},
#             'best_epoch': int(checkpoint['epoch']) + 1,
#             'hidden_dim': args.hidden_dim,
#             'dropout': args.dropout,
#             'loss_weights': {
#                 'mse': args.mse_weight,
#                 'var': args.var_weight,
#                 'spectral': args.spectral_weight
#             },
#             # 'version': 'V2-Fixed'
#         }

#         with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
#             json.dump(metrics, f, indent=2)

#         print(f"  ✓ Results saved to {subject_dir}")

#         # Memory cleanup
#         del model
#         del optimizer
#         del encoder
#         torch.cuda.empty_cache()

#         return metrics

#     except Exception as e:
#         print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
#         import traceback
#         traceback.print_exc()

#         if 'model' in locals():
#             del model
#         torch.cuda.empty_cache()

#         return None

# ##############################################################################################################################
# def train_single_subject_MAE_test(args, subject_id, device, num_patches, dataset):
#     """
#     Train full fine-tuning reconstruction model for a single subject (V2-Fixed)

#     Args:
#         args: Command line arguments
#         subject_id: Target subject ID
#         device: CUDA device
#         num_patches: Number of patches (A) for current dataset
#         dataset: Pre-loaded EEGDataset instance

#     Returns:
#         dict: Results dictionary with metrics, or None if failed
#     """
#     try:
#         # Create subject-specific output directory
#         subject_dir = os.path.join(args.output_root, args.dataset, args.recon_type, args.mask_type, args.fitune_type, f'subject_{subject_id:02d}')
#         os.makedirs(subject_dir, exist_ok=True)

#         print(f"\n{'='*80}")
#         print(f"Subject {subject_id:02d} - Loading Data")
#         print(f"{'='*80}")

#         # Create dataloaders with within-class chronological split
#         if args.dataset_suffle:
#             train_loader, val_loader, test_loader = create_single_subject_dataloaders(
#                 dataset=dataset,
#                 target_subject_id=subject_id,
#                 train_size =args.train_size,
#                 test_size=args.test_size,
#                 val_size=args.val_size,
#                 batch_size=args.batch_size,
#                 random_state=args.seed,
#                 num_workers=args.num_workers
#             )
#         else:
#             # Create dataloaders for cognitive task
#             train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
#                 dataset=dataset,
#                 target_subject_id=subject_id,
#                 train_size =args.train_size,
#                 test_size=args.test_size,
#                 val_size=args.val_size,
#                 batch_size=args.batch_size,
#                 random_state=args.seed,
#                 num_workers=args.num_workers
#             )


#         train_size = len(train_loader.dataset)
#         val_size = len(val_loader.dataset)
#         test_size = len(test_loader.dataset)

#         print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

#         # Check minimum samples
#         if train_size < args.min_samples_per_class:
#             raise ValueError(f"Insufficient training samples: {train_size}")

#         in_chans, ch_names, chan_number = get_dataset_name(args)
#         print('in_chans',in_chans)
#         # sample = dataset[0]
#         # C = sample['eeg'].shape[0]
        
#         # Create encoder (fresh initialization for each subject)
#         encoder = create_model(
#             args.model,
#             pretrained=False,
#             use_mean_pooling=args.use_mean_pooling,
#             init_scale=args.init_scale,
#             use_rel_pos_bias=args.rel_pos_bias,
#             use_abs_pos_emb=args.abs_pos_emb,
#             init_values=args.layer_scale_init_value,
#             qkv_bias=args.qkv_bias,
#             num_patches_per_channel_input=num_patches,
#             # force_patch_embed=True
#         )
        
#         # Load LaBraM weights
#         load_vanilla_labram_weights(
#             encoder,
#             args.pretrained_path,
#             args.model_key,
#             args.model_prefix
#         )
#         residual_net = ResEEGNet(
#             num_channels=chan_number,
#             num_patches=num_patches,
#             patch_size=args.patch_size,
#             embed_dim=args.embed_dim,
#             hidden_dim=args.residual_hidden_dim,
#             dropout=args.residual_dropout
#         )
#         Encoderwith_Residual=EncoderWithResidual(
#             encoder = encoder,
#             residual_net = residual_net,
#             in_chans = in_chans,
#             embed_dim=args.embed_dim
#         )

#         # Create model with enhanced head
#         if args.fitune_type=='full':
#             model = LaBraM_FullFinetune_MAE(
#                 encoder=encoder,
#                 patch_size=args.patch_size,
#                 embed_dim=args.embed_dim,
#                 hidden_dim=args.hidden_dim,
#                 dropout=args.dropout,
#                 in_chans = in_chans,
#                 mask_ratio = args.mask_ratio,
#                 mask_type = args.mask_type,
#                 keep_channel = args.keep_channel,
#                 mask_EEG = args.mask_EEG,
#                 num_channels = chan_number,
#             ).to(device)
#             trainable_params = [p for p in model.parameters() if p.requires_grad]
#             optimizer = torch.optim.AdamW(
#                 trainable_params,
#                 lr=args.lr,
#                 weight_decay=args.weight_decay
#             )
            
#             # optimizer = torch.optim.AdamW(
#             # model.parameters(),
#             # lr=args.lr,
#             # weight_decay=args.weight_decay
#             # )
            
#         if args.fitune_type=='sgd':
#             model = LaBraM_FullFinetune_MAE(
#                 encoder=encoder,
#                 patch_size=args.patch_size,
#                 embed_dim=args.embed_dim,
#                 hidden_dim=args.hidden_dim,
#                 dropout=args.dropout,
#                 in_chans = in_chans,
#                 mask_ratio = args.mask_ratio,
#                 mask_type = args.mask_type,
#                 keep_channel = args.keep_channel,
#                 mask_EEG = args.mask_EEG,
#                 num_channels = chan_number,
#             ).to(device)
            
#             args.lr=args.sgd_lr
#             args.min_lr =args.min_sgd_lr
#             trainable_params = [p for p in model.parameters() if p.requires_grad]
#             optimizer = torch.optim.SGD( # model.parameters(),
#                 trainable_params,
#                 lr=args.lr,
#                 momentum=args.momentum,
#                 weight_decay=args.weight_decay)
            
#         elif args.fitune_type=='lora':
#             model = LaBraM_LoRA_MAE_new(
#                 encoder=encoder,
#                 patch_size=args.patch_size,
#                 embed_dim = args.embed_dim,
#                 hidden_dim=args.hidden_dim,
#                 dropout=args.dropout,
#                 in_chans = in_chans,
#                 mask_ratio = args.mask_ratio,
#                 mask_type = args.mask_type,
#                 keep_channel = args.keep_channel,
#                 mask_EEG = args.mask_EEG,
#                 target_modules = args.target_modules,
#                 lora_r=args.lora_r, 
#                 lora_alpha=args.lora_alpha,
#                 lora_dropout=args.lora_dropout,
#                 num_channels = chan_number,
#             ).to(device)
#             trainable_params = [p for p in model.parameters() if p.requires_grad]
#             optimizer = torch.optim.AdamW(
#                 trainable_params,
#                 lr=args.lr,
#                 weight_decay=args.weight_decay
#             )            
#             # optimizer = torch.optim.AdamW(
#             # model.parameters(),
#             # lr=args.lr,
#             # weight_decay=args.weight_decay
#             # )
            
#         elif args.fitune_type=='resnet':
#             model = LaBraM_Residual_MAE(
#                 encoder=encoder,
#                 encoder_with_residual=Encoderwith_Residual,
#                 residual_net=residual_net,
#                 patch_size=args.patch_size,
#                 embed_dim=args.embed_dim,
#                 hidden_dim=args.head_hidden_dim,
#                 dropout=args.head_dropout,
#                 in_chans = in_chans,
#                 mask_ratio = args.mask_ratio,
#                 mask_type = args.mask_type,
#                 keep_channel = args.keep_channel,
#                 mask_EEG = args.mask_EEG,
#                 num_channels = chan_number,
#             ).to(device) 
#             trainable_params = [p for p in model.parameters() if p.requires_grad]
#             optimizer = torch.optim.AdamW(
#                 trainable_params,
#                 lr=args.lr,
#                 weight_decay=args.weight_decay
#             )    
#             # optimizer = torch.optim.AdamW(
#             # model.parameters(),
#             # lr=args.lr,
#             # weight_decay=args.weight_decay
#             # )   

#         if args.keep_channel:
#             # criterion = ReconstructionLoss_channel(
#             #     mse_weight=args.mse_weight, 
#             #     spectral_weight=args.spectral_weight, 
#             #     correlation_weight=args.var_weight
#             # )
#             criterion_MAE = ReconstructionLoss_channel_MAE_test(
#                 mse_weight=args.mse_weight, 
#                 correlation_weight=args.var_weight
#             )
#             criterion_MLP = ReconstructionLoss_channel_MLP(
#                 mse_weight=args.mse_weight, 
#                 correlation_weight=args.var_weight
#             )
#         else:
#             criterion = ReconstructionLoss(
#                 mse_weight=args.mse_weight,
#                 var_weight=args.var_weight,
#                 spectral_weight=args.spectral_weight
#             )
            
#         def count_parameters_with_names(model):
#             """
#             计算模型中的总参数数和可训练参数数，并显示可训练模块的名称
#             """
#             total_params = 0
#             trainable_params = 0
#             trainable_modules = []

#             for name, param in model.named_parameters():
#                 total_params += param.numel()
#                 if param.requires_grad:
#                     trainable_params += param.numel()
#                     trainable_modules.append(name)  # 收集可训练模块的名称

#             return total_params, trainable_params, trainable_modules

#         def calculate_parameter_percentage_with_names(model):
#             """
#             计算可训练参数占总参数的比例，并返回可训练模块的名称
#             """
#             total_params, trainable_params, trainable_modules = count_parameters_with_names(model)
#             trainable_percentage = (trainable_params / total_params) * 100
#             return total_params, trainable_params, trainable_percentage, trainable_modules

#         total_params, trainable_params, trainable_percentage, trainable_modules = calculate_parameter_percentage_with_names(model)

#         print(f"Total parameters: {total_params}")
#         print(f"Trainable parameters: {trainable_params}")
#         print(f"Trainable parameters percentage: {trainable_percentage:.2f}%")
#         # print("\nTrainable modules:")
#         # for module_name in trainable_modules:
#         #     print(f"- {module_name}")
        
        
        
        
#         scaler = GradScaler() if args.use_amp else None

#         best_val_r2 = -float('inf')
#         best_val_std_ratio = 0.0
#         patience_counter = 0
#         best_val_FC = -float('inf')

#         print(f"\n{'='*80}")
#         print(f"Subject {subject_id:02d} - Training ({args.fitune_type})")
#         print(f"{'='*80}")
#         print(f"Mixed Loss: MSE={args.mse_weight}, Var={args.var_weight}, Spec={args.spectral_weight}")

#         # Training loop
#         for epoch in range(args.epochs):
#             if epoch < args.warmup_epochs:
#                 lr_mult = (epoch + 1) / args.warmup_epochs
#                 current_lr = args.lr * lr_mult
#             else:
#                 progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
#                 current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = current_lr

#             # Train
#             model.train()
#             train_loss = 0
#             # train_loss_components = {'mse': 0, 'variance': 0, 'spectral': 0}
#             train_loss_components = {'mse': 0, 'correlation': 0}

#             for batch in train_loader:
#                 eeg = batch['eeg'].to(device)
#                 optimizer.zero_grad()
#                 if args.keep_channel:
#                     outputs, mask, embedding = model(eeg)
#                     B, C, A, T = eeg.shape
#                     target = eeg
#                     loss, loss_dict = criterion_MAE(outputs, target, mask,embedding)
#                     loss.backward()
#                     optimizer.step()
#                 else:
#                     outputs, mask,embedding = model(eeg)
#                     B, C, A, T = eeg.shape
#                     target = eeg.reshape(B * C, A, T)
#                     loss, loss_dict = criterion_MAE(outputs, target, mask,embedding)
#                     loss.backward()
#                     optimizer.step()

#                 train_loss += loss.item()
#                 for key in train_loss_components.keys():
#                     train_loss_components[key] += loss_dict[key]

#             avg_train_loss = train_loss / len(train_loader)
#             avg_train_components = {k: v / len(train_loader) for k, v in train_loss_components.items()}

#             # Validation
#             val_loss, val_r2, val_pcc, val_std_ratio, val_hf_ratio, val_loss_components,val_FC = \
#                 evaluate_reconstruction_enhanced_val(model, val_loader, criterion_MAE, device, use_amp=args.use_amp,keep_channel = args.keep_channel)

#             # Print progress
#             if (epoch + 1) % 10 == 0 or epoch == 0:
#                 print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
#                       f"TrainLoss={avg_train_loss:.4f} "
#                       f"(MSE:{avg_train_components['mse']:.4f}, "
#                       f"Corr:{avg_train_components['correlation']:.4f}, "
#                       f"ValR2={val_r2:.4f}, ValPCC={val_pcc:.4f}, "
#                       f"StdRatio={val_std_ratio:.4f} "
#                       f"LR={current_lr:.6f},"
#                       f"FC={val_FC:.6f},")

#             # Early stopping
#             # if val_r2 > best_val_r2:
#             #     best_val_r2 = val_r2
#             #     best_val_std_ratio = val_std_ratio
#             #     patience_counter = 0
#             if val_FC > best_val_FC:
#                 best_val_FC = val_FC
#                 best_val_std_ratio = val_std_ratio
#                 patience_counter = 0
            
#                 if args.fitune_type=='resnet':
#                     torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'encoder_state_dict': model.encoder.state_dict(),
#                     'EncoderWithResidual_state_dict':model.encoder_with_residual.state_dict(),
#                     'val_r2': val_r2,
#                     'val_pcc': val_pcc,
#                     'val_std_ratio': val_std_ratio,
#                     'val_hf_ratio': val_hf_ratio,
#                     'hidden_dim': args.hidden_dim,
#                     'dropout': args.dropout,
#                     'loss_weights': {
#                         'mse': args.mse_weight,
#                         'var': args.var_weight,
#                         'spectral': args.spectral_weight
#                     }
#                 }, os.path.join(subject_dir, 'best_model.pth'))
                              
#                 else:
#                     torch.save({
#                         'epoch': epoch,
#                         'model_state_dict': model.state_dict(),
#                         'encoder_state_dict': model.encoder.state_dict(),
#                         'val_r2': val_r2,
#                         'val_pcc': val_pcc,
#                         'val_std_ratio': val_std_ratio,
#                         'val_hf_ratio': val_hf_ratio,
#                         'hidden_dim': args.hidden_dim,
#                         'dropout': args.dropout,
#                         'loss_weights': {
#                             'mse': args.mse_weight,
#                             'var': args.var_weight,
#                             'spectral': args.spectral_weight
#                         },
#                         'model_type':args.fitune_type
#                     }, os.path.join(subject_dir, 'best_model.pth'))
#             else:
#                 patience_counter += 1
#                 if patience_counter >= args.patience:
#                     print(f"  Early stopping at epoch {epoch+1}")
#                     break

#         # Final test evaluation
#         print(f"\n{'='*80}")
#         print(f"Subject {subject_id:02d} - Final Test Evaluation")
#         print(f"{'='*80}")

#         # checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'))
#         # model.load_state_dict(checkpoint['model_state_dict'])
#         checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
#         model.load_state_dict(checkpoint['model_state_dict'])

#         test_loss, test_r2, test_pcc, test_std_ratio, test_hf_ratio, test_loss_components = \
#             evaluate_reconstruction_enhanced(
#                 model, test_loader, criterion_MLP, device, args.use_amp, keep_channel = args.keep_channel)

#         print(f"  Test R²: {test_r2:.4f}, Test PCC: {test_pcc:.4f}")
#         print(f"  Std Ratio: {test_std_ratio:.4f} {'✓ GOOD' if test_std_ratio > 0.7 else '✗ SMOOTHED'}")
#         print(f"  High-Freq Ratio: {test_hf_ratio:.4f}")
#         # print(f"  Loss Components: MSE={test_loss_components['mse']:.4f}, "
#         #       f"Var={test_loss_components['variance']:.4f}, "
#         #       f"Spec={test_loss_components['spectral']:.4f}")
#         print(f"  Loss Components: MSE={test_loss_components['mse']:.4f}, "
#               f"Var={test_loss_components['correlation']:.4f}")

#         # Save metrics
#         metrics = {
#             'subject_id': int(subject_id),
#             'train_size': train_size,
#             'val_size': val_size,
#             'test_size': test_size,
#             'best_val_r2': float(best_val_r2),
#             'best_val_std_ratio': float(best_val_std_ratio),
#             'test_r2': float(test_r2),
#             'test_pcc': float(test_pcc),
#             'test_std_ratio': float(test_std_ratio),
#             'test_hf_ratio': float(test_hf_ratio),
#             'test_loss_components': {k: float(v) for k, v in test_loss_components.items()},
#             'best_epoch': int(checkpoint['epoch']) + 1,
#             'hidden_dim': args.hidden_dim,
#             'dropout': args.dropout,
#             'loss_weights': {
#                 'mse': args.mse_weight,
#                 'var': args.var_weight,
#                 'spectral': args.spectral_weight
#             },
#             # 'version': 'V2-Fixed'
#         }

#         with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
#             json.dump(metrics, f, indent=2)

#         print(f"  ✓ Results saved to {subject_dir}")

#         # Memory cleanup
#         del model
#         del optimizer
#         del encoder
#         torch.cuda.empty_cache()

#         return metrics

#     except Exception as e:
#         print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
#         import traceback
#         traceback.print_exc()

#         if 'model' in locals():
#             del model
#         torch.cuda.empty_cache()

#         return None



# def train_single_subject_MLP(args, subject_id, device, num_patches, dataset):
#     """
#     Train full fine-tuning reconstruction model for a single subject (V2-Fixed)

#     Args:
#         args: Command line arguments
#         subject_id: Target subject ID
#         device: CUDA device
#         num_patches: Number of patches (A) for current dataset
#         dataset: Pre-loaded EEGDataset instance

#     Returns:
#         dict: Results dictionary with metrics, or None if failed
#     """
#     try:
#         # Create subject-specific output directory
#         subject_dir = os.path.join(args.output_root, args.dataset, args.recon_type, 
#                                    args.mask_type, args.fitune_type, f'subject_{subject_id:02d}')
#         os.makedirs(subject_dir, exist_ok=True)

#         print(f"\n{'='*80}")
#         print(f"Subject {subject_id:02d} - Loading Data")
#         print(f"{'='*80}")

#         # Create dataloaders with within-class chronological split
#         train_loader, val_loader, test_loader = create_single_subject_dataloaders(
#             dataset=dataset,
#             target_subject_id=subject_id,
#             test_size=args.test_size,
#             val_size=args.val_size,
#             batch_size=args.batch_size,
#             random_state=args.seed,
#             num_workers=args.num_workers
#         )

#         train_size = len(train_loader.dataset)
#         val_size = len(val_loader.dataset)
#         test_size = len(test_loader.dataset)

#         print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

#         # Check minimum samples
#         if train_size < args.min_samples_per_class:
#             raise ValueError(f"Insufficient training samples: {train_size}")

#         in_chans, ch_names, chan_number = get_dataset_name(args)
        
#         encoder = create_model(
#             args.model,
#             pretrained=False,
#             use_mean_pooling=args.use_mean_pooling,
#             init_scale=args.init_scale,
#             use_rel_pos_bias=args.rel_pos_bias,
#             use_abs_pos_emb=args.abs_pos_emb,
#             init_values=args.layer_scale_init_value,
#             qkv_bias=args.qkv_bias,
#             num_patches_per_channel_input=num_patches,
#             # force_patch_embed=True
#         )

#         # Load LaBraM weights
#         load_vanilla_labram_weights(
#             encoder,
#             args.pretrained_path,
#             args.model_key,
#             args.model_prefix
#         )
#         residual_net = ResEEGNet(
#             num_channels=chan_number,
#             num_patches=num_patches,
#             patch_size=args.patch_size,
#             embed_dim=args.embed_dim,
#             hidden_dim=args.residual_hidden_dim,
#             dropout=args.residual_dropout
#         )
        

#         # Create model with enhanced head
#         if args.fitune_type=='full':
#             model = LaBraM_FullFinetune_MLP(
#                 encoder=encoder,
#                 patch_size=args.patch_size,
#                 embed_dim=args.embed_dim,
#                 hidden_dim=args.hidden_dim,
#                 dropout=args.dropout,
#                 in_chans = in_chans,
#                 keep_channel = args.keep_channel,
#                 mask_EEG = args.mask_EEG,
#                 num_channels = chan_number,
#             ).to(device)
            
#             # optimizer = torch.optim.AdamW(
#             # model.parameters(),
#             # lr=args.lr,
#             # weight_decay=args.weight_decay
#             # )
#             trainable_params = [p for p in model.parameters() if p.requires_grad]
#             optimizer = torch.optim.AdamW(
#                 trainable_params,
#                 lr=args.lr,
#                 weight_decay=args.weight_decay
#             )   
                        
#         if args.fitune_type=='sgd':
#             model = LaBraM_FullFinetune_MLP(
#                 encoder=encoder,
#                 patch_size=args.patch_size,
#                 embed_dim=args.embed_dim,
#                 hidden_dim=args.hidden_dim,
#                 dropout=args.dropout,
#                 in_chans = in_chans,
#                 keep_channel = args.keep_channel,
#                 mask_EEG = args.mask_EEG,
#                 num_channels = chan_number,
#             ).to(device)
#             args.lr=args.sgd_lr
#             args.min_lr =args.min_sgd_lr
#             # optimizer = torch.optim.SGD(
#             #     model.parameters(),
#             #     lr=args.lr,
#             #     momentum=args.momentum,
#             #     weight_decay=args.weight_decay)
#             trainable_params = [p for p in model.parameters() if p.requires_grad]
#             optimizer = torch.optim.SGD(
#                 trainable_params,
#                 lr=args.lr,
#                 momentum=args.momentum,
#                 weight_decay=args.weight_decay)
            
#         elif args.fitune_type=='lora':
#             model = LaBraM_LoRA_MLP(
#                 encoder=encoder,
#                 patch_size=args.patch_size,
#                 embed_dim = args.embed_dim,
#                 hidden_dim=args.hidden_dim,
#                 dropout=args.dropout,
#                 in_chans = in_chans,
#                 keep_channel = args.keep_channel,
#                 mask_EEG = args.mask_EEG,
#                 num_channels = chan_number,
#             ).to(device)
#             trainable_params = [p for p in model.parameters() if p.requires_grad]
#             optimizer = torch.optim.AdamW(
#                 trainable_params,
#                 lr=args.lr,
#                 weight_decay=args.weight_decay
#             )                
#             # optimizer = torch.optim.AdamW(
#             #     model.parameters(),
#             #     lr=args.lr,
#             #     weight_decay=args.weight_decay
#             #     )
            
#         elif args.fitune_type=='resnet':
#             model = LaBraM_Residual_MLP(
#                 encoder=encoder,
#                 residual_net=residual_net,
#                 patch_size=args.patch_size,
#                 embed_dim=args.embed_dim,
#                 hidden_dim=args.head_hidden_dim,
#                 dropout=args.head_dropout,
#                 in_chans = in_chans,
#             ).to(device) 
#             trainable_params = [p for p in model.parameters() if p.requires_grad]
#             optimizer = torch.optim.AdamW(
#                 trainable_params,
#                 lr=args.lr,
#                 weight_decay=args.weight_decay
#             )    
#             # optimizer = torch.optim.AdamW(
#             # model.parameters(),
#             # lr=args.lr,
#             # weight_decay=args.weight_decay
#             # )   

#         if args.keep_channel:
#             criterion = ReconstructionLoss_channel(
#                 mse_weight=args.mse_weight, 
#                 spectral_weight=args.spectral_weight, 
#                 correlation_weight=args.var_weight
#             )
#         else:
#             criterion = ReconstructionLoss(
#                 mse_weight=args.mse_weight,
#                 var_weight=args.var_weight,
#                 spectral_weight=args.spectral_weight
#             )

#         scaler = GradScaler() if args.use_amp else None

#         best_val_r2 = -float('inf')
#         best_val_std_ratio = 0.0
#         patience_counter = 0

#         print(f"\n{'='*80}")
#         print(f"Subject {subject_id:02d} - Training (V2-Fixed: {args.fitune_type})")
#         print(f"{'='*80}")
#         print(f"Mixed Loss: MSE={args.mse_weight}, Var={args.var_weight}, Spec={args.spectral_weight}")

#         # Training loop
#         for epoch in range(args.epochs):
#             if epoch < args.warmup_epochs:
#                 lr_mult = (epoch + 1) / args.warmup_epochs
#                 current_lr = args.lr * lr_mult
#             else:
#                 progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
#                 current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = current_lr

#             # Train
#             model.train()
#             train_loss = 0
#             train_loss_components = {'mse': 0, 'correlation': 0}

#             # for batch in train_loader:
#             #     eeg = batch['eeg'].to(device)

#             #     optimizer.zero_grad()
#             for batch in train_loader:
#                 eeg = batch['eeg'].to(device)
#                 optimizer.zero_grad()
#                 if args.keep_channel:
#                     if args.use_amp:
#                         with torch.amp.autocast('cuda'):
#                             outputs = model(eeg)
#                             # B, C, A, T = eeg.shape
#                             target = eeg
#                             loss, loss_dict = criterion(outputs, target)
#                         scaler.scale(loss).backward()
#                         scaler.step(optimizer)
#                         scaler.update()
#                     else:
#                         outputs = model(eeg)
#                         # B, C, A, T = eeg.shape
#                         target = eeg
#                         loss, loss_dict = criterion(outputs, target)
#                         loss.backward()
#                         optimizer.step()
#                 else:
#                     if args.use_amp:
#                         with torch.amp.autocast('cuda'):
#                             outputs = model(eeg)
#                             B, C, A, T = eeg.shape
#                             target = eeg.reshape(B * C, A, T)
#                             loss, loss_dict = criterion(outputs, target)
#                         scaler.scale(loss).backward()
#                         scaler.step(optimizer)
#                         scaler.update()
#                     else:
#                         outputs = model(eeg)
#                         B, C, A, T = eeg.shape
#                         target = eeg.reshape(B * C, A, T)
#                         loss, loss_dict = criterion(outputs, target)
#                         loss.backward()
#                         optimizer.step()
                        
#                 # if args.use_amp:
#                 #     with torch.amp.autocast('cuda'):
#                 #         outputs = model(eeg)
#                 #         B, C, A, T = eeg.shape
#                 #         target = eeg.reshape(B * C, A, T)
#                 #         loss, loss_dict = criterion(outputs, target)
#                 #     scaler.scale(loss).backward()
#                 #     scaler.step(optimizer)
#                 #     scaler.update()
#                 # else:
#                 #     outputs = model(eeg)
#                 #     B, C, A, T = eeg.shape
#                 #     target = eeg.reshape(B * C, A, T)
#                 #     loss, loss_dict = criterion(outputs, target)
#                 #     loss.backward()
#                 #     optimizer.step()

#                 train_loss += loss.item()
#                 for key in train_loss_components.keys():
#                     train_loss_components[key] += loss_dict[key]

#             avg_train_loss = train_loss / len(train_loader)
#             avg_train_components = {k: v / len(train_loader) for k, v in train_loss_components.items()}

#             # Validation
#             val_loss, val_r2, val_pcc, val_std_ratio, val_hf_ratio, val_loss_components = \
#                 evaluate_reconstruction_enhanced(
#                     model, val_loader, criterion, device, use_amp=args.use_amp,keep_channel = args.keep_channel
#                 )

#             # Print progress
#             if (epoch + 1) % 10 == 0 or epoch == 0:
#                 print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
#                       f"TrainLoss={avg_train_loss:.4f} "
#                       f"(MSE:{avg_train_components['mse']:.4f}, "
#                       f"Var:{avg_train_components['variance']:.4f}, "
#                       f"Spec:{avg_train_components['spectral']:.4f}), "
#                       f"ValR2={val_r2:.4f}, ValPCC={val_pcc:.4f}, "
#                       f"StdRatio={val_std_ratio:.4f} "
#                       f"LR={current_lr:.6f}")

#             # Early stopping
#             if val_r2 > best_val_r2:
#                 best_val_r2 = val_r2
#                 best_val_std_ratio = val_std_ratio
#                 patience_counter = 0
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'encoder_state_dict': model.encoder.state_dict(),
#                     'val_r2': val_r2,
#                     'val_pcc': val_pcc,
#                     'val_std_ratio': val_std_ratio,
#                     'val_hf_ratio': val_hf_ratio,
#                     'hidden_dim': args.hidden_dim,
#                     'dropout': args.dropout,
#                     'loss_weights': {
#                         'mse': args.mse_weight,
#                         'var': args.var_weight,
#                         'spectral': args.spectral_weight
#                     }
#                 }, os.path.join(subject_dir, 'best_model.pth'))
#             else:
#                 patience_counter += 1
#                 if patience_counter >= args.patience:
#                     print(f"  Early stopping at epoch {epoch+1}")
#                     break

#         # Final test evaluation
#         print(f"\n{'='*80}")
#         print(f"Subject {subject_id:02d} - Final Test Evaluation")
#         print(f"{'='*80}")

#         # checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'))
#         # model.load_state_dict(checkpoint['model_state_dict'])
        
#         checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
#         model.load_state_dict(checkpoint['model_state_dict'])

#         test_loss, test_r2, test_pcc, test_std_ratio, test_hf_ratio, test_loss_components = \
#             evaluate_reconstruction_enhanced(
#                 model, test_loader, criterion, device, args.use_amp,keep_channel = args.keep_channel
#             )

#         print(f"  Test R²: {test_r2:.4f}, Test PCC: {test_pcc:.4f}")
#         print(f"  Std Ratio: {test_std_ratio:.4f} {'✓ GOOD' if test_std_ratio > 0.7 else '✗ SMOOTHED'}")
#         print(f"  High-Freq Ratio: {test_hf_ratio:.4f}")
#         print(f"  Loss Components: MSE={test_loss_components['mse']:.4f}, "
#               f"Var={test_loss_components['variance']:.4f}, "
#               f"Spec={test_loss_components['spectral']:.4f}")

#         # Save metrics
#         metrics = {
#             'subject_id': int(subject_id),
#             'train_size': train_size,
#             'val_size': val_size,
#             'test_size': test_size,
#             'best_val_r2': float(best_val_r2),
#             'best_val_std_ratio': float(best_val_std_ratio),
#             'test_r2': float(test_r2),
#             'test_pcc': float(test_pcc),
#             'test_std_ratio': float(test_std_ratio),
#             'test_hf_ratio': float(test_hf_ratio),
#             'test_loss_components': {k: float(v) for k, v in test_loss_components.items()},
#             'best_epoch': int(checkpoint['epoch']) + 1,
#             'hidden_dim': args.hidden_dim,
#             'dropout': args.dropout,
#             'loss_weights': {
#                 'mse': args.mse_weight,
#                 'var': args.var_weight,
#                 'spectral': args.spectral_weight
#             },
#             'version': 'V2-Fixed'
#         }

#         with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
#             json.dump(metrics, f, indent=2)

#         print(f"  ✓ Results saved to {subject_dir}")

#         # Memory cleanup
#         del model
#         del optimizer
#         del encoder
#         torch.cuda.empty_cache()

#         return metrics

#     except Exception as e:
#         print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
#         import traceback
#         traceback.print_exc()

#         if 'model' in locals():
#             del model
#         torch.cuda.empty_cache()

#         return None
    
    
    


def load_pretrained_encoder(encoder, checkpoint_path, method, device):
    """
    Load pretrained encoder from reconstruction checkpoint

    Args:
        encoder: LaBraM encoder model
        checkpoint_path: Path to checkpoint
        method: 'full', 'sgd', 'lora', or 'resnet'
        device: Device to load on

    Returns:
        Loaded encoder
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract encoder state dict based on method
    if method in ['full', 'sgd','lora']:
        # Full fine-tuning: state dict has 'encoder_state_dict' or 'model_state_dict'
        if 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']

    # elif method == 'lora':
    #     # LoRA: state dict has LoRA weights + encoder weights
    #     if 'encoder_state_dict' in checkpoint:
    #         state_dict = checkpoint['encoder_state_dict']
       
    elif method == 'resnet':
        # Residual: state dict has encoder + residual adapter
        if 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']
        elif 'model_state_dict' in checkpoint:
            full_state = checkpoint['model_state_dict']
            state_dict = OrderedDict()
            for key, value in full_state.items():
                if key.startswith('encoder.'):
                    new_key = key.replace('encoder.', '')
                    state_dict[new_key] = value
        else:
            state_dict = checkpoint

    else:
        raise ValueError(f"Unknown method: {method}")

    # Load state dict (strict=False to allow missing/extra keys)
    missing_keys, unexpected_keys = encoder.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"  Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  Unexpected keys: {len(unexpected_keys)}")

    return encoder

def train_single_subject_cls_test(args, subject_id, device, num_patches, dataset):
    """Train transfer learning classifier for single subject with frozen encoder"""
    try:
        # Create subject-specific output directory
        output_path = os.path.join(args.output_root, args.dataset,  args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        # Check for checkpoint resumption
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data (Cognitive Task)")
        print(f"{'='*80}")

        
        if args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        else:
            # Create dataloaders for cognitive task
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset) if not args.fewshot else 0
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")
        
        in_chans, ch_names, chan_number = get_dataset_name(args)
        print('in_chans',in_chans)

        # Create encoder
        print(f"\n[Step 1] Creating Encoder ({args.model})...")
        encoder = create_model(
            args.model,
            pretrained=False,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            qkv_bias=args.qkv_bias,
            num_patches_per_channel_input=num_patches,
        )
        
        residual_net = ResEEGNet(
            num_channels=chan_number,
            num_patches=num_patches,
            patch_size=200,
            embed_dim=args.embed_dim,
            hidden_dim=args.residual_hidden_dim,
            dropout=args.residual_dropout
        )
        
        Encoderwith_Residual=EncoderWithResidual(
            encoder = encoder,
            residual_net = residual_net,
            in_chans = in_chans,
            embed_dim=args.embed_dim
        )
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,  # Targets Q/K/V/Projection layers
        )
        
        # Load pretrained encoder from reconstruction task
        checkpoint_path = os.path.join(args.recon_results_dir, args.dataset, f'subject_{subject_id:02d}','best_model.pth')
        print(f"  Checkpoint: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {checkpoint_path}")

        if args.cls_fitune_type =='lora':
            encoder_with_lora = get_peft_model(encoder, lora_config)
            encoder = load_pretrained_encoder(encoder_with_lora, checkpoint_path, args.fitune_type, device)
            for name, param in encoder.named_parameters():
                param.requires_grad = False
            lora_config_cls = LoraConfig(
                r=args.cls_lora_r,
                lora_alpha=args.cls_lora_alpha,
                lora_dropout=args.cls_lora_dropout,
                target_modules=args.cls_target_modules,
                bias="none"
            )
            encoder_cls_lora = get_peft_model(encoder, lora_config_cls)
            model = SimpleFinetuneClassifier_lora(
                encoder=encoder_cls_lora ,
                num_channels= chan_number,
                in_chans = in_chans,
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)
        else:
            encoder_with_lora = get_peft_model(encoder, lora_config)
            encoder_with_lora = load_pretrained_encoder(encoder_with_lora, checkpoint_path, args.fitune_type, device)
            model = SimpleFinetuneClassifier(
                encoder=encoder_with_lora,
                encoder_free=args.encoder_free,
                num_channels= chan_number,
                in_chans = in_chans,
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)
                
        # Setup training (only classification head parameters)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"\n  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training (Linear Probing, Frozen Encoder)")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        for epoch in range(args.epochs):
            # Learning rate scheduling
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Train
            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']

                eeg = eeg.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Save the best model (based on training loss or other metrics)
            if epoch == 0 or avg_train_loss < best_train_loss:  # Save based on the lowest training loss
                best_train_loss = avg_train_loss
                print(f"  Saving Best Model at Epoch {epoch+1} with Train Loss: {best_train_loss:.4f}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'classifier_state_dict': model.classifier.state_dict(),
                    'train_loss': best_train_loss
                }, os.path.join(subject_dir, 'best_model.pth'))

            # Print progress
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                f"TrainLoss={avg_train_loss:.4f}, "
                f"LR={current_lr:.6f}")

        # Final test evaluation
        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")


        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_f1, test_cm = evaluate(
            model, test_loader, criterion, device, args.use_amp
        )

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        # Save metrics
        metrics = {
            'subject_id': int(subject_id),
            'method': args.fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc) if not args.fewshot else None,
            'best_val_f1': float(best_val_f1) if not args.fewshot else None,
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1 if not args.fewshot else None,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification'
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        # Memory cleanup
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None


def train_single_subject_cls(args, subject_id, device, num_patches, dataset):
    """Train transfer learning classifier for single subject with frozen encoder"""
    try:
        # Create subject-specific output directory
        output_path = os.path.join(args.output_root, args.dataset,  args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        # Check for checkpoint resumption
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data (Cognitive Task)")
        print(f"{'='*80}")

        
        if args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        else:
            # Create dataloaders for cognitive task
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")
        
        in_chans, ch_names, chan_number = get_dataset_name(args)
        print('in_chans',in_chans)

        # Create encoder
        print(f"\n[Step 1] Creating Encoder ({args.model})...")
        encoder = create_model(
            args.model,
            pretrained=False,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            qkv_bias=args.qkv_bias,
            num_patches_per_channel_input=num_patches,
        )
        
        residual_net = ResEEGNet(
            num_channels=chan_number,
            num_patches=num_patches,
            patch_size=200,
            embed_dim=args.embed_dim,
            hidden_dim=args.residual_hidden_dim,
            dropout=args.residual_dropout
        )
        
        Encoderwith_Residual=EncoderWithResidual(
            encoder = encoder,
            residual_net = residual_net,
            in_chans = in_chans,
            embed_dim=args.embed_dim
        )
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,  # Targets Q/K/V/Projection layers
        )
        checkpoint_path = os.path.join(args.recon_results_dir, args.dataset, f'subject_{subject_id:02d}','best_model.pth')
        print(f"  Checkpoint: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {checkpoint_path}")

        # if args.fitune_type=='resnet':
        #     res_encoder = load_pretrained_encoder(Encoderwith_Residual, checkpoint_path, args.fitune_type, device)
        #     model = SimpleFinetuneClassifier_resnet(
        #         encoder=res_encoder,
        #         encoder_free=args.encoder_free,
        #         num_channels= chan_number,
        #         in_chans = in_chans,
        #         num_classes= args.num_classes,
        #         embed_dim= args.embed_dim
        #     ).to(device)
            
        if args.cls_fitune_type =='lora':
            encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
            encoder = create_lora(encoder,args)
            for name, param in encoder.named_parameters():
                if param.requires_grad:  # 如果参数是可训练的
                    print(f"Trainable parameter: {name}")            
            # for name, param in encoder.named_parameters():
            #     if 'lora' not in name:  # 只冻结非LoRA参数
            #         param.requires_grad = False
            #     else:
            #         param.requires_grad = True
            # for name, param in encoder_cls_lora.named_parameters():
            #     if param.requires_grad:
            #         print(f"可训练: {name}")      
            model = SimpleClassifier(
                encoder=encoder ,
                num_channels= chan_number,
                in_chans = in_chans,
                args = args,  
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)
            
        elif args.cls_fitune_type =='full':
            encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
            for name, param in encoder.named_parameters():
                param.requires_grad = True
            # encoder_with_lora = get_peft_model(encoder, lora_config)
            model = SimpleClassifier(
                encoder=encoder,
                num_channels= chan_number,
                in_chans = in_chans,
                args = args,  
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)
            
        elif args.cls_fitune_type =='linear':
            encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
            for name, param in encoder.named_parameters():
                param.requires_grad = False
            model = SimpleClassifier(
                encoder=encoder,
                num_channels= chan_number,
                in_chans = in_chans,
                args = args,  
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)

        # Setup training (only classification head parameters)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"\n  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None

        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training (Linear Probing, Frozen Encoder)")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        # Training loop
        for epoch in range(args.epochs):
            # Learning rate scheduling
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Train
            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']

                eeg = eeg.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_loss, val_acc, val_f1, val_cm = evaluate(
                model, val_loader, criterion, device, args.use_amp
            )

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'classifier_state_dict': model.classifier.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Final test evaluation
        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_f1, test_cm = evaluate(
            model, test_loader, criterion, device, args.use_amp
        )

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        # Save metrics
        metrics = {
            'subject_id': int(subject_id),
            'method': args.fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification'
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        # Memory cleanup
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None


def train_single_subject_cls_derangement(args, data_subject_id, ckpt_subject_id, device, num_patches, dataset):
    """Train transfer learning classifier for single subject with frozen encoder"""
    try:
        valid_fitune_types = ['lora', 'full']
        valid_cls_fitune_types = ['lora', 'full', 'linear']
        if args.fitune_type not in valid_fitune_types:
            raise ValueError(f"fitune_type must be one of {valid_fitune_types}")
        if args.cls_fitune_type not in valid_cls_fitune_types:
            raise ValueError(f"cls_fitune_type must be one of {valid_cls_fitune_types}")
        # [修改 2]: 区分输出路径，避免错配实验覆盖基线结果，文件夹命名包含两个 ID
        is_mismatch = (data_subject_id != ckpt_subject_id)
        folder_suffix = "_mismatch" if is_mismatch else ""
        output_path = os.path.join(args.output_root, args.dataset,  args.cls_fitune_type + folder_suffix)
        
        subject_dir_name = f'subject_data{data_subject_id:02d}_ckpt{ckpt_subject_id:02d}'
        subject_dir = os.path.join(output_path, subject_dir_name)
        os.makedirs(subject_dir, exist_ok=True)

        # Check for checkpoint resumption
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Data Sub {data_subject_id:02d} with Ckpt Sub {ckpt_subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Data Subject {data_subject_id:02d} | Ckpt Subject {ckpt_subject_id:02d} - Loading Data (Cognitive Task)")
        print(f"{'='*80}")

        
        if args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=data_subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=data_subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        else:
            # Create dataloaders for cognitive task
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=data_subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")
################################################################        
        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, data_subject_id)
            print('in_chans',in_chans)    
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)
        if args.model =='Brainomni':
            position, sensor_type, valid_electrodes = extract_1020_electrode_positions(ch_names,device)    
            # print(position.shape)           
#############################################################################        
        checkpoint_path = os.path.join(args.recon_results_dir, args.dataset, f'subject_{ckpt_subject_id:02d}','best_model.pth')
        print(f"  Checkpoint: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {checkpoint_path}")
        
        print(f"\n[Step 1] Creating Encoder ({args.model})...")
        if args.model =='labram_base_patch200_200':
            encoder = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches).to(device)
            
            if args.fitune_type =='lora':
                encoder_with_lora = create_lora(encoder,args)
                encoder = load_pretrained_encoder(encoder_with_lora, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
        elif args.model =='CBraMod':
            encoder = CBraMod().to(device)
            if args.fitune_type =='lora':
                encoder = apply_lora_to_model(encoder, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    encoder = apply_lora_to_model(encoder, r=args.cls_lora_r, alpha=args.cls_lora_alpha, dropout=args.cls_lora_dropout)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
        elif args.model =='Brainomni':
            from brainomni.model import BrainOmni
            with open(os.path.join('/mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/models/checkpoint/BrainOmni_tiny', "model_cfg.json")) as f:
                cfg = json.load(f)
            encoder = BrainOmni(**cfg)
            if args.fitune_type =='lora':
                encoder = create_lora(encoder,args)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False    
                                    
        if args.model =='Brainomni':
            model = SimpleClassifier_brainomni(
                encoder=encoder,
                position =position,
                sensortype =sensor_type,
                args = args,
                num_classes=args.num_classes,
            ).to(device)
        else:
            model = SimpleClassifier_multi_model(
                encoder=encoder,
                num_channels= chan_number,
                in_chans = in_chans,
                args = args,  
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)     
                            
        for name, param in model.named_parameters():
            if param.requires_grad:  # 如果参数是可训练的
                print(f"Trainable parameter: {name}")           
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"\n  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None

        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {data_subject_id:02d} - Training (Linear Probing, Frozen Encoder)")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        # Training loop
        for epoch in range(args.epochs):
            # Learning rate scheduling
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Train
            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']

                eeg = eeg.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss, val_acc, val_f1, val_cm = evaluate(
                model, val_loader, criterion, device, args.use_amp
            )

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            # Early stopping 逻辑保持不变...
            if val_f1 > best_val_f1:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # ====================================================================
        # 从这里开始向下替换！
        # ====================================================================
        print("  [Memory] Freeing up GPU memory before final test evaluation...")
        if 'optimizer' in locals():
            del optimizer
        if 'eeg' in locals(): del eeg
        if 'labels' in locals(): del labels
        if 'logits' in locals(): del logits
        if 'loss' in locals(): del loss
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Final test evaluation
        print(f"\n{'='*80}")
        print(f"Data Sub {data_subject_id:02d} | Ckpt Sub {ckpt_subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # [修改] 强制加入 no_grad()，防止 evaluate 内部意外累积计算图导致显存溢出
        with torch.no_grad():
            test_loss, test_acc, test_f1, test_cm = evaluate(
                model, test_loader, criterion, device, args.use_amp
            )

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        # Save metrics
        metrics = {
            'data_subject_id': int(data_subject_id), 
            'ckpt_subject_id': int(ckpt_subject_id), 
            'is_mismatch': bool(is_mismatch),             
            'method': args.cls_fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification'
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        # ================== 收尾阶段的安全清理 ==================
        if 'model' in locals(): del model
        if 'encoder' in locals(): del encoder
        gc.collect()
        torch.cuda.empty_cache()
        # ==============================================================

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Data Sub {data_subject_id:02d} | Ckpt Sub {ckpt_subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals(): del model
        if 'encoder' in locals(): del encoder
        if 'optimizer' in locals(): del optimizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def train_single_subject_cls_new_1(args, subject_id, device, num_patches, dataset):
    """Train transfer learning classifier for single subject with frozen encoder"""
    try:
        valid_fitune_types = ['lora', 'full']
        valid_cls_fitune_types = ['lora', 'full', 'linear']
        if args.fitune_type not in valid_fitune_types:
            raise ValueError(f"fitune_type must be one of {valid_fitune_types}")
        if args.cls_fitune_type not in valid_cls_fitune_types:
            raise ValueError(f"cls_fitune_type must be one of {valid_cls_fitune_types}")
        # Create subject-specific output directory
        output_path = os.path.join(args.output_root, args.dataset,  args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        # Check for checkpoint resumption
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data (Cognitive Task)")
        print(f"{'='*80}")

        
        if args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        else:
            # Create dataloaders for cognitive task
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")
################################################################        
        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)    
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)
        if args.model =='Brainomni':
            position, sensor_type, valid_electrodes = extract_1020_electrode_positions(ch_names,device)    
            # print(position.shape)           
#############################################################################        
        checkpoint_path = os.path.join(args.recon_results_dir, args.dataset, f'subject_{subject_id:02d}','best_model.pth')
        print(f"  Checkpoint: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {checkpoint_path}")
        
        print(f"\n[Step 1] Creating Encoder ({args.model})...")
        if args.model =='labram_base_patch200_200':
            encoder = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches).to(device)
            
            if args.fitune_type =='lora':
                encoder_with_lora = create_lora(encoder,args)
                encoder = load_pretrained_encoder(encoder_with_lora, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
        elif args.model =='CBraMod':
            encoder = CBraMod().to(device)
            if args.fitune_type =='lora':
                encoder = apply_lora_to_model(encoder, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    encoder = apply_lora_to_model(encoder, r=args.cls_lora_r, alpha=args.cls_lora_alpha, dropout=args.cls_lora_dropout)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
        elif args.model =='Brainomni':
            from brainomni.model import BrainOmni
            with open(os.path.join('/mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/models/checkpoint/BrainOmni_tiny', "model_cfg.json")) as f:
                cfg = json.load(f)
            encoder = BrainOmni(**cfg)
            if args.fitune_type =='lora':
                encoder = create_lora(encoder,args)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False    
                                    
        if args.model =='Brainomni':
            model = SimpleClassifier_brainomni(
                encoder=encoder,
                position =position,
                sensortype =sensor_type,
                args = args,
                num_classes=args.num_classes,
            ).to(device)
        else:
            model = SimpleClassifier_multi_model(
                encoder=encoder,
                num_channels= chan_number,
                in_chans = in_chans,
                args = args,  
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)     
                            
        for name, param in model.named_parameters():
            if param.requires_grad:  # 如果参数是可训练的
                print(f"Trainable parameter: {name}")           
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"\n  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None

        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training (Linear Probing, Frozen Encoder)")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        # Training loop
        for epoch in range(args.epochs):
            # Learning rate scheduling
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Train
            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']

                eeg = eeg.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_loss, val_acc, val_f1, val_cm = evaluate(
                model, val_loader, criterion, device, args.use_amp
            )

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            # Early stopping
            if val_f1 > best_val_f1:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Final test evaluation
        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_f1, test_cm = evaluate(
            model, test_loader, criterion, device, args.use_amp
        )

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        # Save metrics (no validation)
        metrics = {
            'subject_id': int(subject_id),
            'method': args.cls_fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': None,
            'best_val_f1': None,
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification'
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        # Memory cleanup
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()


def train_single_subject_cls_new(args, subject_id, device, num_patches, dataset):
    """Train transfer learning classifier for single subject with frozen encoder"""
    try:
        valid_fitune_types = ['lora', 'full']
        valid_cls_fitune_types = ['lora', 'full', 'linear']
        if args.fitune_type not in valid_fitune_types:
            raise ValueError(f"fitune_type must be one of {valid_fitune_types}")
        if args.cls_fitune_type not in valid_cls_fitune_types:
            raise ValueError(f"cls_fitune_type must be one of {valid_cls_fitune_types}")
        # Create subject-specific output directory
        output_path = os.path.join(args.output_root, args.dataset,  args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        # Check for checkpoint resumption
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data (Cognitive Task)")
        print(f"{'='*80}")

        
        if args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        else:
            # Create dataloaders for cognitive task
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")
################################################################        
        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)    
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)
        if args.model =='Brainomni':
            position, sensor_type, valid_electrodes = extract_1020_electrode_positions(ch_names,device)    
            # print(position.shape)           
#############################################################################        
        checkpoint_path = os.path.join(args.recon_results_dir, args.dataset, f'subject_{subject_id:02d}','best_model.pth')
        print(f"  Checkpoint: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {checkpoint_path}")
        
        print(f"\n[Step 1] Creating Encoder ({args.model})...")
        if args.model =='labram_base_patch200_200':
            encoder = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches).to(device)
            
            if args.fitune_type =='lora':
                encoder_with_lora = create_lora(encoder,args)
                encoder = load_pretrained_encoder(encoder_with_lora, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
        elif args.model =='CBraMod':
            encoder = CBraMod().to(device)
            if args.fitune_type =='lora':
                encoder = apply_lora_to_model(encoder, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    encoder = apply_lora_to_model(encoder, r=args.cls_lora_r, alpha=args.cls_lora_alpha, dropout=args.cls_lora_dropout)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
        elif args.model =='Brainomni':
            from brainomni.model import BrainOmni
            with open(os.path.join('/mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/models/checkpoint/BrainOmni_tiny', "model_cfg.json")) as f:
                cfg = json.load(f)
            encoder = BrainOmni(**cfg)
            if args.fitune_type =='lora':
                encoder = create_lora(encoder,args)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False    

        elif args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)
            
            if args.fitune_type =='lora':
                encoder = create_lora(encoder,args)
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        if param.is_floating_point(): 
                            param.requires_grad = True 
                        else:
                            param.requires_grad = False
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        if param.is_floating_point(): 
                            param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False    
                                    
        if args.model =='Brainomni':
            model = SimpleClassifier_brainomni(
                encoder=encoder,
                position =position,
                sensortype =sensor_type,
                args = args,
                num_classes=args.num_classes,
            ).to(device)
        elif args.model =='BIOT':
            model = SimpleClassifier_BIOT(
                encoder=encoder, 
                args=args, 
                ch_names=ch_names, 
                num_classes=args.num_classes
            ).to(device) 
        else:
            model = SimpleClassifier_multi_model(
                encoder=encoder,
                num_channels= chan_number,
                in_chans = in_chans,
                args = args,  
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)     
                            
        for name, param in model.named_parameters():
            if param.requires_grad:  # 如果参数是可训练的
                print(f"Trainable parameter: {name}")           
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"\n  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None

        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training (Linear Probing, Frozen Encoder)")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        # Training loop
        for epoch in range(args.epochs):
            # Learning rate scheduling
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Train
            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']

                eeg = eeg.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_loss, val_acc, val_f1, val_cm = evaluate(
                model, val_loader, criterion, device, args.use_amp
            )

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            # Early stopping
            if val_f1 > best_val_f1:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Final test evaluation
        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_f1, test_cm = evaluate(
            model, test_loader, criterion, device, args.use_amp
        )

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        # Save metrics
        metrics = {
            'subject_id': int(subject_id),
            'method': args.cls_fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification'
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        # Memory cleanup
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None
    

def train_single_subject_cls_lora_merging(args, subject_id, device, num_patches, dataset):
    """Train transfer learning classifier for single subject with frozen encoder"""
    try:
        valid_fitune_types = ['lora', 'full']
        valid_cls_fitune_types = ['lora', 'full', 'linear']
        if args.fitune_type not in valid_fitune_types:
            raise ValueError(f"fitune_type must be one of {valid_fitune_types}")
        if args.cls_fitune_type not in valid_cls_fitune_types:
            raise ValueError(f"cls_fitune_type must be one of {valid_cls_fitune_types}")
        # Create subject-specific output directory
        output_path = os.path.join(args.output_root, args.dataset,  args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        # Check for checkpoint resumption
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data (Cognitive Task)")
        print(f"{'='*80}")

        
        if args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        else:
            # Create dataloaders for cognitive task
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")
################################################################        
        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)    
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)
        if args.model =='Brainomni':
            position, sensor_type, valid_electrodes = extract_1020_electrode_positions(ch_names,device)    
            # print(position.shape)           
#############################################################################        
        checkpoint_path = os.path.join(args.recon_results_dir, args.dataset, f'subject_{subject_id:02d}','best_model.pth')
        print(f"  Checkpoint: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {checkpoint_path}")
        
        print(f"\n[Step 1] Creating Encoder ({args.model})...")
        if args.model =='labram_base_patch200_200':
            encoder = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches).to(device)
            
            if args.fitune_type =='lora':
                encoder_with_lora = create_lora(encoder,args)
                encoder = load_pretrained_encoder(encoder_with_lora, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
        elif args.model =='CBraMod':
            encoder = CBraMod().to(device)
            if args.fitune_type =='lora':
                encoder = apply_lora_to_model(encoder, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    encoder = apply_lora_to_model(encoder, r=args.cls_lora_r, alpha=args.cls_lora_alpha, dropout=args.cls_lora_dropout)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
        elif args.model =='Brainomni':
            from brainomni.model import BrainOmni
            with open(os.path.join('/mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/models/checkpoint/BrainOmni_tiny', "model_cfg.json")) as f:
                cfg = json.load(f)
            encoder = BrainOmni(**cfg)
            if args.fitune_type =='lora':
                encoder = create_lora(encoder,args)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False    

        elif args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)
            
            if args.fitune_type =='lora':
                encoder = create_lora(encoder,args)
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        if param.is_floating_point(): 
                            param.requires_grad = True 
                        else:
                            param.requires_grad = False
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        if param.is_floating_point(): 
                            param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False    
        
        encoder_task_1 = encoder
        
        
                                    
        if args.model =='Brainomni':
            model = SimpleClassifier_brainomni(
                encoder=encoder,
                position =position,
                sensortype =sensor_type,
                args = args,
                num_classes=args.num_classes,
            ).to(device)
        elif args.model =='BIOT':
            model = SimpleClassifier_BIOT(
                encoder=encoder, 
                args=args, 
                ch_names=ch_names, 
                num_classes=args.num_classes,
            ).to(device) 
        else:
            model = SimpleClassifier_multi_model(
                encoder=encoder,
                num_channels= chan_number,
                in_chans = in_chans,
                args = args,  
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)     
                            
        for name, param in model.named_parameters():
            if param.requires_grad:  # 如果参数是可训练的
                print(f"Trainable parameter: {name}")           
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"\n  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None

        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training (Linear Probing, Frozen Encoder)")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        # Training loop
        for epoch in range(args.epochs):
            # Learning rate scheduling
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Train
            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']

                eeg = eeg.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_loss, val_acc, val_f1, val_cm = evaluate(
                model, val_loader, criterion, device, args.use_amp
            )

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            # Early stopping
            if val_f1 > best_val_f1:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Final test evaluation
        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_f1, test_cm = evaluate(
            model, test_loader, criterion, device, args.use_amp
        )

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        # Save metrics
        metrics = {
            'subject_id': int(subject_id),
            'method': args.cls_fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification'
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        # Memory cleanup
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None
    

import torch.nn.utils as utils
def train_single_subject_cls_multi_task_mismatch(args, data_subject_id, ckpt_subject_id, device, num_patches, dataset):
    """Train transfer learning classifier for single subject with frozen encoder"""
    label_mapping = {
        10: 0,
        11: 1,
        12: 2,
        20: 3,
        21: 4,
        30: 5,
        31: 6,
        32: 7,
        40: 8,
        41: 9,
        42: 10
    }
    try:
        valid_fitune_types = ['lora', 'full']
        valid_cls_fitune_types = ['lora', 'full', 'linear']
        if args.fitune_type not in valid_fitune_types:
            raise ValueError(f"fitune_type must be one of {valid_fitune_types}")
        if args.cls_fitune_type not in valid_cls_fitune_types:
            raise ValueError(f"cls_fitune_type must be one of {valid_cls_fitune_types}")
        # Create subject-specific output directory
        is_mismatch = (data_subject_id != ckpt_subject_id)
        folder_suffix = "_mismatch" if is_mismatch else ""
        output_path = os.path.join(args.output_root, args.dataset,  args.cls_fitune_type + folder_suffix)
        
        subject_dir_name = f'subject_data{data_subject_id:02d}_ckpt{ckpt_subject_id:02d}'
        subject_dir = os.path.join(output_path, subject_dir_name)
        os.makedirs(subject_dir, exist_ok=True)

        # Check for checkpoint resumption
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {data_subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {data_subject_id:02d} - Loading Data")
        print(f"{'='*80}")

        if args.multi_task:
            train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
                dataset= dataset,
                target_subject_id=data_subject_id,
                target_task_id=[1,2,3,4],
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        elif args.multitask_fewshot:
            train_loader, val_loader, test_loader= create_multitask_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=data_subject_id,
                target_task_id=[1,2,3,4],
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=data_subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=data_subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )


        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        """
        in chans
        """
        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, data_subject_id)
            print('in_chans',in_chans)    
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)
            
        if args.model =='Brainomni':
            position, sensor_type, valid_electrodes = extract_1020_electrode_positions(ch_names,device)   
        
        checkpoint_path = os.path.join(args.recon_results_dir, args.dataset, f'subject_{ckpt_subject_id:02d}','best_model.pth')
        print(f"  Checkpoint: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {checkpoint_path}")
        
        print(f"\n[Step 1] Creating Encoder ({args.model})...")

        """LaBraM"""
        if args.model =='labram_base_patch200_200':
            encoder = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches).to(device)
            
            if args.fitune_type =='lora':
                encoder_with_lora = create_lora(encoder,args)
                encoder = load_pretrained_encoder(encoder_with_lora, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

            """cBraMod"""           
        elif args.model =='CBraMod':
            encoder = CBraMod().to(device)
            if args.fitune_type =='lora':
                encoder = apply_lora_to_model(encoder, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    encoder = apply_lora_to_model(encoder, r=args.cls_lora_r, alpha=args.cls_lora_alpha, dropout=args.cls_lora_dropout)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

            """Brainomni"""
        elif args.model =='Brainomni':
            from brainomni.model import BrainOmni
            with open(os.path.join('/mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/models/checkpoint/BrainOmni_tiny', "model_cfg.json")) as f:
                cfg = json.load(f)
            encoder = BrainOmni(**cfg)
            if args.fitune_type =='lora':
                encoder = create_lora(encoder,args)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False    

            """BIOT"""
        elif args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)
            
            if args.fitune_type =='lora':
                encoder = create_lora(encoder,args)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        if param.is_floating_point(): 
                            param.requires_grad = True 
                        else:
                            param.requires_grad = False
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    if args.model =='BIOT':
                        for name, param in encoder.named_parameters():
                            if param.is_floating_point(): 
                                param.requires_grad = True 
                    else:
                        for name, param in encoder.named_parameters():
                            param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False    
                                    
        if args.model =='Brainomni':
            model = SimpleClassifier_brainomni(
                encoder=encoder,
                position =position,
                sensortype =sensor_type,
                args = args,
                num_classes=args.num_classes,
            ).to(device)
        elif args.model =='BIOT':
            model = SimpleClassifier_BIOT(
                encoder=encoder, 
                args=args, 
                ch_names=ch_names, 
                num_classes=args.num_classes,
            ).to(device) 
        else:
            model = SimpleClassifier_multi_model(
                encoder=encoder,
                num_channels= chan_number,
                in_chans = in_chans,
                args = args,  
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)     

        # for name, param in model.named_parameters():
        #     # if param.requires_grad:  # 如果参数是可训练的
        #     #     print(f"Trainable parameter: {name}")           
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        # print(f"\n  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None

        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {data_subject_id:02d} - Training (Linear Probing, Frozen Encoder)")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        # Training loop
        for epoch in range(args.epochs):
            # Learning rate scheduling
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Train
            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']
                    mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])

                eeg = eeg.to(device)
                mapped_labels = mapped_labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, mapped_labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, mapped_labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_loss, val_acc, val_f1, val_cm = evaluate_multitask(
                model, val_loader, criterion, device, label_mapping, args.use_amp
            )

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Final test evaluation
        print(f"\n{'='*80}")
        print(f"Subject {data_subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_f1, test_cm = evaluate_multitask(
            model, test_loader, criterion, device,label_mapping, args.use_amp
        )

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        # Save metrics
        metrics = {
            'data_subject_id': int(data_subject_id), # 记录数据来源
            'ckpt_subject_id': int(ckpt_subject_id), # 记录权重来源
            'is_mismatch': bool(is_mismatch),        # 错配对照组标记
            'method': args.cls_fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification'
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        # Memory cleanup
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {data_subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None


import torch.nn.utils as utils
def train_single_subject_cls_multi_task(args, subject_id, device, num_patches, dataset):
    """Train transfer learning classifier for single subject with frozen encoder"""
    label_mapping = {
        10: 0,
        11: 1,
        12: 2,
        20: 3,
        21: 4,
        30: 5,
        31: 6,
        32: 7,
        40: 8,
        41: 9,
        42: 10
    }
    try:
        valid_fitune_types = ['lora', 'full']
        valid_cls_fitune_types = ['lora', 'full', 'linear']
        if args.fitune_type not in valid_fitune_types:
            raise ValueError(f"fitune_type must be one of {valid_fitune_types}")
        if args.cls_fitune_type not in valid_cls_fitune_types:
            raise ValueError(f"cls_fitune_type must be one of {valid_cls_fitune_types}")
        # Create subject-specific output directory
        output_path = os.path.join(args.output_root, args.dataset,  args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        # Check for checkpoint resumption
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")

        if args.multi_task:
            train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
                dataset= dataset,
                target_subject_id=subject_id,
                target_task_id=[1,2,3,4],
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        elif args.multitask_fewshot:
            train_loader, val_loader, test_loader= create_multitask_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                target_task_id=[1,2,3,4],
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )


        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        """
        in chans
        """
        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)    
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)
            
        if args.model =='Brainomni':
            position, sensor_type, valid_electrodes = extract_1020_electrode_positions(ch_names,device)   
        
        checkpoint_path = os.path.join(args.recon_results_dir, args.dataset, f'subject_{subject_id:02d}','best_model.pth')
        print(f"  Checkpoint: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {checkpoint_path}")
        
        print(f"\n[Step 1] Creating Encoder ({args.model})...")

        """LaBraM"""
        if args.model =='labram_base_patch200_200':
            encoder = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches).to(device)
            
            if args.fitune_type =='lora':
                encoder_with_lora = create_lora(encoder,args)
                encoder = load_pretrained_encoder(encoder_with_lora, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

            """cBraMod"""           
        elif args.model =='CBraMod':
            encoder = CBraMod().to(device)
            if args.fitune_type =='lora':
                encoder = apply_lora_to_model(encoder, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    encoder = apply_lora_to_model(encoder, r=args.cls_lora_r, alpha=args.cls_lora_alpha, dropout=args.cls_lora_dropout)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False


            """Brainomni"""
        elif args.model =='Brainomni':
            from brainomni.model import BrainOmni
            with open(os.path.join('/mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/models/checkpoint/BrainOmni_tiny', "model_cfg.json")) as f:
                cfg = json.load(f)
            encoder = BrainOmni(**cfg)
            if args.fitune_type =='lora':
                encoder = create_lora(encoder,args)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False    

            """BIOT"""
        elif args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)
            
            if args.fitune_type =='lora':
                encoder = create_lora(encoder,args)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        if param.is_floating_point(): 
                            param.requires_grad = True 
                        else:
                            param.requires_grad = False
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    if args.model =='BIOT':
                        for name, param in encoder.named_parameters():
                            if param.is_floating_point(): 
                                param.requires_grad = True 
                    else:
                        for name, param in encoder.named_parameters():
                            param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False    
                                    
        if args.model =='Brainomni':
            model = SimpleClassifier_brainomni(
                encoder=encoder,
                position =position,
                sensortype =sensor_type,
                args = args,
                num_classes=args.num_classes,
            ).to(device)
        elif args.model =='BIOT':
            model = SimpleClassifier_BIOT(
                encoder=encoder, 
                args=args, 
                ch_names=ch_names, 
                num_classes=args.num_classes,
            ).to(device) 
        else:
            model = SimpleClassifier_multi_model(
                encoder=encoder,
                num_channels= chan_number,
                in_chans = in_chans,
                args = args,  
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)     

        # for name, param in model.named_parameters():
        #     # if param.requires_grad:  # 如果参数是可训练的
        #     #     print(f"Trainable parameter: {name}")           
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        # print(f"\n  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None

        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training (Linear Probing, Frozen Encoder)")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        # Training loop
        for epoch in range(args.epochs):
            # Learning rate scheduling
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Train
            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']
                    mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])

                eeg = eeg.to(device)
                mapped_labels = mapped_labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, mapped_labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, mapped_labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_loss, val_acc, val_f1, val_cm = evaluate_multitask(
                model, val_loader, criterion, device, label_mapping, args.use_amp
            )

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Final test evaluation
        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_f1, test_cm = evaluate_multitask(
            model, test_loader, criterion, device,label_mapping, args.use_amp
        )

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        # Save metrics
        metrics = {
            'subject_id': int(subject_id),
            'method': args.cls_fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification'
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        # Memory cleanup
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None
    
        

def train_single_subject_cls_old(args, subject_id, device, num_patches, dataset):
    """Train transfer learning classifier for single subject with frozen encoder"""
    try:
        # Create subject-specific output directory
        output_path = os.path.join(args.output_root, args.dataset, args.recon_type,args.mask_type, args.fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        # Check for checkpoint resumption
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data (Cognitive Task)")
        print(f"{'='*80}")
        
        if args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        else:
            # Create dataloaders for cognitive task
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        # # Create dataloaders for cognitive task
        # train_loader, val_loader, test_loader = create_single_subject_dataloaders(
        #     dataset=dataset,
        #     target_subject_id=subject_id,
        #     train_size =args.train_size,
        #     test_size=args.test_size,
        #     val_size=args.val_size,
        #     batch_size=args.batch_size,
        #     random_state=args.seed,
        #     num_workers=args.num_workers
        # )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")
        
        in_chans, ch_names, chan_number = get_dataset_name(args)
        print('in_chans',in_chans)

        # Create encoder
        print(f"\n[Step 1] Creating Encoder ({args.model})...")
        encoder = create_model(
            args.model,
            pretrained=False,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            qkv_bias=args.qkv_bias,
            num_patches_per_channel_input=num_patches,
        )
        
        residual_net = ResEEGNet(
            num_channels=chan_number,
            num_patches=num_patches,
            patch_size=200,
            embed_dim=args.embed_dim,
            hidden_dim=args.residual_hidden_dim,
            dropout=args.residual_dropout
        )
        
        Encoderwith_Residual=EncoderWithResidual(
            encoder = encoder,
            residual_net = residual_net,
            in_chans = in_chans,
            embed_dim=args.embed_dim
        )
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q", "k", "v", "proj"])
        
        # 在train.py中，调用get_peft_model之前，添加以下代码：
        # print("Encoder module names:")
        # for name, _ in encoder.named_modules():
        #     print(name)
        
        
        # Load pretrained encoder from reconstruction task
        print(f"\n[Step 2] Loading Pretrained Encoder ({args.fitune_type})...")
        # checkpoint_path = get_checkpoint_path(args.fitune_type, subject_id, args.recon_results_dir, args.dataset)
        # f'subject_{subject_id:02d}',
        # 'best_model.pth'
        checkpoint_path = os.path.join(args.recon_results_dir, args.fitune_type, f'subject_{subject_id:02d}','best_model.pth')
        print(f"  Checkpoint: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {checkpoint_path}")

        if args.fitune_type=='resnet':
            res_encoder = load_pretrained_encoder(Encoderwith_Residual, checkpoint_path, args.fitune_type, device)
            model = SimpleFinetuneClassifier_resnet(
                encoder=res_encoder,
                encoder_free=args.encoder_free,
                num_channels= chan_number,
                in_chans = in_chans,
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)
            
        elif args.fitune_type =='lora':
            encoder_with_lora = get_peft_model(encoder, lora_config)
            encoder = load_pretrained_encoder(encoder_with_lora, checkpoint_path, args.fitune_type, device)
            model = SimpleFinetuneClassifier(
                encoder=encoder,
                encoder_free=args.encoder_free,
                num_channels= chan_number,
                in_chans = in_chans,
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)
        else:
            encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
            model = SimpleFinetuneClassifier(
                encoder=encoder,
                encoder_free=args.encoder_free,
                num_channels= chan_number,
                in_chans = in_chans,
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)
                
        # Setup training (only classification head parameters)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"\n  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None

        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training (Linear Probing, Frozen Encoder)")
        print(f"Method: {args.fitune_type.upper()}")
        print(f"{'='*80}")

        # Training loop
        for epoch in range(args.epochs):
            # Learning rate scheduling
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Train
            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']

                eeg = eeg.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_loss, val_acc, val_f1, val_cm = evaluate(
                model, val_loader, criterion, device, args.use_amp
            )

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'classifier_state_dict': model.classifier.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Final test evaluation
        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_f1, test_cm = evaluate(
            model, test_loader, criterion, device, args.use_amp
        )

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        # Save metrics
        metrics = {
            'subject_id': int(subject_id),
            'method': args.fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification'
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        # Memory cleanup
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None



def train_single_subject_baseline_cls(args, subject_id, device, num_patches, dataset):
    """Train transfer learning classifier for single subject with frozen encoder"""
    try:
        # Create subject-specific output directory
        output_path = os.path.join(args.output_root, args.dataset, args.fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        # Check for checkpoint resumption
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data (Cognitive Task)")
        print(f"{'='*80}")

        # Create dataloaders for cognitive task
        if args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        else:
            # Create dataloaders for cognitive task
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
            
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")
        
        in_chans, ch_names, chan_number = get_dataset_name(args)
        print('in_chans',in_chans)

        # Create encoder
        print(f"\n[Step 1] Creating Encoder ({args.model})...")
        encoder = create_model(
            args.model,
            pretrained=False,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            qkv_bias=args.qkv_bias,
            num_patches_per_channel_input=num_patches,
        )

        load_vanilla_labram_weights(
            encoder, 
            args.pretrained_path, 
            model_key='model', # 对应 labram-base.pth 内部的键
            model_prefix=''    # 对应前缀过滤
        )
        
        # lora_config = LoraConfig(
        #     r=args.lora_r,
        #     lora_alpha=args.lora_alpha,
        #     lora_dropout=args.lora_dropout,
        #     target_modules=args.target_modules,  # Targets Q/K/V/Projection layers
        # )
        
        # [Step 2.5] 根据方法进行包装 (LoRA 或 ResNet)
        if  args.fitune_type == 'lora':
            # 先加载了原始权重，再进行 LoRA 包装
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.target_modules,
            )
            encoder_with_lora = get_peft_model(encoder, lora_config)
            model = SimpleFinetuneClassifier(
                encoder=encoder_with_lora,
                encoder_free=args.encoder_free,
                num_channels=chan_number,
                in_chans=in_chans,
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        elif args.fitune_type == 'sgd':
            model = SimpleFinetuneClassifier(
                encoder=encoder,
                encoder_free=args.encoder_free,
                num_channels=chan_number,
                in_chans=in_chans,
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)
            
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            args.lr=args.sgd_lr
            args.min_lr =args.min_sgd_lr
            optimizer = torch.optim.SGD(
                trainable_params,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
        # 打印模型的所有子模块，以检查模块名称是否匹配
            # for name, module in encoder.named_modules():
            #     print(name)

        else: 
            model = SimpleFinetuneClassifier(
                encoder=encoder,
                encoder_free=args.encoder_free,
                num_channels=chan_number,
                in_chans=in_chans,
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)
            
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=args.lr,
                weight_decay=args.weight_decay
            )                
        # Setup training (only classification head parameters)
        # trainable_params = [p for p in model.parameters() if p.requires_grad]
        # print(f"\n  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None

        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training (Linear Probing, Frozen Encoder)")
        print(f"Method: {args.fitune_type.upper()}")
        print(f"{'='*80}")

        # Training loop
        for epoch in range(args.epochs):
            # Learning rate scheduling
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Train
            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']

                eeg = eeg.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_loss, val_acc, val_f1, val_cm = evaluate(
                model, val_loader, criterion, device, args.use_amp
            )

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'classifier_state_dict': model.classifier.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Final test evaluation
        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_f1, test_cm = evaluate(
            model, test_loader, criterion, device, args.use_amp
        )

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        # Save metrics
        metrics = {
            'subject_id': int(subject_id),
            'method': args.fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification'
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        # Memory cleanup
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None
    
    
def train_single_subject_baseline_cls_new(args, subject_id, device, num_patches, dataset):
    """Train transfer learning classifier for single subject with frozen encoder"""
    try:
        # Create subject-specific output directory
        output_path = os.path.join(args.output_root, args.dataset, args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        # Check for checkpoint resumption
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data (Cognitive Task)")
        print(f"{'='*80}")

        # Create dataloaders for cognitive task
        if args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        else:
            # Create dataloaders for cognitive task
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
            
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")
        
########################### in chans #################################            
        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)    
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)
            
        if args.model =='Brainomni':
            position, sensor_type, valid_electrodes = extract_1020_electrode_positions(ch_names,device)    
            # print(position.shape)    
            
########################### in chans ################################# 
        print(f"\n[Step 1] Creating Encoder ({args.model})...")

        # Create encoder
        if args.model == 'labram_base_patch200_200':
       
            encoder = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches,
            )

            load_vanilla_labram_weights(
                encoder, 
                args.pretrained_path, 
                model_key='model', # 对应 labram-base.pth 内部的键
                model_prefix=''    # 对应前缀过滤
            )
            
        elif args.model == 'CBraMod':
            encoder=CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
            )
            map_location = torch.device(args.device)
            encoder.load_state_dict(torch.load(args.pretrained_path, map_location=map_location,weights_only=False))
            
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
            encoder=get_brainomni_model(args.pretrained_path).to(device)
            
        elif args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)
            encoder.load_state_dict(torch.load(args.pretrained_path, map_location=device, weights_only=False))    
            
        if  args.cls_fitune_type == 'lora':
            if args.model == 'CBraMod':
                encoder = apply_lora_to_model(encoder, r=args.cls_lora_r, alpha=args.cls_lora_alpha, dropout=args.cls_lora_dropout)
            else:
                encoder = create_lora_cls(encoder,args)
            for name, param in encoder.named_parameters():
                if 'lora' not in name:  # 只冻结非LoRA参数
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            # trainable = [p for p in encoder.parameters() if p.requires_grad]
            # total = sum(p.numel() for p in encoder.parameters())
            # trainable_count = sum(p.numel() for p in trainable)
            # print(f"trainable: {trainable_count:,} / {total:,}")

        elif args.cls_fitune_type == 'full':
            if args.model =='BIOT':
                for name, param in encoder.named_parameters():
                    if param.is_floating_point():  # 只设置浮点数的梯度
                        param.requires_grad = True
            else:
                for name, param in encoder.named_parameters():
                    param.requires_grad = True 
                    
        elif args.cls_fitune_type == 'linear':
            for name, param in encoder.named_parameters():
                param.requires_grad = False
                
        if args.model =='Brainomni':
            model = SimpleClassifier_brainomni(
                encoder=encoder,
                position =position,
                sensortype =sensor_type,
                args = args,
                num_classes=args.num_classes,
            ).to(device)
        elif args.model=='BIOT':
            model = SimpleClassifier_BIOT(
                encoder=encoder, 
                args=args, 
                ch_names=ch_names, 
                num_classes=args.num_classes,
            ).to(device) 
        else:
            model = SimpleClassifier(
                encoder=encoder,
                num_channels=chan_number,
                in_chans=in_chans,
                args=args,
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)
        trainable = [p for p in encoder.parameters() if p.requires_grad]
        total = sum(p.numel() for p in encoder.parameters())
        trainable_count = sum(p.numel() for p in trainable)
        print(f"trainable: {trainable_count:,} / {total:,}")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )                     
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None
        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training (Linear Probing, Frozen Encoder)")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        # Training loop
        for epoch in range(args.epochs):
            # Learning rate scheduling
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Train
            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']

                eeg = eeg.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            
            train_loss, train_acc, train_f1,  train_cm = evaluate(
                model, train_loader, criterion, device, args.use_amp
            )
            # Validation
            val_loss, val_acc, val_f1, val_cm = evaluate(
                model, val_loader, criterion, device, args.use_amp
            )
            if epoch ==0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                }, os.path.join(subject_dir, 'original_model.pth'))

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"Train_acc={train_acc:.4f}",
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'classifier_state_dict': model.classifier.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Final test evaluation
        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_f1, test_cm = evaluate(
            model, test_loader, criterion, device, args.use_amp
        )

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        # Save metrics
        metrics = {
            'subject_id': int(subject_id),
            'method': args.cls_fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification'
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        # Memory cleanup
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None    
    
    
##############################################################################################################################
def train_Contrastiv_for_subject(args, subject_id, device, num_patches, dataset):
    """
    Train full fine-tuning reconstruction model for a single subject (V2-Fixed)

    Args:
        args: Command line arguments
        subject_id: Target subject ID
        device: CUDA device
        num_patches: Number of patches (A) for current dataset
        dataset: Pre-loaded EEGDataset instance

    Returns:
        dict: Results dictionary with metrics, or None if failed
    """
    try:
        # Create subject-specific output directory
        subject_dir = os.path.join(args.output_root, args.dataset, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")
        
        # Load data
        train_loader, val_loader, test_loader = create_contrastive_dataloaders(
            dataset=dataset,
            target_subject_id=subject_id,
            train_size=args.train_size,
            test_size=args.test_size,
            val_size=args.val_size,
            batch_size=args.batch_size,
            random_state=args.seed,
            num_workers=args.num_workers
        )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # Check minimum samples
        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        if test_size == 0:
            print("Warning: Test set is empty! Skipping test evaluation.")
            test_metrics = None  # Set test_metrics to None or some default value if needed.
        else:
            test_metrics = True  # Flag indicating that test set is not empty

        in_chans, ch_names, chan_number = get_dataset_name(args)
        print('in_chans', in_chans)
        
        encoder_sub = create_model(
            args.model,
            pretrained=False,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            qkv_bias=args.qkv_bias,
            num_patches_per_channel_input=num_patches,
        )   
        load_vanilla_labram_weights(encoder_sub, args.pretrained_path, args.model_key, args.model_prefix)
        
        
        
        # lora_config = LoraConfig(
        #     r=args.lora_r,
        #     lora_alpha=args.lora_alpha,
        #     lora_dropout=args.lora_dropout,
        #     target_modules=args.target_modules,
        # )
        # encoder_sub = get_peft_model(encoder_sub, lora_config)
        encoder_sub = encoder_sub.to(device)

        # Freeze all parameters except LoRA layers
        # for name, module in encoder_sub.named_modules():
        #     if isinstance(module, nn.Module):
        #         for param in module.parameters():
        #             param.requires_grad = False

        # for name, module in encoder_sub.named_modules():
        #     if 'lora' in name:  # Only unfreeze LoRA layers
        #         for param in module.parameters():
        #             param.requires_grad = True
        # 正确做法：只冻结base_model，不解冻LoRA
        # for name, param in encoder_sub.named_parameters():
        #     if 'lora' not in name:  # 只冻结非LoRA参数
        #         param.requires_grad = False
        #     else:
        #         param.requires_grad = True
        # Set up optimizer for only trainable parameters (LoRA layers)
        for name, param in encoder_sub.named_parameters():
            if param.requires_grad:
                print(f"可训练: {name}")
        
        for name, param in encoder_sub.named_parameters():
            param.requires_grad = True
        # Parameter details
        # total_params, trainable_params, trainable_percentage, trainable_modules = calculate_parameter_percentage_with_names(encoder_group)
        # print(f"encoder_group Total parameters: {total_params}")
        # print(f"encoder_group Trainable parameters: {trainable_params}")
        # print(f"encoder_group Trainable parameters percentage: {trainable_percentage:.2f}%")
        
        total_params, trainable_params, trainable_percentage, trainable_modules = calculate_parameter_percentage_with_names(encoder_sub)
        print(f"encoder_sub Total parameters: {total_params}")
        print(f"encoder_sub Trainable parameters: {trainable_params}")
        print(f"encoder_sub Trainable parameters percentage: {trainable_percentage:.2f}%")
        # for name, param in encoder_sub.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        # for name, param in encoder_sub.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         print(f"{name} 有梯度")
        #     elif param.requires_grad and param.grad is None:
        #         print(f"警告: {name} 可训练但梯度为None")

        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder_sub.parameters()), lr=args.lr)
        trainable_params = [p for p in encoder_sub.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training ({args.fitune_type})")
        print(f"{'='*80}")
        # for name, param in encoder_sub.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
            
        # Training loop
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            encoder_sub.train()
            # encoder_group.eval()
            train_loss = 0

            for data, contrastive_data, label, contrastive_label in train_loader:
                if not data.requires_grad:
                    data = data.requires_grad_(True)
                if not contrastive_data.requires_grad:
                    contrastive_data = contrastive_data.requires_grad_(True)

                B, C, A, T = data.shape
                optimizer.zero_grad()
                data = data.to(device)
                contrastive_data = contrastive_data.to(device)
                       
                # Forward pass
                # x_patch = encoder_sub.debug_patch_embed_gradient(data)
                embedding_sub = encoder_sub.forward_features(data, input_chans=in_chans)
                embedding_group = encoder_sub.forward_features(contrastive_data, input_chans=in_chans)
                # print(f"embedding_sub.requires_grad: {embedding_sub.requires_grad}")
                # print(f"embedding_group.requires_grad: {embedding_group.requires_grad}")

                # Compute InfoNCE loss
                loss = info_nce_loss(embedding_sub, embedding_group, temperature=0.1)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation loss
            encoder_sub.eval()
            # encoder_group.eval()
            val_loss = 0
            with torch.no_grad():
                for val_data, val_contrastive_data,label, contrastive_label in val_loader:
                    val_data = val_data.to(device)
                    val_contrastive_data = val_contrastive_data.to(device)
                    val_embedding_sub = encoder_sub.forward_features(val_data, input_chans=in_chans)
                    val_embedding_group = encoder_sub.forward_features(val_contrastive_data, input_chans=in_chans)
                    val_loss += info_nce_loss(val_embedding_sub, val_embedding_group, temperature=0.1).item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")

            # Checkpoint saving and early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder_sub.state_dict(),
                    'val_loss': avg_val_loss,
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                


        losses = {
        'train_losses': train_losses,
        'val_losses': val_losses}
        # Final test evaluation if test set is available
        if test_metrics is not None:  # Check if test set is not empty
            print(f"\n{'='*80}")
            print(f"Subject {subject_id:02d} - Final Test Evaluation")
            print(f"{'='*80}")

            checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'))
            encoder_sub.load_state_dict(checkpoint['encoder_state_dict'])

            # Test metrics
            test_loss = 0
            for test_data, test_contrastive_data,label, contrastive_label in test_loader:
                test_data = test_data.to(device)
                test_contrastive_data = test_contrastive_data.to(device)                
                test_embedding_sub = encoder_sub.forward_features(test_data, input_chans=in_chans)
                test_embedding_group = encoder_sub.forward_features(test_contrastive_data, input_chans=in_chans)
                test_loss += info_nce_loss(test_embedding_sub, test_embedding_group, temperature=0.1).item()

            avg_test_loss = test_loss / len(test_loader)

            # Save metrics
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss':avg_train_loss,
                'val_size': val_size,
                'val_loss':avg_val_loss,
                'test_size': test_size,
                'test_loss': avg_test_loss,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        else:
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss':avg_train_loss,
                'val_size': val_size,
                'val_loss':avg_val_loss,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        # Cleanup
        torch.cuda.empty_cache()

        return metrics, losses

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'encoder_sub' in locals():
            del encoder_sub
        torch.cuda.empty_cache()

        return None, None



##############################################################################################################################
def train_for_subject_embedding(args, subject_id, device, num_patches, dataset, encoder_sub,encoder_group):
    """
    Train full fine-tuning reconstruction model for a single subject (V2-Fixed)

    Args:
        args: Command line arguments
        subject_id: Target subject ID
        device: CUDA device
        num_patches: Number of patches (A) for current dataset
        dataset: Pre-loaded EEGDataset instance

    Returns:
        dict: Results dictionary with metrics, or None if failed
    """
    try:
        # Create subject-specific output directory
        subject_dir = os.path.join(args.output_root, args.dataset, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")
        
        if args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        else:
            # Create dataloaders for cognitive task
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # Check minimum samples
        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        if test_size == 0:
            print("Warning: Test set is empty! Skipping test evaluation.")
            test_metrics = None  # Set test_metrics to None or some default value if needed.
        else:
            test_metrics = True  # Flag indicating that test set is not empty

        in_chans, ch_names, chan_number = get_dataset_name(args)
        print('in_chans', in_chans)
        
        # load_vanilla_labram_weights(encoder_group, args.pretrained_path, args.model_key, args.model_prefix)
        encoder_group = encoder_group.to(device)
        for name, param in encoder_group.named_parameters():
            param.requires_grad = False
        
        # encoder_sub = create_model(
        #     args.model,
        #     pretrained=False,
        #     use_mean_pooling=args.use_mean_pooling,
        #     init_scale=args.init_scale,
        #     use_rel_pos_bias=args.rel_pos_bias,
        #     use_abs_pos_emb=args.abs_pos_emb,
        #     init_values=args.layer_scale_init_value,
        #     qkv_bias=args.qkv_bias,
        #     num_patches_per_channel_input=num_patches,
        # )   
        

        # lora_config = LoraConfig(
        #     r=args.lora_r,
        #     lora_alpha=args.lora_alpha,
        #     lora_dropout=args.lora_dropout,
        #     target_modules=args.target_modules,
        # )
        # encoder_sub = get_peft_model(encoder_sub, lora_config)
        load_vanilla_labram_weights(encoder_sub, args.pretrained_path, args.model_key, args.model_prefix)
        encoder_sub = encoder_sub.to(device)
        for name, param in encoder_sub.named_parameters():
            param.requires_grad = True

        # 正确做法：只冻结base_model，不解冻LoRA
        # for name, param in encoder_sub.named_parameters():
        #     if 'lora' not in name:  # 只冻结非LoRA参数
        #         param.requires_grad = False
        #     else:
        #         param.requires_grad = True
        # Set up optimizer for only trainable parameters (LoRA layers)
        for name, param in encoder_sub.named_parameters():
            if param.requires_grad:
                print(f"group_sub 可训练: {name}")
                
        for name, param in encoder_group.named_parameters():
            if param.requires_grad:
                print(f"group_encoder 可训练: {name}")
        

        # Parameter details
        # total_params, trainable_params, trainable_percentage, trainable_modules = calculate_parameter_percentage_with_names(encoder_group)
        # print(f"encoder_group Total parameters: {total_params}")
        # print(f"encoder_group Trainable parameters: {trainable_params}")
        # print(f"encoder_group Trainable parameters percentage: {trainable_percentage:.2f}%")
        
        total_params, trainable_params, trainable_percentage, trainable_modules = calculate_parameter_percentage_with_names(encoder_sub)
        print(f"encoder_sub Total parameters: {total_params}")
        print(f"encoder_sub Trainable parameters: {trainable_params}")
        print(f"encoder_sub Trainable parameters percentage: {trainable_percentage:.2f}%")
        # for name, param in encoder_sub.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        # for name, param in encoder_sub.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         print(f"{name} 有梯度")
        #     elif param.requires_grad and param.grad is None:
        #         print(f"警告: {name} 可训练但梯度为None")

        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder_sub.parameters()), lr=args.lr)
        trainable_params = [p for p in encoder_sub.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
      

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training ({args.fitune_type})")
        print(f"{'='*80}")
        # for name, param in encoder_sub.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        masking_strategy = Embedding_MaskingStrategy(mask_ratio=args.mask_ratio, mask_type=args.mask_type)    
        # Training loop
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            encoder_sub.train()
            encoder_group.eval()
            train_loss = 0
            train_loss_intra =0
            train_loss_cross =0

            for batch in train_loader:
                # if not data.requires_grad:
                #     data = data.requires_grad_(True)
                # if not contrastive_data.requires_grad:
                #     contrastive_data = contrastive_data.requires_grad_(True)
                optimizer.zero_grad()
                eeg = batch['eeg'].to(device)
                B, C, A, T = eeg.shape
                   # Forward pass
                # x_patch = encoder_sub.debug_patch_embed_gradient(data)
                if args.embedding_high_level:
                    mask_eeg,mask = masking_strategy(eeg)
                    embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                    embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                    loss, loss_intra, loss_cross = info_embedding_loss_4d_channel(embedding_mask, embedding_truth, temperature=0.1, batch_size= B, channel=C, patch=A, dimension=T)
                else:
                    mask_eeg,mask = masking_strategy(eeg)
                    embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                    embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans)
                    loss, loss_intra, loss_cross = info_embedding_loss(embedding_mask, embedding_truth, temperature=0.1)
                # high level feature from two encoders

                # print(f"embedding_sub.requires_grad: {embedding_sub.requires_grad}")
                # print(f"embedding_group.requires_grad: {embedding_group.requires_grad}")

                # Compute InfoNCE loss
                
                loss.backward()
                optimizer.step()

                # Track the different losses separately
                train_loss += loss.item()
                train_loss_intra += loss_intra.item()
                train_loss_cross += loss_cross.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_train_loss_intra = train_loss_intra / len(train_loader)
            avg_train_loss_cross = train_loss_cross / len(train_loader)
            
            train_losses.append(avg_train_loss)

            # # Validation loss
            # encoder_sub.eval()
            # encoder_group.eval()
            # val_loss = 0
            # with torch.no_grad():
            #     for val_data, val_contrastive_data,label, contrastive_label in val_loader:
            #         val_data = val_data.to(device)
            #         val_contrastive_data = val_contrastive_data.to(device)
            #         val_embedding_sub = encoder_sub.forward_features(val_data, input_chans=in_chans)
            #         val_embedding_group = encoder_group.forward_features(val_contrastive_data, input_chans=in_chans)
            #         val_loss += info_nce_loss_loop(val_embedding_sub, val_embedding_group, temperature=0.1).item()

            # avg_val_loss = val_loss / len(val_loader)
            # val_losses.append(avg_val_loss)
            # Validation loss
            # Validation loss
            encoder_sub.eval()
            encoder_group.eval()
            val_loss = 0
            val_loss_intra = 0
            val_loss_cross = 0
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch['eeg'].to(device)  # 获取数据
                    B, C, A, T = eeg.shape  # 添加形状提取
                    if args.embedding_high_level:
                        mask_eeg,mask = masking_strategy(eeg)
                        val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans, return_patch_tokens=True)
                        val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans, return_patch_tokens=True)
                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = info_embedding_loss_4d_channel(val_embedding_sub, val_embedding_group, temperature=0.7, batch_size=B, channel=C, patch=A, dimension=T)
                    else:
                        mask_eeg,mask = masking_strategy(eeg)
                        val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                        val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = info_embedding_loss(val_embedding_sub, val_embedding_group, temperature=0.7)
                    
                    val_loss += val_loss_curr.item()
                    val_loss_intra += val_loss_intra_curr.item()
                    val_loss_cross += val_loss_cross_curr.item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_loss_intra = val_loss_intra / len(val_loader)
            avg_val_loss_cross = val_loss_cross / len(val_loader)

            val_losses.append(avg_val_loss)

            # if (epoch + 1) % 10 == 0 or epoch == 0:
            #     print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, "
                    f"Loss MSE={avg_train_loss_intra:.4f}, Loss Cross={avg_train_loss_cross:.4f}, "
                    f"Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")

            # Checkpoint saving and early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder_sub.state_dict(),
                    'val_loss': avg_val_loss,
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        losses = {
        'train_losses': train_losses,
        'val_losses': val_losses}
        # Final test evaluation if test set is available
        if test_metrics is not None:  # Check if test set is not empty
            print(f"\n{'='*80}")
            print(f"Subject {subject_id:02d} - Final Test Evaluation")
            print(f"{'='*80}")

            checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'))
            encoder_sub.load_state_dict(checkpoint['encoder_state_dict'])

            # Test metrics
            test_loss = 0
            test_loss_intra = 0
            test_loss_cross = 0
            encoder_sub.eval()
            encoder_group.eval()
            with torch.no_grad():
                for batch in test_loader:
                    eeg = batch['eeg'].to(device)  # 获取数据
                    B, C, A, T = eeg.shape  # 添加形状提取
                    
                    if args.embedding_high_level:
                        mask_eeg,mask = masking_strategy(eeg)
                        test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans, return_patch_tokens=True)
                        test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans, return_patch_tokens=True)
                        test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = info_embedding_loss_4d_channel(test_embedding_sub, test_embedding_group, temperature=0.1, batch_size=B, channel=C, patch=A, dimension=T)
                    else:
                        mask_eeg,mask = masking_strategy(eeg)
                        test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                        test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = info_embedding_loss(test_embedding_sub, test_embedding_group, temperature=0.1)
                    
                    test_loss += test_loss_curr.item()
                    test_loss_intra += test_loss_intra_curr.item()
                    test_loss_cross += test_loss_cross_curr.item()

            avg_test_loss = test_loss / len(test_loader)
            avg_test_loss_intra = test_loss_intra / len(test_loader)
            avg_test_loss_cross = test_loss_cross / len(test_loader)

            # Save metrics
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'train_loss_intra': avg_train_loss_intra,
                'train_loss_cross': avg_train_loss_cross,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'val_loss_intra': avg_val_loss_intra,
                'val_loss_cross': avg_val_loss_cross,
                'test_size': test_size,
                'test_loss': avg_test_loss,
                'test_loss_intra': avg_test_loss_intra,
                'test_loss_cross': avg_test_loss_cross,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        else:
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        # Cleanup
        torch.cuda.empty_cache()

        return metrics, losses

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'encoder_sub' in locals():
            del encoder_sub
        torch.cuda.empty_cache()

        return None, None
    
    
    
# def train_for_subject_embedding_with_label(args, subject_id, device, num_patches, dataset,encoder_group):
#     """
#     Train full fine-tuning reconstruction model for a single subject (V2-Fixed)

#     Args:
#         args: Command line arguments
#         subject_id: Target subject ID
#         device: CUDA device
#         num_patches: Number of patches (A) for current dataset
#         dataset: Pre-loaded EEGDataset instance

#     Returns:
#         dict: Results dictionary with metrics, or None if failed
#     """
#     try:
#         # Create subject-specific output directory
#         subject_dir = os.path.join(args.output_root, args.dataset, f'subject_{subject_id:02d}')
#         os.makedirs(subject_dir, exist_ok=True)

#         print(f"\n{'='*80}")
#         print(f"Subject {subject_id:02d} - Loading Data")
#         print(f"{'='*80}")
        
#         if args.multi_task:
#             train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
#                 dataset= dataset,
#                 target_subject_id=subject_id,
#                 target_task_id=args.multi_task_ids,
#                 train_size =args.train_size,
#                 test_size=args.test_size,
#                 val_size=args.val_size,
#                 batch_size =args.batch_size,
#                 random_state=args.seed,
#                 num_workers=args.num_workers
#             )
#         elif args.dataset_suffle:
#             train_loader, val_loader, test_loader = create_single_subject_dataloaders(
#                 dataset=dataset,
#                 target_subject_id=subject_id,
#                 train_size =args.train_size,
#                 test_size=args.test_size,
#                 val_size=args.val_size,
#                 batch_size=args.batch_size,
#                 random_state=args.seed,
#                 num_workers=args.num_workers
#             )
#         else:
#             # Create dataloaders for cognitive task
#             train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
#                 dataset=dataset,
#                 target_subject_id=subject_id,
#                 train_size =args.train_size,
#                 test_size=args.test_size,
#                 val_size=args.val_size,
#                 batch_size=args.batch_size,
#                 random_state=args.seed,
#                 num_workers=args.num_workers
#             )

#         train_size = len(train_loader.dataset)
#         val_size = len(val_loader.dataset)
#         test_size = len(test_loader.dataset)

#         print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

#         # Check minimum samples
#         if train_size < args.min_samples_per_class:
#             raise ValueError(f"Insufficient training samples: {train_size}")

#         if test_size == 0:
#             print("Warning: Test set is empty! Skipping test evaluation.")
#             test_metrics = None  # Set test_metrics to None or some default value if needed.
#         else:
#             test_metrics = True  # Flag indicating that test set is not empty

# ########################### in chans #################################            
#         if args.dataset=='cogbci':
#             in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
#             print('in_chans',in_chans)    
#         else:
#             in_chans, ch_names, chan_number = get_dataset_name(args)
#             print('in_chans', in_chans)
            
#         # if args.model =='brainomni':
#         #     positions, sensor_types, valid_electrodes = extract_1020_electrode_positions(ch_names)
            
# ########################### in chans #################################   
        
        
#         # load_vanilla_labram_weights(encoder_group, args.pretrained_path, args.model_key, args.model_prefix)
        
#         # encoder_group = encoder_group.to(device)
#         for name, param in encoder_group.named_parameters():
#             param.requires_grad = False
#         if args.model =='labram_base_patch200_200':
#             encoder_sub = create_model(
#                 args.model,
#                 pretrained=False,
#                 use_mean_pooling=args.use_mean_pooling,
#                 init_scale=args.init_scale,
#                 use_rel_pos_bias=args.rel_pos_bias,
#                 use_abs_pos_emb=args.abs_pos_emb,
#                 init_values=args.layer_scale_init_value,
#                 qkv_bias=args.qkv_bias,
#                 num_patches_per_channel_input=num_patches
#             ).to(device)
            
#             load_vanilla_labram_weights(encoder_sub, args.pretrained_path, args.model_key, args.model_prefix)
#             if args.fitune_type == 'lora':
#                 encoder_sub = create_lora(encoder_sub, args)
#                 for name, param in encoder_sub.named_parameters():
#                     if 'lora' not in name:  # 只冻结非LoRA参数
#                         param.requires_grad = False
#                     else:
#                         param.requires_grad = True
#             else:
#                 for name, param in encoder_sub.named_parameters():
#                     param.requires_grad = True
                    
    
#         elif args.model =='CBraMod':
#             encoder_sub = CBraMod().to(device)
#             encoder_sub.load_state_dict(torch.load(args.pretrained_path, map_location=device))
#             # for name, module in encoder_sub.named_modules():
#             #     print(f"Module Name: {name}, Module Type: {type(module)}")
#             if args.fitune_type == 'lora':
#                 encoder_sub = apply_lora_to_model(encoder_sub, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
#                 # freeze_non_lora_parameters(encoder_sub)
#                 for name, param in encoder_sub.named_parameters():
#                     if 'lora' not in name:  # 只冻结非LoRA参数
#                         param.requires_grad = False
#                     else:
#                         param.requires_grad = True
#             else:
#                 for name, param in encoder_sub.named_parameters():
#                     param.requires_grad = True
#         # elif args.model == 'FEMBA':
#         #     raise ValueError(f"Unsupported model type: {args.model}")    


#         for name, param in encoder_sub.named_parameters():
#             if param.requires_grad:
#                 print(f"group_sub 可训练: {name}")
                
#         for name, param in encoder_group.named_parameters():
#             if param.requires_grad:
#                 print(f"group_encoder 可训练: {name}")
        

#         # Parameter details
#         # total_params, trainable_params, trainable_percentage, trainable_modules = calculate_parameter_percentage_with_names(encoder_group)
#         # print(f"encoder_group Total parameters: {total_params}")
#         # print(f"encoder_group Trainable parameters: {trainable_params}")
#         # print(f"encoder_group Trainable parameters percentage: {trainable_percentage:.2f}%")
#         # for name, module in encoder_sub.named_modules():
#         #     print(f"Module Name: {name}, Module Type: {type(module)}")
#         total_params, trainable_params, trainable_percentage, trainable_modules = calculate_parameter_percentage_with_names(encoder_sub)
#         print(f"encoder_sub Total parameters: {total_params}")
#         print(f"encoder_sub Trainable parameters: {trainable_params}")
#         print(f"encoder_sub Trainable parameters percentage: {trainable_percentage:.2f}%")
#         # for name, param in encoder_sub.named_parameters():
#         #     print(f"{name}: requires_grad={param.requires_grad}")
#         # for name, param in encoder_sub.named_parameters():
#         #     if param.requires_grad and param.grad is not None:
#         #         print(f"{name} 有梯度")
#         #     elif param.requires_grad and param.grad is None:
#         #         print(f"警告: {name} 可训练但梯度为None")

#         # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder_sub.parameters()), lr=args.lr)
#         trainable_params = [p for p in encoder_sub.parameters() if p.requires_grad]
#         optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
#         best_val_loss = float('inf')
#         patience_counter = 0
#         train_losses = []
#         val_losses = []
      

#         print(f"\n{'='*80}")
#         print(f"Subject {subject_id:02d} - Training ({args.fitune_type})")
#         print(f"{'='*80}")
#         # for name, param in encoder_sub.named_parameters():
#         #     print(f"{name}: requires_grad={param.requires_grad}")
#         masking_strategy = Embedding_MaskingStrategy(mask_ratio=args.mask_ratio, mask_type=args.mask_type)    
#         # Training loop
#         for epoch in range(args.epochs):
#             if epoch < args.warmup_epochs:
#                 lr_mult = (epoch + 1) / args.warmup_epochs
#                 current_lr = args.lr * lr_mult
#             else:
#                 progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
#                 current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = current_lr

#             encoder_sub.train()
#             encoder_group.eval()
#             train_loss = 0
#             train_loss_intra =0
#             train_loss_cross =0

#             for batch in train_loader:
#                 # if not data.requires_grad:
#                 #     data = data.requires_grad_(True)
#                 # if not contrastive_data.requires_grad:
#                 #     contrastive_data = contrastive_data.requires_grad_(True)
#                 optimizer.zero_grad()
#                 eeg = batch['eeg'].to(device)
#                 label = batch['label'].to(device)
#                 B, C, A, T = eeg.shape
#                    # Forward pass
#                 # x_patch = encoder_sub.debug_patch_embed_gradient(data)
#                 if args.embedding_high_level:
#                     mask_eeg, mask = masking_strategy(eeg)
#                     if args.model =='CBraMod':
#                         embedding_mask = encoder_sub(mask_eeg)
#                         embedding_truth = encoder_group(eeg)
#                     elif args.model =='labram_base_patch200_200':
#                         embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
#                         embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
    
#                     loss, loss_intra, loss_cross = label_aware_contrastive_4d_channel(embedding_mask, embedding_truth, label, args,
#                                                                                       temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, 
#                                                                                       batch_size= B, channel=C, patch=A, dimension=T)
#                 else:
#                     mask_eeg,mask = masking_strategy(eeg)
#                     embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
#                     embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans)
#                     loss, loss_intra, loss_cross = label_aware_contrastive(embedding_mask, embedding_truth, label, temperature=0.07)
                    
#                 loss.backward()
#                 optimizer.step()

#                 # Track the different losses separately
#                 train_loss += loss.item()
#                 train_loss_intra += loss_intra.item()
#                 train_loss_cross += loss_cross.item()

#             avg_train_loss = train_loss / len(train_loader)
#             avg_train_loss_intra = train_loss_intra / len(train_loader)
#             avg_train_loss_cross = train_loss_cross / len(train_loader)
            
#             train_losses.append(avg_train_loss)

#             # # Validation loss
#             # encoder_sub.eval()
#             # encoder_group.eval()
#             # val_loss = 0
#             # with torch.no_grad():
#             #     for val_data, val_contrastive_data,label, contrastive_label in val_loader:
#             #         val_data = val_data.to(device)
#             #         val_contrastive_data = val_contrastive_data.to(device)
#             #         val_embedding_sub = encoder_sub.forward_features(val_data, input_chans=in_chans)
#             #         val_embedding_group = encoder_group.forward_features(val_contrastive_data, input_chans=in_chans)
#             #         val_loss += info_nce_loss_loop(val_embedding_sub, val_embedding_group, temperature=0.1).item()

#             # avg_val_loss = val_loss / len(val_loader)
#             # val_losses.append(avg_val_loss)
#             # Validation loss
#             # Validation loss
#             encoder_sub.eval()
#             encoder_group.eval()
#             val_loss = 0
#             val_loss_intra = 0
#             val_loss_cross = 0
#             with torch.no_grad():
#                 for batch in val_loader:
#                     eeg = batch['eeg'].to(device)  # 获取数据
#                     label = batch['label'].to(device)
#                     B, C, A, T = eeg.shape  # 添加形状提取
#                     if args.embedding_high_level:
#                         mask_eeg, mask = masking_strategy(eeg)
#                         if args.model =='CBraMod':
#                             val_embedding_sub = encoder_sub(mask_eeg)
#                             val_embedding_group = encoder_group(eeg)
#                         elif args.model =='labram_base_patch200_200':
#                             val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
#                             val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
#                         val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive_4d_channel(val_embedding_sub, val_embedding_group, label, args,
#                                                                                                                      temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, 
#                                                                                                                      batch_size= B, channel=C, patch=A, dimension=T)
#                     # if args.embedding_high_level:
#                     #     mask_eeg, mask = masking_strategy(eeg)
#                     #     val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans, return_patch_tokens=True)
#                     #     val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans, return_patch_tokens=True)
#                     #     val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive_4d_channel(val_embedding_sub, val_embedding_group, label, temperature=0.07, batch_size=B, channel=C, patch=A, dimension=T)
#                     else:
#                         mask_eeg, mask = masking_strategy(eeg)
#                         val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
#                         val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
#                         val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive(val_embedding_sub, val_embedding_group, label, temperature=0.07)
                    
#                     val_loss += val_loss_curr.item()
#                     val_loss_intra += val_loss_intra_curr.item()
#                     val_loss_cross += val_loss_cross_curr.item()

#             avg_val_loss = val_loss / len(val_loader)
#             avg_val_loss_intra = val_loss_intra / len(val_loader)
#             avg_val_loss_cross = val_loss_cross / len(val_loader)

#             val_losses.append(avg_val_loss)

#             # if (epoch + 1) % 10 == 0 or epoch == 0:
#             #     print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")
#             if (epoch + 1) % 10 == 0 or epoch == 0:
#                 print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, "
#                     f"Loss MSE={avg_train_loss_intra:.4f}, Loss Cross={avg_train_loss_cross:.4f}, "
#                     f"Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")

#             # Checkpoint saving and early stopping
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 patience_counter = 0
#                 torch.save({
#                     'epoch': epoch,
#                     'encoder_state_dict': encoder_sub.state_dict(),
#                     'val_loss': avg_val_loss,
#                 }, os.path.join(subject_dir, 'best_model.pth'))
#             else:
#                 patience_counter += 1
#                 if patience_counter >= args.patience:
#                     print(f"Early stopping at epoch {epoch + 1}")
#                     break

#         losses = {
#         'train_losses': train_losses,
#         'val_losses': val_losses}
#         # Final test evaluation if test set is available
#         if test_metrics is not None:  # Check if test set is not empty
#             print(f"\n{'='*80}")
#             print(f"Subject {subject_id:02d} - Final Test Evaluation")
#             print(f"{'='*80}")

#             checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'))
#             encoder_sub.load_state_dict(checkpoint['encoder_state_dict'])

#             # Test metrics
#             test_loss = 0
#             test_loss_intra = 0
#             test_loss_cross = 0
#             encoder_sub.eval()
#             encoder_group.eval()
#             with torch.no_grad():
#                 for batch in test_loader:
#                     eeg = batch['eeg'].to(device)  # 获取数据
#                     label = batch['label'].to(device)
#                     B, C, A, T = eeg.shape  # 添加形状提取
#                     if args.embedding_high_level:
#                         mask_eeg, mask = masking_strategy(eeg)
#                         if args.model =='CBraMod':
#                             test_embedding_sub = encoder_sub(mask_eeg)
#                             test_embedding_group = encoder_group(eeg)
#                         elif args.model =='labram_base_patch200_200':
#                             test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
#                             test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
#                         test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = label_aware_contrastive_4d_channel(test_embedding_sub, test_embedding_group, label,args, 
#                                                                                                                         temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, 
#                                                                                                                         batch_size= B, channel=C, patch=A, dimension=T)
#                     else:
#                         mask_eeg,mask = masking_strategy(eeg)
#                         test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
#                         test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
#                         test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = label_aware_contrastive(test_embedding_sub, test_embedding_group,label, temperature=0.07)
                    
#                     test_loss += test_loss_curr.item()
#                     test_loss_intra += test_loss_intra_curr.item()
#                     test_loss_cross += test_loss_cross_curr.item()

#             avg_test_loss = test_loss / len(test_loader)
#             avg_test_loss_intra = test_loss_intra / len(test_loader)
#             avg_test_loss_cross = test_loss_cross / len(test_loader)

#             # Save metrics
#             metrics = {
#                 'subject_id': int(subject_id),
#                 'train_size': train_size,
#                 'train_loss': avg_train_loss,
#                 'train_loss_intra': avg_train_loss_intra,
#                 'train_loss_cross': avg_train_loss_cross,
#                 'val_size': val_size,
#                 'val_loss': avg_val_loss,
#                 'val_loss_intra': avg_val_loss_intra,
#                 'val_loss_cross': avg_val_loss_cross,
#                 'test_size': test_size,
#                 'test_loss': avg_test_loss,
#                 'test_loss_intra': avg_test_loss_intra,
#                 'test_loss_cross': avg_test_loss_cross,
#                 'best_epoch': checkpoint['epoch'] + 1,
#             }

#             with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
#                 json.dump(metrics, f, indent=2)

#             print(f"  ✓ Results saved to {subject_dir}")

#         else:
#             metrics = {
#                 'subject_id': int(subject_id),
#                 'train_size': train_size,
#                 'train_loss': avg_train_loss,
#                 'val_size': val_size,
#                 'val_loss': avg_val_loss,
#                 'best_epoch': checkpoint['epoch'] + 1,
#             }

#             with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
#                 json.dump(metrics, f, indent=2)

#             print(f"  ✓ Results saved to {subject_dir}")
#         # Cleanup
#         torch.cuda.empty_cache()

#         return metrics, losses

#     except Exception as e:
#         print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
#         import traceback
#         traceback.print_exc()

#         if 'encoder_sub' in locals():
#             del encoder_sub
#         torch.cuda.empty_cache()

#         return None, None
    

def train_for_subject_embedding_with_label_test(args, subject_id, device, num_patches, dataset,encoder_group):
    """
    Train full fine-tuning reconstruction model for a single subject (V2-Fixed)

    Args:
        args: Command line arguments
        subject_id: Target subject ID
        device: CUDA device
        num_patches: Number of patches (A) for current dataset
        dataset: Pre-loaded EEGDataset instance

    Returns:
        dict: Results dictionary with metrics, or None if failed
    """
    try:
        # Create subject-specific output directory
        subject_dir = os.path.join(args.output_root, args.dataset, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")
        
        if args.multi_task:
            train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
                dataset= dataset,
                target_subject_id=subject_id,
                target_task_id=args.multi_task_ids,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        else:
            # Create dataloaders for cognitive task
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # Check minimum samples
        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        if test_size == 0:
            print("Warning: Test set is empty! Skipping test evaluation.")
            test_metrics = None  # Set test_metrics to None or some default value if needed.
        else:
            test_metrics = True  # Flag indicating that test set is not empty

########################### in chans #################################            
        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)    
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)
            
        if args.model =='Brainomni':
            position, sensor_type, valid_electrodes = extract_1020_electrode_positions(ch_names,device)    
            # print(position.shape)        
########################### in chans #################################   
        
        
        # load_vanilla_labram_weights(encoder_group, args.pretrained_path, args.model_key, args.model_prefix)
        
##################### frozen encoder_group #####################################################
        for name, param in encoder_group.named_parameters():
            param.requires_grad = False
##################### setup encoder_sub #####################################################  
####################### labram ################################################            
        if args.model =='labram_base_patch200_200':
            encoder_sub = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches
            ).to(device)
            load_vanilla_labram_weights(encoder_sub, args.pretrained_path, args.model_key, args.model_prefix)
            if args.fitune_type == 'lora':
                encoder_sub = create_lora(encoder_sub, args)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:  # 只冻结非LoRA参数
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True
#######################CbraMod ################################################                    
    
        elif args.model =='CBraMod':
            encoder_sub = CBraMod().to(device)
            encoder_sub.load_state_dict(torch.load(args.pretrained_path, map_location=device))
            # for name, module in encoder_sub.named_modules():
            #     print(f"Module Name: {name}, Module Type: {type(module)}")
            if args.fitune_type == 'lora':
                encoder_sub = apply_lora_to_model(encoder_sub, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
                # freeze_non_lora_parameters(encoder_sub)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:  # 只冻结非LoRA参数
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True
        # elif args.model == 'FEMBA':
        #     encoder_sub = FEMBA(seq_length=1280, num_channels=22, num_classes=0, embed_dim=35, num_blocks=4)    

####################### BrainOmni ################################################  
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
            encoder_sub=get_brainomni_model(args.pretrained_path).to(device)
            if args.fitune_type == 'lora':
                encoder_sub=create_lora(encoder_sub,args)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:  # 只冻结非LoRA参数
                        param.requires_grad = False
                    else:
                        param.requires_grad = True          
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True
                    
        for name, param in encoder_sub.named_parameters():
            if param.requires_grad:
                print(f"group_sub 可训练: {name}")
                
        for name, param in encoder_group.named_parameters():
            if param.requires_grad:
                print(f"group_encoder 可训练: {name}")
        
        total_params, trainable_params, trainable_percentage, trainable_modules = calculate_parameter_percentage_with_names(encoder_sub)
        print(f"encoder_sub Total parameters: {total_params}")
        print(f"encoder_sub Trainable parameters: {trainable_params}")
        print(f"encoder_sub Trainable parameters percentage: {trainable_percentage:.2f}%")
        # for name, param in encoder_sub.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        # for name, param in encoder_sub.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         print(f"{name} 有梯度")
        #     elif param.requires_grad and param.grad is None:
        #         print(f"警告: {name} 可训练但梯度为None")

        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder_sub.parameters()), lr=args.lr)
        trainable_params = [p for p in encoder_sub.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
      

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training ({args.fitune_type})")
        print(f"{'='*80}")
        # for name, param in encoder_sub.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        masking_strategy = Embedding_MaskingStrategy(mask_ratio=args.mask_ratio, mask_type=args.mask_type)    
        # Training loop
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            encoder_sub.train()
            encoder_group.eval()
            train_loss = 0
            train_loss_intra =0
            train_loss_cross =0

            for batch in train_loader:
                # if not data.requires_grad:
                #     data = data.requires_grad_(True)
                # if not contrastive_data.requires_grad:
                #     contrastive_data = contrastive_data.requires_grad_(True)
                optimizer.zero_grad()
                eeg = batch['eeg'].to(device)
                label = batch['label'].to(device)
                B, C, A, T = eeg.shape
                   # Forward pass
                # x_patch = encoder_sub.debug_patch_embed_gradient(data)
                if args.embedding_high_level:
                    mask_eeg, mask = masking_strategy(eeg)
                    if args.model =='CBraMod':
                        embedding_mask = encoder_sub(mask_eeg)
                        embedding_truth = encoder_group(eeg)
                    elif args.model =='labram_base_patch200_200':
                        embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                        embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                    elif args.model =='Brainomni':
                        eeg = eeg.reshape(B, C, A * T)
                        mask_eeg = mask_eeg.reshape(B, C, A * T)
                        # print(B)
                        positions = position.unsqueeze(0).repeat(B, 1, 1).to(device)
                        sensor_types = sensor_type.unsqueeze(0).repeat(B, 1).to(device)
                        embedding_mask = encoder_sub.encode(mask_eeg, positions,sensor_types)
                        embedding_truth = encoder_group.encode(eeg, positions,sensor_types)
    
                    loss, loss_intra, loss_cross = label_aware_contrastive_4d_channel(embedding_mask, embedding_truth, label, args,
                                                                                      temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, 
                                                                                      batch_size= B, channel=C, patch=A, dimension=T)
                else:
                    mask_eeg,mask = masking_strategy(eeg)
                    embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                    embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans)
                    loss, loss_intra, loss_cross = label_aware_contrastive(embedding_mask, embedding_truth, label, temperature=0.07)
                    
                loss.backward()
                optimizer.step()

                # Track the different losses separately
                train_loss += loss.item()
                train_loss_intra += loss_intra.item()
                train_loss_cross += loss_cross.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_train_loss_intra = train_loss_intra / len(train_loader)
            avg_train_loss_cross = train_loss_cross / len(train_loader)
            
            train_losses.append(avg_train_loss)

            encoder_sub.eval()
            encoder_group.eval()
            val_loss = 0
            val_loss_intra = 0
            val_loss_cross = 0
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch['eeg'].to(device)  # 获取数据
                    label = batch['label'].to(device)
                    B, C, A, T = eeg.shape  # 添加形状提取
                    if args.embedding_high_level:
                        mask_eeg, mask = masking_strategy(eeg)
                        if args.model =='CBraMod':
                            val_embedding_sub = encoder_sub(mask_eeg)
                            val_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                            val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                            
                        elif args.model =='Brainomni':
                            eeg = eeg.reshape(B, C, A * T)
                            mask_eeg = mask_eeg.reshape(B, C, A * T)
                            positions = position.unsqueeze(0).repeat(B, 1, 1).to(device)
                            sensor_types = sensor_type.unsqueeze(0).repeat(B, 1).to(device)
                            val_embedding_sub = encoder_sub.encode(mask_eeg, positions,sensor_types)
                            val_embedding_group = encoder_group.encode(eeg, positions,sensor_types)
                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive_4d_channel(val_embedding_sub, val_embedding_group, label, args,
                                                                                                                     temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, 
                                                                                                                     batch_size= B, channel=C, patch=A, dimension=T)
                    # if args.embedding_high_level:
                    #     mask_eeg, mask = masking_strategy(eeg)
                    #     val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans, return_patch_tokens=True)
                    #     val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans, return_patch_tokens=True)
                    #     val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive_4d_channel(val_embedding_sub, val_embedding_group, label, temperature=0.07, batch_size=B, channel=C, patch=A, dimension=T)
                    else:
                        mask_eeg, mask = masking_strategy(eeg)
                        val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                        val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive(val_embedding_sub, val_embedding_group, label, temperature=0.07)
                    
                    val_loss += val_loss_curr.item()
                    val_loss_intra += val_loss_intra_curr.item()
                    val_loss_cross += val_loss_cross_curr.item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_loss_intra = val_loss_intra / len(val_loader)
            avg_val_loss_cross = val_loss_cross / len(val_loader)

            val_losses.append(avg_val_loss)

            # if (epoch + 1) % 10 == 0 or epoch == 0:
            #     print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, "
                    f"Loss MSE={avg_train_loss_intra:.4f}, Loss Cross={avg_train_loss_cross:.4f}, "
                    f"Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")

            # Checkpoint saving and early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder_sub.state_dict(),
                    'val_loss': avg_val_loss,
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        losses = {
        'train_losses': train_losses,
        'val_losses': val_losses}
        # Final test evaluation if test set is available
        if test_metrics is not None:  # Check if test set is not empty
            print(f"\n{'='*80}")
            print(f"Subject {subject_id:02d} - Final Test Evaluation")
            print(f"{'='*80}")

            checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'))
            encoder_sub.load_state_dict(checkpoint['encoder_state_dict'])

            # Test metrics
            test_loss = 0
            test_loss_intra = 0
            test_loss_cross = 0
            encoder_sub.eval()
            encoder_group.eval()
            with torch.no_grad():
                for batch in test_loader:
                    eeg = batch['eeg'].to(device)  # 获取数据
                    label = batch['label'].to(device)
                    B, C, A, T = eeg.shape  # 添加形状提取
                    if args.embedding_high_level:
                        mask_eeg, mask = masking_strategy(eeg)
                        if args.model =='CBraMod':
                            test_embedding_sub = encoder_sub(mask_eeg)
                            test_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                            test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                        elif args.model =='Brainomni':
                            eeg = eeg.reshape(B, C, A * T)
                            mask_eeg = mask_eeg.reshape(B, C, A * T)
                            positions = position.unsqueeze(0).repeat(B, 1, 1).to(device)
                            sensor_types = sensor_type.unsqueeze(0).repeat(B, 1).to(device)
                            test_embedding_sub = encoder_sub.encode(mask_eeg, positions,sensor_types)
                            test_embedding_group = encoder_group.encode(eeg, positions,sensor_types)   
                                                     
                        test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = label_aware_contrastive_4d_channel(test_embedding_sub, test_embedding_group, label,args, 
                                                                                                                        temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, 
                                                                                                                        batch_size= B, channel=C, patch=A, dimension=T)
                    else:
                        mask_eeg,mask = masking_strategy(eeg)
                        test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                        test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = label_aware_contrastive(test_embedding_sub, test_embedding_group,label, temperature=0.07)
                    
                    test_loss += test_loss_curr.item()
                    test_loss_intra += test_loss_intra_curr.item()
                    test_loss_cross += test_loss_cross_curr.item()

            avg_test_loss = test_loss / len(test_loader)
            avg_test_loss_intra = test_loss_intra / len(test_loader)
            avg_test_loss_cross = test_loss_cross / len(test_loader)

            # Save metrics
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'train_loss_intra': avg_train_loss_intra,
                'train_loss_cross': avg_train_loss_cross,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'val_loss_intra': avg_val_loss_intra,
                'val_loss_cross': avg_val_loss_cross,
                'test_size': test_size,
                'test_loss': avg_test_loss,
                'test_loss_intra': avg_test_loss_intra,
                'test_loss_cross': avg_test_loss_cross,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        else:
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        # Cleanup
        torch.cuda.empty_cache()

        return metrics, losses

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'encoder_sub' in locals():
            del encoder_sub
        torch.cuda.empty_cache()

        return None, None



import torch.nn.utils as utils
def train_for_subject_embedding_with_label(args, subject_id, device, num_patches, dataset,encoder_group):
    """
    Train full fine-tuning reconstruction model for a single subject (V2-Fixed)

    Args:
        args: Command line arguments
        subject_id: Target subject ID
        device: CUDA device
        num_patches: Number of patches (A) for current dataset
        dataset: Pre-loaded EEGDataset instance

    Returns:
        dict: Results dictionary with metrics, or None if failed
    """
    try:
        # Create subject-specific output directory
        subject_dir = os.path.join(args.output_root, args.dataset, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")
        
        if args.multi_task:
            train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
                dataset= dataset,
                target_subject_id=subject_id,
                target_task_id=args.multi_task_ids,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        elif args.train_count is not None:
            # Create dataloaders for cognitive task
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_control(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                train_count = args.train_count
            )
        else:
            # Create dataloaders for cognitive task
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # Check minimum samples
        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        if test_size == 0:
            print("Warning: Test set is empty! Skipping test evaluation.")
            test_metrics = None  # Set test_metrics to None or some default value if needed.
        else:
            test_metrics = True  # Flag indicating that test set is not empty

########################### in chans #################################            
        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)    
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)
            
        if args.model =='Brainomni':
            position, sensor_type, valid_electrodes = extract_1020_electrode_positions(ch_names,device)    
            # print(position.shape)        
########################### in chans #################################   
        
        
        # load_vanilla_labram_weights(encoder_group, args.pretrained_path, args.model_key, args.model_prefix)
        
##################### frozen encoder_group #####################################################
        for name, param in encoder_group.named_parameters():
            param.requires_grad = False
##################### setup encoder_sub #####################################################  
        force_student_random = getattr(args, 'student_random_init', False)
####################### labram ################################################            
        if args.model =='labram_base_patch200_200':
            encoder_sub = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches
            ).to(device)
            
            # 只有在传了路径 且 没有开启强制随机初始化 时，才加载权重
            if args.pretrained_path and not force_student_random:
                load_vanilla_labram_weights(encoder_sub, args.pretrained_path, args.model_key, args.model_prefix)
            else:
                print("⚠️ [encoder_sub] Student uses RANDOM INITIALIZATION for LaBraM.")
                
            if args.fitune_type == 'lora':
                encoder_sub = create_lora(encoder_sub, args)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:  # 只冻结非LoRA参数
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True

####################### CbraMod ################################################                    
        elif args.model =='CBraMod':
            encoder_sub = CBraMod().to(device)
            
            if args.pretrained_path and not force_student_random:
                encoder_sub.load_state_dict(torch.load(args.pretrained_path, map_location=device))
            else:
                print("⚠️ [encoder_sub] Student uses RANDOM INITIALIZATION for CBraMod.")
                
            if args.fitune_type == 'lora':
                encoder_sub = apply_lora_to_model(encoder_sub, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:  
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True

####################### BrainOmni ################################################  
        elif args.model =='Brainomni':
            from brainomni.model import BrainOmni
            # 给加载函数加一个 load_weights 控制开关
            def get_brainomni_model(ckpt_path, load_weights=True):
                with open(os.path.join(ckpt_path, "model_cfg.json")) as f:
                    cfg = json.load(f)
                model = BrainOmni(**cfg)
                pt_path = os.path.join(ckpt_path, "BrainOmni.pt")
                if load_weights and os.path.exists(pt_path):
                    print("Loading BrainOmni weights for Student...")
                    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True), strict=False)
                else:
                    print("⚠️ [encoder_sub] Student uses RANDOM INITIALIZATION for BrainOmni.")
                return model
                
            # 根据开关决定是否加载权重
            should_load = not force_student_random
            encoder_sub = get_brainomni_model(args.pretrained_path, load_weights=should_load).to(device)
            
            if args.fitune_type == 'lora':
                encoder_sub=create_lora(encoder_sub,args)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name: 
                        param.requires_grad = False
                    else:
                        param.requires_grad = True          
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True

####################### BIOT ################################################  
        elif args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder_sub = BIOTEncoder(emb_size=256, heads=8, depth=4, n_channels=18).to(device)
            
            if args.pretrained_path and not force_student_random:
                encoder_sub.load_state_dict(torch.load(args.pretrained_path, map_location=device))    
            else:
                print("⚠️ [encoder_sub] Student uses RANDOM INITIALIZATION for BIOT.")
                
            if args.fitune_type == 'lora':
                encoder_sub=create_lora(encoder_sub,args)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' in name.lower():
                        param.requires_grad = True
                    else:
                        param.requires_grad = False       
            else:
                for name, param in encoder_sub.named_parameters():
                    if param.is_floating_point():  
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                    
        for name, param in encoder_sub.named_parameters():
            if param.requires_grad:
                print(f"group_sub 可训练: {name}")
                
        for name, param in encoder_group.named_parameters():
            if param.requires_grad:
                print(f"group_encoder 可训练: {name}")
        
        total_params, trainable_params, trainable_percentage, trainable_modules = calculate_parameter_percentage_with_names(encoder_sub)
        print(f"encoder_sub Total parameters: {total_params}")
        print(f"encoder_sub Trainable parameters: {trainable_params}")
        print(f"encoder_sub Trainable parameters percentage: {trainable_percentage:.2f}%")

        trainable_params = [p for p in encoder_sub.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
      

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training ({args.fitune_type})")
        print(f"{'='*80}")
        # for name, param in encoder_sub.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        masking_strategy = Embedding_MaskingStrategy(mask_ratio=args.mask_ratio, mask_type=args.mask_type) 
        max_grad_norm = 1.0   
        # Training loop
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            encoder_sub.train()
            encoder_group.eval()
            train_loss = 0
            train_loss_intra =0
            train_loss_cross =0

            for batch in train_loader:
                # if not data.requires_grad:
                #     data = data.requires_grad_(True)
                # if not contrastive_data.requires_grad:
                #     contrastive_data = contrastive_data.requires_grad_(True)
                optimizer.zero_grad()
                eeg = batch['eeg'].to(device)
                label = batch['label'].to(device)
                B, C, A, T = eeg.shape
                   # Forward pass
                # x_patch = encoder_sub.debug_patch_embed_gradient(data)
                if args.embedding_high_level:
                    mask_eeg, mask = masking_strategy(eeg)
                    if args.model =='CBraMod':
                        embedding_mask = encoder_sub(mask_eeg)
                        embedding_truth = encoder_group(eeg)
                    elif args.model =='labram_base_patch200_200':
                        embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                        embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                    elif args.model =='Brainomni':
                        eeg = eeg.reshape(B, C, A * T)
                        mask_eeg = mask_eeg.reshape(B, C, A * T)
                        # print(B)
                        positions = position.unsqueeze(0).repeat(B, 1, 1).to(device)
                        sensor_types = sensor_type.unsqueeze(0).repeat(B, 1).to(device)
                        embedding_mask = encoder_sub.encode(mask_eeg, positions,sensor_types)
                        embedding_truth = encoder_group.encode(eeg, positions,sensor_types)


                    loss, loss_intra, loss_cross = label_aware_contrastive_4d_channel(embedding_mask, embedding_truth, label, args,
                                                                                      temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, 
                                                                                      batch_size= B, channel=C, patch=A, dimension=T)
                else:
                    if args.model =='BIOT':
                        mask_eeg, mask = masking_strategy(eeg)
                        eeg = eeg_projector.channel_project(eeg, ch_names)
                        eeg_mask = eeg_projector.channel_project(mask_eeg, ch_names)
                        embedding_mask = encoder_sub(eeg_mask)
                        embedding_truth = encoder_group(eeg)
                    elif args.model =='labram_base_patch200_200':
                        mask_eeg,mask = masking_strategy(eeg)
                        embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                        embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans)
                        
                    loss, loss_intra, loss_cross = label_aware_contrastive(embedding_mask, embedding_truth, label, 
                                                                           temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, )
                    
                loss.backward()
                utils.clip_grad_norm_(encoder_sub.parameters(), max_grad_norm)
                optimizer.step()

                # Track the different losses separately
                train_loss += loss.item()
                train_loss_intra += loss_intra.item()
                train_loss_cross += loss_cross.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_train_loss_intra = train_loss_intra / len(train_loader)
            avg_train_loss_cross = train_loss_cross / len(train_loader)
            
            train_losses.append(avg_train_loss)

            encoder_sub.eval()
            encoder_group.eval()
            val_loss = 0
            val_loss_intra = 0
            val_loss_cross = 0
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch['eeg'].to(device)  # 获取数据
                    label = batch['label'].to(device)
                    B, C, A, T = eeg.shape  # 添加形状提取
                    if args.embedding_high_level:
                        mask_eeg, mask = masking_strategy(eeg)
                        if args.model =='CBraMod':
                            val_embedding_sub = encoder_sub(mask_eeg)
                            val_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                            val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                            
                        elif args.model =='Brainomni':
                            eeg = eeg.reshape(B, C, A * T)
                            mask_eeg = mask_eeg.reshape(B, C, A * T)
                            positions = position.unsqueeze(0).repeat(B, 1, 1).to(device)
                            sensor_types = sensor_type.unsqueeze(0).repeat(B, 1).to(device)
                            val_embedding_sub = encoder_sub.encode(mask_eeg, positions,sensor_types)
                            val_embedding_group = encoder_group.encode(eeg, positions,sensor_types)
                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive_4d_channel(val_embedding_sub, val_embedding_group, label, args,
                                                                                                                     temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, 
                                                                                                                     batch_size= B, channel=C, patch=A, dimension=T)
                    else:
                        if args.model =='BIOT':
                            mask_eeg, mask = masking_strategy(eeg)
                            eeg = eeg_projector.channel_project(eeg, ch_names)
                            eeg_mask = eeg_projector.channel_project(mask_eeg, ch_names)
                            val_embedding_sub = encoder_sub(eeg_mask)
                            val_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            mask_eeg, mask = masking_strategy(eeg)
                            val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                            val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive(val_embedding_sub, val_embedding_group, label, 
                                                                                                          temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, )
                    
                    val_loss += val_loss_curr.item()
                    val_loss_intra += val_loss_intra_curr.item()
                    val_loss_cross += val_loss_cross_curr.item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_loss_intra = val_loss_intra / len(val_loader)
            avg_val_loss_cross = val_loss_cross / len(val_loader)

            val_losses.append(avg_val_loss)

            # if (epoch + 1) % 10 == 0 or epoch == 0:
            #     print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, "
                    f"Loss MSE={avg_train_loss_intra:.4f}, Loss Cross={avg_train_loss_cross:.4f}, "
                    f"Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")

            # Checkpoint saving and early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder_sub.state_dict(),
                    'val_loss': avg_val_loss,
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        losses = {
        'train_losses': train_losses,
        'val_losses': val_losses}
        # Final test evaluation if test set is available
        if test_metrics is not None:  # Check if test set is not empty
            print(f"\n{'='*80}")
            print(f"Subject {subject_id:02d} - Final Test Evaluation")
            print(f"{'='*80}")

            checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'))
            encoder_sub.load_state_dict(checkpoint['encoder_state_dict'])

            # Test metrics
            test_loss = 0
            test_loss_intra = 0
            test_loss_cross = 0
            encoder_sub.eval()
            encoder_group.eval()
            with torch.no_grad():
                for batch in test_loader:
                    eeg = batch['eeg'].to(device)  # 获取数据
                    label = batch['label'].to(device)
                    B, C, A, T = eeg.shape  # 添加形状提取
                    if args.embedding_high_level:
                        mask_eeg, mask = masking_strategy(eeg)
                        if args.model =='CBraMod':
                            test_embedding_sub = encoder_sub(mask_eeg)
                            test_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                            test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                        elif args.model =='Brainomni':
                            eeg = eeg.reshape(B, C, A * T)
                            mask_eeg = mask_eeg.reshape(B, C, A * T)
                            positions = position.unsqueeze(0).repeat(B, 1, 1).to(device)
                            sensor_types = sensor_type.unsqueeze(0).repeat(B, 1).to(device)
                            test_embedding_sub = encoder_sub.encode(mask_eeg, positions,sensor_types)
                            test_embedding_group = encoder_group.encode(eeg, positions,sensor_types)   
                                                     
                        test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = label_aware_contrastive_4d_channel(test_embedding_sub, test_embedding_group, label,args, 
                                                                                                                        temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, 
                                                                                                                        batch_size= B, channel=C, patch=A, dimension=T)
                    else:
                        if args.model =='BIOT':
                            mask_eeg, mask = masking_strategy(eeg)
                            eeg = eeg_projector.channel_project(eeg, ch_names)
                            eeg_mask = eeg_projector.channel_project(mask_eeg, ch_names)
                            test_embedding_sub = encoder_sub(eeg_mask)
                            test_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            mask_eeg,mask = masking_strategy(eeg)
                            test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                            test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = label_aware_contrastive(test_embedding_sub, test_embedding_group,label, 
                                                                                                             temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, )
                    
                    test_loss += test_loss_curr.item()
                    test_loss_intra += test_loss_intra_curr.item()
                    test_loss_cross += test_loss_cross_curr.item()

            avg_test_loss = test_loss / len(test_loader)
            avg_test_loss_intra = test_loss_intra / len(test_loader)
            avg_test_loss_cross = test_loss_cross / len(test_loader)

            # Save metrics
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'train_loss_intra': avg_train_loss_intra,
                'train_loss_cross': avg_train_loss_cross,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'val_loss_intra': avg_val_loss_intra,
                'val_loss_cross': avg_val_loss_cross,
                'test_size': test_size,
                'test_loss': avg_test_loss,
                'test_loss_intra': avg_test_loss_intra,
                'test_loss_cross': avg_test_loss_cross,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        else:
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        # Cleanup
        torch.cuda.empty_cache()

        return metrics, losses

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'encoder_sub' in locals():
            del encoder_sub
        torch.cuda.empty_cache()

        return None, None


import torch.nn.utils as utils
def train_for_subject_embedding_with_label_dynamics(args, subject_id, device, num_patches, dataset,encoder_group):
    """
    Train full fine-tuning reconstruction model for a single subject (V2-Fixed)

    Args:
        args: Command line arguments
        subject_id: Target subject ID
        device: CUDA device
        num_patches: Number of patches (A) for current dataset
        dataset: Pre-loaded EEGDataset instance

    Returns:
        dict: Results dictionary with metrics, or None if failed
    """
    try:
        # Create subject-specific output directory
        subject_dir = os.path.join(args.output_root, args.dataset, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")
        
        if args.multi_task:
            train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
                dataset= dataset,
                target_subject_id=subject_id,
                target_task_id=args.multi_task_ids,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
            
        else:
            # Create dataloaders for cognitive task
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # Check minimum samples
        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        if test_size == 0:
            print("Warning: Test set is empty! Skipping test evaluation.")
            test_metrics = None  # Set test_metrics to None or some default value if needed.
        else:
            test_metrics = True  # Flag indicating that test set is not empty

########################### in chans #################################            
        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)    
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)
            
        if args.model =='Brainomni':
            position, sensor_type, valid_electrodes = extract_1020_electrode_positions(ch_names,device)    
            # print(position.shape)        
########################### in chans #################################   
        
        
        # load_vanilla_labram_weights(encoder_group, args.pretrained_path, args.model_key, args.model_prefix)
        
##################### frozen encoder_group #####################################################
        for name, param in encoder_group.named_parameters():
            param.requires_grad = False
##################### setup encoder_sub #####################################################  
####################### labram ################################################            
        if args.model =='labram_base_patch200_200':
            encoder_sub = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches
            ).to(device)
            load_vanilla_labram_weights(encoder_sub, args.pretrained_path, args.model_key, args.model_prefix)
            if args.fitune_type == 'lora':
                encoder_sub = create_lora(encoder_sub, args)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:  # 只冻结非LoRA参数
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True
#######################CbraMod ################################################                    
    
        elif args.model =='CBraMod':
            encoder_sub = CBraMod().to(device)
            encoder_sub.load_state_dict(torch.load(args.pretrained_path, map_location=device))
            # for name, module in encoder_sub.named_modules():
            #     print(f"Module Name: {name}, Module Type: {type(module)}")
            if args.fitune_type == 'lora':
                encoder_sub = apply_lora_to_model(encoder_sub, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
                # freeze_non_lora_parameters(encoder_sub)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:  # 只冻结非LoRA参数
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True
        # elif args.model == 'FEMBA':
        #     encoder_sub = FEMBA(seq_length=1280, num_channels=22, num_classes=0, embed_dim=35, num_blocks=4)    

####################### BrainOmni ################################################  
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
            encoder_sub=get_brainomni_model(args.pretrained_path).to(device)
            if args.fitune_type == 'lora':
                encoder_sub=create_lora(encoder_sub,args)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:  # 只冻结非LoRA参数
                        param.requires_grad = False
                    else:
                        param.requires_grad = True          
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True
####################### BrainOmni ################################################  
        elif args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder_sub = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)
            encoder_sub.load_state_dict(torch.load(args.pretrained_path, map_location=device))
            if args.fitune_type == 'lora':
                encoder_sub=create_lora(encoder_sub,args)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' in name.lower():
                        # LoRA参数可训练
                        param.requires_grad = True
                    else:
                        param.requires_grad = False       
            else:
                for name, param in encoder_sub.named_parameters():
                    if param.is_floating_point():  # 只设置浮点数的梯度
                        param.requires_grad = True
                    else:
                        # 整数类型参数（如index）保持requires_grad=False
                        param.requires_grad = False
                    
        for name, param in encoder_sub.named_parameters():
            if param.requires_grad:
                print(f"group_sub 可训练: {name}")
                
        for name, param in encoder_group.named_parameters():
            if param.requires_grad:
                print(f"group_encoder 可训练: {name}")
        
        total_params, trainable_params, trainable_percentage, trainable_modules = calculate_parameter_percentage_with_names(encoder_sub)
        print(f"encoder_sub Total parameters: {total_params}")
        print(f"encoder_sub Trainable parameters: {trainable_params}")
        print(f"encoder_sub Trainable parameters percentage: {trainable_percentage:.2f}%")

        trainable_params = [p for p in encoder_sub.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
      

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training ({args.fitune_type})")
        print(f"{'='*80}")
        # for name, param in encoder_sub.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        masking_strategy = Embedding_MaskingStrategy(mask_ratio=args.mask_ratio, mask_type=args.mask_type) 
        max_grad_norm = 1.0   
        # Training loop
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            encoder_sub.train()
            encoder_group.eval()
            train_loss = 0
            train_loss_intra =0
            train_loss_cross =0

            for batch in train_loader:
                # if not data.requires_grad:
                #     data = data.requires_grad_(True)
                # if not contrastive_data.requires_grad:
                #     contrastive_data = contrastive_data.requires_grad_(True)
                optimizer.zero_grad()
                eeg = batch['eeg'].to(device)
                label = batch['label'].to(device)
                B, C, A, T = eeg.shape
                   # Forward pass
                # x_patch = encoder_sub.debug_patch_embed_gradient(data)
                if args.embedding_high_level:
                    mask_eeg, mask = masking_strategy(eeg)
                    if args.model =='CBraMod':
                        embedding_mask = encoder_sub(mask_eeg)
                        embedding_truth = encoder_group(eeg)
                    elif args.model =='labram_base_patch200_200':
                        embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                        embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                    elif args.model =='Brainomni':
                        eeg = eeg.reshape(B, C, A * T)
                        mask_eeg = mask_eeg.reshape(B, C, A * T)
                        # print(B)
                        positions = position.unsqueeze(0).repeat(B, 1, 1).to(device)
                        sensor_types = sensor_type.unsqueeze(0).repeat(B, 1).to(device)
                        embedding_mask = encoder_sub.encode(mask_eeg, positions,sensor_types)
                        embedding_truth = encoder_group.encode(eeg, positions,sensor_types)


                    loss, loss_intra, loss_cross = label_aware_contrastive_4d_channel_dynamics(embedding_mask, embedding_truth, label, args,
                                                                                                epoch=epoch, warmup_epoch=args.warmup_epochs,
                                                                                                temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, lambda_mse=args.lambda_mse,
                                                                                                batch_size= B, channel=C, patch=A, dimension=T)
                else:
                    if args.model =='BIOT':
                        mask_eeg, mask = masking_strategy(eeg)
                        eeg = eeg_projector.channel_project(eeg, ch_names)
                        eeg_mask = eeg_projector.channel_project(mask_eeg, ch_names)
                        embedding_mask = encoder_sub(eeg_mask)
                        embedding_truth = encoder_group(eeg)
                    elif args.model =='labram_base_patch200_200':
                        mask_eeg,mask = masking_strategy(eeg)
                        embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                        embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans)
                        
                    loss, loss_intra, loss_cross = label_aware_contrastive_dynamics(embedding_mask, embedding_truth, label, 
                                                                                    epoch=epoch, warmup_epoch=args.warmup_epochs,
                                                                           temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha)
                    
                loss.backward()
                utils.clip_grad_norm_(encoder_sub.parameters(), max_grad_norm)
                optimizer.step()

                # Track the different losses separately
                train_loss += loss.item()
                train_loss_intra += loss_intra.item()
                train_loss_cross += loss_cross.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_train_loss_intra = train_loss_intra / len(train_loader)
            avg_train_loss_cross = train_loss_cross / len(train_loader)
            
            train_losses.append(avg_train_loss)

            encoder_sub.eval()
            encoder_group.eval()
            val_loss = 0
            val_loss_intra = 0
            val_loss_cross = 0
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch['eeg'].to(device)  # 获取数据
                    label = batch['label'].to(device)
                    B, C, A, T = eeg.shape  # 添加形状提取
                    if args.embedding_high_level:
                        mask_eeg, mask = masking_strategy(eeg)
                        if args.model =='CBraMod':
                            val_embedding_sub = encoder_sub(mask_eeg)
                            val_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                            val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                            
                        elif args.model =='Brainomni':
                            eeg = eeg.reshape(B, C, A * T)
                            mask_eeg = mask_eeg.reshape(B, C, A * T)
                            positions = position.unsqueeze(0).repeat(B, 1, 1).to(device)
                            sensor_types = sensor_type.unsqueeze(0).repeat(B, 1).to(device)
                            val_embedding_sub = encoder_sub.encode(mask_eeg, positions,sensor_types)
                            val_embedding_group = encoder_group.encode(eeg, positions,sensor_types)
                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive_4d_channel_dynamics(val_embedding_sub, val_embedding_group, label, args,
                                                                                                                     epoch=epoch, warmup_epoch=args.warmup_epochs,
                                                                                                                     temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, 
                                                                                                                     batch_size= B, channel=C, patch=A, dimension=T)
                    # if args.embedding_high_level:
                    #     mask_eeg, mask = masking_strategy(eeg)
                    #     val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans, return_patch_tokens=True)
                    #     val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans, return_patch_tokens=True)
                    #     val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive_4d_channel(val_embedding_sub, val_embedding_group, label, temperature=0.07, batch_size=B, channel=C, patch=A, dimension=T)
                    else:
                        if args.model =='BIOT':
                            mask_eeg, mask = masking_strategy(eeg)
                            eeg = eeg_projector.channel_project(eeg, ch_names)
                            eeg_mask = eeg_projector.channel_project(mask_eeg, ch_names)
                            val_embedding_sub = encoder_sub(eeg_mask)
                            val_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            mask_eeg, mask = masking_strategy(eeg)
                            val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                            val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive_dynamics(val_embedding_sub, val_embedding_group, label, 
                                                                                                                    epoch=epoch, warmup_epoch=args.warmup_epochs,
                                                                                                                    temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, )
                    
                    val_loss += val_loss_curr.item()
                    val_loss_intra += val_loss_intra_curr.item()
                    val_loss_cross += val_loss_cross_curr.item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_loss_intra = val_loss_intra / len(val_loader)
            avg_val_loss_cross = val_loss_cross / len(val_loader)

            val_losses.append(avg_val_loss)

            # if (epoch + 1) % 10 == 0 or epoch == 0:
            #     print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, "
                    f"Loss MSE={avg_train_loss_intra:.4f}, Loss Cross={avg_train_loss_cross:.4f}, "
                    f"Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")

            # Checkpoint saving and early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder_sub.state_dict(),
                    'val_loss': avg_val_loss,
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        losses = {
        'train_losses': train_losses,
        'val_losses': val_losses}
        # Final test evaluation if test set is available
        if test_metrics is not None:  # Check if test set is not empty
            print(f"\n{'='*80}")
            print(f"Subject {subject_id:02d} - Final Test Evaluation")
            print(f"{'='*80}")

            checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'))
            encoder_sub.load_state_dict(checkpoint['encoder_state_dict'])

            # Test metrics
            test_loss = 0
            test_loss_intra = 0
            test_loss_cross = 0
            encoder_sub.eval()
            encoder_group.eval()
            with torch.no_grad():
                for batch in test_loader:
                    eeg = batch['eeg'].to(device)  # 获取数据
                    label = batch['label'].to(device)
                    B, C, A, T = eeg.shape  # 添加形状提取
                    if args.embedding_high_level:
                        mask_eeg, mask = masking_strategy(eeg)
                        if args.model =='CBraMod':
                            test_embedding_sub = encoder_sub(mask_eeg)
                            test_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                            test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                        elif args.model =='Brainomni':
                            eeg = eeg.reshape(B, C, A * T)
                            mask_eeg = mask_eeg.reshape(B, C, A * T)
                            positions = position.unsqueeze(0).repeat(B, 1, 1).to(device)
                            sensor_types = sensor_type.unsqueeze(0).repeat(B, 1).to(device)
                            test_embedding_sub = encoder_sub.encode(mask_eeg, positions,sensor_types)
                            test_embedding_group = encoder_group.encode(eeg, positions,sensor_types)   
                                                     
                        test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = label_aware_contrastive_4d_channel_dynamics(test_embedding_sub, test_embedding_group, label,args, 
                                                                                                                                epoch=epoch, warmup_epoch=args.warmup_epochs,
                                                                                                                                temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, 
                                                                                                                                batch_size= B, channel=C, patch=A, dimension=T)
                    else:
                        if args.model =='BIOT':
                            mask_eeg, mask = masking_strategy(eeg)
                            eeg = eeg_projector.channel_project(eeg, ch_names)
                            eeg_mask = eeg_projector.channel_project(mask_eeg, ch_names)
                            test_embedding_sub = encoder_sub(eeg_mask)
                            test_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            mask_eeg,mask = masking_strategy(eeg)
                            test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                            test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = label_aware_contrastive_dynamics(test_embedding_sub, test_embedding_group,label, 
                                                                                                             epoch=epoch, warmup_epoch=args.warmup_epochs,
                                                                                                             temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, )
                    
                    test_loss += test_loss_curr.item()
                    test_loss_intra += test_loss_intra_curr.item()
                    test_loss_cross += test_loss_cross_curr.item()

            avg_test_loss = test_loss / len(test_loader)
            avg_test_loss_intra = test_loss_intra / len(test_loader)
            avg_test_loss_cross = test_loss_cross / len(test_loader)

            # Save metrics
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'train_loss_intra': avg_train_loss_intra,
                'train_loss_cross': avg_train_loss_cross,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'val_loss_intra': avg_val_loss_intra,
                'val_loss_cross': avg_val_loss_cross,
                'test_size': test_size,
                'test_loss': avg_test_loss,
                'test_loss_intra': avg_test_loss_intra,
                'test_loss_cross': avg_test_loss_cross,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        else:
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        # Cleanup
        torch.cuda.empty_cache()

        return metrics, losses

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'encoder_sub' in locals():
            del encoder_sub
        torch.cuda.empty_cache()

        return None, None


def train_for_subject_embedding_with_dynamics(args, subject_id, device, num_patches, dataset,encoder_group):
    """
    Train full fine-tuning reconstruction model for a single subject (V2-Fixed)

    Args:
        args: Command line arguments
        subject_id: Target subject ID
        device: CUDA device
        num_patches: Number of patches (A) for current dataset
        dataset: Pre-loaded EEGDataset instance

    Returns:
        dict: Results dictionary with metrics, or None if failed
    """
    try:
        # Create subject-specific output directory
        subject_dir = os.path.join(args.output_root, args.dataset, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")
        
        if args.multi_task:
            train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
                dataset= dataset,
                target_subject_id=subject_id,
                target_task_id=args.multi_task_ids,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        else:
            # Create dataloaders for cognitive task
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # Check minimum samples
        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        if test_size == 0:
            print("Warning: Test set is empty! Skipping test evaluation.")
            test_metrics = None  # Set test_metrics to None or some default value if needed.
        else:
            test_metrics = True  # Flag indicating that test set is not empty

########################### in chans #################################            
        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)    
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)
            
        if args.model =='Brainomni':
            position, sensor_type, valid_electrodes = extract_1020_electrode_positions(ch_names,device)    
            # print(position.shape)        
########################### in chans #################################   
        
        
        # load_vanilla_labram_weights(encoder_group, args.pretrained_path, args.model_key, args.model_prefix)
        
##################### frozen encoder_group #####################################################
        for name, param in encoder_group.named_parameters():
            param.requires_grad = False
##################### setup encoder_sub #####################################################  
####################### labram ################################################            
        if args.model =='labram_base_patch200_200':
            encoder_sub = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches
            ).to(device)
            load_vanilla_labram_weights(encoder_sub, args.pretrained_path, args.model_key, args.model_prefix)
            if args.fitune_type == 'lora':
                encoder_sub = create_lora(encoder_sub, args)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:  # 只冻结非LoRA参数
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True
#######################CbraMod ################################################                    
    
        elif args.model =='CBraMod':
            encoder_sub = CBraMod().to(device)
            encoder_sub.load_state_dict(torch.load(args.pretrained_path, map_location=device))
            # for name, module in encoder_sub.named_modules():
            #     print(f"Module Name: {name}, Module Type: {type(module)}")
            if args.fitune_type == 'lora':
                encoder_sub = apply_lora_to_model(encoder_sub, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
                # freeze_non_lora_parameters(encoder_sub)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:  # 只冻结非LoRA参数
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True
        # elif args.model == 'FEMBA':
        #     encoder_sub = FEMBA(seq_length=1280, num_channels=22, num_classes=0, embed_dim=35, num_blocks=4)    

####################### BrainOmni ################################################  
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
            encoder_sub=get_brainomni_model(args.pretrained_path).to(device)
            if args.fitune_type == 'lora':
                encoder_sub=create_lora(encoder_sub,args)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:  # 只冻结非LoRA参数
                        param.requires_grad = False
                    else:
                        param.requires_grad = True          
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True
####################### BrainOmni ################################################  
        elif args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder_sub = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)
            encoder_sub.load_state_dict(torch.load(args.pretrained_path, map_location=device))
            if args.fitune_type == 'lora':
                encoder_sub=create_lora(encoder_sub,args)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' in name.lower():
                        # LoRA参数可训练
                        param.requires_grad = True
                    else:
                        param.requires_grad = False       
            else:
                for name, param in encoder_sub.named_parameters():
                    if param.is_floating_point():  # 只设置浮点数的梯度
                        param.requires_grad = True
                    else:
                        # 整数类型参数（如index）保持requires_grad=False
                        param.requires_grad = False
                    
        for name, param in encoder_sub.named_parameters():
            if param.requires_grad:
                print(f"group_sub 可训练: {name}")
                
        for name, param in encoder_group.named_parameters():
            if param.requires_grad:
                print(f"group_encoder 可训练: {name}")
        
        total_params, trainable_params, trainable_percentage, trainable_modules = calculate_parameter_percentage_with_names(encoder_sub)
        print(f"encoder_sub Total parameters: {total_params}")
        print(f"encoder_sub Trainable parameters: {trainable_params}")
        print(f"encoder_sub Trainable parameters percentage: {trainable_percentage:.2f}%")

        trainable_params = [p for p in encoder_sub.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
      

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training ({args.fitune_type})")
        print(f"{'='*80}")
        # for name, param in encoder_sub.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        masking_strategy = Embedding_MaskingStrategy(mask_ratio=args.mask_ratio, mask_type=args.mask_type) 
        max_grad_norm = 1.0   
        # Training loop
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            encoder_sub.train()
            encoder_group.eval()
            train_loss = 0
            train_loss_intra =0
            train_loss_cross =0

            for batch in train_loader:
                # if not data.requires_grad:
                #     data = data.requires_grad_(True)
                # if not contrastive_data.requires_grad:
                #     contrastive_data = contrastive_data.requires_grad_(True)
                optimizer.zero_grad()
                eeg = batch['eeg'].to(device)
                label = batch['label'].to(device)
                B, C, A, T = eeg.shape
                   # Forward pass
                # x_patch = encoder_sub.debug_patch_embed_gradient(data)
                if args.embedding_high_level:
                    mask_eeg, mask = masking_strategy(eeg)
                    if args.model =='CBraMod':
                        embedding_mask = encoder_sub(mask_eeg)
                        embedding_truth = encoder_group(eeg)
                    elif args.model =='labram_base_patch200_200':
                        embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                        embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                    elif args.model =='Brainomni':
                        eeg = eeg.reshape(B, C, A * T)
                        mask_eeg = mask_eeg.reshape(B, C, A * T)
                        # print(B)
                        positions = position.unsqueeze(0).repeat(B, 1, 1).to(device)
                        sensor_types = sensor_type.unsqueeze(0).repeat(B, 1).to(device)
                        embedding_mask = encoder_sub.encode(mask_eeg, positions,sensor_types)
                        embedding_truth = encoder_group.encode(eeg, positions,sensor_types)


                    loss, loss_intra, loss_cross = contrastive_4d_channel_dynamics(embedding_mask, embedding_truth, label, args,
                                                                                                epoch=epoch, warmup_epoch=args.warmup_epochs,
                                                                                                temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, 
                                                                                                batch_size= B, channel=C, patch=A, dimension=T)
                else:
                    if args.model =='BIOT':
                        mask_eeg, mask = masking_strategy(eeg)
                        eeg = eeg_projector.channel_project(eeg, ch_names)
                        eeg_mask = eeg_projector.channel_project(mask_eeg, ch_names)
                        embedding_mask = encoder_sub(eeg_mask)
                        embedding_truth = encoder_group(eeg)
                    elif args.model =='labram_base_patch200_200':
                        mask_eeg,mask = masking_strategy(eeg)
                        embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                        embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans)
                        
                    loss, loss_intra, loss_cross = contrastive_dynamics(embedding_mask, embedding_truth, label, 
                                                                                    epoch=epoch, warmup_epoch=args.warmup_epochs,
                                                                           temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha)
                    
                loss.backward()
                utils.clip_grad_norm_(encoder_sub.parameters(), max_grad_norm)
                optimizer.step()

                # Track the different losses separately
                train_loss += loss.item()
                train_loss_intra += loss_intra.item()
                train_loss_cross += loss_cross.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_train_loss_intra = train_loss_intra / len(train_loader)
            avg_train_loss_cross = train_loss_cross / len(train_loader)
            
            train_losses.append(avg_train_loss)

            encoder_sub.eval()
            encoder_group.eval()
            val_loss = 0
            val_loss_intra = 0
            val_loss_cross = 0
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch['eeg'].to(device)  # 获取数据
                    label = batch['label'].to(device)
                    B, C, A, T = eeg.shape  # 添加形状提取
                    if args.embedding_high_level:
                        mask_eeg, mask = masking_strategy(eeg)
                        if args.model =='CBraMod':
                            val_embedding_sub = encoder_sub(mask_eeg)
                            val_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                            val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                            
                        elif args.model =='Brainomni':
                            eeg = eeg.reshape(B, C, A * T)
                            mask_eeg = mask_eeg.reshape(B, C, A * T)
                            positions = position.unsqueeze(0).repeat(B, 1, 1).to(device)
                            sensor_types = sensor_type.unsqueeze(0).repeat(B, 1).to(device)
                            val_embedding_sub = encoder_sub.encode(mask_eeg, positions,sensor_types)
                            val_embedding_group = encoder_group.encode(eeg, positions,sensor_types)
                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = contrastive_4d_channel_dynamics(val_embedding_sub, val_embedding_group, label, args,
                                                                                                                     epoch=epoch, warmup_epoch=args.warmup_epochs,
                                                                                                                     temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, 
                                                                                                                     batch_size= B, channel=C, patch=A, dimension=T)
                    # if args.embedding_high_level:
                    #     mask_eeg, mask = masking_strategy(eeg)
                    #     val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans, return_patch_tokens=True)
                    #     val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans, return_patch_tokens=True)
                    #     val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive_4d_channel(val_embedding_sub, val_embedding_group, label, temperature=0.07, batch_size=B, channel=C, patch=A, dimension=T)
                    else:
                        if args.model =='BIOT':
                            mask_eeg, mask = masking_strategy(eeg)
                            eeg = eeg_projector.channel_project(eeg, ch_names)
                            eeg_mask = eeg_projector.channel_project(mask_eeg, ch_names)
                            val_embedding_sub = encoder_sub(eeg_mask)
                            val_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            mask_eeg, mask = masking_strategy(eeg)
                            val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                            val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = contrastive_dynamics(val_embedding_sub, val_embedding_group, label, 
                                                                                                                    epoch=epoch, warmup_epoch=args.warmup_epochs,
                                                                                                                    temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, )
                    
                    val_loss += val_loss_curr.item()
                    val_loss_intra += val_loss_intra_curr.item()
                    val_loss_cross += val_loss_cross_curr.item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_loss_intra = val_loss_intra / len(val_loader)
            avg_val_loss_cross = val_loss_cross / len(val_loader)

            val_losses.append(avg_val_loss)

            # if (epoch + 1) % 10 == 0 or epoch == 0:
            #     print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, "
                    f"Loss MSE={avg_train_loss_intra:.4f}, Loss Cross={avg_train_loss_cross:.4f}, "
                    f"Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")

            # Checkpoint saving and early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder_sub.state_dict(),
                    'val_loss': avg_val_loss,
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        losses = {
        'train_losses': train_losses,
        'val_losses': val_losses}
        # Final test evaluation if test set is available
        if test_metrics is not None:  # Check if test set is not empty
            print(f"\n{'='*80}")
            print(f"Subject {subject_id:02d} - Final Test Evaluation")
            print(f"{'='*80}")

            checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'))
            encoder_sub.load_state_dict(checkpoint['encoder_state_dict'])

            # Test metrics
            test_loss = 0
            test_loss_intra = 0
            test_loss_cross = 0
            encoder_sub.eval()
            encoder_group.eval()
            with torch.no_grad():
                for batch in test_loader:
                    eeg = batch['eeg'].to(device)  # 获取数据
                    label = batch['label'].to(device)
                    B, C, A, T = eeg.shape  # 添加形状提取
                    if args.embedding_high_level:
                        mask_eeg, mask = masking_strategy(eeg)
                        if args.model =='CBraMod':
                            test_embedding_sub = encoder_sub(mask_eeg)
                            test_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                            test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                        elif args.model =='Brainomni':
                            eeg = eeg.reshape(B, C, A * T)
                            mask_eeg = mask_eeg.reshape(B, C, A * T)
                            positions = position.unsqueeze(0).repeat(B, 1, 1).to(device)
                            sensor_types = sensor_type.unsqueeze(0).repeat(B, 1).to(device)
                            test_embedding_sub = encoder_sub.encode(mask_eeg, positions,sensor_types)
                            test_embedding_group = encoder_group.encode(eeg, positions,sensor_types)   
                                                     
                        test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = contrastive_4d_channel_dynamics(test_embedding_sub, test_embedding_group, label,args, 
                                                                                                                                epoch=epoch, warmup_epoch=args.warmup_epochs,
                                                                                                                                temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, 
                                                                                                                                batch_size= B, channel=C, patch=A, dimension=T)
                    else:
                        if args.model =='BIOT':
                            mask_eeg, mask = masking_strategy(eeg)
                            eeg = eeg_projector.channel_project(eeg, ch_names)
                            eeg_mask = eeg_projector.channel_project(mask_eeg, ch_names)
                            test_embedding_sub = encoder_sub(eeg_mask)
                            test_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            mask_eeg,mask = masking_strategy(eeg)
                            test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                            test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = contrastive_dynamics(test_embedding_sub, test_embedding_group,label, 
                                                                                                             epoch=epoch, warmup_epoch=args.warmup_epochs,
                                                                                                             temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, )
                    
                    test_loss += test_loss_curr.item()
                    test_loss_intra += test_loss_intra_curr.item()
                    test_loss_cross += test_loss_cross_curr.item()

            avg_test_loss = test_loss / len(test_loader)
            avg_test_loss_intra = test_loss_intra / len(test_loader)
            avg_test_loss_cross = test_loss_cross / len(test_loader)

            # Save metrics
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'train_loss_intra': avg_train_loss_intra,
                'train_loss_cross': avg_train_loss_cross,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'val_loss_intra': avg_val_loss_intra,
                'val_loss_cross': avg_val_loss_cross,
                'test_size': test_size,
                'test_loss': avg_test_loss,
                'test_loss_intra': avg_test_loss_intra,
                'test_loss_cross': avg_test_loss_cross,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        else:
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        # Cleanup
        torch.cuda.empty_cache()

        return metrics, losses

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'encoder_sub' in locals():
            del encoder_sub
        torch.cuda.empty_cache()

        return None, None



def train_for_subject_embedding_task_transder(args, subject_id, device, num_patches, dataset,encoder_group):
    """
    Train full fine-tuning reconstruction model for a single subject (V2-Fixed)

    Args:
        args: Command line arguments
        subject_id: Target subject ID
        device: CUDA device
        num_patches: Number of patches (A) for current dataset
        dataset: Pre-loaded EEGDataset instance

    Returns:
        dict: Results dictionary with metrics, or None if failed
    """
    try:
        # Create subject-specific output directory
        subject_dir = os.path.join(args.output_root, args.dataset, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")
        
        if args.multi_task:
            train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
                dataset= dataset,
                target_subject_id=subject_id,
                target_task_id=args.multi_task_ids,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        else:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # Check minimum samples
        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        if test_size == 0:
            print("Warning: Test set is empty! Skipping test evaluation.")
            test_metrics = None  # Set test_metrics to None or some default value if needed.
        else:
            test_metrics = True  # Flag indicating that test set is not empty

        in_chans, ch_names, chan_number = get_dataset_name(args)
        print('in_chans', in_chans)
        
        
        encoder_group = encoder_group.to(device)
        for name, param in encoder_group.named_parameters():
            param.requires_grad = False
            
        if args.model =='labram_base_patch200_200':
            encoder_sub = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches
            ).to(device)
            load_vanilla_labram_weights(encoder_sub, args.pretrained_path, args.model_key, args.model_prefix)
            if args.fitune_type == 'lora':
                encoder_sub = create_lora(encoder_sub, args)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:  # 只冻结非LoRA参数
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True
                    
    
        elif args.model =='CBraMod':
            encoder_sub = CBraMod().to(device)
            encoder_sub.load_state_dict(torch.load(args.pretrained_path, map_location=device))
            # for name, module in encoder_sub.named_modules():
            #     print(f"Module Name: {name}, Module Type: {type(module)}")
            if args.fitune_type == 'lora':
                encoder_sub = apply_lora_to_model(encoder_sub, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
                # freeze_non_lora_parameters(encoder_sub)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:  # 只冻结非LoRA参数
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True

        for name, param in encoder_sub.named_parameters():
            if param.requires_grad:
                print(f"group_sub 可训练: {name}")
                
        for name, param in encoder_group.named_parameters():
            if param.requires_grad:
                print(f"group_encoder 可训练: {name}")
        
        total_params, trainable_params, trainable_percentage, trainable_modules = calculate_parameter_percentage_with_names(encoder_sub)
        print(f"encoder_sub Total parameters: {total_params}")
        print(f"encoder_sub Trainable parameters: {trainable_params}")
        print(f"encoder_sub Trainable parameters percentage: {trainable_percentage:.2f}%")

        trainable_params = [p for p in encoder_sub.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
      

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training ({args.fitune_type})")
        print(f"{'='*80}")
        masking_strategy = Embedding_MaskingStrategy(mask_ratio=args.mask_ratio, mask_type=args.mask_type)    
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            encoder_sub.train()
            encoder_group.eval()
            train_loss = 0
            train_loss_intra =0
            train_loss_cross =0

            for batch in train_loader:
                optimizer.zero_grad()
                eeg = batch['eeg'].to(device)
                label = batch['label'].to(device)
                B, C, A, T = eeg.shape
                label = labels_to_task_id(label, device)
                   # Forward pass
                # x_patch = encoder_sub.debug_patch_embed_gradient(data)
                if args.embedding_high_level:
                    mask_eeg, mask = masking_strategy(eeg)
                    if args.model =='CBraMod':
                        embedding_mask = encoder_sub(mask_eeg)
                        embedding_truth = encoder_group(eeg)
                    elif args.model =='labram_base_patch200_200':
                        embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                        embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
    
                    loss, loss_intra, loss_cross = label_aware_contrastive_4d_channel(embedding_mask, embedding_truth, label, args, 
                                                                                      temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, 
                                                                                      batch_size= B, channel=C, patch=A, dimension=T)
                else:
                    mask_eeg,mask = masking_strategy(eeg)
                    embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                    embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans)
                    loss, loss_intra, loss_cross = label_aware_contrastive(embedding_mask, embedding_truth, label, temperature=0.07)
                    
                loss.backward()
                optimizer.step()

                # Track the different losses separately
                train_loss += loss.item()
                train_loss_intra += loss_intra.item()
                train_loss_cross += loss_cross.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_train_loss_intra = train_loss_intra / len(train_loader)
            avg_train_loss_cross = train_loss_cross / len(train_loader)
            
            train_losses.append(avg_train_loss)

            # # Validation loss
            # encoder_sub.eval()
            # encoder_group.eval()
            # val_loss = 0
            # with torch.no_grad():
            #     for val_data, val_contrastive_data,label, contrastive_label in val_loader:
            #         val_data = val_data.to(device)
            #         val_contrastive_data = val_contrastive_data.to(device)
            #         val_embedding_sub = encoder_sub.forward_features(val_data, input_chans=in_chans)
            #         val_embedding_group = encoder_group.forward_features(val_contrastive_data, input_chans=in_chans)
            #         val_loss += info_nce_loss_loop(val_embedding_sub, val_embedding_group, temperature=0.1).item()

            # avg_val_loss = val_loss / len(val_loader)
            # val_losses.append(avg_val_loss)
            # Validation loss
            # Validation loss
            encoder_sub.eval()
            encoder_group.eval()
            val_loss = 0
            val_loss_intra = 0
            val_loss_cross = 0
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch['eeg'].to(device)  # 获取数据
                    label = batch['label'].to(device)
                    label = labels_to_task_id(label, device)
                    B, C, A, T = eeg.shape  # 添加形状提取
                    if args.embedding_high_level:
                        mask_eeg, mask = masking_strategy(eeg)
                        if args.model =='CBraMod':
                            val_embedding_sub = encoder_sub(mask_eeg)
                            val_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                            val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive_4d_channel(val_embedding_sub, val_embedding_group, label, args,
                                                                                                                     temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, 
                                                                                                                     batch_size= B, channel=C, patch=A, dimension=T)
                    # if args.embedding_high_level:
                    #     mask_eeg, mask = masking_strategy(eeg)
                    #     val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans, return_patch_tokens=True)
                    #     val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans, return_patch_tokens=True)
                    #     val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive_4d_channel(val_embedding_sub, val_embedding_group, label, temperature=0.07, batch_size=B, channel=C, patch=A, dimension=T)
                    else:
                        mask_eeg, mask = masking_strategy(eeg)
                        val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                        val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive(val_embedding_sub, val_embedding_group, label, temperature=0.07)
                    
                    val_loss += val_loss_curr.item()
                    val_loss_intra += val_loss_intra_curr.item()
                    val_loss_cross += val_loss_cross_curr.item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_loss_intra = val_loss_intra / len(val_loader)
            avg_val_loss_cross = val_loss_cross / len(val_loader)

            val_losses.append(avg_val_loss)

            # if (epoch + 1) % 10 == 0 or epoch == 0:
            #     print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, "
                    f"Loss MSE={avg_train_loss_intra:.4f}, Loss Cross={avg_train_loss_cross:.4f}, "
                    f"Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")

            # Checkpoint saving and early stopping
            if avg_val_loss < best_val_loss-1e-4:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder_sub.state_dict(),
                    'val_loss': avg_val_loss,
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        losses = {
        'train_losses': train_losses,
        'val_losses': val_losses}
        # Final test evaluation if test set is available
        if test_metrics is not None:  # Check if test set is not empty
            print(f"\n{'='*80}")
            print(f"Subject {subject_id:02d} - Final Test Evaluation")
            print(f"{'='*80}")

            checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'))
            encoder_sub.load_state_dict(checkpoint['encoder_state_dict'])

            # Test metrics
            test_loss = 0
            test_loss_intra = 0
            test_loss_cross = 0
            encoder_sub.eval()
            encoder_group.eval()
            with torch.no_grad():
                for batch in test_loader:
                    eeg = batch['eeg'].to(device)  # 获取数据
                    label = batch['label'].to(device)
                    label = labels_to_task_id(label, device)
                    B, C, A, T = eeg.shape  # 添加形状提取
                    if args.embedding_high_level:
                        mask_eeg, mask = masking_strategy(eeg)
                        if args.model =='CBraMod':
                            test_embedding_sub = encoder_sub(mask_eeg)
                            test_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                            test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                        test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = label_aware_contrastive_4d_channel(test_embedding_sub, test_embedding_group, label,args, 
                                                                                                                        temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha,
                                                                                                                        batch_size= B, channel=C, patch=A, dimension=T)
                    else:
                        mask_eeg,mask = masking_strategy(eeg)
                        test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                        test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = label_aware_contrastive(test_embedding_sub, test_embedding_group,label, temperature=0.07)
                    
                    test_loss += test_loss_curr.item()
                    test_loss_intra += test_loss_intra_curr.item()
                    test_loss_cross += test_loss_cross_curr.item()

            avg_test_loss = test_loss / len(test_loader)
            avg_test_loss_intra = test_loss_intra / len(test_loader)
            avg_test_loss_cross = test_loss_cross / len(test_loader)

            # Save metrics
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'train_loss_intra': avg_train_loss_intra,
                'train_loss_cross': avg_train_loss_cross,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'val_loss_intra': avg_val_loss_intra,
                'val_loss_cross': avg_val_loss_cross,
                'test_size': test_size,
                'test_loss': avg_test_loss,
                'test_loss_intra': avg_test_loss_intra,
                'test_loss_cross': avg_test_loss_cross,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        else:
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        # Cleanup
        torch.cuda.empty_cache()

        return metrics, losses

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'encoder_sub' in locals():
            del encoder_sub
        torch.cuda.empty_cache()

        return None, None








##################################################################################################
############################################# 对比学习+ MAE #######################################
def train_for_subject_embedding_mask(args, subject_id, device, num_patches, dataset,encoder_group):
    """
    Train full fine-tuning reconstruction model for a single subject (V2-Fixed)

    Args:
        args: Command line arguments
        subject_id: Target subject ID
        device: CUDA device
        num_patches: Number of patches (A) for current dataset
        dataset: Pre-loaded EEGDataset instance

    Returns:
        dict: Results dictionary with metrics, or None if failed
    """
    try:
        # Create subject-specific output directory
        subject_dir = os.path.join(args.output_root, args.dataset, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")
        
        if args.multi_task:
            train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
                dataset= dataset,
                target_subject_id=subject_id,
                target_task_id=args.multi_task_ids,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # Check minimum samples
        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        if test_size == 0:
            print("Warning: Test set is empty! Skipping test evaluation.")
            test_metrics = None  # Set test_metrics to None or some default value if needed.
        else:
            test_metrics = True  # Flag indicating that test set is not empty

        in_chans, ch_names, chan_number = get_dataset_name(args)
        print('in_chans', in_chans)
        
        
        # encoder_group = encoder_group.to(device)
        for name, param in encoder_group.named_parameters():
            param.requires_grad = False
        
        if args.model =='labram_base_patch200_200':
            encoder_sub = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches
            ).to(device)
            
            load_vanilla_labram_weights(encoder_sub, args.pretrained_path, args.model_key, args.model_prefix)
            if args.fitune_type == 'lora':
                encoder_sub = create_lora(encoder_sub, args)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:  # 只冻结非LoRA参数
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True
                    
    
        elif args.model =='CBraMod':
            encoder_sub = CBraMod().to(device)
            encoder_sub.load_state_dict(torch.load(args.pretrained_path, map_location=device))
 
            if args.fitune_type == 'lora':
                encoder_sub = apply_lora_to_model(encoder_sub, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:  
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True
            

        for name, param in encoder_sub.named_parameters():
            if param.requires_grad:
                print(f"group_sub 可训练: {name}")
                
        for name, param in encoder_group.named_parameters():
            if param.requires_grad:
                print(f"group_encoder 可训练: {name}")
        
        total_params, trainable_params, trainable_percentage, trainable_modules = calculate_parameter_percentage_with_names(encoder_sub)
        print(f"encoder_sub Total parameters: {total_params}")
        print(f"encoder_sub Trainable parameters: {trainable_params}")
        print(f"encoder_sub Trainable parameters percentage: {trainable_percentage:.2f}%")

        trainable_params = [p for p in encoder_sub.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
      
        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training ({args.fitune_type})")
        print(f"{'='*80}")

        masking_strategy = Embedding_MaskingStrategy(mask_ratio=args.mask_ratio, mask_type=args.mask_type)    
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            encoder_sub.train()
            encoder_group.eval()
            train_loss = 0
            train_loss_intra =0
            train_loss_cross =0

            for batch in train_loader:
                optimizer.zero_grad()
                eeg = batch['eeg'].to(device)
                label = batch['label'].to(device)
                B, C, A, T = eeg.shape
                if args.embedding_high_level:
                    mask_eeg, mask = masking_strategy(eeg)
                    if args.model =='CBraMod':
                        embedding_mask = encoder_sub(mask_eeg)
                        embedding_truth = encoder_group(eeg)
                    elif args.model =='labram_base_patch200_200':
                        embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                        embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
    # subject_embeddings, group_embeddings, mask, labels, args,
                    loss = label_aware_contrastive_4d_channel_mask(embedding_mask, embedding_truth, mask, label, args, temperature=0.07, alpha=0, batch_size= B, channel=C, patch=A, dimension=T)
                else:
                    mask_eeg,mask = masking_strategy(eeg)
                    embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                    embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans)
                    loss = label_aware_contrastive_mask(embedding_mask, embedding_truth, label, temperature=0.07)
                    
                loss.backward()
                optimizer.step()

                # Track the different losses separately
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            train_losses.append(avg_train_loss)

################################ validation #####################################################################
            encoder_sub.eval()
            encoder_group.eval()
            val_loss = 0
            val_loss_intra = 0
            val_loss_cross = 0
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch['eeg'].to(device)  # 获取数据
                    label = batch['label'].to(device)
                    B, C, A, T = eeg.shape  # 添加形状提取
                    if args.embedding_high_level:
                        mask_eeg, mask = masking_strategy(eeg)
                        if args.model =='CBraMod':
                            val_embedding_sub = encoder_sub(mask_eeg)
                            val_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                            val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                        val_loss_curr = label_aware_contrastive_4d_channel_mask(val_embedding_sub, val_embedding_group, mask, label, args, 
                                                                           temperature=0.07, alpha=0, batch_size= B, channel=C, patch=A, dimension=T)

                    else:
                        mask_eeg, mask = masking_strategy(eeg)
                        val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                        val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        val_loss_curr = label_aware_contrastive_mask(val_embedding_sub, val_embedding_group, label, temperature=0.07)
                        
                    val_loss += val_loss_curr.item()


            avg_val_loss = val_loss / len(val_loader)

            val_losses.append(avg_val_loss)


            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, "
                    f"Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")

            # Checkpoint saving and early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder_sub.state_dict(),
                    'val_loss': avg_val_loss,
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        losses = {
        'train_losses': train_losses,
        'val_losses': val_losses}
        
        # Final test evaluation if test set is available
        if test_metrics is not None:  # Check if test set is not empty
            print(f"\n{'='*80}")
            print(f"Subject {subject_id:02d} - Final Test Evaluation")
            print(f"{'='*80}")
            
            checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'))
            encoder_sub.load_state_dict(checkpoint['encoder_state_dict'])

            # Test metrics
            test_loss = 0
            test_loss_intra = 0
            test_loss_cross = 0
            encoder_sub.eval()
            encoder_group.eval()
            with torch.no_grad():
                for batch in test_loader:
                    eeg = batch['eeg'].to(device)  # 获取数据
                    label = batch['label'].to(device)
                    B, C, A, T = eeg.shape  # 添加形状提取
                    if args.embedding_high_level:
                        mask_eeg, mask = masking_strategy(eeg)
                        if args.model =='CBraMod':
                            test_embedding_sub = encoder_sub(mask_eeg)
                            test_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                            test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                        test_loss_curr = label_aware_contrastive_4d_channel_mask(test_embedding_sub, test_embedding_group, mask, label, args, 
                                                                                temperature=0.07, alpha =0, batch_size= B, channel=C, patch=A, dimension=T)
                    else:
                        mask_eeg,mask = masking_strategy(eeg)
                        test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                        test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        test_loss_curr = label_aware_contrastive_mask(test_embedding_sub, test_embedding_group, label, temperature=0.07)
                    
                    test_loss += test_loss_curr.item()

            avg_test_loss = test_loss / len(test_loader)


            # Save metrics
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'test_size': test_size,
                'test_loss': avg_test_loss,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        else:
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        # Cleanup
        torch.cuda.empty_cache()

        return metrics, losses

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'encoder_sub' in locals():
            del encoder_sub
        torch.cuda.empty_cache()

        return None, None

















    
def train_for_subject_embedding_multitask(args, subject_id, device, num_patches, dataset,encoder_group):
    try:
        # Create subject-specific output directory
        subject_dir = os.path.join(args.output_root, args.dataset, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")
        
        if args.multi_task:
            train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
                dataset= dataset,
                target_subject_id=subject_id,
                target_task_id=args.target_task_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        elif  args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        else:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # Check minimum samples
        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        if test_size == 0:
            print("Warning: Test set is empty! Skipping test evaluation.")
            test_metrics = None  # Set test_metrics to None or some default value if needed.
        else:
            test_metrics = True  # Flag indicating that test set is not empty

        in_chans, ch_names, chan_number = get_dataset_name(args)
        print('in_chans', in_chans)
        
        # load_vanilla_labram_weights(encoder_group, args.pretrained_path, args.model_key, args.model_prefix)
        
        # encoder_group = encoder_group.to(device)
        for name, param in encoder_group.named_parameters():
            param.requires_grad = False
        if args.model =='labram_base_patch200_200':
            encoder_sub = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches
            ).to(device)
            load_vanilla_labram_weights(encoder_sub, args.pretrained_path, args.model_key, args.model_prefix)
            if args.fitune_type == 'lora':
                encoder_sub = create_lora(encoder_sub, args)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:  # 只冻结非LoRA参数
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True
                    
    
        elif args.model =='CBraMod':
            encoder_sub = CBraMod().to(device)
            encoder_sub.load_state_dict(torch.load(args.pretrained_path, map_location=device))
            # for name, module in encoder_sub.named_modules():
            #     print(f"Module Name: {name}, Module Type: {type(module)}")
            if args.fitune_type == 'lora':
                encoder_sub = apply_lora_to_model(encoder_sub, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
                # freeze_non_lora_parameters(encoder_sub)
                for name, param in encoder_sub.named_parameters():
                    if 'lora' not in name:  # 只冻结非LoRA参数
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for name, param in encoder_sub.named_parameters():
                    param.requires_grad = True
            
        for name, param in encoder_sub.named_parameters():
            if param.requires_grad:
                print(f"group_sub 可训练: {name}")
                
        for name, param in encoder_group.named_parameters():
            if param.requires_grad:
                print(f"group_encoder 可训练: {name}")

        total_params, trainable_params, trainable_percentage, trainable_modules = calculate_parameter_percentage_with_names(encoder_sub)
        print(f"encoder_sub Total parameters: {total_params}")
        print(f"encoder_sub Trainable parameters: {trainable_params}")
        print(f"encoder_sub Trainable parameters percentage: {trainable_percentage:.2f}%")


        trainable_params = [p for p in encoder_sub.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
      

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training ({args.fitune_type})")
        print(f"{'='*80}")
        # for name, param in encoder_sub.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        masking_strategy = Embedding_MaskingStrategy(mask_ratio=args.mask_ratio, mask_type=args.mask_type)    
        # Training loop
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr)
        
        # for epoch in range(args.epochs):
        #     if epoch < args.warmup_epochs:
        #         lr_mult = (epoch + 1) / args.warmup_epochs
        #         current_lr = args.lr * lr_mult
        #     else:
        #         scheduler.step()
        #         # progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        #         # current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = current_lr
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                # Warm-up 阶段：线性增长学习率
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
                # 更新学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                # 在 warmup 结束后，使用 CosineAnnealingLR 更新学习率
                scheduler.step()
                
            encoder_sub.train()
            encoder_group.eval()
            train_loss = 0
            train_loss_intra =0
            train_loss_cross =0

            for batch in train_loader:
                # if not data.requires_grad:
                #     data = data.requires_grad_(True)
                # if not contrastive_data.requires_grad:
                #     contrastive_data = contrastive_data.requires_grad_(True)
                optimizer.zero_grad()
                eeg = batch['eeg'].to(device)
                label = batch['label'].to(device)
                B, C, A, T = eeg.shape
                   # Forward pass
                # x_patch = encoder_sub.debug_patch_embed_gradient(data)
                if args.embedding_high_level:
                    mask_eeg, mask = masking_strategy(eeg)
                    if args.model =='CBraMod':
                        embedding_mask = encoder_sub(mask_eeg)
                        embedding_truth = encoder_group(eeg)
                    elif args.model =='labram_base_patch200_200':
                        embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                        embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                    loss, loss_intra, loss_cross = label_aware_contrastive_4d_channel(embedding_mask, embedding_truth, label, args,temperature=0.07, batch_size= B, channel=C, patch=A, dimension=T)
                else:
                    mask_eeg,mask = masking_strategy(eeg)
                    embedding_mask = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                    embedding_truth = encoder_group.forward_features(eeg, input_chans=in_chans)
                    loss, loss_intra, loss_cross = label_aware_contrastive(embedding_mask, embedding_truth, label, temperature=0.07)
                    
                loss.backward()
                optimizer.step()

                # Track the different losses separately
                train_loss += loss.item()
                train_loss_intra += loss_intra.item()
                train_loss_cross += loss_cross.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_train_loss_intra = train_loss_intra / len(train_loader)
            avg_train_loss_cross = train_loss_cross / len(train_loader)
            
            train_losses.append(avg_train_loss)

            # # Validation loss
            # encoder_sub.eval()
            # encoder_group.eval()
            # val_loss = 0
            # with torch.no_grad():
            #     for val_data, val_contrastive_data,label, contrastive_label in val_loader:
            #         val_data = val_data.to(device)
            #         val_contrastive_data = val_contrastive_data.to(device)
            #         val_embedding_sub = encoder_sub.forward_features(val_data, input_chans=in_chans)
            #         val_embedding_group = encoder_group.forward_features(val_contrastive_data, input_chans=in_chans)
            #         val_loss += info_nce_loss_loop(val_embedding_sub, val_embedding_group, temperature=0.1).item()

            # avg_val_loss = val_loss / len(val_loader)
            # val_losses.append(avg_val_loss)
            # Validation loss
            # Validation loss
            encoder_sub.eval()
            encoder_group.eval()
            val_loss = 0
            val_loss_intra = 0
            val_loss_cross = 0
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch['eeg'].to(device)  # 获取数据
                    label = batch['label'].to(device)
                    B, C, A, T = eeg.shape  # 添加形状提取
                    if args.embedding_high_level:
                        mask_eeg, mask = masking_strategy(eeg)
                        if args.model =='CBraMod':
                            val_embedding_sub = encoder_sub(mask_eeg)
                            val_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                            val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive_4d_channel(val_embedding_sub, val_embedding_group, label, args,temperature=0.07, batch_size= B, channel=C, patch=A, dimension=T)
                    # if args.embedding_high_level:
                    #     mask_eeg, mask = masking_strategy(eeg)
                    #     val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans, return_patch_tokens=True)
                    #     val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans, return_patch_tokens=True)
                    #     val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive_4d_channel(val_embedding_sub, val_embedding_group, label, temperature=0.07, batch_size=B, channel=C, patch=A, dimension=T)
                    else:
                        mask_eeg, mask = masking_strategy(eeg)
                        val_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                        val_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = label_aware_contrastive(val_embedding_sub, val_embedding_group, label, temperature=0.07)
                    
                    val_loss += val_loss_curr.item()
                    val_loss_intra += val_loss_intra_curr.item()
                    val_loss_cross += val_loss_cross_curr.item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_loss_intra = val_loss_intra / len(val_loader)
            avg_val_loss_cross = val_loss_cross / len(val_loader)

            val_losses.append(avg_val_loss)

            # if (epoch + 1) % 10 == 0 or epoch == 0:
            #     print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, "
                    f"Loss MSE={avg_train_loss_intra:.4f}, Loss Cross={avg_train_loss_cross:.4f}, "
                    f"Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")

            # Checkpoint saving and early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder_sub.state_dict(),
                    'val_loss': avg_val_loss,
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        losses = {
        'train_losses': train_losses,
        'val_losses': val_losses}
        # Final test evaluation if test set is available
        if test_metrics is not None:  # Check if test set is not empty
            print(f"\n{'='*80}")
            print(f"Subject {subject_id:02d} - Final Test Evaluation")
            print(f"{'='*80}")

            checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'))
            encoder_sub.load_state_dict(checkpoint['encoder_state_dict'])

            # Test metrics
            test_loss = 0
            test_loss_intra = 0
            test_loss_cross = 0
            encoder_sub.eval()
            encoder_group.eval()
            with torch.no_grad():
                for batch in test_loader:
                    eeg = batch['eeg'].to(device)  # 获取数据
                    label = batch['label'].to(device)
                    B, C, A, T = eeg.shape  # 添加形状提取
                    if args.embedding_high_level:
                        mask_eeg, mask = masking_strategy(eeg)
                        if args.model =='CBraMod':
                            test_embedding_sub = encoder_sub(mask_eeg)
                            test_embedding_group = encoder_group(eeg)
                        elif args.model =='labram_base_patch200_200':
                            test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans,return_patch_tokens=True)
                            test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans,return_patch_tokens=True)
                        test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = label_aware_contrastive_4d_channel(test_embedding_sub, test_embedding_group, label,args, temperature=0.07, batch_size= B, channel=C, patch=A, dimension=T)
                    else:
                        mask_eeg,mask = masking_strategy(eeg)
                        test_embedding_sub = encoder_sub.forward_features(mask_eeg, input_chans=in_chans)
                        test_embedding_group = encoder_group.forward_features(eeg, input_chans=in_chans)
                        test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = label_aware_contrastive(test_embedding_sub, test_embedding_group,label, temperature=0.07)
                    
                    test_loss += test_loss_curr.item()
                    test_loss_intra += test_loss_intra_curr.item()
                    test_loss_cross += test_loss_cross_curr.item()

            avg_test_loss = test_loss / len(test_loader)
            avg_test_loss_intra = test_loss_intra / len(test_loader)
            avg_test_loss_cross = test_loss_cross / len(test_loader)

            # Save metrics
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'train_loss_intra': avg_train_loss_intra,
                'train_loss_cross': avg_train_loss_cross,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'val_loss_intra': avg_val_loss_intra,
                'val_loss_cross': avg_val_loss_cross,
                'test_size': test_size,
                'test_loss': avg_test_loss,
                'test_loss_intra': avg_test_loss_intra,
                'test_loss_cross': avg_test_loss_cross,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        else:
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        # Cleanup
        torch.cuda.empty_cache()

        return metrics, losses

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'encoder_sub' in locals():
            del encoder_sub
        torch.cuda.empty_cache()

        return None, None


##############################################################################################################################
def train_Contrastiv_for_subject_2loop(args, subject_id, device, num_patches, dataset, encoder_sub,encoder_group):
    """
    Train full fine-tuning reconstruction model for a single subject (V2-Fixed)

    Args:
        args: Command line arguments
        subject_id: Target subject ID
        device: CUDA device
        num_patches: Number of patches (A) for current dataset
        dataset: Pre-loaded EEGDataset instance

    Returns:
        dict: Results dictionary with metrics, or None if failed
    """
    try:
        # Create subject-specific output directory
        subject_dir = os.path.join(args.output_root, args.dataset, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")
        
        # Load data
        train_loader, val_loader, test_loader = create_contrastive_dataloaders(
            dataset=dataset,
            target_subject_id=subject_id,
            train_size=args.train_size,
            test_size=args.test_size,
            val_size=args.val_size,
            batch_size=args.batch_size,
            random_state=args.seed,
            num_workers=args.num_workers
        )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # Check minimum samples
        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        if test_size == 0:
            print("Warning: Test set is empty! Skipping test evaluation.")
            test_metrics = None  # Set test_metrics to None or some default value if needed.
        else:
            test_metrics = True  # Flag indicating that test set is not empty

        in_chans, ch_names, chan_number = get_dataset_name(args)
        print('in_chans', in_chans)
        
        encoder_group = encoder_group.to(device)
        for name, param in encoder_group.named_parameters():
            param.requires_grad = False
        
        encoder_sub = create_model(
            args.model,
            pretrained=False,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            qkv_bias=args.qkv_bias,
            num_patches_per_channel_input=num_patches,
        )   
        load_vanilla_labram_weights(encoder_sub, args.pretrained_path, args.model_key, args.model_prefix)

        # lora_config = LoraConfig(
        #     r=args.lora_r,
        #     lora_alpha=args.lora_alpha,
        #     lora_dropout=args.lora_dropout,
        #     target_modules=args.target_modules,
        # )
        # encoder_sub = get_peft_model(encoder_sub, lora_config)
        encoder_sub = encoder_sub.to(device)
        for name, param in encoder_sub.named_parameters():
            param.requires_grad = True

        # 正确做法：只冻结base_model，不解冻LoRA
        # for name, param in encoder_sub.named_parameters():
        #     if 'lora' not in name:  # 只冻结非LoRA参数
        #         param.requires_grad = False
        #     else:
        #         param.requires_grad = True
        # Set up optimizer for only trainable parameters (LoRA layers)
        for name, param in encoder_sub.named_parameters():
            if param.requires_grad:
                print(f"group_sub 可训练: {name}")
                
        for name, param in encoder_group.named_parameters():
            if param.requires_grad:
                print(f"group_encoder 可训练: {name}")
        

        # Parameter details
        # total_params, trainable_params, trainable_percentage, trainable_modules = calculate_parameter_percentage_with_names(encoder_group)
        # print(f"encoder_group Total parameters: {total_params}")
        # print(f"encoder_group Trainable parameters: {trainable_params}")
        # print(f"encoder_group Trainable parameters percentage: {trainable_percentage:.2f}%")
        
        total_params, trainable_params, trainable_percentage, trainable_modules = calculate_parameter_percentage_with_names(encoder_sub)
        print(f"encoder_sub Total parameters: {total_params}")
        print(f"encoder_sub Trainable parameters: {trainable_params}")
        print(f"encoder_sub Trainable parameters percentage: {trainable_percentage:.2f}%")
        # for name, param in encoder_sub.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        # for name, param in encoder_sub.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         print(f"{name} 有梯度")
        #     elif param.requires_grad and param.grad is None:
        #         print(f"警告: {name} 可训练但梯度为None")

        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder_sub.parameters()), lr=args.lr)
        trainable_params = [p for p in encoder_sub.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
      

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training ({args.fitune_type})")
        print(f"{'='*80}")
        # for name, param in encoder_sub.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
            
        # Training loop
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            encoder_sub.train()
            encoder_group.eval()
            train_loss = 0
            train_loss_intra =0
            train_loss_cross =0

            for data, contrastive_data, label, contrastive_label in train_loader:
                if not data.requires_grad:
                    data = data.requires_grad_(True)
                if not contrastive_data.requires_grad:
                    contrastive_data = contrastive_data.requires_grad_(True)

                B, C, A, T = data.shape
                optimizer.zero_grad()
                data = data.to(device)
                contrastive_data = contrastive_data.to(device)
                       
                # Forward pass
                # x_patch = encoder_sub.debug_patch_embed_gradient(data)
                if args.embedding_high_level:
                    embedding_sub = encoder_sub.forward_features(data, input_chans=in_chans,return_patch_tokens=True)
                    embedding_group = encoder_group.forward_features(contrastive_data, input_chans=in_chans,return_patch_tokens=True)
                    loss, loss_intra, loss_cross = info_nce_loss_4d_channel(embedding_sub, embedding_group, temperature=0.1, batch_size= B, channel=C, patch=A, dimension=T)
                else:
                    embedding_sub = encoder_sub.forward_features(data, input_chans=in_chans)
                    embedding_group = encoder_group.forward_features(contrastive_data, input_chans=in_chans)
                    loss, loss_intra, loss_cross = info_nce_loss_loop(embedding_sub, embedding_group, temperature=0.1)
                # high level feature from two encoders

                # print(f"embedding_sub.requires_grad: {embedding_sub.requires_grad}")
                # print(f"embedding_group.requires_grad: {embedding_group.requires_grad}")

                # Compute InfoNCE loss
                
                loss.backward()
                optimizer.step()

                # Track the different losses separately
                train_loss += loss.item()
                train_loss_intra += loss_intra.item()
                train_loss_cross += loss_cross.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_train_loss_intra = train_loss_intra / len(train_loader)
            avg_train_loss_cross = train_loss_cross / len(train_loader)
            
            train_losses.append(avg_train_loss)

            # # Validation loss
            # encoder_sub.eval()
            # encoder_group.eval()
            # val_loss = 0
            # with torch.no_grad():
            #     for val_data, val_contrastive_data,label, contrastive_label in val_loader:
            #         val_data = val_data.to(device)
            #         val_contrastive_data = val_contrastive_data.to(device)
            #         val_embedding_sub = encoder_sub.forward_features(val_data, input_chans=in_chans)
            #         val_embedding_group = encoder_group.forward_features(val_contrastive_data, input_chans=in_chans)
            #         val_loss += info_nce_loss_loop(val_embedding_sub, val_embedding_group, temperature=0.1).item()

            # avg_val_loss = val_loss / len(val_loader)
            # val_losses.append(avg_val_loss)
            # Validation loss
            encoder_sub.eval()
            encoder_group.eval()
            val_loss = 0
            val_loss_intra = 0
            val_loss_cross = 0
            with torch.no_grad():
                for val_data, val_contrastive_data, label, contrastive_label in val_loader:
                    val_data = val_data.to(device)
                    val_contrastive_data = val_contrastive_data.to(device)
                    if args.embedding_high_level:
                        val_embedding_sub = encoder_sub.forward_features(val_data, input_chans=in_chans,return_patch_tokens=True)
                        val_embedding_group = encoder_group.forward_features(val_contrastive_data, input_chans=in_chans,return_patch_tokens=True)
                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = info_nce_loss_4d_channel(val_embedding_sub, val_embedding_group, temperature=0.7,
                                                                                                           batch_size= B, channel=C, patch=A, dimension=T)   
                    else:                 
                        val_embedding_sub = encoder_sub.forward_features(val_data, input_chans=in_chans)
                        val_embedding_group = encoder_group.forward_features(val_contrastive_data, input_chans=in_chans)
                        val_loss_curr, val_loss_intra_curr, val_loss_cross_curr = info_nce_loss_loop(val_embedding_sub, val_embedding_group, temperature=0.7)

                    
                    val_loss += val_loss_curr.item()
                    val_loss_intra += val_loss_intra_curr.item()
                    val_loss_cross += val_loss_cross_curr.item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_loss_intra = val_loss_intra / len(val_loader)
            avg_val_loss_cross = val_loss_cross / len(val_loader)
            
            val_losses.append(avg_val_loss)

            # if (epoch + 1) % 10 == 0 or epoch == 0:
            #     print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, "
                    f"Loss Intra={avg_train_loss_intra:.4f}, Loss Cross={avg_train_loss_cross:.4f}, "
                    f"Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")

            # Checkpoint saving and early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder_sub.state_dict(),
                    'val_loss': avg_val_loss,
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                


        losses = {
        'train_losses': train_losses,
        'val_losses': val_losses}
        # Final test evaluation if test set is available
        if test_metrics is not None:  # Check if test set is not empty
            print(f"\n{'='*80}")
            print(f"Subject {subject_id:02d} - Final Test Evaluation")
            print(f"{'='*80}")

            checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'))
            encoder_sub.load_state_dict(checkpoint['encoder_state_dict'])

            # Test metrics
            test_loss = 0
            test_loss_intra =0
            test_loss_cross =0
            for test_data, test_contrastive_data,label, contrastive_label in test_loader:
                test_data = test_data.to(device)
                test_contrastive_data = test_contrastive_data.to(device)                

                if args.embedding_high_level:
                    test_embedding_sub = encoder_sub.forward_features(test_data, input_chans=in_chans,return_patch_tokens=True)
                    test_embedding_group = encoder_group.forward_features(test_contrastive_data, input_chans=in_chans,  return_patch_tokens=True)
                    test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = info_nce_loss_4d_channel(test_embedding_sub, test_embedding_group, temperature=0.1,
                                                                                                          batch_size= B, channel=C, patch=A, dimension=T) 
                else:
                    test_embedding_sub = encoder_sub.forward_features(test_data, input_chans=in_chans)
                    test_embedding_group = encoder_group.forward_features(test_contrastive_data, input_chans=in_chans)
                    test_loss_curr, test_loss_intra_curr, test_loss_cross_curr = info_nce_loss_loop(test_embedding_sub, test_embedding_group, temperature=0.1)                
                
                test_loss += test_loss_curr.item()
                test_loss_intra += test_loss_intra_curr.item()
                test_loss_cross += test_loss_cross_curr.item()

            avg_test_loss = test_loss / len(test_loader)
            avg_test_loss_intra = test_loss_intra / len(test_loader)
            avg_test_loss_cross = test_loss_cross / len(test_loader)

            # Save metrics
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss': avg_train_loss,
                'train_loss_intra': avg_train_loss_intra,
                'train_loss_cross': avg_train_loss_cross,
                'val_size': val_size,
                'val_loss': avg_val_loss,
                'val_loss_intra': avg_val_loss_intra,
                'val_loss_cross': avg_val_loss_cross,
                'test_size': test_size,
                'test_loss': avg_test_loss,
                'test_loss_intra': avg_test_loss_intra,
                'test_loss_cross': avg_test_loss_cross,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        else:
            metrics = {
                'subject_id': int(subject_id),
                'train_size': train_size,
                'train_loss':avg_train_loss,
                'val_size': val_size,
                'val_loss':avg_val_loss,
                'best_epoch': checkpoint['epoch'] + 1,
            }

            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Results saved to {subject_dir}")

        # Cleanup
        torch.cuda.empty_cache()

        return metrics, losses

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'encoder_sub' in locals():
            del encoder_sub
        torch.cuda.empty_cache()

        return None, None
    
    


def train_single_subject_baseline_multitask(args, subject_id, device, num_patches, dataset):
    """Train transfer learning classifier for single subject with frozen encoder"""
    if args.dataset=='m3cv':
        label_mapping = {
            10: 0,
            11: 1,
            12: 2,
            20: 3,
            21: 4,
            30: 5,
            31: 6,
            32: 7,
            40: 8,
            41: 9,
            42: 10
        }
        target_task_ids =[1,2,3,4]
    else:
        label_mapping = {
            10: 0,
            11: 1,
            12: 2,
            20: 3,
            21: 4,
            22: 5
        }
        target_task_ids =[1,2]
                
    try:
        # Create subject-specific output directory
        output_path = os.path.join(args.output_root, args.dataset, args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        # Check for checkpoint resumption
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")

        # Create dataloaders for cognitive task
        if args.multi_task:
            train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
                dataset= dataset,
                target_subject_id=subject_id,
                target_task_id=target_task_ids,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        # elif args.multitask_fewshot:
        #     train_loader, val_loader, test_loader= create_multitask_single_subject_few_shot_dataloaders(
        #         dataset=dataset,
        #         target_subject_id=subject_id,
        #         target_task_id=[1,2,3,4],
        #         n_shot = args.n_shot,
        #         batch_size =args.batch_size,
        #         random_state=args.seed,
        #         num_workers=args.num_workers,
        #         val_size=args.val_size,
        #         oversample=args.oversample
        #     )
        elif args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")
        
        """
        in chans
        """
        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)    
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)
            
        if args.model =='Brainomni':
            position, sensor_type, valid_electrodes = extract_1020_electrode_positions(ch_names,device)  

        # Create encoder
        """LaBraM"""
        if args.model == 'labram_base_patch200_200':
       
            encoder = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches,
            )
            load_vanilla_labram_weights(
                encoder, 
                args.pretrained_path, 
                model_key='model', # 对应 labram-base.pth 内部的键
                model_prefix=''    # 对应前缀过滤
            )

            """CBraMod"""
        elif args.model == 'CBraMod':
            encoder=CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
            map_location = torch.device(args.device)
            encoder.load_state_dict(torch.load(args.pretrained_path, map_location=map_location,weights_only=False))

            """BrainOmni"""
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
            encoder=get_brainomni_model(args.pretrained_path).to(device)

            """BIOT"""
        elif args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)
            map_location = torch.device(args.device)
            encoder.load_state_dict(torch.load(args.pretrained_path, map_location=map_location,weights_only=False))
            

        if args.cls_fitune_type == 'lora':
            if args.model=='CBraMod':
                encoder = apply_lora_to_model(encoder, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
            else:
                encoder = create_lora_cls(encoder,args)
            for name, param in encoder.named_parameters():
                if 'lora' not in name:  # 只冻结非LoRA参数
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        elif args.cls_fitune_type == 'full':
            if args.model=='BIOT':
                for name, param in encoder.named_parameters():
                    if param.is_floating_point(): 
                        param.requires_grad = True 
            else:
                for name, param in encoder.named_parameters():
                    param.requires_grad = True

        elif args.cls_fitune_type == 'linear':
            for name, param in encoder.named_parameters():
                param.requires_grad = False   

                

        if args.model =='Brainomni':
            model = SimpleClassifier_brainomni(
                encoder=encoder,
                position =position,
                sensortype =sensor_type,
                args = args,
                num_classes=args.num_classes,
            ).to(device)
        elif args.model =='BIOT':
            model = SimpleClassifier_BIOT(
                encoder=encoder, 
                args=args, 
                ch_names=ch_names, 
                num_classes=args.num_classes,
            ).to(device) 
        else:
            model = SimpleClassifier_multi_model(
                encoder=encoder,
                num_channels= chan_number,
                in_chans = in_chans,
                args = args,  
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)  
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"\n  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None
        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training (Linear Probing, Frozen Encoder)")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        # Training loop
        for epoch in range(args.epochs):
            # Learning rate scheduling
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Train
            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']
                    # print("type(eeg):", type(eeg))
                    labels = torch.tensor([label_mapping[label.item()] for label in labels])
                    # labels=mapped_labels

                eeg = eeg.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_loss, train_acc, train_f1, val_cm = evaluate_multitask(
                model, train_loader, criterion, device, label_mapping, args.use_amp
            )
            # Validation
            val_loss, val_acc, val_f1, val_cm = evaluate_multitask(
                model, val_loader, criterion, device, label_mapping, args.use_amp
            )

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"TrainAcc={train_acc:.4f}, trainF1={train_f1:.4f}, "
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'classifier_state_dict': model.classifier.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Final test evaluation
        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_f1, test_cm = evaluate_multitask(
            model, test_loader, criterion, device, label_mapping, args.use_amp
        )

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        # Save metrics
        metrics = {
            'subject_id': int(subject_id),
            'method': args.cls_fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification'
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        # Memory cleanup
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None        

class SimpleClassifierWithRest(nn.Module):
    """
    支持 rest_eeg 和 task_eeg 双输入的分类器
    将两个 embedding 拼接后进行分类
    """
    def __init__(self, encoder, num_channels=61, in_chans=None, args=None, 
                 num_classes=3, embed_dim=200, fusion='concat'):
        super().__init__()
        self.encoder = encoder
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.model = args.model if args else 'labram_base_patch200_200'
        self.fusion = fusion  # 'concat', 'add', 'mean'
        
        # 根据融合方式确定分类器输入维度
        if fusion == 'concat':
            classifier_input_dim = embed_dim * 2
        else:
            classifier_input_dim = embed_dim
            
        self.classifier_CBM = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes),
        )
        self.classifier = nn.Linear(classifier_input_dim, num_classes)

    def forward(self, task_eeg, rest_eeg=None):
        """
        Args:
            task_eeg: (B, C, A, T) - 任务 EEG
            rest_eeg: (B, C, A, T) - 静息态 EEG (可选)
        Returns:
            logits: (B, num_classes)
        """
        # 提取 task embedding
        if self.model == 'CBraMod':
            task_features = self.encoder(task_eeg)
            task_features = rearrange(task_features, 'b c s d -> b d c s')
            # 全局平均池化
            task_embedding = F.adaptive_avg_pool2d(task_features, (1, 1)).squeeze(-1).squeeze(-1)
        else:
            task_embedding = self.encoder.forward_features(task_eeg, input_chans=self.in_chans)
        
        # 如果有 rest_eeg，提取 rest embedding 并融合
        if rest_eeg is not None:
            if self.model == 'CBraMod':
                rest_features = self.encoder(rest_eeg)
                rest_features = rearrange(rest_features, 'b c s d -> b d c s')
                rest_embedding = F.adaptive_avg_pool2d(rest_features, (1, 1)).squeeze(-1).squeeze(-1)
            else:
                rest_embedding = self.encoder.forward_features(rest_eeg, input_chans=self.in_chans)
            
            # 融合 embedding
            if self.fusion == 'concat':
                combined_embedding = torch.cat([task_embedding, rest_embedding], dim=-1)
            elif self.fusion == 'add':
                combined_embedding = task_embedding + rest_embedding
            elif self.fusion == 'mean':
                combined_embedding = (task_embedding + rest_embedding) / 2
            else:
                combined_embedding = task_embedding
        else:
            # 如果没有 rest_eeg，使用 task_embedding 的两倍（保持维度一致）
            if self.fusion == 'concat':
                combined_embedding = torch.cat([task_embedding, task_embedding], dim=-1)
            else:
                combined_embedding = task_embedding
        
        logits = self.classifier(combined_embedding)
        return logits

class SimpleClassifier_BIOT_WithRest(nn.Module):
    """
    支持 rest_eeg 和 task_eeg 双输入的 BIOT 分类器
    使用 BIOT_DataProjector 进行通道投影，然后将两个 embedding 拼接后进行分类
    """
    def __init__(self, encoder, args, ch_names, num_classes=3, embed_dim=256, fusion='concat'):
        super().__init__()
        self.encoder = encoder
        self.eeg_projector = BIOT_DataProjector()
        self.embed_dim = embed_dim
        self.model = args.model
        self.ch_names = ch_names
        self.fusion = fusion  # 'concat', 'add', 'mean', 'task_only'
        
        # 根据融合方式确定分类器输入维度
        if fusion == 'concat':
            classifier_input_dim = embed_dim * 2
        else:
            classifier_input_dim = embed_dim
            
        self.classifier = nn.Linear(classifier_input_dim, num_classes)

    def forward(self, task_eeg, rest_eeg=None):
        """
        Args:
            task_eeg: (B, C, A, T) - 任务 EEG
            rest_eeg: (B, C, A, T) - 静息态 EEG (可选)
        Returns:
            logits: (B, num_classes)
        """
        # 处理 task EEG
        task_projected = self.eeg_projector.channel_project(task_eeg, self.ch_names)
        task_features = self.encoder(task_projected)
        
        # 如果有 rest_eeg，提取 rest embedding 并融合
        if rest_eeg is not None:
            rest_projected = self.eeg_projector.channel_project(rest_eeg, self.ch_names)
            rest_features = self.encoder(rest_projected)
            
            # 融合 embedding
            if self.fusion == 'concat':
                combined_features = torch.cat([task_features, rest_features], dim=-1)
            elif self.fusion == 'add':
                combined_features = task_features + rest_features
            elif self.fusion == 'mean':
                combined_features = (task_features + rest_features) / 2
            else:
                combined_features = task_features
        else:
            # 如果没有 rest_eeg
            if self.fusion == 'concat':
                # 需要保持维度一致，将 task_features 复制一份
                combined_features = torch.cat([task_features, task_features], dim=-1)
            else:
                combined_features = task_features
        
        logits = self.classifier(combined_features)
        return logits

def evaluate_multitask_with_rest(model, task_loader, rest_loader, criterion, device, label_mapping, use_amp=False):
    """
    评估函数，支持同时传入 task 和 rest 数据
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for (task_batch, rest_batch) in zip(task_loader, rest_loader):
            # 处理 task 数据
            if isinstance(task_batch, (tuple, list)):
                eeg, labels = task_batch
            else:
                eeg = task_batch['eeg']
                labels = task_batch['label']
                labels = torch.tensor([label_mapping[label.item()] for label in labels])

            eeg = eeg.to(device)
            labels = labels.to(device)
            
            # 处理 rest 数据
            if isinstance(rest_batch, (tuple, list)):
                rest_eeg, _ = rest_batch
            else:

                rest_eeg = rest_batch['eeg']
            rest_eeg = rest_eeg.to(device)
            rest_eeg = prepare_rest_data(rest_eeg, eeg.size(0), device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(eeg, rest_eeg)
                    loss = criterion(logits, labels)
            else:
                logits = model(eeg, rest_eeg)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(task_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, f1, cm
        
def prepare_rest_data(rest_eeg, target_batch_size, device):
    """
    确保 rest_eeg 的 batch size 与 task_eeg 匹配
    如果不够则重复，如果太多则截断
    """
    rest_eeg = rest_eeg.to(device)
    current_size = rest_eeg.size(0)
    
    if current_size < target_batch_size:
        # 重复 rest_eeg 直到达到目标 batch size
        repeat_times = (target_batch_size // current_size) + 1
        rest_eeg = rest_eeg.repeat(repeat_times, 1, 1, 1)[:target_batch_size]
    elif current_size > target_batch_size:
        # 截断
        rest_eeg = rest_eeg[:target_batch_size]
    
    return rest_eeg
from itertools import cycle
def train_single_subject_baseline_multitask_rest(args, subject_id, device, num_patches, dataset, rest_dataset):
    """Train transfer learning classifier for single subject with frozen encoder"""
    if args.dataset=='m3cv':
        label_mapping = {
            10: 0, 11: 1, 12: 2,
            20: 3, 21: 4,
            30: 5, 31: 6, 32: 7,
            40: 8, 41: 9, 42: 10
        }
        target_task_ids = [1, 2, 3, 4]
    else:
        label_mapping = {
            0: 0, 1: 1, 2: 2,
            10: 3, 11: 4, 12: 5
        }
        target_task_ids = [0, 1]
                
    try:
        # Create subject-specific output directory
        output_path = os.path.join(args.output_root, args.dataset, args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        # Check for checkpoint resumption
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")

        # Create dataloaders for cognitive task
        if args.multi_task:
            train_loader, val_loader, test_loader = create_multitask_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                target_task_id=target_task_ids,
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
        # elif args.multitask_fewshot:
        #     train_loader, val_loader, test_loader= create_multitask_single_subject_few_shot_dataloaders(
        #         dataset=dataset,
        #         target_subject_id=subject_id,
        #         target_task_id=[1,2,3,4],
        #         n_shot = args.n_shot,
        #         batch_size =args.batch_size,
        #         random_state=args.seed,
        #         num_workers=args.num_workers,
        #         val_size=args.val_size,
        #         oversample=args.oversample
        #     )
        elif args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")
        
        """
        in chans
        """
        if args.dataset == 'cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans', in_chans)    
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)
            
        if args.model == 'Brainomni':
            position, sensor_type, valid_electrodes = extract_1020_electrode_positions(ch_names, device)  

        # Create encoder
        """LaBraM"""
        if args.model == 'labram_base_patch200_200':
            encoder = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches,
            )
            load_vanilla_labram_weights(
                encoder, 
                args.pretrained_path, 
                model_key='model',
                model_prefix=''
            )

            """CBraMod"""
        elif args.model == 'CBraMod':
            encoder=CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
            map_location = torch.device(args.device)
            encoder.load_state_dict(torch.load(args.pretrained_path, map_location=map_location,weights_only=False))

            """BrainOmni"""
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
            encoder=get_brainomni_model(args.pretrained_path).to(device)

            """BIOT"""
        elif args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)
            map_location = torch.device(args.device)
            encoder.load_state_dict(torch.load(args.pretrained_path, map_location=map_location,weights_only=False))
            

        if args.cls_fitune_type == 'lora':
            if args.model=='CBraMod':
                encoder = apply_lora_to_model(encoder, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
            else:
                encoder = create_lora_cls(encoder,args)
            for name, param in encoder.named_parameters():
                if 'lora' not in name:  # 只冻结非LoRA参数
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        elif args.cls_fitune_type == 'full':
            if args.model=='BIOT':
                for name, param in encoder.named_parameters():
                    if param.is_floating_point(): 
                        param.requires_grad = True 
            else:
                for name, param in encoder.named_parameters():
                    param.requires_grad = True

        elif args.cls_fitune_type == 'linear':
            for name, param in encoder.named_parameters():
                param.requires_grad = False   

                

        if args.model =='Brainomni':
            model = SimpleClassifier_brainomni(
                encoder=encoder,
                position =position,
                sensortype =sensor_type,
                args = args,
                num_classes=args.num_classes,
            ).to(device)
        elif args.model =='BIOT':
            model = SimpleClassifier_BIOT_WithRest(
                encoder=encoder, 
                args=args, 
                ch_names=ch_names, 
                num_classes=args.num_classes,
                embed_dim=args.embed_dim,
                fusion='concat'
            ).to(device) 
        else:
            model = SimpleClassifierWithRest(
                encoder=encoder,
                num_channels=chan_number,
                in_chans=in_chans,
                args=args,  
                num_classes=args.num_classes,
                embed_dim=args.embed_dim,
                fusion='concat'  # 或 'add', 'mean'
            ).to(device)  
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"\n  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None
        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training (Linear Probing, Frozen Encoder)")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        # Training loop
        for epoch in range(args.epochs):
            # Learning rate scheduling
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Train
            model.train()
            train_loss = 0

            # 训练循环
            for task_batch, rest_batch in zip(train_loader, cycle(train_rest)):
                # 处理 task 数据
                if isinstance(task_batch, (tuple, list)):
                    eeg, labels = task_batch
                else:
                    eeg = task_batch['eeg']
                    labels = task_batch['label']
                    labels = torch.tensor([label_mapping[label.item()] for label in labels])

                eeg = eeg.to(device)
                labels = labels.to(device)
                batch_size = eeg.size(0)
                
                # 处理 rest 数据 - 确保 batch size 匹配
                if isinstance(rest_batch, (tuple, list)):
                    rest_eeg, _ = rest_batch
                else:
                    rest_eeg = rest_batch['eeg']
                
                rest_eeg = prepare_rest_data(rest_eeg, batch_size, device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_loss, train_acc, train_f1, val_cm = evaluate_multitask(
                model, train_loader, criterion, device, label_mapping, args.use_amp
            )
            # Validation
            val_loss, val_acc, val_f1, val_cm = evaluate_multitask_with_rest(
                model, val_loader, val_rest, criterion, device, label_mapping, args.use_amp
            )


            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'classifier_state_dict': model.classifier.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Final test evaluation
        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_f1, test_cm = evaluate_multitask(
            model, test_loader, criterion, device, label_mapping, args.use_amp
        )

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        # Save metrics
        metrics = {
            'subject_id': int(subject_id),
            'method': args.cls_fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification'
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        # Memory cleanup
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None        


import torch.nn.utils as utils
def train_single_subject_cls_multi_task_cog_time(args, subject_id, device, num_patches, dataset):
    """Train transfer learning classifier for single subject with frozen encoder"""
    label_mapping = {
        0: 0,
        1: 1,
        2: 2,
        10: 3,
        11: 4,
        12: 5,
    }
    try:
        valid_fitune_types = ['lora', 'full']
        valid_cls_fitune_types = ['lora', 'full', 'linear']
        if args.fitune_type not in valid_fitune_types:
            raise ValueError(f"fitune_type must be one of {valid_fitune_types}")
        if args.cls_fitune_type not in valid_cls_fitune_types:
            raise ValueError(f"cls_fitune_type must be one of {valid_cls_fitune_types}")
        # Create subject-specific output directory
        output_path = os.path.join(args.output_root, args.dataset,  args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        # Check for checkpoint resumption
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")

        if args.multi_task:
            train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
                dataset= dataset,
                target_subject_id=subject_id,
                target_task_id=[0,1],
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        elif args.multitask_fewshot:
            train_loader, val_loader, test_loader= create_multitask_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                target_task_id=[0, 1],
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )


        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        """
        in chans
        """
        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)    
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)
            
        if args.model =='Brainomni':
            position, sensor_type, valid_electrodes = extract_1020_electrode_positions(ch_names,device)   
        
        checkpoint_path = os.path.join(args.recon_results_dir, args.dataset, f'subject_{subject_id:02d}','best_model.pth')
        print(f"  Checkpoint: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {checkpoint_path}")
        
        print(f"\n[Step 1] Creating Encoder ({args.model})...")

        """LaBraM"""
        if args.model =='labram_base_patch200_200':
            encoder = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches).to(device)
            
            if args.fitune_type =='lora':
                encoder_with_lora = create_lora(encoder,args)
                encoder = load_pretrained_encoder(encoder_with_lora, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

            """cBraMod"""           
        elif args.model =='CBraMod':
            encoder = CBraMod().to(device)
            if args.fitune_type =='lora':
                encoder = apply_lora_to_model(encoder, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    encoder = apply_lora_to_model(encoder, r=args.cls_lora_r, alpha=args.cls_lora_alpha, dropout=args.cls_lora_dropout)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

            """Brainomni"""
        elif args.model =='Brainomni':
            from brainomni.model import BrainOmni
            with open(os.path.join('/mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/models/checkpoint/BrainOmni_tiny', "model_cfg.json")) as f:
                cfg = json.load(f)
            encoder = BrainOmni(**cfg)
            if args.fitune_type =='lora':
                encoder = create_lora(encoder,args)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False    

            """BIOT"""
        elif args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)
            
            if args.fitune_type =='lora':
                encoder = create_lora(encoder,args)
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        if param.is_floating_point(): 
                            param.requires_grad = True 
                        else:
                            param.requires_grad = False
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        if param.is_floating_point(): 
                            param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False    
                                    
        if args.model =='Brainomni':
            model = SimpleClassifier_brainomni(
                encoder=encoder,
                position =position,
                sensortype =sensor_type,
                args = args,
                num_classes=args.num_classes,
            ).to(device)
        elif args.model =='BIOT':
            model = SimpleClassifier_BIOT(
                encoder=encoder, 
                args=args, 
                ch_names=ch_names, 
                num_classes=args.num_classes,
            ).to(device) 
        else:
            model = SimpleClassifier_multi_model(
                encoder=encoder,
                num_channels= chan_number,
                in_chans = in_chans,
                args = args,  
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)     

        # for name, param in model.named_parameters():
        #     # if param.requires_grad:  # 如果参数是可训练的
        #     #     print(f"Trainable parameter: {name}")           
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        # print(f"\n  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None


        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training (Linear Probing, Frozen Encoder)")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        # Training loop
        for epoch in range(args.epochs):
            # Learning rate scheduling
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Train
            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']
                    mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])

                eeg = eeg.to(device)
                mapped_labels = mapped_labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, mapped_labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, mapped_labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Print progress (no validation)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"LR={current_lr:.6f}")

            # Save model at the last epoch (no early stopping, no validation)
            if epoch == args.epochs - 1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                }, os.path.join(subject_dir, 'best_model.pth'))

        # Final test evaluation
        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_f1, test_cm = evaluate_multitask_cog(
            model, test_loader, criterion, device, args.use_amp
        )

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        # Save metrics (no validation performed)
        metrics = {
            'subject_id': int(subject_id),
            'method': args.cls_fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': None,
            'best_val_f1': None,
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification'
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        # Memory cleanup
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None




def train_single_subject_cls_scratch_all(args, subject_id, device, num_patches, dataset):
    """
    Train a classifier from scratch (Random Initialization) for a single subject.
    Compatible with LaBraM, CBraMod, Brainomni, and BIOT.
    """
    label_mapping = {0: 0, 1: 1, 2: 2, 10: 3, 11: 4, 12: 5}
    
    try:
        # Create subject-specific output directory
        output_path = os.path.join(args.output_root, args.dataset, "scratch_baseline")
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        # Check for checkpoint resumption
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained (Scratch), skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"🌟 Subject {subject_id:02d} - Classification Baseline (Train from Scratch)")
        print(f"{'='*80}")

        # ==========================================
        # 1. Loading Data (Keeping your exact dataloader logic)
        # ==========================================
        if args.multi_task:
            train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
                dataset=dataset, target_subject_id=subject_id, target_task_id=[0,1],
                train_size=args.train_size, test_size=args.test_size, val_size=args.val_size,
                batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers
            )
        elif getattr(args, 'dataset_suffle', False):
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset, target_subject_id=subject_id, train_size=args.train_size,
                test_size=args.test_size, val_size=args.val_size, batch_size=args.batch_size,
                random_state=args.seed, num_workers=args.num_workers
            )
        else:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders_order(
                dataset=dataset, target_subject_id=subject_id, train_size=args.train_size,
                test_size=args.test_size, val_size=args.val_size, batch_size=args.batch_size,
                random_state=args.seed, num_workers=args.num_workers
            )

        train_size, val_size, test_size = len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)
        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < getattr(args, 'min_samples_per_class', 1):
            raise ValueError(f"Insufficient training samples: {train_size}")

        # Extract Channels and Positions
        if args.dataset == 'cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            
        if args.model == 'Brainomni':
            position, sensor_type, valid_electrodes = extract_1020_electrode_positions(ch_names, device)   

        # ==========================================
        # 2. Creating Encoder (Random Initialization)
        # ==========================================
        print(f"\n[Step 1] Creating Random Initialized Encoder ({args.model})...")

        if args.model == 'labram_base_patch200_200':
            encoder = create_model(
                args.model, pretrained=False,
                use_mean_pooling=getattr(args, 'use_mean_pooling', True),
                init_scale=getattr(args, 'init_scale', 0.001),
                use_rel_pos_bias=getattr(args, 'rel_pos_bias', True),
                use_abs_pos_emb=getattr(args, 'abs_pos_emb', False),
                init_values=getattr(args, 'layer_scale_init_value', 0.1),
                qkv_bias=getattr(args, 'qkv_bias', True),
                num_patches_per_channel_input=num_patches
            ).to(device)

        elif args.model == 'CBraMod':
            encoder = CBraMod().to(device)

        elif args.model == 'Brainomni':
            from brainomni.model import BrainOmni
            with open(os.path.join('/mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/models/checkpoint/BrainOmni_tiny', "model_cfg.json")) as f:
                cfg = json.load(f)
            encoder = BrainOmni(**cfg).to(device)

        elif args.model == 'BIOT':
            from src.biot import BIOTEncoder
            # BIOT_DataProjector logic will be handled inside SimpleClassifier_BIOT
            encoder = BIOTEncoder(emb_size=256, heads=8, depth=4, n_channels=18).to(device)

        else:
            raise ValueError(f"Unsupported model: {args.model}")

        if getattr(args, 'cls_fitune_type', '') == 'scratch_lora':
            print("  [Ablation] Applying LoRA to Randomly Initialized Encoder...")
            # 1. 注入 LoRA 结构
            encoder = create_lora(encoder, args)
            # 2. 冻结随机的 Base Weights，只放开 LoRA 参数
            for name, param in encoder.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            print("  [Baseline] Full Train from Scratch...")
            # 原本的 Full Scratch：暴力放开所有参数
            for param in encoder.parameters():
                param.requires_grad = True

        # ==========================================
        # 3. Wrapping with Classification Head
        # ==========================================
        if args.model == 'Brainomni':
            model = SimpleClassifier_brainomni(
                encoder=encoder, position=position, sensortype=sensor_type, 
                args=args, num_classes=args.num_classes
            ).to(device)
        elif args.model == 'BIOT':
            model = SimpleClassifier_BIOT(
                encoder=encoder, args=args, ch_names=ch_names, num_classes=args.num_classes
            ).to(device) 
        else:
            model = SimpleClassifier_multi_model(
                encoder=encoder, num_channels=chan_number, in_chans=in_chans, 
                args=args, num_classes=args.num_classes, embed_dim=getattr(args, 'embed_dim', 200)
            ).to(device)     

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"  ✓ Total Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")

        # ==========================================
        # 4. Optimizer & Loss
        # ==========================================
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=getattr(args, 'label_smoothing', 0.0))
        
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None

        best_val_acc, best_val_f1 = 0.0, 0.0
        patience_counter = 0

        # ==========================================
        # 5. Training Loop
        # ==========================================
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                current_lr = args.lr * ((epoch + 1) / args.warmup_epochs)
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg, labels = batch['eeg'], batch['label']
                    
                mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])
                eeg, mapped_labels = eeg.to(device), mapped_labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, mapped_labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, mapped_labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_loss, val_acc, val_f1, val_cm = evaluate_multitask_cog(
                model, val_loader, criterion, device, args.use_amp
            )

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: TrainLoss={avg_train_loss:.4f}, ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, LR={current_lr:.6f}")

            if val_acc > best_val_acc:
                best_val_acc, best_val_f1 = val_acc, val_f1
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # ==========================================
        # 6. Final Test Evaluation
        # ==========================================
        print(f"\n{'='*80}\nSubject {subject_id:02d} - Final Test Evaluation\n{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_acc, test_f1, test_cm = evaluate_multitask_cog(
            model, test_loader, criterion, device, args.use_amp
        )

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")

        # Save metrics
        metrics = {
            'subject_id': int(subject_id),
            'method': 'train_from_scratch',
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'train_from_scratch', 
            'source_task': 'none',
            'target_task': 'cognitive_classification'
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        del model, optimizer, encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()
        if 'model' in locals(): del model
        torch.cuda.empty_cache()
        return None
    

def train_single_subject_cls_multi_task_cog(args, subject_id, device, num_patches, dataset):
    """Train transfer learning classifier for single subject with frozen encoder"""
    label_mapping = {
        10: 0,
        11: 1,
        12: 2,
        20: 3,
        21: 4,
        22: 5,
    }
    try:
        valid_fitune_types = ['lora', 'full']
        valid_cls_fitune_types = ['lora', 'full', 'linear']
        if args.fitune_type not in valid_fitune_types:
            raise ValueError(f"fitune_type must be one of {valid_fitune_types}")
        if args.cls_fitune_type not in valid_cls_fitune_types:
            raise ValueError(f"cls_fitune_type must be one of {valid_cls_fitune_types}")
        # Create subject-specific output directory
        output_path = os.path.join(args.output_root, args.dataset,  args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        # Check for checkpoint resumption
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")

        if args.multi_task:
            train_loader, val_loader, test_loader= create_multitask_single_subject_dataloaders(
                dataset= dataset,
                target_subject_id=subject_id,
                target_task_id=[1,2],
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )
        elif args.multitask_fewshot:
            train_loader, val_loader, test_loader= create_multitask_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                target_task_id=[1,2],
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )


        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")

        """
        in chans
        """
        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)    
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)
            
        if args.model =='Brainomni':
            position, sensor_type, valid_electrodes = extract_1020_electrode_positions(ch_names,device)   
        
        checkpoint_path = os.path.join(args.recon_results_dir, args.dataset, f'subject_{subject_id:02d}','best_model.pth')
        print(f"  Checkpoint: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {checkpoint_path}")
        
        print(f"\n[Step 1] Creating Encoder ({args.model})...")

        """LaBraM"""
        if args.model =='labram_base_patch200_200':
            encoder = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches).to(device)
            
            if args.fitune_type =='lora':
                encoder_with_lora = create_lora(encoder,args)
                encoder = load_pretrained_encoder(encoder_with_lora, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

            """cBraMod"""           
        elif args.model =='CBraMod':
            encoder = CBraMod().to(device)
            if args.fitune_type =='lora':
                encoder = apply_lora_to_model(encoder, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    encoder = apply_lora_to_model(encoder, r=args.cls_lora_r, alpha=args.cls_lora_alpha, dropout=args.cls_lora_dropout)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif args.cls_fitune_type =='full':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    # encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False

            """Brainomni"""
        elif args.model =='Brainomni':
            from brainomni.model import BrainOmni
            with open(os.path.join('/mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/models/checkpoint/BrainOmni_tiny', "model_cfg.json")) as f:
                cfg = json.load(f)
            encoder = BrainOmni(**cfg)
            if args.fitune_type =='lora':
                encoder = create_lora(encoder,args)
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True 
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False    

            """BIOT"""
        elif args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)
            
            if args.fitune_type =='lora':
                encoder = create_lora(encoder,args)
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if args.cls_fitune_type =='lora':
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True 
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        if param.is_floating_point(): 
                            param.requires_grad = True 
                        else:
                            param.requires_grad = False
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False
                        
            elif args.fitune_type =='full':
                encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
                if args.cls_fitune_type =='lora':
                    encoder = create_lora_cls(encoder,args)
                    for name, param in encoder.named_parameters():
                        if 'lora' not in name:  # 只冻结非LoRA参数
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            
                elif args.cls_fitune_type =='full':
                    for name, param in encoder.named_parameters():
                        if param.is_floating_point(): 
                            param.requires_grad = True
                elif args.cls_fitune_type =='linear':
                    for name, param in encoder.named_parameters():
                        param.requires_grad = False    
                                    
        if args.model =='Brainomni':
            model = SimpleClassifier_brainomni(
                encoder=encoder,
                position =position,
                sensortype =sensor_type,
                args = args,
                num_classes=args.num_classes,
            ).to(device)
        elif args.model =='BIOT':
            model = SimpleClassifier_BIOT(
                encoder=encoder, 
                args=args, 
                ch_names=ch_names, 
                num_classes=args.num_classes,
            ).to(device) 
        else:
            model = SimpleClassifier_multi_model(
                encoder=encoder,
                num_channels= chan_number,
                in_chans = in_chans,
                args = args,  
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)     

        # for name, param in model.named_parameters():
        #     # if param.requires_grad:  # 如果参数是可训练的
        #     #     print(f"Trainable parameter: {name}")           
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        # print(f"\n  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None

        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training (Linear Probing, Frozen Encoder)")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        # Training loop
        for epoch in range(args.epochs):
            # Learning rate scheduling
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Train
            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']
                    mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])

                eeg = eeg.to(device)
                mapped_labels = mapped_labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, mapped_labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, mapped_labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_loss, val_acc, val_f1, val_cm = evaluate_multitask_cog(
                model, val_loader, criterion, device, args.use_amp
            )

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Final test evaluation
        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_f1, test_cm = evaluate_multitask_cog(
            model, test_loader, criterion, device, args.use_amp
        )

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        # Save metrics
        metrics = {
            'subject_id': int(subject_id),
            'method': args.cls_fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification'
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        # Memory cleanup
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None


def train_single_subject_baseline_multitask_rest_signal(args, subject_id, device, num_patches, dataset,rest_dataset):
    """Train transfer learning classifier for single subject with frozen encoder"""
    if args.dataset=='m3cv':
        label_mapping = {
            10: 0,
            11: 1,
            12: 2,
            20: 3,
            21: 4,
            30: 5,
            31: 6,
            32: 7,
            40: 8,
            41: 9,
            42: 10
        }
        target_task_ids =[1,2,3,4]
    else:
        label_mapping = {
            0: 0,
            1: 1,
            2: 2,
            10: 3,
            11: 4,
            12: 5
        }
        target_task_ids =[0, 1]
                
    try:
        # Create subject-specific output directory
        output_path = os.path.join(args.output_root, args.dataset, args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)

        # Check for checkpoint resumption
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Subject {subject_id:02d} already trained, skipping...")
            with open(metrics_path, 'r') as f:
                return json.load(f)

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Loading Data")
        print(f"{'='*80}")

        # Create dataloaders for cognitive task
        if args.multi_task:
            train_loader, val_loader, test_loader = create_rest_normalized_dataloaders(
                task_dataset=dataset,
                rest_dataset=rest_dataset,
                target_subject_id=subject_id,
                target_task_id=target_task_ids,
                train_size=args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                normalize_per_channel=True,    # 每个通道独立标准化
                use_global_rest=False          # 使用当前被试的rest数据，设为True则使用所有被试
            )
        # elif args.multitask_fewshot:
        #     train_loader, val_loader, test_loader= create_multitask_single_subject_few_shot_dataloaders(
        #         dataset=dataset,
        #         target_subject_id=subject_id,
        #         target_task_id=[1,2,3,4],
        #         n_shot = args.n_shot,
        #         batch_size =args.batch_size,
        #         random_state=args.seed,
        #         num_workers=args.num_workers,
        #         val_size=args.val_size,
        #         oversample=args.oversample
        #     )
        elif args.fewshot:
            train_loader, val_loader, test_loader= create_single_subject_few_shot_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                n_shot = args.n_shot,
                batch_size =args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers,
                val_size=args.val_size,
                oversample=args.oversample
            )
        elif args.dataset_suffle:
            train_loader, val_loader, test_loader = create_single_subject_dataloaders(
                dataset=dataset,
                target_subject_id=subject_id,
                train_size =args.train_size,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                random_state=args.seed,
                num_workers=args.num_workers
            )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

        if train_size < args.min_samples_per_class:
            raise ValueError(f"Insufficient training samples: {train_size}")
        
        """
        in chans
        """
        if args.dataset=='cogbci':
            in_chans, ch_names, chan_number = get_cogbci_dataset(args, subject_id)
            print('in_chans',in_chans)    
        else:
            in_chans, ch_names, chan_number = get_dataset_name(args)
            print('in_chans', in_chans)
            
        if args.model =='Brainomni':
            position, sensor_type, valid_electrodes = extract_1020_electrode_positions(ch_names,device)  

        # Create encoder
        """LaBraM"""
        if args.model == 'labram_base_patch200_200':
       
            encoder = create_model(
                args.model,
                pretrained=False,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
                num_patches_per_channel_input=num_patches,
            )
            load_vanilla_labram_weights(
                encoder, 
                args.pretrained_path, 
                model_key='model', # 对应 labram-base.pth 内部的键
                model_prefix=''    # 对应前缀过滤
            )

            """CBraMod"""
        elif args.model == 'CBraMod':
            encoder=CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
            map_location = torch.device(args.device)
            encoder.load_state_dict(torch.load(args.pretrained_path, map_location=map_location,weights_only=False))

            """BrainOmni"""
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
            encoder=get_brainomni_model(args.pretrained_path).to(device)

            """BIOT"""
        elif args.model =='BIOT':
            eeg_projector = BIOT_DataProjector()
            from src.biot import BIOTEncoder
            encoder = BIOTEncoder(emb_size=256, heads=8, depth=4,n_channels=18).to(device)
            map_location = torch.device(args.device)
            encoder.load_state_dict(torch.load(args.pretrained_path, map_location=map_location,weights_only=False))
            

        if args.cls_fitune_type == 'lora':
            if args.model=='CBraMod':
                encoder = apply_lora_to_model(encoder, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
            else:
                encoder = create_lora_cls(encoder,args)
            for name, param in encoder.named_parameters():
                if 'lora' not in name:  # 只冻结非LoRA参数
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        elif args.cls_fitune_type == 'full':
            if args.model=='BIOT':
                for name, param in encoder.named_parameters():
                    if param.is_floating_point(): 
                        param.requires_grad = True 
            else:
                for name, param in encoder.named_parameters():
                    param.requires_grad = True

        elif args.cls_fitune_type == 'linear':
            for name, param in encoder.named_parameters():
                param.requires_grad = False   

                

        if args.model =='Brainomni':
            model = SimpleClassifier_brainomni(
                encoder=encoder,
                position =position,
                sensortype =sensor_type,
                args = args,
                num_classes=args.num_classes,
            ).to(device)
        elif args.model =='BIOT':
            model = SimpleClassifier_BIOT(
                encoder=encoder, 
                args=args, 
                ch_names=ch_names, 
                num_classes=args.num_classes,
            ).to(device) 
        else:
            model = SimpleClassifier_multi_model(
                encoder=encoder,
                num_channels= chan_number,
                in_chans = in_chans,
                args = args,  
                num_classes=args.num_classes,
                embed_dim=args.embed_dim
            ).to(device)  
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"\n  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None
        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience_counter = 0

        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Training (Linear Probing, Frozen Encoder)")
        print(f"Method: {args.cls_fitune_type.upper()}")
        print(f"{'='*80}")

        # Training loop
        for epoch in range(args.epochs):
            # Learning rate scheduling
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Train
            model.train()
            train_loss = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    eeg, labels = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']
                    # print("type(eeg):", type(eeg))
                    labels = torch.tensor([label_mapping[label.item()] for label in labels])
                    # labels=mapped_labels

                eeg = eeg.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_loss, train_acc, train_f1, val_cm = evaluate_multitask(
                model, train_loader, criterion, device, label_mapping, args.use_amp
            )
            # Validation
            val_loss, val_acc, val_f1, val_cm = evaluate_multitask(
                model, val_loader, criterion, device, label_mapping, args.use_amp
            )

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"TrainLoss={avg_train_loss:.4f}, "
                      f"TrainAcc={train_acc:.4f}, trainF1={train_f1:.4f}, "
                      f"ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, "
                      f"LR={current_lr:.6f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'classifier_state_dict': model.classifier.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Final test evaluation
        print(f"\n{'='*80}")
        print(f"Subject {subject_id:02d} - Final Test Evaluation")
        print(f"{'='*80}")

        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_f1, test_cm = evaluate_multitask(
            model, test_loader, criterion, device, label_mapping, args.use_amp
        )

        print(f"  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix:\n{test_cm}")

        # Save metrics
        metrics = {
            'subject_id': int(subject_id),
            'method': args.cls_fitune_type,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_cm': test_cm.tolist(),
            'best_epoch': int(checkpoint['epoch']) + 1,
            'transfer_method': 'linear_probing',
            'source_task': 'resting_reconstruction',
            'target_task': 'cognitive_classification'
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Results saved to {subject_dir}")

        # Memory cleanup
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"  ✗ ERROR for Subject {subject_id:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

        return None        
