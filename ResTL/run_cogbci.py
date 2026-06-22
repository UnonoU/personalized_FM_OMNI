"""
CogBCI One-Click Training and Testing Script for ResTL

Usage:
    python run_cogbci.py -g 0 -te 1 --task all -e 100 --checkpoint ./checkpoints/cogbci/s1
"""
# python /mnt/dataset4/yinuo/personlized_FM/1_rebuttal/MICCAI2024-ResTL-master/MICCAI2024-ResTL-master/run_cogbci.py -g 2 -te 11 --task all -e 100 --checkpoint ./checkpoints/cogbci/s11
import os
import argparse
import numpy as np
import torch
import model
import utils
from torch.utils.data import DataLoader, Subset
import warnings
from train import *
from cogbci_loader import CogBCIDataset, CogBCIRestDataset, COGBCI_CONFIG

warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description="CogBCI Training and Testing")
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-te", "--test", type=str, default='1', help='Test subject ID (e.g., "1" or "1,2,3")')
parser.add_argument("-m", "--model", type=str, default='EEGNet', choices=['EEGNet', 'Conformer'])
parser.add_argument("-e", "--epoch", type=int, default=100)
parser.add_argument("-l", "--learningrate", type=float, default=0.0005)
parser.add_argument("-b", "--batch", type=int, default=128)
parser.add_argument("--task", type=str, default='all', help='Tasks: all, 0, 1, or "0,1"')
parser.add_argument("--cdist", type=float, default=1e-5)
parser.add_argument("--checkpoint", type=str, required=True, help='Checkpoint directory')
parser.add_argument("--skip_train", action='store_true', help='Skip training, only test')
parser.add_argument("--skip_test", action='store_true', help='Skip testing, only train')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

torch.multiprocessing.set_sharing_strategy('file_system')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = device

# Parse test subjects
test_subjects = list(map(int, args.test.split(',')))
train_subjects = [idx for idx in range(1, 30) if idx not in test_subjects]  # 29 subjects

# Parse task IDs
if args.task == 'all':
    target_task_id = [0, 1]
else:
    target_task_id = list(map(int, args.task.split(',')))

# CogBCI Configuration
args.datanum = 6   # 6 classes
args.chnum = 62    # 62 channels
args.sessionnum = 1
args.size = 200    # 200 timepoints
args.multiproto = 1
args.train = train_subjects
args.hyper = [1, 0.5, 0.05, 0]
args.hyper_finetune = [1, 0, 0, 0]

print("=" * 70)
print("CogBCI ResTL Training & Testing")
print("=" * 70)
print(f"Test subjects: {test_subjects}")
print(f"Train subjects: {len(train_subjects)} subjects")
print(f"Tasks: {target_task_id}")
print(f"Model: {args.model}")
print(f"Epochs: {args.epoch}")
print(f"Batch size: {args.batch}")
print(f"Checkpoint: {args.checkpoint}")
print("=" * 70)

# File paths
task_h5_path = COGBCI_CONFIG['multitask_file']
rest_h5_path = COGBCI_CONFIG['resting_file']

# Create checkpoint directory
os.makedirs(args.checkpoint, exist_ok=True)

# Loss and optimizer
loss = torch.nn.CrossEntropyLoss().to(device)

# ===== TRAINING PHASE =====
if not args.skip_train:
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING")
    print("=" * 70)
    
    # Create dataset with all training subjects
    print("\nLoading training data...")
    dataset = CogBCIDataset(
        task_h5_path=task_h5_path,
        rest_h5_path=rest_h5_path,
        target_subject_id=None,
        target_task_id=target_task_id,
        n_classes=args.datanum,
        normalize=True,
        normalize_per_channel=True,
        is_training='train'
    )
    
    # Filter to training subjects
    train_mask = np.isin(dataset.subject_ids, train_subjects)
    train_indices = np.where(train_mask)[0]
    
    # Use last training subject for validation
    val_subject = train_subjects[-1] if len(train_subjects) > 1 else train_subjects[0]
    val_mask = (dataset.subject_ids == val_subject)
    val_indices = np.where(val_mask)[0]
    
    # Remove validation from training
    train_mask = train_mask & (~val_mask)
    train_indices = np.where(train_mask)[0]
    
    print(f"\nTraining samples: {len(train_indices)}")
    print(f"Validation subject: {val_subject} ({len(val_indices)} samples)")
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    data_tr = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )
    data_val = DataLoader(
        val_dataset,
        batch_size=min(len(val_indices), args.batch),
        shuffle=False,
        num_workers=8
    )
    
    # Initialize model
    net = model.Distribution(args).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learningrate)
    
    # Training manager
    manage = utils.find_best_model(args.checkpoint, args)
    manage.code_copy(os.path.join(args.checkpoint, 'run'))
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(args.epoch):
        net.train()
        x, y, output, acc_epoch, loss_epoch, count_epoch = one_epoch_distribution(
            net, data_tr, optimizer, loss, args.hyper, args, train='train'
        )
        print(f'Epoch {epoch+1:3d}/{args.epoch} | Train Acc: {acc_epoch:.4f} | Loss: {loss_epoch["Total"]:.4f}')
        
        net.eval()
        with torch.no_grad():
            x, y, output, acc_epoch, loss_epoch, count_epoch = one_epoch_distribution(
                net, data_val, optimizer, loss, args.hyper, args, train='val'
            )
            manage.update(net, args.checkpoint, epoch, acc_epoch, loss_epoch['CLS'])
            print(f'           | Val Acc:   {acc_epoch:.4f} | Loss: {loss_epoch["Total"]:.4f}')
    
    manage.training_finish(net, args.checkpoint)
    print(f"\n✓ Training completed! Best model saved to {args.checkpoint}")

# ===== TESTING PHASE =====
if not args.skip_test:
    print("\n" + "=" * 70)
    print("PHASE 2: TESTING")
    print("=" * 70)
    
    for test_subject in test_subjects:
        print(f"\n--- Testing Subject {test_subject} ---")
        
        # Load test dataset
        dataset = CogBCIDataset(
            task_h5_path=task_h5_path,
            rest_h5_path=rest_h5_path,
            target_subject_id=test_subject,
            target_task_id=target_task_id,
            n_classes=args.datanum,
            normalize=True,
            normalize_per_channel=True,
            is_training='test'
        )
        
        # Load resting data
        dataset_RS = CogBCIRestDataset(
            rest_h5_path=rest_h5_path,
            target_subject_id=test_subject,
            normalize=True,
            normalize_per_channel=True,
            n_classes=args.datanum,
            max_samples=50  # Use only 50 resting state samples to speed up
        )
        
        # Initialize model
        net = model.Distribution(args).to(device)
        
        # Load best checkpoint
        checkpoint_path = os.path.join(args.checkpoint, 'model-best.pth')
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            continue
        
        restore = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        net.load_state_dict(restore, strict=True)
        print(f"Loaded checkpoint: {checkpoint_path}")
        
        # Test results manager
        manage = utils.test_model(args.checkpoint, args, f'result_subject{test_subject}')
        
        # Test-time adaptation with resting state
        for session in range(args.sessionnum):
            torch.cuda.empty_cache()
            net.load_state_dict(restore, strict=True)
            restore_before = net.state_dict()
            
            # Generate signals from resting state
            data_test_RS = DataLoader(dataset_RS, batch_size=1, shuffle=False, num_workers=8)
            RS, RS_update, RS_update_label = generate_signal_from_RS(
                net, data_test_RS, loss, args.hyper, args, train='train'
            )
            RS_update = torch.cat(RS_update, dim=0)
            RS = torch.cat(RS, dim=0)
            RS_update_label = torch.cat(RS_update_label, dim=0)
            net.load_state_dict(restore_before, strict=True)
            
            # Update resting dataset
            # Use only RS_update (generated pseudo-labeled signal)
            dataset_RS.testset[session]['x'] = RS_update
            dataset_RS.testset[session]['y'] = RS_update_label
            
            # Fine-tune on resting state
            net.train()
            data_test_RS = DataLoader(dataset_RS, batch_size=args.batch, shuffle=True, num_workers=8)
            net.load_state_dict(restore_before)
            optimizer = torch.optim.Adam(net.parameters(), lr=args.learningrate/10)
            kd_loss = torch.nn.KLDivLoss().to(args.device)
            
            print(f"Fine-tuning on resting state (session {session})...")
            for epoch in range(10):
                args.cdist = 0
                x, y, output, acc_epoch, loss_epoch, count_epoch = one_epoch_distribution(
                    net, data_test_RS, optimizer, kd_loss, args.hyper_finetune, args, train='RS'
                )
            
            # Test on task data
            dataset.is_training = 'test'
            dataset.finetuning = False
            data_test = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
            
            net.eval()
            with torch.no_grad():
                x, y, output, acc_epoch, loss_epoch, count_epoch = one_epoch_distribution(
                    net, data_test, optimizer, loss, args.hyper, args, train='val'
                )
                
                result_key = f"S{test_subject}_Session{session}"
                print(f"Result: {result_key} | Samples: {count_epoch} | Acc: {acc_epoch:.4f} | Loss: {loss_epoch['Total']:.4f}")
                manage.total_result(result_key, count_epoch, round(acc_epoch, 7), round(loss_epoch['Total'], 7))

print("\n" + "=" * 70)
print("ALL DONE!")
print("=" * 70)
