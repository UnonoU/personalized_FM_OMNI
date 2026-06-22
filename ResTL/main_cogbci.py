import os
import argparse
import numpy as np
import torch
import model
import utils
from torch.utils.data import DataLoader
import warnings
from train import *
from cogbci_loader import CogBCIDataset, CogBCIRestDataset, create_cogbci_dataloaders, COGBCI_CONFIG

warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description="Cross-subject")
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-te", "--test", type=str, default='9')
parser.add_argument("-m", "--model", type=str, default='EEGNet')
parser.add_argument("-e", "--epoch", type=int, default=100)
parser.add_argument("-l", "--learningrate", type=float, default=0.0005)
parser.add_argument("-t", "--is_training", type=str, default='train')
parser.add_argument("-b", "--batch", type=int, default=128)
parser.add_argument("-d", "--data", type=str, default='BCI4_2b')
parser.add_argument("--size", type=int, default=200, help='Time dimension (default: 200 for CogBCI)')  # CogBCI uses 200 timepoints
parser.add_argument("--rest", type=bool, default=False)
parser.add_argument("--cdist", type=float, default=1e-5)
parser.add_argument("--checkpoint", type=str)

parser.add_argument("--task", type=str, default='all', help='CogBCI tasks: all, 1, 2, 3, or comma-separated like "1,2"')
parser.add_argument("--max_rs_samples", type=int, default=50, help='Max resting state samples for testing (default: 50, use 0 for all)')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

torch.multiprocessing.set_sharing_strategy('file_system')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = device
test = list(map(int, args.test.split(',')))

# ===== Dataset Configuration =====
if args.data == 'BCI4_2b':
    test = list(map(int, args.test.split(',')))
    train = [idx for idx in list(range(1, 10)) if not idx in test]
    args.datanum = 2
    args.chnum = 3
    args.sessionnum = 5
    
elif args.data == 'OpenBMI':
    test = list(map(int, args.test.split(',')))
    train = [idx for idx in list(range(1, 55)) if not idx in test]
    args.datanum = 2
    args.chnum = 62
    args.sessionnum = 2
    
elif args.data == 'BCI4_2a':
    test = list(map(int, args.test.split(',')))
    train = [idx for idx in list(range(1, 10)) if not idx in test]
    args.datanum = 4
    args.chnum = 22
    args.sessionnum = 2
    
elif args.data == 'm3cv':
    test = list(map(int, args.test.split(',')))
    train = [idx for idx in list(range(1, 29)) if not idx in test]
    args.datanum = 11
    args.chnum = 62
    args.sessionnum = 1

elif args.data == 'cogbci':
    # CogBCI Configuration
    test = list(map(int, args.test.split(',')))
    train = [idx for idx in list(range(1, 30)) if idx not in test]  # 29 subjects (1-29)
    args.datanum = 6  # 6 classes
    args.chnum = 62    # 62 channels
    args.sessionnum = 1  # Session 1 only
    args.size = 200    # CogBCI has 200 timepoints @ 200Hz
    
    # Parse task IDs
    if args.task == 'all':
        target_task_id = [0,1]  # All tasks: Flanker, MATB, Nback
    else:
        target_task_id = list(map(int, args.task.split(',')))
    
    # Get file paths (only Session 1)
    task_h5_path = COGBCI_CONFIG['multitask_file']
    rest_h5_path = COGBCI_CONFIG['resting_file']
    
    print(f"\n{'='*60}")
    print(f"CogBCI Dataset Configuration (Session 1 only)")
    print(f"{'='*60}")
    print(f"Tasks: {target_task_id}")
    print(f"Test subject: {test}")
    print(f"Train subjects: {train}")
    print(f"Task data: {task_h5_path}")
    print(f"Rest data: {rest_h5_path}")
    print(f"{'='*60}\n")

args.train = train

# Handle multiproto parameter for Conformer model
if args.model == 'Conformer':
    args.multiproto = 1  # Default value
else:
    args.multiproto = 1

net = model.Distribution(args).to(device)
args.hyper = [1, 0.5, 0.05, 0]
args.hyper_finetune = [1, 0, 0, 0]
loss = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.learningrate)

# ===== Training Phase =====
if args.is_training == 'train':
    os.makedirs(args.checkpoint, exist_ok=True)
    
    if args.data == 'cogbci':
        # Use CogBCI data loader with leave-one-subject-out
        # For training: use all subjects except test subject
        # For validation: use a portion of training data
        
        # For simplicity, we use subject 1 as validation during training of others
        # Or use a split of the training subjects
        
        # Create dataset with all training subjects
        dataset = CogBCIDataset(
            task_h5_path=task_h5_path,
            rest_h5_path=rest_h5_path,
            target_subject_id=None,  # Load all subjects
            target_task_id=target_task_id,
            n_classes=args.datanum,
            normalize=True,
            normalize_per_channel=True,
            is_training='train'
        )
        
        # Filter to only include training subjects
        train_mask = np.isin(dataset.subject_ids, train)
        train_indices = np.where(train_mask)[0]
        
        # Use last training subject for validation
        val_subject = train[-1] if len(train) > 1 else train[0]
        val_mask = (dataset.subject_ids == val_subject)
        val_indices = np.where(val_mask)[0]
        
        # Remove validation subject from training
        train_mask = train_mask & (~val_mask)
        train_indices = np.where(train_mask)[0]
        
        print(f"\nTraining on {len(train_indices)} samples")
        print(f"Validation on subject {val_subject}: {len(val_indices)} samples")
        
        from torch.utils.data import Subset
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
        
    else:
        # Original placeholder for other datasets
        dataset = []  # Set dataset
        data_tr = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=8, drop_last=True)
        data_val = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=8)
    
    manage = utils.find_best_model(args.checkpoint, args)
    manage.code_copy(os.path.join(args.checkpoint, 'run'))

    for epoch in range(args.epoch):
        net.train()
        x, y, output, acc_epoch, loss_epoch, count_epoch = one_epoch_distribution(
            net, data_tr, optimizer, loss, args.hyper, args, train='train'
        )
        print('Epoch  : ', epoch + 1, 'Acc: ', round(acc_epoch, 7), 
              'Loss: ', np.round(loss_epoch['Total'], 5))

        net.eval()
        with torch.no_grad():
            x, y, output, acc_epoch, loss_epoch, count_epoch = one_epoch_distribution(
                net, data_val, optimizer, loss, args.hyper, args, train='val'
            )
            manage.update(net, args.checkpoint, epoch, acc_epoch, loss_epoch['CLS'])
            print('Val    : ', epoch + 1, 'Acc: ', round(acc_epoch, 7), 
                  'Loss: ', np.round(loss_epoch['Total'], 5))

    manage.training_finish(net, args.checkpoint)

# ===== Testing Phase =====
elif args.is_training == 'test':
    manage = utils.test_model(args.checkpoint, args, 'result')
    
    if args.data == 'cogbci':
        # Load test subject data
        dataset = CogBCIDataset(
            task_h5_path=task_h5_path,
            rest_h5_path=rest_h5_path,
            target_subject_id=test[0],  # Test subject
            target_task_id=target_task_id,
            n_classes=args.datanum,
            normalize=True,
            normalize_per_channel=True,
            is_training='test'
        )
        
        # Load resting data for test subject
        # max_samples: limit resting state samples to speed up testing
        max_rs = args.max_rs_samples if args.max_rs_samples > 0 else None
        dataset_RS = CogBCIRestDataset(
            rest_h5_path=rest_h5_path,
            target_subject_id=test[0],
            normalize=True,
            normalize_per_channel=True,
            n_classes=args.datanum,
            max_samples=max_rs
        )
        
    else:
        # Original placeholder
        dataset = []
        dataset_RS = []
    
    restore = torch.load(os.path.join(args.checkpoint, 'model-best.pth'), 
                         map_location=torch.device('cpu'))
    net.load_state_dict(restore, strict=True)

    for session in range(args.sessionnum):
        torch.cuda.empty_cache()
        net.load_state_dict(restore, strict=True)
        restore_before = net.state_dict()
        
        data_test_RS = DataLoader(dataset_RS, batch_size=1, shuffle=False, num_workers=8)
        RS, RS_update, RS_update_label = generate_signal_from_RS(
            net, data_test_RS, loss, args.hyper, args, train='train'
        )
        RS_update = torch.cat(RS_update, dim=0)
        RS = torch.cat(RS, dim=0)
        RS_update_label = torch.cat(RS_update_label, dim=0)
        net.load_state_dict(restore_before, strict=True)

        # Use only RS_update (generated pseudo-labeled signal)
        # RS_update has the same time dimension (200) as model input
        dataset_RS.testset[session]['x'] = RS_update
        dataset_RS.testset[session]['y'] = RS_update_label

        # Switch to dict mode for one_epoch_distribution
        dataset_RS.set_return_dict(True)
        
        net.train()
        data_test_RS = DataLoader(dataset_RS, batch_size=args.batch, shuffle=True, num_workers=8)
        net.load_state_dict(restore_before)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learningrate/10)
        kd_loss = torch.nn.KLDivLoss().to(args.device)
        
        for epoch in range(10):
            args.cdist = 0
            x, y, output, acc_epoch, loss_epoch, count_epoch = one_epoch_distribution(
                net, data_test_RS, optimizer, kd_loss, args.hyper_finetune, args, train='RS'
            )

        dataset.is_training = 'test'
        dataset.finetuning = False
        data_test = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
        
        net.eval()
        with torch.no_grad():
            x, y, output, acc_epoch, loss_epoch, count_epoch = one_epoch_distribution(
                net, data_test, optimizer, loss, args.hyper, args, train='val'
            )
            result_key = f"{test[0]}S{session}"
            print(f'{result_key}: ', count_epoch, 'Acc: ', round(acc_epoch, 7), 
                  'Loss: ', round(loss_epoch['Total'], 7))
            manage.total_result(result_key, count_epoch, round(acc_epoch, 7), 
                               round(loss_epoch['Total'], 7))
