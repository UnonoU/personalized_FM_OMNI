# CogBCI Dataset Usage for ResTL

## 概述

CogBCI 数据集加载器已为 ResTL 框架配置完成，**仅使用 Session 1 数据**。

## 数据文件

- **任务数据**: `/mnt/dataset4/fuxy/m3cv/processed_cogbci_multitask/cogbci_multitask_session1.h5`
- **静息态数据**: `/mnt/dataset4/fuxy/m3cv/processed_cogbci_session1/cogbci_resting.h5`

## 数据规格

| 参数 | 值 |
|------|-----|
| 被试数 | 29 |
| 类别数 | 6 |
| 通道数 | 62 |
| 时间点数 | 200 (@ 200Hz) |
| 任务类型 | Task 0 (labels 0,1,2), Task 1 (labels 10,11,12) |

## 标签映射

| 原始标签 | 映射后 | 任务 |
|----------|--------|------|
| 0, 1, 2 | 0, 1, 2 | Task 0 |
| 10, 11, 12 | 3, 4, 5 | Task 1 |

## 快速开始

### 方法一：单被试一键运行（推荐）

```bash
# 训练+测试被试1
python run_cogbci.py -g 0 -te 1 --task all -e 100 --checkpoint ./checkpoints/cogbci/s1

# 仅训练
python run_cogbci.py -g 0 -te 1 --task all -e 100 --checkpoint ./checkpoints/cogbci/s1 --skip_test

# 仅测试（已有训练好的模型）
python run_cogbci.py -g 0 -te 1 --task all --checkpoint ./checkpoints/cogbci/s1 --skip_train

# 使用特定任务
python run_cogbci.py -g 0 -te 1 --task 0 -e 100 --checkpoint ./checkpoints/cogbci/s1_task0

# 使用 EEG-Conformer 模型
python run_cogbci.py -g 0 -te 1 --task all -m Conformer -e 100 --checkpoint ./checkpoints/cogbci/s1_conformer
```

### 方法二：批量运行所有被试

```bash
# 运行所有 29 个被试（留一被试交叉验证）
python run_all_subjects.py -g 0 --all -e 100 --task all

# 运行特定被试
python run_all_subjects.py -g 0 --subjects 1,2,3,4,5 -e 100

# 使用 Conformer 模型
python run_all_subjects.py -g 0 --all -m Conformer -e 150
```

### 方法三：分步运行（原始方式）

```bash
# 训练
python main_cogbci.py -g 0 -d cogbci -m EEGNet -te 1 --task all -e 100 --checkpoint ./checkpoints/cogbci/s1

# 测试
python main_cogbci.py -g 0 -d cogbci -m EEGNet -te 1 --task all -t test --checkpoint ./checkpoints/cogbci/s1
```

## 参数说明

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `-te` | 测试被试ID | 1-29 |
| `--task` | 使用哪些任务 | `all`, `0`, `1`, `0,1` |
| `-m` | 模型 | `EEGNet`, `Conformer` |
| `-e` | 训练轮数 | 任意整数 |
| `-b` | 批次大小 | 任意整数 |
| `--checkpoint` | 检查点目录 | 路径 |

## 输出文件

运行后会在 checkpoint 目录生成：

```
checkpoints/cogbci/s1/
├── model-best.pth       # 最佳模型
├── model-last.pth       # 最后模型
├── log.txt              # 训练日志
├── result.txt           # 测试结果
└── run/                 # 代码备份
    ├── main_cogbci.py
    ├── cogbci_loader.py
    └── ...
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `run_cogbci.py` | 一键训练+测试脚本 |
| `run_all_subjects.py` | 批量运行所有被试 |
| `main_cogbci.py` | 原训练/测试脚本 |
| `cogbci_loader.py` | CogBCI 数据加载器 |

## 示例代码

```python
from cogbci_loader import CogBCIDataset, COGBCI_CONFIG

# 加载数据
dataset = CogBCIDataset(
    task_h5_path=COGBCI_CONFIG['multitask_file'],
    rest_h5_path=COGBCI_CONFIG['resting_file'],
    target_subject_id=1,      # 指定被试
    target_task_id=[0, 1],    # 使用所有任务
    n_classes=6,
    normalize=True
)

# 获取样本
sample = dataset[0]
print(sample['x_anc'].shape)  # (1, 200, 62)
print(sample['label'].shape)  # (6,)
```
