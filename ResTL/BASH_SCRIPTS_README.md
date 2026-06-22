# CogBCI Bash 脚本使用指南

## 脚本说明

| 脚本 | 功能 |
|------|------|
| `run_cogbci.sh` | 单个被试训练+测试 |
| `run_all_subjects.sh` | 批量运行所有29个被试 |
| `run_subjects_range.sh` | 运行指定范围的被试 |

## 使用方法

### 1. 单个被试 (`run_cogbci.sh`)

```bash
# 基本用法 (默认: subject=1, gpu=0, epochs=100, task=all)
bash run_cogbci.sh

# 指定被试、GPU和轮数
bash run_cogbci.sh 1 0 100

# 完整参数: subject gpu epochs task model batch_size
bash run_cogbci.sh 1 0 100 all EEGNet 128

# 使用 Conformer 模型
bash run_cogbci.sh 1 0 100 all Conformer 64

# 只使用 Task 0
bash run_cogbci.sh 1 0 100 0

# 只使用 Task 1
bash run_cogbci.sh 1 0 100 1
```

**参数顺序**: `subject_id` `gpu_id` `epochs` `task` `model` `batch_size`

### 2. 批量运行所有被试 (`run_all_subjects.sh`)

```bash
# 基本用法 (默认: gpu=0, epochs=100, task=all, model=EEGNet)
bash run_all_subjects.sh

# 指定GPU和轮数
bash run_all_subjects.sh 0 150

# 完整参数: gpu epochs task model batch_size
bash run_all_subjects.sh 0 100 all EEGNet 128

# 使用 Conformer 模型
bash run_all_subjects.sh 0 150 all Conformer 64
```

**参数顺序**: `gpu_id` `epochs` `task` `model` `batch_size`

**特点**:
- 自动运行所有29个被试
- 自动记录日志到 `checkpoints/cogbci/run_all_YYYYMMDD_HHMMSS.log`
- 自动汇总所有结果到 `checkpoints/cogbci/all_results_YYYYMMDD_HHMMSS.txt`
- 显示每个被试的运行时间和成功率

### 3. 指定范围被试 (`run_subjects_range.sh`)

```bash
# 运行被试 1-5
bash run_subjects_range.sh 1 5

# 运行被试 10-15，使用GPU 1
bash run_subjects_range.sh 10 15 1

# 完整参数: start end gpu epochs task model batch_size
bash run_subjects_range.sh 1 10 0 100 all EEGNet 128
```

**参数顺序**: `start_subject` `end_subject` `gpu_id` `epochs` `task` `model` `batch_size`

## 快速开始示例

```bash
cd /mnt/dataset4/yinuo/personlized_FM/1_rebuttal/MICCAI2024-ResTL-master/MICCAI2024-ResTL-master

# 示例1: 测试被试1，快速运行（10轮）
bash run_cogbci.sh 1 0 10

# 示例2: 运行前5个被试
bash run_subjects_range.sh 1 5

# 示例3: 运行所有被试，使用Conformer模型
bash run_all_subjects.sh 0 150 all Conformer

# 示例4: 只使用Task 0运行所有被试
bash run_all_subjects.sh 0 100 0
```

## 输出结构

```
checkpoints/cogbci/
├── subject_1/
│   ├── model-best.pth
│   ├── model-last.pth
│   ├── log.txt
│   └── result_subject1.txt
├── subject_2/
│   └── ...
├── run_all_20240125_143022.log      # 批量运行日志
└── all_results_20240125_143022.txt  # 结果汇总
```

## 查看结果

```bash
# 查看单个被试结果
cat checkpoints/cogbci/subject_1/result_subject1.txt

# 查看批量运行日志
cat checkpoints/cogbci/run_all_*.log | tail -50

# 查看所有结果汇总
cat checkpoints/cogbci/all_results_*.txt
```

## 注意事项

1. **Conda环境**: 脚本会自动激活 `EEGPT` 环境，请确保已安装
2. **GPU内存**: 如果显存不足，请减小batch size:
   ```bash
   bash run_cogbci.sh 1 0 100 all EEGNet 64
   ```
3. **断点续跑**: 如果某个被试失败，可以单独重新运行该被试
4. **日志**: 批量运行时日志会同时输出到屏幕和文件

## 常见问题

### Q: 如何只运行失败的被试？
```bash
# 例如被试3和7失败了
bash run_cogbci.sh 3 0 100
bash run_cogbci.sh 7 0 100
```

### Q: 如何并行运行多个被试？
```bash
# 终端1: 运行被试1-10
bash run_subjects_range.sh 1 10 0 &

# 终端2: 运行被试11-20
bash run_subjects_range.sh 11 20 1 &

# 终端3: 运行被试21-29
bash run_subjects_range.sh 21 29 2 &
```

### Q: 如何修改默认参数？
编辑脚本文件顶部的默认参数：
```bash
GPU=${1:-0}        # 修改默认GPU
EPOCHS=${3:-100}   # 修改默认轮数
```
