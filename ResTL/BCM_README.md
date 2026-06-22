# BCM (Baseline Correction Module) Implementation

基于 Kwak et al. 2023 论文实现的基线校正模块，与 ResTL 框架集成。

## 核心特点

| 特性 | BCM | ResTL |
|------|-----|-------|
| 方法 | 显式估计增益矩阵和背景特征 | 隐式特征分解 |
| 训练数据 | 需要多个被试的静息态+任务态 | 多个被试 |
| 核心损失 | 三元组损失 (Triplet Loss) | 分布学习 + 对比学习 |
| 测试适应 | 1分钟基线平均 | 伪标签生成+微调 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `bcm_model.py` | BCM 核心实现（BaselineCorrectionModule + 损失函数 + 完整模型） |
| `run_bcm_cogbci.py` | BCM 训练/测试主脚本 |
| `run_bcm.sh` | 单被试运行脚本 |

## 快速开始

### 1. 运行单个被试

```bash
# 基本用法
bash run_bcm.sh 1 0 100

# 完整参数: subject gpu epochs task model batch_size m alpha
bash run_bcm.sh 1 0 100 all EEGNet 64 5.0 0.1

# 使用 Conformer
bash run_bcm.sh 1 0 100 all Conformer 32 5.0 0.1

# 只使用 Task 0
bash run_bcm.sh 1 0 100 0 EEGNet 64 5.0 0.1
```

### 2. 直接使用 Python

```bash
# 训练+测试
python run_bcm_cogbci.py \
    -g 0 \
    -te 1 \
    -m EEGNet \
    -e 100 \
    -b 64 \
    --task all \
    --m 5.0 \
    --alpha 0.1 \
    --checkpoint ./checkpoints/bcm/s1

# 仅训练
python run_bcm_cogbci.py \
    -g 0 -te 1 -m EEGNet -e 100 \
    --checkpoint ./checkpoints/bcm/s1 \
    --skip_test

# 仅测试
python run_bcm_cogbci.py \
    -g 0 -te 1 -m EEGNet \
    --checkpoint ./checkpoints/bcm/s1 \
    --skip_train
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-g` | 0 | GPU ID |
| `-te` | 1 | 测试被试ID |
| `-m` | EEGNet | 骨干网络 (EEGNet/Conformer) |
| `-e` | 100 | 训练轮数 |
| `-b` | 64 | Batch size |
| `--task` | all | 任务选择 (all/0/1/0,1) |
| `--m` | 5.0 | 特征幅度约束 (CogBCI推荐5.0) |
| `--alpha` | 0.1 | 不变损失权重 (EEGNet: 0.1, MIN2Net: 1.0) |

## 核心模块说明

### BaselineCorrectionModule

```python
class BaselineCorrectionModule(nn.Module):
    """
    基线校正模块 - 估计被试特定的增益矩阵和背景特征
    
    训练阶段：单样本配对 (随机选择基线)
        s_target = U( (G^i)^+ * x_task - E[s_back^i] )
    
    测试阶段：多基线平均 (1分钟静息态分割平均)
        s_target = U( mean(G^+) * x_task - mean(s_back) )
    """
```

### SubjectInvariantLoss

```python
class SubjectInvariantLoss(nn.Module):
    """
    三元组损失 - 使用余弦距离
    
    对于每个anchor：
        - Hardest Positive: 最远的同类样本
        - Hardest Negative: 最近的不同类样本
        - Loss = max(0, d(a,p) - d(a,n))
    """
```

## 与 ResTL 对比

### 相同点
- 都使用静息态 EEG 进行被试自适应
- 都可以使用 EEGNet/Conformer 作为骨干网络
- 都在 CogBCI 数据集上测试

### 不同点

| 维度 | BCM | ResTL |
|------|-----|-------|
| 核心思想 | 显式去除被试特定成分 | 特征分解为任务相关+被试相关 |
| 训练方式 | 需要配对 (任务, 基线) | 三元组采样 |
| 测试适应 | 1分钟基线直接平均 | 伪标签生成+网络微调 |
| 损失函数 | CE + α * Triplet | CE + Dist + Sub + Sample |
| 被试间迁移 | 显式不变性约束 | 隐式分布学习 |

## 预期结果

在 CogBCI 数据集上的预期性能：

```
Subject 1 | Accuracy: ~65-75%  (取决于任务划分)
Subject 2 | Accuracy: ~65-75%
...
```

## 注意事项

1. **数据准备**: BCM 需要配对数据 (任务样本 + 基线样本)，训练时随机配对，测试时使用1分钟基线
2. **超参数敏感性**:
   - `m`: 特征幅度，影响归一化后的特征尺度
   - `alpha`: 三元组损失权重，过大可能导致训练不稳定
3. **显存使用**: BCM 需要同时加载任务和基线数据，batch size 建议比 ResTL 小
4. **训练稳定性**: 三元组损失在早期可能不稳定，建议观察几轮后再判断

## 常见问题

### Q: BCM 和 ResTL 哪个更好？
BCM 和 ResTL 是针对不同场景的两种方法：
- BCM: 显式建模被试变异，需要多个被试训练，适合被试间迁移
- ResTL: 隐式分解特征，单被试即可适应，适合快速个性化

### Q: 如何调整超参数？
```bash
# 如果训练不稳定，减小 alpha
bash run_bcm.sh 1 0 100 all EEGNet 64 5.0 0.05

# 如果特征幅度不合适，调整 m
bash run_bcm.sh 1 0 100 all EEGNet 64 3.0 0.1
```

### Q: 可以在其他数据集上使用吗？
可以，需要调整参数：
- OpenBMI: `m=1.0` (论文建议)
- BCIC IV 2a/2b: `m=5.0`
- 时间维度不同需要调整特征维度计算

## 引用

```bibtex
@article{kwak2023subject,
  title={Subject-Invariant Deep Neural Networks Based on Baseline Correction for EEG Motor Imagery Brain-Computer Interfaces},
  author={Kwak, Yeonghun and Kim, Joongwoo and Kim, June Sic},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023}
}
```
