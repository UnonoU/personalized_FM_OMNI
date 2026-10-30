# lwh.bdf 的 LaBraM 预处理

## 原始数据交互分析 Notebook

打开 [notebooks/lwh_bdf_exploration.ipynb](notebooks/lwh_bdf_exploration.ipynb)，选择 **Python (EEGPT)** 内核并从上到下运行。已配置本次 `lwh.bdf` 的绝对路径，包含通道表与电极位置、交互波形、事件时间线、质量筛查、PSD、频段热图、时频图、固定窗口和事件对齐切块，以及已有 LaBraM HDF5 的对照。每幅图均附可修改的绘图代码。

默认只读取需要的片段，质量与频谱统计取均匀分布的 5 个 60 秒片段；开启 `FULL_QUALITY_SCAN` 才会逐块扫描全记录。`EVENT_LABELS` / `STATE_PAIRS` 等待真实实验编码表，未确认的静息/任务状态保留 `unknown`。原始文件只读，表格导出默认关闭。

交互运行依赖已安装在 EEGPT 环境。重建环境时可运行：

```bash
python -m pip install -r notebooks/requirements.txt
python -m ipykernel install --user --name eegpt --display-name "Python (EEGPT)"
```

## 运行预处理脚本

在项目根目录运行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate EEGPT
python preprocess_lwh_labram.py --source-unit uV
```

默认输入为已挂载的 `~/rustfs/oi-data/EMOVID_RAW/20261022161317_20261022_lwh/20261022161317_20261022_lwh/lwh/lwh.bdf`，输出为本地 `processed/lwh_labram.hdf5`。输入只读；已有输出不会被覆盖，重跑时请使用新的 `--output` 路径。本次实际生成的文件约 234 MiB，会占用本地空间。

## 处理规则与官方依据

- 根据 [LaBraM README](https://github.com/935963004/LaBraM#running-experiments) 和 [eegUtils.py](https://github.com/935963004/LaBraM/blob/main/dataset_maker/shock/utils/eegUtils.py)：保留 EEG，0.1–75 Hz FIR 带通，50 Hz 陷波，降采样到 200 Hz，以微伏保存。
- 移除本记录的 ECG 和 4 路 EOG。保留真实存在的 59 路 EEG，不补造缺失电极，也不强制凑成 62/64 路。
- 依据 [utils.py 的 standard_1020/get_input_chans](https://github.com/935963004/LaBraM/blob/main/utils.py) 排序并保存位置索引。`input_chans[0]` 为 CLS 的 0，后续为各 EEG 电极的 1 起始位置索引。词表已与官方代码逐项核对。
- 使用原始采集参考；不额外执行平均参考、ICA、伪迹剔除或 Z-score。官方通用预处理流程没有要求这些附加步骤。
- 依据 [官方 HDF5 制作脚本](https://github.com/935963004/LaBraM/blob/main/dataset_maker/make_h5dataset_for_pretrain.py) 和 [data_processor/dataset.py](https://github.com/935963004/LaBraM/blob/main/data_processor/dataset.py)，保存连续的 `(channels, time)` 数据，而不是预先复制出重叠窗口。
- 官方 CNT 示例丢弃末尾 10 秒。本脚本默认保留整个记录以保留全部时间信息；需要复用这一处理时可加 `--trim-tail-seconds 10`。

## 单位校准

这份 BDF 的物理单位字符串是 `uV/mm`，用户已确认其实际数值按微伏标定。当前 MNE 1.4.2 不将该字符串识别为微伏，因此脚本用 `--source-unit uV` 明确声明真实单位，将 MNE 内部数值先校正为伏，再以微伏导出。若原文件已使用标准 `uV`，同一参数不会再次乘以百万。该路径已通过与标准微伏 BDF 的数值对照测试。

对其他文件若不知道其真实标定，不要照搬这个声明。`--source-unit auto` 只接受标准 V/mV/uV 单位，遇到 `uV/mm` 会停止，不会凭振幅猜测单位。

## 输出结构

```text
/                         attrs: processing_status, metadata_json
└── lwh/
    ├── eeg               float32，(59, 1036200)，单位 uV
    │                     attrs: chOrder, rsFreq=200, lFreq=0.1, hFreq=75
    ├── input_chans       60 个索引，包括 CLS
    └── events/
        ├── description
        ├── onset_original_seconds
        ├── onset_output_seconds
        ├── duration_seconds
        ├── sample_original
        ├── sample_output
        └── in_output
```

事件码保持原始字符串，不解释成情绪标签。原始时间以秒保存；`sample_output` 使用 200 Hz 下的最近采样点，索引从 0 开始。裁剪处理时仍保存原始事件，并通过 `in_output` 标记事件起点是否位于输出内，避免静默丢失事件。输出不包含虚构的分类标签，也不自动划分训练/验证/测试集。

## 模型输入

[官方训练代码](https://github.com/935963004/LaBraM/blob/main/engine_for_pretraining.py) 在模型输入前执行 `/100` 并按每 200 个采样点分成一个 patch。HDF5 中的值没有提前除以 100，也没有 Z-score；使用官方训练代码时不要再次归一化。

在本项目中手工取一个 8 秒窗口：

```python
import h5py
import torch

with h5py.File("processed/lwh_labram.hdf5", "r") as f:
    eeg = f["lwh/eeg"][:, :1600]  # 59 通道 × 8 秒 × 200 Hz
    ch_names = f["lwh/eeg"].attrs["chOrder"].tolist()
    input_chans = f["lwh/input_chans"][:].tolist()

x = torch.from_numpy(eeg.copy()).float().reshape(1, 59, 8, 200) / 100.0
# model(x, input_chans=input_chans)
```

本项目修改版模型实例化示例：

```python
from src.modeling_finetune import labram_base_patch200_200

model = labram_base_patch200_200(
    num_classes=0, num_patches_per_channel_input=8, init_values=0.1
)
```

这个实例化示例不加载预训练权重。实际训练或提取预训练特征时还需加载对应 checkpoint；随机初始化前向测试仅用于确认接口和张量形状。

输出兼容官方 `SingleShockDataset`/`ShockDataset` 的预训练结构。本项目当前 `src.data_loader.EEGDataset` 要求另一种含 labels/subject_ids 的 HDF5，并默认 Z-score，不能直接用它读取本输出。监督微调需另外提供事件/试次与情绪类别的对应表。

## 检查与验证

只检查文件头，不扫描整个远程信号：

```bash
python preprocess_lwh_labram.py --inspect
```

处理一个至少 34 秒的片段用于调试：

```bash
python preprocess_lwh_labram.py --source-unit uV \
  --start-seconds 600 --duration-seconds 60 \
  --output processed/lwh_labram_60s.hdf5
```

0.1 Hz FIR 滤波器较长，脚本拒绝少于 34 秒的片段。正常分析应先过滤整个连续记录再取窗口，避免每个短窗口单独滤波带来的边缘效应。调试片段的边缘结果不等同于完整记录处理后再裁剪。

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests \
  -p test_preprocess_lwh_labram.py -v
```

测试覆盖单位缩放、10 Hz 保留/50 Hz 抑制/高频混叠抑制、通道索引、200 Hz 事件对齐、裁剪及不覆盖已有输出。真实记录的完整性、官方读取器和模型前向检查结果见 `analysis/lwh_bdf/preprocessing_validation.json`。

读取挂载的 BDF 仍会产生网络流量；MNE 打开含注释的 BDF 时即使 `preload=False` 也可能扫描整个文件。完整记录在内存中滤波，本次处理约需数 GB 内存；没有把原始 BDF 另存到本地。
