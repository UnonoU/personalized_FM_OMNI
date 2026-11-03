# EMOVID 的 LaBraM 预处理与 S3 分析

## 在 EMOVID 流程上生成 LaBraM 数据（S3）

带情绪标签的 EMOVID 数据使用新增入口：

```bash
conda activate EEGPT

# 前置：已有 EMOVID S3 输出；尚未生成时先运行
python -m preprocessing.common.emovid_pipeline --resume --no-mount

# 全量转换，默认复用已验证的输出
python -m preprocessing.labram.emovid_labram

# 也可只转换一份录制
python -m preprocessing.labram.emovid_labram --recording 20261022161317_20261022_lwh
```

默认从 `s3://personalized-fm/processed/manifest.json` 读取 EMOVID 索引、分段报告、
质控和固定版本的 Lance 表，生成到 `s3://personalized-fm/processed_labram/`。
可通过 `--source-uri`、`--output-uri`、`--profile` 更改位置和凭据 profile。
`--no-resume` 强制重算；临时 HDF5 写入系统临时目录，上传、校验后自动清除，
可用 `--work-dir` 指定暂存目录。读取和分析均通过 S3 API，不依赖本地挂载。

这条派生流程保留 **EMOVID 的 0.3–75 Hz 带通、50 Hz 陷波、200 Hz 重采样、
视频/休息边界内的完整 3 秒窗口及原始标签/QC**。
与下方旧的连续 HDF5 脚本相比，它不会改用 0.1 Hz 下限，也不会重新解释实验标签。

原 Lance 保存的是 `clip(x_uV/200, -1, 1)`，因此转换器会从原始 S3 BDF/EDF
按块重新读取和滤波，取得**裁剪前的 μV**；不会用 `normalized*200` 冒充恢复原信号。
每个窗口重新按 EMOVID 规则编码后，必须与固定版本 Lance 的波形和质控完全一致，
才允许发布结果。原始对象按 ETag/VersionId 固定快照，软件环境和数值配置也会核对。

S3 输出结构：

```text
s3://personalized-fm/processed_labram/
  <采集目录名>.<内容哈希前16位>.labram.hdf5
  <采集目录名>.summary.json
  <采集目录名>.validation.json
  manifest.json
  summary.json
  progress.json
```

HDF5 包含 `data: [N,59,600]`（float32、μV、无幅度裁剪）、`labels`、
`emotion_labels`（rest 为 -1）、`qc_pass`、`sample_ids`、`subject_ids`、`session_ids`、
`segment_type`、`rows_json`（完整原始窗口元数据）、`channel_names` 和 `input_chans`。
通道按 LaBraM 词表重排，位置索引与波形顺序一致。
所有窗口都保留，主标签仍是 negative=0、neutral=1、positive=2、rest=3。
S3 上传后逐字节流式核对 SHA-256；summary 和 manifest 作为就绪索引，引用已验证的内容文件。

直接从 S3 读取模型输入：

```python
from torch.utils.data import DataLoader
from preprocessing.labram.labram_dataset import S3LaBraMDataset

with S3LaBraMDataset() as dataset:
    # 默认只选 qc_pass=true 的 emotion 窗口。
    batch = next(iter(DataLoader(dataset, batch_size=8)))
    x = batch['eeg']       # [B,59,3,200]，已经完成 μV/100
    y = batch['label']     # 0/1/2，torch.long
    input_chans = dataset.input_chans  # 包含 CLS，长度60
    # logits = model(x.to(device), input_chans=input_chans)
```

模型输入已除以 100，直接调用模型时不要再次缩放。若使用官方训练循环（循环内已有
`/100` 和 patch 重排），应传入 `read_window(i)['eeg_uv']` 的 `[59,600]` μV 数据，
并按官方循环要求提供标签，而非直接传入上面的 `batch['eeg']`。
本项目 `src.data_loader.EEGDataset` 默认 Z-score，不应替代这个 S3 加载器。

本项目模型实例化时设置 `num_patches_per_channel_input=3`、`num_classes=3`。
使用预训练模型时还需按所用 checkpoint 处理时间嵌入等参数的形状并加载权重；
转换器与加载器只负责输入数据，不会自动加载模型权重。

物理单位下的分析与元数据读取：

```python
with S3LaBraMDataset(qc_only=False, segment_type=None) as dataset:
    window = dataset.read_window(0)
    eeg_uv = window['eeg_uv']     # [59,600]，用于波形/PSD
    names = window['channel_names']
    rows = dataset.metadata()   # 包括原被试、session、video 和 sample_id
```

加载器使用 S3 Range 请求和有上限的内存缓存，不下载完整 HDF5 到本地。
支持 PyTorch DataLoader；每个 worker 单独建立客户端及文件句柄。
训练/验证/测试划分仍由调用方按被试、session 或视频分组处理，`split` 保留 `unspecified`。

逐步分析见 [EMOVID → LaBraM S3 Notebook](notebooks/emovid_labram_s3_walkthrough.ipynb)，
包含标签/QC 分布、μV 波形和 PSD、与原 Lance 的数值对照、模型输入批次。
依赖使用现有 EEGPT 环境中的 NumPy、SciPy、MNE、Lance、PyArrow、h5py、boto3、PyTorch；
Notebook 附加依赖仍在 `notebooks/requirements.txt`，无需额外安装 S3 文件系统库。

本次实际运行（2026-11-03）已转换 23 份录制、3026 个窗口，其中情绪窗口 2843 个、
休息窗口 183 个；2758 个窗口通过 QC，其中可供默认三分类加载器读取的情绪窗口为 2592 个。
所有窗口都通过转换前的原 Lance 精确核对，全部 S3 HDF5 通过上传后 SHA-256 核对，
再次全量运行全部命中缓存。上述 Notebook 已实际执行并保存表格与图像。
另已验证双 worker 的 S3 DataLoader，以及随机初始化 LaBraM base 的 `[2,59,3,200] → [2,3]`
有限值前向；该检查仅验证接口，不评估预训练特征或分类性能。

## 原有连续 HDF5 探索流程

以下 `preprocess_lwh_labram.py` 用于单份 BDF 的连续信号探索；带标签的 S3 派生流程见上文。

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
python -m preprocessing.labram.preprocess_lwh_labram --source-unit uV
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
python -m preprocessing.labram.preprocess_lwh_labram --inspect
```

处理一个至少 34 秒的片段用于调试：

```bash
python -m preprocessing.labram.preprocess_lwh_labram --source-unit uV \
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
