EEG 预处理代码
==============

代码按用途分在 ``common/`` 和 ``labram/`` 两个子目录。
所有命令从项目根目录运行，使用已有的 EEGPT 环境。

目录结构
--------

::

    preprocessing/
      common/                         # 通用预处理与共享 S3 工具
        emovid_pipeline.py            # EMOVID 统一入口
        preprocess_emovid_rules.py    # 读取、滤波、重采样、切窗、质控、归一化
        emovid_segments.py            # 事件、时钟对齐和分段
        batch_emovid_segments.py      # 批量扫描与标签日志匹配
        validate_emovid_rules.py      # 输出校验
        emovid_storage.py             # S3 输出和挂载
        read_rustfs.py                # S3 客户端和凭据配置
        s3_hdf5.py                    # 通用 HDF5 Range 读取
        mvnn_process/                 # 图片刺激预处理与 MVNN 参考代码
      labram/                         # LaBraM 数据转换与模型接口
        emovid_labram.py              # EMOVID → LaBraM S3 HDF5
        labram_dataset.py             # S3 数据加载器
        labram_contract.py            # 通道映射、格式和接口校验
        preprocess_lwh_labram.py      # 单份 BDF → 连续 EEG HDF5

通用 EMOVID 流程
---------------

::

    python -m preprocessing.common.emovid_pipeline --help
    python -m preprocessing.common.emovid_pipeline --resume

    from preprocessing.common.emovid_pipeline import PreprocessingOptions, preprocess_dataset
    from preprocessing.common import preprocess_emovid_rules as prep

说明见 ``../EMOVID_PREPROCESSING.rst`` 和 ``../EMOVID_MAPPING.rst``。
演示见 ``../notebooks/emovid_bdf_preprocessing_walkthrough.ipynb``。

LaBraM 转换与读取
-----------------

复用 EMOVID 的分段、标签和质控，从原始 S3 EEG 重建裁剪前的 μV，
逐窗核对原 Lance 后写入 ``s3://personalized-fm/processed_labram/``。

::

    python -m preprocessing.labram.emovid_labram
    python -m preprocessing.labram.emovid_labram --recording 20261022161317_20261022_lwh

    from preprocessing.labram.labram_dataset import S3LaBraMDataset

加载器默认筛选通过 QC 的情绪窗口，输出 ``[59,3,200]`` 的 μV/100 张量；
``read_window`` 提供未缩放的 μV 分析数据。
分析见 ``../notebooks/emovid_labram_s3_walkthrough.ipynb``。
原有的单 BDF 连续信号转换入口为：

::

    python -m preprocessing.labram.preprocess_lwh_labram --source-unit uV

说明见 ``../PREPROCESS_LABRAM.md``，探索 Notebook 为
``../notebooks/lwh_bdf_exploration.ipynb``。

MVNN 参考流程
-------------

``common/mvnn_process/`` 保留图片刺激预处理与 MVNN 参考实现。
原实验路径和外部资源要求见 ``../EMOVID_PREPROCESSING.rst``；
其中 ``mvnn.py`` 还需要原项目配套的 ``contracts.py``（当前项目未提供）。

Notebook 与检查
---------------

Notebook 保留在 ``notebooks/``，从项目根目录或 ``notebooks/`` 启动内核均可。
附加依赖仍由 ``notebooks/requirements.txt`` 管理。

::

    python -m unittest discover -s tests -v
