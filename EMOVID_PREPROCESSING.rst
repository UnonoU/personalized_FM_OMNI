EMOVID 初步预处理与 official_subjects 流程对比
=============================================

代码与输入
----------

统一入口为 ``emovid_pipeline.py``，一次完成 S3 扫描、事件匹配、分段、预处理、
质控校验及存储。``preprocess_emovid_rules.py`` 保留数值处理函数，其命令行也转入
同一个统一入口。默认扫描 ``s3://oi-data/EMOVID_RAW/``；也可以指定单个 BDF。
使用 ``--from-manifest`` 时可复用已有映射，不需要重复搜索标签日志。
它对通过分段检查的情绪片段和休息片段读取实际 EEG、滤波、重采样、切窗、归一化，
直接写入 ``s3://personalized-fm/processed/``。
已排除的短休息、未闭合片段、没有日志或没有 BDF 导出的采集不会混入输出。
本地 ``/home/zhangyn/rustfs/`` 挂载到此 S3 processed 前缀，是相同对象的文件系统视图。
处理时仅在 ``/tmp/emovid-preprocess`` 暂存单份录制的数组，完成后自动清除，
不再向项目 ``processed/`` 目录累积波形和历史检查点。
旧的 ``processed/emovid_rules_v1`` 已在逐份校验后替换为指向
``/home/zhangyn/rustfs`` 的兼容软链接，约 1.8 GB 本地生成副本已清理。

参考固定在 dataset-rules commit ``502c9d1dea27d9930813aa11076ef086e4ba2de3``：

* 参数示例：https://github.com/Omni-Intel/dataset-rules/blob/502c9d1dea27d9930813aa11076ef086e4ba2de3/examples/eeg-pretrain/metadata.toml
* 基础表规范：https://github.com/Omni-Intel/dataset-rules/blob/502c9d1dea27d9930813aa11076ef086e4ba2de3/rules/lance-rules.cn.md
* 元数据规范：https://github.com/Omni-Intel/dataset-rules/blob/502c9d1dea27d9930813aa11076ef086e4ba2de3/rules/metadata-rules.cn.md

该仓库是规范和 TOML 示例，不包含可直接执行的预处理代码，也不要求所有 EEG 数据集
必须使用某一组滤波参数。本实现参考示例的滤波、重采样和归一化参数，按用户要求
采用 3 秒无补零窗口，另外明确选择了
MNE 零相位 FIR 和 SciPy polyphase 重采样。由于需要保留情绪标签，输出选择
``profile = "base"``，不声明为禁止标签的 ``pretrain`` profile。
当前数据版本为 ``0.2.0``；原有 10 秒补零规则已替换为严格的 3 秒完整窗口。

逐步运行与质量可视化 Notebook
----------------------------

``notebooks/emovid_bdf_preprocessing_walkthrough.ipynb`` 以
``20261022161317_20261022_lwh/lwh.bdf`` 为例，保存实际运行的表格与图像。
在 VS Code 或 Jupyter 中选择 ``Python (EEGPT)`` 内核，依次运行即可；需要
本机 ``rustfs`` profile 和该 BDF 已生成的同名 Lance、summary、segments 文件。
从项目根目录或 notebooks 目录启动均可，不依赖本地 S3 挂载。

Notebook 重新搜索标签日志并验证映射，展示设备时间、日志时钟残差、绝对时间、
标签与 3 秒窗边界；随后拆开显示原始 μV、带通、50 Hz 陷波、200 Hz 重采样、
裁剪前质控及归一化。包含电极位置、波形、滤波响应、PSD、时频图、窗口质量分布、
通道异常热图和通过/失败片段对照。图中的电极坐标来自标准模板。

其中第 2.1–2.2 节单独展示“日志与设备墙钟差”：对每个已匹配触发，比较
``BDF 文件头起点 + 设备事件秒数`` 与 ``日志绝对时间零点 + relative_time_sec``，
绘制有符号差值随时间的变化、各 run 的分布，并与拟合后的时钟残差对照。
生产报告的 ``wall_clock_difference_seconds`` 是各 run 的 ``median(abs(设备−日志))``，
不是最大差值或拟合残差；30 秒用于候选日志匹配，不代表片段边界允许误差 30 秒。
``wall_clock_events`` 保存逐事件绝对时间与差值，``wall_clock_runs`` 保存汇总，
``wall_clock_origins`` 记录用于恢复日志墙钟零点的事件与时间来源。

第 5.1 节区分“去掉直流/慢变化造成整体位置改变”和“波动幅度改变”，显示同一通道、
同一个 3 秒窗在原始、带通、陷波、降采样各阶段的均值、峰峰值及标准差，并单独叠加
陷波后降采样前后的 μV 波形。原始信号与完整处理结果的图例明确包含带通、陷波和重采样；
这些图尚未进行除以 200 的归一化。原始波形减均值的辅助图仅用于显示，不改变生产处理。

默认重新计算该 BDF 全部可用标签窗口，并逐窗核对 S3 Lance 的波形和质控字段。
统计明确区分完整录制、有标签完整窗口和通过质控的窗口，不把未标记区间当作 rest。
只按块读取原始数据，不保存新的波形副本、不覆盖生产结果。需要更换示例时修改
``RAW_URI``；需要交互选择窗口/通道时启用 ``ENABLE_WIDGETS``，默认静态图不依赖控件。
这是当前 EMOVID 流程的演示，与旧的 ``lwh_bdf_exploration.ipynb`` 中 LaBraM 探索流程不同。

处理步骤
--------

1. 检查 S3 原始对象的 ETag、大小与分段时快照一致，并检查 EDF/BDF 的最终文件长度。
   仅通过 Range 请求读取需要的波形块及上下文，不下载完整原始文件。
2. 按规范化通道名选择并排序 59 个头皮 EEG 通道。通道名转大写，剥离 ``EEG `` 前缀；
   ECG、EOG、触发和 annotation 通道不进入模型数据。缺失或重复必需通道会报错。
3. 解码有符号 24 位 BDF 或 16 位 EDF，应用每道 physical/digital min/max 线性标定。
   随后根据显式单位声明换算为 μV，避免非标准单位被读取器误当作 V 而放大一百万倍。
4. 保留采集参考；不做 CAR、ICA、刺激前基线校正或 MVNN。
5. MNE ``filter_data``：0.3–75 Hz，零相位 FIR、Hamming、firwin。
   然后 ``notch_filter``：50 Hz、0.25 Hz notch width、1 Hz transition bandwidth。
6. SciPy ``resample_poly``：1000→200 Hz 时使用 up=1、down=5。
   各块起点与全局重采样网格对齐，防止独立块的采样点偏移。
7. 在每个完整 emotion/rest 片段内部切不重叠 3 秒窗，不跨视频或标签边界。
   不足 3 秒的尾段直接舍弃，完全不补零；每窗恰好 600 个采样点。
   例如 10 秒视频得到 3+3+3 秒，剩余 1 秒舍弃；相邻同标签视频也不合并。
8. 对真实的 3 秒 EEG 计算质控指标：有限值检查、大于 ±200 μV 的样本占比、每道标准差。
   默认 clip fraction 超过 1%，或任一道标准差不超过 1e-4 μV 时 ``qc_pass=false``。
   阈值是本项目的初步选择，不是 dataset-rules 强制指定的阈值。
9. 按 ``clip(x_μV / 200, -1, 1)`` 归一化并转 float32；每行逻辑形状为 ``[59, 600]``。
   ``qc_pass=false`` 的窗口保留诊断信息，训练时应显式过滤。

处理块默认含左右各 15 秒上下文。当前 1000 Hz 下带通核为 11001 点、陷波核为
6601 点，合成半支持宽度为 8.8 秒；额外上下文覆盖重采样核。
因此切块内部结果与连续处理一致，而不是在每个短窗口边缘重新起滤波器。
原始录制的真实首尾使用 reflect_limited 边界处理。

单位声明
--------

本批 BDF 物理单位字段写为非标准的 ``uV/mm``。用户已于 2026-11-03 明确确认，
这些物理数值按 μV 理解。因此当前状态为 ``source_unit_status = "confirmed"``，
复现时使用 ``--source-unit uV --unit-status confirmed``。
``metadata.toml``、处理配置、运行摘要和检查点已同步更新，并记录确认来源。

单位确认时只更新了状态与来源记录，未改变当时的波形和质控结果。
当前 3 秒版本沿用已确认的 μV 标定，并针对新的窗口重新计算质控。
该确认来自用户，不代表脚本额外执行了设备物理校准测量。
若实际单位或数值标定后续发生变化，应指定正确参数重跑。
不要将归一化后的值再次当作 μV 输入另一个除幅度流程。

输出与读取
----------

::

    /home/zhangyn/rustfs/                 # s3://personalized-fm/processed/
      20261022161317_20261022_lwh.lance/
      20261022161317_20261022_lwh.metadata.toml
      20261022161317_20261022_lwh.versions.toml
      20261022161317_20261022_lwh.segments.json
      20261022161317_20261022_lwh.summary.json
      20261022161317_20261022_lwh.validation.json
      ...                               # 其他采集目录同样命名
      manifest.json                     # 所有采集、表路径与处理状态的索引
      summary.json                      # 全量统计
      preprocessing_config.json         # 单位、参数与软件版本
      validation.json                   # 全量校验状态
      progress.json                     # 中断进度

命名使用 ``EMOVID_RAW/`` 下的最外层采集目录名。例如：

::

    输入：s3://oi-data/EMOVID_RAW/20261022161317_20261022_lwh/20261022161317_20261022_lwh/lwh/lwh.bdf
    输出：s3://personalized-fm/processed/20261022161317_20261022_lwh.lance
    挂载：/home/zhangyn/rustfs/20261022161317_20261022_lwh.lance

不会用 ``lwh.lance`` 或哈希名作为这些表的名称。同一采集目录下若发现多份不同的有效
BDF，程序会报告命名冲突，不会静默覆盖；同一 BDF 的重复导出先按内容标识去重。
每份录制有自己的 Lance 表及元数据，顶层 manifest 索引这些表。

主表 ``label`` 使用 negative=0、neutral=1、positive=2、rest=3。
其中 rest 是独立的采集状态，不是第四种情绪；``emotion_label`` 对 emotion 保持原来的
0/1/2 三分类，对 rest 为空。这样主表满足基础规范的非空主标签要求。
训练三分类情绪模型时选 ``segment_type = 'emotion'`` 并使用 ``emotion_label``。

每行 ``shape = original_shape = [59, 600]``，``valid_length = 35400``，
``padding = false``，``time_padding = false``，所有 600 个时间点均来自真实 EEG。

::

    import lance
    import numpy as np

    dataset = lance.dataset('/home/zhangyn/rustfs/20261022161317_20261022_lwh.lance')
    scanner = dataset.scanner(
        filter="qc_pass = true AND segment_type = 'emotion'", batch_size=8)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            eeg = np.asarray(row['data'], dtype=np.float32).reshape(row['shape'])
            label = row['emotion_label']

窗口同时保存 BDF URI、events URI、session/run/video/trial/attempt、原始采样点、
200 Hz 采样点、有效时长、clip fraction 和质控原因。所有采样点从完整原始录制起点计数。
``split`` 暂为 ``unspecified``，没有擅自随机划分训练集。后续应按被试、session 或
video 分组划分，尤其不能把同一视频的邻近窗口随机分到训练集和测试集。

与 preprocess_official_subjects.py 的差异
---------------------------------------

下表比较当前目录中的 ``processed/preprocess_official_subjects.py``；
其 ``PREPROCESSING_CONTRACT``、``preprocess_session`` 和 derived 分支分别定义这些行为。
本实现没有修改该脚本或 ``processed/mvnn.py``。

====================  ========================================  ==============================================
项目                  本次 EMOVID 初步预处理                     preprocess_official_subjects.py
====================  ========================================  ==============================================
任务与标签            情绪视频三分类，另保留休息状态             图片诱发响应；标签来自 image_id/index
输入                  S3 EDF/BDF 及已对齐的片段                  sample-major float32 波形文件、sidecar 与图片事件表
同步                  EMOVID 150/151、rest 日志与设备标记匹配    hardware_verified、offset、epoch boundary，并核对触发道 101
通道                  按名称选 59 EEG，道名大写、显式重排        校验前 59 道必须等于 FULL59 的既定顺序
参考                  保留采集参考                               在连续数据上对 59 道做 CAR
带通                  0.3–75 Hz，零相位 FIR                      0.5–40 Hz，4 阶 Butterworth SOS，sosfiltfilt
工频                  50 Hz 显式 FIR 陷波                        没有单独的 notch 步骤
重采样                200 Hz，1000 Hz 输入时 resample_poly(1,5)  250 Hz，resample_poly(1,4)
切段                  视频/休息内部 3 秒窗；不足 3 秒尾段舍弃     从刺激前 -0.2 秒到 +1.2 秒抽 epoch
基线                  无刺激前基线校正                           减去 -0.2～0 秒均值，再保留 0～1.2 秒
形状                  59×600，固定 3 秒，无时间 padding           59×300，固定 1.2 秒，无时间 padding
幅度处理              μV 除以 200 后裁剪至 [-1,1]                full_preprocessed 保留原幅度数值，不做该归一化
质控                  pre-clip 占比、每道平坦度、有限值           max-channel PTP >500 μV 或全局 std ≤1e-4 等规则拒绝
质控失败              通过 qc_pass/reasons 显式标记并保留         在 shared 波形输出中剔除并记录 rejected
MVNN                  初步阶段不拟合、不应用                     derived 分支支持 full59 和 visual17 的 MVNN
存储                  S3 上按采集目录命名的 Lance 及 TOML        按 subject/session 保存 NPZ 与 JSON，多输出分支
数据划分              split=unspecified                          MVNN 只使用当前 image protocol 的 train image 行拟合
====================  ========================================  ==============================================

特别注意，同样的数字事件码可能在不同采集范式中有不同含义。EMOVID 的 101 是
session_start，不能套用图片流程中“触发道 101 是待抽取图片事件”的规则。
长情绪视频也没有本脚本能够证实的图片式 -0.2～0 秒静息基线，因此没有照搬 ERP 基线。

official 脚本的 ``full_preprocessed`` 与 MVNN 派生分支需要区分。
前者经过 CAR/滤波/重采样/基线处理，但不做 MVNN；后者从 train image 行逐 trial
计算 shrinkage covariance，先按 image 平均，再对 image 等权平均，求 59×59 whitener。
它不是 z-score，也不是每个通道独立除标准差。

当前 ``processed/mvnn.py`` 的正式 API 还依赖相对导入的 ``.contracts``，并检查
59 道顺序、250 Hz 与预处理 contract hash；其图片预处理输入由上述流程生成 59×300。
当前文件夹没有 contracts.py，
不能把这个单文件直接 import 后应用到本次 200 Hz、3 秒、幅度归一化后的数据。
若之后增加 EMOVID 的 MVNN，需要单独定义相容的契约和不泄漏测试数据的拟合集合。

本次实际结果
------------

当前输入为 23 份 BDF 中通过映射检查的 456 段情绪视频和 10 段长休息。
按 3 秒、不补零的规则生成 3026 个窗口：2843 个情绪窗口、183 个休息窗口。
不足 3 秒的尾段累计约 699.435 秒已舍弃。补零窗口数为 0。
其中 2758 个窗口通过初步质控、268 个窗口标记为不通过；通过质控的
positive/neutral/negative 分别为 863/883/846 个，另有 166 个休息窗口通过。
逐录制和分类质控统计保存在 ``summary.json``，训练时筛选 ``qc_pass=true``。

共 38 项测试通过，覆盖搜索分段、预处理、命名冲突、存储路径与断点续跑。对每份 Lance 的
全量结构、数值范围、标签、采样点、固定 600 点长度、不重叠切窗和无补零校验，结果写入
``/home/zhangyn/rustfs/<采集目录名>.validation.json``，顶层 validation.json 汇总状态。
迁移时对 S3 读回信号的 SHA256 与原预处理数组逐份比较，确保数值没有改变。

复现
----

::

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate EEGPT
    # 全量：默认直接写 S3，自动挂载本地读取视图
    python emovid_pipeline.py --resume

    # 单个 BDF
    python emovid_pipeline.py \
      --raw 's3://oi-data@us-east-1/EMOVID_RAW/20261022161317_20261022_lwh/20261022161317_20261022_lwh/lwh/lwh.bdf' \
      --resume

    # 仅恢复挂载（如服务器重启后）
    python emovid_pipeline.py --mount-only

    # 无 FUSE 环境下仍可以直接写 S3
    python emovid_pipeline.py --resume --no-mount

常用参数为 ``--raw-root``、``--label-root``、``--output-uri``、``--output-root``、
``--profile`` 和 ``--resume``。默认单位为本批数据已确认的 μV。
``--from-manifest <manifest.json>`` 可复用映射；``--import-existing <旧输出目录>``
配合该参数可导入尚未迁移的检查点，并在 S3 读回验证数值一致。

    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
      -s tests -p '*emovid*.py' -v

Python 接口：

::

    from emovid_pipeline import PreprocessingOptions, preprocess_dataset

    result = preprocess_dataset(PreprocessingOptions(
        raw=None,                         # None 表示全量，也可以填写单个 BDF 的 S3 URI
        output_root='/home/zhangyn/rustfs',
        output_uri='s3://personalized-fm/processed',
        resume=True,
    ))

``--resume`` 校验源对象、映射、数值处理参数及已完成表的版本和行数后复用结果。
失败或尚未写完的录制可以重试，已有表不得被另一份源数据覆盖。
Lance 直接通过 S3 SDK 写入，不依赖 FUSE 模拟随机写入；rclone 的磁盘波形缓存关闭。
凭据复用 ``rustfs`` profile，不写入代码、TOML 或结果 JSON。

实际结果以输出目录中的 summary.json 和 validation.json 为准。
