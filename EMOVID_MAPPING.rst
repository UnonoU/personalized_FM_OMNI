EMOVID 原始 EEG、事件日志与情绪标签映射
====================================

完整预处理的统一入口为 ``python emovid_pipeline.py --resume``，结果直接保存在
``s3://personalized-fm/processed/``，并挂载至 ``/home/zhangyn/rustfs/``。
Lance 及配套报告以最外层采集目录名命名，例如 ``20261022161317_20261022_lwh.lance``。
下面的 ``batch_emovid_segments.py`` 用于单独运行映射阶段；其本地 JSON 仅是小型分析报告。

全量批处理
----------

``batch_emovid_segments.py`` 扫描整个 ``EMOVID_RAW/``，按连续设备录制文件中的
session_start/session_end 划分多次实验启动，分别匹配相应的事件日志。
这解决了“一份 BDF 对应多个 events.json”的情况，标签不会只取最近的一份日志。
对没有正常 session_end 的中断录制，保留中断之前边界完整的片段并报告异常。

::

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate EEGPT
    python batch_emovid_segments.py \
      --raw-root 's3://oi-data@us-east-1/EMOVID_RAW/' \
      --label-root 's3://oi-data/EMOVID-261029/' \
      --min-segment-seconds 3 \
      --output-dir analysis/emovid_batch \
      --resume

默认忽略不足 3 秒的片段。该过滤对 emotion 和 rest 都生效，因此大部分 1–2 秒的
试次间短休息只保留在诊断结果中，不进入可用片段表。可调整
``--min-segment-seconds``；这是时长筛选，并不代表已经完成 EEG 信号质量评估。

批处理会列举全部对象一次，优先选择合并后的命名 BDF/EDF，不重复处理其 data.bdf
信号组件；依据原始导出文件和标记文件的 ETag、大小识别重复导出，并保存别名。
默认使用 4 个工作线程，只读取原始文件头、标记文件及日志，不下载完整 EEG 波形。
没有 EDF/BDF 导出的 NDF 目录写入 skipped_collections；单条错误不会停止其他记录。

日志目录时间需要落在设备录制开始前 600 秒至录制结束的窗口内。随后还需通过
公共标记匹配、时钟残差和实际墙钟时间验证。默认将 EDF/BDF 文件头按 Asia/Shanghai
解释，日志采用 payload.timestamp_unix_sec，缺少时回退到目录时间。
墙钟差异不得超过 30 秒，避免相同事件码模式误配到另一场录制。
``--search-padding-seconds``、``--wall-tolerance-seconds``、``--timezone`` 可调整这些参数。

批量输出目录包含：

* ``usable_segments.csv``：边界与标签检查通过且时长达标的片段，含原始 URI、
  events URI、起止秒数、原始采样点和 200 Hz 采样点，可用于后续 EEG 裁剪。
* ``all_segments.csv``：所有已识别片段，含短片段和排除原因。
* ``manifest.json``：全量扫描与处理统计、每条录制状态、匹配证据、异常与未匹配目录。
* ``recordings.csv``：每条录制的状态摘要。
* ``recordings/*.json``：每条原始录制的完整分段结果和各事件日志的对齐证据。
* ``inventory.json``：扫描到的源对象、日志清单和未重复处理的导出组件/副本。

``success`` 表示录制中的各次实验启动均完成映射且未发现配对或采集中断问题；
``partial`` 表示可用片段已保留，同时存在采集中断、未闭合事件或未匹配的实验启动；
``unmatched`` 表示未找到通过验证的日志。可用表仍包含 partial 录制中检查通过的完整片段。
``usable`` 仅表示本脚本的映射、标签及最小时长条件通过。

``--resume`` 只复用源对象清单、ETag、代码和参数均未变化的成功/部分成功结果。
失败条目会重试；汇总表会根据本轮结果重新生成。可在同一输出目录继续运行。
程序不会写入 S3，输出为分段映射和索引，不包含 EEG 滤波、重采样后的波形文件。

本次实际全量运行识别出 32 个采集目录、24 份去重后的 BDF、65 份标签事件日志。
23 份 BDF 找到了通过验证的标签映射，其中 8 份记录了中断或边界异常；
1 份未找到时间窗内的日志，8 个目录尚无 EDF/BDF 导出。
按当前 3 秒阈值过滤了 1608 个短片段，输出可用 emotion 456 段（每类 152 段）和
long_rest 10 段。另有一段情绪视频因未完整播放而排除。重复导出的 3 份副本没有重复计数。

批处理和单文件流程的测试：

::

    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p '*emovid_segments.py' -v

单文件处理
----------

入口为 ``emovid_segments.py``。依赖现有 EEGPT 环境中的 boto3 和 MNE；S3
使用 ``read_rustfs.client()`` 的本地凭据和 RustFS endpoint，不需要把密钥写进代码。
核心搜索、对齐和分段函数使用 Python 标准库，可以独立导入。

运行用户提供的示例
------------------

::

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate EEGPT
    python emovid_segments.py \
      --raw 's3://oi-data@us-east-1/EMOVID_RAW/20261023102529_20261023_lwh/20261023102529_20261023_lwh/lwh/lwh.edf' \
      --output-sfreq 200 \
      --output analysis/emovid_lwh_20261023/segments.json

输出文件必须是新文件。示例结果已保存在上述路径，重复运行时请换一个输出文件名。
实际存储对象为同路径的 ``lwh.bdf``，不存在 ``lwh.edf``；程序发现指定文件不存在时，
只尝试同目录同名的另一种 EDF/BDF 扩展名，并在结果中同时保留请求路径与实际路径。

搜索与对齐规则
--------------

1. 从原始文件目录解析 ``20261023102529``，从标签目录解析
   ``20261023_102630``，在 ``EMOVID-261029`` 下列举 events.json 和
   events_part_NNN.json。支持 S3 分页和本地目录。
2. 仅保留同一天、默认相差不超过 600 秒的候选，按绝对时间差排序。
   示例选中 ``13467638/session_01/20261023_102630/events.json``，相差 +61 秒。
   lwh 与数字 subject_id 不做未经证实的姓名映射；可用 ``--subject-id`` 过滤。
3. 两个候选的时间距离之差不超过默认 30 秒时，拒绝自动选择。
   多个 EEG part 处于同一时间目录也会触发此规则，需要用 ``--events`` 指明 part。
   ``--max-delta-seconds``、``--ambiguity-seconds`` 可调整搜索参数。
4. 读取设备实际标记序列与 events.json 的公共事件码，通过顺序匹配建立对应关系。
   至少需要三个公共标记、80% 的日志标记覆盖率，以及合理的时钟斜率和拟合残差。
   时间匹配只用于定位日志；它不能作为 EEG 裁剪的时间偏移。
5. 片段的硬件标记边界使用 EDF/BDF 中的真实 onset。日志本身的 sample_index
   属于 PsychoPy TCP 录制，不能直接当作设备 BDF 的采样点。
   长休息没有硬件标记，使用两侧公共标记之间的日志时钟插值，输出明确记录其估计依据。
   优先采用完整精度的 flip callback 单调时钟；缺少该字段时采用 relative_time_sec。
   不对公共标记范围之外的事件进行外推。

默认从原始文件同目录的 ``evt.bdf`` 获取硬件标记，只下载小型标记文件和原始文件头。
两者的文件头开始时间必须一致。若没有 evt.bdf，可以通过 ``--annotations`` 提供
EDF/BDF 标记文件，或提供包含 ``onset_seconds,code`` 列的 CSV。
CSV 的 onset 必须相对于当前原始录制起点。完整 EDF/BDF 也可以作为本地输入。
断续 EDF+D/BDF+D 和混合采样率信号需要单独处理，目前拒绝自动映射。

仅搜索对应文件时添加 ``--search-only``；已知日志时使用：

::

    python emovid_segments.py \
      --raw 's3://oi-data/EMOVID_RAW/20261023102529_20261023_lwh/20261023102529_20261023_lwh/lwh/lwh.bdf' \
      --events 's3://oi-data/EMOVID-261029/13467638/session_01/20261023_102630/events.json' \
      --output analysis/emovid_lwh_20261023/explicit_segments.json

分段定义
--------

* ``150/151``：emotion，标签取事件 payload.three_class_label，保留
  original_label、video_id、trial_id、attempt_id。仅情绪视频带三分类标签，
  不把普通视频或休息推断为 neutral。
* ``156/157`` 或旧的 ``136/137``：rest / short_rest，试次间短休息。
* ``rest_start/rest_end``：rest / long_rest，只存在于日志，不能靠 BDF 数字码恢复。
  这类事件可能继承上一条情绪视频的 payload；程序不会给长休息赋该视频的情绪标签。
* ``110/111``：rest / baseline；``112/113``：eyes_open_baseline；
  ``114/115``：eyes_closed_baseline。保留它们与试次间休息的区别。

边界按 subject/session/eeg_part 以及适用的 trial/attempt/video/rest_index 配对，
不跨会话或播放尝试连接。缺失边界写入 pairing_issues，不自动用下一个视频补全。
跳过/中断的视频、标签缺失或冲突、缺少真实硬件边界的片段保留诊断信息，
但 ``usable=false``。没有事件标记的空隙不自动算作 rest。

三分类 ID 是本项目的显式约定：negative=0、neutral=1、positive=2，
并非 EEG 事件码。采集仓库没有提供这个整数顺序。

输出及使用
----------

JSON 顶层包含候选映射、协议来源 commit、原始文件头、时钟对齐质量、segments 和
pairing_issues。每段包含秒数、原始 start_sample/stop_sample、标签和可用性。
全部区间采用半开区间 ``[start, stop)``，采样点采用最近整数舍入。
``--output-sfreq 200`` 会额外提供 output_start_sample/output_stop_sample，
其起点仍然是未裁剪的完整原始记录，并不会自动应用 LaBraM 预处理的 crop。

::

    import json
    import mne

    result = json.load(open('analysis/emovid_lwh_20261023/segments.json'))
    raw = mne.io.read_raw_bdf('/path/to/lwh.bdf', preload=False)
    for segment in result['segments']:
        if segment['usable'] and segment['segment_type'] == 'emotion':
            eeg = raw.get_data(start=segment['start_sample'], stop=segment['stop_sample'])
            label = segment['emotion_label_id']

若输入为 EDF，使用 ``mne.io.read_raw_edf``。对已裁剪/重采样的数据使用这些边界时，
须先扣除实际 crop_start_seconds，再按目标采样率换算。

验证
----

::

    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_emovid_segments.py' -v

实际示例的识别结果为 29 段 emotion、103 段 short_rest、1 段 long_rest，
未发现 baseline 标记。短休息与长休息不能据此等同于睁眼/闭眼静息基线。

协议来源固定为采集仓库 commit 637058b3b2a431132a4643e56444d03a3bb5567d：

* https://github.com/18yiba/visual-video-task/blob/637058b3b2a431132a4643e56444d03a3bb5567d/src/video_eeg/utils/markers.py
* https://github.com/18yiba/visual-video-task/blob/637058b3b2a431132a4643e56444d03a3bb5567d/src/video_eeg/experiment/emotion_runner.py
* https://github.com/18yiba/visual-video-task/blob/637058b3b2a431132a4643e56444d03a3bb5567d/src/video_eeg/storage/session_recorder.py
