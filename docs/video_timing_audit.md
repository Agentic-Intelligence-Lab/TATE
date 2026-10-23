# Bowl / hanging / drawer 视频速度检查（2026-10-13）

**结论：进入 TATE 的源 MP4 已经是现在的播放速度，校正前后数据集打包和 WiLoR 可视化没有进一步加速。现有数据不足以确定采集现场的真实动作速度，也不足以排除更早的采集/首次编码阶段存在帧率问题。**

## 检查范围

| 任务 | 源数据 | TATE 实验 | 源 ego / 真机条数 | 校正前 / 后条数 |
|---|---|---|---:|---:|
| bowl | `/mnt/data/xule/Data_TATE_0930/bowl_{ego,arx}` | `bowl_1` | 100 / 100 | 100 / 74 |
| hanging | `/mnt/data/xule/Data_TATE_0930/hanging_{ego,arx}` | `hanging_1` | 100 / 100 | 100 / 91 |
| drawer | `/mnt/data/xule/Data_TATE_0930/drawer_{ego,arx}` | `drawer_1012` | 100 / 100 | 100 / 85 |

检查了 300 个原始 ego 头部视频、300 个原始真机头部视频、300 个 `wilor_eef_vis.mp4` 的流信息；另外逐一核对两个导出版本所有三路相机视频与源视频的文件身份。未指定具体播放器或仿真回放文件，因此结论针对这些磁盘 MP4，不包含未观察到的播放器倍速设置或实时仿真运行速度。

## 排除 TATE 后处理加速的证据

| 任务 | 导出视频检查数（三路相机、两个版本） | 与源文件同设备号 / inode | WiLoR 可视化帧数、FPS、时长完全匹配 |
|---|---:|---:|---:|
| bowl | 522 | 522 / 522 | 100 / 100 |
| hanging | 573 | 573 / 573 | 100 / 100 |
| drawer | 555 | 555 / 555 | 100 / 100 |
| 合计 | **1650** | **1650 / 1650** | **300 / 300** |

这里的“同 inode”表示导出视频与源视频是同一份底层文件的硬链接，内容和编码时间轴完全相同。不是只比较了文件名或 nominal FPS。

所有 550 条导出 episodes 的源帧索引均连续，保留帧数与对应源 episode 一致；视频片段起止时间也一致。校正失败导致的是整条 episode 未进入校正后集合，没有发现删除无效帧后仍以 30 FPS 播放的情况。

注意：校正后会重新编号。例如 bowl 校正后 `file-000.mp4` 对应源 episode 1，而校正前 `file-000.mp4` 对应源 episode 0。直接拿同名文件比较可能比较到不同演示。本次全部按 `source_episode_index` 对齐。

源 ego、源真机和可视化流信息均为 **30 FPS**。额外固定抽查每任务源 episode 0、50、99 的源视频及 WiLoR 可视化，共 18 个视频，实际解码其帧显示时间戳（PTS）：时间轴符合 `frame_index / 30`，最大偏差仅约 `3.33e-7 s`（输出小数精度范围）。

源码与实测一致：

- [preprocess/batch/lerobot.py](../preprocess/batch/lerobot.py) 的 `_materialize_file` 使用硬链接或原样复制；本次所有导出实测均为硬链接。
- 该文件 `build_episode_table` 保留 `tate.source_frame_index`、`tate.source_timestamp`，输出 timestamp 只减去片段起始时间。
- [preprocess/visualize_wilor_cache.py](../preprocess/visualize_wilor_cache.py) 使用缓存/源视频 FPS，每个缓存帧写一帧；缺失某只手只跳过该手的绘制，不跳过整帧。
- [preprocess/batch/dataset.py](../preprocess/batch/dataset.py) 的片段裁剪使用 `setpts=PTS-STARTPTS`，只重置起点；本批没有时间裁剪，导出链路没有执行该重编码分支。

## 源视频本身的长度

以下是各 100 条**完整视频**的统计，包含动作前后停顿。Ego 与真机不是逐条配对，不能把两者时长比当作真实动作加速倍数。

| 任务 | 原始 ego 平均时长 | 原始 ego 中位数 | 原始 ego 范围 | 原始真机平均时长 |
|---|---:|---:|---:|---:|
| bowl | 9.19 s | 8.97 s | 6.97–14.40 s | 14.48 s |
| hanging | 10.65 s | 9.97 s | 6.97–20.03 s | 11.71 s |
| drawer | 12.92 s | 12.67 s | 11.07–17.97 s | 20.14 s |

例如源 episode 0：bowl 为 376 帧 / 12.533333 秒，hanging 为 601 帧 / 20.033333 秒，drawer 为 428 帧 / 14.266667 秒。它们的 WiLoR 可视化均保留完全相同的帧数和时长。

## 为什么不能据此断言“采集现场动作就是快”

目前所谓“原始数据”已经是首次导出的 LeRobot 数据集，不是带相机硬件时间戳的原始采集流：

- 源 parquet 只有 `observation.state`、`action`、`timestamp`、`frame_index`、`episode_index`、`index`、`task_index`；没有采集 wall-clock、相机 device timestamp 或实际采集 FPS 字段。
- 全部源 ego 时间戳与 `frame_index / 30` 一致，最大差异小于 `9e-7 s`，符合 float32 数值精度。这个规则时间轴并不是一份可独立证明实际采样间隔的记录。
- 源目录没有保留原始图像帧文件。已检查目录及项目引用中未找到与这批演示对应的逐帧真实时钟或采集性能日志。服务器上其他采集器代码不能证明这批数据具体由哪个版本、什么参数录制。

因此现有证据支持：**TATE 处理后的速度与输入源文件一致。** 剩余两种情况尚无法区分：

1. 演示者本来做得快，且实际采集/播放均为 30 FPS。
2. 实际采样低于 30 FPS，但首次导出仍按 30 FPS 编码，导致源 MP4 就已快放。

第二种情况只是需要排查的可能性，当前没有证据确认其发生。举例：如果实际每秒只采到 15 帧，却将所有帧按 30 FPS 编码，就会播放成 2 倍速；仅查看编码 FPS 和 `frame_index/30` 时间戳无法发现这一点。

要继续定因，需要同一批采集的相机硬件/系统逐帧时间戳，或记录 episode 真实起止时刻的日志，连同当时采集程序及 FPS 配置。用真实采集时长与 MP4 时长对比，才能判断是否在采集/首次编码阶段压缩了时间。

## 可复查文件

- [汇总及 PTS 明细 JSON](../outputs/video_timing_audit_20261013/audit.json)
- [900 个视频流信息 CSV](../outputs/video_timing_audit_20261013/video_streams.csv)
- [1650 个导出—源视频对应及文件身份 CSV](../outputs/video_timing_audit_20261013/export_source_identity.csv)
- [检查脚本](../scripts/audit_video_timing.py)

服务器输出：`/mnt/data/xule/TATE/outputs/video_timing_audit_20261013`。所有操作只读源数据，没有修改视频、FPS、时间戳或校正结果。

```bash
cd /mnt/data/xule/TATE
/mnt/data/envs/lifego/bin/python \
  outputs/video_timing_audit_20261013/audit_video_timing.py \
  --output outputs/video_timing_audit_20261013
```
