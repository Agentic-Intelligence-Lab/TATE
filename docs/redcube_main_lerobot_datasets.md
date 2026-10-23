# 随机红方块主实验 LeRobot 数据集

服务器：`root@8.160.160.64`，SSH 端口 `1018`。

目标根目录：`/mnt/data/xule/TATE/output/lerobot/local`。注意这里是 **output**，不是已有实验的 **outputs**。

## 十组配置

以下目录均位于目标根目录；训练 repo ID 为 `local/` 加目录名。

| 配置 | 目录名 | 有效训练帧 | 输出 episodes |
|---|---|---:|---:|
| R20 | `arx_eef_redcube_random_main_r20_nodropout_train` | 7,324 | 20 |
| R40 | `arx_eef_redcube_random_main_r40_nodropout_train` | 14,709 | 40 |
| R70 | `arx_eef_redcube_random_main_r70_nodropout_train` | 26,106 | 70 |
| R100 | `arx_eef_redcube_random_main_r100_nodropout_train` | 37,085 | 100 |
| Raw-E20R20 | `arx_eef_redcube_random_main_e20r20_raw_nodropout_train` | 13,447 | 40 |
| TATE-E20R20 | `arx_eef_redcube_random_main_e20r20_tate_nodropout_train` | 13,447 | 40 |
| Raw-E50R20 | `arx_eef_redcube_random_main_e50r20_raw_nodropout_train` | 22,190 | 70 |
| TATE-E50R20 | `arx_eef_redcube_random_main_e50r20_tate_nodropout_train` | 22,190 | 70 |
| Raw-E100R20 | `arx_eef_redcube_random_main_e100r20_raw_nodropout_train` | 36,970 | 121 |
| TATE-E100R20 | `arx_eef_redcube_random_main_e100r20_tate_nodropout_train` | 36,970 | 121 |

E/R 表示独立源演示数，不是输出 episode 数。E100 中源 ego episode 7 有 EEF 缺测，拆成两个连续片段，故 100 条源 ego 输出 101 个 episodes；动作不跨越缺测区间。E20/E50 没有选中这条源演示。

## 来源和数据预算

- 原始视频及真机实测关节：`/mnt/data/xule/Data_TATE_0930/stack_redcube_arx_ego_1`。这个数据集同时包含 100 条 ego 和 100 条真机，按已审计身份清单区分，不按奇偶直接推断。
- ego-raw EEF：`outputs/paired_redcube_20261010/preprocess/paired_redcube_w*/artifacts/eef/finger_center_hys085/episode_*/eef_raw.json`；逐条精确路径及 SHA-256 记录在审计文件中。目录名包含 paired 不代表 Raw 标签经过配对校正。
- ego-tate：`outputs/unpaired_redcube_20261011/corrected/unpaired_bridge/episode_*/eef_corrected.json`。使用无配对 SceneBridge 的固定主实验，不使用旧的配对回归输出。
- 校正模型：`outputs/unpaired_redcube_20261011/20261011/model.json`。

R20 就是校正模型使用的 20 条真机：

`2, 22, 52, 54, 58, 66, 68, 78, 80, 86, 102, 108, 112, 116, 126, 128, 132, 156, 176, 178`

R20 ⊂ R40 ⊂ R70 ⊂ R100。额外真机按固定种子 `20261016` 随机打乱并依次加入。

E20 ⊂ E50 ⊂ E100。按 ego 自己的初始场景二维坐标建立 3×3 分位数网格，按格子样本数比例逐步抽取，格内使用固定种子 `20261016` 打乱。选择不使用对应真机轨迹或校正误差。Raw/TATE 的源 IDs 和有效帧严格相同。

**现有校正器使用全部 100 条 ego 和 R20 拟合。** E20/E50 只代表策略训练数据量，不是整个流程仅使用 20/50 条 ego；没有为小子集重新拟合校正器。每个 TATE 数据集的 provenance 均记录完整校正预算。

## 训练语义

- LeRobot v2.1，30 FPS，单一 train split，直接合并两域，不加入额外采样权重。
- `observation.state` 和 `action` 都是 16D EEF：每臂 xyz（米）、xyzw 四元数、夹爪；夹爪 0=open、1=closed。
- state 来自源帧 t，action 来自源帧 t+1；每个视频保留末尾动作对应的额外一帧。
- ego 直接读取 EEF JSON，不经过 IK→FK；真机从实测关节做 FK。
- 仅右臂参与 state/action loss，左臂为 canonical identity，前 8 维 mask=false。
- ego 的 image mask 为 head-only；真机三路相机全开；不增加 camera dropout。
- 全长视频沿用源文件；缺测分段按源帧索引裁剪并设置 30Hz 时间轴，避免 ffmpeg 自动同步丢帧。
- 均匀抽帧时，E20R20 / E50R20 / E100R20 的真机占比分别约 54.47% / 33.01% / 19.81%，不是按演示条数计算的比例。
- 每组统计量仅由该组导出的样本计算；OpenPI 所需的 normalization assets 仍应在训练准备阶段按各组计算。

## 构建与验证

```bash
cd /mnt/data/xule/TATE
PY=thirdparty/openpi/.venv/bin/python
$PY training/build_redcube_main_datasets.py
$PY training/validate_redcube_main_datasets.py
HF_HUB_OFFLINE=1 $PY training/validate_redcube_main_loader.py
```

构建器拒绝覆盖已完成结果；`--resume` 只用于中断恢复，校验计划一致后跳过已完成组，把未完成目录移到审计目录保存。ffmpeg 未出现在 PATH 时使用环境内 imageio_ffmpeg 自带的可执行文件。

审计目录：`/mnt/data/xule/TATE/output/lerobot/main_redcube_audit/`。

- `plan.json`：全部子集 IDs、抽样顺序、场景分层边界、源路径和实验预算。
- `eef_source_hashes.json`：200 个 Raw/TATE EEF 文件的路径与 SHA-256。
- `complete.json`：十组最终计数；生成成功的标志。
- `validation.json`：全量标签与源 EEF/真机 FK 一致、有效帧、掩码、视频长度、Raw/TATE 对照一致性。
- `loader_validation.json`：真实 LeRobot 加载、三路视频解码、50 步动作窗口及 episode 边界填充验证。
- 每个数据集的 `cotrain_provenance.json`：逐片段源 episode、源帧范围和校正器预算。

仅生成和验证数据，不启动策略训练。
