# 随机 bowl 主实验 LeRobot 数据集

服务器：`root@8.160.160.64`，SSH 端口 `1020`。

输出根目录：`/mnt/data/xule/TATE/output/lerobot/local`（单数 output）。任务是将黄色碗叠到粉色碗上，使用右臂。

## 配置

沿用随机红方块主实验：R20、R40、R70、R100，以及 E20/E50/E100 + R20 的 Raw/TATE 两个版本，共十组。直接合并两域，不额外设置采样权重；单一 train split，不增加 camera dropout。

| 配置 | 目录名 | 有效训练帧 | 输出 episodes |
|---|---|---:|---:|
| R20 | `arx_eef_bowl_random_main_r20_nodropout_train` | 5,835 | 20 |
| R40 | `arx_eef_bowl_random_main_r40_nodropout_train` | 11,673 | 40 |
| R70 | `arx_eef_bowl_random_main_r70_nodropout_train` | 20,432 | 70 |
| R100 | `arx_eef_bowl_random_main_r100_nodropout_train` | 29,191 | 100 |
| Raw-E20R20 | `arx_eef_bowl_random_main_e20r20_raw_nodropout_train` | 12,182 | 40 |
| TATE-E20R20 | `arx_eef_bowl_random_main_e20r20_tate_nodropout_train` | 12,182 | 40 |
| Raw-E50R20 | `arx_eef_bowl_random_main_e50r20_raw_nodropout_train` | 21,500 | 70 |
| TATE-E50R20 | `arx_eef_bowl_random_main_e50r20_tate_nodropout_train` | 21,500 | 70 |
| Raw-E100R20 | `arx_eef_bowl_random_main_e100r20_raw_nodropout_train` | 36,772 | 120 |
| TATE-E100R20 | `arx_eef_bowl_random_main_e100r20_tate_nodropout_train` | 36,772 | 120 |

100 条 ego 的 31,037 个源帧生成 30,937 个 t→t+1 样本；每条去掉最后一个无下一帧的 state，视频仍保留该尾帧。没有缺测分段，源演示数与输出 episode 数一致。

训练 repo ID 为 `local/` 加目录名。训练数据根目录是 `/mnt/data/xule/TATE/output/lerobot`。

## 来源与预算

- 原始视频和真机实测关节：`/mnt/data/xule/Data_TATE_1016/bowl_random`。
- 校正前：`/mnt/data/xule/TATE/outputs/unpaired_bowl_20261016/ego_raw/episode_XXXXXX/eef_raw.json`。
- 校正后：`/mnt/data/xule/TATE/outputs/unpaired_bowl_20261016/ego_corrected/unpaired_bridge/episode_XXXXXX/eef_corrected.json`。
- 校正清单：`outputs/unpaired_bowl_20261016/ego_corrected/manifest.json`。
- 校正模型：`outputs/unpaired_bowl_20261016/20261016/model.json`。
- 已审计身份清单：`outputs/paired_bowl_20261016/pairing_verified.json`；这里只取两域身份，不使用逐条跨域匹配进行标签生成或子集选择。

100 条真机和 100 条 ego 来自同一个原始采集数据集。经既有运动审计，真机为偶数 episode，ego 为奇数 episode。使用固定主模型的无配对校正结果，没有混入其他参考抽样轮次。

R20 使用模型实际使用的 20 条真机（以下按 ID 排序）：

`14, 18, 30, 36, 46, 56, 64, 68, 86, 92, 104, 106, 116, 130, 132, 134, 158, 164, 172, 190`

R20 ⊂ R40 ⊂ R70 ⊂ R100；其余真机按固定种子 `20261019` 打乱后依次加入。E20 ⊂ E50 ⊂ E100；按 ego 自身初始场景坐标（黄碗中心减粉碗中心）构造 3×3 分位数网格，按各格比例嵌套抽样，格内使用相同固定种子打乱。Raw/TATE 的源演示和有效帧严格一致。

**校正器使用全部 100 条 ego 和 R20 拟合。** E20/E50 指策略训练使用的 ego 数量，不代表校正阶段只使用了 20/50 条 ego。每个 TATE 数据集都记录完整校正器预算。

## 标签和视频语义

- LeRobot v2.1，30 FPS；16D state/action：左右臂各 xyz（米）、xyzw 四元数、二值夹爪。
- ego 直接读取 EEF JSON，保留已校正标签，不经过 IK→FK。
- 真机从采集的 14D 关节状态做 FK；只右臂参与 loss，左臂使用 canonical identity 且前八维 mask=false。
- state 来自源帧 t，action 来自源帧 t+1；每个视频保留最后 action 对应的额外一帧。
- ego image mask 为 `[True, False, False]`，真机为 `[True, True, True]`。
- **bowl 真机夹爪采用 `raw > -1.65` 为 closed=1，其余为 open=0。** 与本次 bowl 校正的单域事件提取规则一致；通用构建器的 -2.6 不适用于这批部分张开的动作。阈值和严格大于比较均写入 provenance，metadata 中的夹爪说明也同步更新。ego 沿用原 EEF 的 grasp_state。
- 元数据统计量仅由各组自身输出计算。OpenPI 的 normalization assets 可在后续训练准备阶段按组计算。

## 复现与验证

```bash
cd /mnt/data/xule/TATE
PY=thirdparty/openpi/.venv/bin/python
$PY training/build_bowl_main_datasets.py
$PY training/validate_bowl_main_datasets.py
HF_HUB_OFFLINE=1 $PY training/validate_bowl_main_loader.py
```

已完成目录受覆盖保护；`--resume` 仅用于中断恢复，要求计划一致。

服务器审计目录：`/mnt/data/xule/TATE/output/lerobot/main_bowl_audit/`。

- `plan.json`：源路径、全部子集 IDs、抽样顺序、场景分层边界及预算。
- `eef_source_hashes.json`：200 个原始/校正 EEF 文件的 SHA-256；校正清单和原文件哈希逐条核验。
- `complete.json`：十组最终帧数、episode 数及域计数。
- `validation.json`：全量原始/校正 EEF 与导出标签比对、真机 FK 与夹爪标签、掩码、视频长度和成对数据一致性。
- `loader_validation.json`：实际 LeRobot 加载、三路视频解码、50 步 action 窗口及 episode 边界填充抽查。
- 每个数据集的 `cotrain_provenance.json`：逐条源演示、源帧范围、夹爪规则及校正模型预算。

仅构建、验证数据，不启动策略训练。
