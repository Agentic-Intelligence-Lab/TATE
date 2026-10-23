# 五任务：真机与 ego 校正前后的 TCP 可视化

输出日期：2026-10-12。服务器与本地均保存在项目下的 `outputs/five_task_eef_visualization_20261012/`。

## 两种主图

- [单条轨迹总览 PNG](../outputs/five_task_eef_visualization_20261012/01_single_episodes.png) / [PDF](../outputs/five_task_eef_visualization_20261012/01_single_episodes.pdf)：每个任务每种数据选择一条，展示 3D、XY、XZ、YZ；圆点为起点，方块为终点。
- [全量空间分布 PNG](../outputs/five_task_eef_visualization_20261012/02_full_distributions.png) / [PDF](../outputs/five_task_eef_visualization_20261012/02_full_distributions.pdf)：全部有效帧的 3D 点云，以及 X/Y/Z 各自的概率密度直方图。
- [共同源帧分布 PNG](../outputs/five_task_eef_visualization_20261012/03_matched_distributions.png) / [PDF](../outputs/five_task_eef_visualization_20261012/03_matched_distributions.pdf)：补充对照，ego 两组只保留共同源 episode、共同源帧及两边均有效的位置；真机保持全量。

深灰 = 真机，橙色 = ego 校正前，蓝色 = ego 校正后。校正后曲线用虚线。

| 任务 | 活动臂 | 真机条数 | 校正前条数 | 校正后条数 | 单条真机 / ego 源 ID | 单条图 | 分布图 |
|---|---|---:|---:|---:|---|---|---|
| cube | 右 | 50 | 50 | 50 | 0 / 0 | [PNG](../outputs/five_task_eef_visualization_20261012/cube_single.png) | [PNG](../outputs/five_task_eef_visualization_20261012/cube_distribution.png) |
| hanging | 右 | 100 | 100 | 91 | 0 / 0 | [PNG](../outputs/five_task_eef_visualization_20261012/hanging_single.png) | [PNG](../outputs/five_task_eef_visualization_20261012/hanging_distribution.png) |
| bowl | 右 | 100 | 100 | 74 | 0 / 1 | [PNG](../outputs/five_task_eef_visualization_20261012/bowl_single.png) | [PNG](../outputs/five_task_eef_visualization_20261012/bowl_distribution.png) |
| cola | 左、右 | 20 | 50 | 50 | 0 / 0 | [PNG](../outputs/five_task_eef_visualization_20261012/cola_single.png) | [PNG](../outputs/five_task_eef_visualization_20261012/cola_distribution.png) |
| drawer | 左、右 | 100 | 100 | 85 | 0 / 0 | [PNG](../outputs/five_task_eef_visualization_20261012/drawer_single.png) | [PNG](../outputs/five_task_eef_visualization_20261012/drawer_distribution.png) |

每张分任务图均有同名 PDF。

## 数据口径

- 位置单位从米换算为厘米。X 向前、Y 向左、Z 向上；坐标原点为各臂零位法兰。双臂分行展示，不将两个原点当成同一世界原点。
- 真机读取 `outputs/arx_real_flange/{task}_arx/episode_*.json` 的 `arms.{side}.T_tcp_in_output_frame`，它由测量关节状态计算得到。
- Ego 直接读取用户指定 `datasets/finger_center_hys085__{none,position_rotation}/data/**/*.parquet` 的 `tate.eef.{side}.pose` 前三维，按 `tate.eef.{side}.valid` 过滤。`observation.state` 是关节角，不拿它当 TCP 坐标，也不做 IK→FK 回转。
- 单条 ego 选择校正前后共同存在、每个活动臂至少有两个有效位置的最小源 ID，以 `tate.source_episode_index` 匹配；真机独立选取最小可用 ID。真机与 ego 不宣称成对演示，不做时间对齐。Bowl 的校正后源 ID 0 缺失，因此选择源 ID 1（导出编号 0）。
- 所有原始有效位置保留，不平滑轨迹、不抽样、不裁剪离群点。无效帧和源帧缺口会断开单条轨迹，避免跨缺测区间连线。
- 全量分布按有效帧统计；长 episode 权重较大。每种数据在同一套 60 个区间上独立归一化，密度积分为 1。组间共享坐标范围；不同任务使用各自的范围。单条图只按所选轨迹设定范围。
- Hanging、bowl、drawer 校正前后集合大小不同，主图反映目录中的实际全量分布，不能把所有分布差异都归因于校正。补图控制 ego 样本集合和逐臂有效性，适合辅助判断校正带来的位置变化。本次未调查 episode 缺失的生成原因。
- 本次图展示 TCP **位置**；“position_rotation”是数据版本名，图中不展示姿态旋转和夹爪状态，也不据此量化校正精度。

完整目录、源 ID、缺失 ID、有效/无效帧数、位置范围及单条选择记录在 [audit.json](../outputs/five_task_eef_visualization_20261012/audit.json)。读取时已核对 parquet 总帧数、导出 episode 数、源帧唯一性、TCP 坐标框架，以及补图两组有效帧数一致性。

## 重现

脚本：[scripts/visualize_five_task_eef.py](../scripts/visualize_five_task_eef.py)。服务器输出目录中另有运行时脚本副本。

```bash
cd /mnt/data/xule/TATE
/mnt/data/envs/lifego/bin/python \
  outputs/five_task_eef_visualization_20261012/visualize_five_task_eef.py \
  --root /mnt/data/xule/TATE \
  --output outputs/five_task_eef_visualization_20261012
```

实验目录：cube=`stack_redcube_v1_1`，hanging=`hanging_1`，bowl=`bowl_1`，cola=`cola_1012`，drawer=`drawer_1012`。Cube 真机目录名为 `stack_redcube_v1_arx`，其余为 `{task}_arx`。
