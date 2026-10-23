# 五任务校正效果评估（2026-10-12）

协议来源：[evaluation/README.md](../evaluation/README.md)，指标实现：[evaluation/metrics.py](../evaluation/metrics.py) 和 [evaluation/aggregate.py](../evaluation/aggregate.py)。用户所写 `rho_re3` 对应仓库字段 `rho_se3`。

## 结果

全部指标越低越好。前三项无量纲，位置单位为毫米，旋转单位为度。Cola、drawer 为两活动臂的等权均值。

| 任务 | 数据 | rho_pos | rho_rot | rho_se3 | D_pos_mm | D_rot_deg |
|---|---|---:|---:|---:|---:|---:|
| cube | 校正前 | 10.145 | 3.756 | 6.148 | 127.50 | 24.70 |
| cube | 校正后 | 1.525 | 1.384 | 1.445 | 19.17 | 9.10 |
| hanging | 校正前 | 11.331 | 6.173 | 8.336 | 182.12 | 36.45 |
| hanging | 校正后 | 5.233 | 4.024 | 4.451 | 84.10 | 23.76 |
| bowl | 校正前 | 8.237 | 2.613 | 4.626 | 170.68 | 25.89 |
| bowl | 校正后 | 2.901 | 2.154 | 2.455 | 60.12 | 21.34 |
| cola | 校正前 | 6.287 | 5.667 | 5.944 | 152.54 | 41.08 |
| cola | 校正后 | 1.654 | 2.018 | 1.798 | 41.39 | 14.54 |
| drawer | 校正前 | 6.612 | 7.167 | 6.880 | 161.84 | 47.50 |
| drawer | 校正后 | 2.125 | 2.461 | 2.277 | 52.02 | 16.18 |

五个任务的五项指标均下降。位置距离降幅分别为：

- cube：85.0%；旋转距离下降 63.2%。
- hanging：53.8%；旋转距离下降 34.8%。
- bowl：64.8%；旋转距离下降 17.6%。
- cola：72.9%；旋转距离下降 64.6%。
- drawer：67.9%；旋转距离下降 65.9%。

### 实际评估样本数

| 任务 | 左臂 ego（前后相同） | 右臂 ego（前后相同） | 留出真机 |
|---|---:|---:|---:|
| cube | — | 50 | 10 |
| hanging | — | 88 | 10 |
| bowl | — | 73 | 9 |
| cola | 49 | 50 | 10 |
| drawer | 74 | 85 | 10 |

五个任务的 `comparison.json` 均为 `comparable: true`：按活动臂核对，校正前后实际参与聚合的 ego–real 配对完全相同。Hanging 和 bowl 的五项指标与服务器既有同数据评估逐值完全一致，复核记录见 `reproduction_validation.json`。

### 分臂指标

| 任务 | 数据 | 臂 | rho_pos | rho_rot | rho_se3 | D_pos_mm | D_rot_deg |
|---|---|---|---:|---:|---:|---:|---:|
| cola | 校正前 | 左 | 5.921 | 5.924 | 5.911 | 161.42 | 45.21 |
| cola | 校正前 | 右 | 6.652 | 5.409 | 5.977 | 143.65 | 36.95 |
| cola | 校正后 | 左 | 1.999 | 1.894 | 1.929 | 54.50 | 14.45 |
| cola | 校正后 | 右 | 1.310 | 2.143 | 1.667 | 28.28 | 14.64 |
| drawer | 校正前 | 左 | 6.759 | 7.495 | 7.116 | 164.97 | 53.58 |
| drawer | 校正前 | 右 | 6.464 | 6.839 | 6.644 | 158.71 | 41.42 |
| drawer | 校正后 | 左 | 2.195 | 2.339 | 2.259 | 53.57 | 16.72 |
| drawer | 校正后 | 右 | 2.056 | 2.582 | 2.296 | 50.47 | 15.64 |

数据文件：[任务级 CSV](../outputs/five_task_correction_eval_20261012/metrics.csv) · [分臂 CSV](../outputs/five_task_correction_eval_20261012/metrics_per_arm.csv) · [完整精度 JSON](../outputs/five_task_correction_eval_20261012/metrics.json)。

## 评估口径

- 比较 `finger_center_hys085__none` 和 `finger_center_hys085__position_rotation`；每任务使用两版本共同完成的源 ego episodes，并按配置检查夹爪事件健康性。
- Cube、hanging、bowl 使用右臂；cola、drawer 分别评估左右臂，任务级结果取两个活动臂的指标算术均值。
- 使用各任务 `cfg/evaluation/{task}_arx.yaml`。前三类及 cola 的事件序列为 `open, close, open`；drawer 为 `open, close, open, close`。
- 仅评估首末夹爪事件之间的片段，重采样为 200 点；位置做无窗口约束 DTW，旋转误差沿该位置 DTW 路径计算。最大缺测插值间隔为 0.25 秒。
- 真机 episode 0–9 为校正参考，10–19 为留出评估集。位置和旋转两个校正文件分别核对，不与留出真机重叠。
- Bowl 真机 episode 18 仅检测到 `open, close`，缺少最后的 `open`，不满足配置要求。输出目录内保存单独的健康 split，评估其余 9 条；未修改仓库原始 split，也未从校正参考集中补样本。
- 真机内部基准只使用该任务的健康留出集，通过两两 DTW 和 leave-one-out 均值计算。每条 ego 与全部留出真机逐一比较；先对真机参考求均值，再对 ego episodes 等权求均值。
- 仓库评估器读取 manifest 指向的 canonical EEF JSON，并核验文件指纹和所指定 LeRobot 数据集的打包 provenance；不是对关节角或 IK→FK 回转结果评估。
- 这是针对留出真机的分布对齐评估，不是真机与 ego 逐条配对的真实误差，也不代表下游策略成功率。

## 指标定义

对每条 ego、每个活动臂，记与各留出真机比较后平均的位置距离为 `D_pos`，平均旋转距离为 `D_rot`，留出真机内部基准为 `F_pos` 和 `F_rot`：

- `D_pos_mm`：沿 DTW 累计欧氏位置距离，除以两段有效重采样点数的平均值，再换算为毫米。此分母不是 DTW 路径长度。
- `D_rot_deg`：沿位置 DTW 路径的 SO(3) 相对旋转角均值，单位为度。
- `rho_pos = D_pos / F_pos`。
- `rho_rot = D_rot / F_rot`。
- `rho_se3 = sqrt(rho_pos * rho_rot)`，先逐 ego 计算，再取均值；不能由表中已聚合的两列 rho 再开方替代。

上述五项越低越好；`rho = 1` 表示相应指标达到留出真机内部差异的尺度。不同任务的归一化基准不同，不能将 rho 的跨任务大小直接当作绝对误差比较。

## 重现和审计

运行入口：[scripts/evaluate_five_task_corrections.py](../scripts/evaluate_five_task_corrections.py)。服务器输出目录为 `/mnt/data/xule/TATE/outputs/five_task_correction_eval_20261012`。

```bash
cd /mnt/data/xule/TATE
/mnt/data/envs/lifego/bin/python \
  outputs/five_task_correction_eval_20261012/evaluate_five_task_corrections.py \
  --output outputs/five_task_correction_eval_20261012 --workers 3 --skip-dispersion
```

运行器直接调用 `evaluation.run_eval.run`，不改动仓库评估代码。为加速，仅在当前进程中将 DTW 原函数用 Numba 编译，并将不受编译器支持的 `argmin(tuple)` 改写为等价的 `argmin(asarray(tuple))`。不开启 fastmath；24 个覆盖随机代价、零代价、常数代价、不同长度和窗口的用例均要求路径索引及累计代价与原 Python 实现完全一致，验证记录保存在各任务的 `dtw_validation.json`。

`--skip-dispersion` 只跳过本次未请求的独立 `eta_disp` 指标，避免额外的 ego–ego 全配对计算；不改变五项请求指标的对齐、真机基准或聚合。首次运行已完成的变体通过原生 `--resume` 复用，因此部分 summary 仍保留已算出的 `eta_disp`，其余明确标为未计算。本报告不比较该列。

各任务还保存 `preflight.json`（真实参考集、健康排除原因、两个校正文件指纹和全部评估源码指纹），以及评估器原生的 `comparison.json/csv`、`summary.json`、`episodes.csv`、`exclusions.json`、`real_to_real_noise_floor.json`。逐配对 alignment NPZ 保留在服务器。
