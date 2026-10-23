# Bowl / hanging 1016 重采数据对比（2026-10-16）

## 结论

- **Bowl 1016 明显优于旧数据，可以作为更好的替代版本。** 校正后位置距离从 60.12 mm 降到 29.04 mm（下降 51.7%），旋转距离从 21.34° 降到 19.17°（下降 10.2%）；100 条全部完成校正，98 条满足评估事件要求。
- **Hanging 1016 的有效子集质量更好，但当前数据集不能直接整体替换旧版。** 校正后位置距离从 84.10 mm 降到 63.72 mm（下降 24.2%），旋转距离从 23.76° 降到 14.97°（下降 37.0%）；但 100 条中只有 62 条完成校正，最终仅 49 条满足完整评估条件。校正后数据中还保留了严重过冲的 episode 34。

## 同口径指标

全部指标越低越好。新旧数据均使用相同的真机参考、相同的 `cfg/evaluation/{task}_arx.yaml` 和相同的留出评估集。真机 0–9 用于拟合，10–19 用于评估；bowl 18 缺少最终 open，按旧评估相同规则排除。

| 任务 | 数据 | 阶段 | rho_pos | rho_rot | rho_se3 | D_pos_mm | D_rot_deg | 实际 ego 数 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| bowl | 旧 | 校正前 | 8.237 | 2.613 | 4.626 | 170.68 | 25.89 | 73 |
| bowl | 旧 | 校正后 | 2.901 | 2.154 | 2.455 | 60.12 | 21.34 | 73 |
| bowl | 1016 | 校正前 | 5.780 | 2.471 | 3.772 | 119.76 | 24.49 | 98 |
| bowl | 1016 | 校正后 | **1.401** | **1.935** | **1.631** | **29.04** | **19.17** | 98 |
| hanging | 旧 | 校正前 | 11.331 | 6.173 | 8.336 | 182.12 | 36.45 | 88 |
| hanging | 旧 | 校正后 | 5.233 | 4.024 | 4.451 | 84.10 | 23.76 | 88 |
| hanging | 1016 | 校正前 | 7.478 | 4.640 | 5.850 | 120.19 | 27.40 | 49 |
| hanging | 1016 | 校正后 | **3.965** | **2.535** | **3.011** | **63.72** | **14.97** | 49 |

[新旧校正后指标图](../outputs/recollected_task_comparison_20261016/metrics_new_vs_previous.png) · [PDF](../outputs/recollected_task_comparison_20261016/metrics_new_vs_previous.pdf)

### 新数据本身与校正分别贡献了什么

| 任务 | 对比 | D_pos 变化 | D_rot 变化 | rho_se3 变化 |
|---|---|---:|---:|---:|
| bowl | 新校正前 vs 旧校正前 | -29.8% | -5.4% | -18.4% |
| bowl | 新校正后 vs 旧校正后 | **-51.7%** | **-10.2%** | **-33.6%** |
| bowl | 新数据自身校正前→后 | -75.8% | -21.7% | -56.8% |
| hanging | 新校正前 vs 旧校正前 | -34.0% | -24.8% | -29.8% |
| hanging | 新校正后 vs 旧校正后 | **-24.2%** | **-37.0%** | **-32.4%** |
| hanging | 新数据自身校正前→后 | -47.0% | -45.4% | -48.5% |

两次实验的位置目标来自同一组真机 0–9，五个位置锚点逐坐标完全一致（最大差异 0 mm）。因此位置结果改善并不是换了真机目标。新采原始数据已经更接近真机；bowl 的位置校正进一步取得更大收益。Hanging 的位置校正相对降幅从旧版 53.8% 变为 47.0%，但新原始数据的位置距离先下降了 34.0%，所以最终仍更好。

旋转拟合反映出的全局偏置和 episode 间离散也都变小：bowl 偏置角从 18.01° 变为 14.09°，每条 ego 的偏置离散从 6.56° 变为 3.91°；hanging 从 32.43° 变为 26.12°，离散从 13.06° 变为 7.26°。这与新采数据校正前的旋转指标改善一致。

## 数据覆盖率与异常

| 任务 | 版本 | 源数据 | 完成校正 | 满足评估事件序列 |
|---|---|---:|---:|---:|
| bowl | 旧 | 100 | 74 | 73 |
| bowl | 1016 | 100 | **100** | **98** |
| hanging | 旧 | 100 | 91 | 88 |
| hanging | 1016 | 100 | **62** | **49** |

Bowl 1016 的覆盖率大幅提高。两条未参与指标的源 episode 73、75 夹爪事件序列不健康。

Hanging 1016 的 38 条校正跳过均由夹爪事件不足导致：21 条只检测到 1 个事件，15 条检测到 2 个，2 条没有事件。完成校正的 62 条中还有 13 条不满足严格的 `open, close, open` 序列，因此指标只统计 49 条。新旧指标比较的是各自健康子集，不是同一批 ego 演示；新 hanging 的数值改善不能抵消有效样本覆盖率下降。

[校正后完整 TCP 分布图](../outputs/recollected_task_comparison_20261016/corrected_position_distributions.png) · [PDF](../outputs/recollected_task_comparison_20261016/corrected_position_distributions.pdf)

按真机全量轨迹的逐轴最小/最大包络统计，bowl 新版落在包络外的有效帧比例从旧版 18.0% 降到 2.4%；hanging 从 22.5% 降到 18.8%。后者仍有显著离群点。

### Hanging episode 34 校正过冲

源 episode 34 检测到五次转换：`open, close, open, close, open`。评估器因序列不等于预期的三次转换而排除它，但批处理校正仍使用前三个事件作为锚点并将它打包进校正后数据集。

- 原始范围最高为 Y=21.36 cm、Z=5.87 cm；校正后达到 Y=65.32 cm、Z=76.56 cm。
- 五个校正锚点的最大位移为 265.53 mm。
- 锚点之间的 `min_bending` 传播在 frame 244 达到 928.10 mm，是最大锚点位移的 3.50 倍。

[episode 34 过冲诊断图](../outputs/recollected_task_comparison_20261016/hanging_episode34_overshoot.png) · [PDF](../outputs/recollected_task_comparison_20261016/hanging_episode34_overshoot.pdf) · [数值审计](../outputs/recollected_task_comparison_20261016/hanging_episode34_overshoot.json)

因此当前建议是：bowl 1016 可以优先替换旧版；hanging 1016 应先按严格事件序列重新过滤，禁止多余事件 episode 进入校正/打包，并处理或移除 min-bending 过冲后再用于训练。这里没有修改任何数据集。

## 输出与复现

- [完整精度汇总 JSON](../outputs/recollected_task_comparison_20261016/comparison_summary.json)
- [指标 CSV](../outputs/recollected_task_comparison_20261016/metrics_comparison.csv)
- [百分比变化 CSV](../outputs/recollected_task_comparison_20261016/percent_changes.csv)
- [分布审计 JSON](../outputs/recollected_task_comparison_20261016/distribution_audit.json)
- [评估脚本](../scripts/evaluate_five_task_corrections.py)
- [对比绘图脚本](../scripts/compare_recollected_bowl_hanging.py)
- [离群诊断脚本](../scripts/plot_hanging_1016_outlier.py)

服务器完整评估输出位于 `/mnt/data/xule/TATE/outputs/recollected_task_comparison_20261016`，其中保留逐 ego–real 配对的 alignment 文件。每个新实验内部的校正前后 `comparison` 均为 `comparable: true`，DTW 编译实现也通过与仓库原 Python 实现逐路径一致的 24 个验证用例。

