# 随机红方块：100 条 ego 校正前后 × 20/40/60 条真机混训数据

服务器：`root@8.160.160.64`，SSH 端口 `1020`。

六组数据均位于 `/mnt/data/xule/TATE/outputs/lerobot/local/`：

| ego 标签 | 真机源演示数 | 数据集目录 | 总训练帧 | 输出 episodes |
|---|---:|---|---:|---:|
| 校正前 | 20 | `arx_eef_redcube_random_cotrain_e100r20_none_nodropout_train` | 36,970 | 121 |
| 校正前 | 40 | `arx_eef_redcube_random_cotrain_e100r40_none_nodropout_train` | 44,414 | 141 |
| 校正前 | 60 | `arx_eef_redcube_random_cotrain_e100r60_none_nodropout_train` | 51,752 | 161 |
| 校正后 | 20 | `arx_eef_redcube_random_cotrain_e100r20_unpaired_bridge_nodropout_train` | 36,970 | 121 |
| 校正后 | 40 | `arx_eef_redcube_random_cotrain_e100r40_unpaired_bridge_nodropout_train` | 44,414 | 141 |
| 校正后 | 60 | `arx_eef_redcube_random_cotrain_e100r60_unpaired_bridge_nodropout_train` | 51,752 | 161 |

每组都包含同样的 **100 条源 ego 演示**。其中一条有 EEF 缺测区间，遵循现有构建器的规则拆分为连续片段，因此输出为 101 个 ego episodes，共 29,646 个有效 state/action 样本。动作不会跨越缺失帧。校正前后保留的源帧和片段完全一致。

## 数据语义及真机选择

- LeRobot v2.1，30 FPS，仅 `train`，不划分 eval。
- `observation.state` / `action` 均为 16D：左臂 xyz、xyzw 四元数、夹爪；右臂同样 8 维。位置单位为米。
- state 来自当前帧，action 来自原始下一帧；每个片段的最后 action 仍有对应的视频尾帧。
- 夹爪为 `0=open, 1=closed`，与现有 cotrain builder 一致。
- 左臂使用 canonical identity，state/action 的前 8 维 mask 为 false，右臂为 true。
- ego 图像掩码为 `[head=True, left=False, right=False]`；真机为三路全开。不额外进行 camera dropout。
- 校正前标签来自 `eef_raw.json`，校正后来自无配对 SceneBridge 主实验 20261011 的 `eef_corrected.json`。
- ego 直接读取这些 EEF 位姿，不做 IK→FK 回转。真机仍按原有 builder 使用采集关节状态做 FK。这样保留已经评测的校正前后标签，不会误把 ego 采集时基本静止的机器人关节当作 ego 动作。

20 条真机选用该校正模型已使用的 20 条参考：

`2, 22, 52, 54, 58, 66, 68, 78, 80, 86, 102, 108, 112, 116, 126, 128, 132, 156, 176, 178`

其余真机按固定种子 20261011 打乱，逐级增加为 40、60 条。20 ⊂ 40 ⊂ 60；每种规模的校正前后使用完全相同的真机数据。因此 20 条混训实验不会额外依赖另一组未计数的校正参考。40、60 条混训仍使用同一批已经校正好的 100 条 ego，**没有重新用 40/60 条真机校正 ego**。

## 生成与复查

新增专用入口复用 `training/build_arx_cotrain_datasets.py` 中的 DatasetWriter、FK、掩码、连续片段和视频裁剪逻辑。原有脚本默认读取关节标签，不适合直接指向这批散落在 worker 目录中的 EEF JSON，因此使用如下入口生成全部六组：

```bash
cd /mnt/data/xule/TATE
PY=thirdparty/openpi/.venv/bin/python
$PY training/build_unpaired_redcube_cotrain.py
```

此命令会拒绝覆盖已完成结果；现有数据集已生成，不需要再次运行。脚本不进行策略训练。

完整审计位于：

```text
/mnt/data/xule/TATE/outputs/unpaired_redcube_20261011/cotrain_build/
  plan.json                 # 六组配置、全部源 episode IDs、嵌套真机集合
  eef_source_hashes.json     # 每条校正前后 EEF 的路径与 SHA-256
  complete.json             # 六组最终计数
  validation.json           # 全量标签、帧索引、视频长度及对照一致性验证
  loader_validation.json    # 实际 LeRobot loader 读取与动作窗口验证
```

每个数据集自身的 `cotrain_provenance.json` 也保存了真实源 episode ID、源帧范围、分段情况、校正模型及数据预算。训练使用的 repo ID 为 `local/` 加上表中的目录名，数据根目录为 `/mnt/data/xule/TATE/outputs/lerobot`。
