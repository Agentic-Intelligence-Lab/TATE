# MINT 与 WiLoR 的校正前 EEF 对比

实现入口：`preprocess/reconstruct_mint.py`；批量实验入口：`scripts/evaluate_mint_raw.py`。
MINT 使用外部官方仓库 https://github.com/wuji-technology/wuji-ego-mint ，模型为其公开的 `mint_v1`。

## 完整结果（2026-10-19）

两任务各完成 50 条 MINT 重建及原始 EEF 导出。使用每任务的 10 条留出真机（episode 10–19），未施加真机位置/旋转校正。下表只比较两方法共同有效的 ego 集合；全部指标越低越好，cola 为两个活动臂的等权均值。

|任务|方法|共同 ego 数|rho_pos|rho_rot|rho_se3|D_pos_mm|D_rot_deg|
|---|---|---:|---:|---:|---:|---:|---:|
|redcube|WiLoR|10|10.106|3.158|5.626|127.014|20.764|
|redcube|MINT|10|13.872|6.505|9.493|174.337|42.775|
|cola|WiLoR|48|6.293|5.725|5.980|152.701|41.490|
|cola|MINT|48|6.052|6.987|6.483|145.004|50.352|

在当前固定 hand2gripper 配置下，MINT 未带来整体改善：redcube 的位置距离增加 37.3%、旋转距离增加 106.0%；cola 的位置距离降低 5.0%，但旋转距离增加 21.4%，综合 rho_se3 增加 8.4%。这不是对 MINT 所有配置或手部重建真值精度的结论，本次仅评估原生 `blend`、无 UKF 的结果。

### 有效率与筛选影响

|任务/臂|WiLoR 事件健康 episodes|MINT 事件健康 episodes|
|---|---:|---:|
|redcube 右臂|50/50|10/50|
|cola 左臂|49/50|48/50|
|cola 右臂|50/50|49/50|

MINT 的所有活动臂帧均能导出 EEF，但有 EEF 不等于事件序列正确。Redcube 的 40 条异常中，29 条只检测到 `open, close`，10 条为 `open, close, open, close`，1 条有更多额外事件。未修改阈值，也未将不合格样本记为零距离。Redcube 的共同集合仅占 20%，因此距离指标必须连同此有效率解读。

### 产物与核验

- [完整报告](../outputs/mint_raw_comparison/report.md)、[任务及分臂 CSV](../outputs/mint_raw_comparison/metrics.csv)、[有效率 CSV](../outputs/mint_raw_comparison/coverage.csv)、[完整精度 JSON](../outputs/mint_raw_comparison/metrics.json)。
- 两任务 `common/comparison.json` 均为 `comparable: true`；redcube 每方法 100 个实际臂级配对，cola 每方法 960 个。共同集合重评估的距离分量与全量中的同一配对逐值一致，见 [结果审计](../outputs/mint_raw_comparison/result_audit.json)。
- 全部 100 条数据的帧数、30 FPS、相机外参、TCP 变换、运动限制和 hand2gripper 配置在两方法间一致，见 [输入审计](../outputs/mint_raw_comparison/input_audit.json)。
- WiLoR 全量五项指标复现仓库既有五任务报告；接入测试 3 项、原有预处理测试 11 项、评估测试 14 项全部通过。
- 本地保存两任务 `all/`、`common/` 汇总、配对距离记录、排除原因和审计文件；逐帧缓存与配对 alignment NPZ 保留在服务器 `/mnt/data/xule/TATE/outputs/mint_raw_{redcube,cola}`。
- 完成后的评估日志为服务器 `outputs/mint_raw_redcube_evaluation.log`、`outputs/mint_raw_cola_evaluation.log`。

## 接口与比较口径

- 官方 `StudentEngine` 预测相机系 MANO 参数，官方 `decode_hand_6d` / `run_mano` 处理腕部原点、左手姿态及 OpenPose 21 点顺序，再转换为 TATE 的 `aria_humanego_21`。
- 坐标单位为米，OpenCV 相机坐标系；不叠加 MINT 预测的相机轨迹，不对尺度做真机拟合。
- MINT 官方推理接口不接收实测相机内参，使用原生预测的相机系平移。缓存中的实测 K/d 用于来源记录和投影检查，不做额外焦距/主点平移修正；这与 WiLoR 的内参处理有所不同，属于本次原生方法对比的一部分。
- 缓存 schema 为 `tate.mint_hands` v1，与 WiLoR 缓存分开。`export_eef_from_cache` 接受两者，使用相同 hand2gripper、相机外参、TCP 变换、夹爪阈值和 EEF 运动限制。
- 默认 32 帧窗口、`blend` 重叠窗口融合、手存在概率阈值 0.5，不启用 UKF；`--hand-mode smooth` 可另开实验。保留原视频帧率和完整 episode 帧区间。
- 只替换手部重建分支；已有 WiLoR 自身的滤波/优化仍属于其基线。这里的“原始 EEF”指未施加真机位置/旋转校正，仍包含既有 EEF 导出配置中的运动限制。
- redcube 评估右臂，cola 评估双臂。真机 split、事件健康检查、200 点重采样、位置 DTW 和旋转距离计算均复用 `evaluation.run_eval`。
- `all/` 反映各方法在完整输入集合上的原生有效率及指标；`common/` 限制为两种方法、所有活动臂都满足事件健康要求的同一批 episodes。需进一步查看 `comparison.json`，确认对齐后实际配对也一致。
- 这是 ego 与留出真机轨迹的分布距离，不是逐帧配对手部重建真值误差。两任务的 rho 基准不同。

## 环境

本次服务器外部源码：`/mnt/data/xule/wuji-ego-mint`；隔离 Python：`/mnt/data/envs/tate-mint/bin/python`，复用 lifego 的 CUDA/PyTorch 包。
官方 checkpoint 下载脚本校验 SHA256：`7b2f0aa5dfd00c271bb2f12c841ccfcc70e81e4052d413eacb5fb42a1bcc36c8`。
本次官方源码 commit：`1bd21e4edd6d5d2d59789ce64ee5450de24a7c2b`。
MANO 使用服务器已有文件，分别链接到官方仓库 `assets/mano/mano_{left,right}`；模型文件不加入 TATE。

## 单视频重建

```bash
python -m preprocess.reconstruct_mint \
  --video /path/episode.mp4 --session outputs/mint_example \
  --mint-repo /mnt/data/xule/wuji-ego-mint \
  --checkpoint /mnt/data/xule/wuji-ego-mint/checkpoints/model.safetensors
python -m preprocess.export_eef \
  --hands outputs/mint_example/preprocess/mint_hands.json \
  --out outputs/mint_example/preprocess/eef_raw.json \
  --grasp-close-ratio 0.85 --grasp-open-ratio 0.90 --grasp-min-frames 10
```

将 `outputs/mint_example/preprocess/mint_hands.json` 传给已有 EEF 导出器的缓存参数即可。缓存记录上游 commit、权重 SHA256、窗口及时间区间；EEF 记录 `source_hand_cache` 和重建来源。

## 批量评估

在服务器 TATE 根目录运行，输出目录必须是新目录，避免混用配置不同的缓存：

```bash
/mnt/data/envs/tate-mint/bin/python -m scripts.evaluate_mint_raw \
  --experiment outputs/experiments/stack_redcube_v1_1 \
  --real-manifest outputs/arx_real_flange/stack_redcube_v1_arx/manifest.json \
  --real-split cfg/evaluation/splits/stack_redcube_v1.json \
  --eval-config cfg/evaluation/stack_redcube_v1_arx.yaml \
  --output outputs/mint_raw_redcube \
  --mint-repo /mnt/data/xule/wuji-ego-mint \
  --checkpoint /mnt/data/xule/wuji-ego-mint/checkpoints/model.safetensors

/mnt/data/envs/tate-mint/bin/python -m scripts.evaluate_mint_raw \
  --experiment outputs/experiments/cola_1012 \
  --real-manifest outputs/arx_real_flange/cola_arx/manifest.json \
  --real-split cfg/evaluation/splits/cola.json \
  --eval-config cfg/evaluation/cola_arx.yaml \
  --output outputs/mint_raw_cola \
  --mint-repo /mnt/data/xule/wuji-ego-mint \
  --checkpoint /mnt/data/xule/wuji-ego-mint/checkpoints/model.safetensors
```

`--limit 1` 用于冒烟验证；`--evaluate-only` 对已经完整导出的缓存/EEF 重新评估。
结果包含 `manifest.json`、`health.json`、`cohort.json`、`all/`、`common/`、逐 episode 缓存和原始 EEF。
两任务完成后运行 `python -m scripts.summarize_mint_raw` 生成 `outputs/mint_raw_comparison/{report.md,metrics.csv,coverage.csv,metrics.json}`。
DTW 使用已有五任务脚本的等价 Numba 加速及路径一致性验证；跳过本次未请求的独立 `eta_disp` 指标。

## 验证

`tests/test_mint_pipeline.py` 检查解剖关键点映射、缺测与时间戳，以及相同关键点经两种缓存 schema 导出的 EEF 完全一致；原有 `test_preprocess_pipeline.py` 验证 WiLoR 导出兼容性。
