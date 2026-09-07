# TATE


python -m preprocess.Preprocess \
  --mps_path outputs/test \
  --video_path DATA/arx_ego_dataset/test.mp4 \
  --cfg_path cfg/preprocess/base/Preprocess.yaml \
  --task serve_bread \
  --no-gif

python real2sim/replay_arx_mujoco.py --viewer

python real2sim/replay_arx_mujoco.py \
  --mode eef \
  --data outputs/test/preprocess/eef.json --viewer

python real2sim/replay_arx_mujoco.py \
  --mode joint \
  --data DATA/arx_ego_dataset/data/chunk-000/file-000.parquet \
  --out outputs/real2sim/file-000_joint_replay.mp4