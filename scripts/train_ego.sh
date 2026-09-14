cd /home/admin/workspace/yuanmingqi/code/TATE/thirdparty/openpi

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  ../../training/train_arx_eef_pytorch.py \
  --repo-id local/arx_eef_stack_cube_arx \
  --dataset-root ../../outputs/lerobot/local/arx_eef_stack_cube_arx \
  --model pi05 \
  --pytorch-weight-path /mnt/workspace/sunxiaoquan/models/pi05_base \
  --batch-size 8