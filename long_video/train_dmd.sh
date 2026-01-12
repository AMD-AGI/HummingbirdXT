ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29590 \
  train.py \
  --config_path self_forcing_dmd_vsink1.yaml \
  --no_visualize \
  --logdir checkpoint/dmd_checkpoints