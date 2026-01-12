ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29490 \
  train.py \
  --config_path configs/self_forcing_ode.yaml \
  --no_visualize \
  --logdir checkpoint/ode_checkpoints