#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

python3 viz_2.py \
  --checkpoint "/workspace/data_nas06/cuongnd36/dashcam/cake/checkpoints_k400_distill_mgd/last.pth" \
  --val_list "/workspace/data_nas06/cuongnd36/dashcam/k400/Kinetics-400/kinetics400_train_list_videos.txt" \
  --val_root "/workspace/data_nas06/cuongnd36/dashcam/k400/Kinetics-400/videos_train" \