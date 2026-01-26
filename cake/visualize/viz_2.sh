#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

python3 viz_2.py \
  --checkpoint "/workspace/data_nas06/cuongnd36/dashcam/checkpoint_old/checkpoints_k400_distill_mgd_freeze_new_adapter/last.pth" \
  --val_list "/workspace/data_nas06/cuongnd36/dashcam/k400/Kinetics-400/kinetics400_train_list_videos.txt" \
  --val_root "/workspace/data_nas06/cuongnd36/dashcam/k400/Kinetics-400/videos_train" \
  --raft_weights "/workspace/data_nas06/cuongnd36/dashcam/train_teacher/checkpoint/raft_large_C_T_SKHT_V2-ff5fadd5.pth" \
  --flow_teacher_weights "/workspace/data_nas06/cuongnd36/dashcam/checkpoints_x3d_of/best_x3d_flow.pth" \
  --stride 8