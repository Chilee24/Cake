#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python3 train.py \
  --train_list "/workspace/data_nas06/cuongnd36/dashcam/k400/Kinetics-400/kinetics400_train_list_videos.txt" \
  --train_root "/workspace/data_nas06/cuongnd36/dashcam/k400/Kinetics-400/videos_train" \
  --val_list "/workspace/data_nas06/cuongnd36/dashcam/k400/Kinetics-400/kinetics400_val_list_videos.txt" \
  --val_root "/workspace/data_nas06/cuongnd36/dashcam/k400/Kinetics-400/videos_val" \
  --raft_weights "/workspace/data_nas06/cuongnd36/dashcam/train_teacher/checkpoint/raft_large_C_T_SKHT_V2-ff5fadd5.pth" \
  --flow_teacher_weights "/workspace/data_nas06/cuongnd36/dashcam/checkpoints_x3d_of/best_x3d_flow.pth" \
  --student_pretrained "/workspace/data_nas06/cuongnd36/dashcam/train_teacher/checkpoint/X3D_S.pyth" \
  --save_dir "./new_mgd_freeze_no_cls" \
  --batch_size 150 \
  --epochs 200 \
  --clip_len 13 \
  --lr_backbone 0.0001 \
  --lr_new_layers 0.001 \
  --workers 8 \
  --distill_type mgd \
  --alpha_distill 2 \
  --freeze_rgb