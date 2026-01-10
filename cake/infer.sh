CUDA_VISIBLE_DEVICES=7 python3 test_30views_k400.py \
  --root /workspace/data_nas06/cuongnd36/dashcam/k400/Kinetics-400/videos_val \
  --test_list /workspace/data_nas06/cuongnd36/dashcam/k400/Kinetics-400/kinetics400_val_list_videos.txt \
  --weights /workspace/data_nas06/cuongnd36/dashcam/cake/new_mse_freeze_no_cls/mse_40.32.pth \
  --clip_len 13