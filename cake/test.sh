CUDA_VISIBLE_DEVICES=0 python test.py \
  --video_train "/workspace/data_nas01/cuongnd36/thumos14/train" \
  --anno_train "/workspace/data_nas01/cuongnd36/thumos14/annotation_train" \
  --video_test "/workspace/data_nas01/cuongnd36/thumos14/test" \
  --anno_test "/workspace/data_nas01/cuongnd36/thumos14/annotation_test" \
  --json_file "/workspace/raid/os_callbot/kiennh/oad/MiniROAD/data_info/video_list.json" \
  --weights "/workspace/data_nas06/cuongnd36/dashcam/cake/new_mse_freeze_no_cls/mse_40.32.pth" \
  --output_dir "/workspace/raid/os_callbot/kiennh/oad/data/cake_4" \
  --batch_size 160


# CUDA_VISIBLE_DEVICES=6 python test_tvseries.py \
#   --video_dir "/workspace/data_nas06/cuongnd36/dashcam/tvseries/video" \
#   --anno_train "/workspace/data_nas06/cuongnd36/dashcam/tvseries/label/GT-train.txt" \
#   --anno_test "/workspace/data_nas06/cuongnd36/dashcam/tvseries/label/GT-test.txt" \
#   --json_file "/workspace/raid/os_callbot/kiennh/oad/MiniROAD/data_info/video_list.json" \
#   --weights "/workspace/data_nas06/cuongnd36/dashcam/cake/new_mse_freeze_no_cls/mse_40.32.pth" \
#   --output_dir "/workspace/raid/os_callbot/kiennh/oad/data/tvseries_4" \
#   --batch_size 220