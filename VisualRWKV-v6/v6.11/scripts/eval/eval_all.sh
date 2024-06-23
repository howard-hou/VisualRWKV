export CUDA_VISIBLE_devices=0

cd "$(dirname "$(dirname "$0")")/.."

python /mnt/data2/Mutil_data/VisualRWKV/VisualRWKV/VisualRWKV-v6/v6.11/scripts/eval/scan_hyperparameter.py \
      --model_path /mnt/data2/Mutil_data/VisualRWKV/VisualRWKV/VisualRWKV-v6/v6.11/out/rwkv1b5-v061_mix665k/rwkv-83.pth \
      --ctx_len 2048 --grid_size -1 \
      --n_embd 2048 --n_layer 24 \
      --eval_dir /mnt/data2/Mutil_data/eval_dataset \
      --image_position middle \
      --task_names MME,gqa,textvqa,scienceqa,vqav2,vizwiz,pope \
      --hyperparameter image_position --hyperparameter_values middle

