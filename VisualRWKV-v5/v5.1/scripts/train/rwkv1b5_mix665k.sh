export CUDA_VISIBLE_DEVICES=0

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"
image_scanning=$1
echo "image_scanning: $image_scanning"

python train.py --model_path out/rwkv1b5-${image_scanning}_pretrain/rwkv-0.pth \
    --wandb "" --proj_dir out/rwkv1b5-${image_scanning}_mix665k \
    --data_file /data/huggingface_datasets/LLaVA-Instruct-150K/shuffled_llava_v1_5_mix665k.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 1024 --epoch_steps 1000 --epoch_count 84 --epoch_begin 0 --epoch_save 0 \
    --micro_bsz 8 --accumulate_grad_batches 8 --n_layer 24 --n_embd 2048 --pre_ffn 0 \
    --lr_init 1e-3 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 0 \
    --image_folder /data/huggingface_datasets/LLaVA-Instruct-150K/images/ \
    --vision_tower_name /data/huggingface_models/openai/clip-vit-large-patch14-336 \
    --freeze_rwkv 24 --detail low --grid_size -1 --image_position first \
    --enable_progress_bar True --image_scanning ${image_scanning}