export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"


python train.py --load_model /data/huggingface_models/BlinkDL/rwkv-7-world/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth \
    --wandb "rwkv0b1-v0701_pretrain" --proj_dir out/rwkv0b1-v0701_pretrain \
    --data_file /data/huggingface_datasets/Infovqa/output.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 2048 --epoch_steps 1000 --epoch_count 4 --epoch_begin 0 --epoch_save 0 \
    --micro_bsz 1 --accumulate_grad_batches 1 --n_layer 12 --n_embd 768 --pre_ffn 0 \
    --lr_init 1e-3 --lr_final 5e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --image_folder /data/huggingface_datasets/Infovqa \
    --vision_tower_path /data/huggingface_models/siglip2-base-patch16-256 \
    --freeze_rwkv 12 --freeze_proj 0 --num_token_per_image 256
