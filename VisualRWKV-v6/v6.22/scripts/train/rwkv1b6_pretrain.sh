export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"


python train.py --load_model /houhaowenT/huggingface_models/BlinkDL/rwkv-6-world/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth \
    --wandb "rwkv1b6-v0622-448-mlp_pretrain" --proj_dir out/rwkv1b6-v0622-448-mlp_pretrain \
    --data_file /houhaowenT/huggingface_datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 256 --epoch_steps 1000 --epoch_count 17 --epoch_begin 0 --epoch_save 0 \
    --micro_bsz 16 --accumulate_grad_batches 4 --n_layer 24 --n_embd 2048 --pre_ffn 0 \
    --lr_init 1e-3 --lr_final 5e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 2 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --image_folder /houhaowenT/huggingface_datasets/LLaVA-Pretrain/images/ \
    --vision_tower_dir /houhaowenT/huggingface_models/ \
    --freeze_rwkv 24 --freeze_proj 0 --proj_type mlp \
    --num_token_per_image 64 --n_state_encoder_layer 6 --enable_state_encoder_pretrain_mode 1 \
    --enable_progress_bar True
