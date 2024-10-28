export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"


python train.py --model_path out/rwkv1b6-v0622-448-mlp_pretrain/rwkv-3.pth \
    --wandb "rwkv1b5-v0622-448-mlp_mix665k" --proj_dir out/rwkv1b5-v0622-448-mlp_mix665k \
    --data_file /houhaowenT/huggingface_datasets/LLaVA-Instruct-150K/shuffled_llava_v1_5_mix665k_5rounds.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 2048 --epoch_steps 1000 --epoch_count 15 --epoch_begin 0 --epoch_save 0 \
    --micro_bsz 16 --accumulate_grad_batches 2 --n_layer 24 --n_embd 2048 --pre_ffn 0 \
    --lr_init 6e-5 --lr_final 1.5e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 4 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --image_folder /houhaowenT/MyDataset/VisualRWKV-Instruct/images/ \
    --vision_tower_dir /houhaowenT/huggingface_models/ \
    --freeze_rwkv 0 --freeze_proj 0 --proj_type mlp \
    --num_token_per_image 1024 --n_state_encoder_layer 6 --state_encoder_max_feature_len 1024 \
    --enable_progress_bar True
