export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"


python train.py --model_path 'out/rwkv1b5-v0612-448-mlp_mix665k/rwkv-14.pth' \
    --wandb "rwkv1b5-v0621_m4instruct615k" --proj_dir out/rwkv1b5-v0621_m4instruct615k \
    --data_file /root/autodl-tmp/M4-Instruct-Data/m4_instruct_annotations_fixed_shuffled.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 256 --epoch_steps 1000 --epoch_count 38 --epoch_begin 0 --epoch_save 19 \
    --micro_bsz 1 --accumulate_grad_batches 4 --n_layer 24 --n_embd 2048 --pre_ffn 0 \
    --lr_init 6e-5 --lr_final 1.5e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 2 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --image_folder /root/autodl-tmp/M4-Instruct-Data/images \
    --vision_tower_dir /root/autodl-tmp/huggingface_models \
    --freeze_rwkv 0 --freeze_proj 0  --num_token_per_image 64 --proj_type mlp \
    --fusion_layer 0 \
    --enable_progress_bar True
