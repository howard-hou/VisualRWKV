export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"

python train.py --load_model /houhaowenT/huggingface_models/BlinkDL/rwkv-6-world/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth \
    --wandb "rwkv7b-v0612-448-mlp_pretrain" --proj_dir out/rwkv7b-v0612-448-mlp_pretrain \
    --data_file /houhaowenT/huggingface_datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 1280 --epoch_steps 1000 --epoch_count 17 --epoch_begin 0 --epoch_save 0 \
    --micro_bsz 32 --accumulate_grad_batches 4 --n_layer 32 --n_embd 4096 --pre_ffn 0 \
    --lr_init 1e-3 --lr_final 5e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --image_folder /houhaowenT/huggingface_datasets/LLaVA-Pretrain/images/ \
    --vision_tower_dir /houhaowenT/huggingface_models/ \
    --freeze_rwkv 32 --freeze_proj 0 --proj_type mlp \
    --enable_progress_bar True
