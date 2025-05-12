export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"


python train.py --load_model /home/rwkvos/howardhwhou/huggingface_models/BlinkDL/rwkv-7-world/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth \
    --wandb "rwkv0b1-v0700_pretrain" --proj_dir out/rwkv0b1-v0700_pretrain \
    --data_file /home/rwkvos/howardhwhou/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 2048 --epoch_steps 1000 --epoch_count 2 --epoch_begin 0 --epoch_save 0 \
    --micro_bsz 32 --accumulate_grad_batches 1 --n_layer 12 --n_embd 768 --pre_ffn 0 \
    --lr_init 1e-3 --lr_final 5e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 8 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --image_folder /home/rwkvos/howardhwhou/LLaVA-Pretrain/images \
    --vision_tower_dir /home/rwkvos/howardhwhou/huggingface_models/ \
    --freeze_rwkv 12 --freeze_proj 0 --proj_type mlp \
    --num_token_per_image 1024
