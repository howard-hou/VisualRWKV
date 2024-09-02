export CUDA_VISIBLE_DEVICES=0,1
export VISION_HOME='/root/autodl-tmp/huggingface_models/'

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"

python train.py --load_model /root/autodl-tmp/huggingface_models/BlinkDL/rwkv-6-world/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth \
    --wandb "" --proj_dir out/rwkv1b5-v0612_pretrain \
    --data_file /root/autodl-tmp/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 1024 --epoch_steps 1000 --epoch_count 35 --epoch_begin 0 --epoch_save 0 \
    --micro_bsz 8 --accumulate_grad_batches 8 --n_layer 24 --n_embd 2048 --pre_ffn 0 \
    --lr_init 1e-3 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 2 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 0 \
    --image_folder /root/autodl-tmp/LLaVA-Pretrain/images/ \
    --freeze_rwkv 24 --grid_size -1 --image_position first \
    --enable_progress_bar True