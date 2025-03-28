export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"


python train.py --load_model "" \
    --wandb "rwkv0b1-v0710_pretrain_imagenet" --proj_dir out/rwkv0b1-v0710_pretrain_imagenet \
    --data_file /home/rwkvos/howardhwhou/LLaVA-Imagenet/train.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 384 --epoch_steps 1000 --epoch_count 200 --epoch_begin 0 --epoch_save 10 \
    --micro_bsz 32 --accumulate_grad_batches 1 --n_layer 12 --n_embd 768 --pre_ffn 0 \
    --lr_init 2e-4 --lr_final 2e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 8 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --image_folder /home/rwkvos/howardhwhou/LLaVA-Imagenet/ \
    --image_size 256 --patch_size 16 --enable_pretrain_mode 1 
