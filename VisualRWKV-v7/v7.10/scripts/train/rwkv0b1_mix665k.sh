export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"


python train.py --model_path out/rwkv0b1-v0710_pretrain_recap/rwkv-3.pth \
    --wandb "rwkv0b1-v0710_mix665k_recap" --proj_dir out/rwkv0b1-v0710_mix665k_recap \
    --data_file /home/rwkvos/howardhwhou/LLaVA-Instruct-150K/shuffled_llava_v1_5_mix665k_5rounds.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 2048 --epoch_steps 1000 --epoch_count 15 --epoch_begin 0 --epoch_save 7 \
    --micro_bsz 16 --accumulate_grad_batches 1 --n_layer 12 --n_embd 768 --pre_ffn 0 \
    --lr_init 6e-5 --lr_final 1.5e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 8 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --image_folder /home/rwkvos/howardhwhou/LLaVA-Instruct-150K/images \
    --image_size 256 --patch_size 16
