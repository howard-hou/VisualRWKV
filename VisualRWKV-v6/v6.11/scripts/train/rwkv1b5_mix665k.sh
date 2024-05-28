export CUDA_VISIBLE_DEVICES=0

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"


python train.py --load_model /mnt/data2/Mutil_data/VisualRWKV/VisualRWKV/VisualRWKV-v6/v6.1/out/rwkv1b5-v061_pretrain/rwkv1b5-v061_pretrain_rwkv.pth \
    --wandb "" --proj_dir out/rwkv1b5-v061_mix665k \
    --data_file /mnt/data2/Mutil_data/Visual_IMage_Data_collection/LLaVA-Instruct-150K/shuffled_llava_v1_5_mix665k_reformatted.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 2048 --epoch_steps 1000 --epoch_count 666 --epoch_begin 0 --epoch_save 111 \
    --micro_bsz 1 --accumulate_grad_batches 128 --n_layer 24 --n_embd 2048 --pre_ffn 0 \
    --lr_init 2e-5 --lr_final 2e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 0 \
    --image_folder /mnt/data2/Mutil_data/Visual_IMage_Data_collection/LLaVA-Instruct-150K/ \
    --freeze_rwkv 12 --freeze_proj 0 --detail low --grid_size -1 --image_position middle \
    --enable_progress_bar True
