export CUDA_VISIBLE_DEVICES=0

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"

python train.py --load_model /mnt/data2/Mutil_data/hf_models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth \
    --wandb "" --proj_dir out/rwkv1b5-v061_pretrain \
    --data_file /mnt/data2/Mutil_data/Visual_IMage_Data_collection/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 1024 --epoch_steps 1000 --epoch_count 140 --epoch_begin 0 --epoch_save 10 \
    --micro_bsz 4 --accumulate_grad_batches 32 --n_layer 24 --n_embd 2048 --pre_ffn 0 \
    --lr_init 1e-3 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 0 \
    --image_folder /mnt/data2/Mutil_data/Visual_IMage_Data_collection/LLaVA-Pretrain/images/ \
    --freeze_rwkv 24 --detail low --grid_size -1 --image_position first \
    --enable_progress_bar True