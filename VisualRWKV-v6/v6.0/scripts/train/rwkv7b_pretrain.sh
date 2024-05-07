export CUDA_VISIBLE_DEVICES=0,1,2,3

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"

python train.py --load_model /houhaowenT/huggingface_models/BlinkDL/rwkv-6-world/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth \
    --wandb "" --proj_dir out/rwkv7b-v060_pretrain \
    --data_file /houhaowenT/huggingface_datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 1024 --epoch_steps 1000 --epoch_count 18 --epoch_begin 0 --epoch_save 0 \
    --micro_bsz 8 --accumulate_grad_batches 4 --n_layer 32 --n_embd 4096 --pre_ffn 0 \
    --lr_init 1e-3 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 4 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 0 \
    --image_folder /houhaowenT/huggingface_datasets/LLaVA-Pretrain/images/ \
    --vision_tower_name /houhaowenT/huggingface_models/openai/clip-vit-large-patch14-336 \
    --freeze_rwkv 32 --freeze_proj 0 --detail low --grid_size -1 --image_position first \
    --enable_progress_bar True