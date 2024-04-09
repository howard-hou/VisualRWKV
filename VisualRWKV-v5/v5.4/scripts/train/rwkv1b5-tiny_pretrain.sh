export CUDA_VISIBLE_DEVICES=0

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"
tiny_att_layer=$1
echo "tiny_att_layer: $tiny_att_layer"

python train.py --load_model /root/autodl-tmp/huggingface_models/BlinkDL/rwkv-5-world/RWKV-5-World-1B5-v2-20231025-ctx4096.pth \
    --wandb "" --proj_dir out/rwkv1b5-tiny${tiny_att_layer}_pretrain \
    --data_file /root/autodl-tmp/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 1024 --epoch_steps 1000 --epoch_count 70 --epoch_begin 0 --epoch_save 0 \
    --micro_bsz 8 --accumulate_grad_batches 8 --n_layer 24 --n_embd 2048 --pre_ffn 0 \
    --lr_init 1e-3 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 0 \
    --image_folder /root/autodl-tmp/LLaVA-Pretrain/images/ \
    --vision_tower_name /root/autodl-tmp/huggingface_models/openai/clip-vit-large-patch14-336 \
    --freeze_rwkv 24 --detail low --grid_size -1 --image_position first \
    --enable_progress_bar True --tiny_att_dim 2048 --tiny_att_layer ${tiny_att_layer} --freeze_tiny_att 0