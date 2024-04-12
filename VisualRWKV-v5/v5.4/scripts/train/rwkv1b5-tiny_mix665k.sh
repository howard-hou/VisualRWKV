export CUDA_VISIBLE_DEVICES=0

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"
tiny_att_layer=$1
freeze_rwkv=$((24-$tiny_att_layer))

echo "tiny_att_layer: $tiny_att_layer"
echo "freeze_rwkv: $freeze_rwkv"

python train.py --model_path out/rwkv1b5-tiny${tiny_att_layer}_clip-sam-dino_pretrain-concatseq/rwkv-69.pth \
    --wandb "" --proj_dir out/rwkv1b5-tiny${tiny_att_layer}_clip-sam-dino_mix665k-concatseq \
    --data_file /root/autodl-tmp/LLaVA-Instruct-150K/shuffled_llava_v1_5_mix665k.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 1024 --epoch_steps 1000 --epoch_count 84 --epoch_begin 0 --epoch_save 0 \
    --micro_bsz 8 --accumulate_grad_batches 16 --n_layer 24 --n_embd 2048 --pre_ffn 0 \
    --lr_init 2e-5 --lr_final 0 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2_offload --grad_cp 0 \
    --image_folder /root/autodl-tmp/LLaVA-Instruct-150K/images/ \
    --vision_tower_clip /root/autodl-tmp/huggingface_models/openai/clip-vit-large-patch14 \
    --vision_tower_sam /root/autodl-tmp/huggingface_models/facebook/sam_vit_b_01ec64.pth \
    --vision_tower_dino /root/autodl-tmp/huggingface_models/facebook/dinov2-large \
    --freeze_rwkv $freeze_rwkv --unified_vision_dim 1024 --image_position first \
    --enable_progress_bar True --tiny_att_dim 2048 --tiny_att_layer ${tiny_att_layer} --freeze_tiny_att 0