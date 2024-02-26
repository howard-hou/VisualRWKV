export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python train.py --model_path "out/rwkv3b-vitl336p14_pretrain/rwkv-34.pth" \
    --wandb "" --proj_dir "out/rwkv3b-vitl336p14-577token_mix665k_8gpu" \
    --data_file "/data/howard/huggingface_datasets/LLaVA-Instruct-150K/shuffled_llava_v1_5_mix665k.json" \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 1024 --epoch_steps 1000 --epoch_count 84 --epoch_begin 0 --epoch_save 10 \
    --micro_bsz 1 --accumulate_grad_batches 16 --n_layer 32 --n_embd 2560 --pre_ffn 0 \
    --lr_init 2e-5 --lr_final 2e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 8 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --image_folder /data/howard/huggingface_datasets/LLaVA-Instruct-150K/images/ \
    --vision_tower_name /data/howard/huggingface_models/openai/clip-vit-large-patch14-336 \
    --freeze_rwkv 0 --freeze_proj 0 --detail low --grid_size -1