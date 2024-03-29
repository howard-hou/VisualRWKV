export CUDA_VISIBLE_DEVICES=0 

python train.py --model_path "out/rwkv1b5_pretrain/rwkv-0.pth" \
    --wandb "" --proj_dir "out/rwkv1b5_mix665k" \
    --data_file "/houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/shuffled_llava_v1_5_mix665k.json" --data_type "json" --vocab_size 65536 \
    --ctx_len 256 --epoch_steps 0 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 8 --accumulate_grad_batches 16 --n_layer 24 --n_embd 2048 --pre_ffn 0 \
    --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0 \
    --image_folder /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/images/ \
    --vision_tower_name /houhaowen/huggingface_models/openai/clip-vit-base-patch32