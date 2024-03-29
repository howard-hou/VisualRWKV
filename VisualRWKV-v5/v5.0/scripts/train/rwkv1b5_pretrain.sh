export CUDA_VISIBLE_DEVICES=0 

python train.py --load_model "/houhaowen/huggingface_models/BlinkDL/rwkv-5-world/RWKV-5-World-1B5-v2-20231025-ctx4096.pth" \
    --wandb "" --proj_dir "out/rwkv1b5_pretrain" \
    --data_file "/houhaowen/huggingface_datasets/liuhaotian/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json" \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 256 --epoch_steps 0 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 16 --accumulate_grad_batches 16 --n_layer 24 --n_embd 2048 --pre_ffn 0 \
    --lr_init 1e-4 --lr_final 1e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0 \
    --image_folder /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Pretrain/images/ \
    --vision_tower_name /houhaowen/huggingface_models/openai/clip-vit-base-patch32 \
    --freeze_rwkv 1