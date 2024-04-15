export CUDA_VISIBLE_DEVICES=0 

python train.py --load_model "" --wandb "" --proj_dir "out/dummy" \
    --data_file "dummy_data/dummy.json" --data_type "json" --vocab_size 65536 \
    --ctx_len 256 --epoch_steps 1000 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 8 --n_layer 6 --n_embd 512 --pre_ffn 0 \
    --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0 \
    --image_folder dummy_data/images/ \
    --vision_tower_name /houhaowen/huggingface_models/openai/clip-vit-base-patch32