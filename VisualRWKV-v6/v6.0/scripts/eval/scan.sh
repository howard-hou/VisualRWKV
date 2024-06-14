

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python /houhaowenT/MyCode/VisualRWKV/VisualRWKV-v6/v6.0/scripts/eval/scan_hyperparameter.py \
--model_path out/rwkv7b-v060_mix665k_20rounds/rwkv-9.pth \
--ctx_len 2048 --grid_size -1 --n_embd 4096 --n_layer 32 \
--eval_dir /houhaowenT/huggingface_datasets/LLaVA-Instruct-150K \
--vision_tower_path /houhaowenT/huggingface_models/openai/clip-vit-large-patch14-336 \
--task_names mme,gqa,textvqa,scienceqa,vqav2,vizwiz,pope  \
--hyperparameter image_position --hyperparameter_values middle