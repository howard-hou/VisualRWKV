model_path=$1
ctx_len=$2
grid_size=$3
# 使用dirname命令获取父目录的路径
parent_dir=$(dirname "${model_path}")

# 使用basename命令获取父目录名称
exp_name=$(basename "${parent_dir}")
echo "exp name: $exp_name, model path: $model_path, ctx len: $ctx_len, grid size: $grid_size"

python evaluate.py \
    --ctx_len $ctx_len --grid_size $grid_size \
    --vision_tower_name /houhaowen/huggingface_models/openai/clip-vit-large-patch14-336 \
    --model_path $model_path \
    --image_folder /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/pope/val2014 \
    --question_file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/pope/llava_pope_test.jsonl \
    --output_file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/pope/answers/$exp_name.jsonl

python eval/eval_pope.py \
    --annotation-dir /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/pope/coco \
    --question-file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/pope/llava_pope_test.jsonl \
    --result-file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/pope/answers/$exp_name.jsonl