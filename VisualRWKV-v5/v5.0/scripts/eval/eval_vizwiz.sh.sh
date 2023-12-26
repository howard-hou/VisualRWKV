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
    --image_folder /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/vizwiz/test \
    --question_file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/vizwiz/llava_test.jsonl \
    --output_file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/vizwiz/answers/$exp_name.jsonl

python eval/convert_vizwiz_for_submission.py \
    --annotation-file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/vizwiz/llava_test.jsonl \
    --result-file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/vizwiz/answers/$exp_name.jsonl \
    --result-upload-file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/vizwiz/answers_upload/$exp_name.json