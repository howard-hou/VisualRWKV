model_path=$1
ctx_len=$2
grid_size=$3
# 使用dirname命令获取父目录的路径
parent_dir=$(dirname "${model_path}")
SPLIT="mmbench_dev_20230712"
# 使用basename命令获取父目录名称
exp_name=$(basename "${parent_dir}")
echo "exp name: $exp_name, model path: $model_path, ctx len: $ctx_len, grid size: $grid_size"

python evaluate.py \
    --ctx_len $ctx_len --grid_size $grid_size \
    --vision_tower_name /houhaowen/huggingface_models/openai/clip-vit-large-patch14-336 \
    --model_path $model_path \
    --dataset_name mmbench \
    --question_file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/mmbench/$SPLIT.tsv \
    --output_file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/mmbench/answers/$SPLIT/$exp_name.jsonl

python eval/convert_mmbench_for_submission.py \
    --annotation-file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/mmbench/$SPLIT.tsv \
    --result-dir /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/mmbench/answers/$SPLIT \
    --upload-dir /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/mmbench/answers_upload/$SPLIT \
    --experiment $exp_name