#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

model_path=/mnt/data2/Mutil_data/VisualRWKV/VisualRWKV/VisualRWKV-v6/v6.11/out/rwkv1b5-v061_mix665k/rwkv-83.pth
ctx_len=2048
grid_size=-1
n_embd=2048
n_layer=24
eval_dir=/mnt/data2/Mutil_data/eval_dataset
image_position=middle
# 使用dirname命令获取父目录的路径
parent_dir=$(dirname "${model_path}")

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"

# 使用basename命令获取父目录名称
exp_name=$(basename "${parent_dir}")
# add ctx_len, grid_size, image_position to exp_name
exp_name="${exp_name}_ctx${ctx_len}_grid${grid_size}_pos${image_position}"
echo "exp name: $exp_name, model path: $model_path"
echo "ctx_len: $ctx_len, grid_size: $grid_size, n_embd: $n_embd, n_layer: $n_layer"
echo "eval dir: $eval_dir"
echo "image position: $image_position"

python evaluate.py \
    --ctx_len $ctx_len --grid_size $grid_size --n_embd $n_embd --n_layer $n_layer \
    --model_path $model_path \
    --image_folder $eval_dir/eval/scienceqa/images/test \
    --question_file $eval_dir/eval/scienceqa/llava_test_CQM-A.json \
    --output_file $eval_dir/eval/scienceqa/answers/$exp_name.jsonl \
    --image_position $image_position
shid
python eval/eval_science_qa.py \
    --base-dir $eval_dir/eval/scienceqa \
    --result-file $eval_dir/eval/scienceqa/answers/$exp_name.jsonl \
    --output-file $eval_dir/eval/scienceqa/answers/$exp_name_output.jsonl \
    --output-result $eval_dir/eval/scienceqa/answers/$exp_name_result.json