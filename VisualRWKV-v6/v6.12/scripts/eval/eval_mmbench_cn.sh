#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

model_path=$1
ctx_len=$2
grid_size=$3
n_embd=$4
n_layer=$5
eval_dir=$6
vision_tower_path=$7
image_position=$8
# 使用dirname命令获取父目录的路径
parent_dir=$(dirname "${model_path}")
# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."
# get the name of the model without extension
model_name=$(basename "${model_path}")
model_name="${model_name%.*}"
# 打印当前工作目录
echo "Current working directory: $(pwd)"
#
SPLIT="mmbench_dev_cn_20231003"
CHUNKS=${#GPULIST[@]}

# 使用basename命令获取父目录名称
exp_name=$(basename "${parent_dir}")
# add model name to exp name
exp_name="${exp_name}_${model_name}"
echo "exp name: $exp_name, model path: $model_path"
echo "ctx_len: $ctx_len, grid_size: $grid_size, n_embd: $n_embd, n_layer: $n_layer"
echo "eval dir: $eval_dir"
echo "vision tower path: $vision_tower_path", "image_position: $image_position"
echo "num of chunks: $CHUNKS"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python evaluate.py \
        --ctx_len $ctx_len --grid_size $grid_size --n_embd $n_embd --n_layer $n_layer \
        --vision_tower_name $vision_tower_path \
        --model_path $model_path \
        --dataset_name mmbench_cn \
        --question_file $eval_dir/eval/mmbench/$SPLIT.tsv \
        --output_file $eval_dir/eval/mmbench/answers/$SPLIT/$exp_name/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --image_position $image_position &
    echo "Started chunk $IDX"
done

wait

output_file=$eval_dir/eval/mmbench/answers/$SPLIT/$exp_name/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $eval_dir/eval/mmbench/answers/$SPLIT/$exp_name/${CHUNKS}_${IDX}.jsonl >> $output_file
done

python eval/convert_mmbench_for_submission.py \
    --annotation-file $eval_dir/eval/mmbench/$SPLIT.tsv \
    --result-dir $eval_dir/eval/mmbench/answers/$SPLIT \
    --upload-dir $eval_dir/eval/mmbench/answers_upload/$SPLIT \
    --experiment $exp_name