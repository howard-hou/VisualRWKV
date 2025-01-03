#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

model_path=$1
ctx_len=$2
n_embd=$3
n_layer=$4
eval_dir=$5
vision_tower_dir=$6

num_token_per_image=1024
# 使用dirname命令获取父目录的路径
parent_dir=$(dirname "${model_path}")
# get the name of the model without extension
model_name=$(basename "${model_path}")
model_name="${model_name%.*}"

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"
#
SPLIT="llava_gqa_testdev_balanced"
CHUNKS=${#GPULIST[@]}

# 使用basename命令获取父目录名称
exp_name=$(basename "${parent_dir}")
# add model name to exp name
exp_name="${exp_name}_${model_name}"
echo "exp name: $exp_name, model path: $model_path"
echo "ctx_len: $ctx_len, n_embd: $n_embd, n_layer: $n_layer"
echo "num_token_per_image: $num_token_per_image"
echo "eval dir: $eval_dir"
echo "vision_tower_dir: $vision_tower_dir"
echo "num of chunks: $CHUNKS"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python evaluate.py \
        --ctx_len $ctx_len --n_embd $n_embd --n_layer $n_layer \
        --num_token_per_image $num_token_per_image \
        --vision_tower_dir $vision_tower_dir \
        --model_path $model_path \
        --image_folder $eval_dir/images/gqa/images \
        --question_file $eval_dir/eval/gqa/$SPLIT.jsonl \
        --output_file $eval_dir/eval/gqa/answers/$SPLIT/$exp_name/${CHUNKS}_${IDX}.jsonl \
        --num_token_per_image $num_token_per_image \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
    echo "Started chunk $IDX"
done

wait

output_file=$eval_dir/eval/gqa/answers/$SPLIT/$exp_name/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $eval_dir/eval/gqa/answers/$SPLIT/$exp_name/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python eval/convert_gqa_for_eval.py --src $output_file \
    --dst $eval_dir/eval/gqa/answers/$SPLIT/$exp_name/testdev_balanced_predictions.json

python eval/eval_gqa.py --questions $eval_dir/eval/gqa/testdev_balanced_questions.json \
    --predictions $eval_dir/eval/gqa/answers/$SPLIT/$exp_name/testdev_balanced_predictions.json