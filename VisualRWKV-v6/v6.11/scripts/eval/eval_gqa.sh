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
#
SPLIT="llava_gqa_testdev_balanced"
CHUNKS=${#GPULIST[@]}

# 使用basename命令获取父目录名称
exp_name=$(basename "${parent_dir}")
# add ctx_len, grid_size, image_position to exp_name
exp_name="${exp_name}_ctx${ctx_len}_grid${grid_size}_pos${image_position}"
echo "exp name: $exp_name, model path: $model_path"
echo "ctx_len: $ctx_len, grid_size: $grid_size, n_embd: $n_embd, n_layer: $n_layer"
echo "eval dir: $eval_dir"
echo "image_position: $image_position"
echo "num of chunks: $CHUNKS"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python evaluate.py \
        --ctx_len $ctx_len --grid_size $grid_size --n_embd $n_embd --n_layer $n_layer \
        --model_path $model_path \
        --image_folder $eval_dir/eval/gqa/images \
        --question_file $eval_dir/eval/gqa/$SPLIT.jsonl \
        --output_file $eval_dir/eval/gqa/answers/$SPLIT/$exp_name/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --image_position $image_position &
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