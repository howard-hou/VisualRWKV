#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

model_path=$1
ctx_len=$2
grid_size=$3
# 使用dirname命令获取父目录的路径
parent_dir=$(dirname "${model_path}")
#
SPLIT="llava_gqa_testdev_balanced"
CHUNKS=${#GPULIST[@]}

# 使用basename命令获取父目录名称
exp_name=$(basename "${parent_dir}")
echo "exp name: $exp_name, model path: $model_path, ctx len: $ctx_len, grid size: $grid_size"

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python evaluate.py \
        --ctx_len $ctx_len --grid_size $grid_size \
        --vision_tower_name /houhaowen/huggingface_models/openai/clip-vit-large-patch14-336 \
        --model_path $model_path \
        --image_folder /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/images/gqa/images \
        --question_file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/gqa/$SPLIT.jsonl \
        --output_file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/gqa/answers/$exp_name/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python evaluate.py \
        --ctx_len $ctx_len --grid_size $grid_size \
        --vision_tower_name /houhaowen/huggingface_models/openai/clip-vit-large-patch14-336 \
        --model_path $model_path \
        --image_folder /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/images/gqa/images \
        --question_file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/gqa/$SPLIT.jsonl \
        --output_file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/gqa/answers/$SPLIT/$exp_name/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done

wait

output_file=/houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/gqa/answers/$SPLIT/$exp_name/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/gqa/answers/$SPLIT/$exp_name/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python eval/convert_gqa_for_eval.py --src $output_file \
    --dst /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/gqa/answers/$SPLIT/$exp_name/testdev_balanced_predictions.json

python eval/eval_gqa.py --questions /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/gqa/testdev_balanced_questions.json \
    --predictions /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/gqa/answers/$SPLIT/$exp_name/testdev_balanced_predictions.json