#!/bin/bash
# 
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

model_path=$1
ctx_len=$2
n_embd=$3
n_layer=$4
eval_dir=$5

# 使用dirname命令获取父目录的路径
parent_dir=$(dirname "${model_path}")
# get the name of the model without extension
model_name=$(basename "${model_path}")
model_name="${model_name%.*}"

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"

# 使用basename命令获取父目录名称
exp_name=$(basename "${parent_dir}")
# add model name to exp name
exp_name="${exp_name}_${model_name}"
echo "exp name: $exp_name, model path: $model_path"
echo "ctx_len: $ctx_len, n_embd: $n_embd, n_layer: $n_layer"
echo "eval dir: $eval_dir"

mkdir -p $eval_dir/eval/textvqa/answers/$exp_name
CHUNKS=${#GPULIST[@]}
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}
    python evaluate.py \
        --ctx_len $ctx_len --n_embd $n_embd --n_layer $n_layer \
        --model_path $model_path \
        --image_folder $eval_dir/images/textvqa/train_images \
        --question_file $eval_dir/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --output_file $eval_dir/eval/textvqa/answers/$exp_name/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
    echo "Started chunk $IDX"
done
wait

output_file=$eval_dir/eval/textvqa/answers/${exp_name}/merge.jsonl
> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $eval_dir/eval/textvqa/answers/${exp_name}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python eval/eval_textvqa.py \
    --annotation-file $eval_dir/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $eval_dir/eval/textvqa/answers/$exp_name/merge.jsonl