#!/bin/bash
# 获取所有可用的GPU
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
CHUNKS=${#GPULIST[@]}

# 使用basename命令获取父目录名称
exp_name=$(basename "${parent_dir}")
# add model name to exp name
exp_name="${exp_name}_${model_name}"
echo "exp name: $exp_name, model path: $model_path"
echo "ctx_len: $ctx_len, n_embd: $n_embd, n_layer: $n_layer"
echo "eval dir: $eval_dir"
echo "num of chunks: $CHUNKS"

mkdir -p $eval_dir/eval/scienceqa/answers/$exp_name

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}
    python evaluate.py \
        --ctx_len $ctx_len --n_embd $n_embd --n_layer $n_layer \
        --model_path $model_path \
        --image_folder $eval_dir/eval/scienceqa/images/test \
        --question_file $eval_dir/eval/scienceqa/llava_test_CQM-A.json \
        --output_file $eval_dir/eval/scienceqa/answers/$exp_name/${CHUNKS}_${IDX}.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
    echo "Started chunk $IDX"
done
wait

# 合并结果文件
output_file=$eval_dir/eval/scienceqa/answers/${exp_name}/merge.jsonl
> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $eval_dir/eval/scienceqa/answers/${exp_name}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python eval/eval_science_qa.py \
    --base-dir $eval_dir/eval/scienceqa \
    --result-file $eval_dir/eval/scienceqa/answers/$exp_name/merge.jsonl \
    --output-file $eval_dir/eval/scienceqa/answers/$exp_name/output.json \
    --output-result $eval_dir/eval/scienceqa/answers/$exp_name/result.json