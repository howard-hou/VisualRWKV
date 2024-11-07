#!/bin/bash
model_path=$1
ctx_len=$2
proj_type=$3
n_embd=$4
n_layer=$5
dataset_path=$6
vision_tower_dir=$7
num_token_per_image=$8
n_state_encoder_layer=$9
# 使用dirname命令获取父目录的路径
parent_dir=$(dirname "${model_path}")
# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."
# get the name of the model without extension
model_name=$(basename "${model_path}")
model_name="${model_name%.*}"
# get the dataset name
dataset_name=$(basename "${dataset_path}")
# 打印当前工作目录
echo "Current working directory: $(pwd)"
output_file=$parent_dir/$dataset_name/$model_name.jsonl

# 使用basename命令获取父目录名称
exp_name=$(basename "${parent_dir}")
# add model name to exp name
exp_name="${exp_name}_${model_name}"
echo "exp name: $exp_name, model path: $model_path"
echo "ctx_len: $ctx_len, proj_type: $proj_type, n_embd: $n_embd, n_layer: $n_layer"
echo "num_token_per_image: $num_token_per_image", "n_state_encoder_layer: $n_state_encoder_layer"
echo "dataset_path: $dataset_path"
echo "vision_tower_dir: $vision_tower_dir"
echo "output file: $output_file"

python evaluate_hfds.py \
    --ctx_len $ctx_len --proj_type $proj_type --n_embd $n_embd --n_layer $n_layer \
    --vision_tower_dir $vision_tower_dir \
    --model_path $model_path \
    --dataset_path $dataset_path \
    --num_token_per_image $num_token_per_image \
    --n_state_encoder_layer $n_state_encoder_layer \
    --state_encoder_max_feature_len 4096


python eval/eval_interleave.py $output_file