model_path=$1
ctx_len=$2
n_embd=$3
n_layer=$4
eval_dir=$5
vision_tower_dir=$6
num_token_per_image=$7
state_encoder_num_token_per_image=$8

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
echo "vision tower dir: $vision_tower_dir"
echo "num token per image: $num_token_per_image"
echo "state encoder num token per image: $state_encoder_num_token_per_image"

python evaluate.py \
    --ctx_len $ctx_len --n_embd $n_embd --n_layer $n_layer \
    --vision_tower_dir $vision_tower_dir \
    --num_token_per_image $num_token_per_image \
    --state_encoder_num_token_per_image $state_encoder_num_token_per_image \
    --model_path $model_path \
    --image_folder $eval_dir/eval/MME/MME_Benchmark_release_version \
    --question_file $eval_dir/eval/MME/llava_mme.jsonl \
    --output_file $eval_dir/eval/MME/answers/$exp_name.jsonl 

cd $eval_dir/eval/MME

python convert_answer_to_mme.py --experiment $exp_name

cd eval_tool

python calculation.py --results_dir answers/$exp_name