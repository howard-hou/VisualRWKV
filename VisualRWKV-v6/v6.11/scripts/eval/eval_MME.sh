model_path=/mnt/data2/Mutil_data/VisualRWKV/VisualRWKV/VisualRWKV-v6/v6.11/out/rwkv1b5-v061_mix665k/rwkv-83.pth
ctx_len=2048
grid_size=-1
n_embd=2048
n_layer=24
eval_dir=/mnt/data2/Mutil_data/eval_dataset
image_position=middle
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
echo "ctx_len: $ctx_len, grid_size: $grid_size, n_embd: $n_embd, n_layer: $n_layer"
echo "eval dir: $eval_dir"
echo "image position: $image_position"

python evaluate.py \
    --ctx_len $ctx_len --grid_size $grid_size --n_embd $n_embd --n_layer $n_layer \
    --model_path $model_path \
    --image_folder $eval_dir/eval/MME/MME_Benchmark_release_version \
    --question_file $eval_dir/eval/MME/llava_mme.jsonl \
    --output_file $eval_dir/eval/MME/answers/$exp_name.jsonl \
    --image_position $image_position

cd $eval_dir/eval/MME

python convert_answer_to_mme.py --experiment $exp_name

cd eval_tool

python calculation.py  --results_dir answers/$exp_name