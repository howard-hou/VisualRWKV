model_path=$1
ctx_len=$2
grid_size=$3
n_embd=$4
n_layer=$5
eval_dir=$6
vision_tower_path=$7
image_position=$8
image_scanning=$9
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
echo "vision tower path: $vision_tower_path", 
echo "image position: $image_position, image scanning: $image_scanning"

python evaluate.py \
    --ctx_len $ctx_len --grid_size $grid_size --n_embd $n_embd --n_layer $n_layer \
    --vision_tower_name $vision_tower_path \
    --model_path $model_path \
    --image_folder $eval_dir/eval/scienceqa/images/test \
    --question_file $eval_dir/eval/scienceqa/llava_test_CQM-A.json \
    --output_file $eval_dir/eval/scienceqa/answers/$exp_name.jsonl \
    --image_position $image_position --image_scanning $image_scanning

python eval/eval_science_qa.py \
    --base-dir $eval_dir/eval/scienceqa \
    --result-file $eval_dir/eval/scienceqa/answers/$exp_name.jsonl \
    --output-file $eval_dir/eval/scienceqa/answers/$exp_name_output.jsonl \
    --output-result $eval_dir/eval/scienceqa/answers/$exp_name_result.json