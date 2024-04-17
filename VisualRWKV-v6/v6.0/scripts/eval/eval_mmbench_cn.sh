model_path=$1
ctx_len=$2
grid_size=$3
n_embd=$4
n_layer=$5
eval_dir=$6
vision_tower_path=$7
# 使用dirname命令获取父目录的路径
parent_dir=$(dirname "${model_path}")
SPLIT="mmbench_dev_cn_20231003"
# 使用basename命令获取父目录名称
exp_name=$(basename "${parent_dir}")
echo "exp name: $exp_name, model path: $model_path"
echo "ctx_len: $ctx_len, grid_size: $grid_size, n_embd: $n_embd, n_layer: $n_layer"
echo "eval dir: $eval_dir"
echo "vision tower path: $vision_tower_path"
echo "split name: $SPLIT"

python evaluate.py \
    --ctx_len $ctx_len --grid_size $grid_size --n_embd $n_embd --n_layer $n_layer \
    --vision_tower_name $vision_tower_path \
    --model_path $model_path \
    --dataset_name mmbench \
    --question_file $eval_dir/eval/mmbench/$SPLIT.tsv \
    --output_file $eval_dir/eval/mmbench/answers/$SPLIT/$exp_name.jsonl

python eval/convert_mmbench_for_submission.py \
    --annotation-file $eval_dir/eval/mmbench/$SPLIT.tsv \
    --result-dir $eval_dir/eval/mmbench/answers/$SPLIT \
    --upload-dir $eval_dir/eval/mmbench/answers_upload/$SPLIT \
    --experiment $exp_name