model_path=/mnt/data2/Mutil_data/VisualRWKV/VisualRWKV/VisualRWKV-v6/v6.11/out/rwkv1b5-v061_mix665k/rwkv-83.pth
ctx_len=2048
grid_size=-1
n_embd=2048
n_layer=24
cd "$(dirname "$(dirname "$0")")/.."
# 使用dirname命令获取父目录的路径
parent_dir=$(dirname "${model_path}")

# 使用basename命令获取父目录名称
exp_name=$(basename "${parent_dir}")
echo "exp name: $exp_name, model path: $model_path"
echo "ctx_len: $ctx_len, grid_size: $grid_size, n_embd: $n_embd, n_layer: $n_layer"

python /mnt/data2/Mutil_data/VisualRWKV/VisualRWKV/VisualRWKV-v6/v6.11/evaluate.py \
    --ctx_len 2048 --grid_size -1 --n_embd 2048 --n_layer 24 \
    --model_path /mnt/data2/Mutil_data/VisualRWKV/VisualRWKV/VisualRWKV-v6/v6.11/out/rwkv1b5-v061_mix665k/rwkv-83.pth \
    --dataset_name seedbench \
    --image_folder /mnt/data2/Mutil_data/eval_dataset/seed_bench/SEED-Bench\
    --question_file /mnt/data2/Mutil_data/eval_dataset/seed_bench/SEED-Bench/rwkv-seed-bench.jsonl \
    --output_file /mnt/data2/Mutil_data/eval_dataset/se+bench/SEED-Bench/rwkv-seed-bench-answer.jsonl


python /mnt/data2/Mutil_data/VisualRWKV/VisualRWKV/VisualRWKV-v6/v6.11/eval/convert_seed_for_submission.py \
    --annotation-file /mnt/data2/Mutil_data/eval_dataset/seed_bench/SEED-Bench/SEED-Bench.json \
    --result-file  /mnt/data2/Mutil_data/eval_dataset/seed_bench/SEED-Bench/rwkv-seed-bench-answer.jsonl\
    --result-upload-file /mnt/data2/Mutil_data/eval_dataset/seed_bench/SEED-Bench/rwkv-v6.11-1.6B.jsonl