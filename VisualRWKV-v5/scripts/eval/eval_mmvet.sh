python eval.py \
    --vision_tower_name /houhaowen/huggingface_models/openai/clip-vit-base-patch32 \
    --model_path out/rwkv1b5_mix665k/rwkv-0.pth \
    --image_folder /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/mm-vet/images \
    --question_file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/mm-vet/llava-mm-vet.jsonl \
    --output_file /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/mm-vet/outputs/rwkv1b5-mix665k.jsonl

mkdir -p /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/mm-vet/outputs/rwkv1b5-mix665k.jsonl \
    --dst /houhaowen/huggingface_datasets/liuhaotian/LLaVA-Instruct-150K/eval/mm-vet/results/rwkv1b5-mix665k.json