ckpt_path=$1
question_file=$2
num_token_per_image=$3
n_state_encoder_layer=$4
state_encoder_max_feature_len=$5
state_encoder_num_token_per_image=$6
# get name of the question file
question_file_name=$(basename $question_file)
# replace extension with json
output_file_name=${question_file_name%.*}.json
# get parent directory of the model path
ckpt_dir=$(dirname $ckpt_path)

# echo
echo "ckpt_path: $ckpt_path"
echo "question_file: $question_file"
echo "question_file_name: $question_file_name"
echo "output_file_name: $output_file_name"
echo "num_token_per_image: $num_token_per_image"
echo "n_state_encoder_layer: $n_state_encoder_layer"
echo "state_encoder_max_feature_len: $state_encoder_max_feature_len"
echo "state_encoder_num_token_per_image: $state_encoder_num_token_per_image"


python evaluate.py \
    --ctx_len 2048 --proj_type mlp --n_embd 2048 --n_layer 24 \
    --vision_tower_dir /root/autodl-tmp/huggingface_models/ \
    --model_path $ckpt_path \
    --image_folder /root/autodl-tmp/huggingface_datasets/Video-MME/output/ \
    --question_file $question_file \
    --output_file $ckpt_dir/Video-MME/$question_file_name \
    --num_token_per_image $num_token_per_image \
    --n_state_encoder_layer $n_state_encoder_layer \
    --state_encoder_max_feature_len $state_encoder_max_feature_len \
    --state_encoder_num_token_per_image $state_encoder_num_token_per_image

python eval/convert_videomme_for_eval.py --src $ckpt_dir/Video-MME/$question_file_name \
        --dst $ckpt_dir/Video-MME/$output_file_name

python eval/eval_your_results.py \
    --results_file $ckpt_dir/Video-MME/$output_file_name \
    --video_duration_type short,medium,long \
    --return_categories_accuracy \
    --return_sub_categories_accuracy \
    --return_task_types_accuracy