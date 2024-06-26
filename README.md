# VisualRWKV: Visual Language model based on RWKV
<p align="center">
  <img src="./rwkv_emoji.png" alt="Logo" width="200">
</p>

📖 [Paper](https://arxiv.org/abs/2406.13362) | 🤗 [Model](https://huggingface.co/howard-hou/visualrwkv-6) | 🐰 [Demo](https://huggingface.co/spaces/howard-hou/VisualRWKV-Gradio-1)

VisualRWKV is a visual language model based on RWKV language model, enabling RWKV to handle various visual tasks.

**VisualRWKV: Exploring Recurrent Neural Networks for Visual Language Models** [[Paper]](https://arxiv.org/abs/2406.13362)

**Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence** [[Paper]](https://arxiv.org/abs/2404.05892)

## News and Updates
* 2024.06.25 🔥 **VisualRWKV-6.0 checkpoints are released!** [[weights]](./MODEL_ZOO.md)
* 2024.05.11 🔥 **VisualRWKV-6.0 code is released!** [[code]](https://github.com/howard-hou/VisualRWKV/tree/main/VisualRWKV-v6/v6.0)
* 2024.03.25 🔥 **VisualRWKV-5.0 is released!**


## Architecture
<p align="center">  
  <img src="./VisualRWKV-arch.png" alt="Logo" width="800">
</p>

## Model Zoo
VisualRWKV weights, checkpoints, and related results are shown in the following markdown[[Model Zoo]](./MODEL_ZOO.md).


## Install
1. Clone this repository and navigate to VisualRWKV folder, VisualRWKV-v6.0 is the stable version.
```bash
git clone https://github.com/howard-hou/VisualRWKV.git
cd VisualRWKV-v6/v6.0
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate visualrwkv
pip install --upgrade pip  # enable PEP 660 support

#reference:
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pytorch-lightning==1.9.5 deepspeed==0.7.0 wandb ninja

best performance:
pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==1.9.5 deepspeed wandb ninja --upgrade
```


## Pre-training and Fine-tuning
**Lastest stable verion is VisualRWKV-v6/v6.0, please cd to the dir VisualRWKV-v6/v6.0 for running the code.**

VisualRWKV training consists of two stages: 
- (1) Pre-training stage: use pretrain dataset to train a projection layer from *frozen pretrained* vision encoder to the *frozen RWKV*; 
- (2) Fine-tuning stage: use visual instruction data, to teach the model to follow visual instructions.

### Pre-training
#### Download LLaVA-Pretrain dataset
You can download the [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).
#### Download RWKV checkpoints for Pre-training
If you want to pretrain by yourself. You can download the RWKV checkpoints from the following links in the table.
| VisualRWKV Version | RWKV 1B6 | RWKV 3B | RWKV 7B |
| --- | --- | --- | --- |
| VisualRWKV-v6 | [RWKV-x060-World-1B6](https://huggingface.co/BlinkDL/rwkv-6-world/blob/main/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth) |[RWKV-x060-World-3B](https://huggingface.co/BlinkDL/rwkv-6-world/blob/main/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth) | [RWKV-x060-World-7B](https://huggingface.co/BlinkDL/rwkv-6-world/blob/main/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth) |

#### Pre-training command
You can refer to the following command to pretrain the VisualRWKV-v6.0 model. Also see scripts in the `scripts/train` directory.
```bash
# here is an example to use 4 GPUs to pretrain a 1B5 RWKV model
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py --load_model /path/to/rwkv/checkpoint \
    --wandb "" --proj_dir path/to/output/ \
    --data_file /path/to/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 1024 --epoch_steps 1000 --epoch_count 9 --epoch_begin 0 --epoch_save 0 \
    --micro_bsz 16 --accumulate_grad_batches 2 --n_layer 24 --n_embd 2048 --pre_ffn 0 \
    --lr_init 1e-3 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 4 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 0 \
    --image_folder /path/to/LLaVA-Pretrain/images/ \
    --vision_tower_name /path/to/openai/clip-vit-large-patch14-336 \
    --freeze_rwkv 24 --detail low --grid_size -1 --image_position first \
    --enable_progress_bar True
```

### Visual Instruction Tuning
#### Prepare data
Please refer to the [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/README.md) project for visual instruction data.

#### Fine-tuning command
You can refer to the following command to fine-tune the VisualRWKV-v6.0 model. Also see scripts in the `scripts/train` directory.
```bash
# here is an example to use 8 GPUs to fine-tune a 1B5 RWKV model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train.py --model_path path/to/pretrained-visualrwkv \
    --wandb "" --proj_dir out/rwkv1b5-v060_mix665k \
    --data_file /path/to/LLaVA-Instruct-150K/shuffled_llava_v1_5_mix665k.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 2048 --epoch_steps 1000 --epoch_count 20 --epoch_begin 0 --epoch_save 5 \
    --micro_bsz 8 --accumulate_grad_batches 2 --n_layer 24 --n_embd 2048 --pre_ffn 0 \
    --lr_init 2e-5 --lr_final 2e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 8 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 0 \
    --image_folder /path/to/LLaVA-Instruct-150K/images/ \
    --vision_tower_name /path/to/openai/clip-vit-large-patch14-336 \
    --freeze_rwkv 0 --freeze_proj 0 --detail low --grid_size -1 --image_position middle \
    --enable_progress_bar True
```
