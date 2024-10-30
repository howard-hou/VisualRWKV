import os
os.environ["RWKV_JIT_ON"] = "0"

import json
from PIL import Image
import pandas as pd
import numpy as np
import math
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from src.rwkv_tokenizer import TRIE_TOKENIZER
from src.dataset import DEFAULT_IMAGE_TOKEN, DEFAULT_STOP_TOKEN, STOP_TOKEN_INDEX, IMAGE_TOKEN_INDEX
from src.config import VISION_TOWER_CHECKPOINT_NAMES
from pynvml import *
nvmlInit()
gpu_h = nvmlDeviceGetHandleByIndex(0)
    

def get_fake_input_image_dict(image_processor, num_images):
    image_dict = {}
    # step 1: generate all zero PIL images
    image_list = [Image.new("RGB", (224, 224), (0, 0, 0)) for _ in range(num_images)]
    # step 2: process the images
    pixel_values = defaultdict(list)
    for image in image_list:
        pixel_value = image_processor(image) # dict with keys 'dino' and 'siglip' and 'sam'
        for k in pixel_value:
            pixel_values[k].append(pixel_value[k])
    # merge by key
    for key in pixel_values:
        image_dict[key] = torch.stack(pixel_values[key], dim=0)

    return image_dict

def get_real_num_token_per_image(num_input_token):
    # choose closest num_token_per_image to num_input_token
    for num_token_per_image in [i**2 for i in range(32, 0, -1)]:
        if num_token_per_image <= num_input_token:
            return num_token_per_image


def eval_model(args):
    import time
    from src.model import VisualRWKV
    args.vision_tower_path = {name: Path(args.vision_tower_dir) / path for name, path in VISION_TOWER_CHECKPOINT_NAMES.items()}
    # Model
    model = VisualRWKV(args).bfloat16().to(args.device)
    tokenizer = TRIE_TOKENIZER("src/rwkv_vocab_v20230424.txt")
    image_processor = model.vit.get_image_transform()

    num_input_token_list = [1]
    for num_input_token in tqdm(num_input_token_list):
        # prepare fake image
        image_dict = get_fake_input_image_dict(image_processor, args.num_fake_images)
        for k in image_dict:
            image_dict[k] = image_dict[k].bfloat16().to(args.device)
        image_dict['num_image_per_sample'] = [len(image_dict[k])] # make sure one sample per step
        # prepare fake text input_ids
        real_num_token_per_image = get_real_num_token_per_image(num_input_token)
        args.num_token_per_image = real_num_token_per_image
        num_image_token = args.num_fake_images * real_num_token_per_image
        num_text_token = num_input_token - num_image_token
        input_ids = [IMAGE_TOKEN_INDEX] * num_image_token + [0] * num_text_token
        print(f"num_input_token={num_input_token}, real_num_token_per_image={real_num_token_per_image}, num_image_token={num_image_token}, num_text_token={num_text_token}")
        assert len(input_ids) == num_input_token, f"len(input_ids)={len(input_ids)}, num_input_token={num_input_token}"
        data_dict = {"input_ids": torch.tensor(input_ids)}
        input_ids = data_dict['input_ids'].unsqueeze(0).to(args.device)

        print("input_ids.shape: ", input_ids.shape)
        print("input_ids: ", input_ids)
        for k in image_dict:
            if isinstance(image_dict[k], torch.Tensor):
                print(f"image_dict[{k}].shape: {image_dict[k].shape}")
            else:
                print(f"{k}: {image_dict[k]}")

        # timeit for inference
        # use os.environ to store, change the model code accordingly
        os.environ["vision_encoder_latency"] = ""
        os.environ["state_encoder_latency"] = ""
        os.environ["decoder_latency"] = ""
        for _ in range(5):
            with torch.inference_mode():
                output_ids, output_logits, output_probs = model.generate(
                    input_ids,
                    images=image_dict,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    max_new_tokens=1,
                    stop_token_idx=STOP_TOKEN_INDEX)
        gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
        vision_encoder_latency = np.mean([float(l) for l in os.environ["vision_encoder_latency"].split() if l][1:]) * 1000 # s to ms
        state_encoder_latency = np.mean([float(l) for l in os.environ["state_encoder_latency"].split() if l][1:]) * 1000 # s to ms
        decoder_latency = np.mean([float(l) for l in os.environ["decoder_latency"].split() if l][1:]) * 1000 # s to ms
        # convert gpu used to GiB
        gpu_used = gpu_info.used / 1024**3
        print(f"num_input_token={num_input_token}, gpu_used={gpu_used:.4f} GiB)")
        print(f"vision_encoder_latency={vision_encoder_latency:.4f} ms, state_encoder_latency={state_encoder_latency:.4f} ms, decoder_latency={decoder_latency:.4f} ms")
        print(f"os.environ['vision_encoder_latency']={os.environ['vision_encoder_latency']}")
        print(f"os.environ['state_encoder_latency']={os.environ['state_encoder_latency']}")
        print(f"os.environ['decoder_latency']={os.environ['decoder_latency']}")
        from pytorch_lightning.utilities.model_summary import ModelSummary
        model_summary = ModelSummary(model)
        print(model_summary)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # arguments to init model
    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--vocab_size", default=65536, type=int)
    parser.add_argument("--ctx_len", default=256, type=int)
    parser.add_argument("--n_layer", default=24, type=int)
    parser.add_argument("--n_embd", default=2048, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_size_a", default=64, type=int)
    parser.add_argument("--head_size_divisor", default=8, type=int)
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--vision_tower_dir",type=str, help="Path to the directory containing the vision tower checkpoints")
    parser.add_argument("--grid_size", type=int, default=8) # -1 for no grid, 0 for cls token, 1 for global avg, 8 for 64 tokens
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--proj_type", default='linear', type=str, choices=['linear', 'mlp'])
    parser.add_argument("--num_token_per_image", type=int, default=64)
    parser.add_argument("--n_state_encoder_layer", default=6, type=int)
    parser.add_argument("--state_encoder_max_feature_len", default=0, type=int)
    parser.add_argument("--state_encoder_num_token_per_image", default=0, type=int)
    # arguments for evaluation
    parser.add_argument("--num_fake_images", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    #
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

    eval_model(args)