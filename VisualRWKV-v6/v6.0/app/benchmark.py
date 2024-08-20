import os
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # if '1' then use CUDA kernel for seq mode (much faster)
# make sure cuda dir is in the same level as modeling_rwkv.py
from modeling_rwkv import RWKV

import gc
import base64
from io import BytesIO
import torch
import torch.nn.functional as F
import time
from PIL import Image
import json
from transformers import CLIPImageProcessor
import argparse
from pathlib import Path
from pynvml import *
nvmlInit()
gpu_h = nvmlDeviceGetHandleByIndex(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('vision_tower_name', type=str, default='openai/clip-vit-large-patch14-336')
    parser.add_argument('--strategy', type=str, default='cuda fp16')
    return parser.parse_args()


def convert_ckpt(model_path):
    # convert visualrwkv to RWKV and vision encoder #######################
    output_dir = Path(model_path).parent
    state_dict = torch.load(model_path, map_location="cpu")
    rwkv_state_dict = {}
    visual_state_dict = {}
    for key in state_dict:
        if key.startswith("rwkv"):
            rwkv_state_dict[key[5:]] = state_dict[key].half()
        else:
            visual_state_dict[key] = state_dict[key].half()
    # save 
    vision_local_path = output_dir / f"visual.pth"
    rwkv_local_path = output_dir / f"rwkv.pth"
    torch.save(rwkv_state_dict, rwkv_local_path)
    torch.save(visual_state_dict, vision_local_path)
    print("rwkv state dict has keys: ", len(rwkv_state_dict), "saved to ", rwkv_local_path)
    print("visual state dict has keys: ", len(visual_state_dict), "saved to ", vision_local_path)
    return rwkv_local_path, vision_local_path


def generate_prompt(instruction):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    return f"\n{instruction}\n\nAssistant:"


def generate(
    ctx,
    image_state,
    token_count=200,
):
    ctx = ctx.strip()
    all_tokens = []
    non_first_token_time = []
    for i in range(int(token_count)):
        if i == 0:
            input_ids = pipeline.encode(ctx)[-ctx_limit:]
            start = time.time()
            out, state = model.forward(tokens=input_ids, state=image_state)
            prefill_text_time = time.time() - start
            print(f"first token time: {prefill_text_time:.3f} seconds")
        else:
            input_ids = [token]
            start = time.time()
            out, state = model.forward(tokens=input_ids, state=state)
            non_first_token_time.append(time.time() - start)
        token = out.argmax()
        all_tokens += [token]
    return prefill_text_time, non_first_token_time


##########################################################################
from rwkv.utils import PIPELINE
ctx_limit = 3500
args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() and "cuda" in args.strategy else "cpu")

rwkv_path, vision_path = convert_ckpt(args.model_path)

model = RWKV(model=str(rwkv_path), strategy=args.strategy)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

##########################################################################
from modeling_vision import VisionEncoder, VisionEncoderConfig
config = VisionEncoderConfig(n_embd=model.args.n_embd, 
                             vision_tower_name=args.vision_tower_name, 
                             grid_size=-1)
visual_encoder = VisionEncoder(config)
vision_state_dict = torch.load(vision_path, map_location='cpu')
visual_encoder.load_state_dict(vision_state_dict, strict=False)
image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower_name)
visual_encoder = visual_encoder.to(device)
##########################################################################
cur_dir = os.path.dirname(os.path.abspath(__file__))
examples = [
    [
        f"{cur_dir}/examples_pizza.jpg",
        "What are steps to cook it?"
    ],
    [
        f"{cur_dir}/examples_bluejay.jpg",
        "what is the name of this bird?",
    ],
    [
        f"{cur_dir}/examples_extreme_ironing.jpg",
        "What is unusual about this image?",
    ],
    [
        f"{cur_dir}/examples_waterview.jpg",
        "What are the things I should be cautious about when I visit here?",
    ],
]


ln0_weight = model.w['blocks.0.ln0.weight'].to(torch.float32).to(device)
ln0_bias = model.w['blocks.0.ln0.bias'].to(torch.float32).to(device)
def compute_image_state(image):
    image = image_processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values']
    image = image.to(device)
    image_features = visual_encoder.encode_images(image.unsqueeze(0)).squeeze(0) # [L, D]
    # apply layer norm to image feature, very important
    image_features = F.layer_norm(image_features, 
                                    (image_features.shape[-1],), 
                                    weight=ln0_weight, 
                                    bias=ln0_bias)
    _, image_state = model.forward(embs=image_features, state=None)
    return image_state

def warmup(image, question):
    print("warm up...")
    image_state = compute_image_state(image)
    input_text = generate_prompt(question)
    generate(input_text, image_state, token_count=10)
    print("warm up done.")
    return image_state


def benchmark(image, question, new_tokens=1):
    print(f"benchmarking... new tokens: {new_tokens}")
    warmup(image, question)
    start = time.time()
    image_state = compute_image_state(image)
    prefill_image_time = time.time() - start
    print(f"image state computation time: {prefill_image_time:.3f} seconds")
    input_text = generate_prompt(question)
    # measure latency
    prefill_text_time_list, non_first_token_time_list = [], []
    for _ in range(5):
        prefill_text_time, non_first_token_time = generate(input_text, 
                                                           image_state, 
                                                           token_count=new_tokens)
        non_first_token_time_list.extend(non_first_token_time)
        prefill_text_time_list.append(prefill_text_time)
    prefill_text_time_avg = sum(prefill_text_time_list[1:]) / len(prefill_text_time_list[1:])
    latency_avg = sum(non_first_token_time_list[1:]) / len(non_first_token_time_list[1:])
    gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    # convert gpu used to GiB
    gpu_used = gpu_info.used / 1024**3
    print(f'new tokens: {new_tokens} - latency: {latency_avg:.3f} seconds - speed: {1 / latency_avg:.3f} tokens/s - GPU used: {gpu_used:.3f} GiB')
    return dict(new_tokens=new_tokens, latency=latency_avg, vram_used=gpu_used, 
                prefill_image_time=prefill_image_time, prefill_text_time=prefill_text_time_avg, 
                speed=1 / latency_avg)

new_tokens_list = [32, 64]

# write to jsonl: benchmark_{strategy}_{model_name}.jsonl
model_name = args.model_path.split('/')[-1].replace('.pth', '')
strategy = args.strategy.replace(' ', '-')
w = open(f'benchmark_{model_name}_{strategy}.jsonl', 'w')
for new_tokens in new_tokens_list:
    image_path, question = examples[0]
    image = Image.open(image_path)
    bench_dict = benchmark(image, question, new_tokens=new_tokens)
    w.write(json.dumps(bench_dict) + '\n')
    gc.collect()
    torch.cuda.empty_cache()

