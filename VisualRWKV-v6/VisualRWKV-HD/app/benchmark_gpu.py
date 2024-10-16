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
from pynvml import *
nvmlInit()
gpu_h = nvmlDeviceGetHandleByIndex(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rwkv_path', type=str)
    parser.add_argument('--vision_path', type=str)
    return parser.parse_args()

ctx_limit = 3500
args = parse_args()
vision_tower_dir = 'openai/clip-vit-large-patch14-336'

model = RWKV(model=args.rwkv_path, strategy='cuda fp16')
from rwkv.utils import PIPELINE, PIPELINE_ARGS
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

##########################################################################
from modeling_vision import VisionEncoder, VisionEncoderConfig
config = VisionEncoderConfig(n_embd=model.args.n_embd, 
                             vision_tower_dir=vision_tower_dir, 
                             grid_size=-1)
visual_encoder = VisionEncoder(config)
vision_state_dict = torch.load(args.vision_path, map_location='cpu')
visual_encoder.load_state_dict(vision_state_dict, strict=False)
image_processor = CLIPImageProcessor.from_pretrained(vision_tower_dir)
visual_encoder = visual_encoder.to(device)
##########################################################################
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
    for i in range(int(token_count)):
        if i == 0:
            input_ids = pipeline.encode(ctx)[-ctx_limit:]
            out, state = model.forward(tokens=input_ids, state=image_state)
        else:
            input_ids = [token]
            out, state = model.forward(tokens=input_ids, state=state)
        token = out.argmax()
        all_tokens += [token]
    return all_tokens


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


def pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format as needed (JPEG, PNG, etc.)
    # Encodes the image data into base64 format as a bytes object
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return base64_image


ln0_weight = model.w['blocks.0.ln0.weight'].to(torch.float32).to(device)
ln0_bias = model.w['blocks.0.ln0.bias'].to(torch.float32).to(device)
def compute_image_state(image, image_cache=None):
    base64_image = pil_image_to_base64(image)
    if image_cache is not None and base64_image in image_cache:
        image_state = image_cache[base64_image]
    else:
        image = image_processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values']
        image = image.to(device)
        image_features = visual_encoder.encode_images(image.unsqueeze(0)).squeeze(0) # [L, D]
        # apply layer norm to image feature, very important
        image_features = F.layer_norm(image_features, 
                                     (image_features.shape[-1],), 
                                      weight=ln0_weight, 
                                      bias=ln0_bias)
        _, image_state = model.forward(embs=image_features, state=None)
        if image_cache is not None:
            image_cache[base64_image] = image_state
    return image_state


def benchmark(image, question, new_tokens=1):
    image_state = compute_image_state(image)
    input_text = generate_prompt(question)
    # measure latency
    latency_list = []
    for _ in range(15):
        start = time.time()
        all_tokens = generate(input_text, image_state, token_count=new_tokens)
        latency = time.time() - start
        latency_list.append(latency)
    latency_avg = sum(latency_list[5:]) / len(latency_list[5:])
    gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    # convert gpu used to GiB
    gpu_used = gpu_info.used / 1024**3
    print(f'new tokens: {new_tokens} - latency: {latency_avg:.3f} seconds - gpu used {gpu_used}')
    return dict(new_tokens=new_tokens, latency=latency_avg, vram_used=gpu_used)

new_tokens_list = [2**i for i in range(16)]
print(new_tokens_list)

w = open('benchmark_gpu.jsonl', 'w')
for new_tokens in new_tokens_list:
    image_path, question = examples[0]
    image = Image.open(image_path)
    bench_dict = benchmark(image, question, new_tokens=new_tokens)
    w.write(json.dumps(bench_dict) + '\n')
    gc.collect()
    torch.cuda.empty_cache()

