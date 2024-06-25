import os
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # if '1' then use CUDA kernel for seq mode (much faster)
# make sure cuda dir is in the same level as modeling_rwkv.py
from modeling_rwkv import RWKV

import gc
import gradio as gr
import base64
from io import BytesIO
from pathlib import Path
import torch
import torch.nn.functional as F
from datetime import datetime
from transformers import CLIPImageProcessor
from huggingface_hub import hf_hub_download
from pynvml import *
nvmlInit()
gpu_h = nvmlDeviceGetHandleByIndex(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ctx_limit = 3500
title = 'ViusualRWKV-v6.0'
visualrwkv_remote_path = "VisualRWKV-v060-1B6-v1.0-20240612.pth"
model_path = hf_hub_download(repo_id="howard-hou/visualrwkv-6", filename=visualrwkv_remote_path)
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
##########################################################################
vision_tower_name = 'openai/clip-vit-large-patch14-336'

model = RWKV(model=str(rwkv_local_path), strategy='cuda fp16')
from rwkv.utils import PIPELINE, PIPELINE_ARGS
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

##########################################################################
from modeling_vision import VisionEncoder, VisionEncoderConfig
config = VisionEncoderConfig(n_embd=model.args.n_embd, 
                             vision_tower_name=vision_tower_name, 
                             grid_size=-1)
visual_encoder = VisionEncoder(config)
vision_state_dict = torch.load(vision_local_path, map_location='cpu')
visual_encoder.load_state_dict(vision_state_dict, strict=False)
image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
visual_encoder = visual_encoder.to(device)
##########################################################################
def generate_prompt(instruction):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    return f"\n{instruction}\n\nAssistant:"

def generate(
    ctx,
    image_state,
    token_count=512,
    temperature=0.2,
    top_p=0.3,
    presencePenalty = 0.0,
    countPenalty = 1.0,
):
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                    alpha_frequency = countPenalty,
                    alpha_presence = presencePenalty,
                    token_ban = [], # ban the generation of some tokens
                    token_stop = [0, 261]) # stop generation whenever you see any token here
    ctx = ctx.strip()
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    for i in range(int(token_count)):
        if i == 0:
            input_ids = pipeline.encode(ctx)[-ctx_limit:]
            out, state = model.forward(tokens=input_ids, state=image_state)
        else:
            input_ids = [token]
            out, state = model.forward(tokens=input_ids, state=state)
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        if '\n\n' in out_str:
            break
        all_tokens += [token]
        for xxx in occurrence:
            occurrence[xxx] *= 0.996        
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        
        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            out_str += tmp
            yield out_str.strip()
            out_last = i + 1

    gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('sampled tokens:', all_tokens)
    print(f'{timestamp} - vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')
    del out
    del state
    gc.collect()
    torch.cuda.empty_cache()
    yield out_str.strip()


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
def compute_image_state(image, prefix_tokens):
    image = image_processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values']
    image = image.to(device)
    image_features = visual_encoder.encode_images(image.unsqueeze(0)).squeeze(0) # [L, D]
    # apply layer norm to image feature, very important
    image_features = F.layer_norm(image_features, 
                                 (image_features.shape[-1],), 
                                 weight=ln0_weight, 
                                 bias=ln0_bias)
    _, image_state = model.forward(tokens=prefix_tokens, embs=image_features, state=None)
    return image_state

def chatbot(image, question):
    if image is None:
        yield "Please upload an image."
        return
    input_text = generate_prompt(question)
    prefix_tokens = pipeline.encode(input_text)[-ctx_limit:]
    image_state = compute_image_state(image, prefix_tokens)
    for output in generate(input_text, image_state):
        yield output

with gr.Blocks(title=title) as demo:
    with gr.Row():
        with gr.Column():
            image = gr.Image(type='pil', label="Image")
        with gr.Column():
            prompt = gr.Textbox(lines=10, label="Prompt", 
                value="Render a clear and concise summary of the photo.")
            with gr.Row():
                submit = gr.Button("Submit", variant="primary")
                clear = gr.Button("Clear", variant="secondary") 
        with gr.Column():
            output = gr.Textbox(label="Output", lines=20)
    data = gr.Dataset(components=[image, prompt], samples=examples, label="Examples", headers=["Image", "Prompt"])
    submit.click(chatbot, [image, prompt], [output])
    clear.click(lambda: None, [], [output])
    data.click(lambda x: x, [data], [image, prompt])

demo.queue(max_size=10)
demo.launch(share=False)