import os
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # if '1' then use CUDA kernel for seq mode (much faster)
# make sure cuda dir is in the same level as modeling_rwkv.py
from modeling_rwkv import RWKV

import gc
import gradio as gr
import base64
from io import BytesIO
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
########################## text rwkv ################################################################
from rwkv.utils import PIPELINE, PIPELINE_ARGS
title = "RWKV-5-World-1B5-v2-20231025-ctx4096"
model_path = hf_hub_download(repo_id="BlinkDL/rwkv-5-world", filename=f"{title}.pth")
model = RWKV(model=model_path, strategy='cuda fp16')
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

def generate_prompt(instruction, input=""):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    input = input.strip().replace('\r\n','\n').replace('\n\n','\n')
    if input:
        return f"""Instruction: {instruction}
Input: {input}
Response:"""
    else:
        return f"""User: hi
Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.
User: {instruction}
Assistant:"""

def evaluate(
    ctx,
    token_count=200,
    temperature=1.0,
    top_p=0.7,
    presencePenalty = 0.1,
    countPenalty = 0.1,
):
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0]) # stop generation whenever you see any token here
    ctx = ctx.strip()
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = None
    for i in range(int(token_count)):
        input_ids = pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token]
        out, state = model.forward(tokens=input_ids, state=state)
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
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
    print(f'{timestamp} - vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')
    del out
    del state
    gc.collect()
    torch.cuda.empty_cache()
    yield out_str.strip()

examples = [
    ["Assistant: Sure! Here is a very detailed plan to create flying pigs:", 333, 1, 0.3, 0, 1],
    ["Assistant: Sure! Here are some ideas for FTL drive:", 333, 1, 0.3, 0, 1],
    ["A few light taps upon the pane made her turn to the window. It had begun to snow again.", 333, 1, 0.3, 0, 1],
    [generate_prompt("Écrivez un programme Python pour miner 1 Bitcoin, avec des commentaires."), 333, 1, 0.3, 0, 1],
    [generate_prompt("東京で訪れるべき素晴らしい場所とその紹介をいくつか挙げてください。"), 333, 1, 0.3, 0, 1],
    [generate_prompt("Write a story using the following information.", "A man named Alex chops a tree down."), 333, 1, 0.3, 0, 1],
    ["Assistant: Here is a very detailed plan to kill all mosquitoes:", 333, 1, 0.3, 0, 1],
    ['''Edward: I am Edward Elric from fullmetal alchemist. I am in the world of full metal alchemist and know nothing of the real world.
Player: Hello Edward. What have you been up to recently?
Edward:''', 333, 1, 0.3, 0, 1],
    [generate_prompt("写一篇关于水利工程的流体力学模型的论文，需要详细全面。"), 333, 1, 0.3, 0, 1],
    ['''“当然可以，大宇宙不会因为这五公斤就不坍缩了。”关一帆说，他还有一个没说出来的想法：也许大宇宙真的会因为相差一个原子的质量而由封闭转为开放。大自然的精巧有时超出想象，比如生命的诞生，就需要各项宇宙参数在几亿亿分之一精度上的精确配合。但程心仍然可以留下她的生态球，因为在那无数文明创造的无数小宇宙中，肯定有相当一部分不响应回归运动的号召，所以，大宇宙最终被夺走的质量至少有几亿吨，甚至可能是几亿亿亿吨。
但愿大宇宙能够忽略这个误差。
程心和关一帆进入了飞船，智子最后也进来了。她早就不再穿那身华丽的和服了，她现在身着迷彩服，再次成为一名轻捷精悍的战士，她的身上佩带着许多武器和生存装备，最引人注目的是那把插在背后的武士刀。
“放心，我在，你们就在！”智子对两位人类朋友说。
聚变发动机启动了，推进器发出幽幽的蓝光，飞船缓缓地穿过了宇宙之门。
小宇宙中只剩下漂流瓶和生态球。漂流瓶隐没于黑暗里，在一千米见方的宇宙中，只有生态球里的小太阳发出一点光芒。在这个小小的生命世界中，几只清澈的水球在零重力环境中静静地飘浮着，有一条小鱼从一只水球中蹦出，跃入另一只水球，轻盈地穿游于绿藻之间。在一小块陆地上的草丛中，有一滴露珠从一片草叶上脱离，旋转着飘起，向太空中折射出一缕晶莹的阳光。''', 333, 1, 0.3, 0, 1],    
]

########################## visual rwkv ################################################################
visual_title = 'ViusualRWKV-v5'
rwkv_remote_path = "rwkv1b5-vitl336p14-577token_mix665k_rwkv.pth"
vision_remote_path = "rwkv1b5-vitl336p14-577token_mix665k_visual.pth"
vision_tower_name = 'openai/clip-vit-large-patch14-336'

model_path = hf_hub_download(repo_id="howard-hou/visualrwkv-5", filename=rwkv_remote_path)
visual_rwkv = RWKV(model=model_path, strategy='cuda fp16')

##########################################################################
from modeling_vision import VisionEncoder, VisionEncoderConfig
config = VisionEncoderConfig(n_embd=visual_rwkv.args.n_embd, 
                             vision_tower_name=vision_tower_name, 
                             grid_size=-1)
visual_encoder = VisionEncoder(config)
vision_local_path = hf_hub_download(repo_id="howard-hou/visualrwkv-5", filename=vision_remote_path)
vision_state_dict = torch.load(vision_local_path, map_location='cpu')
visual_encoder.load_state_dict(vision_state_dict, strict=False)
image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
visual_encoder = visual_encoder.to(device)
##########################################################################
def visual_generate_prompt(instruction):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    return f"\n{instruction}\n\nAssistant:"

def generate(
    ctx,
    image_state,
    token_count=200,
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
            out, state = visual_rwkv.forward(tokens=input_ids, state=image_state)
        else:
            input_ids = [token]
            out, state = visual_rwkv.forward(tokens=input_ids, state=state)
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
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
    print(f'{timestamp} - vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')
    del out
    del state
    gc.collect()
    torch.cuda.empty_cache()
    yield out_str.strip()


##########################################################################
cur_dir = os.path.dirname(os.path.abspath(__file__))
visual_examples = [
    [
        f"{cur_dir}/examples_pizza.jpg",
        "What are steps to cook it?"
    ],
    [
        f"{cur_dir}/examples_bluejay.jpg",
        "what is the name of this bird?",
    ],
    [
        f"{cur_dir}/examples_woman_and_dog.png",
        "describe this image",
    ],
]


def pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format as needed (JPEG, PNG, etc.)
    # Encodes the image data into base64 format as a bytes object
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return base64_image

image_cache = {}
ln0_weight = visual_rwkv.w['blocks.0.ln0.weight'].to(torch.float32).to(device)
ln0_bias = visual_rwkv.w['blocks.0.ln0.bias'].to(torch.float32).to(device)
def compute_image_state(image):
    base64_image = pil_image_to_base64(image)
    if base64_image in image_cache:
        image_state = image_cache[base64_image]
    else:
        image = image_processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values'].to(device)
        image_features = visual_encoder.encode_images(image.unsqueeze(0)).squeeze(0) # [L, D]
        # apply layer norm to image feature, very important
        image_features = F.layer_norm(image_features, 
                                    (image_features.shape[-1],), 
                                    weight=ln0_weight, 
                                    bias=ln0_bias)
        _, image_state = visual_rwkv.forward(embs=image_features, state=None)
        image_cache[base64_image] = image_state
    return image_state

def chatbot(image, question):
    if image is None:
        yield "Please upload an image."
        return
    image_state = compute_image_state(image)
    input_text = visual_generate_prompt(question)
    for output in generate(input_text, image_state, temperature=1.0):
        yield output


##################################################################################################################
with gr.Blocks(title=title) as demo:
    gr.HTML(f"<div style=\"text-align: center;\">\n<h1>RWKV-5 World v2 - {title}</h1>\n</div>")
    with gr.Tab("Raw Generation"):
        gr.Markdown(f"This is [RWKV-5 World v2](https://huggingface.co/BlinkDL/rwkv-5-world) with 1.5B params - a 100% attention-free RNN [RWKV-LM](https://github.com/BlinkDL/RWKV-LM). Supports all 100+ world languages and code. And we have [200+ Github RWKV projects](https://github.com/search?o=desc&p=1&q=rwkv&s=updated&type=Repositories). *** Please try examples first (bottom of page) *** (edit them to use your question). Demo limited to ctxlen {ctx_limit}.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=2, label="Prompt", value="Assistant: Sure! Here is a very detailed plan to create flying pigs:")
                token_count = gr.Slider(10, 333, label="Max Tokens", step=10, value=333)
                temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
                presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=0)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=1)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                output = gr.Textbox(label="Output", lines=5)
        data = gr.Dataset(components=[prompt, token_count, temperature, top_p, presence_penalty, count_penalty], samples=examples, label="Example Instructions", headers=["Prompt", "Max Tokens", "Temperature", "Top P", "Presence Penalty", "Count Penalty"])
        submit.click(evaluate, [prompt, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
        clear.click(lambda: None, [], [output])
        data.click(lambda x: x, [data], [prompt, token_count, temperature, top_p, presence_penalty, count_penalty])
    with gr.Tab("Visual RWKV"):
        with gr.Row():
            with gr.Column():
                image = gr.Image(type='pil', label="Image")
            with gr.Column():
                prompt = gr.Textbox(lines=8, label="Prompt", 
                    value="Render a clear and concise summary of the photo.")
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary") 
            with gr.Column():
                output = gr.Textbox(label="Output", lines=10)
        data = gr.Dataset(components=[image, prompt], samples=visual_examples, label="Examples", headers=["Image", "Prompt"])
        submit.click(chatbot, [image, prompt], [output])
        clear.click(lambda: None, [], [output])
        data.click(lambda x: x, [data], [image, prompt])

demo.queue(concurrency_count=1, max_size=10)
demo.launch(share=False)