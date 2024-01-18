import gradio as gr
import os, gc
import base64
from io import BytesIO
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from huggingface_hub import hf_hub_download

ctx_limit = 3500
title = 'ViusualRWKV-v5'
rwkv_remote_path = "rwkv1b5-vitl336p14-577token_mix665k_rwkv.pth"
vision_remote_path = "rwkv1b5-vitl336p14-577token_mix665k_visual.pth"
vision_tower_name = 'openai/clip-vit-large-patch14-336'

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # if '1' then use CUDA kernel for seq mode (much faster)

from modeling_vision import VisionEncoder, VisionEncoderConfig
from modeling_rwkv import RWKV
model_path = hf_hub_download(repo_id="howard-hou/visualrwkv-5", filename=rwkv_remote_path)
model = RWKV(model=model_path, strategy='cpu fp32')
from rwkv.utils import PIPELINE, PIPELINE_ARGS
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

##########################################################################
config = VisionEncoderConfig(n_embd=model.args.n_embd, 
                             vision_tower_name=vision_tower_name, 
                             grid_size=-1)
visual_encoder = VisionEncoder(config)
vision_local_path = hf_hub_download(repo_id="howard-hou/visualrwkv-5", filename=vision_remote_path)
vision_state_dict = torch.load(vision_local_path, map_location='cpu')
visual_encoder.load_state_dict(vision_state_dict, strict=False)
image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
##########################################################################
def generate_prompt(instruction):
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
            out, state = model.forward(tokens=input_ids, state=image_state)
        else:
            input_ids = [token]
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

    del out
    del state
    gc.collect()
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

image_cache = {}
def compute_image_state(image):
    base64_image = pil_image_to_base64(image)
    if base64_image in image_cache:
        image_state = image_cache[base64_image]
    else:
        image = image_processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values']
        image_features = visual_encoder.encode_images(image.unsqueeze(0)).squeeze(0) # [L, D]
        # apply layer norm to image feature, very important
        image_features = F.layer_norm(image_features, 
                                    (image_features.shape[-1],), 
                                    weight=model.w['blocks.0.ln0.weight'], 
                                    bias=model.w['blocks.0.ln0.bias'])
        _, image_state = model.forward(embs=image_features, state=None)
        image_cache[base64_image] = image_state
    return image_state

def chatbot(image, question):
    if image is None:
        yield "Please upload an image."
        return
    image_state = compute_image_state(image)
    input_text = generate_prompt(question)
    for output in generate(input_text, image_state):
        yield output

with gr.Blocks(title=title) as demo:
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
    data = gr.Dataset(components=[image, prompt], samples=examples, label="Examples", headers=["Image", "Prompt"])
    submit.click(chatbot, [image, prompt], [output])
    clear.click(lambda: None, [], [output])
    data.click(lambda x: x, [data], [image, prompt])

demo.queue(max_size=10)
demo.launch(share=False)