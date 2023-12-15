import os
os.environ["RWKV_JIT_ON"] = "1"

import json
from PIL import Image
import math
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from src.rwkv_tokenizer import TRIE_TOKENIZER
from src.dataset import DEFAULT_IMAGE_TOKEN, process_image_tokens_in_conversations, preprocess
from src.utils import Conversation, gpt4v_crop
from transformers import CLIPImageProcessor


def eval_model(args):
    from src.model import VisualRWKV
    model_path = Path(args.model_path)
    model_name = model_path.parent.name
    # Model
    model = VisualRWKV(args)
    msg = model.load_state_dict(torch.load(model_path))
    print("msg of loading model: ", msg)
    model = model.bfloat16().to(args.device)
    tokenizer = TRIE_TOKENIZER("src/rwkv_vocab_v20230424.txt")
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower_name)

    questions = [json.loads(q) for q in open(args.question_file)]
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    image_folder = Path(args.image_folder)

    out_file = open(output_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = DEFAULT_IMAGE_TOKEN + '\n' + line["text"]

        conv = Conversation(id=idx, roles=["human", "gpt"], conversations=[])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], "")

        conversations = process_image_tokens_in_conversations(conv.conversations)

        image = Image.open(image_folder / image_file)
        if args.detail == 'high':
            image = [image] + gpt4v_crop(image)
            image_tensor = image_processor(images=image, return_tensors='pt')['pixel_values']
            image_tensor = image_tensor.unsqueeze(0)
        else:
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.bfloat16().to(args.device)

        data_dict = preprocess(
            conversations,
            tokenizer,
            has_image=True,
            ctx_len=args.ctx_len,
            pad_token_id=0,
            do_pad_to_max_length=False)
        
        input_ids = data_dict['input_ids'].unsqueeze(0).to(args.device)
        cur_prompt = data_dict['input_text']

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=128)

        output = tokenizer.decode(output_ids).split('\n\n')[0]

        out_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": output,
                                   "model_id": model_name,
                                   "metadata": {
                                       "image_file": image_file,
                                   }}, ensure_ascii=False) + "\n")
        out_file.flush()
    out_file.close()

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
    parser.add_argument("--vision_tower_name", default="openai/clip-vit-base-patch32", type=str)  # openai/clip-vit-base-patch32
    parser.add_argument("--grid_size", type=int, default=8) # 0 for no grid, 1 for 1 token(global avg), 8 for 64 tokens
    parser.add_argument("--detail", type=str, default="high")
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    # arguments for evaluation
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--question_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    #
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

    eval_model(args)