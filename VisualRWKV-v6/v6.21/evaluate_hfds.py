import os
os.environ["RWKV_JIT_ON"] = "1"

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
from datasets import load_from_disk
from src.rwkv_tokenizer import TRIE_TOKENIZER
from src.dataset import DEFAULT_IMAGE_TOKEN, DEFAULT_STOP_TOKEN, STOP_TOKEN_INDEX
from src.dataset import process_image_tokens_in_conversations, preprocess
from src.utils import Conversation
from src.config import VISION_TOWER_CHECKPOINT_NAMES


    
def get_input_image_dict(image_list, image_processor):
    image_dict = {}
    pixel_values = defaultdict(list)
    for image in image_list:
        image = image.convert("RGB") # make sure it's RGB
        pixel_value = image_processor(image) # dict with keys 'dino' and 'siglip' and 'sam'
        for k in pixel_value:
            pixel_values[k].append(pixel_value[k])
    # merge by key
    for key in pixel_values:
        image_dict[key] = torch.stack(pixel_values[key], dim=0)
    return image_dict


def prepare_conversations(line):
    if "question" in line:
        input_text = line['question']

        conv = Conversation(id=line['sample_id'], roles=["human", "gpt"], conversations=[])
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], "")

        conversations = conv.conversations
    elif "conversations" in line:
        conv = Conversation(id=line['sample_id'], roles=["human", "gpt"], conversations=[])
        for msg in line["conversations"]:
            if msg['from'] == "human":
                conv.append_message(conv.roles[0], msg['value'])
            elif msg['from'] == "gpt":
                conv.append_message(conv.roles[1], msg['value'])
            else:
                raise ValueError(f"Unknown role {msg['from']}")
        # if last message is from human, add an empty message from gpt
        if msg['from'] == "human":
            conv.append_message(conv.roles[1], "")
        conversations = conv.conversations
    else:
        raise ValueError("Invalid input line, no 'question' or 'conversations' field")
    return conversations


def eval_model(args):
    from src.model import VisualRWKV
    model_path = Path(args.model_path)
    exp_name = model_path.parent.name
    model_name = model_path.stem
    args.vision_tower_path = {name: Path(args.vision_tower_dir) / path for name, path in VISION_TOWER_CHECKPOINT_NAMES.items()}
    # Model
    model = VisualRWKV(args)
    msg = model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    print("msg of loading model: ", msg)
    model = model.bfloat16().to(args.device)
    tokenizer = TRIE_TOKENIZER("src/rwkv_vocab_v20230424.txt")
    image_processor = model.vit.get_image_transform()

    dataset = load_from_disk(args.dataset_path)
    if isinstance(dataset, dict):
        dataset = dataset[args.split]
    # output to the same dir of the model
    dataset_path = Path(args.dataset_path)
    output_dir = model_path.parent / dataset_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}.jsonl"

    out_file = open(output_file, "w")
    pbar = tqdm(total=len(dataset))
    update_every = len(dataset) // 100
    for i, line in enumerate(dataset):
        idx = line['sample_id']
        image_keys = [k for k in line.keys() if 'image' in k]
        image_list = [line[k] for k in image_keys if line[k] is not None]
        image_dict = get_input_image_dict(image_list, image_processor)
        for k in image_dict:
            image_dict[k] = image_dict[k].bfloat16().to(args.device)
        image_dict['num_image_per_sample'] = [len(image_dict[k])] # make sure one sample per step

        conversations = prepare_conversations(line)

        data_dict = preprocess(
            conversations,
            tokenizer,
            has_image=True,
            ctx_len=args.ctx_len,
            num_token_per_image=args.num_token_per_image,
            pad_token_id=0,
            do_pad_to_max_length=False)
        
        input_ids = data_dict['input_ids'].unsqueeze(0).to(args.device)
        cur_prompt = data_dict['input_text']
        if i == 0:
            print("input_ids.shape: ", input_ids.shape)
            print("input_ids: ", input_ids)
            print("cur_prompt: ", cur_prompt)
            print("num_input_images: ", len(image_list))
            for k in image_dict:
                print(f"image_dict[{k}].shape: {image_dict[k].shape}")

        with torch.inference_mode():
            output_ids, output_logits, output_probs = model.generate(
                input_ids,
                images=image_dict,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=args.max_new_tokens,
                stop_token_idx=STOP_TOKEN_INDEX)

        output = tokenizer.decode(output_ids).split(DEFAULT_STOP_TOKEN)[0].strip()
        # avg logit
        avg_logit = sum(output_logits) / len(output_logits)
        # geometric mean of probs
        avg_prob = np.prod(output_probs) ** (1.0 / len(output_probs))

        out_str = json.dumps({"question_id": idx,
                              "prompt": cur_prompt,
                              "text": output,
                              "avg_logit": str(round(avg_logit, 3)),
                              "avg_prob": str(round(avg_prob, 3)),
                              "model_id": model_name,
                              "metadata": {
                                  "sub_task": line.get("sub_task", None),
                                  "question_type": line.get("question_type", None),
                                  "answer": line.get("answer", None),
                              }}, ensure_ascii=False)
        out_file.write(out_str + "\n")
        # update progress bar
        if i % update_every == 0 and i != 0:
            pbar.update(update_every)
        out_file.flush()
    out_file.close()
    pbar.close()

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
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--proj_type", default='linear', type=str, choices=['linear', 'mlp'])
    parser.add_argument("--num_token_per_image", type=int, default=16)
    # arguments for evaluation
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default='test')
    args = parser.parse_args()
    #
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

    eval_model(args)