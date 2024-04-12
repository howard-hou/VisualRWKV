import os
os.environ["RWKV_JIT_ON"] = "1"

import json
from PIL import Image
import pandas as pd
import math
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from src.rwkv_tokenizer import TRIE_TOKENIZER
from src.dataset import DEFAULT_IMAGE_TOKEN, DEFAULT_STOP_TOKEN, STOP_TOKEN_INDEX
from src.dataset import process_image_tokens_in_conversations, preprocess
from src.utils import Conversation, gpt4v_crop, load_image_from_base64
from transformers import CLIPImageProcessor


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False


def load_questions(file_path):
    file_path = Path(file_path)
    suffix = file_path.suffix
    if suffix == ".jsonl":
        questions = [json.loads(q) for q in open(file_path)]
    elif suffix == ".json":
        questions = json.load(open(file_path))
    elif suffix == ".tsv":
        questions = pd.read_table(file_path).to_dict("records")
    else:
        raise ValueError("Unsupported file type: {}".format(suffix))
    return questions


def get_question_id(line):
    if "question_id" in line:
        return line["question_id"]
    elif "id" in line:
        return line["id"]
    elif "index" in line:
        return line["index"]
    else:
        raise ValueError("Cannot find question id in line: {}".format(line))


def get_options(line, options):
    parsed_options = []
    for option in options:
        option_value = line[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def get_input_text_mmbench(line, lang='en'):
    all_options = ['A', 'B', 'C', 'D']
    options = get_options(line, all_options)
    question = line['question']
    hint = line['hint']
    if not is_none(hint):
        question = hint + '\n' + question
    for option_char, option in zip(all_options[:len(options)], options):
        question = question + '\n' + option_char + '. ' + option
    question = DEFAULT_IMAGE_TOKEN + '\n' + question
    if lang == 'cn':
        question = question + '\n' + "请直接回答选项字母。"
    else:
        question = question + '\n' + "Answer with the option's letter from the given choices directly."
    return question
    

def get_input_text(line, dataset_name):
    if dataset_name == "mmbench":
        return get_input_text_mmbench(line)
    elif dataset_name == "mmbench_cn":
        return get_input_text_mmbench(line, lang='cn')
    else:
        if "text" in line:
            return DEFAULT_IMAGE_TOKEN + '\n' + line["text"]
        elif "conversations" in line:
            return line["conversations"][0]["value"]
        else:
            raise ValueError("Cannot find input text in line: {}".format(line))
    
def get_input_image_tensor(line, image_folder, image_processor, detail):
    if "image" in line:
        image_file = line["image"]
        if image_folder is not None:
            image = Image.open(image_folder / image_file)
        else: # image is base64 encoded
            image = load_image_from_base64(image_file)
        if args.detail == 'high':
            image = [image] + gpt4v_crop(image)
            image_tensor = image_processor(images=image, return_tensors='pt')['pixel_values']
        else:
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    else:
        # image does not exist in the data, fill with zeros
        if detail == 'high':
            crop_size = image_processor.crop_size
            image_tensor = torch.zeros(7, 3, crop_size['height'], crop_size['width'])
        else:
            crop_size = image_processor.crop_size
            image_tensor = torch.zeros(1, 3, crop_size['height'], crop_size['width'])
    return image_tensor

def eval_model(args):
    from src.model import VisualRWKV
    model_path = Path(args.model_path)
    model_name = model_path.parent.name
    # Model
    # adjust arguments for tiny attention, from int to list
    args.tiny_att_layer = [i for i in range(args.n_layer)][-args.tiny_att_layer:] if args.tiny_att_layer > 0 else []
    print("tiny_att_layer: ", args.tiny_att_layer)
    model = VisualRWKV(args)
    msg = model.load_state_dict(torch.load(model_path), strict=False)
    print("msg of loading model: ", msg)
    model = model.bfloat16().to(args.device)
    tokenizer = TRIE_TOKENIZER("src/rwkv_vocab_v20230424.txt")
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower_name)

    questions = load_questions(args.question_file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    image_folder = Path(args.image_folder) if args.image_folder is not None else None

    out_file = open(output_file, "w")
    pbar = tqdm(total=len(questions))
    update_every = len(questions) // 100
    for i, line in enumerate(questions):
        idx = get_question_id(line)
        input_text = get_input_text(line, dataset_name=args.dataset_name)

        conv = Conversation(id=idx, roles=["human", "gpt"], conversations=[])
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], "")

        conversations = process_image_tokens_in_conversations(conv.conversations, image_position=args.image_position)

        image_tensor = get_input_image_tensor(line, image_folder, image_processor, args.detail)
        image_tensor = image_tensor.unsqueeze(0).bfloat16().to(args.device)

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
                max_new_tokens=args.max_new_tokens,
                stop_token_idx=STOP_TOKEN_INDEX)

        output = tokenizer.decode(output_ids).split(DEFAULT_STOP_TOKEN)[0].strip()

        out_str = json.dumps({"question_id": idx,
                              "prompt": cur_prompt,
                              "text": output,
                              "model_id": model_name,
                              "metadata": {
                                  "image_file": line.get("image", None),
                              }}, ensure_ascii=False)
        out_file.write(out_str + "\n")
        # update progress bar
        if i % update_every == 0:
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
    parser.add_argument("--vision_tower_clip", default="", type=str)  # empty means not use 
    parser.add_argument("--vision_tower_sam", default="", type=str)  # empty means not use 
    parser.add_argument("--vision_tower_dino", default="", type=str)  # empty means not use
    parser.add_argument("--grid_size", type=int, default=8) # -1 for no grid, 0 for cls token, 1 for global avg, 8 for 64 tokens
    parser.add_argument("--detail", type=str, default="low")
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--tiny_att_dim", default=-1, type=int)  # tiny attention dim
    parser.add_argument("--tiny_att_layer", default=-1, type=int)  # tiny attention @ last num layers
    parser.add_argument("--clip_unfreeze_layers", default=0, type=int) 
    parser.add_argument("--sam_unfreeze_layers", default=0, type=int) 
    parser.add_argument("--dino_unfreeze_layers", default=0, type=int) 
    parser.add_argument("--unified_vision_dim", default=1024, type=int)
    # arguments for evaluation
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--question_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset_name", type=str, default="default")
    parser.add_argument("--image_position", default='first', type=str)  # 'first' or 'last' or ''middle
    args = parser.parse_args()
    #
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

    eval_model(args)