########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, re, copy
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from typing import Dict, Optional, Sequence, Any
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


def process_image_tokens_in_conversations(
    conversations: Sequence[str],
) -> Dict:
    """
    Process image tokens within conversations.
    """
    for sentence in conversations:
        if DEFAULT_IMAGE_TOKEN in sentence['value']:
            sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            sentence['value'] = re.sub(r"\n(\s*\n)+", '\n', sentence['value'])
            sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
            sentence['value'] = sentence['value'].strip()
        else:
            sentence['value'] = re.sub(r"\n(\s*\n)+", '\n', sentence['value'].strip())

    return conversations


def _add_speaker_and_signal(conversations):
    """Add speaker and start/end signal on each round."""
    END_SIGNAL = "\n\n"
    for sentence in conversations:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = "User"
        elif from_str.lower() == "gpt":
            from_str = "Assistant"
        else:
            raise ValueError(f"Unknown speaker: {from_str}, must be human or gpt.")
        sentence["value"] = (from_str + ": " +
                             sentence["value"] + END_SIGNAL)
    return conversations

def tokenize_with_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX):
    prompt_chunks = [tokenizer.encode(chunk) for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    input_ids = []
    for chunk in prompt_chunks:
        input_ids.extend(chunk)
        input_ids.append(image_token_index)

    return input_ids[:-1] # remove last image token



def mask_targets_from_human(targets, tokenized_lens, speakers):
    cur_idx = 0
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            targets[cur_idx:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def preprocess(conversations, tokenizer, has_image):
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add \n\n after each round;
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = _add_speaker_and_signal(conversations)
    conversation = "".join([sentence["value"] for sentence in conversations])
    if has_image:
        input_ids = tokenize_with_image_token(conversation, tokenizer)
    else:
        input_ids = tokenizer.encode(conversation)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = copy.deepcopy(input_ids)
    if has_image:
        tokenized_lens = [len(tokenize_with_image_token(s["value"], tokenizer)) for s in conversations]
    else:
        tokenized_lens = [len(tokenizer.encode(s["value"])) for s in conversations]
    speakers = [sentence["from"] for sentence in conversations]
    mask_targets_from_human(targets, tokenized_lens, speakers)
    return dict(input_ids=input_ids, labels=targets)



class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.vocab_size = args.vocab_size
        list_data_dict = json.load(open(args.data_file, "r"))

        rank_zero_info("Formatting inputs...")
        self.tokenizer = args.tokenizer
        self.list_data_dict = list_data_dict


    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        args = self.args
        sample = self.list_data_dict[idx]
        if 'image' in sample:
            image_file = self.list_data_dict[idx]['image']
            image_folder = args.image_folder
            processor = args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            conversations = process_image_tokens_in_conversations(copy.deepcopy(sample["conversations"]))
        else:
            conversations = copy.deepcopy(sample["conversations"])

        data_dict = preprocess(
            conversations,
            self.tokenizer,
            has_image=('image' in sample))
        
        # image exist in the data
        if 'image' in sample:
            data_dict['images'] = image
        else:
            # image does not exist in the data, fill with zeros
            crop_size = args.image_processor.crop_size
            data_dict['images'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict
    

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data.json")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--image_folder", type=str, default="images")
    parser.add_argument("--image_processor", type=str, default="images")
    parser.add_argument("--tokenizer", type=str, default="images")
    parser.add_argument("--epoch_steps", type=int, default=100)
    parser.add_argument("--micro_bsz", type=int, default=1)
    args = parser.parse_args()
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    from transformers import CLIPImageProcessor
    args.tokenizer = TRIE_TOKENIZER("rwkv_vocab_v20230424.txt")
    args.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dataset = MyDataset(args)
    print(dataset[0])
