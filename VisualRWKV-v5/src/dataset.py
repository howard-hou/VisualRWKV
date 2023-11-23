########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys, copy
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
    Process image tokens within conversations by moving the image token to the end of the sentence.
    which is better for the rkwv to learn.
    """
    for sentence in conversations:
        if DEFAULT_IMAGE_TOKEN in sentence['value']:
            sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            sentence['value'] = sentence['value'] + '\n' + DEFAULT_IMAGE_TOKEN
            sentence['value'] = sentence['value'].strip()

    return conversations


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = "USER"
        elif from_str.lower() == "gpt":
            from_str = "ASSISTANT"
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess(conversations, tokenizer, has_image):
    # add end signal and concatenate together
    header = f"\n\n"
    conversation = _add_speaker_and_signal(header, conversations)
    print(conversation)



class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.vocab_size = args.vocab_size
        list_data_dict = json.load(open(args.data_file, "r"))

        rank_zero_info("Formatting inputs...")
        self.tokenizer = args.tokenizer
        self.list_data_dict = list_data_dict


    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        sample = self.list_data_dict[idx]
        if 'image' in sample:
            image_file = self.list_data_dict[idx]['image']
            image_folder = args.image_folder
            # processor = args.image_processor
            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sample = process_image_tokens_in_conversations(copy.deepcopy(sample["conversations"]))
        else:
            sample = copy.deepcopy(sample["conversations"])

        data_dict = preprocess(
            sample,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[idx]))


        return None
    

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
    dataset = MyDataset(args)
    print(dataset[0])
