########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, os, re, copy
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from typing import Dict, List, Sequence, Any
from .utils import gpt4v_crop, largest_3n_plus_2_prime
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
STOP_TOKEN_INDEX = 261
DEFAULT_STOP_TOKEN = "\n\n"


def get_all_human_conversation(conversations: Sequence[Dict]) -> str:
    """
    Get all human conversation from a list of conversations.
    """
    human_conv_list = []
    for sentence in conversations:
        if sentence["from"].lower() == "human":
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                human_conv = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                human_conv = human_conv.replace("Answer the question using a single word or phrase.", '').strip()
                human_conv = re.sub(r"\n(\s*\n)+", '\n', human_conv)
                human_conv_list.append(human_conv)
            else:
                human_conv_list.append(sentence['value'].strip())
    return "\n".join(human_conv_list)


def process_image_tokens_in_conversations(
    conversations: Sequence[Dict],
    image_position: str = "first", # "first", "middle" or "last"
) -> Sequence[Dict]:
    """
    Process image tokens within conversations.
    image first, then text
    replace \n\n with \n
    """
    if image_position == 'middle':
        all_human_conv = get_all_human_conversation(conversations)
    for sentence in conversations:
        if DEFAULT_IMAGE_TOKEN in sentence['value']:
            sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            sentence['value'] = re.sub(r"\n(\s*\n)+", '\n', sentence['value'])
            if image_position == "first":
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
            elif image_position == "middle":
                sentence['value'] = all_human_conv + '\n' + DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
            elif image_position == "last":
                sentence['value'] = sentence['value'] + '\n' + DEFAULT_IMAGE_TOKEN
            else:
                raise ValueError(f"Unknown image_position: {image_position}, must be first, middle or last.")
            sentence['value'] = sentence['value'].strip()
        else:
            sentence['value'] = re.sub(r"\n(\s*\n)+", '\n', sentence['value'].strip())

    return conversations

def process_tokens_in_conversations(
    conversations: Sequence[Dict],
) -> Sequence[Dict]:
    """
    Process tokens within conversations.
    replace \n\n with \n
    """
    for sentence in conversations:
        sentence['value'] = sentence['value'].strip()
        sentence['value'] = re.sub(r"\n(\s*\n)+", '\n', sentence['value'])

    return conversations


def _add_speaker_and_signal(conversations):
    """Add speaker and start/end signal on each round."""
    for sentence in conversations:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = "User"
        elif from_str.lower() == "gpt":
            from_str = "Assistant"
        else:
            raise ValueError(f"Unknown speaker: {from_str}, must be human or gpt.")
        
        if sentence["value"]: # for training, add end signal
            sentence["value"] = (from_str + ": " + sentence["value"] + DEFAULT_STOP_TOKEN)
        else: # for inference, not add end signal and no whitespace after colon
            sentence["value"] = from_str + ":"
    return conversations


def tokenize_with_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX):
    prompt_chunks = [tokenizer.encode(chunk) for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    input_ids = []
    for chunk in prompt_chunks:
        input_ids.extend(chunk)
        input_ids.append(image_token_index)

    return input_ids[:-1] # remove last image token


def mask_targets(targets, tokenized_lens, speakers):
    '''
    1. mask human words with IGNORE_INDEX.
    2. mask assistant begin signal with IGNORE_INDEX. Assistant: -> [5585, 41693, 59] 3 tokens
    '''
    cur_idx = 0
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            targets[cur_idx:cur_idx + tokenized_len] = IGNORE_INDEX
        if speaker == "gpt":
            targets[cur_idx:cur_idx + 3] = IGNORE_INDEX
        cur_idx += tokenized_len


def pad_to_max_len(input_ids, targets, max_len, pad_token_id):
    # keep the first max_len tokens to make sure instruction complete
    input_ids = input_ids[:max_len]
    targets = targets[:max_len]
    padding_len = max_len - len(input_ids)
    if padding_len <= 0:
        return input_ids, targets
    # input_ids and targets are tensors
    input_ids = torch.cat([input_ids, torch.tensor([pad_token_id] * padding_len, dtype=torch.long)])
    targets = torch.cat([targets, torch.tensor([IGNORE_INDEX] * padding_len, dtype=torch.long)])
    return input_ids, targets


def preprocess(conversations, tokenizer, has_image, ctx_len, pad_token_id=0, do_pad_to_max_length=True):
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add \n\n after each round;
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    5. Pad to max length.
    """
    # add end signal and concatenate together
    conversations = _add_speaker_and_signal(conversations)
    input_text = "".join([sentence["value"] for sentence in conversations])
    input_ids, tokenized_lens, speakers = [], [], []
    for conversation in conversations:
        if has_image:
            conv_ids = tokenize_with_image_token(conversation["value"], tokenizer)
        else:
            conv_ids = tokenizer.encode(conversation["value"])
        input_ids.extend(conv_ids)
        tokenized_lens.append(len(conv_ids))
        speakers.append(conversation["from"])
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = copy.deepcopy(input_ids)
    mask_targets(targets, tokenized_lens, speakers)
    if do_pad_to_max_length:
        input_ids, targets = pad_to_max_len(input_ids, targets, ctx_len, pad_token_id)
    return dict(input_ids=input_ids, labels=targets, input_text=input_text)


class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.vocab_size = args.vocab_size
        self.tokenizer = args.tokenizer
        self.list_data_dict = json.load(open(args.data_file, "r"))
        # shuffle the data, but deterministically
        self.list_data_dict_reverse = [x for x in reversed(self.list_data_dict)]
        self.data_size = len(self.list_data_dict)
        self.magic_prime = largest_3n_plus_2_prime(self.data_size)
        self.samples_per_epoch = self.args.epoch_steps * self.args.real_bsz

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        step = epoch * self.samples_per_epoch + (idx * world_size) + rank
        # use a magic prime to sample the dataset deterministically yet randomly enough
        sample_idx = (step * step * step) % self.magic_prime
        # first epoch use the original data, then use the reversed data(avoid overfitting)
        # normally, we don't train for more than 2 epoch
        if step < self.magic_prime: # first epoch
            sample = self.list_data_dict[sample_idx]
        else: # when step >= self.magic_prime, means the second epoch
            sample = self.list_data_dict_reverse[sample_idx]

        if 'image' in sample:
            image_file = sample['image']
            image_folder = args.image_folder
            processor = args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if args.detail == 'high':
                image = [image] + gpt4v_crop(image)
                image = processor(images=image, return_tensors='pt')['pixel_values']
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values']
            conversations = process_image_tokens_in_conversations(copy.deepcopy(sample["conversations"]), 
                                                                  image_position=args.image_position)
        else:
            conversations = process_tokens_in_conversations(copy.deepcopy(sample["conversations"]))

        data_dict = preprocess(
            conversations,
            self.tokenizer,
            has_image=('image' in sample),
            ctx_len=args.ctx_len,
            pad_token_id=0)
        
        # image exist in the data
        if 'image' in sample:
            data_dict['images'] = image
        else:
            # image does not exist in the data, fill with zeros
            if args.detail == 'high':
                crop_size = args.image_processor.crop_size
                data_dict['images'] = torch.zeros(7, 3, crop_size['height'], crop_size['width'])
            else:
                crop_size = args.image_processor.crop_size
                data_dict['images'] = torch.zeros(1, 3, crop_size['height'], crop_size['width'])
        return data_dict