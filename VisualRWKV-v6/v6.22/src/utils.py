import json, time, random, os
import numpy as np
import dataclasses
import torch
from torch.nn import functional as F
from pytorch_lightning.utilities import rank_zero_info
from typing import List, Dict
from PIL import Image
from io import BytesIO
import base64
from collections import defaultdict
# possible image configurations: (1:2), (2:1), (1:1), (1:3), (3:1)
POSSIBLE_RESOLUTIONS = [(448, 896), (896, 448), (896, 896), (448, 1344), (1344, 448)]

time_slot = {}
time_ref = time.time_ns()

def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def enable_state_encoder_pretrain_mode(model):
    # 1. freeze the RWKV model
    model.requires_grad_(False)
    # 2. unfreeze the state encoder
    model.state_encoder.requires_grad_(True)
    # 3. unfreeze the cross layer in blocks
    for block in model.rwkv.blocks:
        block.cross.requires_grad_(True)
    # 4. unfreeze the projection layer
    model.proj.requires_grad_(True)


def load_visualrwkv_from_checkpoint(model, model_path):
    '''
    reuse some of the visual-rwkv blocks to initialize the cross block
    '''
    rank_zero_info(f"loading visual rwkv model from {model_path}")
    ckpt_state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    # use pos_embed from pretrained model
    if "vit.dino_featurizer.pos_embed" in ckpt_state_dict:
        del ckpt_state_dict["vit.dino_featurizer.pos_embed"]
    if "vit.siglip_featurizer.pos_embed" in ckpt_state_dict:
        del ckpt_state_dict["vit.siglip_featurizer.pos_embed"]

    msg = model.load_state_dict(ckpt_state_dict, strict=False)
    msg = {
        'missing_keys': compress_parameter_names(msg.missing_keys), 
        'unexpected_keys': compress_parameter_names(msg.unexpected_keys)
    }
    rank_zero_info(f"msg from loading visual rwkv model: {msg}")
    return model


def load_image_state_encoder_from_checkpoint(model, path):
    '''
    reuse some of the visual-rwkv blocks to initialize the image state encoder
    '''
    rank_zero_info(f"loading image state encoder from {path}")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    n_state_encoder_layer = len(model.state_encoder.blocks)
    new_state_dict = {}
    for k in state_dict:
        if "rwkv.blocks" in k:
            layer_id = int(k.split(".")[2])
            if layer_id < n_state_encoder_layer:
                # change the key name
                new_k = k.replace("rwkv.blocks", "blocks")
                new_state_dict[new_k] = state_dict[k]
    msg = model.state_encoder.load_state_dict(new_state_dict, strict=False)
    msg = {
            'missing_keys': compress_parameter_names(msg.missing_keys), 
            'unexpected_keys': compress_parameter_names(msg.unexpected_keys)
        }
    rank_zero_info(f"msg from loading image state encoder: {msg}")
    return model


def load_rwkv_from_pretrained(model, path):
    model.rwkv.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=False)
    rank_zero_info(f"Loaded pretrained RWKV from {path}")
    return model


def largest_3n_plus_2_prime(x):
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                return False
        return True
    
    # Integer division to obtain an integer n such that 3n+2 < x
    n = x // 3  
    while True:
        num = 3 * n + 2
        if num < x and is_prime(num):
            return num
        n -= 1


def split_image_into_tiles(image, n, m):
    '''
    Split the image into n x m tiles.
    '''
    width, height = image.size
    tile_width = width // m
    tile_height = height // n
    tiles = []
    # left top, right top, left bottom, right bottom
    for i in range(n):
        for j in range(m):
            x = j * tile_width
            y = i * tile_height
            tiles.append(image.crop((x, y, x + tile_width, y + tile_height)))
    return tiles


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = downscaled_width * downscaled_height
        wasted_resolution = abs((width * height) - effective_resolution)

        if wasted_resolution < min_wasted_resolution:
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def single_image_to_multi_image_strategy(image, best_resolution):
    """
    Splits a single image into n x m tiles and returns a list of images.

    Args:
        image (PIL.Image): The image to be split.
        n (int): The number of rows.
        m (int): The number of columns.

    Returns:
        list: A list of PIL.Image objects.
    """
    if best_resolution == (896, 896):
        image_area = image.size[0] * image.size[1]
        if image_area <= (896 * 896):
            return [image]
        else: # large image needs to be split
            n, m = 2, 2
    if best_resolution == (448, 896):
        n, m = 2, 1
    if best_resolution == (896, 448):
        n, m = 1, 2
    if best_resolution == (448, 1344):
        n, m = 3, 1
    if best_resolution == (1344, 448):
        n, m = 1, 3
    tiles = split_image_into_tiles(image, n, m)
    return [image] + tiles


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    id: str
    roles: List[str]
    conversations: List[Dict[str, str]]

    def append_message(self, role, message):
        d = {"from": role, "value": message}
        self.conversations.append(d)


def compress_parameter_names(parameter_names):
    compressed = defaultdict(set)
    for weight in parameter_names:
        parts = weight.split('.')
        # find the block number which is a number
        split_index = None
        for i, part in enumerate(parts):
            if part.isdigit():
                block = part
                split_index = i
                break
        if split_index is not None:
            block = parts[split_index]  # 提取block号
            rest = '.'.join(parts[split_index+1:])  # 剩余部分
            prefix = '.'.join(parts[:split_index]) # 
            compressed[(prefix, rest)].add(block)
        else:
            compressed[(weight, '')].add('')

    # 格式化输出，合并具有相同rest部分的block号
    output = []
    for (prefix, rest), blocks in compressed.items():
        if rest and blocks:
            blocks = sorted([int(b) for b in blocks])
            block_range = '{' + ','.join(map(str, blocks)) + '}'
            output.append(f'{prefix}.{block_range}.{rest}')
        else:
            output.append(prefix)
    return output