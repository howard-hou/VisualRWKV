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
    # 3. unfreeze the readout in blocks
    for block in model.rwkv.blocks:
        block.att.mem_read.requires_grad_(True)
        block.att.mem_gate.requires_grad_(True)
        block.att.time_mem_w1.requires_grad_(True)
        block.att.time_mem_w2.requires_grad_(True)
        block.att.time_mem_r.requires_grad_(True)
        block.att.time_mem_g.requires_grad_(True)
    # 4. unfreeze the projection layer
    model.proj.requires_grad_(True)


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


# --------------------------------------------------------
# 3D sine-cosine position embedding
# References:
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# --------------------------------------------------------
def get_3d_sincos_pos_embed(embed_dim, grid_size_x, grid_size_y, grid_size_z):
    """
    grid_size_x, grid_size_y, grid_size_z: ints of the grid dimensions in three axes
    return:
    pos_embed: [grid_size_x*grid_size_y*grid_size_z, embed_dim]
    """
    grid_x = np.arange(grid_size_x, dtype=np.float32)
    grid_y = np.arange(grid_size_y, dtype=np.float32)
    grid_z = np.arange(grid_size_z, dtype=np.float32)
    grid_xx, grid_yy, grid_zz = np.meshgrid(grid_x, grid_y, grid_z)
    grid = np.stack([grid_xx, grid_yy, grid_zz], axis=0)

    grid = grid.reshape([3, 1, grid_size_x, grid_size_y, grid_size_z])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    # embed_dim should be divisible by 6
    embed_dim_z = embed_dim // 6 * 2
    embed_dim_y = embed_dim // 6 * 2
    embed_dim_x = embed_dim - embed_dim_z - embed_dim_y

    # use a third of dimensions to encode each axis
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim_x, grid[0])  # (X*Y*Z, D/3)
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim_y, grid[1])  # (X*Y*Z, D/3)
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim_z, grid[2])  # (X*Y*Z, D/3)

    emb = np.concatenate([emb_x, emb_y, emb_z], axis=1)  # (X*Y*Z, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
def fold_tensor_by_layer(packed_image_features, n_layer):
    """
    按照指定层数对张量进行折叠操作

    Args:
        packed_image_features (torch.Tensor): 形状为 [B, L, D] 的张量
        n_layer (int): 用于折叠的层数

    Returns:
        torch.Tensor: 折叠后的张量，形状为 [B * n_layer, L // n_layer, D]（当L能被n_layer整除时）
        或 [B * n_layer, (L + n_layer - L % n_layer) // n_layer, D]（当L不能被n_layer整除时）
    """
    batch_size, length, dimension = packed_image_features.shape

    # 检查是否需要补齐
    if length % n_layer!= 0:
        padding_length = n_layer - length % n_layer
        padded_tensor = torch.zeros((batch_size, length + padding_length, dimension),
                                    device=packed_image_features.device,
                                    dtype=packed_image_features.dtype)
        padded_tensor[:, -length:, :] = packed_image_features
        length += padding_length

        folded_tensor = torch.reshape(padded_tensor, (batch_size * n_layer, length // n_layer, dimension))
    else:
        folded_tensor = torch.reshape(packed_image_features, (batch_size * n_layer, length // n_layer, dimension))

    return folded_tensor