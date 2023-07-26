from typing import List, Tuple
from pathlib import Path

import torch
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode
from omegaconf import DictConfig, OmegaConf
from visualrwkv.model import VisualRWKV
from visualrwkv.components.adapter import AdapterPretrain
from visualrwkv.components.rwkv_rnn import RWKV
from visualrwkv.model_config import AdapterConfig, RWKVConfig, VisualRWKVConfig

_MODELS = {
    "VisualRWKV-small": "",
}
MODEL_NAME2IMAGE_SIZE = {
    "VisualRWKV-small": 224,
    "VisualRWKV-world-7b": 224,
}


def _get_image_transform(image_size):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return transform_test


def _load_adapter(adapter, state_dict):
    # Create a new dictionary with updated keys
    key_prefix = "adapter."
    new_state_dict = {}
    for key, value in state_dict.items():
        if key_prefix in key:
            new_key = key.replace(key_prefix, "")
            new_state_dict[new_key] = value
    load_msg = adapter.load_state_dict(new_state_dict)
    print("load adapter: ", load_msg)
    return adapter


def _load_projection(projection, state_dict):
    key_prefix = "llm.llm_proj."
    new_state_dict = {}
    for key, value in state_dict.items():
        if key_prefix in key:
            new_key = key.replace(key_prefix, "")
            new_state_dict[new_key] = value
    load_msg = projection.load_state_dict(new_state_dict)
    print("load projection: ", load_msg)
    return projection


def get_model_config(model_name: str, rwkv_path, device) -> DictConfig:
    model_config = OmegaConf.create()
    model_config.adapter = AdapterConfig()
    if device == "cuda":
        rwkv_strategy = "cuda fp16"
    elif device == "cpu":
        rwkv_strategy = "cpu fp32"
    else:
        raise ValueError(f"device {device} not supported")
    model_config.rwkv = RWKVConfig(rwkv_path, rwkv_strategy)
    model_config.visualrwkv = VisualRWKVConfig(model_name=model_name)
    return model_config


def build_model(state_dict: dict, model_config):
    # first step: build adapter
    adapter_cfg = model_config.adapter
    adapter = AdapterPretrain(
        vit_model=adapter_cfg.vit_model,
        t5_model=adapter_cfg.t5_model,
        num_task_embeddings=adapter_cfg.num_task_embeddings,
        feature_size=adapter_cfg.feature_size,
        is_freeze_vit=adapter_cfg.is_freeze_vit,
        max_length=adapter_cfg.max_length,
    )
    adapter = _load_adapter(adapter, state_dict)
    # second step: build rwkv
    rwkv_cfg = model_config["rwkv"]
    rwkv = RWKV(rwkv_cfg.model_path, rwkv_cfg.strategy)
    # third step: build projection
    adapter_hidden_size = adapter.hidden_size
    rwkv_hidden_size = rwkv.w["emb.weight"].shape[1]
    projection = torch.nn.Linear(adapter_hidden_size, rwkv_hidden_size)
    projection = _load_projection(projection, state_dict)
    # fourth step: build visualrwkv
    visualrwkv_cfg = model_config["visualrwkv"]
    visualrwkv = VisualRWKV(
        adapter=adapter,
        rwkv_model=rwkv,
        llm_proj=projection,
        model_name=visualrwkv_cfg.model_name,
        max_length=visualrwkv_cfg.max_length,
        rnn_strategy=rwkv_cfg.strategy,
    )
    return visualrwkv


def print_model_info(model):
    print("after model loaded: ")
    # print model dtype and device
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}, {param.device}")


def load(model_name, adapter_path, rwkv_path, device, use_bf16=False) -> None:
    state_dict = torch.load(adapter_path, map_location="cpu")["state_dict"]
    model_config = get_model_config(model_name, rwkv_path, device)
    model = build_model(state_dict, model_config)
    image_size = MODEL_NAME2IMAGE_SIZE[model_name]
    image_transfrom = _get_image_transform(image_size)
    if use_bf16:
        model.adapter.vit = model.adapter.vit.bfloat16()
    model = model.to(device)
    # print_model_info(model)
    model.eval()
    return model, image_transfrom
