"""
Vision backbone that returns concatenated features from SAM, DINOv2 and SigLIP
"""
import os
from pathlib import Path
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict,  Optional, Protocol, Tuple, Union

import timm
import torch
import torch.nn as nn
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
from torchvision.transforms import Compose, Resize
from .sam import build_sam_vit_b
from transformers import AutoImageProcessor

# base_vision
########################################################################################################
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper

########################################################################################################
# === Interface for an Image Transform ===
class ImageTransform(Protocol):
    def __call__(self, img: Image, **kwargs: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...
    

@dataclass
class SamDinoSigLIPImageTransform:
    dino_image_transform: ImageTransform
    siglip_image_transform: ImageTransform
    sam_image_transform: ImageTransform

    def __post_init__(self) -> None:
        self.image_size = {"dino": 384, "siglip": 384, "sam": 1024}

    def __call__(self, img: Image, **kwargs: str) -> Dict[str, torch.Tensor]:
        sam_pixel_values = self.sam_image_transform(img, return_tensors='pt')['pixel_values'][0]
        return {"dino": self.dino_image_transform(img, **kwargs), 
                "siglip": self.siglip_image_transform(img, **kwargs),
                "sam": sam_pixel_values}


class SamDinoSigLIPViTBackbone(nn.Module):
    def __init__(self, vision_tower_dir: dict, default_image_size: int = 384) -> None:
        super().__init__()
        self.default_image_size = default_image_size
        self.dino_timm_path_or_url = "vit_large_patch14_reg4_dinov2.lvd142m"
        self.siglip_timm_path_or_url = "vit_so400m_patch14_siglip_384"
        self.sam_ckpt_path_or_url = vision_tower_dir["sam"]

        # Initialize both Featurizers (ViTs) by downloading from HF / TIMM Hub if necessary
        self.dino_featurizer = timm.create_model(
            self.dino_timm_path_or_url, pretrained=True, num_classes=0, img_size=default_image_size, 
            pretrained_cfg_overlay=dict(file=vision_tower_dir["dino"])
        )
        self.dino_featurizer.eval()

        self.siglip_featurizer = timm.create_model(
            self.siglip_timm_path_or_url, pretrained=True,  num_classes=0, 
            pretrained_cfg_overlay=dict(file=vision_tower_dir['siglip'])
        )
        self.siglip_featurizer.eval()

        self.sam_featurizer = build_sam_vit_b(checkpoint=self.sam_ckpt_path_or_url)

        # Monkey-Patch the `forward()` function of the featurizers to ensure FSDP-compatibility
        #   => Note: By default set `get_intermediate_layers` to return the *SECOND-TO-LAST* layer patches!
        #   => TODO (siddk) Remove after resolution of https://github.com/pytorch/pytorch/issues/109385
        self.dino_featurizer.forward = unpack_tuple(
            partial(self.dino_featurizer.get_intermediate_layers, n={len(self.dino_featurizer.blocks) - 2})
        )
        self.siglip_featurizer.forward = unpack_tuple(
            partial(self.siglip_featurizer.get_intermediate_layers, n={len(self.siglip_featurizer.blocks) - 2})
        )

        # Get Configs for _both_ Featurizers =>> Note :: Override default image size for larger resolution models
        self.dino_data_cfg = timm.data.resolve_model_data_config(self.dino_featurizer)
        self.dino_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        self.siglip_data_cfg = timm.data.resolve_model_data_config(self.siglip_featurizer)
        self.siglip_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # Initialize *both* Transformszh
        default_dino_transform = timm.data.create_transform(**self.dino_data_cfg, is_training=False)
        default_siglip_transform = timm.data.create_transform(**self.siglip_data_cfg, is_training=False)
        default_sam_transform = AutoImageProcessor.from_pretrained('facebook/sam-vit-base')

        # Fix =>> SigLIP default transform resizes to *larger* than `self.default_image_size` (crops image)!!
        assert isinstance(default_siglip_transform, Compose), "Unexpected `default_image_transform`!"
        assert isinstance(sl_resize_transform := default_siglip_transform.transforms[0], Resize)
        default_siglip_transform = Compose(
            [
                Resize(self.default_image_size, interpolation=sl_resize_transform.interpolation),
                *default_siglip_transform.transforms[1:],
            ]
        )

        assert isinstance(default_dino_transform, Compose), "Unexpected `default_dino_image_transform`!"
        assert isinstance(default_siglip_transform, Compose), "Unexpected `default_siglip_image_transform`!"
        assert isinstance(dino_resize_transform := default_dino_transform.transforms[0], Resize)
        assert isinstance(siglip_resize_transform := default_siglip_transform.transforms[0], Resize)

        sig_target_size = (self.default_image_size, self.default_image_size)
        dino_target_size = (self.default_image_size, self.default_image_size)
        dino_transform = Compose(
            [
                Resize(dino_target_size, interpolation=dino_resize_transform.interpolation),
                *default_dino_transform.transforms[1:],
            ]
        )
        siglip_transform = Compose(
            [
                Resize(sig_target_size, interpolation=siglip_resize_transform.interpolation),
                *default_siglip_transform.transforms[1:],

            ]
        )

        self.image_transform = SamDinoSigLIPImageTransform(dino_transform, siglip_transform, default_sam_transform)


    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Runs the transformed image/pixel tensors through each vision backbone, returning concatenated patches."""
        dino_patches = self.dino_featurizer(pixel_values["dino"])[0]
        siglip_patches = self.siglip_featurizer(pixel_values["siglip"])[0]
        sam_patches = self.sam_featurizer(pixel_values["sam"])
        B, C, H, W = sam_patches.shape
        sam_patches = sam_patches.view(B, C, H*W).permute(0, 2, 1)
        return torch.cat([dino_patches, siglip_patches, sam_patches], dim=2)

    def get_image_transform(self) -> ImageTransform:
        return self.image_transform

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.dino_data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.dino_featurizer.embed_dim + self.siglip_featurizer.embed_dim + self.sam_featurizer.output_dim

    @property
    def num_patches(self) -> int:
        assert self.dino_featurizer.patch_embed.num_patches == self.siglip_featurizer.patch_embed.num_patches
        return self.dino_featurizer.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16