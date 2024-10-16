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
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from PIL import Image
from torchvision.transforms import Compose, Resize
from .sam import build_sam_vit_b

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
        self.image_size = {"dino": 448, "siglip": 448, "sam": 1024}

    def __call__(self, img: Image, **kwargs: str) -> Dict[str, torch.Tensor]:
        return {"dino": self.dino_image_transform(img, **kwargs), 
                "siglip": self.siglip_image_transform(img, **kwargs),
                "sam": self.sam_image_transform(img, **kwargs)}


class SamDinoSigLIPViTBackbone(nn.Module):
    def __init__(self, vision_tower_path: dict, default_image_size: int = 448) -> None:
        super().__init__()
        self.default_image_size = default_image_size
        self.dino_timm_path_or_url = "vit_large_patch14_reg4_dinov2.lvd142m"
        self.siglip_timm_path_or_url = "vit_so400m_patch14_siglip_384"
        self.sam_ckpt_path_or_url = vision_tower_path["sam"]

        # Initialize both Featurizers (ViTs) by downloading from HF / TIMM Hub if necessary
        self.dino_featurizer = timm.create_model(
            self.dino_timm_path_or_url, pretrained=True, num_classes=0, img_size=default_image_size, 
            pretrained_cfg_overlay=dict(file=vision_tower_path["dino"])
        )
        self.dino_featurizer.eval()

        self.siglip_featurizer = timm.create_model(
            self.siglip_timm_path_or_url, pretrained=True,  num_classes=0, img_size=default_image_size,
            pretrained_cfg_overlay=dict(file=vision_tower_path['siglip'])
        )
        self.siglip_featurizer.eval()

        self.sam_featurizer = build_sam_vit_b(checkpoint=self.sam_ckpt_path_or_url)
        self.sam_featurizer.eval()

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
        #default_sam_transform = AutoImageProcessor.from_pretrained('facebook/sam-vit-base')

        sig_target_size = (self.default_image_size, self.default_image_size)
        dino_target_size = (self.default_image_size, self.default_image_size)
        sam_target_size = (1024, 1024)
        dino_transform = Compose(
            [
                Resize(dino_target_size, interpolation=default_dino_transform.transforms[0].interpolation),
                *default_dino_transform.transforms[2:],
            ]
        )
        print('dino_transform:', dino_transform)
        siglip_transform = Compose(
            [
                Resize(sig_target_size, interpolation=default_siglip_transform.transforms[0].interpolation),
                *default_siglip_transform.transforms[2:],

            ]
        )
        print('siglip_transform:', siglip_transform)
        sam_transform = Compose(
            [
                Resize(sam_target_size, interpolation=default_dino_transform.transforms[0].interpolation),
                *default_dino_transform.transforms[2:],
            ]
        )
        print('default_sam_transform:', sam_transform)
        self.image_transform = SamDinoSigLIPImageTransform(dino_transform, siglip_transform, sam_transform)


    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Runs the transformed image/pixel tensors through each vision backbone, returning concatenated patches."""
        # step 1
        B, N, C, H, W = pixel_values["dino"].shape
        # reshape the input tensor to B * N, C, H, W
        dino_pixels = pixel_values["dino"].view(B * N, C, H, W)
        dino_patches = self.dino_featurizer(dino_pixels)
        if isinstance(dino_patches, list):
            dino_patches = dino_patches[0]
        # reshape back to B, N, L, D
        _, L, D = dino_patches.shape
        dino_patches = dino_patches.view(B, N, L, D)
        # step 2
        B, N, C, H, W = pixel_values["siglip"].shape
        # reshape the input tensor to B * N, C, H, W
        siglip_pixels = pixel_values["siglip"].view(B * N, C, H, W)
        siglip_patches = self.siglip_featurizer(siglip_pixels)
        if isinstance(siglip_patches, list):
            siglip_patches = siglip_patches[0]
        # reshape back to B, N, L, D
        _, L, D = siglip_patches.shape
        siglip_patches = siglip_patches.view(B, N, L, D)
        # step 3
        B, N, C, H, W = pixel_values["sam"].shape
        # reshape the input tensor to B * N, C, H, W
        sam_pixels = pixel_values["sam"].view(B * N, C, H, W)
        sam_patches = self.sam_featurizer(sam_pixels)
        _, C, H, W = sam_patches.shape
        sam_patches = sam_patches.view(B, N, C, H, W).view(B, N, C, H*W).permute(0, 1, 3, 2)
        # output: B, N, L, D
        return dino_patches, siglip_patches, sam_patches

    def get_image_transform(self) -> ImageTransform:
        return self.image_transform

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.dino_data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        # global feature dimension + local feature dimension
        return (self.dino_featurizer.embed_dim + self.siglip_featurizer.embed_dim + self.sam_featurizer.output_dim) * 2

    @property
    def num_patches(self) -> int:
        assert self.dino_featurizer.patch_embed.num_patches == self.siglip_featurizer.patch_embed.num_patches
        return self.dino_featurizer.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
    

# === Interface for an Image Feature Extractor ===
def fuse_image_features(dino_patches, siglip_patches, sam_patches):
    '''
    fuse image features from DINO, SigLIP and SAM. Special designed for VisualRWKV-UHD.
    dino_patches: [B, N, L, D]
    siglip_patches: [B, N, L, D]
    sam_patches: [B, N, L, D]
    '''
    B, N, L, _ = dino_patches.shape
    H_or_W = int(L**0.5) # 1024 -> 32
    # concat global image features
    global_features = torch.cat([dino_patches[:, 0, :, :], siglip_patches[:, 0, :, :], sam_patches[:, 0, :, :]], dim=-1)
    # adaptive_avg_pool2d over all tiles
    output_size = H_or_W // 2 # 32 -> 16
    dino_tiles, siglip_tiles, sam_tiles = [], [], []
    for i in range(1, N):
        one_dino = dino_patches[:, i, :, :].view(B, H_or_W, H_or_W, -1).permute(0, 3, 1, 2)
        one_siglip = siglip_patches[:, i, :, :].view(B, H_or_W, H_or_W, -1).permute(0, 3, 1, 2)
        one_sam = sam_patches[:, i, :, :].view(B, H_or_W, H_or_W, -1).permute(0, 3, 1, 2)
        # adaptive_avg_pool2d
        one_dino = F.adaptive_avg_pool2d(one_dino, output_size)
        one_siglip = F.adaptive_avg_pool2d(one_siglip, output_size)
        one_sam = F.adaptive_avg_pool2d(one_sam, output_size)
        dino_tiles.append(one_dino)
        siglip_tiles.append(one_siglip)
        sam_tiles.append(one_sam)
    # 先拼接水平 (W 方向上) 部分
    dino_top = torch.cat([dino_tiles[0], dino_tiles[1]], dim=3) # 拼接左上和右上部分
    siglip_top = torch.cat([siglip_tiles[0], siglip_tiles[1]], dim=3)
    sam_top = torch.cat([sam_tiles[0], sam_tiles[1]], dim=3)
    dino_bottom = torch.cat([dino_tiles[2], dino_tiles[3]], dim=3) # 拼接左下和右下部分
    siglip_bottom = torch.cat([siglip_tiles[2], siglip_tiles[3]], dim=3)
    sam_bottom = torch.cat([sam_tiles[2], sam_tiles[3]], dim=3)
    # 再拼接垂直 (H 方向上) 部分
    dino_features = torch.cat([dino_top, dino_bottom], dim=2)
    siglip_features = torch.cat([siglip_top, siglip_bottom], dim=2)
    sam_features = torch.cat([sam_top, sam_bottom], dim=2)
    # reshape B, C, H, W to B, H*W, C
    dino_features = dino_features.permute(0, 2, 3, 1).reshape(B, -1, dino_features.shape[1])
    siglip_features = siglip_features.permute(0, 2, 3, 1).reshape(B, -1, siglip_features.shape[1])
    sam_features = sam_features.permute(0, 2, 3, 1).reshape(B, -1, sam_features.shape[1])
    # concat global image features with local features at D dimension
    image_features = torch.cat([global_features, dino_features, siglip_features, sam_features], dim=-1)
    return image_features



class VisualFeatureExtractor(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.image_feature_folder = Path(args.image_feature_folder)
        self.vit = SamDinoSigLIPViTBackbone(args.vision_tower_path)
        self.vit.requires_grad_(False)

    def encode_images(self, images):
        '''
        encode image to get image features before projection.
        '''
        dino_patches, siglip_patches, sam_patches = self.vit(images)
        image_features = fuse_image_features(dino_patches, siglip_patches, sam_patches)
        return image_features
    
    @torch.inference_mode()
    def forward(self, samples):
        image_features = self.encode_images(samples['images'])
        return image_features
    
    def predict_step(self, batch):
        image_features = self(batch)
        # replace all image suffix with .npz
        image_feature_file_names = [Path(f).with_suffix('.npz') for f in batch['image_file']]
        # save image features to np
        image_feature_file_paths = [self.image_feature_folder / f for f in image_feature_file_names]
        for f, features in zip(image_feature_file_paths, image_features):
            # make parent directory
            f.parent.mkdir(parents=True, exist_ok=True)
            # use float16 to save space
            features_fp16 = features.float().cpu().numpy().astype(np.float16)
            np.savez(f, features=features_fp16)