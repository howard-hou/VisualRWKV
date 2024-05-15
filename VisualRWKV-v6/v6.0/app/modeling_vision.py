from transformers import CLIPVisionModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class VisionEncoderConfig:
    n_embd: int = 2048
    vision_tower_name: str = 'openai/clip-vit-large-patch14-336'
    grid_size: int = -1 # -1: no grid pooling, 0: take cls token, 1: global avg pooling, 2, 3, 4, ...: grid pooling

class VisionEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vit = CLIPVisionModel.from_pretrained(args.vision_tower_name)
        self.proj = nn.Linear(self.vit.config.hidden_size, args.n_embd, bias=False)

    def encode_images(self, images):
        B, N, C, H, W = images.shape
        images = images.view(B*N, C, H, W)
        image_features = self.vit(images).last_hidden_state
        L, D = image_features.shape[1], image_features.shape[2]
        # rerange [B*N, L, D] -> [B, N, L, D]
        image_features = image_features.view(B, N, L, D)[:, 0, :, :]
        image_features = self.grid_pooling(image_features)
        return self.proj(image_features)
    
    def grid_pooling(self, image_features):
        cls_features = image_features[:, 0:1, :]
        image_features = image_features[:, 1:, :] #drop cls token
        if self.args.grid_size == -1: # no grid pooling
            return torch.cat((image_features, cls_features), dim=1)
        if self.args.grid_size == 0: # take cls token
            return cls_features
        if self.args.grid_size == 1: # global avg pooling
            return torch.cat((image_features.mean(dim=1, keepdim=True), cls_features), dim=1)
        B, L, D = image_features.shape
        H_or_W = int(L**0.5)
        image_features = image_features.view(B, H_or_W, H_or_W, D)
        grid_stride = H_or_W // self.args.grid_size
        image_features = F.avg_pool2d(image_features.permute(0, 3, 1, 2), 
                                      padding=0,
                                      kernel_size=grid_stride, 
                                      stride=grid_stride)
        image_features = image_features.permute(0, 2, 3, 1).view(B, -1, D)
        return torch.cat((image_features, cls_features), dim=1)