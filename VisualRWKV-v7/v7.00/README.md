# VisualRWKV: A Visual-Enhanced RWKV

## tl;dr
VisualRWKV-MultiImage: Support multiple images in VisualRWKV.

## Important Changes:
- Data format is changed. Now it is different from the LLaVA format. So LLaVA can not be directly used in this version.
- Image feature insertion logic is changed. Now it can support any number of images.
- Therefore, this version can support multiple images from single image splits, multi-images and video frames.

## 版本差异（v7.00）
- 使用自定义的 `SamDinoSigLIPViTBackbone` 视觉骨干，并在数据管线中同时处理 dino/sam/siglip 三路图像输入，配套 `multi_image_collate_fn` 合并多图特征。相比后续版本的单一 SigLIP/torchvision 预处理方式更重。  
- `encode_images` 支持 mini-batch 方式提取图像特征并清理缓存，面向显存压力较大的场景。  
