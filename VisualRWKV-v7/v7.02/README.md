# VisualRWKV: A Visual-Enhanced RWKV

## tl;dr
VisualRWKV-MultiImage: Support multiple images in VisualRWKV.

## Important Changes:
- Data format is changed. Now it is different from the LLaVA format. So LLaVA can not be directly used in this version.
- Image feature insertion logic is changed. Now it can support any number of images.
- Therefore, this version can support multiple images from single image splits, multi-images and video frames.

## 版本差异（v7.02）
- 引入 `image_to_regions` 将单图切分为多个 region，并把 image token 数量扩展为 region 数；纯文本样本不再添加 dummy image token。  
- `multi_image_collate_fn` 回归并将图像拼成扁平的 `(BN, C, H, W)`，训练 dataloader 显式使用该 collate_fn。  
- `encode_images` 简化为直接处理扁平 batch，不再依赖 `(B, N, ...)` 的 reshape。  
