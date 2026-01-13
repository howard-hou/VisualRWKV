# VisualRWKV: A Visual-Enhanced RWKV

## tl;dr
VisualRWKV-MultiImage: Support multiple images in VisualRWKV.

## Important Changes:
- Data format is changed. Now it is different from the LLaVA format. So LLaVA can not be directly used in this version.
- Image feature insertion logic is changed. Now it can support any number of images.
- Therefore, this version can support multiple images from single image splits, multi-images and video frames.

## 版本差异（v7.04）
- `multi_image_collate_fn` 改回固定形状 `(B, N, C, H, W)`，并恢复按 region 数量填充 image token。  
- `encode_images` 回到 `(B, N, L, D)` 处理路径，`compress_visual_tokens` 以 N 维聚合，损失计算也回到 v7.02 风格。  
- `num_token_per_image` 默认值调回 256；脚本侧做了清理（保留 `diff_stem_delete_common.py`、`rename.sh`，移除多余数据处理脚本）。  
