# VisualRWKV: A Visual-Enhanced RWKV

## tl;dr
VisualRWKV-MultiImage: Support multiple images in VisualRWKV.

## Important Changes:
- Data format is changed. Now it is different from the LLaVA format. So LLaVA can not be directly used in this version.
- Image feature insertion logic is changed. Now it can support any number of images.
- Therefore, this version can support multiple images from single image splits, multi-images and video frames.

## 版本差异（v7.03）
- 新增 Visual Token Compressor（VTC），提供 `n_vtc_layer` 与 `compress_visual_tokens`，默认 `num_token_per_image=1024`，并支持从 RWKV block 初始化 VTC 权重。  
- 图像编码支持每样本不同 tile 数量的右对齐 padding；`multi_image_collate_fn` 传回 `images` 列表与 `tile_counts`。  
- `select_best_resolution` 默认策略改为 `closest`；新增 `process_pdfs_to_md.py` 与 `scripts/train/test.sh`。  
