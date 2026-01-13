# VisualRWKV: A Visual-Enhanced RWKV

## tl;dr
VisualRWKV-MultiImage: Support multiple images in VisualRWKV.

## Important Changes:
- Data format is changed. Now it is different from the LLaVA format. So LLaVA can not be directly used in this version.
- Image feature insertion logic is changed. Now it can support any number of images.
- Therefore, this version can support multiple images from single image splits, multi-images and video frames.

## 版本差异（v7.01）
- 视觉骨干改为 `transformers.SiglipVisionModel`，并统一使用 `MLPWithContextGating` 投影，去掉了 v7.00 的 `SamDinoSigLIPViTBackbone` 组合式特征。  
- 数据读取支持 `.json/.jsonl`，并引入可重复的随机打乱（`sample_idx_mapping`）；文本样本会自动注入一个 dummy image token。  
- 图像预处理简化为单张图的 `AutoImageProcessor` 输出，缺图时回退到零张量；`sample_id` 统一转为字符串。  
