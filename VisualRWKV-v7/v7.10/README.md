# VisualRWKV: A Visual-Enhanced RWKV

## tl;dr
VisualRWKV-MultiImage: Support multiple images in VisualRWKV.

## Important Changes:
- Data format is changed. Now it is different from the LLaVA format. So LLaVA can not be directly used in this version.
- Image feature insertion logic is changed. Now it can support any number of images.
- Therefore, this version can support multiple images from single image splits, multi-images and video frames.

## 版本差异（v7.10）
- 视觉侧切回 `SamDinoSigLIPViTBackbone`，并引入带 mask 的 `Block` 与新的 `VRWKV/VBlock` 结构，用于视觉预训练/分类分支（含 ImageNet head）。  
- 训练/评估管线改为基于 `patch_size` 与 `image_size` 计算视觉 token 数，图像预处理使用 torchvision transforms；新增 `enable_pretrain_mode` 与 ImageNet 训练/评估脚本。  
- 数据集支持多图列表与 `num_image_per_sample`，并携带 `class_id`（`cls_labels`）用于分类任务；工具函数回归固定 `POSSIBLE_RESOLUTIONS` 的 tile 切分策略。  
