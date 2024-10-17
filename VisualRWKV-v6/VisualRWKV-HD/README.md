# VisualRWKV-HD and UHD: Advancing High-Resolution Processing for Visual Language Models
📖 [Paper](https://arxiv.org/html/2410.11665) | 🤗 [Model](https://huggingface.co/howard-hou/visualrwkv-6)

VisualRWKV-HD and UHD are two advancements in high-resolution processing for visual language models.

## tl;dr
VisualRWKV-HD: High-Definition, up to 1080x1080 pixels.

## Introduction

The paper "VisualRWKV-HD and UHD: Advancing High-Resolution Processing for Visual Language Models" introduces two advanced models, VisualRWKV-HD and VisualRWKV-UHD, designed to enhance the processing of high-resolution visual inputs within Visual Language Models (VLMs). These models are particularly focused on tasks requiring fine-grained visual details and document analysis.

Summary of the Paper's Contributions:
Ensemble of Encoders (HD): The VisualRWKV-HD model integrates three pre-trained encoders (SigLIP, DINOv2, and SAM), which are optimized for handling images at resolutions up to 1024x1024. This allows for a detailed analysis of visual inputs, leading to significant improvements across benchmarks, especially for tasks like TextVQA.

Lossless Downsampling Strategy: A novel lossless downsampling technique is introduced to ensure that the high-resolution encoders can align with low-resolution modules without losing critical information during the training process. This method effectively maintains detail, reducing computational costs while enhancing performance.

UHD Strategy: VisualRWKV-UHD splits an input image into four segments to process both high and low-resolution features, ultimately supporting resolutions up to 4096x4096. This method helps balance coarse and fine image features, maintaining high performance on visually complex tasks.

MLP with Context Gating: The model introduces an MLP with Context Gating mechanism, replacing linear projection layers. This adjustment reduces feature competition and enhances training stability, improving the overall performance of the model.

Performance: VisualRWKV-HD and UHD models show notable improvements over baseline models across various benchmarks, such as DocVQA, InfographicVQA, and ChartQA. These tasks require the model to capture intricate details, and the high-resolution processing capabilities of these models lead to better visual understanding.

Key Findings:
Efficiency: The models balance high-resolution visual processing with computational efficiency, allowing them to outperform other high-resolution models like LLaVA-UHD while maintaining competitive inference speeds and reduced memory usage.
Applications: These models excel in tasks involving text-rich visuals and complex document analysis, making them suitable for applications in fields like automated document processing and visual question answering.
The VisualRWKV-HD and UHD models represent a significant step forward in visual language model development, especially in handling high-resolution and visually complex tasks efficiently.

# Citation
If you find this repo useful, please consider citing our works:
```bib
@inproceedings{Li2024VisualRWKVHDAU,
  title={VisualRWKV-HD and UHD: Advancing High-Resolution Processing for Visual Language Models},
  author={Zihang Li and Haowen Hou},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:273350770}
}
```

