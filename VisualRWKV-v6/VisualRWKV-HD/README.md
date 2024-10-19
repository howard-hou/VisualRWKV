# VisualRWKV-HD and UHD: Advancing High-Resolution Processing for Visual Language Models
📖 [Paper](https://arxiv.org/html/2410.11665) | 🤗 [Model](https://huggingface.co/howard-hou/visualrwkv-6)

VisualRWKV-HD and UHD are two advancements in high-resolution processing for visual language models.

## tl;dr
VisualRWKV-HD: High-Definition, up to 1080x1080 pixels.

## Introduction

The paper "VisualRWKV-HD and UHD: Advancing High-Resolution Processing for Visual Language Models" introduces two advanced models, VisualRWKV-HD and VisualRWKV-UHD, designed to enhance the processing of high-resolution visual inputs within Visual Language Models (VLMs). These models are particularly focused on tasks requiring fine-grained visual details and document analysis.

## Summary of the Paper's Contributions:
Ensemble of Encoders (HD): The VisualRWKV-HD model integrates three pre-trained encoders (SigLIP, DINOv2, and SAM), which are optimized for handling images at resolutions up to 1024x1024. This allows for a detailed analysis of visual inputs, leading to significant improvements across benchmarks, especially for tasks like TextVQA.

Lossless Downsampling Strategy: A novel lossless downsampling technique is introduced to ensure that the high-resolution encoders can align with low-resolution modules without losing critical information during the training process. This method effectively maintains detail, reducing computational costs while enhancing performance.

UHD Strategy: VisualRWKV-UHD splits an input image into four segments to process both high and low-resolution features, ultimately supporting resolutions up to 4096x4096. This method helps balance coarse and fine image features, maintaining high performance on visually complex tasks.
The following picture is the overview of VisualRWKV-HD and UHD Strategy. 
![](https://i-blog.csdnimg.cn/direct/4733ffae0ea3442db830388222137aa4.png#pic_center)


MLP with Context Gating: The model introduces an MLP with Context Gating mechanism, replacing linear projection layers. This adjustment reduces feature competition and enhances training stability, improving the overall performance of the model.

Performance: VisualRWKV-HD and UHD models show notable improvements over baseline models across various benchmarks, such as DocVQA, InfographicVQA, and ChartQA. These tasks require the model to capture intricate details, and the high-resolution processing capabilities of these models lead to better visual understanding.

The following table presents a performance comparison of various models, highlighting their resolution and task metrics.
| Method                  | LLM                   | Resolution | SQA      | TextQA   | GQA     | VizWiz | MME               | POPE  | MMB/MMB-CN     |
|-------------------------|-----------------------|------------|----------|----------|---------|--------|-------------------|-------|-----------------|
| MobileVLM 1.7B         | MobileLLaMA-1.4B     | 336        | 54.7     | -        | 56.1    | -      | 1196.2/-          | 84.5  | 53.2/-          |
| Mini-Gemini            | Gemma-2B             | 336        | -        | -        | -       | -      | 1341/312          | -     | 59.8/-          |
| TinyLLaVa-v1           | TinyLlama-1.1B       | -          | 59.4     | -        | 57.5    | -      | -                 | -     | -               |
| VisualRWKV              | VisualRWKV6-1.6B     | 336        | 59.1     | 43.6     | 55.2    | -      | 1204.9/-          | 83.2  | 55.8/53.2       |
| **VisualRWKV-HD**       | **VisualRWKV6-1.6B** | **448**    | **54.39**| **54.71**| **60.84**| **54.97**| **1378.62/266.07** | **86.0** | **60.31/55.41** |
| **VisualRWKV-UHD**      | **VisualRWKV6-1.6B** | **448**    | **56.97**| **56.31**| **59.52**| **49.88**| **1321.33/232.14** | **85.3** | **58.42/52.84** |

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

