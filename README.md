______________________________________________________________________

<div align="center">

# VisualRWKV

![rwkv logo](rwkv_emoji.png)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

</div>

## Description

VisualRWKV is the visual-enhanced version of the [RWKV language model](https://github.com/BlinkDL/RWKV-LM), enabling RWKV to handle various visual tasks. 

By utilizing a loosely coupled adapter design, visual capabilities can be effortlessly enhanced while preserving the performance of the RWKV language model. This approach allows for easy integration and interchangeability without compromising the core functionality of RWKV.

## Usage
### Installation
```bash
# clone project
git clone https://github.com/howard-hou/VisualRWKV
cd VisualRWKV

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10
conda activate myenv

# install requirements
pip install -r requirements.txt
```
### Download Checkpoint

RWKV models are here, please download LLM and its adapter.

| Model Name         | Link                                                                                                             |
|-------------------|------------------------------------------------------------------------------------------------------------------|
| RWKV-4 169M       | [RWKV-4 169M](https://huggingface.co/BlinkDL/rwkv-4-pile-169m/blob/main/RWKV-4-Pile-169M-20220807-8023.pth)       |
| RWKV-1b5 raven    | [RWKV-1b5 raven](https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/RWKV-4-Raven-1B5-v12-Eng98%25-Other2%25-20230520-ctx4096.pth) |
| Visual Adapter    | [adapted to RWKV-4 169M](https://huggingface.co/howard-hou/VisualRWKV/blob/main/rwkv169_coco-vg-sbu-cc_t5small_vit224.ckpt) |
| Visual Adapter    | [adapted to RWKV-1b5 raven](https://huggingface.co/howard-hou/VisualRWKV/resolve/main/rwkv1b5raven_coco-vg-sbu-cc_t5small_vit224.ckpt) |

## Zero-shot Performance

### Zero-shot Visual Question Answering

| model | dataset | split      | overall | other | yes/no | number |
|-----|---------|------------|---------|-------|--------|--------|
|RWKV-1b5 raven| vqav2   | validation | 43.7    | 34.35 | 60.67  | 30.25  |
|RWKV-4 169M   | vqav2   | validation | 15.41   | 23.59 | 0.14   | 28.42  |

### Zero-shot Image Captioning

| model | dataset | split | Bleu_1 | Bleu_2 | Bleu_3 | Bleu_4 | METEOR | ROUGE_L | CIDEr | SPICE |
|-------|-------------------|-------------|--------|--------|--------|--------|--------|---------|-------|-------|
|RWKV-1b5 raven| coco | test | 0.6911 | 0.5143 | 0.3652 | 0.2542 | 0.2376 | 0.5018  | 0.8658 | 0.1728|
|RWKV-1b5 raven| nocaps | val | 0.6779 | 0.5027 | 0.3521 | 0.242  | 0.2062 | 0.4639  | 0.5988 | 0.0915|
|RWKV-4 169M| coco | test | 0.6762 | 0.4957 | 0.3446 | 0.2332 | 0.2202 | 0.488   | 0.768  | 0.1562|
|RWKV-4 169M| nocaps | val | 0.6561 | 0.4783 | 0.3261 | 0.2142 | 0.1918 | 0.4538  | 0.5184 | 0.0792|

* note: model is trained on coo_caption, so it is not zero-shot; but it is zero-shot on nocaps_caption



### Example

```python
import torch
import visualrwkv
import os
from PIL import Image
from visualrwkv.utlis import postprocess_response
from tqdm import tqdm

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"

device = "cuda" if torch.cuda.is_available() else "cpu"

# download checkpoint to your dir
adapter_path = "your_dir/rwkv169_coco-vg-sbu-cc_t5small_vit224.ckpt"
rwkv_path = "your_dir/RWKV-4-Pile-169M-20220807-8023.pth"
# now only support VisualRWKV-small, will support more models in the future
model, preprocess = visualrwkv.load(
    model_name="VisualRWKV-small", adapter_path=adapter_path, rwkv_path=rwkv_path
)

instruction = ["describe the image."]
max_new_tokens = 20  # use to control the length of the generated text
image = preprocess(Image.open("demo.png")).unsqueeze(0).to(device)
image_embs = model.adapter.forward_task_embs(image)
outputs = model.generate(
    image_embs, instruction=instruction, max_new_tokens=max_new_tokens
)
decoded = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
decoded = postprocess_response(decoded)
print(decoded) 
# ['couple in the water with a dog on the beach']

# test in cpu, 400 ms per image
```