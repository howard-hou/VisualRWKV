______________________________________________________________________

<div align="center">

# VisualRWKV

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

</div>

## Description

VisualRWKV is the visual-enhanced version of the RWKV language model, enabling RWKV to handle various visual tasks.

## Usage
### Installation
```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install requirements
pip install -r requirements.txt
```
### Download Checkpoint

RWKV models are here, RWKV-4 169M:
https://huggingface.co/BlinkDL/rwkv-4-pile-169m/blob/main/RWKV-4-Pile-169M-20220807-8023.pth

Visual Adapter model:
https://huggingface.co/BlinkDL/rwkv-4-pile-169m


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