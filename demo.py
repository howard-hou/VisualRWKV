import torch
import visualrwkv
import os
from PIL import Image
from visualrwkv.utlis import postprocess_response
from tqdm import tqdm

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"

device = "cuda" if torch.cuda.is_available() else "cpu"

adapter_path = "/Users/howardhwhou/Downloads/rwkv169_coco-vg-sbu-cc_t5small_vit224.ckpt"
rwkv_path = "/Users/howardhwhou/Downloads/RWKV-4-Pile-169M-20220807-8023.pth"
model, preprocess = visualrwkv.load(
    model_name="VisualRWKV-small", adapter_path=adapter_path, rwkv_path=rwkv_path
)

instruction = ["describe the image."]
max_new_tokens = 20  # use to control the length of the generated text
for _ in tqdm(range(100)):
    image = preprocess(Image.open("demo.png")).unsqueeze(0).to(device)
    image_embs = model.adapter.forward_task_embs(image)
    outputs = model.generate(
        image_embs, instruction=instruction, max_new_tokens=max_new_tokens
    )
    decoded = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    decoded = postprocess_response(decoded)
print(decoded)
# test in cpu, 400 ms per image
