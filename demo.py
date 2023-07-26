import torch
import time
import visualrwkv
import os
from PIL import Image
from visualrwkv.utlis import postprocess_response
import argparse

os.environ["RWKV_JIT_ON"] = "1"
# RWKV_CUDA_ON not support jet, please don't use it
os.environ["RWKV_CUDA_ON"] = "0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--rwkv_path", type=str, default=None)
    args = parser.parse_args()
    return args


args = parse_args()
device_list = ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda"]

for device in device_list:
    print(f"now testing in {device}")

    model, preprocess = visualrwkv.load(
        model_name="VisualRWKV-small",
        adapter_path=args.adapter_path,
        rwkv_path=args.rwkv_path,
        device=device,
    )

    instruction = ["describe the image."]
    max_new_tokens = 20  # use to control the length of the generated text
    start_time = time.time()
    for _ in range(1):
        image = preprocess(Image.open("demo.png")).unsqueeze(0).to(device)
        image_embs = model.adapter.forward_task_embs(image)
        outputs = model.generate(
            image_embs, instruction=instruction, max_new_tokens=max_new_tokens
        )
        decoded = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded = postprocess_response(decoded)
    print("output", decoded)
    elapsed_time = (time.time() - start_time) / 1 * 1000  # ms
    print(f"time cost: {elapsed_time} ms per image using {device}")
    print("-" * 50)
