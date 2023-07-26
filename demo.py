import torch
import time
import visualrwkv
import os
from PIL import Image
from pathlib import Path
from visualrwkv.utlis import postprocess_response
import argparse

os.environ["RWKV_JIT_ON"] = "1"
# RWKV_CUDA_ON not support jet, please don't use it
os.environ["RWKV_CUDA_ON"] = "0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--rwkv_path", type=str, default=None)
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["VisualRWKV-small", "VisualRWKV-world-7b"],
        default="VisualRWKV-small",
    )
    args = parser.parse_args()
    return args


args = parse_args()
device_list = ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda"]
# glob png or jpeg images

caption_image_list = sorted(Path("images/caption").glob("*.png")) + sorted(
    Path("images/caption").glob("*.jpeg")
)
vqa_image_list = sorted(Path("images/vqa").glob("*.png")) + sorted(
    Path("images/vqa").glob("*.jpeg")
)

for device in device_list:
    print(f"now testing in {device}")

    model, preprocess = visualrwkv.load(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        rwkv_path=args.rwkv_path,
        device=device,
    )

    instruction = ["describe the image."]
    max_new_tokens = 20  # use to control the length of the generated text
    start_time = time.time()
    for image_path in caption_image_list:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_embs = model.adapter.forward_task_embs(image)
        outputs = model.generate(
            image_embs, instruction=instruction, max_new_tokens=max_new_tokens
        )
        decoded = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded = postprocess_response(decoded)
        print(f"caption of {image_path.name}: ", decoded[0])

    for image_path in vqa_image_list:
        questoin = image_path.stem
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_embs = model.adapter.forward_task_embs(image)
        outputs = model.generate(
            image_embs, instruction=[questoin], max_new_tokens=max_new_tokens
        )
        decoded = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded = postprocess_response(decoded)
        print(f"Question: {questoin} Answer: {decoded[0]}")
    elapsed_time = (time.time() - start_time) / len(caption_image_list) * 1000  # ms
    print(f"time cost: {elapsed_time} ms per image using {device}")
    print("-" * 50)
