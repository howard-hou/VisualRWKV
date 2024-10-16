import datasets
import sys
from pathlib import Path
import json
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--save_image', action='store_true')
    return parser.parse_args()

args = parse_args()
print(f'now running {args.input_path}, only use the first image, output to {args.output_dir}, save_image={args.save_image}')
input_path = Path(args.input_path)
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)
ds_name = input_path.stem
ds = datasets.load_from_disk(str(input_path))
if "train" in ds:
    ds = ds["train"]

image_output = output_dir / ds_name
image_output.mkdir(exist_ok=True)
json_output = output_dir / f"{ds_name}.json"

data = []
for line_id, sample in enumerate(tqdm(ds)):
    sample_id = f"{ds_name}_{line_id}"
    img_path_list = []
    for img_id, image in enumerate(sample["images"]):
        img_name = f"{ds_name}_{line_id}_{img_id}.jpg"
        img_path = image_output / img_name
        if args.save_image:
            image = image.convert("RGB")
            image.save(img_path)
        img_path_list.append(img_name)
    convs = []
    # only take the first image
    img_path_list = img_path_list[:1]
    image_prifix = "\n".join(len(img_path_list) * ["<image>"])
    instruction = image_prifix + "\n" + sample["texts"][0]["user"]
    convs.append({"from": "human", "value": instruction})
    convs.append({"from": "gpt", "value": sample["texts"][0]["assistant"]})
    # if conversation is more than one round
    for conv in sample["texts"][1:]:
        human_conv = {"from": "human", "value": conv["user"]}
        gpt_conv = {"from": "gpt", "value": conv["assistant"]}
        convs.append(human_conv)
        convs.append(gpt_conv)
    image_path = str(Path(ds_name) / img_path_list[0]) 
    data.append({"id": sample_id, "image": image_path, "conversations": convs})
json.dump(data, open(json_output, "w"), indent=2, ensure_ascii=False)