import datasets
import sys
from pathlib import Path
import json
from tqdm import tqdm

input_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
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
        image = image.convert("RGB")
        image.save(img_path)
        img_path_list.append(img_name)
    convs = []
    image_prifix = "\n".join(len(img_path_list) * ["<image>"])
    instruction = image_prifix + "\n" + sample["texts"][0]["user"]
    convs.append({"from": "human", "value": instruction, "image": img_path_list})
    convs.append({"from": "gpt", "value": sample["texts"][0]["assistant"]})
    # if conversation is more than one round
    for conv in sample["texts"][1:]:
        human_conv = {"from": "human", "value": conv["user"]}
        gpt_conv = {"from": "gpt", "value": conv["assistant"]}
        convs.append(human_conv)
        convs.append(gpt_conv)
    data.append({"id": sample_id, "image_dir": ds_name, "conversations": convs})
json.dump(data, open(json_output, "w"), indent=2, ensure_ascii=False)