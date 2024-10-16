import datasets
import sys
from pathlib import Path
import json
from tqdm import tqdm
# possible image configurations: (1:2), (2:1), (1:1), (1:3), (3:1)
POSSIBLE_RESOLUTIONS = [(448, 896), (896, 448), (896, 896), (448, 1344), (1344, 448)]

def calc_image_width_to_height_ratio(image):
    return image.width / image.height


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = downscaled_width * downscaled_height
        wasted_resolution = abs((width * height) - effective_resolution)

        if wasted_resolution < min_wasted_resolution:
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit



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
        print('image size:', image.size)
        print('width to height ratio:', calc_image_width_to_height_ratio(image))
        best_resolution = select_best_resolution(image.size, POSSIBLE_RESOLUTIONS)
        print('best resolution:', best_resolution)
        print('best resolution width to height ratio:', best_resolution[0] / best_resolution[1])
        break
        #image.save(img_path)
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
    break
json.dump(data, open(json_output, "w"), indent=2, ensure_ascii=False)