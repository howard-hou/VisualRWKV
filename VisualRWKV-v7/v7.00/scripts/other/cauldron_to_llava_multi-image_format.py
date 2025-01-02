import datasets
import sys
from pathlib import Path
import json
from tqdm import tqdm
# possible image configurations: (1:2), (2:1), (1:1), (1:3), (3:1)
POSSIBLE_RESOLUTIONS = [(448, 896), (896, 448), (896, 896), (448, 1344), (1344, 448)]


def split_image_into_tiles(image, n, m):
    '''
    Split the image into n x m tiles.
    '''
    width, height = image.size
    tile_width = width // m
    tile_height = height // n
    tiles = []
    # left top, right top, left bottom, right bottom
    for i in range(n):
        for j in range(m):
            x = j * tile_width
            y = i * tile_height
            tiles.append(image.crop((x, y, x + tile_width, y + tile_height)))
    return tiles


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


def single_image_to_multi_image_strategy(image, best_resolution):
    """
    Splits a single image into n x m tiles and returns a list of images.

    Args:
        image (PIL.Image): The image to be split.
        n (int): The number of rows.
        m (int): The number of columns.

    Returns:
        list: A list of PIL.Image objects.
    """
    if best_resolution == (896, 896):
        image_area = image.size[0] * image.size[1]
        if image_area <= (896 * 896):
            return [image]
        else: # large image needs to be split
            n, m = 2, 2
    if best_resolution == (448, 896):
        n, m = 2, 1
    if best_resolution == (896, 448):
        n, m = 1, 2
    if best_resolution == (448, 1344):
        n, m = 3, 1
    if best_resolution == (1344, 448):
        n, m = 1, 3
    tiles = split_image_into_tiles(image, n, m)
    return [image] + tiles



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
        image = image.convert("RGB")
        best_resolution = select_best_resolution(image.size, POSSIBLE_RESOLUTIONS)
        image_list = single_image_to_multi_image_strategy(image, best_resolution)
        for split_id, img in enumerate(image_list):
            img_name = f"{ds_name}_{line_id}_{img_id}_{split_id}.jpg"
            img_path = image_output / img_name
            img.save(img_path)
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