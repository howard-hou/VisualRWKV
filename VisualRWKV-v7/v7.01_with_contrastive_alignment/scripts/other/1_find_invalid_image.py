import json
import sys
import datetime
from pathlib import Path
from PIL import Image

def is_image_valid(image_path_list):
    error_list = []
    for img_path in image_path_list:
        try:
            img = Image.open(img_path).convert('RGB')
            if img.size[0] < 10 or img.size[1] < 10:
                error = (img_path, 'image_too_small')
                error_list.append(error)
            if img.size[0] * img.size[1] > 4096 * 4096:
                error = (img_path, 'image_too_large')
                error_list.append(error)
        except Exception as e:
            # dict of invalid images
            error = (img_path, str(e).strip().split()[0])
            error_list.append(error)
    return error_list

input_json = sys.argv[1]
image_folder = Path(sys.argv[2])
output_file = input_json.replace('.json', '_invalid_images.json')

data = json.load(open(input_json))
print(f'Checking images in {input_json}...')
print(f'Invalid images will be saved in {output_file}')

json_with_invalid_images = []
num_not_exist_image = 0
num_cannot_open_image = 0
num_not_exist_line = 0
num_cannot_open_line = 0
num_not_exist_and_cannot_open_line = 0

total_images = 0
total_lines = len(data)

for i, line in enumerate(data):
    if 'image' not in line or not line['image']:
        continue
    image_exist, image_not_exist = [], []
    for img in line['image']:
        img_path = image_folder / img
        if img_path.exists():
            image_exist.append(str(img_path))
        else:
            image_not_exist.append(str(img_path))
    num_not_exist_image += len(image_not_exist)
    total_images += len(line['image'])
    # case 1: image all exist
    if image_exist and not image_not_exist:
        error_list = is_image_valid(image_exist)
        if error_list:
            invalid_images = {**line, 'invalid_images': error_list}
            json_with_invalid_images.append(invalid_images)
            num_cannot_open_image += len(error_list)
            num_cannot_open_line += 1
    # case 2: image all not exist
    if image_not_exist and not image_exist:
        invalid_images = {**line, 'invalid_images': [(img, 'not_exist') for img in image_not_exist]}
        json_with_invalid_images.append(invalid_images)
        num_not_exist_line += 1
    # case 3: mix of exist and not exist
    if image_exist and image_not_exist:
        invalid_images = [(img, 'not_exist') for img in image_not_exist]
        error_list = is_image_valid(image_exist)
        if error_list:
            invalid_images += error_list
            num_cannot_open_image += len(error_list)
        invalid_images = {**line, 'invalid_images': invalid_images}
        json_with_invalid_images.append(invalid_images)
        if error_list: # case 3.1: exist images cannot open
            num_not_exist_and_cannot_open_line += 1
        else:
            num_not_exist_line += 1

    if i % 1000 == 0:
        print(f'{datetime.datetime.now()} - {i}/{len(data)} lines checked')


print(f'Lines with invalid image found: {len(json_with_invalid_images)} out of {total_lines} lines')
print(f'Total images: {total_images}, not exist images: {num_not_exist_image}, cannot open images: {num_cannot_open_image}')
print(f'Lines with not exist images: {num_not_exist_line}, lines with cannot open images: {num_cannot_open_line}, lines with both: {num_not_exist_and_cannot_open_line}')

if json_with_invalid_images:
    json.dump(json_with_invalid_images, open(output_file, 'w'), indent=2, ensure_ascii=False)