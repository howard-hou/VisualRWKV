import json
import sys
import datetime
from pathlib import Path
from PIL import Image

def is_image_valid(image_path_list):
    error_list = []
    for img_path in image_path_list:
        try:
            Image.open(img_path).convert('RGB')
        except Exception as e:
            # dict of invalid images
            error = {img_path: str(e)}
            error_list.append(error)
    return error_list

input_json = sys.argv[1]
image_folder = Path(sys.argv[2])
output_file = input_json.replace('.json', '_invalid_images.json')

data = json.load(open(input_json))
print(f'Checking images in {input_json}...')
print(f'Invalid images will be saved in {output_file}')

invalid_images = []
for i, line in enumerate(data):
    if 'image' not in line:
        continue
    image_path_list = [str(image_folder / img) for img in line['image']]
    error_list = is_image_valid(image_path_list)
    invalid_images.extend(error_list)
    if i % 1000 == 0:
        print(f'{datetime.datetime.now()} - {i}/{len(data)} images checked')

with open(output_file, 'w') as f:
    for invalid_image in invalid_images:
        f.write(json.dumps(invalid_image) + '\n')

print(f'Invalid images found: {len(invalid_images)}')