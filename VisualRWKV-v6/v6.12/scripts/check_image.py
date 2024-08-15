from pathlib import Path
import sys
from tqdm import tqdm
import json
from PIL import Image


def is_image_valid(line, image_not_found, try_open=False, do_resize=False):
    image_dir = line["image_dir"]
    for conv in line["conversations"]:
        if "image" in conv:
            for img in conv["image"]:
                img_path = image_folder / image_dir / img
                if img_path in image_not_found:
                    print(f"Image not found: {img_path}")
                    return False
                if try_open:
                    try:
                        img = Image.open(img_path)
                        # DecompressionBombWarning: Image size (92150000 pixels) exceeds limit of 89478485 pixels, could be decompression bomb DOS attack.
                        # resize the image to larger side of 1024
                        if do_resize:
                            if img.size[0] * img.size[1] > 1024 * 1024:
                                ratio = max(img.size) / 1024
                                img = img.resize((int(img.size[0] / ratio), int(img.size[1] / ratio)))
                                img.save(img_path)
                                print(f"image is too large, resize image {img_path} to {img.size}")
                    except Exception as e:
                        print(f"Error opening image {img_path}: {e}")
                        # 
                        return False
    return True
        

image_folder = Path(sys.argv[2])

# step1: get all image paths in the folder recursively
child_in_folder = [c for c in image_folder.iterdir()]
all_images_in_folder = []
for d in tqdm(child_in_folder, desc="walk through image folder"):
    all_images_in_folder.extend(list(d.glob('**/*')))
print('all images in folder:', len(all_images_in_folder))
all_images_in_folder_set = set(all_images_in_folder)

# step2: get all image paths in the json file
data = json.load(open(sys.argv[1]))
print("input data:", len(data))
all_images_in_json = []
for line in tqdm(data, desc="walk through json file"):
    if "image_dir" not in line:
        continue
    image_dir = line["image_dir"]
    for conv in line["conversations"]:
        if "image" in conv:
            for img in conv["image"]:
                all_images_in_json.append(image_folder / image_dir / img)
print('all images in json:', len(all_images_in_json))
all_images_in_json_set = set(all_images_in_json)
print('all images in json set:', len(all_images_in_json_set))

# step3: check if all images in json are in the folder
image_not_found = []
for img in all_images_in_json:
    if img not in all_images_in_folder_set:
        image_not_found.append(img)
print('image not found:', len(image_not_found))

# step4: check if all images in folder are in the json
image_not_used = []
for img in all_images_in_folder:
    if img not in all_images_in_json_set:
        image_not_used.append(img)
print('image not used:', len(image_not_used))
from collections import Counter
not_used_counter = Counter([str(img).split('/')[1] for img in image_not_used])
all_images_in_folder_counter = Counter([str(img).split('/')[1] for img in all_images_in_folder])
for k, v in not_used_counter.items():
    o = dict(not_used=v, all=all_images_in_folder_counter[k], not_used_ratio=round(v/all_images_in_folder_counter[k], 3))
    print(f'image not used of {k}: ', o)

# step5: check if all images in json are valid
new_data = []
for line in tqdm(data, desc="check image in json"):
    if "image_dir" not in line:
        new_data.append(line)
        continue
    image_dir = line["image_dir"]
    if not is_image_valid(line, image_not_found):
        continue
    new_data.append(line)

print('keep data:', len(new_data))
output_path = sys.argv[1].replace(".json", "_valid.json")
json.dump(new_data, open(output_path, "w"), indent=2, ensure_ascii=False)
    
        
