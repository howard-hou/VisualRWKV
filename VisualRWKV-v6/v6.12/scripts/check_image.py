from pathlib import Path
import sys
from tqdm import tqdm
import json
from PIL import Image

data = json.load(open(sys.argv[1]))
image_folder = Path(sys.argv[2])

for line in tqdm(data):
    image_dir = line["image_dir"]
    for conv in line["conversations"]:
        if "image" in conv:
            for img in conv["image"]:
                img_path = image_folder / image_dir / img
                if not img_path.exists():
                    print(f"Image not found: {img_path}")
                    continue
                try:
                    img = Image.open(img_path)
                    # DecompressionBombWarning: Image size (92150000 pixels) exceeds limit of 89478485 pixels, could be decompression bomb DOS attack.
                    # resize the image to larger side of 1024
                    if img.size[0] * img.size[1] > 1024 * 1024:
                        ratio = max(img.size) / 1024
                        img = img.resize((int(img.size[0] / ratio), int(img.size[1] / ratio)))
                        img.save(img_path)
                        print(f"image is too large, resize image {img_path} to {img.size}")
                except Exception as e:
                    print(f"Error opening image {img_path}: {e}")
                    continue
        
