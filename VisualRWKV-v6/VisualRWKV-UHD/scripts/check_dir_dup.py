from pathlib import Path
import sys
from tqdm import tqdm
from collections import Counter

input_path = Path(sys.argv[1])
file_list = list(input_path.glob('*.json'))

import json
stat_list = []
for file in tqdm(file_list, desc="loading json files"):
    data = json.load(open(file))
    image_list = []
    for line in data:
        if "image" not in line:
            continue
        image_list.append(line["image"])
    counter = Counter(image_list)
    top3 = [f"{k}:{v}" for k, v in counter.most_common(3)]
    stat = dict(name=file.name, total=len(image_list), unique=len(counter), top3=top3)
    stat_list.append(stat)
json.dump(stat_list, open(input_path.with_name(input_path.stem + '_stat.json'), 'w'), indent=2)