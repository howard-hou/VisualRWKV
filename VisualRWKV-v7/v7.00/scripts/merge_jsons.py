from pathlib import Path
import sys
from tqdm import tqdm

input_path = Path(sys.argv[1])
file_list = list(input_path.glob('*.json'))

import json
json_list = [json.load(open(file)) for file in tqdm(file_list, desc="loading json files")]

new_data = []
for data in tqdm(json_list, desc="processing json files"):
    for line in data:
        new_data.append(line)
print('keep data:', len(new_data))
sub_name = f"{len(new_data) // 1000}k"
output_path = sys.argv[2] + f"_{sub_name}.json"
json.dump(new_data, open(output_path, "w"), indent=2, ensure_ascii=False)