import sys
import random
from datasets import load_from_disk
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

ds = load_from_disk(sys.argv[1])['test']
subtask2idx = defaultdict(list)
for i, sample in enumerate(tqdm(ds)):
    sub_task = sample['sub_task']
    subtask2idx[sub_task].append(i)

total_idx = []
for s in subtask2idx:
    idx = subtask2idx[s]
    if len(idx) > 500:
        samples = random.sample(idx, k=500)
        total_idx.extend(samples)
    else:
        total_idx.extend(idx)

sorted_idx = sorted(total_idx)
ds_lite = ds.select(sorted_idx)
print('lite dataset:', ds_lite)
output_path = Path(sys.argv[1]).parent / (Path(sys.argv[1]).name + '-lite')
ds_lite.save_to_disk(str(output_path))