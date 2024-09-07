import json
import sys
from pathlib import Path


# 设置文件夹路径
orig_json = sys.argv[1]
fix_json = Path(sys.argv[2])
output_file = orig_json.replace('.json', '_fixed.json')

orig_data = json.load(open(orig_json))
id2orig = {}
for line in orig_data:
    sample_id = line.get("sample_id", None)
    if sample_id is None:
        sample_id = line.get('id', None)
    if sample_id is None:
        print(f"Sample id not found in {line}")
        continue
    if isinstance(line['image'], list):
        image = sorted([Path(img).stem for img in line['image']])
        image = '-'.join(image)
    else:
        image = line['image']
    ins = line['conversations'][0]['value']
    unique_id = f"{sample_id}_{image}_{ins}"
    id2orig[unique_id] = line

fix_data = json.load(open(fix_json))
id2fix = {}
for line in fix_data:
    sample_id = line.get("sample_id", None)
    if sample_id is None:
        sample_id = line.get('id', None)
    if sample_id is None:
        print(f"Sample id not found in {line}")
        continue
    if isinstance(line['image'], list):
        image = sorted([Path(img).stem for img in line['image']])
        image = '-'.join(image)
    else:
        image = line['image']
    ins = line['conversations'][0]['value']
    unique_id = f"{sample_id}_{image}_{ins}"
    id2fix[unique_id] = line

# use fix data to update orig data
fix_cnt = 0
for unique_id, fix_line in id2fix.items():
    if unique_id in id2orig:
        orig_line = id2orig[unique_id]
        orig_line['image'] = fix_line['image']
        fix_cnt += 1
print(f"Fixed {fix_cnt} lines, total {len(fix_data)} lines in fix data")

json.dump(orig_data, open(output_file, 'w'), indent=2, ensure_ascii=False)

