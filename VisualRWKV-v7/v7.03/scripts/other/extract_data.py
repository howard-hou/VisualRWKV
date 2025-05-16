# extract data from the raw data json, based on the given image set

import json
import sys
from collections import Counter

def analyze_data(data, key):
    # know the distribution of the data
    image_list = []
    full_image_list = []
    for line in data:
        if key in line:
            image_set = line[key].split("/")[0]
            image_list.append(image_set)
            if key == "image":
                full_image_list.append(line[key])
            if key == "image_dir":
                for l in line["conversations"]:
                    if "image" in l:
                        full_image_list.extend(l["image"])
    counter = Counter(image_list)
    print("data distribution:", counter)
    print("total data:", len(image_list))
    print("total unique data:", len(set(full_image_list)))

def extract_data(data, analyze_key, keep_key):
    # extract the data based on the given image set
    new_data = []
    for line in data:
        if analyze_key in line:
            dataset = line[analyze_key].split("/")[0]
            if dataset in keep_key:
                new_data.append(line)
    print('keep data:', len(new_data))
    return new_data

data = json.load(open(sys.argv[1]))
analyze_key = sys.argv[2]
analyze_data(data, analyze_key)
if ',' in sys.argv[3]:
    keep_key = sys.argv[3].split(",")
    new_data = extract_data(data, analyze_key, keep_key)
    sub_name = f"{len(new_data) // 1000}k"
    output_path = sys.argv[1].replace(".json", f"_{sub_name}.json")
    json.dump(new_data, open(output_path, "w"), indent=2)
