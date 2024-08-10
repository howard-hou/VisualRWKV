# extract data from the raw data json, based on the given image set

import json
import sys
from collections import Counter

def analyze_data(data, key):
    # know the distribution of the data
    image_sets = []
    for line in data:
        if key in line:
            image_set = line[key].split("/")[0]
            image_sets.append(image_set)
    counter = Counter(image_sets)
    print(counter)

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
