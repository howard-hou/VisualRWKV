import json
import sys
import random
random.seed(222)

input_json = sys.argv[1]
output_file = input_json.replace('.json', '_shuffled.json')

data = json.load(open(input_json))
random.shuffle(data)
json.dump(data, open(output_file, 'w'), indent=2, ensure_ascii=False)
