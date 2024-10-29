from pathlib import Path
import sys
from tqdm import tqdm
import copy

input_path = Path(sys.argv[1])

import json
data = json.load(open(input_path))
print('all data:', len(data))

# grouping data by key: image
data_dict = {}
for data in tqdm(data, desc="grouping data"):
    image = data['image']
    if image not in data_dict:
        data_dict[image] = []
    data_dict[image].append(data)


def clean_conversations(convs):
    # change from to human and gpt
    # pair the conversation
    cleaned_convs = []
    for i in range(0, len(convs), 2):
        convs[i]['from'] = 'human'
        convs[i + 1]['from'] = 'gpt'
        cleaned_convs.append(convs[i])
        cleaned_convs.append(convs[i + 1])
    # step 2: remove non-first <image>\n from human
    for conv in cleaned_convs[2:]:
        conv['value'] = conv['value'].replace('<image>\n', '')
    return cleaned_convs



def split_large_multi_round_conversations(data_list, max_rounds=10):
    new_data_list = []
    for data in data_list:
        conversations = data["conversations"]
        # convert to conv pairs
        conv_pairs = []
        for i in range(0, len(conversations), 2):
            conv_pairs.append(conversations[i:i+2])
        # split large multi-round conversations into smaller ones
        if len(conv_pairs) > max_rounds:
            for i in range(0, len(conv_pairs), max_rounds):
                new_conversations = []
                # add <image> to first round
                conv_human, conv_gpt = conv_pairs[i]
                if '<image>' not in conv_human["value"]:
                    conv_human["value"] = "<image>\n" + conv_human["value"]
                new_conversations.append(conv_human)
                new_conversations.append(conv_gpt)
                # add the rest rounds
                for conv_human, conv_gpt in conv_pairs[i+1:i+max_rounds]:
                    new_conversations.append(conv_human)
                    new_conversations.append(conv_gpt)
                #
                new_data = copy.deepcopy(data)
                new_data["conversations"] = new_conversations
                new_data_list.append(new_data)
        else:
            new_data_list.append(data)
    return new_data_list

# merge data
multi_round_data = []
for image, data_list in tqdm(data_dict.items(), desc="merging data"):
    all_convs = set()
    for data in data_list:
        # use json to remove duplicates
        convs = json.dumps(data['conversations'])
        all_convs.add(convs)
    all_convs_dedupe = []
    for convs in all_convs:
        all_convs_dedupe.extend(json.loads(convs))
    all_convs_dedupe = clean_conversations(all_convs_dedupe)
    multi_round_data.append({
        'id': data_list[0]['id'],
        'image': image,
        'conversations': all_convs_dedupe
    })
print('multi-round data:', len(multi_round_data))

multi_round_split_data = split_large_multi_round_conversations(multi_round_data, max_rounds=10)
print('multi-round split data:', len(multi_round_split_data))

json.dump(multi_round_split_data, open(input_path.with_name(input_path.stem + '_multi_round.json'), 'w'), indent=2)
