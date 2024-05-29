# 1. clean missing data, like 000000178275, which gpt has no response
# 2. split large multi-round conversations into smaller ones

import json
import argparse
import copy
import random
random.seed(22)

def clean_missing_data(data_list):
    new_data_list = []
    for data in data_list:
        conversations = data["conversations"]
        new_conversations = []
        # conv pairs
        for i in range(0, len(conversations), 2):
            if conversations[i+1]["from"] == "gpt" and conversations[i+1]["value"]:
                new_conversations.append(conversations[i])
                new_conversations.append(conversations[i+1])
        if new_conversations:
            new_data = copy.deepcopy(data)
            new_data["conversations"] = new_conversations
            new_data_list.append(new_data)
    return new_data_list

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
                if i != 0 and "Answer the question using a single word or phrase." in data["conversations"][0]["value"]:
                    conv_human["value"] += "\nAnswer the question using a single word or phrase."
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str)
    parser.add_argument("--max_rounds", type=int, default=20)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data_list = json.load(open(args.data_file))
    print("original data size:", len(data_list))
    data_list = clean_missing_data(data_list)
    print("cleaned data size:", len(data_list))
    data_list = split_large_multi_round_conversations(data_list,
                                                      max_rounds=args.max_rounds)
    print("reformatted data size:", len(data_list))
    if args.shuffle:
        random.shuffle(data_list)
        print("reformatted data size after shuffle:", len(data_list))
    json.dump(data_list, 
              open(args.data_file.replace(".json", f"_{args.max_rounds}rounds.json"), "w"),
              indent=2, ensure_ascii=False)
