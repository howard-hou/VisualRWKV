from src.rwkv_tokenizer import TRIE_TOKENIZER
from src.dataset import process_image_tokens_in_conversations, _add_speaker_and_signal, tokenize_with_image_token
import argparse
import json
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str, default="data.json")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    tokenizer = TRIE_TOKENIZER("src/rwkv_vocab_v20230424.txt")
    data_list = json.load(open(args.data_file))
    ctx_lens = []
    for data in tqdm(data_list):
        conversations = data["conversations"]
        conversations = process_image_tokens_in_conversations(conversations)
        conversations = _add_speaker_and_signal(conversations)
        conversation = "".join([sentence["value"] for sentence in conversations])
        if "image" in data:
            input_ids = tokenize_with_image_token(conversation, tokenizer)
        else:
            input_ids = tokenizer.encode(conversation)
        ctx_lens.append(len(input_ids))
    print("max ctx len", max(ctx_lens))
    print("min ctx len", min(ctx_lens))
    print("avg ctx len", sum(ctx_lens) / len(ctx_lens))
    print("median ctx len", sorted(ctx_lens)[len(ctx_lens) // 2])
    print("std ctx len", np.std(ctx_lens))
    print("80% ctx len", sorted(ctx_lens)[int(len(ctx_lens) * 0.80)])
    print("90% ctx len", sorted(ctx_lens)[int(len(ctx_lens) * 0.90)])
    print("95% ctx len", sorted(ctx_lens)[int(len(ctx_lens) * 0.95)])
    print("99% ctx len", sorted(ctx_lens)[int(len(ctx_lens) * 0.99)])

