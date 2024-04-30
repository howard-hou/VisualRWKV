from src.rwkv_tokenizer import TRIE_TOKENIZER
from src.dataset import preprocess, IGNORE_INDEX
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
    ctx_lens, valid_label_lens = [], []
    for data in tqdm(data_list):
        data_dict = preprocess(
            data["conversations"],
            tokenizer,
            has_image=('image' in data),
            ctx_len=None,
            pad_token_id=0,
            do_pad_to_max_length=False)
        
        ctx_lens.append(len(data_dict["input_ids"]))
        valid_label_lens.append(sum(data_dict["labels"] != IGNORE_INDEX))

    print("max ctx len", max(ctx_lens))
    print("min ctx len", min(ctx_lens))
    print("avg ctx len", sum(ctx_lens) / len(ctx_lens))
    print("median ctx len", sorted(ctx_lens)[len(ctx_lens) // 2])
    print("std ctx len", np.std(ctx_lens))
    print("80% ctx len", sorted(ctx_lens)[int(len(ctx_lens) * 0.80)])
    print("90% ctx len", sorted(ctx_lens)[int(len(ctx_lens) * 0.90)])
    print("95% ctx len", sorted(ctx_lens)[int(len(ctx_lens) * 0.95)])
    print("99% ctx len", sorted(ctx_lens)[int(len(ctx_lens) * 0.99)])

