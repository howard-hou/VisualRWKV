from src.rwkv_tokenizer import TRIE_TOKENIZER
from src.dataset import preprocess, IGNORE_INDEX
import argparse
import json
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str, default="data.json")
    parser.add_argument("--debug_mode", action="store_true")
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
        
        valid_labels = [l for l in data_dict["labels"].tolist() if l != IGNORE_INDEX]
        if args.debug_mode:
            print(data_dict["input_text"])
            print(tokenizer.decode(valid_labels))

        ctx_lens.append(len(data_dict["input_ids"]))
        valid_label_lens.append(len(valid_labels))

    print(f"file name: {args.data_file}")
    print("max ctx len", max(ctx_lens))
    print("min ctx len", min(ctx_lens))
    print("avg ctx len", round(sum(ctx_lens) / len(ctx_lens), 3))
    print("median ctx len", sorted(ctx_lens)[len(ctx_lens) // 2])
    print("std ctx len", round(np.std(ctx_lens), 3))
    print("80% ctx len", sorted(ctx_lens)[int(len(ctx_lens) * 0.80)])
    print("90% ctx len", sorted(ctx_lens)[int(len(ctx_lens) * 0.90)])
    print("95% ctx len", sorted(ctx_lens)[int(len(ctx_lens) * 0.95)])
    print("99% ctx len", sorted(ctx_lens)[int(len(ctx_lens) * 0.99)])
    # print valid label length
    print("max valid label len", max(valid_label_lens))
    print("min valid label len", min(valid_label_lens))
    print("avg valid label len", round(sum(valid_label_lens) / len(valid_label_lens), 3))
    print("median valid label len", sorted(valid_label_lens)[len(valid_label_lens) // 2])
    print("std valid label len", round(np.std(valid_label_lens), 3))
    print("80% valid label len", sorted(valid_label_lens)[int(len(valid_label_lens) * 0.80)])
    print("90% valid label len", sorted(valid_label_lens)[int(len(valid_label_lens) * 0.90)])
    print("95% valid label len", sorted(valid_label_lens)[int(len(valid_label_lens) * 0.95)])
    print("99% valid label len", sorted(valid_label_lens)[int(len(valid_label_lens) * 0.99)])