from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from src.dataset import preprocess, IGNORE_INDEX, IMAGE_TOKEN_INDEX, load_data_file
from src.dataset import process_image_tokens_in_conversations, process_tokens_in_conversations
import argparse
import json
from tqdm import tqdm
import numpy as np
import copy


def truncate_labels(input_ids, labels, max_len):
    valid_labels = [l for l in labels if l != IGNORE_INDEX]
    if valid_labels:
        return input_ids[:max_len], labels[:max_len]
    else:
        return input_ids[-max_len:], labels[-max_len:]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str, default="data.json")
    parser.add_argument("--num_token_per_image", type=int, default=256)
    parser.add_argument("--max_ctx_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--debug_mode", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.debug_mode:
        print("arguments:",args)
    tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")
    data_list = load_data_file(args.data_file)
    ctx_lens, valid_label_lens, num_conv_rounds = [], [], []
    for data in tqdm(data_list):
        conversations = process_tokens_in_conversations(
                copy.deepcopy(data["conversations"]))

        data_dict = preprocess(
            conversations,
            tokenizer,
            ctx_len=None,
            pad_token_id=0,
            num_token_per_image=args.num_token_per_image,
            do_pad_to_max_length=False)

        data_dict["input_ids"] = data_dict["input_ids"].tolist()
        data_dict["labels"] = data_dict["labels"].tolist()
        valid_labels = [l for l in data_dict["labels"] if l != IGNORE_INDEX]
        if args.debug_mode and len(valid_labels) == 0:
            debug_msg = dict(
                id=data["id"],
                image=data.get("image", None),
                ctx_len = len(data_dict["input_ids"]),
                valid_label_len = len(valid_labels),
                num_rounds = len(conversations),
                input_text = data_dict["input_text"],
            )
            print(json.dumps(debug_msg))

        ctx_lens.append(len(data_dict["input_ids"]))
        valid_label_lens.append(len(valid_labels))
        num_conv_rounds.append(len(conversations) // 2)

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
    # print number of conversation rounds
    print("max num rounds", max(num_conv_rounds))
    print("min num rounds", min(num_conv_rounds))
    print("avg num rounds", round(sum(num_conv_rounds) / len(num_conv_rounds), 3))
    print("median num rounds", sorted(num_conv_rounds)[len(num_conv_rounds) // 2])
    print("std num rounds", round(np.std(num_conv_rounds), 3))
    print("80% num rounds", sorted(num_conv_rounds)[int(len(num_conv_rounds) * 0.80)])
    print("90% num rounds", sorted(num_conv_rounds)[int(len(num_conv_rounds) * 0.90)])
    print("95% num rounds", sorted(num_conv_rounds)[int(len(num_conv_rounds) * 0.95)])
    print("99% num rounds", sorted(num_conv_rounds)[int(len(num_conv_rounds) * 0.99)])
    #
    if args.batch_size is not None:
        # convert to batch 
        batch_valid_label_lens = [valid_label_lens[i:i+args.batch_size] for i in range(0, len(valid_label_lens), args.batch_size)]
        print(batch_valid_label_lens[:10])
        # sum over batch
        batch_valid_label_lens = [sum(batch) for batch in batch_valid_label_lens]
        print(f"batch size: {args.batch_size}")
        print("max batch valid label", max(batch_valid_label_lens))
        print("min batch valid label len", min(batch_valid_label_lens))
        print("avg batch valid label len", round(sum(batch_valid_label_lens) / len(batch_valid_label_lens), 3))
        print("std batch valid label len", round(np.std(batch_valid_label_lens), 3))
        print("median batch valid label len", sorted(batch_valid_label_lens)[len(batch_valid_label_lens) // 2])