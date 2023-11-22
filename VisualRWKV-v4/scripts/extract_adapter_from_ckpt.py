import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_ckpt_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    return args


def extract_adapter_from_state_dict(state_dict):
    # Create a new dictionary with updated keys
    key_prefix = "adapter."
    new_state_dict = {}
    for key, value in state_dict.items():
        if key_prefix in key:
            new_state_dict[key] = value
    key_prefix = "llm.llm_proj."
    for key, value in state_dict.items():
        if key_prefix in key:
            new_state_dict[key] = value
    return new_state_dict


def main(raw_ckpt_path, output_path):
    ckpt = torch.load(raw_ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    new_state_dict = extract_adapter_from_state_dict(state_dict)
    ckpt["state_dict"] = new_state_dict
    torch.save(ckpt, output_path)


if __name__ == "__main__":
    args = parse_args()
    main(args.raw_ckpt_path, args.output_path)
