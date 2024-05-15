from argparse import ArgumentParser
import torch
from pathlib import Path

parser = ArgumentParser()

parser.add_argument("model_path", type=str, default=None, help="path of visualrwkv model") 
parser.add_argument("output_dir", type=str, default=None, help="path of output file")

args = parser.parse_args()
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
model_name = Path(args.model_path).parent.name
state_dict = torch.load(args.model_path, map_location="cpu")
# print(state_dict.keys())
rwkv_state_dict = {}
visual_state_dict = {}
for key in state_dict:
    if key.startswith("rwkv"):
        rwkv_state_dict[key[5:]] = state_dict[key].half()
    else:
        visual_state_dict[key] = state_dict[key].half()
print("rwkv state dict has keys: ", len(rwkv_state_dict))
print("visual state dict has keys: ", len(visual_state_dict))
# save 
torch.save(rwkv_state_dict, output_dir / f"{model_name}_rwkv.pth")
torch.save(visual_state_dict, output_dir / f"{model_name}_visual.pth")
