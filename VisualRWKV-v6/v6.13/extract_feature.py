########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    import torch
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--micro_bsz", default=8, type=int)
    parser.add_argument("--image_folder", type=str, default="images")
    parser.add_argument("--image_feature_folder", type=str)
    parser.add_argument("--vision_tower_dir",type=str)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logging.warning("CUDA is not available, using CPU... This will be very slow.")
    ########################################################################################################
    from pathlib import Path
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from src.vision import VisualFeatureExtractor
    from src.dataset import FeatureDataset
    from src.config import VISION_TOWER_CHECKPOINT_NAMES
    args.vision_tower_path = {name: Path(args.vision_tower_dir) / path for name, path in VISION_TOWER_CHECKPOINT_NAMES.items()}
    model = VisualFeatureExtractor(args).bfloat16().to(device)
    args.image_processor = model.vit.get_image_transform()

    train_data = FeatureDataset(args)
    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, 
                             persistent_workers=False, drop_last=False)
    image_feature_folder = Path(args.image_feature_folder)
    image_feature_folder.mkdir(exist_ok=True, parents=True)
    # extract feature, save to disk
    for batch in tqdm(data_loader, desc="Extracting feature"):
        # send batch image to device
        for key in batch['images']:
            batch['images'][key] = batch['images'][key].bfloat16().to(device)
        model.predict_step(batch)
