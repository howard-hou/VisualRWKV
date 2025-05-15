import datasets
import sys
from pathlib import Path
import json
import random
from tqdm import tqdm

wiki_ss_corpus_instructions = [
    "Please extract all the text from this Wikipedia page image.",
    "Convert this image of a Wikipedia page into plain text, exactly as shown.",
    "Transcribe the text from this Wikipedia screenshot without summarizing.",
    "Perform OCR on this image of a Wikipedia article and return the full text.",
    "Read and convert all visible text from this Wikipedia page image.",
    "Extract every word from the Wikipedia article image as-is.",
    "Accurately convert this Wikipedia page image into its textual content.",
    "Turn this image of a Wikipedia page into raw text line by line.",
    "Get all the textual information from this screenshot of a Wikipedia page.",
    "Retrieve all visible text from this Wikipedia article image exactly.",
    "Detect and output the full text shown in this Wikipedia screenshot.",
    "Parse this Wikipedia image and return all recognized characters and text.",
    "Read the entire text content from this image of a Wikipedia page.",
    "Identify and extract all textual elements from the Wikipedia page image.",
    "Use OCR to capture all the text in this Wikipedia article image."
]


def get_sample_id(sample, ds_name, line_id):
    """
    Generates a sample ID based on the dataset name and line ID.
    """
    if 'wiki-ss-corpus' in ds_name:
        return sample['docid']
    return f"{ds_name}_{line_id}"

def get_instruction(sample, ds_name):
    """
    Generates an instruction string based on the dataset name.
    """
    if 'wiki-ss-corpus' in ds_name:
        return random.choice(wiki_ss_corpus_instructions)
    raise NotImplementedError(f"Dataset {ds_name} not supported for instruction generation.")

def get_response(sample, ds_name):
    """
    Generates a response string based on the dataset name.
    """
    if 'wiki-ss-corpus' in ds_name:
        return 'title:\n' + sample['title'] + '\n' + 'text:\n' + sample['text']
    raise NotImplementedError(f"Dataset {ds_name} not supported for response generation.")

input_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
output_dir.mkdir(exist_ok=True)
ds_name = input_path.stem
ds = datasets.load_from_disk(str(input_path))
if "train" in ds:
    ds = ds["train"]

image_output = output_dir / ds_name
image_output.mkdir(exist_ok=True)
json_output = output_dir / f"{ds_name}.jsonl"

data = []
for line_id, sample in enumerate(tqdm(ds)):
    sample_id = get_sample_id(sample, ds_name, line_id)
    image = sample['image'].convert("RGB")
    img_path = image_output / f"{sample_id}.jpg"
    image.save(img_path)
    img_relative_path = str(img_path.relative_to(output_dir))

    convs = []
    image_prifix = "<image>"
    instruction = image_prifix + "\n" + get_instruction(sample, ds_name)
    convs.append({"from": "human", "value": instruction})
    convs.append({"from": "gpt", "value": get_response(sample, ds_name).strip()})
    data.append({"id": sample_id, "image": img_relative_path, "conversations": convs})

# Save the data to a JSONL file
with open(json_output, "w") as f:
    for entry in data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")