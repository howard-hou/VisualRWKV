import os
import json
import argparse
import pandas as pd
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--upload-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    upload_dir = Path(args.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_table(args.annotation_file)

    cur_df = df.copy()
    cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
    cur_df.insert(6, 'prediction', None)
    for pred in open(os.path.join(args.result_dir, args.experiment, 'merge.jsonl')):
        pred = json.loads(pred)
        cur_df.loc[df['index'] == pred['question_id'], 'prediction'] = pred['text']

    cur_df.to_excel(upload_dir / f"{args.experiment}.xlsx", index=False, engine='openpyxl')
