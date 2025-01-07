import os
import argparse
import json
import re
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    return parser.parse_args()



def eval_single(annotation_file, result_file):
    experiment_name = Path(result_file).parent.name
    annotations = json.load(open(annotation_file))
    annotations = {ann['id']: ann for ann in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    for result in results:
        annotation = annotations[result['question_id']]
        gt_answer = annotation['conversations'][-1]['value'].replace('The category is ', '').lower()
        pred_answer = result['text'].replace('The category is ', '').lower()
        pred_list.append({
            "pred_answer": pred_answer,
            "gt_answer": gt_answer,
        })

    correct = 0
    for pred in pred_list:
        if pred['pred_answer'] == pred['gt_answer']:
            correct += 1
    accuracy = correct / len(pred_list)
    print('Task: Imagenet Experiment: {} Samples: {} Accuracy: {:.2f}%'.format(
        experiment_name, len(pred_list), accuracy * 100))


if __name__ == "__main__":
    args = get_args()
    eval_single(args.annotation_file, args.result_file)
