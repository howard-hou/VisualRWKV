import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_file', type=str)
    parser.add_argument('result_file', type=str)
    return parser.parse_args()


def eval_single(annotation_file, result_file):
    annotations = [json.loads(line) for line in open(annotation_file)]
    annotations = {annotation['question_id']: annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    for result in results:
        annotation = annotations[result['question_id']]
        answers = [ans.strip().lower() for ans in annotation['answers']]
        pred_list.append({
            "pred_answer": result['text'].strip().lower(),
            "gt_answers": answers,
        })

    correct = 0
    for pred in pred_list:
        if pred['pred_answer'] in pred['gt_answers']:
            correct += 1
    acc = correct / len(pred_list)
    print("result_file: {}, acc: {:.2f}%".format(result_file, acc * 100))


if __name__ == "__main__":
    args = get_args()

    eval_single(args.annotation_file, args.result_file)