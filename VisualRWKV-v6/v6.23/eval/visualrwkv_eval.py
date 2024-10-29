import argparse
import json
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_file', type=str)
    parser.add_argument('result_file', type=str)
    # print badcase
    parser.add_argument('--print_badcase', action='store_true')
    return parser.parse_args()


def eval_single(annotation_file, result_file):
    annotations = [json.loads(line) for line in open(annotation_file)]
    annotations = {annotation['question_id']: annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    for result in results:
        annotation = annotations[result['question_id']]
        answers = [ans.strip().lower().strip('.') for ans in annotation['answers']]
        pred_list.append({
            "pred_answer": result['text'].strip().lower().strip('.'),
            "gt_answers": answers,
        })

    correct = 0
    badcase_list = []
    for pred in pred_list:
        pred_answer = pred['pred_answer']
        if pred_answer in pred['gt_answers']:
            correct += 1
        else:
            badcase_list.append(dict(pred_answer=pred_answer, gt_answers=pred['gt_answers']))
    acc = correct / len(pred_list)
    print("result_file: {}, acc: {:.2f}%".format(result_file, acc * 100))
    if args.print_badcase:
        # sample 10 from badcase_list
        badcase_sampled = random.sample(badcase_list, min(10, len(badcase_list)))
        for i, badcase in enumerate(badcase_sampled):
            print("badcase {}: pred: {}, gt: {}".format(i, badcase['pred_answer'], badcase['gt_answers']))


if __name__ == "__main__":
    args = get_args()

    eval_single(args.annotation_file, args.result_file)