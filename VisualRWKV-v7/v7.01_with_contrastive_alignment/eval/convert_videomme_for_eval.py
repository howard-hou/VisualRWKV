import os
import json
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--dst", type=str)
args = parser.parse_args()

group_by_video = defaultdict(list)
for line in open(args.src):
    line = json.loads(line)
    vid = line['metadata']['video_id']
    group_by_video[vid].append(line)

def video_group_to_dict(group):
    questions = []
    for item in group:
        question_dict = {
            "question_id": item['question_id'],
            "task_type": item['metadata']['task_type'],
            "question": item['metadata']['question'],
            "options": item['metadata']['options'],
            "answer": item['metadata']['answer'],
            "response": item['text']
        }
        questions.append(question_dict)
    return {
        "video_id": group[0]['metadata']['video_id'], 
        "duration": group[0]['metadata']['duration'], 
        "domain": group[0]['metadata']['domain'], 
        "sub_category": group[0]['metadata']['sub_category'], 
        "questions": questions
    }

all_answers = []
for vid, group in group_by_video.items():
    all_answers.append(video_group_to_dict(group))


with open(args.dst, 'w') as f:
    json.dump(all_answers, f, ensure_ascii=False, indent=2)