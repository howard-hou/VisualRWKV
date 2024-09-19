import argparse
import json
import os

from openai import OpenAI
from tqdm import tqdm

client = OpenAI()


def get_eval(content: str, max_tokens: int):
    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=[{
            'role': 'system',
            'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
        }, {
            'role': 'user',
            'content': content,
        }],
        temperature=0.2,  # TODO: figure out which temperature is best for evaluation
        max_tokens=max_tokens,
    )

    return completion.choices[0].message


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-b', '--badcase')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    badcase_list = json.load(open(os.path.expanduser(args.badcase)))

    output = []
    for badcase in tqdm(badcase_list):
        prompt = 'I will provide a question, a gold answer, and a predicted answer. Based on the question and the gold answer, determine if the predicted answer is correct. Answer only with "yes" or "no."'
        content = (f'[Question]\n{badcase["questionText"]}\n'
            f'[goldAnswer]\n{badcase["goldAnswer"]}\n'
            f'[predictedAnswer]\n{badcase["predictedAnswer"]}\n'
            f'[Instruction]\n{prompt}\n')
        review = get_eval(content, args.max_tokens)
        o = badcase.copy()
        o['gpt_review'] = review.content.lower()
        output.append(o)
    # json dump the output
    json.dump(output, open(args.output, 'w'), indent=2)