import json
import argparse
from datasets import load_from_disk
from pathlib import Path


subtitle_template = '''
{Frames}
This video's subtitles are listed below:
{Subtitles} 
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option. 
{Question}
Answer the question using a single word or phrase.
'''

subtitle_free_template = '''
{Frames}
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option. 
{Question}
Answer the question using a single word or phrase.
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='Path to the input dataset')
    parser.add_argument('data_folder', type=str, help='Path to the data folder')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of frames to use')
    parser.add_argument('--use_subtitles', action='store_true', help='Use subtitles')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = load_from_disk(args.dataset_path)['test']
    data_folder = Path(args.data_folder)
    vid2folder = {f.stem: f for f in data_folder.iterdir() if f.is_dir()}
    output_path = Path(args.dataset_path).name + f'_{args.num_frames}frames' + ('_subtitles' if args.use_subtitles else '') + '.jsonl'
    w = open(output_path, 'w')
    for i, example in enumerate(dataset):
        vid = example['videoID']
        frames = ' '.join(['<image>'] * args.num_frames)
        question = example['question'] + '\n' + '\n'.join(example['options'])
        if args.use_subtitles:
            subtitles = open(vid2folder[vid] / 'subtitles.txt').read()
            instruction = subtitle_template.format(Frames=frames, Subtitles=subtitles, Question=question).strip()
        else:
            instruction = subtitle_free_template.format(Frames=frames, Question=question).strip()
        out = dict(question_id=example['question_id'], video=vid, text=instruction)
        out['metadata'] = {k: example[k] for k in example}
        w.write(json.dumps(out, ensure_ascii=False) + '\n')
    w.close()

        