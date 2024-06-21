# use to scan the hyper parameter of image_position: first, middle and last
import argparse
import subprocess
from pathlib import Path

def generate_cmd(args, task, value, script_dir):
    arg_list = [args.model_path, args.ctx_len, args.grid_size, args.n_embd, args.n_layer, args.eval_dir, args.vision_tower_path, args.image_position]
    arg_name_to_index = {"model_path": 0, "ctx_len": 1, "grid_size": 2, "n_embd": 3, "n_layer": 4, "eval_dir": 5, "vision_tower_path": 6, "image_position": 7}
    arg_list[arg_name_to_index[args.hyperparameter]] = value
    cmd = f"bash {script_dir}/eval_{task}.sh {' '.join(map(str, arg_list))}"
    return cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--ctx_len", type=int)
    parser.add_argument("--grid_size", type=int) # -1 for no grid, 0 for cls token, 1 for global avg, 8 for 64 tokens
    parser.add_argument("--n_embd", type=int)
    parser.add_argument("--n_layer", type=int)
    parser.add_argument("--eval_dir", type=str)
    parser.add_argument("--vision_tower_path", type=str)
    parser.add_argument("--image_position", type=str, default="first")
    parser.add_argument("--task_names", type=str, help="comma separated list of task names")
    parser.add_argument("--hyperparameter", type=str, help="hyperparameter to scan")
    parser.add_argument("--hyperparameter_values", type=str, help="comma separated list of hyperparameter values")

    args = parser.parse_args()
    print('args:', args)
    # get script directory
    script_dir = Path(__file__).parent

    task_names = args.task_names.split(',')
    hyperparameter_values = args.hyperparameter_values.split(',')
    for task in task_names:
        for value in hyperparameter_values:
            print(f"Running task: {task} with hyperparameter: {args.hyperparameter} and value: {value}")
            cmd = generate_cmd(args, task, value, script_dir)
            print('cmd:', cmd)
            # run cmd
            subprocess.run(cmd, shell=True)
            print(f"Completed task: {task} with hyperparameter: {args.hyperparameter} and value: {value}")

