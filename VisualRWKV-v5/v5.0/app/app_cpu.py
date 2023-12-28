import gradio as gr
import os, gc
from datetime import datetime
from huggingface_hub import hf_hub_download

ctx_limit = 3500
title = "rwkv1b5-vitl336p14-577token_mix665k_rwkv"

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # if '1' then use CUDA kernel for seq mode (much faster)

from rwkv.model import RWKV
model_path = hf_hub_download(repo_id="howard-hou/visualrwkv-5", filename=f"{title}.pth")
model = RWKV(model=model_path, strategy='cpu fp32')
from rwkv.utils import PIPELINE, PIPELINE_ARGS
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

##########################################################################


def generate_prompt(instruction, input=""):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    input = input.strip().replace('\r\n','\n').replace('\n\n','\n')
    if input:
        return f"""Instruction: {instruction}

Input: {input}

Response:"""
    else:
        return f"""User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: {instruction}

Assistant:"""

def evaluate(
    ctx,
    token_count=200,
    temperature=1.0,
    top_p=0.7,
    presencePenalty = 0.1,
    countPenalty = 0.1,
):
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0]) # stop generation whenever you see any token here
    ctx = ctx.strip()
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = None
    for i in range(int(token_count)):
        out, state = model.forward(pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token], state)
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        all_tokens += [token]
        for xxx in occurrence:
            occurrence[xxx] *= 0.996        
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        
        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            out_str += tmp
            yield out_str.strip()
            out_last = i + 1

    del out
    del state
    gc.collect()
    yield out_str.strip()

examples = [
    ["Assistant: Sure! Here is a very detailed plan to create flying pigs:", 333, 1, 0.3, 0, 1],
    ["Assistant: Sure! Here are some ideas for FTL drive:", 333, 1, 0.3, 0, 1],
    ["A few light taps upon the pane made her turn to the window. It had begun to snow again.", 333, 1, 0.3, 0, 1],
    [generate_prompt("Écrivez un programme Python pour miner 1 Bitcoin, avec des commentaires."), 333, 1, 0.3, 0, 1],
    [generate_prompt("東京で訪れるべき素晴らしい場所とその紹介をいくつか挙げてください。"), 333, 1, 0.3, 0, 1],
    [generate_prompt("Write a story using the following information.", "A man named Alex chops a tree down."), 333, 1, 0.3, 0, 1],
    ["Assistant: Here is a very detailed plan to kill all mosquitoes:", 333, 1, 0.3, 0, 1],
    ['''Edward: I am Edward Elric from fullmetal alchemist. I am in the world of full metal alchemist and know nothing of the real world.

Player: Hello Edward. What have you been up to recently?

Edward:''', 333, 1, 0.3, 0, 1],
    [generate_prompt("写一篇关于水利工程的流体力学模型的论文，需要详细全面。"), 333, 1, 0.3, 0, 1],
    ['''“当然可以，大宇宙不会因为这五公斤就不坍缩了。”关一帆说，他还有一个没说出来的想法：也许大宇宙真的会因为相差一个原子的质量而由封闭转为开放。大自然的精巧有时超出想象，比如生命的诞生，就需要各项宇宙参数在几亿亿分之一精度上的精确配合。但程心仍然可以留下她的生态球，因为在那无数文明创造的无数小宇宙中，肯定有相当一部分不响应回归运动的号召，所以，大宇宙最终被夺走的质量至少有几亿吨，甚至可能是几亿亿亿吨。
但愿大宇宙能够忽略这个误差。
程心和关一帆进入了飞船，智子最后也进来了。她早就不再穿那身华丽的和服了，她现在身着迷彩服，再次成为一名轻捷精悍的战士，她的身上佩带着许多武器和生存装备，最引人注目的是那把插在背后的武士刀。
“放心，我在，你们就在！”智子对两位人类朋友说。
聚变发动机启动了，推进器发出幽幽的蓝光，飞船缓缓地穿过了宇宙之门。
小宇宙中只剩下漂流瓶和生态球。漂流瓶隐没于黑暗里，在一千米见方的宇宙中，只有生态球里的小太阳发出一点光芒。在这个小小的生命世界中，几只清澈的水球在零重力环境中静静地飘浮着，有一条小鱼从一只水球中蹦出，跃入另一只水球，轻盈地穿游于绿藻之间。在一小块陆地上的草丛中，有一滴露珠从一片草叶上脱离，旋转着飘起，向太空中折射出一缕晶莹的阳光。''', 333, 1, 0.3, 0, 1],    
]

##########################################################################

with gr.Blocks(title=title) as demo:
    gr.HTML(f"<div style=\"text-align: center;\">\n<h1>VisualRWKV-5.0 - {title}</h1>\n</div>")
    with gr.Tab("Raw Generation"):
        gr.Markdown(f"This is [RWKV-5 World v2](https://huggingface.co/BlinkDL/rwkv-5-world) with 1.5B params - a 100% attention-free RNN [RWKV-LM](https://github.com/BlinkDL/RWKV-LM). Supports all 100+ world languages and code. And we have [200+ Github RWKV projects](https://github.com/search?o=desc&p=1&q=rwkv&s=updated&type=Repositories). *** Please try examples first (bottom of page) *** (edit them to use your question). Demo limited to ctxlen {ctx_limit}.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=2, label="Prompt", value="Assistant: Sure! Here is a very detailed plan to create flying pigs:")
                token_count = gr.Slider(10, 333, label="Max Tokens", step=10, value=333)
                temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
                presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=0)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=1)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                output = gr.Textbox(label="Output", lines=5)
        data = gr.Dataset(components=[prompt, token_count, temperature, top_p, presence_penalty, count_penalty], 
                          samples=examples, label="Example Instructions", 
                          headers=["Prompt", "Max Tokens", "Temperature", "Top P", "Presence Penalty", "Count Penalty"])
        submit.click(evaluate, [prompt, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
        clear.click(lambda: None, [], [output])
        data.click(lambda x: x, [data], [prompt, token_count, temperature, top_p, presence_penalty, count_penalty])

demo.queue(concurrency_count=1, max_size=10)
demo.launch(share=False)