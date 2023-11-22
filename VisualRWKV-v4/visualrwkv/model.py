import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from .components.rwkv_tokenizer import WorldTokenizer
from .components.rwkv_rnn import RWKV
from .components.adapter import AdapterPretrain
import contextlib


class VisualRWKV(nn.Module):
    def __init__(
        self,
        adapter: AdapterPretrain,
        rwkv_model: RWKV,
        llm_proj: nn.Linear,
        model_name: str,
        max_length: int = 64,
        rnn_strategy: str = "cuda fp16",
    ):
        super(VisualRWKV, self).__init__()
        self.model = rwkv_model
        self.adapter = adapter
        self.llm_proj = llm_proj
        # config tokenizer
        if "world" in model_name:
            self.tokenizer = WorldTokenizer("rwkv_vocab_v20230424.txt")
            print("Using tokenizer: ", self.tokenizer)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = "<|padding|>"
            self.tokenizer.bos_token = self.tokenizer.eos_token
            print("Using tokenizer: ", self.tokenizer)
        # begin of image token and end of image token
        self.boi_token_ids = self.tokenizer("image:").input_ids
        self.eoi_token_ids = self.tokenizer("\n").input_ids
        # begin of response token
        self.bor_token = "response:"
        self.bor_token_ids = self.tokenizer(self.bor_token).input_ids
        # end of response token
        self.eor_token = "\n\n"
        self.eor_token_ids = self.tokenizer(self.eor_token).input_ids
        # begin of instruction token
        self.boinstr_token = "instruction:"
        self.boinstr_token_ids = self.tokenizer(self.boinstr_token).input_ids
        # end of instruction token
        self.eoinstr_token = "\n"
        self.eoinstr_token_ids = self.tokenizer(self.eoinstr_token).input_ids
        self.max_length = max_length
        self.rnn_strategy = rnn_strategy
        #
        self.freeze_llm()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def freeze_llm(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def tokenize_response(self, response):
        # response is on the right side of the image, so it should be padded to the right
        # \n\n is stop token
        resp_inputs = [self.bor_token + r + self.eor_token for r in response]
        self.tokenizer.padding_side = "right"
        resp_tokens = self.tokenizer(
            resp_inputs,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return resp_tokens

    def tokenize_instruction(self, instruction):
        # response is on the left side of the image, so it should be padded to the left
        instr_inputs = [
            self.boinstr_token + i + self.eoinstr_token for i in instruction
        ]
        self.tokenizer.padding_side = "left"
        instr_tokens = self.tokenizer(
            instr_inputs,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return instr_tokens

    def process_image(self, image_embs):
        img_device = image_embs.device
        emb_device = self.model.w["emb.weight"].device
        batch_size = image_embs.shape[0]
        image_embs_llm = self.llm_proj(image_embs)
        # begin of image token id
        boi_token_ids = torch.zeros(
            batch_size, len(self.boi_token_ids), dtype=torch.long, device=emb_device
        )
        for i in range(len(self.boi_token_ids)):
            boi_token_ids[:, i] = self.boi_token_ids[i]
        boi_embs = self.model.w["emb.weight"][boi_token_ids].to(img_device)
        # end of image token id
        eoi_token_ids = torch.zeros(
            batch_size, len(self.eoi_token_ids), dtype=torch.long, device=emb_device
        )
        for i in range(len(self.eoi_token_ids)):
            eoi_token_ids[:, i] = self.eoi_token_ids[i]
        eoi_embs = self.model.w["emb.weight"][eoi_token_ids].to(img_device)
        # concatenate
        image_embs_llm = torch.cat([boi_embs, image_embs_llm, eoi_embs], dim=1)
        return image_embs_llm

    @torch.no_grad()
    def generate(self, image_embs, instruction=None, max_new_tokens=25):
        device = image_embs.device
        image_embs_llm = self.process_image(image_embs)
        #
        inputs_embeds = image_embs_llm
        # add instruction if provided
        if instruction is not None:
            instr_tokens = self.tokenize_instruction(instruction)
            instruction_embs = self.model.w["emb.weight"][instr_tokens.input_ids]
            instruction_embs = instruction_embs.to(device)
            # instruction first, then image embeddings match the pretraining pattern
            inputs_embeds = torch.cat([instruction_embs, inputs_embeds], dim=1)

        outputs = self.rnn_generate(inputs_embeds, max_new_tokens=max_new_tokens)

        return outputs

    def cast_to_rnn_strategy(self, x):
        if "bf16" in self.rnn_strategy:
            return x.bfloat16()
        elif "fp16" in self.rnn_strategy:
            return x.half()
        elif "fp32" in self.rnn_strategy:
            return x.float()
        else:
            raise NotImplementedError(f"Unsupported rnn_strategy: {self.rnn_strategy}")

    def rnn_generate(self, inputs_embeds, max_new_tokens=10):
        # use rnn mode to generate
        batch_size, _, n_embd = inputs_embeds.shape
        device = inputs_embeds.device
        # cast inputs_embeds to rnn_strategy
        inputs_embeds = self.cast_to_rnn_strategy(inputs_embeds)
        # important: normalize inputs_embeds by ln0
        inputs_embeds = F.layer_norm(
            inputs_embeds,
            (n_embd,),
            weight=self.model.w["blocks.0.ln0.weight"],
            bias=self.model.w["blocks.0.ln0.bias"],
        )
        # add bor token
        bor_token_ids = torch.zeros(
            batch_size,
            len(self.bor_token_ids),
            dtype=torch.long,
            device=self.model.w["emb.weight"].device,
        )
        for i in range(len(self.bor_token_ids)):
            bor_token_ids[:, i] = self.bor_token_ids[i]
        bor_token_embs = self.model.w["emb.weight"][bor_token_ids].to(device)
        inputs_embeds = torch.cat([inputs_embeds, bor_token_embs], dim=1)
        #
        outputs = torch.zeros(
            batch_size,
            max_new_tokens,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        for i, sample_emb in enumerate(inputs_embeds):
            outputs[i] = self.rnn_generate_one_sample(sample_emb, max_new_tokens)
        return outputs

    def rnn_generate_one_sample(self, sample_emb, max_new_tokens=10):
        """generate sequence from sample_emb using greedy search"""
        #
        next_token_logit, state = self.model(embs=sample_emb, state=None)
        next_token = torch.argmax(next_token_logit, dim=-1)
        next_token_emb = self.model.w["emb.weight"][next_token].unsqueeze(0)
        output = torch.zeros(max_new_tokens, dtype=torch.long, device=next_token.device)
        output[0] = next_token
        for i in range(1, max_new_tokens):
            next_token_logit, state = self.model(embs=next_token_emb, state=state)
            next_token = torch.argmax(next_token_logit, dim=-1)
            next_token_emb = self.model.w["emb.weight"][next_token].unsqueeze(0)
            output[i] = next_token
        return output
