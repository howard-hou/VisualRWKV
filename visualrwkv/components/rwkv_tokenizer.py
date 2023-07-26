########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import os
from typing import Any
import torch
from dataclasses import dataclass


class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to: list
    values: set

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while fr != None:
            if fr.ch != None:
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>" % (ret[::-1], self.values)

    def add(self, key: bytes, idx: int = 0, val=None):
        if idx == len(key):
            if val is None:
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int = 0):
        u: TRIE = self
        ch: int = key[idx]

        while u.to[ch] is not None:
            u = u.to[ch]
            idx += 1
            if u.values:
                ret = idx, u, u.values
            if idx == len(key):
                break
            ch = key[idx]
        return ret


class TRIE_TOKENIZER:
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = []  # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[: l.index(" ")])
            x = eval(l[l.index(" ") : l.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(" ") :])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encodeBytes(self, src: bytes):
        idx: int = 0
        tokens = []
        while idx < len(src):
            _idx: int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert idx != _idx
            _, token = next(iter(values))
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b"".join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        try:
            return self.decodeBytes(tokens).decode("utf-8")
        except:
            return "\ufffd"  # bad utf-8

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode("utf-8")
            except:
                pass
            print(f"{repr(s)}{i}", end=" ")
        print()


class WorldTokenizerOutput:
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def to(self, device):
        return WorldTokenizerOutput(input_ids=self.input_ids.to(device))


class WorldTokenizer:
    def __init__(self, file_name):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.tokenizer = TRIE_TOKENIZER(file_dir + f"/{file_name}")
        self.padding_side = "right"
        self.eos_token = "\n\n"
        self.pad_token_id = 0

    def encode_batch(self, inputs):
        # inputs: list of strings
        return [self.tokenizer.encode(s) for s in inputs]

    def decode(self, token_list):
        all_tokens = []
        out_str = ""
        out_last = 0
        for i in range(len(token_list)):
            token = token_list[i]
            all_tokens += [token]
            tmp = self.tokenizer.decode(all_tokens[out_last:])
            if "\ufffd" not in tmp:  # is valid utf-8 string?
                out_str += tmp
                out_last = i + 1
        return out_str

    def batch_decode(self, inputs, skip_special_tokens=True):
        # if inputs is tensor, convert to list
        if isinstance(inputs, torch.Tensor):
            if inputs.device != "cpu":
                inputs = inputs.cpu()
            inputs = inputs.tolist()
        return [self.decode(s) for s in inputs]

    def __call__(
        self,
        inputs,
        padding="longest",
        truncation=True,
        max_length=None,
        return_tensors=None,
        **kwargs,
    ):
        is_inputs_str = isinstance(inputs, str)
        if is_inputs_str:
            inputs = [inputs]
        # inputs: list of strings
        x = self.encode_batch(inputs)
        # adjust length
        if truncation:
            if max_length is not None:
                x = [s[:max_length] for s in x]
        # padding
        if padding == "longest":
            max_length = max(map(len, x))
        else:
            raise NotImplementedError
        if self.padding_side == "right":
            x = [s + [self.pad_token_id] * (max_length - len(s)) for s in x]
        else:
            x = [[self.pad_token_id] * (max_length - len(s)) + s for s in x]

        if is_inputs_str:
            x = x[0]
        # return
        if return_tensors is None:
            # return list of lists
            return WorldTokenizerOutput(input_ids=x)
        elif return_tensors == "pt":
            return WorldTokenizerOutput(input_ids=torch.tensor(x))
        else:
            raise NotImplementedError
