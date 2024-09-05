########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
from transformers import CLIPVisionModel
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from einops import rearrange
from fla.ops.rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6, native_recurrent_rwkv6

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam
from .dataset import IGNORE_INDEX, IMAGE_TOKEN_INDEX

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
    r = rearrange(r, 'b l (h d) -> b h l d', h = H)
    k = rearrange(k, 'b l (h d) -> b h l d', h = H)
    v = rearrange(v, 'b l (h d) -> b h l d', h = H)
    w = rearrange(-torch.exp(w), 'b l (h d) -> b h l d', h = H)
    o, state = native_recurrent_rwkv6(r, k, v, w, u=u, scale=1., initial_state=s, output_final_state=True)
    x = rearrange(o, 'b h l d -> b l (h d)')
    return x, state

########################################################################################################

class RWKV_Tmix_x060_state(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w

    @MyFunction
    def jit_func_2(self, x, g, state):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x, state

    def forward(self, x, state):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.jit_func(x)
        x, state = RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u=self.time_faaaa, s=state)

        return self.jit_func_2(x, g, state)

########################################################################################################

class RWKV_CMix_x060(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv
    
########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x060_state(args, layer_id)
        self.ffn = RWKV_CMix_x060(args, layer_id)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)
        
    def forward(self, x, state):
        if self.layer_id == 0:
            x = self.ln0(x)

        xx, state = self.att(self.ln1(x), state)
        x = x + xx
        x = x + self.ffn(self.ln2(x))

        return x, state


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optim_groups = [{"params": trainable_params, "weight_decay": self.args.weight_decay}]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False



class VisualRWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rwkv = RWKV(args)
        if len(args.load_model) > 0:
            self.load_rwkv_from_pretrained(args.load_model)
        self.vit = CLIPVisionModel.from_pretrained(args.vision_tower_name)
        self.vit.requires_grad_(False)
        self.proj = nn.Linear(self.vit.config.hidden_size, args.n_embd, bias=False)
        self.n_head = args.dim_att // args.head_size_a

    def load_rwkv_from_pretrained(self, path):
        load_msg = self.rwkv.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        rank_zero_info(f"Loaded pretrained RWKV from {path}")
        rank_zero_info(load_msg)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
    
    def freeze_rwkv(self, num_layers_to_freeze=0):
        # freeze all layers including embedding and lm head
        if num_layers_to_freeze == self.args.n_layer:
            self.rwkv.requires_grad_(False)
        # otherwise, freeze only the first num_layers_to_freeze layers
        for i, block in enumerate(self.rwkv.blocks):
            if i < num_layers_to_freeze:
                for p in block.parameters():
                    p.requires_grad_(False)
            else:
                for p in block.parameters():
                    p.requires_grad_(True)
        # freeze embedding if num_layers_to_freeze != 0
        if num_layers_to_freeze == 0:
            self.rwkv.emb.requires_grad_(True)
        else:
            self.rwkv.emb.requires_grad_(False)

    def enable_state_tuning(self):
        # fine-tune time state all the time
        for block in self.rwkv.blocks:
            for p in block.att.time_state.parameters():
                p.requires_grad_(True)

    def freeze_proj(self):
        self.proj.requires_grad_(False)

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        name_of_trainable_params = [n for n, p in self.named_parameters() if p.requires_grad]
        rank_zero_info(f"Name of trainable parameters in optimizers: {name_of_trainable_params}")
        rank_zero_info(f"Number of trainable parameters in optimizers: {len(trainable_params)}")
        optim_groups = [{"params": trainable_params, "weight_decay": self.args.weight_decay}]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)

    def forward(self, samples):
        x, targets, image_features = self.preparing_embedding(samples)
        logits = self.unidirectional_forward(x, x_emb=image_features)
        return logits, targets

    def unidirectional_forward(self, x, x_emb):
        args = self.args
        if args.dropout > 0:
            x = self.rwkv.drop0(x)

        init_states = torch.zeros(args.n_layer, x.size(0), self.n_head, args.head_size_a, args.head_size_a, 
                                  device=x.device, dtype=torch.float32).requires_grad_(True)
        for i, block in enumerate(self.rwkv.blocks):
            if args.grad_cp == 1:
                x_emb, state = deepspeed.checkpointing.checkpoint(block, x_emb, None)
                x, _ = deepspeed.checkpointing.checkpoint(block, x, state)
            else:
                x_emb, state = block(x_emb, init_states[i])
                x, _ = block(x, state)
        
        x = self.rwkv.ln_out(x)
        x = self.rwkv.head(x)
        return x
    
    def training_step(self, batch, batch_idx):
        logits, targets = self(batch)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        # calculate valid length for each sample
        valid_lengths = (shift_labels != IGNORE_INDEX).sum(1) # [B, T] -> [B]
        # if valid length is 0, set it to 1, to avoid division by zero
        valid_lengths = torch.max(valid_lengths, torch.ones_like(valid_lengths))
        # calculate loss， loss of IGNORE_INDEX will be set to 0
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1),
                               ignore_index=IGNORE_INDEX,
                               reduction='none')
        # Average the loss by valid label length
        loss = loss.view(shift_labels.size()).sum(1) / valid_lengths # [B*T] -> [B, T] -> [B]
        loss = loss.mean() # average over batch
        return L2Wrap.apply(loss, logits)
    
    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all

    def print_all_grad(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    print(name, param.grad.sum().item())
    
    def encode_images(self, images):
        B, N, C, H, W = images.shape
        images = images.view(B*N, C, H, W)
        image_features = self.vit(images).last_hidden_state
        L, D = image_features.shape[1], image_features.shape[2]
        # rerange [B*N, L, D] -> [B, N, L, D]
        image_features = image_features.view(B, N, L, D)[:, 0, :, :]
        image_features = self.grid_pooling(image_features)
        return self.proj(image_features)
    
    def grid_pooling(self, image_features):
        cls_features = image_features[:, 0:1, :]
        image_features = image_features[:, 1:, :] #drop cls token
        if self.args.grid_size == -1: # no grid pooling
            return torch.cat((image_features, cls_features), dim=1)
        if self.args.grid_size == 0: # take cls token
            return cls_features
        if self.args.grid_size == 1: # global avg pooling
            return torch.cat((image_features.mean(dim=1, keepdim=True), cls_features), dim=1)
        B, L, D = image_features.shape
        H_or_W = int(L**0.5)
        image_features = image_features.view(B, H_or_W, H_or_W, D)
        grid_stride = H_or_W // self.args.grid_size
        image_features = F.avg_pool2d(image_features.permute(0, 3, 1, 2), 
                                      padding=0,
                                      kernel_size=grid_stride, 
                                      stride=grid_stride)
        image_features = image_features.permute(0, 2, 3, 1).view(B, -1, D)
        return torch.cat((image_features, cls_features), dim=1)

    def get_max_image_token_indice(self, samples):
        max_image_token_indice = 0
        for cur_input_ids in samples["input_ids"]:
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 1:
                image_token_indice = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0][0]
                max_image_token_indice = max(max_image_token_indice, image_token_indice)
        return max_image_token_indice
    
    def truncate_input(self, new_input_embeds, new_labels):
        # prioritize retaining the labels at the beginning
        # if there are no valid labels at the beginning, retain the labels from the end
        truncated_input_embeds = []
        truncated_labels = []
        for x, y in zip(new_input_embeds, new_labels):
            valid_labels = [i for i in y[:self.args.ctx_len] if i != IGNORE_INDEX]
            if valid_labels:
                truncated_input_embeds.append(x[:self.args.ctx_len])
                truncated_labels.append(y[:self.args.ctx_len])
            else:
                truncated_input_embeds.append(x[-self.args.ctx_len:])
                truncated_labels.append(y[-self.args.ctx_len:])
        return truncated_input_embeds, truncated_labels
   
    def preparing_embedding(self, samples, truncate=True):
        device, label_dtype = samples["labels"].device, samples["labels"].dtype
        emb_dtype = samples["images"].dtype
        ### prepare image features
        image_features  = self.encode_images(samples["images"])
        ### prepare input token
        new_input_embeds, new_labels = [], []
        for idx, cur_input_ids in enumerate(samples["input_ids"]):
            cur_labels = samples["labels"][idx]
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0: # no image in this sample
                # mask image feature, set to 0
                image_features[idx] = torch.zeros_like(image_features[idx])
                cur_new_input_embeds = self.rwkv.emb(cur_input_ids)
                cur_new_labels = cur_labels
            elif num_images == 1: # only one image in this sample
                image_token_indice = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0][0]
                # right part of the image token is the instruction
                cur_new_input_embeds = self.rwkv.emb(cur_input_ids[image_token_indice+1:])
                cur_new_labels = cur_labels[image_token_indice+1:]
            else:
                raise ValueError(f"Too many images in one sample: {num_images}, should be 0 or 1.")
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        # Truncate sequences to max length as image embeddings can make the sequence longer
        # keep the first `ctx_len` tokens, to make sure instruction complete
        if truncate:
            new_input_embeds, new_labels = self.truncate_input(new_input_embeds, new_labels)
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        new_input_embeds_padded = torch.zeros((batch_size, max_len, self.args.n_embd), dtype=emb_dtype, device=device)
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=label_dtype, device=device)
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            new_input_embeds_padded[i, :cur_len] = cur_new_embed
            new_labels_padded[i, :cur_len] = cur_new_labels
        return new_input_embeds_padded, new_labels_padded, image_features
    
    def generate(self, input_ids, images, do_sample, temperature, top_p, max_new_tokens, stop_token_idx) -> list[int]:
        ''' one mode to generate, only generate one sample at a time
        # input_ids: [1, seq_len]
        # images: [1, 3, 224, 224]
        # do_sample: bool
        # temperature: float
        # top_p: float
        # max_new_tokens: int
        '''
        # prepare samples
        sampels = {"input_ids": input_ids, "images": images, "labels": torch.full_like(input_ids, IGNORE_INDEX)}
        # prepare embedding, x: [1, seq_len, n_embd]
        x, _, image_features = self.preparing_embedding(sampels, truncate=False)
        # generate
        generated = []
        for i in range(max_new_tokens):
            logits = self.bidirectional_forward(x)[:, -1, :]
            if do_sample:
                raise NotImplementedError
            else: # greedy
                # [1, vocab_size] -> [1, 1]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_token.item())
            if generated[-1] == stop_token_idx:
                break
            x = torch.cat((x, self.rwkv.emb(next_token)), dim=-2)
            x = x[:, -self.args.ctx_len:, :] # truncate
        return generated
