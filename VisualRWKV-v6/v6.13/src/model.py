########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
from pathlib import Path
import numpy as np
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

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam
from .dataset import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from .vision import SamDinoSigLIPViTBackbone

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
wkv6_cuda = load(name="wkv6", sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
    
class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)

########################################################################################################

class RWKV_Tmix_x060(MyModule):
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
            if args.n_embd >= 4096:
                D_MIX_LORA = 64
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            D_DECAY_LORA = 64
            if args.n_embd >= 4096:
                D_DECAY_LORA = 128
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
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.jit_func(x)
        x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)

        return self.jit_func_2(x, g)

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

        self.att = RWKV_Tmix_x060(args, layer_id)
        self.ffn = RWKV_CMix_x060(args, layer_id)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)
        
    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


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

    def forward(self, x):
        args = self.args
        # B, T, D = x.size()
        # assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        if args.dropout > 0:
            x = self.drop0(x)

        for block in self.blocks:
            if args.grad_cp == 1:
                x = deepspeed.checkpointing.checkpoint(block, x)
            else:
                x = block(x)

        x = self.ln_out(x)

        x = self.head(x)

        return x

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits = self(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all


class MLPWithContextGating(nn.Module):
    def __init__(self, in_dim, n_embd):
        super().__init__()
        self.proj = nn.Linear(in_dim, n_embd, bias=False)
        self.n_embd = n_embd
        self.gate = nn.Linear(n_embd, n_embd, bias=False)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        # x: [B, T, D]
        x = self.proj(x)
        gating = torch.sigmoid(self.gate(x))
        return self.o_proj(x * gating)


class MLPWithMultiheadAttentionGating(nn.Module):
    def __init__(self, in_dim, n_embd):
        super().__init__()
        self.proj = nn.Linear(in_dim, n_embd, bias=False)
        self.n_embd = n_embd
        self.num_heads = n_embd // 64
        # key, query, value projections for all heads
        self.qkv_proj = nn.Linear(n_embd, n_embd * 3, bias=False)
        self.attn = nn.MultiheadAttention(n_embd, self.num_heads)
        self.gate = nn.Sigmoid()
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        # x: [B, T, D]
        x = self.proj(x)
        q, k ,v  = self.qkv_proj(x).split(self.n_embd, dim=2)
        attn_output, _ = self.attn(q, k, v)
        attn_gating = self.gate(attn_output)
        return self.o_proj(x * attn_gating)


def fuse_image_features(dino_patches, siglip_patches, sam_patches):
    '''
    fuse image features from DINO, SigLIP and SAM. Special designed for VisualRWKV-UHD.
    dino_patches: [B, N, L, D]
    siglip_patches: [B, N, L, D]
    sam_patches: [B, N, L, D]
    '''
    B, N, L, _ = dino_patches.shape
    H_or_W = int(L**0.5) # 1024 -> 32
    # concat global image features
    global_features = torch.cat([dino_patches[:, 0, :, :], siglip_patches[:, 0, :, :], sam_patches[:, 0, :, :]], dim=-1)
    # adaptive_avg_pool2d over all tiles
    output_size = H_or_W // 2 # 32 -> 16
    dino_tiles, siglip_tiles, sam_tiles = [], [], []
    for i in range(1, N):
        one_dino = dino_patches[:, i, :, :].view(B, H_or_W, H_or_W, -1).permute(0, 3, 1, 2)
        one_siglip = siglip_patches[:, i, :, :].view(B, H_or_W, H_or_W, -1).permute(0, 3, 1, 2)
        one_sam = sam_patches[:, i, :, :].view(B, H_or_W, H_or_W, -1).permute(0, 3, 1, 2)
        # adaptive_avg_pool2d
        one_dino = F.adaptive_avg_pool2d(one_dino, output_size)
        one_siglip = F.adaptive_avg_pool2d(one_siglip, output_size)
        one_sam = F.adaptive_avg_pool2d(one_sam, output_size)
        dino_tiles.append(one_dino)
        siglip_tiles.append(one_siglip)
        sam_tiles.append(one_sam)
    # 先拼接水平 (W 方向上) 部分
    dino_top = torch.cat([dino_tiles[0], dino_tiles[1]], dim=3) # 拼接左上和右上部分
    siglip_top = torch.cat([siglip_tiles[0], siglip_tiles[1]], dim=3)
    sam_top = torch.cat([sam_tiles[0], sam_tiles[1]], dim=3)
    dino_bottom = torch.cat([dino_tiles[2], dino_tiles[3]], dim=3) # 拼接左下和右下部分
    siglip_bottom = torch.cat([siglip_tiles[2], siglip_tiles[3]], dim=3)
    sam_bottom = torch.cat([sam_tiles[2], sam_tiles[3]], dim=3)
    # 再拼接垂直 (H 方向上) 部分
    dino_features = torch.cat([dino_top, dino_bottom], dim=2)
    siglip_features = torch.cat([siglip_top, siglip_bottom], dim=2)
    sam_features = torch.cat([sam_top, sam_bottom], dim=2)
    # reshape B, C, H, W to B, H*W, C
    dino_features = dino_features.permute(0, 2, 3, 1).reshape(B, -1, dino_features.shape[1])
    siglip_features = siglip_features.permute(0, 2, 3, 1).reshape(B, -1, siglip_features.shape[1])
    sam_features = sam_features.permute(0, 2, 3, 1).reshape(B, -1, sam_features.shape[1])
    # concat global image features with local features at D dimension
    image_features = torch.cat([global_features, dino_features, siglip_features, sam_features], dim=-1)
    return image_features


class VisualRWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rwkv = RWKV(args)
        if len(args.load_model) > 0:
            self.load_rwkv_from_pretrained(args.load_model)
        self.vit = SamDinoSigLIPViTBackbone(args.vision_tower_path)
        self.freeze_vit()
        if args.proj_type == "linear":
            self.proj = nn.Linear(self.vit.embed_dim, args.n_embd, bias=False)
        else:
            self.proj = MLPWithContextGating(self.vit.embed_dim, args.n_embd)

    def load_rwkv_from_pretrained(self, path):
        self.rwkv.load_state_dict(torch.load(path, map_location="cpu"))
        rank_zero_info(f"Loaded pretrained RWKV from {path}")

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def freeze_vit(self):
        self.vit.requires_grad_(False)
    
    def freeze_rwkv(self, num_layers_to_freeze):
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

    def freeze_emb(self):
        self.rwkv.emb.requires_grad_(False)

    def freeze_proj(self):
        self.proj.requires_grad_(False)

    def configure_optimizers(self):
        zero_weight_decay_group = [p for p in self.parameters() if len(p.squeeze().shape) < 2 and p.requires_grad]
        # add weight decay to len(p.squeeze().shape) >= 2
        weight_decay_group = [p for p in self.parameters() if len(p.squeeze().shape) >= 2 and p.requires_grad] 

        name_of_trainable_params = [n for n, p in self.named_parameters() if p.requires_grad]
        rank_zero_info(f"Name of trainable parameters in optimizers: {name_of_trainable_params}")
        rank_zero_info(f"Number of trainable parameters in optimizers: {len(name_of_trainable_params)}")
        optim_groups = []
        if zero_weight_decay_group:
            optim_groups += [{"params": zero_weight_decay_group, "weight_decay": 0.0}]
        if weight_decay_group:
            if self.args.weight_decay > 0:
                optim_groups += [{"params": weight_decay_group, "weight_decay": self.args.weight_decay}]
                rank_zero_info(f"Number of parameters with weight decay: {len(weight_decay_group)}, with value: {self.args.weight_decay}")
            else:
                optim_groups += [{"params": weight_decay_group, "weight_decay": 0.0}]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)

    def forward(self, samples):
        x, targets, image_features = self.preparing_embedding(samples)
        # bidirectional
        logits = self.bidirectional_forward(x)
        return logits, targets

    def bidirectional_forward(self, x, x_emb=None):
        args = self.args
        if args.dropout > 0:
            x = self.rwkv.drop0(x)

        for i, block in enumerate(self.rwkv.blocks):
            do_reverse = (i % 2 == 1)
            if do_reverse: # reverse
                x[:, self.img_start:self.img_end, :] = x[:, self.img_start:self.img_end, :].flip(1)
            
            if args.grad_cp == 1:
                x = deepspeed.checkpointing.checkpoint(block, x)
            else:
                x = block(x)
            
            if do_reverse: # reverse back
                x[:, self.img_start:self.img_end, :] = x[:, self.img_start:self.img_end, :].flip(1)

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
    
    def encode_images(self, images):
        dino_patches, siglip_patches, sam_patches = self.vit(images)
        image_features = fuse_image_features(dino_patches, siglip_patches, sam_patches)
        return self.proj(image_features)
    
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
        emb_dtype = samples["images"]['dino'].dtype
        ### prepare image features
        image_features  = self.encode_images(samples["images"]) #
        ### prepare input token
        new_input_embeds = []
        new_labels = []
        max_image_token_indice = self.get_max_image_token_indice(samples)
        self.img_start = max_image_token_indice
        self.img_end = max_image_token_indice + image_features.shape[1]
        for idx, cur_input_ids in enumerate(samples["input_ids"]):
            cur_labels = samples["labels"][idx]
            cur_new_input_ids = torch.zeros(max_image_token_indice, dtype=cur_input_ids.dtype, device=device)
            cur_new_labels = torch.full((max_image_token_indice,), IGNORE_INDEX, device=device, dtype=label_dtype)
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0: # no image in this sample
                # mask image feature, set to 0
                image_features[idx] = torch.zeros_like(image_features[idx])
            elif num_images == 1: # only one image in this sample
                image_token_indice = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0][0]
                # first text part, left paded
                cur_new_input_ids[-image_token_indice:] = cur_input_ids[:image_token_indice]
                cur_new_labels[-image_token_indice:] = cur_labels[:image_token_indice]
            else:
                raise ValueError(f"Too many images in one sample: {num_images}, should be 0 or 1.")
            # convert to list
            cur_new_input_embeds = [self.rwkv.emb(cur_new_input_ids)]
            cur_new_labels = [cur_new_labels]
            # image part
            cur_image_features = image_features[idx]
            cur_new_input_embeds.append(cur_image_features)
            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=device, dtype=label_dtype))
            # last text part
            if num_images == 1:
                cur_new_input_embeds.append(self.rwkv.emb(cur_input_ids[image_token_indice+1:]))
                cur_new_labels.append(cur_labels[image_token_indice+1:])
            else: # no image
                cur_new_input_embeds.append(self.rwkv.emb(cur_input_ids))
                cur_new_labels.append(cur_labels)
            # concat them
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

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
        # images: a dict of dino, siglip and sam features, each with shape [1, 3, H_dino, W_dino], [1, 3, H_siglip, W_siglip], [1, 3, H_sam, W_sam]
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
        generated_tokens = []
        generated_token_logits = []
        generated_token_probs = []
        for i in range(max_new_tokens):
            logits = self.bidirectional_forward(x)[:, -1, :]
            if do_sample:
                raise NotImplementedError
            else: # greedy
                # [1, vocab_size] -> [1, 1]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                next_token_logit = logits.gather(-1, next_token)
                probs = torch.softmax(logits, dim=-1)
                next_token_prob = probs.gather(-1, next_token)
            generated_tokens.append(next_token.item())
            generated_token_logits.append(next_token_logit.item())
            generated_token_probs.append(next_token_prob.item())
            if generated_tokens[-1] == stop_token_idx:
                break
            x = torch.cat((x, self.rwkv.emb(next_token)), dim=-2)
            x = x[:, -self.args.ctx_len:, :] # truncate
        return generated_tokens, generated_token_logits, generated_token_probs


class VisualFeatureExtractor(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.image_feature_folder = Path(args.image_feature_folder)
        self.vit = SamDinoSigLIPViTBackbone(args.vision_tower_path)
        self.vit.requires_grad_(False)

    def encode_images(self, images):
        '''
        encode image to get image features before projection.
        '''
        dino_patches, siglip_patches, sam_patches = self.vit(images)
        image_features = fuse_image_features(dino_patches, siglip_patches, sam_patches)
        return image_features
    
    @torch.inference_mode()
    def forward(self, samples):
        image_features = self.encode_images(samples['images'])
        return image_features
    
    def predict_step(self, batch):
        image_features = self(batch)
        # replace all image suffix with .npz
        image_feature_file_names = [f.with_suffix('.npz') for f in batch['image_file']]
        # save image features to np
        image_feature_file_paths = [self.image_feature_folder / f for f in image_feature_file_names]
        for f, features in zip(image_feature_file_paths, image_features):
            np.savez(f, features=features.cpu().numpy())