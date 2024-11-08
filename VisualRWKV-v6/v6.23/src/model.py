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
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from pytorch_lightning.strategies import DeepSpeedStrategy

if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam
from .dataset import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from .vision import SamDinoSigLIPViTBackbone
from .utils import get_cross_block_indices, compress_parameter_names

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
# Cross-Attention Layer
########################################################################################################
class CrossAttentionLayer(MyModule):
    def __init__(self, args, layer_id):
        super(CrossAttentionLayer, self).__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        self.query = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        # zero output linear
        self.output.weight.data.zero_()

    def forward(self, query, key_value):
        batch_size = query.shape[0]

        # 将查询、键和值投影到多头注意力的维度
        query = self.query(query)
        key = self.key(key_value)
        value = self.value(key_value)

        # 拆分多头
        query = query.view(batch_size, -1, self.n_head, self.head_size).transpose(1, 2)
        key = key.view(batch_size, -1, self.n_head, self.head_size).transpose(1, 2)
        value = value.view(batch_size, -1, self.n_head, self.head_size).transpose(1, 2)

        # 使用scaled_dot_product_attention计算注意力
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.head_size)

        # 投影输出
        output = self.output(attn_output)

        return output

class MLP(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.c_fc = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(args.dim_ffn, args.n_embd, bias=False)
        # zero output linear
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        return x
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


class CrossAttentionBlock(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = CrossAttentionLayer(args, layer_id)
        self.ffn = MLP(args, layer_id)
        
    def forward(self, x, image_features):
        x = x + self.att(self.ln1(x), image_features)
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


class HybridRWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.cross_blocks = nn.ModuleList([CrossAttentionBlock(args, i) for i in range(args.n_cross_layer)])
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
        raise NotImplementedError

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
        self.gate = nn.Linear(in_dim, in_dim, bias=False)
        self.o_proj = nn.Linear(in_dim, n_embd, bias=False)
        self.ln_v = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x: [B, T, D]
        gating = torch.sigmoid(self.gate(x))
        return self.ln_v(self.o_proj(x * gating))


class VisualRWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rwkv = HybridRWKV(args)
        self.vit = SamDinoSigLIPViTBackbone(args.vision_tower_path)
        self.freeze_vit()
        self.proj = self.init_proj(args)
        
    def init_proj(self, args):
        if args.proj_type == "linear":
            proj = nn.Linear(self.vit.embed_dim, args.n_embd, bias=False)
        elif args.proj_type == "mlp":
            proj = MLPWithContextGating(self.vit.embed_dim, args.n_embd)
        else:
            raise ValueError(f"Unknown projection type: {args.proj_type}")
        return proj

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
        compressed_name_of_trainable_params = compress_parameter_names(name_of_trainable_params)
        rank_zero_info(f"Name of trainable parameters in optimizers: {compressed_name_of_trainable_params}")
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
        logits = self.forward_with_image_features(x, image_features)
        return logits, targets
    
    def forward_with_image_features(self, x, image_features):
        cross_block_indices = get_cross_block_indices(len(self.rwkv.blocks), len(self.rwkv.cross_blocks))
        total_blocks = len(self.rwkv.blocks) + len(self.rwkv.cross_blocks)
        block_index, cross_block_index = 0, 0
        for i in range(total_blocks):
            if i in cross_block_indices:
                if self.args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(self.rwkv.cross_blocks[cross_block_index], x, image_features)
                else:
                    x = self.rwkv.cross_blocks[cross_block_index](x, image_features)
                cross_block_index += 1
            else:
                if self.args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(self.rwkv.blocks[block_index], x)
                else:
                    x = self.rwkv.blocks[block_index](x)
                block_index += 1

        x = self.ln_out(x)
        x = self.head(x)
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

    def adaptive_pooling(self, image_features, output_size):
        B, L, D = image_features.shape
        H_or_W = int(L**0.5)
        image_features = image_features.view(B, H_or_W, H_or_W, D).permute(0, 3, 1, 2)
        image_features = F.adaptive_avg_pool2d(image_features, output_size).view(B, D, -1).permute(0, 2, 1)
        return image_features
    
    def encode_images(self, images: dict, minibatch_size=4) -> torch.Tensor:
        '''
        mini-batch image feature extraction:
        load feature from disk, RWKV-1.6B only occupies 9GB of GPU memory, but computing feature occupies 40GB of GPU memory. 
        This is because there are many intermediate variables and caches during the feature extraction process. 
        Therefore, images are input in mini-batches, where only a portion of the image features are extracted at a time, 
        then the cache is cleared, and then they are concat together.
        '''
        # mini-batch: split images every minibatch_size images equally
        N = len(images['siglip'])
        if N <= minibatch_size:
            image_features_orig = self.vit(images).detach()
            torch.cuda.empty_cache()
        else:
            image_features = []
            for i in range(0, N, minibatch_size):
                minibatch_images = {k: v[i:i+minibatch_size] for k, v in images.items()}
                minibatch_features = self.vit(minibatch_images).detach()
                torch.cuda.empty_cache()
                image_features.append(minibatch_features)
            image_features_orig = torch.cat(image_features, dim=0)
        image_features_pooled = self.adaptive_pooling(image_features_orig, 
                                                      output_size=int(self.args.num_token_per_image ** 0.5))
        return self.proj(image_features_pooled), self.proj(image_features_orig)
    
   
    def preparing_embedding(self, samples):
        if "images" not in samples:
            return self.rwkv.emb(samples["input_ids"]), samples["labels"]
        ### prepare image features
        image_features_pooled, image_features_orig = self.encode_images(samples["images"])
        B_IMG, L_IMG, D_IMG = image_features_pooled.shape
        selected_image_features = image_features_pooled.view(-1, D_IMG)
        ### prepare input token
        input_embeds = self.rwkv.emb(samples["input_ids"])
        B, L, D = input_embeds.shape
        input_embeds = input_embeds.view(B * L, D)
        input_ids = samples["input_ids"].view(B * L)
        selected = (input_ids == IMAGE_TOKEN_INDEX)
        selected_sum = selected.sum()
        if selected_sum != B_IMG*L_IMG:
            # truncate the image_features, wrong way to handle this, but it is fine for now
            selected_image_features = selected_image_features[:selected_sum]
            sample_id = ':::'.join(samples["sample_id"])
            rank_zero_warn(f"\nsample_id: {sample_id}, image tokens: {selected_sum}, but image features: {B_IMG*L_IMG}\n")
        # fill the image features to the input_embeds
        input_embeds[selected] = selected_image_features
        # pack image features
        packed_image_features = self.pack_image_features(image_features_orig, samples["images"]['num_image_per_sample'],
                                    max_feature_len=self.args.state_encoder_max_feature_len if self.args.state_encoder_max_feature_len !=0 else None,
                                    num_token_per_image=self.args.state_encoder_num_token_per_image if self.args.state_encoder_num_token_per_image !=0 else None) 
        return input_embeds.view(B, L, D), samples["labels"], packed_image_features

    def pack_image_features(self, image_features: torch.Tensor, num_image_per_sample: list, max_feature_len=None, num_token_per_image=None):
        ''' two modes:
            1. pack image features to the same length: set max_feature_len to a fixed value
            2. fix image tokens per image  
        '''
        # make sure max_feature_len or num_token_per_image is provided
        assert max_feature_len is not None or num_token_per_image is not None, "max_feature_len or num_token_per_image should be provided"
        # make sure only one of them is provided
        assert max_feature_len is None or num_token_per_image is None, "max_feature_len and num_token_per_image are exclusive"
        #
        max_num_image = max(num_image_per_sample)
        if max_feature_len is not None:
            if max_num_image * image_features.shape[1] > max_feature_len:
                available_token_per_image = max_feature_len // max_num_image
                # find the closest number of tokens per image less than x**2
                choices = [int(x**2) for x in range(32, 0, -1)]
                for token in choices:
                    if token <= available_token_per_image:
                        num_token_per_image = token
                        break
            else:
                num_token_per_image = image_features.shape[1]
            real_feature_len = max_num_image * num_token_per_image
        else:
            real_feature_len = max_num_image * num_token_per_image

        output_size = int(num_token_per_image ** 0.5)
        # init with zeros
        packed_image_features = torch.zeros(len(num_image_per_sample), real_feature_len, image_features.shape[-1], 
                                        device=image_features.device, dtype=image_features.dtype)
        # split
        image_features = image_features.split(num_image_per_sample, dim=0)
        for i, feat_tup in enumerate(image_features):
            image_feature = torch.cat(list(feat_tup), dim=0) # [num_image*T, D]
            if image_feature.size(0) > real_feature_len: # adaptive pooling to H/2, W/2
                image_feature = image_feature.view(len(feat_tup), -1, image_feature.size(-1))
                image_feature = self.adaptive_pooling(image_feature, output_size=output_size)
                image_feature = image_feature.view(-1, image_feature.size(-1))
            packed_image_features[i, -image_feature.size(0):] = image_feature # left padding
        return packed_image_features

    
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
        x, _, packed_image_features = self.preparing_embedding(sampels)
        image_states = self.state_encoder(packed_image_features)
        # generate
        generated_tokens = []
        generated_token_logits = []
        generated_token_probs = []
        for i in range(max_new_tokens):
            logits = self.forward_with_image_states(x, image_states)[:, -1, :]
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
