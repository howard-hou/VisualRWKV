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
from transformers import CLIPVisionConfig, CLIPVisionModel, AutoImageProcessor, AutoModel
from src.sam import build_sam_vit_b
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam
from .dataset import IGNORE_INDEX, IMAGE_TOKEN_INDEX

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
wkv5_cuda = load(name="wkv5", sources=["cuda/wkv5_op.cpp", f"cuda/wkv5_cuda.cu"],
                verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
    
class WKV_5(torch.autograd.Function):
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
            eew = (torch.exp(ew)).contiguous()
            ctx.save_for_backward(r, k, v, eew, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y)
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
            r, k, v, eew, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gw = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            wkv5_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
            gw = torch.sum(gw, 0).view(H, C//H)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w, u):
    return WKV_5.apply(B, T, C, H, r, k, v, w, u)

########################################################################################################

class RWKV_TimeMix_RWKV5(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        assert HEAD_SIZE == self.head_size # change HEAD_SIZE to match args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        self.head_size_divisor = args.head_size_divisor

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_head, self.head_size))
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

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
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att)

    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        return r, k, v, g

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x / self.head_size_divisor).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g = self.jit_func(x)

        x = RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w=self.time_decay, u=self.time_faaaa)

        return self.jit_func_2(x, g)

########################################################################################################

class RWKV_ChannelMix(MyModule):
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
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################
class TinyAttention(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tiny_att_dim = args.tiny_att_dim
        self.head_size = args.head_size_a
        self.n_head = args.tiny_att_dim // args.head_size_a
        self.tiny_ln = nn.LayerNorm(args.n_embd)
        self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
        self.tiny_k = nn.Linear(args.merged_vision_dim, args.tiny_att_dim, bias=False)
        self.tiny_v = nn.Linear(args.merged_vision_dim, args.tiny_att_dim, bias=False)
        self.tiny_o = nn.Linear(args.tiny_att_dim, args.n_embd, bias=False)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, x_emb):
        L = x_emb.size(1)
        B, T, _ = x.size()
        xx = self.tiny_ln(x)
        q = self.tiny_q(xx).view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        k = self.tiny_k(x_emb).view(B, L, self.n_head, self.head_size).transpose(1, 2) # (B, nh, L, hs)
        v = self.tiny_v(x_emb).view(B, L, self.n_head, self.head_size).transpose(1, 2) # (B, nh, L, hs)
        # cross-attention: (B, nh, T, hs) x (B, nh, hs, L) -> (B, nh, T, L)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, L) x (B, nh, L, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, self.tiny_att_dim) # re-assemble all head outputs side by side
        return self.tiny_o(y)

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(args, 0)
        else:
            self.att = RWKV_TimeMix_RWKV5(args, layer_id)

        self.ffn = RWKV_ChannelMix(args, layer_id)

        if args.tiny_att_dim > 0 and self.layer_id in args.tiny_att_layer:
            self.tiny_att = TinyAttention(args)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)
        
    def forward(self, x, x_emb=None):
        args = self.args

        if self.layer_id == 0:
            x = self.ln0(x)

        if self.args.dropout == 0:
            if self.layer_id == 0 and args.pre_ffn > 0:
                x = x + self.ffnPre(self.ln1(x))
            else:
                x = x + self.att(self.ln1(x))
            # tiny attention
            if args.tiny_att_dim > 0 and self.layer_id in args.tiny_att_layer:
                x = x + self.tiny_att(x, x_emb)
            # ffn
            x = x + self.ffn(self.ln2(x))
        else:
            if self.layer_id == 0 and args.pre_ffn > 0:
                x = self.drop0(x + self.ffnPre(self.ln1(x)))
            else:
                x = self.drop0(x + self.att(self.ln1(x)))
            # tiny attention
            if args.tiny_att_dim > 0 and self.layer_id in args.tiny_att_layer:
                x = x + self.tiny_att(x, x_emb)
            # ffn
            x = self.drop1(x + self.ffn(self.ln2(x)))

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

    def forward(self, x, x_emb=None):
        args = self.args
        # B, T, D = x.size()
        # assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        if args.dropout > 0:
            x = self.drop0(x)

        if args.tiny_att_dim > 0 and x_emb is not None:
            for block in self.blocks:
                if args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
                else:
                    x = block(x, x_emb)
        else:
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

class SamProjector(nn.Module):
    def __init__(self, hidden_size, unified_vision_dim):
        super().__init__()
        self.down_sampler = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(hidden_size*2, hidden_size*4, kernel_size=3, stride=2, padding=1, bias=False),
        )
        self.projector = nn.Linear(hidden_size*4, unified_vision_dim, bias=False)
        self.layernorm = nn.LayerNorm(unified_vision_dim)


    def forward(self, x):
        x = self.down_sampler(x)
        x = x.flatten(2).permute(0, 2, 1)
        return self.layernorm(self.projector(x))


class VisualRWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        merged_vision_dim = 0
        # load clip first
        if args.vision_tower_clip:
            self.clip = CLIPVisionModel.from_pretrained(args.vision_tower_clip)
            self.clip_projector = nn.Sequential(
                nn.Linear(self.clip.config.hidden_size, args.unified_vision_dim, bias=False),
                nn.LayerNorm(args.unified_vision_dim)
            )
            self.freeze_clip(args.clip_unfreeze_layers)
            merged_vision_dim += args.unified_vision_dim
        # load sam
        if args.vision_tower_sam:
            self.sam = build_sam_vit_b(checkpoint=args.vision_tower_sam)
            self.sam_projector = SamProjector(self.sam.hidden_size, args.unified_vision_dim)
            self.freeze_sam(args.sam_unfreeze_layers)
            merged_vision_dim += args.unified_vision_dim
        # load dino
        if args.vision_tower_dino:
            self.dino = AutoModel.from_pretrained(args.vision_tower_dino)
            self.dino_projector = nn.Sequential(
                nn.Linear(self.dino.config.hidden_size, args.unified_vision_dim, bias=False),
                nn.LayerNorm(args.unified_vision_dim)
            )
            self.freeze_dino(args.dino_unfreeze_layers)
            merged_vision_dim += args.unified_vision_dim
        args.merged_vision_dim = merged_vision_dim
        # load rwkv
        self.rwkv = RWKV(args)
        if len(args.load_model) > 0:
            self.load_rwkv_from_pretrained(args.load_model)

    def load_rwkv_from_pretrained(self, path):
        self.rwkv.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        rank_zero_info(f"Loaded pretrained RWKV from {path}")

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def freeze_clip(self, unfreeze_layers=0):
        self.clip.requires_grad_(False)
        if unfreeze_layers > 0:
            for p in self.clip.vision_model.encoder.layers[-unfreeze_layers:].parameters():
                p.requires_grad = True
            self.clip.vision_model.post_layernorm.requires_grad_(True)
    
    def freeze_sam(self, unfreeze_layers=0):
        self.sam.requires_grad_(False)
        if unfreeze_layers > 0:
            for p in self.sam.blocks[-unfreeze_layers:].parameters():
                p.requires_grad = True
            self.sam.neck.requires_grad_(True)

    def freeze_dino(self, unfreeze_layers=0):
        self.dino.requires_grad_(False)
        if unfreeze_layers > 0:
            for p in self.dino.encoder.layer[-unfreeze_layers:].parameters():
                p.requires_grad = True
            self.dino.layernorm.requires_grad_(True)

    def freeze_rwkv(self, num_layers_to_freeze=0, freeze_tiny_att=False):
        # freeze all layers including embedding and lm head
        if num_layers_to_freeze == self.args.n_layer:
            self.rwkv.requires_grad_(False)
        # otherwise, freeze only the first num_layers_to_freeze layers
        for i, block in enumerate(self.rwkv.blocks):
            if i < num_layers_to_freeze:
                for n, p in block.named_parameters():
                    if 'tiny_att' in n and not freeze_tiny_att:
                        p.requires_grad_(True)
                    else:
                        p.requires_grad_(False)
            else:
                for p in block.parameters():
                    p.requires_grad_(True)
        # freeze embedding if num_layers_to_freeze != 0
        if num_layers_to_freeze == 0:
            self.rwkv.emb.requires_grad_(True)
        else:
            self.rwkv.emb.requires_grad_(False)


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
        logits = self.rwkv(x, x_emb=image_features)
        return logits, targets
    
    def training_step(self, batch, batch_idx):
        logits, targets = self(batch)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1))
        return L2Wrap.apply(loss, logits)
    
    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all
    
    def encode_images(self, clip_images=None, sam_images=None, dino_images=None):
        image_features = []
        if clip_images is not None:
            # [bs, 257, *] -> [bs, 256, *]
            clip_features = self.clip(clip_images).last_hidden_state[:, 1:]
            clip_features = self.clip_projector(clip_features)
            image_features.append(clip_features)
        if sam_images is not None:
            # [bs, 1024, 16, 16] -> [bs, 256, 1024]
            sam_features = self.sam(sam_images)
            sam_features = self.sam_projector(sam_features)
            image_features.append(sam_features)
        if dino_images is not None:
            # [bs, 257, *] -> [bs, 256, *]
            dino_features = self.dino(dino_images).last_hidden_state[:, 1:]
            dino_features = self.dino_projector(dino_features)
            image_features.append(dino_features)
        # concat at the last dim
        image_features = torch.cat(image_features, dim=-1)
        return image_features
   
    def preparing_embedding(self, samples, truncate=True):
        device, label_dtype = samples["labels"].device, samples["labels"].dtype
        clip_images = samples.get("clip_images", None)
        sam_images = samples.get("sam_images", None)
        dino_images = samples.get("dino_images", None)
        image_features  = self.encode_images(clip_images, sam_images, dino_images)
        emb_dtype = image_features.dtype
        # prepare text embedding
        new_input_embeds = []
        new_labels = []
        for idx, cur_input_ids in enumerate(samples["input_ids"]):
            new_input_embeds.append(self.rwkv.emb(cur_input_ids))
            new_labels.append(samples["labels"][idx])
        # keep the first `ctx_len` tokens, to make sure instruction complete
        if truncate:
            new_input_embeds = [x[:self.args.ctx_len] for x in new_input_embeds]
            new_labels = [x[:self.args.ctx_len] for x in new_labels]
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
        sampels = {"input_ids": input_ids, "clip_images": images["clip_images"], 
                   "sam_images": images["sam_images"], "dino_images": images["dino_images"], 
                   "labels": torch.full_like(input_ids, IGNORE_INDEX)}
        # prepare embedding, x: [1, seq_len, n_embd]
        x, _, image_features = self.preparing_embedding(sampels, truncate=False)
        # generate
        generated = []
        for i in range(max_new_tokens):
            logits = self.rwkv(x, x_emb=image_features)[:, -1, :]
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
