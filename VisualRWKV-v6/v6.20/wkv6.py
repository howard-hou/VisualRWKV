from typing import Optional
import time
import torch
from torch.utils.cpp_extension import load
from torch.nn import functional as F
import numpy as np
from einops import rearrange
np.set_printoptions(precision=4, suppress=True, linewidth=200)
# turn off TF32 for higher accuracy
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

# step1: set up exp parameters
B = 2
T = 1024
C = 2048
HEAD_SIZE = 64

H = C // HEAD_SIZE
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base

def val(x):
    return x.detach().float().cpu().numpy()

def LOSS(y):
    return ((y * y) - torch.tanh(y)).sum()



DEVICE = 'cuda'
DTYPE = torch.bfloat16
require_grad = True
set_seed(42)

# step2: set up inputs
with torch.no_grad():
    r = torch.empty(B, T, C, device=DEVICE, dtype=DTYPE).uniform_(-1, 1).requires_grad_(require_grad)
    k = torch.empty(B, T, C, device=DEVICE, dtype=DTYPE).uniform_(-1, 1).requires_grad_(require_grad)
    v = torch.empty(B, T, C, device=DEVICE, dtype=DTYPE).uniform_(-1, 1).requires_grad_(require_grad)
    w = torch.empty(B, T, C, device=DEVICE, dtype=DTYPE).uniform_(-8, 1).requires_grad_(require_grad)
    u = torch.empty(H, HEAD_SIZE, device=DEVICE, dtype=DTYPE).uniform_(-1, 1).requires_grad_(require_grad)
    s = torch.empty(B, H, HEAD_SIZE, HEAD_SIZE, device=DEVICE, dtype=DTYPE).uniform_(-1, 1).requires_grad_(require_grad)

def clear_grad():   
    r.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    w.requires_grad_()
    u.requires_grad_()
    s.requires_grad_()
    if r.grad is not None: r.grad.data.zero_()
    if k.grad is not None: k.grad.data.zero_()
    if v.grad is not None: v.grad.data.zero_()
    if w.grad is not None: w.grad.data.zero_()
    if u.grad is not None: u.grad.data.zero_()
    if s.grad is not None: s.grad.data.zero_()


# step3: load cuda kernel as baseline

CUDA_KERNEL_VERSION = 'v1c2_noexp'
wkv6_fp32_cuda = load(name="wkv6_fp32", sources=["cuda/fp32_wkv6_op_noexp.cpp", f"cuda/fp32_wkv6_cuda_v1c2_noexp.cu"],
                verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={T}"])
    
class WKV_6_FP32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert r.dtype == torch.float
            assert k.dtype == torch.float
            assert v.dtype == torch.float
            assert w.dtype == torch.float
            assert u.dtype == torch.float
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
            ctx.save_for_backward(r, k, v, w, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.float, memory_format=torch.contiguous_format).uniform_(-100, 100)
            wkv6_fp32_cuda.forward(B, T, C, H, r, k, v, w, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.float
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, w, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float, memory_format=torch.contiguous_format).uniform_(-100, 100)
            wkv6_fp32_cuda.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA_FP32(B, T, C, H, r, k, v, w, u):
    return WKV_6_FP32.apply(B, T, C, H, r.float(), k.float(), v.float(), w.float(), u.float())

######################################################################################################

wkv6_cuda = load(name="wkv6", sources=["cuda/wkv6_op_noexp.cpp", f"cuda/wkv6_cuda_v1c2_noexp.cu"],
                verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={T}"])
    
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
            ctx.save_for_backward(r, k, v, w, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format).uniform_(-100, 100)
            wkv6_cuda.forward(B, T, C, H, r, k, v, w, u, y)
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
            r, k, v, w, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format).uniform_(-100, 100)
            wkv6_cuda.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            # print(val(gw))
            # exit(0)
            return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r.bfloat16(), k.bfloat16(), v.bfloat16(), w.bfloat16(), u.bfloat16())

# step4: naive implementation from fla
def naive_recurrent_rwkv6_fla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False
):
    orig_dtype = q.dtype
    B, H, T, K, V = *q.shape, v.shape[-1]
    q, k, v, w, u = map(lambda x: x.float(), (q, k, v, w, u))
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    o = torch.zeros_like(v)

    if scale is None:
        scale = K ** -0.5

    if initial_state is not None:
        h += initial_state

    for i in range(T):
        q_i = q[:, :, i, :] * scale
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        w_i = w[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        o_i = (h + u[None, ..., None] * kv_i) * q_i[..., None]
        o[:, :, i] = o_i.sum(-2)
        h = h * w_i[..., None] + kv_i
    ht = h if output_final_state else None
    return o.to(orig_dtype), ht

def run_naive_recurrent_fla(B, T, C, H, r, k, v, w, u, s):
    r = r.view(B,T,H,-1).transpose(1,2)
    k = k.view(B,T,H,-1).transpose(1,2)
    v = v.view(B,T,H,-1).transpose(1,2)
    w = -torch.exp(w.view(B,T,H,-1).transpose(1,2))
    o, final_state = naive_recurrent_rwkv6_fla(r, k, v, w, u=u, scale=1, initial_state=s, output_final_state=True)
    return o.transpose(1,2).reshape(B,T,C), final_state

# step6: run baseline implementation
print(f'start exp: B={B} T={T} C={C} HEAD_SIZE={HEAD_SIZE} DTYPE={DTYPE} DEVICE={DEVICE}')
clear_grad()
start_time = time.time()
y32 = RUN_CUDA_FP32(B, T, C, H, r, k, v, w, u)
end_time = time.time()
print('cuda fp32 time:', end_time - start_time)
LOSS(y32).backward()
gr = r.grad.data.clone()
gk = k.grad.data.clone()
gv = v.grad.data.clone()
gw = w.grad.data.clone()
gu = u.grad.data.clone()
clear_grad()

start_time = time.time()
y16 = RUN_CUDA(B, T, C, H, r, k, v, w, u)
end_time = time.time()
print('cuda time:', end_time - start_time)
LOSS(y16).backward()
gr2 = r.grad.data.clone()
gk2 = k.grad.data.clone()
gv2 = v.grad.data.clone()
gw2 = w.grad.data.clone()
gu2 = u.grad.data.clone()
clear_grad()

# step7: run fla naive implementation
start_time = time.time()
y_naive_fla, state_naive_fla = run_naive_recurrent_fla(B, T, C, H, r, k, v, w, u, None)
end_time = time.time()
print('fla naive time:', end_time - start_time)
LOSS(y_naive_fla).backward()
gr3 = r.grad.data.clone()
gk3 = k.grad.data.clone()
gv3 = v.grad.data.clone()
gw3 = w.grad.data.clone()
gu3 = u.grad.data.clone()
clear_grad()

print('fla naive err ratio:')
print('y', get_err_ratio(y_naive_fla, y32))
print('gr', get_err_ratio(gr3, gr))
print('gk', get_err_ratio(gk3, gk))
print('gv', get_err_ratio(gv3, gv))
print('gw', get_err_ratio(gw3, gw))
print('gu', get_err_ratio(gu3, gu))

# step5: my naive implementation
def naive_recurrent_rwkv6_my(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False
):
    orig_dtype = r.dtype
    B, H, T, K, V = *r.shape, v.shape[-1]

    r, k, v, w, u = map(lambda x: x.float(), (r, k, v, w, u))
    wkv_state = torch.zeros(B, H, K, V, dtype=torch.float32, device=r.device)
    o = torch.zeros_like(v)

    if scale is None:
        scale = K ** -0.5

    if initial_state is not None:
        wkv_state += initial_state

    for i in range(T):
        kv = k[:, :, i:i+1, :].mT @ v[:, :, i:i+1, :]
        out = r[:, :, i:i+1, :] @ (wkv_state + u.mT * kv)
        wkv_state = w[:, :, i:i+1, :].exp().mT * wkv_state + kv
        o[:, :, i:i+1, :] = out

    if output_final_state:
        return o.to(orig_dtype), wkv_state
    else:
        return o.to(orig_dtype), None
    
def run_naive_recurrent_my(B, T, C, H, r, k, v, w, u, s):
    r = r.view(B,T,H,-1).transpose(1,2)
    k = k.view(B,T,H,-1).transpose(1,2)
    v = v.view(B,T,H,-1).transpose(1,2)
    w = -torch.exp(w.view(B,T,H,-1).transpose(1,2))
    u = u.view(1, H, 1, -1)
    o, final_state = naive_recurrent_rwkv6_my(r, k, v, w, u=u, scale=1, initial_state=s, output_final_state=True)
    return o.transpose(1,2).reshape(B,T,C), final_state

start_time = time.time()
y_naive_my, state_naive_my = run_naive_recurrent_my(B, T, C, H, r, k, v, w, u, None)
end_time = time.time()
print('my naive time:', end_time - start_time)
LOSS(y_naive_my).backward()
gr4 = r.grad.data.clone()
gk4 = k.grad.data.clone()
gv4 = v.grad.data.clone()
gw4 = w.grad.data.clone()
gu4 = u.grad.data.clone()
clear_grad()
print("max abs y error: ", (y32 - y_naive_my).abs().max().item())
print("max abs state error: ", (state_naive_my - state_naive_fla).abs().max().item())
print('my naive err ratio:')
print('y', get_err_ratio(y_naive_my, y32))
print('gr', get_err_ratio(gr4, gr))
print('gk', get_err_ratio(gk4, gk))
print('gv', get_err_ratio(gv4, gv))
print('gw', get_err_ratio(gw4, gw))
print('gu', get_err_ratio(gu4, gu))

# step8: chunk naive implementation
def naive_chunk_rwkv6(
    q,
    k,
    v,
    w,
    u,
    chunk_size=32,
    initial_state=None,
    output_final_state=True,
):
    assert q.shape[-2] % chunk_size == 0
    orig_dtype = q.dtype
    num_chunk = q.shape[-2] // chunk_size
    u = u.unsqueeze(0)

    q, k, v, w = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size).float(), (q, k, v, w))

    w_cumsum = w.cumsum(-2)

    kw = k * (w_cumsum[..., -1, None, :] - w_cumsum).exp()
    wkv = kw.transpose(-1, -2) @ v

    # Initialize wkv_new as a list to accumulate the results
    if initial_state is not None:
        wkv_new = [initial_state]
    else:
        wkv_new = [torch.zeros_like(wkv[:, :, 0])]

    # Use a differentiable way to update wkv_new
    for i in range(num_chunk - 1):
        new_value = (wkv_new[-1] * w_cumsum[:, :, i, -1, :, None].exp()) + wkv[:, :, i]
        wkv_new.append(new_value)

    # Stack the list to create the final wkv_new tensor
    wkv_new = torch.stack(wkv_new, dim=2)

    o_inter = torch.einsum('b h n d p, b h n c d -> b h n c p', wkv_new, (q * (w_cumsum - w).exp()))

    o_intra = torch.zeros_like(o_inter)
    for i in range(chunk_size):
        attn = (q[:, :, :, i, None] * k * (w_cumsum[:, :, :, i, None] - w[:, :, :, i, None] - w_cumsum).exp()).sum(-1)
        mask = (torch.arange(0, chunk_size) < i).to(attn.device)
        attn.masked_fill_(~mask, 0)
        intra_inter_o = (attn.unsqueeze(-1) * v).sum(-2)
        intra_intra_o = (q[:, :, :, i] * u.unsqueeze(2) * k[:, :, :, i]).sum(-1).unsqueeze(-1) * v[:, :, :, i]
        o_intra[:, :, :, i] = intra_inter_o + intra_intra_o
    o = o_inter + o_intra
    # output wkv state should be (b h n d p) -> (b, h, d, p)
    return rearrange(o, 'b h n c d -> b h (n c) d').to(orig_dtype), wkv_new.sum(2) if output_final_state else None

def run_naive_chunk(B, T, C, H, r, k, v, w, u, s):
    r = r.view(B,T,H,-1).transpose(1,2)
    k = k.view(B,T,H,-1).transpose(1,2)
    v = v.view(B,T,H,-1).transpose(1,2)
    w = -torch.exp(w.view(B,T,H,-1).transpose(1,2))
    #u = u.view(1, H, 1, -1)
    o, final_state = naive_chunk_rwkv6(r, k, v, w, u=u, initial_state=s, output_final_state=True)
    return o.transpose(1,2).reshape(B,T,C), final_state

start_time = time.time()
y_naive_chunk, state_naive_chunk = run_naive_chunk(B, T, C, H, r, k, v, w, u, None)
end_time = time.time()
print('my chunk time:', end_time - start_time)
LOSS(y_naive_chunk).backward()
gr5 = r.grad.data.clone()
gk5 = k.grad.data.clone()
gv5 = v.grad.data.clone()
gw5 = w.grad.data.clone()
gu5 = u.grad.data.clone()
clear_grad()
print("max abs y error: ", (y32 - y_naive_chunk).abs().max().item())
print("max abs state error: ", (state_naive_chunk - state_naive_my).abs().max().item())
print('my chunk err ratio:')
print('y', get_err_ratio(y_naive_chunk, y32))
print('gr', get_err_ratio(gr5, gr))
print('gk', get_err_ratio(gk5, gk))
print('gv', get_err_ratio(gv5, gv))
print('gw', get_err_ratio(gw5, gw))
print('gu', get_err_ratio(gu5, gu))