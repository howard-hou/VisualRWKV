from typing import Optional
import time, sys
import torch
from torch.utils.cpp_extension import load
from torch.nn import functional as F
import numpy as np
from einops import rearrange
from fla.ops.rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6
np.set_printoptions(precision=4, suppress=True, linewidth=200)
# turn off TF32 for higher accuracy
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

test_task = sys.argv[1]
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
    initial_state = torch.empty(B, H, HEAD_SIZE, HEAD_SIZE, device=DEVICE, dtype=DTYPE).uniform_(-1, 1).requires_grad_(require_grad)

def clear_grad():   
    r.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    w.requires_grad_()
    u.requires_grad_()
    initial_state.requires_grad_()
    if r.grad is not None: r.grad.data.zero_()
    if k.grad is not None: k.grad.data.zero_()
    if v.grad is not None: v.grad.data.zero_()
    if w.grad is not None: w.grad.data.zero_()
    if u.grad is not None: u.grad.data.zero_()
    if initial_state.grad is not None: initial_state.grad.data.zero_()


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

# step: run fla chunk implementation
def run_chunk_rwkv6_fla(B, T, C, H, r, k, v, w, u, s):
    r = r.view(B,T,H,-1).transpose(1,2)
    k = k.view(B,T,H,-1).transpose(1,2)
    v = v.view(B,T,H,-1).transpose(1,2)
    w = -torch.exp(w.view(B,T,H,-1).transpose(1,2))
    o, final_state = chunk_rwkv6(r, k, v, w, u=u, scale=1, initial_state=s, output_final_state=True)
    return o.transpose(1,2).reshape(B,T,C), final_state

if test_task == 'chunk_rwkv6':
    print("#"*50 + "fla chunk implementation" + "#"*50)
    for i in range(5):
        start_time = time.time()
        y_chunk_fla, state_chunk_fla = run_chunk_rwkv6_fla(B, T, C, H, r, k, v, w, u, s=None)
        end_time = time.time()
        print(f'fla chunk time {i}:', end_time - start_time)
    LOSS(y_chunk_fla).backward()
    gr6 = r.grad.data.clone()
    gk6 = k.grad.data.clone()
    gv6 = v.grad.data.clone()
    gw6 = w.grad.data.clone()
    gu6 = u.grad.data.clone()
    clear_grad()
    print("max abs y error: ", (y32 - y_chunk_fla).abs().max().item())
    print("max abs state error: ", (state_chunk_fla - state_naive_fla).abs().max().item())
    print('fla chunk err ratio:')
    print('y', get_err_ratio(y_chunk_fla, y32))
    print('gr', get_err_ratio(gr6, gr))
    print('gk', get_err_ratio(gk6, gk))
    print('gv', get_err_ratio(gv6, gv))
    print('gw', get_err_ratio(gw6, gw))
    print('gu', get_err_ratio(gu6, gu))
    print("#"*100)


# step: run fla fused implementation
def run_fused_rwkv6_fla(B, T, C, H, r, k, v, w, u, s):
    r = r.view(B,T,H,-1).transpose(1,2)
    k = k.view(B,T,H,-1).transpose(1,2)
    v = v.view(B,T,H,-1).transpose(1,2)
    w = -torch.exp(w.view(B,T,H,-1).transpose(1,2))
    o, final_state = fused_recurrent_rwkv6(r, k, v, w, u=u, scale=1, initial_state=s, output_final_state=True)
    return o.transpose(1,2).reshape(B,T,C), final_state
if test_task == 'fused_rwkv6':
    print("#"*50 + "fla fused implementation" + "#"*50)
    for i in range(5):
        start_time = time.time()
        y_fused_fla, state_fused_fla = run_fused_rwkv6_fla(B, T, C, H, r, k, v, w, u, s=None)
        end_time = time.time()
        print(f'fla fused time {i}:', end_time - start_time)
    LOSS(y_fused_fla).backward()
    gr7 = r.grad.data.clone()
    gk7 = k.grad.data.clone()
    gv7 = v.grad.data.clone()
    gw7 = w.grad.data.clone()
    gu7 = u.grad.data.clone()
    clear_grad()
    print("max abs y error: ", (y32 - y_fused_fla).abs().max().item())
    print("max abs state error: ", (state_fused_fla - state_naive_fla).abs().max().item())
    print('fla fused err ratio:')
    print('y', get_err_ratio(y_fused_fla, y32))
    print('gr', get_err_ratio(gr7, gr))
    print('gk', get_err_ratio(gk7, gk))
    print('gv', get_err_ratio(gv7, gv))
    print('gw', get_err_ratio(gw7, gw))
    print('gu', get_err_ratio(gu7, gu))
    print("#"*100)

if test_task == 'fused_rwkv6_state_reuse':
    # # Check reuse the first state
    y_fused_fla, state = run_fused_rwkv6_fla(B, T, C, H, r, k, v, w, u, s=initial_state)
    y_fused_fla2, _ = run_fused_rwkv6_fla(B, T, C, H, r, k, v, w, u, s=state)
    LOSS(y_fused_fla2).backward()
    gr8 = r.grad.data.clone()
    gk8 = k.grad.data.clone()
    gv8 = v.grad.data.clone()
    gw8 = w.grad.data.clone()
    gu8 = u.grad.data.clone()
    gh8 = initial_state.grad.data.clone()
    clear_grad()
    print("grad of initial state, max and sum: ", gh8.abs().max().item(), gh8.abs().sum().item())
    print("grad of state, max and sum: ", state.abs().max().item(), state.abs().sum().item())

if test_task == 'chunk_rwkv6_proj':
    # check projection grad
    import torch.nn as nn
    T_img = 1
    print("#"*50 + "check chunk rwkv6 projection layer for img + text" + "#"*50)
    image_feature = torch.randn(B, T_img, C, device=DEVICE, dtype=DTYPE)
    proj_layer = nn.Linear(C, C, bias=False, device=DEVICE, dtype=DTYPE)
    img = proj_layer(image_feature)
    linear_r = nn.Linear(C, C, bias=False, device=DEVICE, dtype=DTYPE)
    linear_w = nn.Linear(C, C, bias=False, device=DEVICE, dtype=DTYPE)
    linear_k = nn.Linear(C, C, bias=False, device=DEVICE, dtype=DTYPE)
    linear_v = nn.Linear(C, C, bias=False, device=DEVICE, dtype=DTYPE)
    linear_r.requires_grad_(False)
    linear_w.requires_grad_(False)
    linear_k.requires_grad_(False)
    linear_v.requires_grad_(False)
    r_img = linear_r(img)
    w_img = linear_w(img)
    k_img = linear_k(img)
    v_img = linear_v(img)
    y_img, img_state = run_fused_rwkv6_fla(B, T_img, C, H, r_img, k_img, v_img, w_img, u, s=None)

    text_emb = torch.randn(B, T, C, device=DEVICE, dtype=DTYPE)
    r_text = linear_r(text_emb)
    w_text = linear_w(text_emb)
    k_text = linear_k(text_emb)
    v_text = linear_v(text_emb)
    y_text, text_state = run_fused_rwkv6_fla(B, T, C, H, r_text, k_text, v_text, w_text, u, s=img_state)

    LOSS(y_text).backward()
    gproj = proj_layer.weight.grad.data.clone()
    clear_grad()
    print("init state is None, grad of projection layer, max and sum: ", gproj.abs().max().item(), gproj.abs().sum().item())
    image_feature = image_feature.detach().clone()
    r_img = linear_r(image_feature)
    w_img = linear_w(image_feature)
    k_img = linear_k(image_feature)
    v_img = linear_v(image_feature)
    y_img, img_state = run_fused_rwkv6_fla(B, T_img, C, H, r_img, k_img, v_img, w_img, u, s=initial_state)

    text_emb = text_emb.detach().clone()
    r_text = linear_r(text_emb)
    w_text = linear_w(text_emb)
    k_text = linear_k(text_emb)
    v_text = linear_v(text_emb)
    y_text, text_state = run_fused_rwkv6_fla(B, T, C, H, r_text, k_text, v_text, w_text, u, s=img_state)
    LOSS(y_text).backward()
    gproj = proj_layer.weight.grad.data.clone()
    clear_grad()
    print("init state is tensor, grad of projection layer, max and sum: ", gproj.abs().max().item(), gproj.abs().sum().item())
