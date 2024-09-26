import torch

B, H, N, T = 2, 3, 4, 5
C = H * N
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


r = torch.randn(B, T, H, N, device=DEVICE).double()
w = torch.ones(B, T, H, N, device=DEVICE).double()
k = torch.randn(B, T, H, N, device=DEVICE).double()
v = torch.randn(B, T, H, N, device=DEVICE).double()
a = torch.randn(B, T, H, N, device=DEVICE).double()
b = torch.randn(B, T, H, N, device=DEVICE).double()

w = torch.exp(-torch.exp(w))

out = torch.zeros((B, T, H, N), device=DEVICE).double()
state = torch.zeros((B, H, N, N), device=DEVICE).double()

for t in range(T):
    kk = k[:, t, :, :]
    rr = r[:, t, :, :]
    vv = v[:, t, :, :]
    aa = a[:, t, :, :]
    bb = b[:, t, :, :]
    
    sab = torch.einsum('bhik,bhk,bhj->bhij', state, aa, bb)
    sa = state @ (aa.unsqueeze(-1))
    sab2 = sa @ bb.unsqueeze(-1).transpose(-1, -2)
    assert torch.allclose(sab, sab2)
    state = state * w[:, t, :, None, :] + sab + torch.einsum('bhj,bhi->bhij', kk, vv)
    out[:, t, :, :] = torch.einsum('bhj,bhij->bhi', rr, state)

out = out.view((B, T, C))
print(out.shape)