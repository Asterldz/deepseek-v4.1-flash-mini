"""Portable differentiable reference operators. No custom CUDA dependency."""
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True)+self.eps) * self.weight.float()).to(x.dtype)


def rope(x, positions, width, theta, inverse=False):
    """Adjacent-pair rotation on the last width channels; positions is [sequence]."""
    freq = theta ** (-torch.arange(0, width, 2, device=x.device, dtype=torch.float32)/width)
    angle = positions.float()[:, None]*freq[None, :]
    shape = [1, positions.numel()] + [1]*(x.ndim-3) + [width//2]
    cos, sin = angle.cos().view(shape), angle.sin().view(shape)
    if inverse:
        sin = -sin
    tail = x[..., -width:].float().reshape(*x.shape[:-1], width//2, 2)
    a, b = tail.unbind(-1)
    rotated = torch.stack((a*cos-b*sin, a*sin+b*cos), -1).flatten(-2)
    return torch.cat((x[..., :-width], rotated.to(x.dtype)), -1)


def gather_positions(values, indices):
    # values [B,K,D], indices [B,Q,T]; KV is NOT duplicated along the head axis.
    b = torch.arange(values.shape[0], device=values.device)[:, None, None]
    return values[b, indices.clamp_min(0)]


def sparse_attention(q, kv, indices, sink, chunk_size):
    """Joint local/global softmax with a learned, zero-value sink; fully differentiable."""
    outputs = []
    for start in range(0, q.shape[1], chunk_size):
        qc, ix = q[:, start:start+chunk_size], indices[:, start:start+chunk_size]
        selected = gather_positions(kv, ix)
        with torch.autocast(device_type=q.device.type, enabled=False):
            scores = torch.einsum('bqhd,bqkd->bqhk', qc.float(), selected.float()) * q.shape[-1]**-0.5
            scores = scores.masked_fill(ix[:, :, None, :] < 0, -torch.inf)
            sink_scores = sink.float()[None, None, :, None].expand(*scores.shape[:-1], 1)
            probs = torch.cat((scores, sink_scores), -1).softmax(-1)[..., :-1]
            outputs.append(torch.einsum('bqhk,bqkd->bqhd', probs, selected.float()).to(q.dtype))
    return torch.cat(outputs, 1)


def window_indices(query_pos, key_pos, window, batch):
    wanted = query_pos[:, None] - window + 1 + torch.arange(window, device=query_pos.device)
    ix = wanted-key_pos[0]
    ix = ix.masked_fill((ix < 0) | (ix >= key_pos.numel()), -1)
    return ix[None].expand(batch, -1, -1)


class HyperConnection(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n, self.d, self.eps = c.hc_streams, c.dim, c.norm_eps
        self.iters = c.hc_sinkhorn_iterations
        size = (2+self.n)*self.n
        self.weight = nn.Parameter(torch.empty(size, self.n*self.d))
        self.base = nn.Parameter(torch.zeros(size))
        self.scale = nn.Parameter(torch.full((3,), 0.01))
        nn.init.normal_(self.weight, std=(self.n*self.d)**-0.5)
        with torch.no_grad():
            self.base[:self.n].fill_(-3.0)
            self.base[0] = 3.0
            self.base[self.n:2*self.n].fill_(-3.0)
            self.base[2*self.n:].view(self.n, self.n).copy_(torch.eye(self.n)*4.0)

    def forward(self, x):
        flat = x.flatten(-2).float()
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True)+self.eps)
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = F.linear(flat, self.weight.float())
        a, b, m = z.split((self.n, self.n, self.n*self.n), -1)
        ba, bb, bm = self.base.split((self.n, self.n, self.n*self.n))
        pre = torch.sigmoid(a*self.scale[0]+ba)+self.eps
        post = 2*torch.sigmoid(b*self.scale[1]+bb)
        comb = (m*self.scale[2]+bm).unflatten(-1, (self.n, self.n)).softmax(-1)+self.eps
        comb = comb/(comb.sum(-2, keepdim=True)+self.eps)
        for _ in range(self.iters-1):
            comb = comb/(comb.sum(-1, keepdim=True)+self.eps)
            comb = comb/(comb.sum(-2, keepdim=True)+self.eps)
        return pre, post, comb

    @staticmethod
    def collapse(x, pre):
        return (x.float()*pre.unsqueeze(-1)).sum(-2).to(x.dtype)

    @staticmethod
    def merge(residual, output, post, comb):
        with torch.autocast(device_type=residual.device.type, enabled=False):
            y = torch.einsum('bsij,bsjd->bsid', comb.float(), residual.float())
        return (y+post.unsqueeze(-1)*output.float().unsqueeze(-2)).to(residual.dtype)
