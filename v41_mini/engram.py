"""Small multi-hash Engram. Addressing is project-defined, not checkpoint compatible."""
import torch
from torch import nn
from torch.nn import functional as F


def prime_buckets(base, count):
    result, x = [], base
    while len(result) < count:
        if x > 1 and all(x % i for i in range(2, int(x**0.5)+1)):
            result.append(x)
        x += 1
    return result


class Engram(nn.Module):
    def __init__(self, c, module_index):
        super().__init__()
        self.c = c
        cols = len(c.engram_orders)*c.engram_heads_per_order
        buckets = prime_buckets(c.engram_bucket_base, cols*len(c.engram_layers))
        self.buckets = buckets[module_index*cols:(module_index+1)*cols]
        offsets, size = [], 0
        for p in self.buckets:
            offsets.append(size)
            size += p
        self.offsets = offsets
        self.table = nn.Embedding(size, c.engram_head_dim)
        self.proj = nn.Linear(cols*c.engram_head_dim, c.dim*(c.hc_streams+1), bias=False)
        self.q_weight = nn.Parameter(torch.ones(c.hc_streams, c.dim))
        self.k_weight = nn.Parameter(torch.ones(c.hc_streams, c.dim))

    def addresses(self, mapped_ids, history=None):
        # +1 reserves 0 for absent left context; document starts reset the history.
        b, s = mapped_ids.shape
        prefix = mapped_ids.new_zeros((b, 3)) if history is None else history
        tokens = torch.cat((prefix, mapped_ids+1), 1)
        indices, col = [], 0
        for order in self.c.engram_orders:
            for head in range(self.c.engram_heads_per_order):
                p = self.buckets[col]
                h = tokens.new_zeros((b, s))
                multiplier = 257+2*head+16*order
                for j in range(order):
                    h = (h*multiplier+tokens[:, 4-order+j:4-order+j+s]) % p
                indices.append(h+self.offsets[col])
                col += 1
        return torch.stack(indices, -1), tokens[:, -3:]

    def forward(self, x, addresses):
        kv = self.proj(self.table(addresses).flatten(-2)).float()
        key, value = kv.split((self.c.hc_streams*self.c.dim, self.c.dim), -1)
        key = key.unflatten(-1, (self.c.hc_streams, self.c.dim))
        h = x.float()
        scale = torch.rsqrt(h.square().mean(-1)+self.c.norm_eps)*torch.rsqrt(key.square().mean(-1)+self.c.norm_eps)
        dot = (h*key*self.q_weight*self.k_weight).sum(-1)*scale*self.c.dim**-0.5
        signed_root = dot.sign()*dot.abs().clamp_min(1e-6).sqrt()
        return (h+signed_root.sigmoid().unsqueeze(-1)*value.unsqueeze(-2)).to(x.dtype)
