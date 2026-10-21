import torch
from torch import nn
from torch.nn import functional as F


class Expert(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Linear(c.dim, c.expert_dim, bias=False)
        self.up = nn.Linear(c.dim, c.expert_dim, bias=False)
        self.down = nn.Linear(c.expert_dim, c.dim, bias=False)
        self.limit = c.swiglu_clamp

    def forward(self, x):
        gate, up = self.gate(x).float(), self.up(x).float()
        return self.down((F.silu(gate.clamp(max=self.limit))*up.clamp(-self.limit, self.limit)).to(x.dtype))


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.router = nn.Linear(c.dim, c.routed_experts, bias=False)
        self.register_buffer('routing_bias', torch.zeros(c.routed_experts))
        self.experts = nn.ModuleList(Expert(c) for _ in range(c.routed_experts))
        self.shared = Expert(c)
        self.topk, self.route_scale = c.topk_experts, c.routed_scale

    def _apply(self, fn, recurse=True):
        # This accumulated control state must retain sub-BF16-ULP updates. Preserve
        # its original values even when a populated model is converted to BF16.
        bias = self.routing_bias
        result = super()._apply(fn, recurse=recurse)
        self.routing_bias = bias.to(device=self.routing_bias.device, dtype=torch.float32)
        return result

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = F.softplus(F.linear(flat.float(), self.router.weight.float())).sqrt()
        indices = (scores+self.routing_bias).topk(self.topk, -1).indices
        weights = scores.gather(-1, indices)
        weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-20)*self.route_scale
        out = self.shared(flat).float()
        # Execute even empty selections: stable DDP graph, zero grads on unvisited experts.
        # Correctness-first; replace Python dispatch with grouped GEMM before a long GPU run.
        for e, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == e)
            y = expert(flat[rows]).float()*weights[rows, slots, None]
            out = out.index_add(0, rows, y)
        counts = torch.bincount(indices.reshape(-1), minlength=len(self.experts)).float()
        probs = (scores/scores.sum(-1, keepdim=True)).view(*shape[:-1], -1)
        chosen = F.one_hot(indices, len(self.experts)).float().sum(-2).view(*shape[:-1], -1)
        freq = chosen.mean(1)/self.topk
        balance = (freq.detach()*probs.mean(1)).sum(-1).mean()*len(self.experts)
        return out.reshape(shape).to(x.dtype), balance, counts

    @torch.no_grad()
    def update_bias(self, counts, rate):
        # Counts are explicitly reduced once per optimizer update by the training loop.
        self.routing_bias.add_(rate*torch.sign(counts.mean()-counts))
        self.routing_bias.sub_(self.routing_bias.mean())
