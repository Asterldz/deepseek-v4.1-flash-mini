"""Whole-parameter mixed optimizer, compatible with ZeRO-1 ownership.

Sinkhorn follows report Algorithm 1. Muon uses a portable Newton-Schulz reference;
this is a project optimizer implementation, not an official training stack.
"""
import math
import torch
from torch.optim import Optimizer


def orthogonal_update(g, steps=5):
    x = g.float()
    transpose = x.shape[-2] > x.shape[-1]
    if transpose:
        x = x.transpose(-1, -2)
    x = x/x.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    for _ in range(steps):
        a = x@x.transpose(-1, -2)
        x = 3.4445*x + (-4.7750*a+2.0315*(a@a))@x
    if transpose:
        x = x.transpose(-1, -2)
    return x/x.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-8)*0.18


def sinkhorn_update(g, steps=11, tau=1e-3, eps=1e-20):
    transposed = g.shape[-2] < g.shape[-1]
    x = g.transpose(-1, -2).clone() if transposed else g.clone()
    row_norm = x.norm(dim=-1, keepdim=True)
    x.masked_fill_(row_norm <= tau*row_norm.mean(-2, keepdim=True), 0)
    for i in range(steps):
        axis = -1 if i % 2 == 0 else -2
        x = x/(x.norm(dim=axis, keepdim=True)+eps)
    x = x*math.sqrt(x.shape[-1])*0.18
    return x.transpose(-1, -2) if transposed else x


class HybridOptimizer(Optimizer):
    def __init__(self, params, lr=2e-4, weight_decay=0.1, momentum=0.95,
                 betas=(0.9, 0.95), eps=1e-8):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                                    betas=betas, eps=eps, kind='adamw', heads=1))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                if grad.is_sparse:
                    raise ValueError("Dense gradients are required for Sinkhorn/momentum updates")
                state = self.state[p]
                if not state:
                    state['step'] = 0
                    state['master'] = p.detach().float().clone()
                    state['momentum'] = torch.zeros_like(p, dtype=torch.float32)
                    if group['kind'] == 'adamw':
                        state['variance'] = torch.zeros_like(p, dtype=torch.float32)
                state['step'] += 1
                master, momentum = state['master'], state['momentum']
                if group['kind'] == 'adamw':
                    b1, b2 = group['betas']
                    momentum.lerp_(grad, 1-b1)
                    state['variance'].lerp_(grad.square(), 1-b2)
                    mean = momentum/(1-b1**state['step'])
                    denom = (state['variance']/(1-b2**state['step'])).sqrt()+group['eps']
                    update = mean/denom
                else:
                    beta = group['momentum']
                    momentum.lerp_(grad, 1-beta)
                    direction = beta*momentum+(1-beta)*grad
                    if group['kind'] == 'sinkhorn':
                        update = sinkhorn_update(direction)
                    else:
                        shape = direction.shape
                        if direction.ndim == 2 and group['heads'] > 1:
                            direction = direction.unflatten(0, (group['heads'], -1))
                        update = orthogonal_update(direction).reshape(shape)
                master.mul_(1-group['lr']*group['weight_decay']).add_(update, alpha=-group['lr'])
                p.copy_(master)
        return loss

    def load_state_dict(self, state_dict):
        # Optimizer base load casts tensors to the parameter dtype; preserve FP32 master/state.
        super().load_state_dict(state_dict)
        for saved_group, group in zip(state_dict['param_groups'], self.param_groups):
            for saved_id, p in zip(saved_group['params'], group['params']):
                original = state_dict['state'].get(saved_id, {})
                for key in ('master', 'momentum', 'variance'):
                    if key in original:
                        self.state[p][key] = original[key].to(device=p.device, dtype=torch.float32).clone()


def parameter_groups(model, lr, optimizer='hybrid', engram_lr_scale=1.0):
    grouped = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        kind, decay, heads, scale = 'muon', 0.1, 1, 1.0
        if name in ('embed.weight', 'head.weight', 'markov_embed.weight', 'markov_head.weight') or name.endswith('engram.table.weight'):
            kind, decay = 'sinkhorn', 0.0
            if 'engram.table' in name:
                scale = engram_lr_scale
        elif p.ndim < 2 or name.endswith(('q_weight', 'k_weight')):
            kind = 'adamw'
            decay = 0.1 if 'norm' in name or name.endswith(('q_weight', 'k_weight')) else 0.0
        elif name.endswith('attention.q_up.weight'):
            heads = model.config.heads
        elif name.endswith('attention.indexer.query.weight'):
            heads = model.config.index_heads
        if optimizer == 'adamw':
            kind = 'adamw'
        key = (kind, decay, heads, scale)
        grouped.setdefault(key, []).append(p)
    return [dict(params=ps, kind=kind, weight_decay=decay, heads=heads,
                 lr=lr*scale, initial_lr=lr*scale, lr_scale=scale)
            for (kind, decay, heads, scale), ps in grouped.items()]
