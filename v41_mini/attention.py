from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F
from .ops import RMSNorm, rope, sparse_attention, window_indices


class Source(NamedTuple):
    kv: torch.Tensor
    index_k: torch.Tensor
    ends: torch.Tensor


class LocalState(NamedTuple):
    kv: torch.Tensor
    positions: torch.Tensor


class Pending(NamedTuple):
    x: torch.Tensor
    positions: torch.Tensor


class Compressor(nn.Module):
    def __init__(self, c, ratio):
        super().__init__()
        self.ratio = ratio
        self.value = nn.Linear(c.dim, c.head_dim, bias=False)
        if ratio > 1:
            self.gate = nn.Linear(c.dim, c.head_dim, bias=False)
        self.norm = RMSNorm(c.head_dim, c.norm_eps)

    def forward(self, x, positions, pending=None):
        if pending is not None:
            x = torch.cat((pending.x, x), 1)
            positions = torch.cat((pending.positions, positions))
        cutoff = x.shape[1]//self.ratio*self.ratio
        tail = Pending(x[:, cutoff:], positions[cutoff:])
        if self.ratio > 1:
            with torch.autocast(device_type=x.device.type, enabled=False):
                value = F.linear(x[:, :cutoff].float(), self.value.weight.float())
                gate = F.linear(x[:, :cutoff].float(), self.gate.weight.float()).unflatten(1, (-1, self.ratio)).softmax(2)
                value = (value.unflatten(1, (-1, self.ratio))*gate).sum(2).to(x.dtype)
        else:
            value = self.value(x[:, :cutoff])
        value = self.norm(value)
        starts = positions[:cutoff:self.ratio]
        ends = positions[self.ratio-1:cutoff:self.ratio]
        return value, starts, ends, tail


class Indexer(nn.Module):
    def __init__(self, c, owns_key):
        super().__init__()
        self.c = c
        self.query = nn.Linear(c.q_rank, c.index_heads*c.index_dim, bias=False)
        self.weights = nn.Linear(c.dim, c.index_heads, bias=False)
        if owns_key:
            self.key = nn.Linear(c.head_dim, c.index_dim, bias=False)
            self.key_norm = RMSNorm(c.index_dim, c.norm_eps)

    def make_key(self, latent, positions):
        key = self.key_norm(self.key(latent.detach()))
        return rope(key, positions, self.c.rope_dim, self.c.rope_theta)

    def score(self, x, qr, positions, key):
        q = self.query(qr.detach()).unflatten(-1, (self.c.index_heads, self.c.index_dim))
        q = rope(q, positions, self.c.rope_dim, self.c.rope_theta)
        weights = self.weights(x.detach()).float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = torch.einsum('bqhd,bkd->bqhk', q.float(), key.float())*self.c.index_dim**-0.5
        return (logits.relu()*weights.unsqueeze(-1)).sum(-2)

    def candidates(self, scores, visible):
        c = self.c
        width = scores.shape[-1]
        padded = F.pad(scores, (0, -width % c.candidate_block_size), value=-torch.inf)
        blocks = padded.unflatten(-1, (-1, c.candidate_block_size)).amax(-1)
        last = ((visible.long().sum(-1)-1)//c.candidate_block_size).clamp_min(0)
        blocks = blocks.scatter(-1, last[None, :, None].expand(scores.shape[0], -1, 1), torch.inf)
        ix = blocks.argsort(dim=-1, descending=True, stable=True)[..., :c.candidate_blocks]
        keep = torch.zeros_like(blocks, dtype=torch.bool).scatter(-1, ix, blocks.gather(-1, ix) > -torch.inf)
        return keep.repeat_interleave(c.candidate_block_size, -1)[..., :width] & visible[None]

    def forward(self, x, qr, qmain, positions, source, candidate_mask=None, produce_candidates=False):
        c = self.c
        if not source.kv.shape[1]:
            empty = torch.full((*x.shape[:2], 1), -1, device=x.device, dtype=torch.long)
            # Keep the indexer parameters in the graph even on very short examples.
            zero = sum(p.sum()*0 for p in self.parameters())
            return empty, None, zero
        selected, masks, losses = [], [], []
        sample_queries = set(torch.linspace(0, x.shape[1]-1, min(c.index_loss_queries, x.shape[1])).long().tolist())
        for start in range(0, x.shape[1], c.query_chunk):
            end = min(start+c.query_chunk, x.shape[1])
            pos = positions[start:end]
            visible = source.ends[None, :] <= pos[:, None]
            scores = self.score(x[:, start:end], qr[:, start:end], pos, source.index_k)
            masked = scores.masked_fill(~visible[None], -torch.inf)
            if candidate_mask is not None:
                masked = masked.masked_fill(~candidate_mask[:, start:end], -torch.inf)
            # ReLU index scores can tie exactly. Stable position-order tie breaking prevents
            # appending masked future keys from changing an earlier query's selected keys.
            ix = masked.argsort(dim=-1, descending=True, stable=True)[..., :c.index_topk]
            selected.append(ix.masked_fill(~torch.isfinite(masked.gather(-1, ix)), -1))
            if produce_candidates:
                masks.append(self.candidates(masked.detach(), visible))
            # Project-specific, sampled main-attention KL target; not an official training recipe.
            rows = [i-start for i in sorted(sample_queries) if start <= i < end]
            if self.training and rows:
                valid = visible[rows].any(-1)
                if valid.any():
                    rows_t = torch.tensor(rows, device=x.device)[valid]
                    allow = visible[rows_t]
                    qm = qmain[:, start:end][:, rows_t].detach().float()
                    with torch.autocast(device_type=x.device.type, enabled=False):
                        teacher = torch.einsum('bqhd,bkd->bqhk', qm, source.kv.detach().float())*c.head_dim**-0.5
                    target = teacher.masked_fill(~allow[None, :, None], -torch.inf).softmax(-1).mean(-2)
                    logp = scores[:, rows_t].masked_fill(~allow[None], -1e30).log_softmax(-1)
                    losses.append((target*(target.clamp_min(1e-20).log()-logp)).sum(-1).mean())
        zero = sum(p.sum()*0 for p in self.parameters())
        loss = torch.stack(losses).mean() if losses else zero
        return torch.cat(selected, 1), torch.cat(masks, 1) if masks else None, loss


class Attention(nn.Module):
    def __init__(self, c, layer_id, global_enabled=True):
        super().__init__()
        self.c, self.layer_id = c, layer_id
        self.global_enabled = global_enabled and c.compress_ratios[layer_id] > 0 if layer_id < c.layers else False
        self.q_down = nn.Linear(c.dim, c.q_rank, bias=False)
        self.q_norm = RMSNorm(c.q_rank, c.norm_eps)
        self.q_up = nn.Linear(c.q_rank, c.heads*c.head_dim, bias=False)
        self.local_kv = nn.Linear(c.dim, c.head_dim, bias=False)
        self.local_norm = RMSNorm(c.head_dim, c.norm_eps)
        self.out_group = nn.Parameter(torch.empty(c.output_groups, c.output_rank, c.heads*c.head_dim//c.output_groups))
        nn.init.normal_(self.out_group, std=(c.heads*c.head_dim//c.output_groups)**-0.5)
        self.out = nn.Linear(c.output_groups*c.output_rank, c.dim, bias=False)
        self.sink = nn.Parameter(torch.zeros(c.heads))
        self.compressor = Compressor(c, c.compress_ratios[layer_id]) if self.global_enabled and layer_id in c.kv_sources else None
        self.indexer = Indexer(c, layer_id in c.kv_sources) if self.global_enabled and layer_id in c.index_sources else None

    def queries(self, x, positions):
        qr = self.q_norm(self.q_down(x))
        q = self.q_up(qr).unflatten(-1, (self.c.heads, self.c.head_dim))
        return qr, rope(q, positions, self.c.rope_dim, self.c.rope_theta)

    def project_output(self, o, positions):
        o = rope(o, positions, self.c.rope_dim, self.c.rope_theta, inverse=True)
        o = o.flatten(-2).unflatten(-1, (self.c.output_groups, -1))
        return self.out(torch.einsum('bsgd,grd->bsgr', o, self.out_group).flatten(-2))

    def build_source(self, x, positions, old=None, pending=None):
        latent, starts, ends, pending = self.compressor(x, positions, pending)
        index_k = self.indexer.make_key(latent, starts)
        main = rope(latent, starts, self.c.rope_dim, self.c.rope_theta)
        if old is not None:
            main = torch.cat((old.kv, main), 1)
            index_k = torch.cat((old.index_k, index_k), 1)
            ends = torch.cat((old.ends, ends))
        return Source(main, index_k, ends), pending

    def forward(self, x, positions, source=None, selection=None, candidates=None,
                local=None, pending=None, source_is_prebuilt=False):
        c = self.c
        qr, q = self.queries(x, positions)
        local_kv = rope(self.local_norm(self.local_kv(x)), positions, c.rope_dim, c.rope_theta)
        key_pos = positions
        if local is not None:
            local_kv = torch.cat((local.kv, local_kv), 1)
            key_pos = torch.cat((local.positions, positions))
        next_local = LocalState(local_kv[:, -c.swa_window:], key_pos[-c.swa_window:])
        indices = window_indices(positions, key_pos, c.swa_window, x.shape[0])
        keys = local_kv
        index_loss = x.new_zeros((), dtype=torch.float32)
        if self.global_enabled:
            if self.compressor is not None and not source_is_prebuilt:
                source, pending = self.build_source(x, positions, source, pending)
            if self.indexer is not None:
                use_candidates = candidates if c.hierarchical_index and self.layer_id > c.candidate_source_layer else None
                selection, fresh_candidates, index_loss = self.indexer(
                    x, qr, q, positions, source, use_candidates,
                    c.hierarchical_index and self.layer_id == c.candidate_source_layer)
                if fresh_candidates is not None:
                    candidates = fresh_candidates
            if source.kv.shape[1]:
                global_ix = torch.where(selection >= 0, selection+local_kv.shape[1], -1)
                indices = torch.cat((indices, global_ix), -1)
                keys = torch.cat((local_kv, source.kv), 1)
        o = sparse_attention(q, keys, indices, self.sink, c.query_chunk)
        if self.global_enabled and not source.kv.shape[1]:
            # A one-token document has no complete ratio-2 group. Retain a zero
            # gradient for its compressor so short documents cannot break DDP.
            o = o+source.kv.sum()*0
        return self.project_output(o, positions), source, selection, candidates, index_loss, next_local, pending
