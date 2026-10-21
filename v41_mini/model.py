from dataclasses import dataclass, field
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .attention import Attention, LocalState, Pending, Source
from .config import ModelConfig
from .engram import Engram
from .experts import MoE
from .ops import HyperConnection, RMSNorm


class Block(nn.Module):
    def __init__(self, c, layer_id):
        super().__init__()
        self.attention = Attention(c, layer_id)
        self.moe = MoE(c)
        self.attn_norm = RMSNorm(c.dim, c.norm_eps)
        self.ffn_norm = RMSNorm(c.dim, c.norm_eps)
        self.hc_attn = HyperConnection(c)
        self.hc_ffn = HyperConnection(c)
        self.engram = Engram(c, c.engram_layers.index(layer_id)) if layer_id in c.engram_layers else None

    def forward(self, h, pre, positions, source, selection, candidates, local=None, pending=None,
                addresses=None, source_is_prebuilt=False):
        if self.engram is not None:
            h = self.engram(h, addresses)
        next_pre, post, comb = self.hc_attn(h)
        x = self.attn_norm(HyperConnection.collapse(h, pre))
        y, source, selection, candidates, index_loss, local, pending = self.attention(
            x, positions, source, selection, candidates, local, pending, source_is_prebuilt)
        h = HyperConnection.merge(h, y, post, comb)
        final_pre, post, comb = self.hc_ffn(h)
        x = self.ffn_norm(HyperConnection.collapse(h, next_pre))
        y, balance, counts = self.moe(x)
        h = HyperConnection.merge(h, y, post, comb)
        return h, final_pre, source, selection, candidates, index_loss, balance, counts, local, pending


@dataclass
class DecodeState:
    sources: dict[int, Source] = field(default_factory=dict)
    local: dict[int, LocalState] = field(default_factory=dict)
    pending: dict[int, Pending] = field(default_factory=dict)
    position: int = 0
    hash_history: torch.Tensor | None = None
    approximate: bool = False
    layer_tokens: list[int] = field(default_factory=list)


@dataclass
class ModelOutput:
    logits: torch.Tensor | None
    loss: torch.Tensor | None
    lm_loss: torch.Tensor | None
    index_loss: torch.Tensor
    balance_loss: torch.Tensor
    routing_counts: torch.Tensor
    hidden: torch.Tensor
    dspark_features: torch.Tensor | None = None


class TextMini(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(Block(config, i) for i in range(config.layers))
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.register_buffer('token_map', torch.arange(config.vocab_size))

    def set_token_map(self, values):
        values = torch.as_tensor(values, device=self.token_map.device)
        if (values.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
                or values.shape != self.token_map.shape
                or (values < 0).any() or (values >= self.config.vocab_size).any()):
            raise ValueError("Invalid Engram token map")
        self.token_map.copy_(values)

    def _initial(self, ids):
        x = self.embed(ids)
        h = x.unsqueeze(-2).expand(-1, -1, self.config.hc_streams, -1)
        pre = torch.zeros(*ids.shape, self.config.hc_streams, device=ids.device, dtype=torch.float32)
        pre[..., 0] = 1
        return h, pre

    def _layers(self, h, pre, mapped, positions, start, end, cache=None, prebuilt_boundary=False,
                collect_features=False):
        source, selection, candidates = None, None, None
        ixloss, balance, counts, features = [], [], [], []
        history = cache.hash_history if cache is not None else None
        for i in range(start, end):
            block = self.blocks[i]
            if i in self.config.kv_sources:
                source = cache.sources.get(i) if cache is not None else None
            if i in self.config.dspark_target_layers and collect_features:
                features.append(h.mean(-2))
            addresses = block.engram.addresses(mapped, history)[0] if block.engram is not None else None
            local = cache.local.get(i) if cache is not None else None
            pending = cache.pending.get(i) if cache is not None else None
            args = (h, pre, positions, source, selection, candidates, local, pending, addresses,
                    prebuilt_boundary and i == self.config.encoder_layers)
            if self.training and self.config.gradient_checkpointing and cache is None:
                out = checkpoint(block, *args, use_reentrant=False)
            else:
                out = block(*args)
            h, pre, source, selection, candidates, il, bl, ct, local, pending = out
            ixloss.append(il)
            balance.append(bl)
            counts.append(ct)
            if cache is not None:
                cache.local[i] = local
                cache.layer_tokens[i] += h.shape[1]
                if i in self.config.kv_sources:
                    cache.sources[i] = source
                    if pending is not None:
                        cache.pending[i] = pending
        return h, pre, torch.stack(ixloss).sum(), torch.stack(balance).mean(), torch.stack(counts), features

    def _hidden(self, ids, collect_features=False):
        h, pre = self._initial(ids)
        positions = torch.arange(ids.shape[1], device=ids.device)
        h, pre, il, bl, counts, features = self._layers(
            h, pre, self.token_map[ids], positions, 0, self.config.layers, collect_features=collect_features)
        return self.norm(HyperConnection.collapse(h, pre)), il/len(self.config.index_sources), bl, counts, features

    def _lm_loss(self, hidden, labels):
        # labels use the standard unshifted convention; -100 masks prompt/pad tokens in SFT.
        hs, targets = hidden[:, :-1].reshape(-1, self.config.dim), labels[:, 1:].reshape(-1)
        count = (targets != -100).sum()
        total = hidden.sum()*0+self.head.weight.reshape(-1)[0]*0
        def chunk_loss(x, y):
            return F.cross_entropy(self.head(x).float(), y, ignore_index=-100, reduction='sum')
        for start in range(0, targets.numel(), self.config.loss_chunk):
            x, y = hs[start:start+self.config.loss_chunk], targets[start:start+self.config.loss_chunk]
            if self.training and torch.is_grad_enabled():
                total = total+checkpoint(chunk_loss, x, y, use_reentrant=False)
            else:
                total = total+chunk_loss(x, y)
        return total/count.clamp_min(1)

    def forward(self, input_ids, labels=None, segment_ids=None, return_logits=True, collect_features=False):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a nonempty [batch, sequence] tensor")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("Sequence exceeds configured context")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids")
        if segment_ids is None:
            hidden, il, bl, counts, features = self._hidden(input_ids, collect_features)
        else:
            # Deliberately conservative packed path: independent document forwards ensure every
            # attention/compression/Engram boundary is isolated, not merely the final CE mask.
            if segment_ids.shape != input_ids.shape or collect_features:
                raise ValueError("Invalid packed input or unsupported packed DSpark feature collection")
            rows, all_il, all_bl, all_counts = [], [], [], []
            labels = labels.clone() if labels is not None else None
            # Parse boundaries on the host once; scalar CUDA comparisons here would
            # otherwise synchronize the device for every token of every microbatch.
            segment_rows = segment_ids.detach().cpu().tolist()
            for b, segment_row in enumerate(segment_rows):
                spans, s = [], 0
                while s < input_ids.shape[1]:
                    e = s+1
                    while e < input_ids.shape[1] and segment_row[e] == segment_row[s]:
                        e += 1
                    if segment_row[s] < 0:
                        spans.append(self.embed.weight.new_zeros(1, e-s, self.config.dim))
                        if labels is not None:
                            labels[b, s:e] = -100
                    else:
                        hid, li, lb, ct, _ = self._hidden(input_ids[b:b+1, s:e])
                        spans.append(hid)
                        all_il.append(li)
                        all_bl.append(lb)
                        all_counts.append(ct)
                        if labels is not None:
                            labels[b, s] = -100
                    s = e
                rows.append(torch.cat(spans, 1))
            if not all_il:
                raise ValueError("Batch contains no valid document")
            hidden = torch.cat(rows, 0)
            il, bl = torch.stack(all_il).mean(), torch.stack(all_bl).mean()
            counts, features = torch.stack(all_counts).sum(0), []
        lm = self._lm_loss(hidden, labels) if labels is not None else None
        loss = lm+self.config.index_loss_weight*il+self.config.balance_loss_weight*bl if lm is not None else None
        logits = self.head(hidden).float() if return_logits else None
        return ModelOutput(logits, loss, lm, il, bl, counts, hidden,
                           torch.cat(features, -1) if features else None)

    @torch.no_grad()
    def update_router_bias(self, counts, rate=0.001):
        for block, count in zip(self.blocks, counts):
            block.moe.update_bias(count, rate)

    @torch.no_grad()
    def prefill(self, input_ids, mode='full'):
        """Create a NEW cache. bounded_replay is intentionally approximate for long inputs."""
        if self.training:
            raise ValueError("Call eval() before inference")
        if mode not in ('full', 'bounded_replay'):
            raise ValueError("Unknown prefill mode")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not 0 < input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError("Reference cache currently supports batch=1 and a nonempty valid context")
        state = DecodeState(layer_tokens=[0]*self.config.layers)
        h, pre = self._initial(input_ids)
        mapped = self.token_map[input_ids]
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        if mode == 'full':
            h, pre, *_ = self._layers(h, pre, mapped, positions, 0, self.config.layers, state)
        else:
            split = self.config.encoder_layers
            h, pre, *_ = self._layers(h, pre, mapped, positions, 0, split, state)
            boundary = self.blocks[split].attn_norm(HyperConnection.collapse(h, pre))
            state.sources[split], state.pending[split] = self.blocks[split].attention.build_source(boundary, positions)
            window = self.config.swa_window
            h, pre, *_ = self._layers(h[:, -window:], pre[:, -window:], mapped[:, -window:],
                                      positions[-window:], split, self.config.layers, state, prebuilt_boundary=True)
            state.approximate = input_ids.shape[1] > window
        state.position = input_ids.shape[1]
        state.hash_history = F.pad(mapped+1, (max(0, 3-mapped.shape[1]), 0))[:, -3:]
        logits = self.head(self.norm(HyperConnection.collapse(h[:, -1:], pre[:, -1:]))).float()
        return logits[:, -1], state

    @torch.no_grad()
    def decode(self, token, state: DecodeState):
        if self.training or token.shape != (1, 1):
            raise ValueError("decode requires eval mode and exactly one token for batch=1")
        if state.position >= self.config.max_seq_len:
            raise ValueError("Decode exceeds configured context")
        h, pre = self._initial(token)
        mapped = self.token_map[token]
        positions = torch.tensor([state.position], device=token.device)
        h, pre, *_ = self._layers(h, pre, mapped, positions, 0, self.config.layers, state)
        state.position += 1
        state.hash_history = torch.cat((state.hash_history, mapped+1), 1)[:, -3:]
        return self.head(self.norm(HyperConnection.collapse(h, pre))).float()[:, -1], state

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=32, temperature=0.0, eos_token_id=None, prefill_mode='full'):
        if max_new_tokens < 0:
            raise ValueError('max_new_tokens must be nonnegative')
        if input_ids.shape[1]+max_new_tokens > self.config.max_seq_len:
            raise ValueError('Prompt plus requested output exceeds configured context')
        if max_new_tokens == 0:
            return input_ids.clone()
        logits, state = self.prefill(input_ids, prefill_mode)
        generated = [input_ids]
        for i in range(max_new_tokens):
            next_id = logits.argmax(-1, keepdim=True) if temperature <= 0 else torch.multinomial((logits/temperature).softmax(-1), 1)
            generated.append(next_id)
            if eos_token_id is not None and next_id.item() == eos_token_id:
                break
            if i+1 < max_new_tokens:
                logits, state = self.decode(next_id, state)
        return torch.cat(generated, 1)
