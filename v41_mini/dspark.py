"""Trainable DSpark-shaped drafter. Acceptance calibration/scheduling is a later stage.

Shared backbone embedding/head are supplied to forward, never registered a second time.
This module deliberately has no speculative acceptance shortcut.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .model import Block
from .ops import HyperConnection, RMSNorm, rope, sparse_attention


class DraftBlock(Block):
    def forward(self, h, pre, main_x, main_positions, draft_positions):
        next_pre, post, comb = self.hc_attn(h)
        x = self.attn_norm(HyperConnection.collapse(h, pre))
        attn, c = self.attention, self.attention.c
        _, q = attn.queries(x, draft_positions)
        main_kv = rope(attn.local_norm(attn.local_kv(main_x)), main_positions, c.rope_dim, c.rope_theta)
        draft_kv = rope(attn.local_norm(attn.local_kv(x)), draft_positions, c.rope_dim, c.rope_theta)
        kv = torch.cat((main_kv, draft_kv), 1)
        # All five base-logit slots see only known/noise inputs, never ground-truth future tokens.
        indices = torch.arange(kv.shape[1], device=x.device)[None, None].expand(x.shape[0], x.shape[1], -1)
        y = attn.project_output(sparse_attention(q, kv, indices, attn.sink, c.query_chunk), draft_positions)
        h = HyperConnection.merge(h, y, post, comb)
        final_pre, post, comb = self.hc_ffn(h)
        x = self.ffn_norm(HyperConnection.collapse(h, next_pre))
        y, balance, counts = self.moe(x)
        return HyperConnection.merge(h, y, post, comb), final_pre, balance, counts


class DSpark(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.main_proj = nn.Linear(len(config.dspark_target_layers)*config.dim, config.dim, bias=False)
        self.main_norm = RMSNorm(config.dim, config.norm_eps)
        self.blocks = nn.ModuleList(DraftBlock(config, config.layers+i) for i in range(config.dspark_layers))
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.markov_embed = nn.Embedding(config.vocab_size, config.dspark_markov_rank)
        self.markov_head = nn.Linear(config.dspark_markov_rank, config.vocab_size, bias=False)
        self.confidence_head = nn.Linear(config.dim+config.dspark_markov_rank, 1, bias=False)

    def forward(self, features, known_token, known_position, embedding_weight, head_weight,
                targets=None, acceptance_targets=None, temperature=1.0):
        """features are causal backbone layer INPUTS before known_token, [B,T,3*d].

        Teacher forcing affects only the Markov head. All shared inputs/weights are detached.
        acceptance_targets, when supplied, must come from actual main-policy verification.
        """
        c = self.config
        if features.shape[1] < 1 or known_token.shape != (features.shape[0], 1):
            raise ValueError('Expected nonempty prefix features and one known token per sample')
        if known_position < features.shape[1]:
            raise ValueError('Prefix features extend beyond known-token position')
        main_x = self.main_norm(self.main_proj(features[:, -c.swa_window:].detach()))
        main_positions = torch.arange(known_position-main_x.shape[1], known_position, device=known_token.device)
        positions = torch.arange(known_position, known_position+c.dspark_block_size, device=known_token.device)
        noise = known_token.new_full((known_token.shape[0], c.dspark_block_size-1), c.dspark_noise_token_id)
        ids = torch.cat((known_token, noise), 1)
        h = F.embedding(ids, embedding_weight.detach()).unsqueeze(-2).expand(-1, -1, c.hc_streams, -1)
        pre = torch.zeros(*ids.shape, c.hc_streams, device=ids.device, dtype=torch.float32)
        pre[..., 0] = 1
        balances = []
        for block in self.blocks:
            h, pre, balance, _ = block(h, pre, main_x, main_positions, positions)
            balances.append(balance)
        raw_hidden = HyperConnection.collapse(h, pre)
        base_logits = F.linear(self.norm(raw_hidden), head_weight.detach()).float()
        if targets is not None:
            if targets.shape != ids.shape:
                raise ValueError('One target is required per draft position')
            previous = torch.cat((known_token, targets[:, :-1]), 1)
            markov = self.markov_embed(previous)
            logits = base_logits+self.markov_head(markov).float()
            sampled = logits.argmax(-1)
        else:
            previous, logit_rows, sampled_rows, markov_rows = known_token[:, 0], [], [], []
            for i in range(c.dspark_block_size):
                m = self.markov_embed(previous)
                score = base_logits[:, i]+self.markov_head(m).float()
                previous = score.argmax(-1) if temperature <= 0 else torch.multinomial((score/temperature).softmax(-1), 1)[:, 0]
                logit_rows.append(score)
                sampled_rows.append(previous)
                markov_rows.append(m)
            logits, sampled, markov = torch.stack(logit_rows, 1), torch.stack(sampled_rows, 1), torch.stack(markov_rows, 1)
        with torch.autocast(device_type=raw_hidden.device.type, enabled=False):
            confidence = F.linear(torch.cat((raw_hidden, markov), -1).float(), self.confidence_head.weight.float()).squeeze(-1)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            loss = loss+c.balance_loss_weight*torch.stack(balances).mean()
        if acceptance_targets is not None:
            if acceptance_targets.shape != confidence.shape:
                raise ValueError('Acceptance labels must match draft positions')
            confidence_loss = F.binary_cross_entropy_with_logits(confidence, acceptance_targets.float())
            loss = confidence_loss if loss is None else loss+confidence_loss
        return dict(logits=logits, tokens=sampled, confidence=confidence.sigmoid(), loss=loss)
