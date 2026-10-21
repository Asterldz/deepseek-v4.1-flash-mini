from dataclasses import replace
import json
from pathlib import Path
import pytest
import torch
from v41_mini import ModelConfig, TextMini


def test_compact_config_module_shapes_and_topology():
    c = ModelConfig.from_json('configs/v41_text_mini_500m.json')
    assert c == ModelConfig.compact()
    with torch.device('meta'):
        m = TextMini(c)
    assert sum(p.numel() for p in m.parameters()) == 501_539_072
    from research.estimate_v41_text_mini import estimate
    ledger = estimate(json.loads(Path('configs/v41_text_mini_500m.json').read_text()))
    assert ledger['total_backbone_including_engram'] == sum(p.numel() for p in m.parameters())
    assert len(m.blocks) == 24
    assert m.embed.weight is not m.head.weight
    assert [i for i, b in enumerate(m.blocks) if b.attention.compressor is not None] == [2, 6, 10, 12]
    assert [i for i, b in enumerate(m.blocks) if b.attention.indexer is not None] == [2, 6, 10, 12, 14, 16, 18, 20]
    assert [i for i, b in enumerate(m.blocks) if b.engram is not None] == [1, 10]
    for block in m.blocks:
        assert block.attention.out_group.shape == (2, 64, 320)
        assert block.moe.topk == 4
        assert len(block.moe.experts) == 32
        assert block.moe.experts[0].gate.weight.shape == (288, 640)
        assert block.moe.router.weight.shape == (32, 640)
    assert ledger['active_matrix_proxy_excluding_lookup_tables'] == 102_514_368


@pytest.mark.parametrize('change', [
    {'layers': 24, 'encoder_layers': 12},
    {'candidate_source_layer': 12},
    {'dspark_target_layers': [37, 38, 40]},
])
def test_reject_incomplete_topology_migration(change):
    with pytest.raises(ValueError):
        replace(ModelConfig(), **change)


def test_compact_grouped_attention_backward_and_cached_generation():
    torch.set_num_threads(2)
    torch.manual_seed(9)
    # Keep the selected attention/group/index dimensions; reduce storage and FFNs for CPU QA.
    c = replace(ModelConfig.compact(), dim=64, expert_dim=32, vocab_size=128,
                engram_bucket_base=17, swa_window=8, index_topk=4,
                candidate_blocks=2, candidate_block_size=4)
    m = TextMini(c)
    ids = torch.randint(0, c.vocab_size, (1, 12))
    out = m(ids, labels=ids)
    torch.testing.assert_close(out.routing_counts.sum(-1), torch.full((24,), ids.numel()*4.0))
    out.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    m.eval()
    with torch.no_grad():
        full = m(ids).logits
        _, state = m.prefill(ids[:, :7])
        for i in range(7, 12):
            actual, state = m.decode(ids[:, i:i+1], state)
            torch.testing.assert_close(actual, full[:, i], rtol=2e-5, atol=2e-5)


def test_top4_routing_is_normalized_and_unselected_experts_have_zero_gradients():
    from v41_mini.experts import MoE
    torch.manual_seed(23)
    c = replace(ModelConfig.compact(), dim=16, expert_dim=8)
    moe = MoE(c)
    x = torch.randn(1, 3, c.dim)
    # Force a known selection so the aggregation can be checked independently of topk.
    with torch.no_grad():
        moe.routing_bias[:4] = 100
        moe.routing_bias[4:] = -100
    actual, _, counts = moe(x)
    flat = x.flatten(0, 1)
    scores = torch.nn.functional.softplus(moe.router(flat)).sqrt()[:, :4]
    weights = c.routed_scale*scores/scores.sum(-1, keepdim=True)
    expected = moe.shared(flat)+sum(moe.experts[i](flat)*weights[:, i:i+1] for i in range(4))
    torch.testing.assert_close(actual.flatten(0, 1), expected)
    torch.testing.assert_close(counts, torch.tensor([3.]*4+[0.]*(c.routed_experts-4)))
    actual.square().mean().backward()
    assert torch.isfinite(moe.router.weight.grad).all() and moe.router.weight.grad.abs().sum() > 0
    for expert in moe.experts[:4]:
        assert expert.down.weight.grad.abs().sum() > 0
    for expert in moe.experts[4:]:
        assert expert.down.weight.grad is not None and expert.down.weight.grad.count_nonzero() == 0
