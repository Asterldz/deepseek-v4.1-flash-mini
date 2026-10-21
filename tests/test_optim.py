import copy
import torch
from v41_mini.optim import HybridOptimizer, sinkhorn_update


def test_sinkhorn_zero_rows_and_scale():
    g = torch.randn(32, 8)
    g[0] = 0
    update = sinkhorn_update(g)
    assert torch.isfinite(update).all() and update[0].abs().sum() == 0
    torch.testing.assert_close(update[1:].square().mean(-1).sqrt(), torch.full((31,), 0.18))


def test_bf16_optimizer_resume_preserves_fp32_state():
    torch.manual_seed(42)
    p = torch.nn.Parameter(torch.randn(8, 16).bfloat16())
    a = HybridOptimizer([dict(params=[p], kind='muon')])
    p.grad = torch.randn_like(p)
    a.step()
    saved = copy.deepcopy(a.state_dict())
    q = torch.nn.Parameter(p.detach().clone())
    b = HybridOptimizer([dict(params=[q], kind='muon')])
    b.load_state_dict(saved)
    assert torch.equal(a.state[p]['master'], b.state[q]['master'])
    p.grad = torch.randn_like(p)
    q.grad = p.grad.clone()
    a.step()
    b.step()
    assert torch.equal(p, q)
    assert torch.equal(a.state[p]['master'], b.state[q]['master'])


def test_dspark_embedding_and_prediction_head_use_sinkhorn():
    from v41_mini import ModelConfig
    from v41_mini.dspark import DSpark
    from v41_mini.optim import parameter_groups
    model = DSpark(ModelConfig.tiny())
    groups = parameter_groups(model, 2e-4)
    kinds = {id(p): g['kind'] for g in groups for p in g['params']}
    assert kinds[id(model.markov_embed.weight)] == 'sinkhorn'
    assert kinds[id(model.markov_head.weight)] == 'sinkhorn'
