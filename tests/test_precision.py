import torch
from v41_mini import ModelConfig, TextMini
from v41_mini.optim import HybridOptimizer, parameter_groups
from v41_mini.experts import MoE


def test_router_bias_stays_fp32_and_small_updates_survive_conversion():
    model = MoE(ModelConfig.tiny())
    with torch.no_grad():
        model.routing_bias.copy_(torch.tensor([1.0001, 1.0001, -1.0001, -1.0001]))
    before = model.routing_bias.clone()
    model.bfloat16()
    assert model.router.weight.dtype == torch.bfloat16
    assert model.routing_bias.dtype == torch.float32
    torch.testing.assert_close(model.routing_bias, before, rtol=0, atol=0)
    model.update_bias(torch.tensor([1., 1., 3., 3.]), 0.001)
    torch.testing.assert_close(model.routing_bias, before+torch.tensor([.001, .001, -.001, -.001]))


def test_bf16_autocast_training_uses_fp32_optimizer_state():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    model = TextMini(ModelConfig.tiny()).bfloat16()
    optimizer = HybridOptimizer(parameter_groups(model, 1e-3))
    ids = torch.randint(0, 128, (1, 12))
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        out = model(ids, labels=ids)
    assert out.loss.dtype == torch.float32 and torch.isfinite(out.loss)
    out.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert all(state['master'].dtype == torch.float32 and state['momentum'].dtype == torch.float32
               for state in optimizer.state.values())


def test_repeated_example_loss_decreases():
    torch.set_num_threads(2)
    torch.manual_seed(12)
    model = TextMini(ModelConfig.tiny())
    optimizer = HybridOptimizer(parameter_groups(model, 0.002, optimizer='adamw'))
    ids = torch.arange(12)[None]+4
    losses = []
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        out = model(ids, labels=ids)
        losses.append(out.lm_loss.item())
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    assert losses[-1] < losses[0]*0.75, losses
