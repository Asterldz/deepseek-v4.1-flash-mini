import torch
import pytest
from v41_mini import ModelConfig, TextMini
from v41_mini.dspark import DSpark


@pytest.mark.parametrize('layers', [40, 24])
def test_dspark_parameter_count_and_frozen_backbone_gradients(layers):
    torch.set_num_threads(2)
    c = ModelConfig.tiny(layers=layers)
    backbone, draft = TextMini(c).eval(), DSpark(c)
    ids = torch.randint(0, c.vocab_size, (1, 14))
    # Backbone graph exists intentionally; the drafter must detach it itself.
    features = backbone(ids[:, :8], collect_features=True).dspark_features
    out = draft(features, ids[:, 8:9], 8, backbone.embed.weight, backbone.head.weight,
                ids[:, 9:14], acceptance_targets=torch.ones(1, 5))
    out['loss'].backward()
    assert all(p.grad is None for p in backbone.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in draft.parameters())
    with torch.device('meta'):
        main_draft = DSpark(ModelConfig() if layers == 40 else ModelConfig.compact())
    assert sum(p.numel() for p in main_draft.parameters()) == (77_457_874 if layers == 40 else 61_708_832)


def test_draft_teacher_forcing_does_not_leak_later_targets():
    torch.manual_seed(8)
    c = ModelConfig.tiny()
    model, draft = TextMini(c).eval(), DSpark(c).eval()
    features = torch.randn(1, 9, 3*c.dim)
    known, targets = torch.randint(0, c.vocab_size, (1, 1)), torch.randint(0, c.vocab_size, (1, 5))
    with torch.no_grad():
        a = draft(features, known, 9, model.embed.weight, model.head.weight, targets)['logits']
        targets[:, 2:] = (targets[:, 2:]+17) % c.vocab_size
        b = draft(features, known, 9, model.embed.weight, model.head.weight, targets)['logits']
    torch.testing.assert_close(a[:, :3], b[:, :3])
