from dataclasses import replace
import torch
import pytest
from v41_mini import ModelConfig, TextMini
from v41_mini.ops import sparse_attention


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    torch.manual_seed(7)


@pytest.fixture(params=[40, 24], ids=['40-layer', '24-layer'])
def small_config(request):
    return ModelConfig.tiny(layers=request.param)


def test_main_parameter_count_without_allocating_weights():
    with torch.device('meta'):
        model = TextMini(ModelConfig.from_json('configs/v41_text_mini.proposal.json'))
    assert sum(p.numel() for p in model.parameters()) == 1_006_125_424
    assert model.embed.weight is not model.head.weight
    assert len(model.blocks) == 40
    assert [i for i, b in enumerate(model.blocks) if b.attention.compressor is not None] == [2, 8, 14, 20]


def test_backward_reaches_indexer_and_encoder_shared_kv(small_config):
    model = TextMini(small_config)
    ids = torch.randint(0, 128, (1, 16))
    out = model(ids, labels=ids)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for i in model.config.index_sources:
        assert model.blocks[i].attention.indexer.query.weight.grad.abs().sum() > 0
    split = model.config.encoder_layers
    assert model.blocks[split].attention.compressor.value.weight.grad.abs().sum() > 0
    assert model.blocks[split-1].moe.shared.up.weight.grad.abs().sum() > 0


def test_single_token_document_keeps_all_parameters_in_backward_graph(small_config):
    model = TextMini(small_config)
    ids = torch.tensor([[4]])
    out = model(ids, labels=ids)
    out.loss.backward()
    assert out.lm_loss.item() == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    _, state = model.prefill(ids)
    logits, _ = model.decode(torch.tensor([[5]]), state)
    torch.testing.assert_close(logits, model(torch.tensor([[4, 5]])).logits[:, -1], rtol=2e-5, atol=2e-5)


def test_generation_context_is_checked_before_work(small_config):
    model = TextMini(replace(small_config, max_seq_len=8)).eval()
    ids = torch.tensor([[4, 5]])
    assert torch.equal(model.generate(ids, max_new_tokens=0), ids)
    with pytest.raises(ValueError, match='context'):
        model.generate(ids, max_new_tokens=7)
    with pytest.raises(ValueError, match='nonnegative'):
        model.generate(ids, max_new_tokens=-1)


@pytest.mark.parametrize('hierarchical', [False, True])
def test_no_future_leakage_including_topk_ties(hierarchical, small_config):
    model = TextMini(replace(small_config, hierarchical_index=hierarchical)).eval()
    ids = torch.randint(0, 128, (1, 24))
    with torch.no_grad():
        full = model(ids).logits
        for n in [7, 8, 9, 15]:
            prefix = model(ids[:, :n]).logits
            torch.testing.assert_close(prefix, full[:, :n], atol=2e-5, rtol=2e-5)
        changed = ids.clone()
        changed[:, 11:] = (changed[:, 11:]+13) % 128
        torch.testing.assert_close(full[:, :11], model(changed).logits[:, :11], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('hierarchical', [False, True])
def test_cached_decode_odd_even_compression_and_window_wrap(hierarchical, small_config):
    model = TextMini(replace(small_config, hierarchical_index=hierarchical)).eval()
    ids = torch.randint(0, 128, (1, 21))
    with torch.no_grad():
        full = model(ids).logits
        logits, state = model.prefill(ids[:, :7])
        torch.testing.assert_close(logits, full[:, 6], atol=2e-5, rtol=2e-5)
        for i in range(7, ids.shape[1]):
            logits, state = model.decode(ids[:, i:i+1], state)
            torch.testing.assert_close(logits, full[:, i], atol=2e-5, rtol=2e-5)
        assert len(state.sources) == 4
        assert len(state.layer_tokens) == small_config.layers
        assert all(v.kv.shape[1] <= model.config.swa_window for v in state.local.values())


def test_bounded_replay_really_skips_decoder_positions(small_config):
    model = TextMini(small_config).eval()
    ids = torch.randint(0, 128, (1, 24))
    fast, state = model.prefill(ids, mode='bounded_replay')
    split = small_config.encoder_layers
    assert state.layer_tokens[:split] == [24]*split
    assert state.layer_tokens[split:] == [8]*(small_config.layers-split)
    assert state.sources[split].kv.shape[1] == 24
    assert state.approximate
    assert torch.isfinite(fast).all()
    logits, state = model.decode(ids[:, :1], state)
    assert torch.isfinite(logits).all()
    full, _ = model.prefill(ids[:, :6])
    fast, state = model.prefill(ids[:, :6], mode='bounded_replay')
    assert not state.approximate
    torch.testing.assert_close(fast, full, atol=2e-5, rtol=2e-5)


def test_packed_documents_reset_attention_compressor_and_engram(small_config):
    model = TextMini(small_config).eval()
    ids = torch.randint(0, 128, (1, 17))
    seg = torch.tensor([[0]*7+[1]*8+[-1]*2])
    with torch.no_grad():
        packed = model(ids, labels=ids, segment_ids=seg)
        a = model(ids[:, :7], labels=ids[:, :7])
        b = model(ids[:, 7:15], labels=ids[:, 7:15])
    torch.testing.assert_close(packed.logits[:, :7], a.logits, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(packed.logits[:, 7:15], b.logits, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(packed.lm_loss, (a.lm_loss*6+b.lm_loss*7)/13)


def test_checkpoint_recomputation_preserves_gradients(small_config):
    c = small_config
    a, b = TextMini(c), TextMini(replace(c, gradient_checkpointing=True))
    b.load_state_dict(a.state_dict())
    ids = torch.randint(0, 128, (1, 12))
    oa, ob = a(ids, labels=ids), b(ids, labels=ids)
    oa.loss.backward()
    ob.loss.backward()
    torch.testing.assert_close(oa.loss, ob.loss)
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, msg=name)


def test_sparse_attention_matches_dense_mask_and_sink():
    q = torch.randn(1, 4, 2, 8, requires_grad=True)
    kv = torch.randn(1, 6, 8, requires_grad=True)
    ix = torch.tensor([[[0, 1, -1], [1, 3, 4], [2, -1, -1], [0, 4, 5]]])
    sink = torch.randn(2, requires_grad=True)
    sparse = sparse_attention(q, kv, ix, sink, 2)
    score = torch.einsum('bqhd,bkd->bqhk', q, kv)/8**0.5
    mask = torch.zeros(1, 4, 6, dtype=torch.bool)
    for i in range(4):
        mask[0, i, ix[0, i][ix[0, i] >= 0]] = True
    score = score.masked_fill(~mask[:, :, None], -torch.inf)
    score = torch.cat((score, sink[None, None, :, None].expand(1, 4, 2, 1)), -1)
    dense = torch.einsum('bqhk,bkd->bqhd', score.softmax(-1)[..., :-1], kv)
    torch.testing.assert_close(sparse, dense)
    gs = torch.autograd.grad(sparse.square().sum(), (q, kv, sink), retain_graph=True)
    gd = torch.autograd.grad(dense.square().sum(), (q, kv, sink))
    for a, b in zip(gs, gd):
        torch.testing.assert_close(a, b)


def test_mhc_constraints_and_engram_incremental_addresses():
    model = TextMini(ModelConfig.tiny())
    hc = model.blocks[0].hc_attn
    pre, post, comb = hc(torch.randn(1, 5, 4, 32))
    assert (pre > 0).all() and (post >= 0).all() and (comb >= 0).all()
    # Twenty finite Sinkhorn iterations approximate, rather than exactly attain, row balance.
    torch.testing.assert_close(comb.sum(-1), torch.ones(1, 5, 4), atol=2e-4, rtol=0)
    torch.testing.assert_close(comb.sum(-2), torch.ones(1, 5, 4), atol=2e-5, rtol=2e-5)
    ids = torch.randint(0, 128, (1, 13))
    engram = model.blocks[1].engram
    full, _ = engram.addresses(ids)
    a, history = engram.addresses(ids[:, :7])
    b, _ = engram.addresses(ids[:, 7:], history)
    assert torch.equal(full, torch.cat((a, b), 1))
