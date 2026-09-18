import copy
import pytest
import torch
from model import KoreanLM, ModelConfig
from training.loss import causal_loss


def tiny(**kwargs):
    data = dict(vocab_size=48, num_hidden_layers=2, hidden_size=32, num_attention_heads=4,
                num_key_value_heads=2, head_dim=8, intermediate_size=64, max_position_embeddings=32)
    data.update(kwargs)
    return KoreanLM(ModelConfig(**data))


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(17)
    torch.set_num_threads(2)


def test_causality_and_cache():
    model = tiny().eval()
    ids = torch.randint(0, 48, (2, 10))
    full = model(ids).logits
    changed = ids.clone(); changed[:, 6:] = (changed[:, 6:]+1) % 48
    torch.testing.assert_close(full[:, :6], model(changed).logits[:, :6])
    first = model(ids[:, :4], use_cache=True)
    outputs, cache = [first.logits], first.past_key_values
    for i in range(4, 10):
        output = model(ids[:, i:i+1], past_key_values=cache, use_cache=True)
        outputs.append(output.logits); cache = output.past_key_values
    torch.testing.assert_close(full, torch.cat(outputs, 1), atol=2e-6, rtol=2e-5)
    assert cache.layers[0][0].shape == (2, 2, 10, 8)
    with pytest.raises(ValueError, match="capacity"):
        model(torch.ones(2, 23, dtype=torch.long), past_key_values=cache)


def test_packing_and_padding_do_not_leak():
    model = tiny().eval()
    ids = torch.randint(0, 48, (1, 10))
    docs = torch.tensor([[0,0,0,0,1,1,1,1,-1,-1]])
    valid = docs >= 0
    original = model(ids, document_ids=docs, attention_mask=valid).logits
    changed = ids.clone(); changed[:, :4] = 40; changed[:, 8:] = 41
    actual = model(changed, document_ids=docs, attention_mask=valid).logits
    torch.testing.assert_close(original[:, 4:8], actual[:, 4:8])
    single = model(ids[:, 4:8]).logits
    torch.testing.assert_close(single, original[:, 4:8], atol=1e-6, rtol=1e-5)
    loss = causal_loss(original, ids, valid, docs)
    assert loss["count"] == 6
    loss["loss_sum"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_eager_sdpa_outputs_and_gradients():
    eager = tiny(backend="eager")
    sdpa = tiny(backend="sdpa"); sdpa.load_state_dict(eager.state_dict())
    ids = torch.randint(0, 48, (2, 8))
    valid = torch.tensor([[0,0,1,1,1,1,1,1], [1,1,1,1,0,0,0,0]], dtype=torch.bool)
    a, b = eager(ids, valid).logits, sdpa(ids, valid).logits
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    causal_loss(a, ids, valid)["loss_sum"].backward()
    causal_loss(b, ids, valid)["loss_sum"].backward()
    for x, y in zip(eager.parameters(), sdpa.parameters()):
        torch.testing.assert_close(x.grad, y.grad, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("profile", ["debug-40m", "base-316m", "scale-1.2b"])
def test_parameter_count_without_allocating_weights(profile):
    config = ModelConfig.load(f"configs/architecture/{profile}.json")
    with torch.device("meta"):
        model = KoreanLM(config)
    assert model.embedding.weight is model.lm_head.weight
    assert sum(p.numel() for p in model.parameters()) == config.parameter_count()


def test_invalid_config_and_empty_targets():
    with pytest.raises(ValueError):
        ModelConfig(head_dim=7)
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"ffn_type": "gelu"})
    model = tiny()
    ids = torch.ones(1, 1, dtype=torch.long)
    loss = causal_loss(model(ids).logits, ids)
    assert loss["count"] == 0 and loss["loss_sum"].item() == 0


def test_cached_right_padding_keeps_correct_positions():
    model = tiny().eval()
    ids = torch.tensor([[1,2,3,0,0]])
    valid = torch.tensor([[1,1,1,0,0]], dtype=torch.bool)
    cache = model(ids, attention_mask=valid, use_cache=True).past_key_values
    actual = model(torch.tensor([[4]]), past_key_values=cache).logits
    expected = model(torch.tensor([[1,2,3,4]])).logits[:, -1:]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_cache():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unavailable")
    model = tiny().cuda().eval()
    ids = torch.randint(0, 48, (1, 8), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full = model(ids).logits
        first = model(ids[:, :5], use_cache=True)
        rest = model(ids[:, 5:], past_key_values=first.past_key_values).logits
    torch.testing.assert_close(full[:, 5:], rest, atol=0.004, rtol=0.04)
