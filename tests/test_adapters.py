import copy
import pytest
import torch
from adapters import attach_adapters, set_adapters_enabled, save_adapter, load_adapter
from test_model import tiny


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_adapter_gradients_restore_and_cache_invalidation(tmp_path, method):
    torch.manual_seed(3)
    base = tiny().eval()
    weights = copy.deepcopy(base.state_dict())
    ids = torch.tensor([[1,2,3]])
    output = base(ids, use_cache=True)
    attach_adapters(base, rank=2, alpha=4, method=method, tokenizer_hash="tok")
    torch.testing.assert_close(base(ids).logits, output.logits, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="Cache"):
        base(ids[:, -1:], past_key_values=output.past_key_values)
    opt = torch.optim.AdamW([p for p in base.parameters() if p.requires_grad], lr=0.01)
    base(ids).logits.square().mean().backward(); opt.step()
    adapted = base(ids).logits.detach()
    assert not torch.allclose(adapted, output.logits)
    set_adapters_enabled(base, False)
    torch.testing.assert_close(base(ids).logits, output.logits)
    assert base.embedding.weight.grad is None
    set_adapters_enabled(base, True)
    save_adapter(base, tmp_path)
    fresh = tiny().eval(); fresh.load_state_dict(weights)
    load_adapter(fresh, tmp_path, "tok")
    torch.testing.assert_close(fresh(ids).logits, adapted)
    wrong = tiny().eval()
    with pytest.raises(ValueError, match="mismatch"):
        load_adapter(wrong, tmp_path, "tok")
