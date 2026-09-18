import copy
import torch
from model import KoreanLM
from training.checkpoint import save_checkpoint, load_checkpoint
from training.dataloader import collate, pack_documents
from training.engine import evaluate, train_update
from training.optimizer import build_optimizer, build_scheduler
from test_model import tiny


def batch():
    return collate(pack_documents([[1,2,3,4,5], [2,3,4]], 8), "cpu")


def test_overfit_and_optimizer_groups():
    torch.set_num_threads(2); torch.manual_seed(2)
    model = tiny()
    opt = build_optimizer(model, learning_rate=0.01)
    scheduler = build_scheduler(opt, 30, warmup_ratio=0)
    assert any(p is model.embedding.weight for p in opt.param_groups[1]["params"])
    params = [id(p) for group in opt.param_groups for p in group["params"]]
    assert len(params) == len(set(params))
    rows = pack_documents([[1,2,3,4,5], [2,3,4]], 8)
    before = evaluate(model, rows)["ce"]
    for _ in range(30):
        train_update(model, opt, scheduler, [batch()])
    assert evaluate(model, rows)["ce"] < before * 0.4


def test_resume_matches_uninterrupted(tmp_path):
    torch.manual_seed(4)
    model = tiny(hidden_dropout=0.1)
    opt = build_optimizer(model); scheduler = build_scheduler(opt, 5)
    train_update(model, opt, scheduler, [batch()])
    save_checkpoint(tmp_path/"model.pt", model, opt, scheduler, 1, 2, {"tokenizer": "abc"})
    train_update(model, opt, scheduler, [batch()])
    expected = copy.deepcopy(model.state_dict())
    resumed = tiny(hidden_dropout=0.1)
    opt2 = build_optimizer(resumed); scheduler2 = build_scheduler(opt2, 5)
    state = load_checkpoint(tmp_path/"model.pt", resumed, opt2, scheduler2, {"tokenizer": "abc"})
    assert state["cursor"] == 2
    train_update(resumed, opt2, scheduler2, [batch()])
    for key, value in expected.items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)


def test_uneven_token_accumulation_matches_full_batch():
    torch.manual_seed(8)
    one = tiny(); two = tiny(); two.load_state_dict(one.state_dict())
    rows = pack_documents([[1,2,3,4,5,6,7,8], [4,5,6]], 8)
    a, b = build_optimizer(one), build_optimizer(two)
    sa, sb = build_scheduler(a, 3), build_scheduler(b, 3)
    train_update(one, a, sa, [collate(rows, "cpu")], z_loss_weight=1e-5)
    train_update(two, b, sb, [collate([row], "cpu") for row in rows], z_loss_weight=1e-5)
    for x, y in zip(one.parameters(), two.parameters()):
        torch.testing.assert_close(x, y, atol=3e-6, rtol=1e-4)


def test_sft_loss_only_targets_response():
    from tokenizer.bpe import BPETokenizer
    from training.finetune import sft_rows
    from training.loss import causal_loss
    tok = BPETokenizer.train(["question answer"], 300)
    rows = sft_rows([{"prompt": "question", "response": "answer"}], tok, 32)
    data = collate(rows, "cpu")
    logits = torch.randn(1, 32, tok.vocab_size, requires_grad=True)
    loss = causal_loss(logits, **data)
    assert loss["count"] == len(tok.encode("answer"))+1
    loss["loss_sum"].backward()
    expected = data["loss_mask"][:, 1:]
    assert (logits.grad[:, :-1][~expected] == 0).all()
