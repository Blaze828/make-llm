from contextlib import nullcontext
import math
import torch
from .dataloader import collate
from .loss import causal_loss
from .execution import require_training_enabled


def train_update(model, optimizer, scheduler, microbatches, z_loss_weight=0.0, clip=1.0, bf16=False):
    """One device, exact valid-token weighted accumulation, one optimizer update."""
    require_training_enabled()
    device = next(model.parameters()).device
    current_lr = optimizer.param_groups[0]["lr"]
    if bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 execution requires a supported CUDA device")
    counts = []
    for batch in microbatches:
        valid = batch["attention_mask"]
        docs = batch["document_ids"]
        keep = valid[:, :-1] & valid[:, 1:] & (docs[:, :-1] == docs[:, 1:])
        if "loss_mask" in batch:
            keep &= batch["loss_mask"][:, 1:]
        counts.append(int(keep.sum()))
    total = sum(counts)
    if not total:
        raise ValueError("No valid next-token targets")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    ce_total, z_total = 0.0, 0.0
    for batch, count in zip(microbatches, counts):
        if not count:
            continue
        batch = {key:value.to(device) for key,value in batch.items()}
        context = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
        with context:
            output = model(**{k: v for k, v in batch.items() if k != "loss_mask"})
            loss = causal_loss(output.logits, **batch, z_loss_weight=z_loss_weight)
        if not torch.isfinite(loss["loss_sum"]):
            raise FloatingPointError("Nonfinite loss; optimizer not stepped")
        (loss["loss_sum"]/total).backward()
        ce_total += loss["ce_sum"].detach().item()
        z_total += loss["z_sum"].detach().item()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    model.invalidate_cache()
    scheduler.step()
    return {"ce": ce_total/total, "z": z_total/total, "tokens": total,
            "grad_norm": float(grad_norm), "lr": current_lr, "next_lr": optimizer.param_groups[0]["lr"]}


@torch.no_grad()
def evaluate(model, rows, batch_size=1):
    previous = model.training
    model.eval()
    ce, count = 0.0, 0
    try:
        for start in range(0, len(rows), batch_size):
            batch = collate(rows[start:start+batch_size], next(model.parameters()).device)
            loss = causal_loss(model(**{k: v for k, v in batch.items() if k != "loss_mask"}).logits, **batch)
            ce += loss["ce_sum"].item(); count += loss["count"]
    finally:
        model.train(previous)
    if not count:
        raise ValueError("Evaluation has no targets")
    return {"ce": ce/count, "perplexity": math.exp(min(ce/count, 80)), "tokens": count}
