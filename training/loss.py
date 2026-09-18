import torch
from torch.nn import functional as F


def causal_loss(logits, input_ids, attention_mask=None, document_ids=None, z_loss_weight=0.0, loss_mask=None):
    """Return token-summed loss and count so accumulation can normalize exactly once."""
    if z_loss_weight < 0:
        raise ValueError("Z-loss weight cannot be negative")
    valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
    keep = valid[:, :-1] & valid[:, 1:]
    if document_ids is not None:
        keep &= document_ids[:, :-1] == document_ids[:, 1:]
    if loss_mask is not None:
        keep &= loss_mask[:, 1:].bool()
    values, targets = logits[:, :-1].float()[keep], input_ids[:, 1:][keep]
    count = targets.numel()
    if not count:
        zero = logits.float().sum() * 0
        return {"loss_sum": zero, "ce_sum": zero, "z_sum": zero, "count": 0}
    ce = F.cross_entropy(values, targets, reduction="sum")
    z = values.logsumexp(-1).square().sum()
    return {"loss_sum": ce+z_loss_weight*z, "ce_sum": ce, "z_sum": z, "count": count}
