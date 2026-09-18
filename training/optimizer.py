import math
import torch


def build_optimizer(model, learning_rate=3e-4, betas=(0.9, 0.95), epsilon=1e-8, weight_decay=0.1):
    decay, no_decay = [], []
    embedding_id = id(model.embedding.weight)
    for _, param in model.named_parameters():
        if param.requires_grad:
            (no_decay if id(param) == embedding_id or param.ndim < 2 else decay).append(param)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=learning_rate, betas=betas, eps=epsilon)


def build_scheduler(optimizer, total_steps, warmup_ratio=0.01, min_lr_ratio=0.1):
    if total_steps <= 0 or not 0 <= warmup_ratio < 1 or not 0 <= min_lr_ratio <= 1:
        raise ValueError("Invalid learning-rate schedule")
    warmup = max(1, int(total_steps*warmup_ratio)) if warmup_ratio else 0
    def multiplier(step):
        if warmup and step < warmup:
            return (step+1)/warmup
        progress = min(1.0, max(0.0, (step-warmup)/max(1, total_steps-warmup-1)))
        return min_lr_ratio + (1-min_lr_ratio)*0.5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
