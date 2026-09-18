"""Unquantized LoRA/DoRA for the named decoder projections."""
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LowRankLinear(nn.Module):
    def __init__(self, base, rank, alpha, dora=False):
        super().__init__()
        self.base = base
        self.enabled = True
        self.scale = alpha/rank
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.magnitude = nn.Parameter(base.weight.detach().float().norm(dim=1).to(base.weight.dtype)) if dora else None

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        if self.magnitude is None:
            return self.base(x) + F.linear(F.linear(x, self.A), self.B)*self.scale
        direction = self.base.weight + (self.B @ self.A)*self.scale
        norm = direction.float().norm(dim=1).detach().clamp_min(1e-8)
        weight = direction * (self.magnitude.float()/norm).to(direction.dtype)[:, None]
        return F.linear(x, weight, self.base.bias)


def base_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def attach_adapters(model, rank=8, alpha=16, method="lora", tokenizer_hash="", targets=TARGETS):
    if hasattr(model, "adapter_manifest"):
        raise ValueError("Model already has adapters")
    if rank <= 0 or alpha <= 0 or method not in ("lora", "dora"):
        raise ValueError("Invalid adapter rank/alpha/method")
    candidates = [(name, module) for name, module in model.named_modules()
                  if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets]
    if not candidates or set(targets) - {name.rsplit(".", 1)[-1] for name, _ in candidates}:
        raise ValueError("Adapter targets did not match model projections")
    manifest = {"version": 1, "method": method, "rank": rank, "alpha": alpha,
                "tokenizer_hash": tokenizer_hash, "base_sha256": base_fingerprint(model),
                "config": model.config.to_dict(), "targets": list(targets),
                "dtype": str(model.embedding.weight.dtype)}
    model.requires_grad_(False)
    for name, module in candidates:
        parent_name, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child, LowRankLinear(module, rank, alpha, method == "dora"))
    model.adapter_manifest = manifest
    model.invalidate_cache()
    return model


def set_adapters_enabled(model, enabled):
    for module in model.modules():
        if isinstance(module, LowRankLinear):
            module.enabled = bool(enabled)
    model.invalidate_cache()


def save_adapter(model, path):
    if not model.adapter_manifest["tokenizer_hash"]:
        raise ValueError("A tokenizer hash is required for a portable adapter")
    state = {name: tensor.detach().cpu() for name, tensor in model.named_parameters() if tensor.requires_grad}
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(state, path / "adapter.pt")
    manifest = dict(model.adapter_manifest)
    manifest["weights_sha256"] = hashlib.sha256((path / "adapter.pt").read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_adapter(model, path, tokenizer_hash):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["tokenizer_hash"] != tokenizer_hash or not tokenizer_hash:
        raise ValueError("Adapter tokenizer mismatch")
    if manifest["config"] != model.config.to_dict() or manifest["base_sha256"] != base_fingerprint(model):
        raise ValueError("Adapter base/config mismatch; load onto its exact original base")
    if manifest["dtype"] != str(model.embedding.weight.dtype):
        raise ValueError("Adapter dtype mismatch")
    if hashlib.sha256((path / "adapter.pt").read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("Adapter weight hash mismatch")
    attach_adapters(model, manifest["rank"], manifest["alpha"], manifest["method"], tokenizer_hash, manifest["targets"])
    state = torch.load(path / "adapter.pt", map_location="cpu", weights_only=True)
    expected = {name for name, param in model.named_parameters() if param.requires_grad}
    if state.keys() != expected:
        raise ValueError("Adapter parameter set mismatch")
    model.load_state_dict(state, strict=False)
    return model
