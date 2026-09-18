import hashlib
import json
import random
from pathlib import Path
import torch


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_checkpoint(path, model, optimizer, scheduler, step, cursor, run_metadata, progress=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"format_version": 1, "config": model.config.to_dict(), "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "step": step, "cursor": cursor, "run_metadata": run_metadata, "progress": progress or {},
             "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, expected_metadata=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["config"] != model.config.to_dict():
        raise ValueError("Checkpoint model config mismatch")
    if expected_metadata is not None and state["run_metadata"] != expected_metadata:
        raise ValueError("Resume tokenizer/data/schedule/config mismatch")
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is None:
            raise ValueError("Training resume requires scheduler")
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state
