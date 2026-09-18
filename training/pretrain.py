"""Executable, bounded single-device pretraining and exact same-run resume."""
import argparse
import json
from pathlib import Path
import random
import torch
from model import KoreanLM, ModelConfig
from tokenizer.bpe import BPETokenizer
from .checkpoint import fingerprint, load_checkpoint, save_checkpoint
from .dataloader import collate, pack_documents
from .engine import evaluate, train_update
from .optimizer import build_optimizer, build_scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Train UTF-8 text, one document per line")
    parser.add_argument("--validation-text", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--recipe", default="configs/training/korean-base.json")
    parser.add_argument("--model-config")
    parser.add_argument("--steps", required=True, type=int, help="Total optimizer-step budget")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int, help="Stop early while preserving the full schedule")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.accumulation) <= 0:
        parser.error("steps, batch size and accumulation must be positive")
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        parser.error("stop-after must be in [1,steps]")
    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    config = ModelConfig.load(args.model_config or recipe["model_config"])
    tok = BPETokenizer.load(args.tokenizer)
    if config.vocab_size != tok.vocab_size:
        raise ValueError(f"Model vocab {config.vocab_size} != actual tokenizer vocab {tok.vocab_size}")
    if not 2 <= args.sequence_length <= config.max_position_embeddings:
        raise ValueError("Sequence length exceeds model capacity or is <2")
    train_text = Path(args.text).read_text(encoding="utf-8").splitlines(keepends=True)
    validation = Path(args.validation_text).read_text(encoding="utf-8").splitlines(keepends=True)
    if set(train_text) & set(validation):
        raise ValueError("Exact train/validation document overlap detected")
    def rows(texts):
        docs = [[tok.special_id("<|bos|>")]+tok.encode(text)+[tok.special_id("<|eos|>")] for text in texts if text.strip()]
        return pack_documents(docs, args.sequence_length, tok.special_id("<|pad|>"))
    train_rows, val_rows = rows(train_text), rows(validation)
    if not train_rows or not val_rows:
        raise ValueError("Both train and validation must have usable documents")
    random.seed(args.seed); torch.manual_seed(args.seed)
    model = KoreanLM(config).to(args.device)
    opt_config = {k: v for k, v in recipe["optimizer"].items()
                  if k in ("learning_rate", "betas", "epsilon", "weight_decay")}
    optimizer = build_optimizer(model, **opt_config)
    scheduler = build_scheduler(optimizer, args.steps, recipe["scheduler"]["warmup_ratio"],
                                recipe["scheduler"]["min_lr_ratio"])
    metadata = {"tokenizer": tok.metadata["tokenizer_sha256"], "train": fingerprint(train_text),
                "validation": fingerprint(validation), "recipe": fingerprint(recipe),
                "steps": args.steps, "batch_size": args.batch_size, "accumulation": args.accumulation,
                "sequence_length": args.sequence_length, "seed": args.seed, "bf16": args.bf16,
                "device": args.device}
    step, cursor = 0, 0
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, metadata)
        step, cursor = state["step"], state["cursor"]
    stop = args.stop_after or args.steps
    while step < stop:
        batches = []
        for _ in range(args.accumulation):
            indices = [(cursor+i) % len(train_rows) for i in range(args.batch_size)]
            cursor += args.batch_size
            batches.append(collate([train_rows[i] for i in indices], args.device))
        metrics = train_update(model, optimizer, scheduler, batches,
                               recipe["z_loss"]["coefficient"], recipe["gradient_clip_global_norm"], args.bf16)
        step += 1
        print(json.dumps({"step": step, **metrics}))
        if step % 100 == 0:
            save_checkpoint(args.output, model, optimizer, scheduler, step, cursor, metadata)
    save_checkpoint(args.output, model, optimizer, scheduler, step, cursor, metadata)
    print(json.dumps({"validation": evaluate(model, val_rows), "checkpoint": args.output}))


if __name__ == "__main__":
    main()
