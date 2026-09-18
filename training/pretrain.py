"""Executable, bounded single-device pretraining and exact same-run resume."""
import argparse
import json
from pathlib import Path
import random
import time
import torch
from model import KoreanLM, ModelConfig
from tokenizer.bpe import BPETokenizer
from .checkpoint import fingerprint, load_checkpoint, save_checkpoint
from .dataloader import collate, pack_documents
from .engine import evaluate, train_update
from .optimizer import build_optimizer, build_scheduler
from .execution import require_training_enabled, require_single_process
from .indexed_data import IndexedRows, shuffled_index
from data_pipeline.common import write_json


def main():
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", help="Small-corpus legacy mode: one document per line")
    inputs.add_argument("--train-manifest", help="Packed-token manifest for bounded-memory training")
    parser.add_argument("--validation-text")
    parser.add_argument("--validation-manifest")
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
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log", help="JSONL event path; defaults beside checkpoint")
    args = parser.parse_args()
    require_training_enabled()
    require_single_process()
    if min(args.save_every,args.eval_every) <= 0: parser.error("Intervals must be positive")
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
    if args.train_manifest:
        if not args.validation_manifest or args.validation_text:
            parser.error("Manifest mode requires --validation-manifest only")
        train_rows, val_rows = IndexedRows(args.train_manifest), IndexedRows(args.validation_manifest)
        for rows, split in ((train_rows,"train"),(val_rows,"validation")):
            m = rows.manifest
            if m["split"] != split or m["tokenizer_sha256"] != tok.metadata["tokenizer_sha256"]:
                raise ValueError("Dataset split/tokenizer mismatch")
            if m["vocab_size"] != config.vocab_size or rows.length != args.sequence_length:
                raise ValueError("Dataset vocabulary/sequence length mismatch")
        if train_rows.manifest["corpus_manifest_sha256"] != val_rows.manifest["corpus_manifest_sha256"]:
            raise ValueError("Train and validation must come from the same grouped-split corpus")
        train_hash, val_hash = train_rows.fingerprint, val_rows.fingerprint
    else:
        if not args.validation_text or args.validation_manifest:
            parser.error("Text mode requires --validation-text only")
        train_text = Path(args.text).read_text(encoding="utf-8").splitlines(keepends=True)
        validation = Path(args.validation_text).read_text(encoding="utf-8").splitlines(keepends=True)
        if set(train_text) & set(validation): raise ValueError("Exact train/validation document overlap")
        def rows(texts):
            docs = [[tok.special_id("<|bos|>")]+tok.encode(text)+[tok.special_id("<|eos|>")] for text in texts if text.strip()]
            return pack_documents(docs,args.sequence_length,tok.special_id("<|pad|>"))
        train_rows, val_rows = rows(train_text), rows(validation)
        train_hash, val_hash = fingerprint(train_text), fingerprint(validation)
    if not train_rows or not val_rows:
        raise ValueError("Both train and validation must have usable documents")
    random.seed(args.seed); torch.manual_seed(args.seed)
    model = KoreanLM(config).to(args.device)
    model.set_gradient_checkpointing(args.gradient_checkpointing)
    opt_config = {k: v for k, v in recipe["optimizer"].items()
                  if k in ("learning_rate", "betas", "epsilon", "weight_decay")}
    optimizer = build_optimizer(model, **opt_config)
    scheduler = build_scheduler(optimizer, args.steps, recipe["scheduler"]["warmup_ratio"],
                                recipe["scheduler"]["min_lr_ratio"])
    metadata = {"tokenizer": tok.metadata["tokenizer_sha256"], "train": train_hash,
                "validation": val_hash, "recipe": fingerprint(recipe),
                "steps": args.steps, "batch_size": args.batch_size, "accumulation": args.accumulation,
                "sequence_length": args.sequence_length, "seed": args.seed, "bf16": args.bf16,
                "device": args.device, "sampler":"affine-v1", "gradient_checkpointing":args.gradient_checkpointing,
                "eval_every":args.eval_every}
    step, cursor = 0, 0
    progress = {"valid_tokens_seen":0, "best_validation_ce":None}
    output = Path(args.output)
    log_path = Path(args.log) if args.log else output.with_suffix(".events.jsonl")
    if not args.resume and (output.exists() or log_path.exists()):
        raise ValueError("Existing run output/log: use --resume or a new path")
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, metadata)
        step, cursor = state["step"], state["cursor"]
        progress.update(state.get("progress", {}))
    stop = args.stop_after or args.steps
    log_path.parent.mkdir(parents=True, exist_ok=True)
    def save(path): save_checkpoint(path,model,optimizer,scheduler,step,cursor,metadata,progress)
    def emit(event):
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event,allow_nan=False)+"\n")
        print(json.dumps(event))
    # Also establish a valid recovery point when resuming to a new output path.
    save(output)
    emit({"event":"start" if not args.resume else "resume", "step":step,"metadata":metadata})
    try:
        while step < stop:
            start = time.perf_counter()
            batches = []
            for _ in range(args.accumulation):
                indices = [shuffled_index(cursor+i,len(train_rows),args.seed) for i in range(args.batch_size)]
                cursor += args.batch_size
                # Accumulation keeps queued microbatches on CPU, not all in VRAM.
                batches.append(collate([train_rows[i] for i in indices], "cpu"))
            metrics = train_update(model, optimizer, scheduler, batches,
                                   recipe["z_loss"]["coefficient"], recipe["gradient_clip_global_norm"], args.bf16)
            step += 1
            progress["valid_tokens_seen"] += metrics["tokens"]
            elapsed = time.perf_counter()-start
            emit({"event":"train","step":step,**metrics,"seconds":elapsed,
                  "tokens_per_second":metrics["tokens"]/elapsed,"valid_tokens_seen":progress["valid_tokens_seen"]})
            if step % args.eval_every == 0 or step == stop:
                result = evaluate(model,val_rows)
                emit({"event":"validation","step":step,**result})
                best = progress["best_validation_ce"]
                if best is None or result["ce"] < best:
                    progress["best_validation_ce"] = result["ce"]
                    save(output.with_name(output.stem+".best"+output.suffix))
            if step % args.save_every == 0: save(output)
        save(output)
        write_json(output.with_suffix(".status.json"), {"status":"completed" if step==args.steps else "stopped",
                                                       "step":step,"progress":progress})
    except (Exception, KeyboardInterrupt) as error:
        # Never overwrite a valid checkpoint with a partially executed optimizer update.
        write_json(output.with_suffix(".status.json"), {"status":"failed_or_interrupted","last_completed_step":step,
                                                       "resume_checkpoint":str(output),"error":type(error).__name__})
        raise


if __name__ == "__main__":
    main()
