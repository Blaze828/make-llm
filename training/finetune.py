"""Prompt/response SFT with frozen base and LoRA or DoRA; response-only loss."""
import argparse
import json
from pathlib import Path
import torch
from model import KoreanLM, ModelConfig
from tokenizer.bpe import BPETokenizer
from adapters import attach_adapters, save_adapter
from .checkpoint import load_checkpoint
from .dataloader import collate
from .engine import evaluate, train_update
from .optimizer import build_optimizer, build_scheduler


def sft_rows(records, tokenizer, context_length):
    rows = []
    for record in records:
        prefix = [tokenizer.special_id("<|bos|>"), tokenizer.special_id("<|user|>")]
        prefix += tokenizer.encode(record["prompt"])
        prefix += [tokenizer.special_id("<|end_turn|>"), tokenizer.special_id("<|assistant|>")]
        answer = tokenizer.encode(record["response"])+[tokenizer.special_id("<|eos|>")]
        ids = prefix+answer
        if len(ids) > context_length:
            raise ValueError("SFT example exceeds context; refusing silent response truncation")
        pad = context_length-len(ids)
        rows.append({"input_ids": torch.tensor(ids+[tokenizer.special_id("<|pad|>")]*pad),
                     "attention_mask": torch.tensor([True]*len(ids)+[False]*pad),
                     "document_ids": torch.tensor([0]*len(ids)+[-1]*pad),
                     "loss_mask": torch.tensor([False]*len(prefix)+[True]*len(answer)+[False]*pad)})
    if not rows:
        raise ValueError("No SFT records")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train", required=True, help="JSONL with prompt and response strings")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--method", choices=["lora", "dora"], default="lora")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    tok = BPETokenizer.load(args.tokenizer)
    state = torch.load(args.base, map_location="cpu", weights_only=True)
    if state["run_metadata"]["tokenizer"] != tok.metadata["tokenizer_sha256"]:
        raise ValueError("Tokenizer/base mismatch")
    model = KoreanLM(ModelConfig.from_dict(state["config"])).to(args.device)
    if args.sequence_length > model.config.max_position_embeddings:
        raise ValueError("SFT sequence length exceeds model context")
    load_checkpoint(args.base, model)
    def read(path):
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    train, validation = read(args.train), read(args.validation)
    if {r["prompt"] for r in train} & {r["prompt"] for r in validation}:
        raise ValueError("Train/validation prompts overlap")
    rows, val_rows = sft_rows(train, tok, args.sequence_length), sft_rows(validation, tok, args.sequence_length)
    attach_adapters(model, args.rank, args.alpha, args.method, tok.metadata["tokenizer_sha256"])
    optimizer = build_optimizer(model, learning_rate=1e-4)
    scheduler = build_scheduler(optimizer, args.steps)
    for step in range(args.steps):
        metrics = train_update(model, optimizer, scheduler, [collate([rows[step % len(rows)]], args.device)], bf16=args.bf16)
        print(json.dumps({"step": step+1, **metrics}))
    save_adapter(model, args.output)
    print(json.dumps({"validation": evaluate(model, val_rows)}))


if __name__ == "__main__":
    main()
