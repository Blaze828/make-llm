import argparse
import torch
from model import KoreanLM, ModelConfig
from tokenizer.bpe import BPETokenizer
from .checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--adapter", help="LoRA/DoRA artifact directory")
    parser.add_argument("--chat", action="store_true", help="Use the SFT user/assistant template")
    args = parser.parse_args()
    tok = BPETokenizer.load(args.tokenizer)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state["run_metadata"]["tokenizer"] != tok.metadata["tokenizer_sha256"]:
        raise ValueError("Tokenizer/checkpoint hash mismatch")
    model = KoreanLM(ModelConfig.from_dict(state["config"])).to(args.device)
    load_checkpoint(args.checkpoint, model)
    if args.adapter:
        from adapters import load_adapter
        load_adapter(model, args.adapter, tok.metadata["tokenizer_sha256"])
    ids = [tok.special_id("<|bos|>")]
    if args.chat:
        ids += [tok.special_id("<|user|>")]+tok.encode(args.prompt)
        ids += [tok.special_id("<|end_turn|>"), tok.special_id("<|assistant|>")]
    else:
        ids += tok.encode(args.prompt)
    result = model.generate(torch.tensor([ids], device=args.device), args.max_new_tokens,
                            eos_token_id=tok.special_id("<|eos|>"))
    print(tok.decode(result[0, len(ids):].tolist()))


if __name__ == "__main__":
    main()
