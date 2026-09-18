"""Small self-contained integration run. Corpus is a code fixture, not a Korean benchmark."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from model import ModelConfig
from tokenizer.bpe import BPETokenizer
from tokenizer.morph_bpe import MecabSurfaceSplitter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output", default="checkpoints/smoke")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty output directory to preserve earlier runs")
    out.mkdir(parents=True, exist_ok=True)
    texts = ["한국어 모델은 문장의 다음 토큰을 예측합니다.\n", "교차로에서 신호등은 차량의 통행 순서를 알려 줍니다.\n",
             "빨간 신호에서는 멈추고 초록 신호에서는 주변을 살핍니다.\n", "학습 데이터의 품질과 중복 제거는 중요합니다.\n",
             "도로의 차량 수를 세고 시간에 따른 변화를 분석합니다.\n", "오늘은 맑은 날씨입니다. 내일은 비가 올 수 있습니다.\n"]
    validation = "한국어로 교통 상황을 설명하는 작은 모델을 시험합니다.\n"
    (out/"train.txt").write_text("".join(texts), encoding="utf-8")
    (out/"validation.txt").write_text(validation, encoding="utf-8")
    tok = BPETokenizer.train(texts, 512, MecabSurfaceSplitter())
    tok.save(out/"tokenizer")
    config = ModelConfig(vocab_size=tok.vocab_size, num_hidden_layers=2, hidden_size=64,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         intermediate_size=192, max_position_embeddings=256)
    (out/"model.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    command = [sys.executable, "-X", "utf8", "-m", "training.pretrain", "--text", str(out/"train.txt"),
               "--validation-text", str(out/"validation.txt"), "--tokenizer", str(out/"tokenizer"),
               "--model-config", str(out/"model.json"), "--steps", str(args.steps), "--output", str(out/"model.pt"),
               "--sequence-length", "128", "--device", args.device]
    if args.bf16:
        command.append("--bf16")
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, "-X", "utf8", "-m", "training.generate", "--checkpoint", str(out/"model.pt"),
                    "--tokenizer", str(out/"tokenizer"), "--prompt", "한국어 모델은", "--max-new-tokens", "12",
                    "--device", args.device], check=True)
    print(f"Smoke completed: {config.parameter_count():,} parameters; artifacts: {out.resolve()}")


if __name__ == "__main__":
    main()
