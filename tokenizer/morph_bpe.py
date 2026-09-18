"""MeCab-ko surface boundaries are used only while learning the vocabulary."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from .bpe import BPETokenizer


class MecabSurfaceSplitter:
    def __init__(self):
        try:
            import mecab_ko
        except ImportError as error:
            raise RuntimeError("Install the Korean analyzer: pip install mecab-ko==1.0.2") from error
        self.tagger = mecab_ko.Tagger()
        dictionary = Path(self.tagger.dictionary_info().filename).parent
        digest = hashlib.sha256()
        for file in sorted(dictionary.glob("*.bin")) + sorted(dictionary.glob("*.dic")):
            digest.update(file.name.encode())
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(chunk)
        self.metadata = {"package": "mecab-ko", "version": importlib.metadata.version("mecab-ko"),
                         "dictionary_version": importlib.metadata.version("mecab-ko-dic"),
                         "dictionary_sha256": digest.hexdigest(), "fallback_documents": 0}

    def __call__(self, text):
        # MeCab returns surface forms, not lemmatized feature strings. Preserve gaps verbatim.
        node = self.tagger.parseToNode(text)
        pieces, cursor = [], 0
        while node:
            surface = node.surface
            if surface:
                start = text.find(surface, cursor)
                if start < 0 or text[cursor:start].strip():
                    self.metadata["fallback_documents"] += 1
                    return [text]
                if start > cursor:
                    pieces.append(text[cursor:start])
                pieces.append(text[start:start+len(surface)])
                cursor = start+len(surface)
            node = node.next
        if text[cursor:].strip():
            self.metadata["fallback_documents"] += 1
            return [text]
        if cursor < len(text):
            pieces.append(text[cursor:])
        return pieces


def main():
    parser = argparse.ArgumentParser(description="Learn Korean morphology-aware BPE from train-only UTF-8 text")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", help="UTF-8 text, one document per line")
    inputs.add_argument("--corpus-manifest", help="Prepared corpus; vocabulary learns from train split only")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/tokenizer/korean-morph-bpe-32k.json")
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--plain", action="store_true", help="Explicit ordinary byte-BPE ablation")
    args = parser.parse_args()
    from training.execution import require_training_enabled
    require_training_enabled()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    splitter = None if args.plain else MecabSurfaceSplitter()
    if args.corpus_manifest:
        from data_pipeline.common import artifact_path, read_jsonl, sha256_file
        manifest_path = Path(args.corpus_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        info = manifest["splits"]["train"]
        source = artifact_path(manifest_path.parent, info["path"])
        if sha256_file(source) != info["sha256"]: raise ValueError("Train split hash mismatch")
        def texts():
            for row in read_jsonl(source):
                if row.get("split") != "train": raise ValueError("Tokenizer received non-training data")
                yield row["text"]
        tok = BPETokenizer.train(texts(), args.vocab_size or config["vocab_size"], splitter, config["special_tokens"])
        tok.metadata["corpus_manifest_sha256"] = sha256_file(manifest_path)
    else:
        with Path(args.text).open(encoding="utf-8", newline="") as stream:
            tok = BPETokenizer.train(stream, args.vocab_size or config["vocab_size"], splitter, config["special_tokens"])
    tok.save(args.output)
    print(json.dumps(tok.metadata, ensure_ascii=False, indent=2))
    if tok.vocab_size != (args.vocab_size or config["vocab_size"]):
        print("Corpus exhausted before target vocabulary. Use actual_vocab_size in the model config.")


if __name__ == "__main__":
    main()
