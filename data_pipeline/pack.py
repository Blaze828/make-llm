"""Stream an already-trained tokenizer over corpus JSONL into fixed-size disk rows."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import numpy as np
from .common import artifact_path, read_jsonl, sha256_file, write_json

DTYPES = {"input_ids":"<i4", "document_ids":"<i8", "attention_mask":"u1"}


def iter_packed(documents, length, pad_id):
    if length < 2: raise ValueError("Sequence length must be >=2")
    ids, groups = [], []
    def row():
        n = len(ids)
        return {"input_ids":ids+[pad_id]*(length-n), "document_ids":groups+[-1]*(length-n),
                "attention_mask":[1]*n+[0]*(length-n)}
    for number, tokens in enumerate(documents):
        if len(tokens) < 2: continue
        if ids and len(ids)+len(tokens) > length:
            yield row(); ids, groups = [], []
        start = 0
        while len(tokens)-start > length:
            ids, groups = tokens[start:start+length], [number]*length
            yield row(); ids, groups = [], []
            start += length-1
        remaining = tokens[start:]
        ids.extend(remaining); groups.extend([number]*len(remaining))
        if len(ids) == length:
            yield row(); ids, groups = [], []
    if ids: yield row()


def pack_split(corpus_manifest, split, tokenizer_path, output, length):
    from tokenizer.bpe import BPETokenizer
    manifest_path = Path(corpus_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    info = manifest["splits"][split]
    source = artifact_path(manifest_path.parent, info["path"])
    if sha256_file(source) != info["sha256"]: raise ValueError("Corpus split hash mismatch")
    output = Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError("Output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    tok = BPETokenizer.load(tokenizer_path)
    def documents():
        for record in read_jsonl(source):
            if record.get("split") != split: raise ValueError("Mixed split input")
            yield [tok.special_id("<|bos|>")]+tok.encode(record["text"])+[tok.special_id("<|eos|>")]
    rows, targets = 0, 0
    with ExitStack() as stack:
        files = {key:stack.enter_context((output/f"{key}.bin").open("wb")) for key in DTYPES}
        for row in iter_packed(documents(), length, tok.special_id("<|pad|>")):
            for key, dtype in DTYPES.items(): np.asarray(row[key], dtype=dtype).tofile(files[key])
            targets += sum(bool(row["attention_mask"][i] and row["attention_mask"][i+1]
                                and row["document_ids"][i] == row["document_ids"][i+1]) for i in range(length-1))
            rows += 1
    if not rows or not targets: raise ValueError("Empty split or no prediction targets")
    result = {"format_version":1,"kind":"packed_tokens","split":split,"rows":rows,"sequence_length":length,
              "valid_targets":targets,"vocab_size":tok.vocab_size,"tokenizer_sha256":tok.metadata["tokenizer_sha256"],
              "corpus_manifest_sha256":sha256_file(manifest_path), "source_sha256":info["sha256"],
              "arrays":{key:{"path":f"{key}.bin","dtype":dtype,"sha256":sha256_file(output/f"{key}.bin")}
                        for key,dtype in DTYPES.items()}}
    write_json(output/"manifest.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-manifest", required=True)
    parser.add_argument("--split", choices=["train","validation","test"], required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-length", type=int, default=1024)
    args = parser.parse_args()
    pack_split(args.corpus_manifest,args.split,args.tokenizer,args.output,args.sequence_length)


if __name__ == "__main__": main()
