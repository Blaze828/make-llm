"""Compare existing tokenizer artifacts; never fits a vocabulary."""
import argparse
from collections import defaultdict
import json
import time
from data_pipeline.common import read_jsonl, sha256_file, write_json
from .bpe import BPETokenizer


def benchmark(tok, source):
    totals = defaultdict(lambda:{"documents":0,"tokens":0,"utf8_bytes":0,"characters":0,"roundtrip_failures":0,"seconds":0.0})
    for row in read_jsonl(source):
        text = row["text"]
        if not text: continue
        domain = row.get("domain","unknown")
        start = time.perf_counter(); ids = tok.encode(text); elapsed = time.perf_counter()-start
        group = totals[domain]
        group["documents"] += 1; group["tokens"] += len(ids); group["utf8_bytes"] += len(text.encode())
        group["characters"] += len(text); group["roundtrip_failures"] += tok.decode(ids)!=text
        group["seconds"] += elapsed
    for group in totals.values():
        group["tokens_per_byte"] = group["tokens"]/group["utf8_bytes"]
        group["bytes_per_second"] = group["utf8_bytes"]/max(group["seconds"],1e-12)
    return dict(totals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer",action="append",required=True,help="label=artifact_directory; may repeat")
    parser.add_argument("--input",required=True,help="Held-out JSONL with text/domain")
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    result = {"source_sha256":sha256_file(args.input),"models":{}}
    for spec in args.tokenizer:
        name,path = spec.split("=",1)
        if name in result["models"]: raise ValueError("Duplicate label")
        tok = BPETokenizer.load(path)
        result["models"][name] = {"vocab_size":tok.vocab_size,"tokenizer_sha256":tok.metadata["tokenizer_sha256"],
                                  "domains":benchmark(tok,args.input)}
    write_json(args.output,result)


if __name__ == "__main__": main()
