"""Compare saved evaluation reports without loading a model."""
import argparse
import json
from pathlib import Path
from data_pipeline.common import write_json


def compare(before, after):
    for key in ("suite_sha256","max_new_tokens","choice_scoring","qa_scoring"):
        if before["provenance"][key] != after["provenance"][key]:
            raise ValueError(f"Incomparable evaluation protocol: {key}")
    same_tokenizer = before["provenance"]["tokenizer_sha256"] == after["provenance"]["tokenizer_sha256"]
    differences = {}
    for section in ("text","tasks"):
        if before[section].keys() != after[section].keys(): raise ValueError("Domain/task coverage mismatch")
        differences[section] = {}
        for domain, old in before[section].items():
            new = after[section][domain]
            keys = ("bits_per_byte","ce","perplexity") if section=="text" else ("accuracy","exact_match","character_f1","mean_repetition","eos_rate")
            differences[section][domain] = {key:{"before":old[key],"after":new[key],"delta":new[key]-old[key]}
                for key in keys if key in old and key in new and old[key] is not None and new[key] is not None
                and (same_tokenizer or key not in ("ce","perplexity"))}
    return {"delta_convention":"after minus before; lower CE/BPB/repetition is better, higher accuracy/EM/F1 is better; EOS rate is descriptive",
            "same_tokenizer":same_tokenizer,"differences":differences,
            "before_provenance":before["provenance"],"after_provenance":after["provenance"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before",required=True)
    parser.add_argument("--after",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.before,args.after)]
    write_json(args.output,compare(*reports))


if __name__ == "__main__": main()
