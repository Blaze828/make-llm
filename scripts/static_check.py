"""Parse source/configuration only. Does not import project modules or allocate a model."""
import ast
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    counts = {"python":0,"json":0,"jsonl_records":0}
    for folder in ("model","tokenizer","training","adapters","data_pipeline","evaluation","scripts","tests"):
        for file in (root/folder).rglob("*.py"):
            ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
            counts["python"] += 1
    for file in (root/"configs").rglob("*.json"):
        json.loads(file.read_text(encoding="utf-8"))
        counts["json"] += 1
    for file in (root/"evaluation/examples").glob("*.jsonl"):
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                assert isinstance(json.loads(line),dict), file
                counts["jsonl_records"] += 1
    for file in (root/"configs/architecture").glob("*.json"):
        c = json.loads(file.read_text(encoding="utf-8"))
        assert c["hidden_size"] == c["num_attention_heads"]*c["head_dim"], file
        assert c["num_attention_heads"] % c["num_key_value_heads"] == 0, file
        assert c["head_dim"] % 2 == 0, file
    print(json.dumps({"static_parsing":"passed", **counts, "training_executed":False}))


if __name__ == "__main__": main()
