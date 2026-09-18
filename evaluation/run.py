"""Run NLL/BPB, choice likelihood and Korean QA fixtures without optimizer updates."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from model import KoreanLM, ModelConfig
from tokenizer.bpe import BPETokenizer
from adapters import load_adapter
from data_pipeline.common import read_jsonl, sha256_file, write_json
from .metrics import answer_scores, repetition


@torch.no_grad()
def text_nll(model, tokenizer, text):
    ids = [tokenizer.special_id("<|bos|>")]+tokenizer.encode(text)
    limit = model.config.max_position_embeddings
    nll, count = 0.0, 0
    for start in range(0,len(ids)-1,limit-1):
        part = torch.tensor([ids[start:start+limit]],device=next(model.parameters()).device)
        logits = model(part).logits[:, :-1].float()
        nll += F.cross_entropy(logits.reshape(-1,logits.shape[-1]),part[:,1:].reshape(-1),reduction="sum").item()
        count += part.shape[1]-1
    return nll, count, len(text.encode("utf-8"))


@torch.no_grad()
def choice_score(model, tok, prompt, choice):
    prefix = [tok.special_id("<|bos|>")]+tok.encode(prompt)
    suffix = tok.encode(choice)
    if not suffix: raise ValueError("Empty choice")
    if len(prefix)+len(suffix) > model.config.max_position_embeddings:
        raise ValueError("Choice prompt exceeds context; explicit dataset revision required")
    ids = torch.tensor([prefix+suffix],device=next(model.parameters()).device)
    logits = model(ids).logits[0,len(prefix)-1:-1].float()
    # Total continuation log probability, not length-normalized; record this policy.
    return -F.cross_entropy(logits,ids[0,len(prefix):],reduction="sum").item()


def evaluate_records(model,tok,records,max_new_tokens,prediction_path):
    text_totals = defaultdict(lambda:[0.0,0,0])
    tasks = defaultdict(lambda:{"count":0,"score_sum":0.0,"character_f1_sum":0.0,"repetition_sum":0.0,"eos_sum":0})
    seen = set()
    with Path(prediction_path).open("w",encoding="utf-8") as stream:
        for row in records:
            if not isinstance(row.get("id"),str) or row["id"] in seen: raise ValueError("Missing or duplicated evaluation id")
            seen.add(row["id"])
            kind, domain = row.get("type"), row.get("domain","korean_general")
            result = {"id":row["id"],"type":kind,"domain":domain}
            if kind == "text":
                nll,count,byte_count = text_nll(model,tok,row["text"])
                if not count or not byte_count: raise ValueError("Empty text evaluation record")
                target = text_totals[domain]
                target[0] += nll; target[1] += count; target[2] += byte_count
                result.update(nll=nll,tokens=count,utf8_bytes=byte_count)
            elif kind == "multiple_choice":
                choices = row["choices"]
                if not isinstance(choices,list) or len(choices)<2: raise ValueError("At least two choices required")
                answer = row["answer_index"]
                if type(answer) is not int or not 0 <= answer < len(choices): raise ValueError("Invalid answer_index")
                scores = [choice_score(model,tok,row["prompt"],c) for c in choices]
                predicted = max(range(len(scores)),key=scores.__getitem__)
                group = tasks[f"{domain}:multiple_choice"]
                group["count"] += 1; group["score_sum"] += predicted==answer
                result.update(scores=scores,predicted_index=predicted,correct=predicted==answer)
            elif kind in ("qa","generation"):
                prefix = [tok.special_id("<|bos|>")]+tok.encode(row["prompt"])
                if len(prefix)+max_new_tokens > model.config.max_position_embeddings: raise ValueError("Generation context overflow")
                ids = torch.tensor([prefix],device=next(model.parameters()).device)
                output = model.generate(ids,max_new_tokens,eos_token_id=tok.special_id("<|eos|>"))[0,len(prefix):].tolist()
                eos = tok.special_id("<|eos|>") in output
                if eos: output = output[:output.index(tok.special_id("<|eos|>"))]
                text = tok.decode(output)
                group = tasks[f"{domain}:{kind}"]
                group["count"] += 1; group["repetition_sum"] += repetition(text); group["eos_sum"] += eos
                result.update(prediction=text,eos=eos,repetition=repetition(text))
                if kind == "qa":
                    if not isinstance(row.get("answers"),list) or not row["answers"] or not all(isinstance(a,str) for a in row["answers"]): raise ValueError("QA needs reference strings")
                    scores = answer_scores(text,row["answers"])
                    group["score_sum"] += scores["exact_match"]
                    group["character_f1_sum"] += scores["character_f1"]
                    result.update(scores)
            else: raise ValueError(f"Unsupported evaluation type: {kind}")
            stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+"\n")
    if not seen: raise ValueError("Empty evaluation suite")
    task_metrics = {}
    for name, values in tasks.items():
        kind = name.rsplit(":",1)[1]
        count = values["count"]
        metrics = {"count":count}
        if kind == "multiple_choice": metrics["accuracy"] = values["score_sum"]/count
        if kind == "qa":
            metrics.update(exact_match=values["score_sum"]/count, character_f1=values["character_f1_sum"]/count)
        if kind in ("qa","generation"):
            metrics.update(mean_repetition=values["repetition_sum"]/count, eos_rate=values["eos_sum"]/count)
        task_metrics[name] = metrics
    return {"records":len(seen),"text":{domain:{"nll":n,"tokens":t,"utf8_bytes":b,"ce":n/t,
                        "perplexity":math.exp(n/t) if n/t < 700 else None,"bits_per_byte":n/(b*math.log(2))}
                        for domain,(n,t,b) in text_totals.items()},
            "tasks":task_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--tokenizer",required=True)
    parser.add_argument("--suite",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--max-new-tokens",type=int,default=64)
    parser.add_argument("--device",default="cpu")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()): raise ValueError("Use an empty evaluation output directory")
    if args.max_new_tokens < 1: raise ValueError("max-new-tokens must be positive")
    output.mkdir(parents=True,exist_ok=True)
    tok = BPETokenizer.load(args.tokenizer)
    state = torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    if state["run_metadata"]["tokenizer"] != tok.metadata["tokenizer_sha256"]: raise ValueError("Tokenizer mismatch")
    model = KoreanLM(ModelConfig.from_dict(state["config"])).to(args.device)
    model.load_state_dict(state["model"])
    if args.adapter: load_adapter(model,args.adapter,tok.metadata["tokenizer_sha256"])
    model.eval()
    results = evaluate_records(model,tok,read_jsonl(args.suite),args.max_new_tokens,output/"predictions.jsonl")
    results["provenance"] = {"checkpoint_sha256":sha256_file(args.checkpoint),"suite_sha256":sha256_file(args.suite),
                             "tokenizer_sha256":tok.metadata["tokenizer_sha256"],"max_new_tokens":args.max_new_tokens,
                             "choice_scoring":"separately encoded continuation, total log likelihood",
                             "qa_scoring":"NFC/lowercase, remove punctuation/whitespace, character F1",
                             "adapter_manifest_sha256":sha256_file(Path(args.adapter)/"manifest.json") if args.adapter else None}
    write_json(output/"results.json",results)


if __name__ == "__main__": main()
