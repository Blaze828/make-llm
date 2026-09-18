"""Analytical budget estimate only: no torch import, no allocations, no training."""
import argparse
import json
import math
from pathlib import Path


def estimate(config, tokens, batch, sequence, accumulation, measured_tps=None):
    d,h,l = config["hidden_size"],config["head_dim"],config["num_hidden_layers"]
    q,k,f,v = config["num_attention_heads"],config["num_key_value_heads"],config["intermediate_size"],config["vocab_size"]
    p = v*d*(1 if config["tie_word_embeddings"] else 2)+l*(2*d*q*h+2*d*k*h+3*d*f+2*d+(2*h if config["qk_norm"]!="none" else 0))+d
    return {"parameters":p,"fp32_parameters_grad_adam_gib":16*p/2**30,
            "bf16_inference_weights_gib":2*p/2**30,"bf16_inference_kv_gib":2*l*batch*sequence*k*h*2/2**30,
            "allocated_input_tokens_per_update":batch*sequence*accumulation,
            "minimum_updates_at_full_utilization":math.ceil(tokens/(batch*sequence*accumulation)),
            "rough_dense_training_flops_6ND":6*p*tokens,
            "hours_at_user_measured_target_tokens_per_second":tokens/measured_tps/3600 if measured_tps else None,
            "limitations":["training state estimate excludes activations, logits, workspaces and allocator",
                           "6ND omits attention length-dependent terms; not a timing prediction",
                           "padding and document boundaries reduce valid targets per update"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/architecture/base-316m.json")
    parser.add_argument("--tokens",required=True,type=int)
    parser.add_argument("--batch",type=int,default=1)
    parser.add_argument("--sequence",type=int,default=1024)
    parser.add_argument("--accumulation",type=int,default=1)
    parser.add_argument("--measured-tokens-per-second",type=float)
    args = parser.parse_args()
    if min(args.tokens,args.batch,args.sequence,args.accumulation)<=0: parser.error("All counts must be positive")
    if args.measured_tokens_per_second is not None and args.measured_tokens_per_second<=0: parser.error("Throughput must be positive")
    c = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.sequence>c["max_position_embeddings"]: parser.error("Sequence exceeds context capacity")
    print(json.dumps(estimate(c,args.tokens,args.batch,args.sequence,args.accumulation,args.measured_tokens_per_second),indent=2))


if __name__ == "__main__": main()
