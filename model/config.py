"""Validated configuration for the dense Korean decoder."""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    num_hidden_layers: int = 24
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 2816
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    qk_norm: str = "headwise_rmsnorm"
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    backend: str = "sdpa"

    def __post_init__(self):
        for key in ("vocab_size", "num_hidden_layers", "hidden_size", "num_attention_heads",
                    "num_key_value_heads", "head_dim", "intermediate_size", "max_position_embeddings"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal query heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError("Invalid GQA ratio or odd RoPE head dimension")
        if self.qk_norm not in ("headwise_rmsnorm", "none") or self.backend not in ("eager", "sdpa"):
            raise ValueError("Unsupported normalization or attention backend")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("Normalization epsilon and RoPE theta must be positive")
        if not all(0 <= p < 1 for p in (self.attention_dropout, self.hidden_dropout)):
            raise ValueError("Dropout must be in [0,1)")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        # Accept the existing design JSON, but never silently ignore architecture changes.
        fixed = {"schema_version": 1, "architecture": "korean_dense_decoder_v1",
                 "attention_type": "full_causal_gqa", "normalization": "pre_rmsnorm",
                 "qk_norm_affine": "shared_across_heads", "rotary_fraction": 1,
                 "rope_scaling": None, "ffn_type": "swiglu", "attention_bias": False,
                 "mlp_bias": False}
        for name, expected in fixed.items():
            if name in data and data.pop(name) != expected:
                raise ValueError(f"Unsupported {name}")
        for name in ("status", "profile"):
            data.pop(name, None)
        unknown = data.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self):
        return asdict(self)

    def parameter_count(self):
        d, h, l = self.hidden_size, self.head_dim, self.num_hidden_layers
        embedding = self.vocab_size * d * (1 if self.tie_word_embeddings else 2)
        return embedding + l * (2*d*self.num_attention_heads*h + 2*d*self.num_key_value_heads*h
                                 + 3*d*self.intermediate_size + 2*d
                                 + (2*h if self.qk_norm != "none" else 0)) + d
