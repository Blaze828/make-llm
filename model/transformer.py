from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .attention import Attention
from .cache import KVCache
from .config import ModelConfig
from .mlp import SwiGLU
from .norm import RMSNorm


@dataclass
class ModelOutput:
    logits: torch.Tensor
    past_key_values: KVCache | None = None


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = Attention(config)
        self.mlp = SwiGLU(config)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x, positions, allowed, past, use_cache):
        value, cache = self.attention(self.attn_norm(x), positions, allowed, past, use_cache)
        x = x + self.dropout(value)
        return x + self.dropout(self.mlp(self.ffn_norm(x))), cache


class KoreanLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight
        self._cache_owner = object()
        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, enabled=True):
        self.gradient_checkpointing = bool(enabled)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def invalidate_cache(self):
        self._cache_owner = object()

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self.invalidate_cache()
        return result

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                document_ids=None, past_key_values=None, use_cache=False):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be nonempty [B,T]")
        if self.training and (use_cache or past_key_values is not None):
            raise ValueError("KV caching is inference-only; call eval() first")
        b, t = input_ids.shape
        device = input_ids.device
        past = past_key_values
        if past is not None and (past.owner is not self._cache_owner or len(past.layers) != len(self.layers)):
            raise ValueError("Cache belongs to another model/adapter revision")
        s = past.length if past else 0
        if s+t > self.config.max_position_embeddings:
            raise ValueError("Context capacity exceeded")
        if past is not None and (past.positions.shape[0] != b or past.positions.device != device):
            raise ValueError("Cache batch or device mismatch")
        valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        last_valid_index = None
        if past is not None:
            indices = torch.arange(s, device=device).expand(b, s)
            last_valid_index = indices.masked_fill(~past.valid, -1).max(1).values
            if (last_valid_index < 0).any():
                raise ValueError("Cannot continue an entirely padded cached sequence")
        if document_ids is None:
            docs = (past.document_ids.gather(1, last_valid_index[:, None]).expand(b, t)
                    if past else torch.zeros_like(input_ids))
        else:
            docs = document_ids
        if valid.shape != input_ids.shape or docs.shape != input_ids.shape:
            raise ValueError("Masks/document IDs must describe current input [B,T]")
        if position_ids is None:
            # Reset positions at packed document boundaries; ignore padding in the counter.
            pos = torch.zeros_like(input_ids)
            prev_pos = past.positions.gather(1, last_valid_index[:, None]).squeeze(1) if past else torch.full((b,), -1, device=device)
            prev_doc = past.document_ids.gather(1, last_valid_index[:, None]).squeeze(1) if past else docs[:, 0]
            for i in range(t):
                current = torch.where(docs[:, i] == prev_doc, prev_pos+1, 0)
                pos[:, i] = torch.where(valid[:, i], current, 0)
                prev_pos = torch.where(valid[:, i], current, prev_pos)
                prev_doc = torch.where(valid[:, i], docs[:, i], prev_doc)
        else:
            pos = position_ids
        if pos.shape != input_ids.shape or (pos < 0).any() or (pos >= self.config.max_position_embeddings).any():
            raise ValueError("Invalid position_ids")
        keys_valid = torch.cat((past.valid, valid), 1) if past else valid
        keys_docs = torch.cat((past.document_ids, docs), 1) if past else docs
        keys_pos = torch.cat((past.positions, pos), 1) if past else pos
        # Sequence indices, not reset RoPE positions, establish the causal order.
        causal = torch.arange(s+t, device=device)[None, :] <= torch.arange(s, s+t, device=device)[:, None]
        allowed = (causal[None] & keys_valid[:, None, :] & valid[:, :, None]
                   & (keys_docs[:, None, :] == docs[:, :, None]))
        x, caches = self.embedding(input_ids), []
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                # Bind the current block: checkpoint recomputation happens after this loop.
                def run(value, block=layer):
                    return block(value, pos, allowed, None, False)[0]
                x = checkpoint(run, x, use_reentrant=False)
                cache = None
            else:
                x, cache = layer(x, pos, allowed, past.layers[i] if past else None, use_cache)
            if use_cache:
                caches.append(cache)
        logits = self.lm_head(self.norm(x))
        cache = KVCache(tuple(caches), keys_pos.detach(), keys_docs.detach(), keys_valid.detach(),
                        self._cache_owner) if use_cache else None
        return ModelOutput(logits, cache)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=32, temperature=0.0, eos_token_id=None):
        if max_new_tokens < 0 or temperature < 0:
            raise ValueError("Generation length/temperature cannot be negative")
        if input_ids.shape[1] + max_new_tokens > self.config.max_position_embeddings:
            raise ValueError("Requested generation exceeds context capacity")
        training = self.training
        self.eval()
        try:
            result, cache = input_ids, None
            done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            for _ in range(max_new_tokens):
                out = self(result if cache is None else result[:, -1:], past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                logits = out.logits[:, -1].float()
                nxt = logits.argmax(-1) if temperature == 0 else torch.multinomial((logits/temperature).softmax(-1), 1).squeeze(-1)
                if eos_token_id is not None:
                    nxt = torch.where(done, eos_token_id, nxt)
                    done |= nxt == eos_token_id
                result = torch.cat((result, nxt[:, None]), 1)
                if done.all():
                    break
            return result
        finally:
            self.train(training)
