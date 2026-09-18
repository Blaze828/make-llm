import math
import torch
from torch import nn
from torch.nn import functional as F
from .norm import RMSNorm
from .rope import apply_rope


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d, h = config.hidden_size, config.head_dim
        self.q_proj = nn.Linear(d, config.num_attention_heads*h, bias=False)
        self.k_proj = nn.Linear(d, config.num_key_value_heads*h, bias=False)
        self.v_proj = nn.Linear(d, config.num_key_value_heads*h, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads*h, d, bias=False)
        self.q_norm = RMSNorm(h, config.rms_norm_eps) if config.qk_norm != "none" else nn.Identity()
        self.k_norm = RMSNorm(h, config.rms_norm_eps) if config.qk_norm != "none" else nn.Identity()

    def forward(self, x, positions, allowed, past=None, use_cache=False):
        b, t, _ = x.shape
        c = self.config
        def heads(proj):
            return proj(x).view(b, t, -1, c.head_dim).transpose(1, 2)
        q, k, v = self.q_norm(heads(self.q_proj)), self.k_norm(heads(self.k_proj)), heads(self.v_proj)
        q, k = apply_rope(q, k, positions, c.rope_theta)
        if past is not None:
            k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
        cache = (k.detach(), v.detach()) if use_cache else None
        # Repetition is temporary; persistent cache retains only KV heads.
        repeats = c.num_attention_heads // c.num_key_value_heads
        k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        mask = allowed.unsqueeze(1)
        dropout = c.attention_dropout if self.training else 0.0
        if c.backend == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout)
        else:
            scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(c.head_dim)
            scores = scores.masked_fill(~mask, -torch.inf)
            # Padding queries have no valid keys. Avoid NaNs in both forward and backward.
            scores = torch.where(mask.any(-1, keepdim=True), scores, torch.zeros_like(scores))
            probs = scores.softmax(-1).masked_fill(~mask, 0).to(v.dtype)
            out = F.dropout(probs, dropout, self.training) @ v
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1)), cache
