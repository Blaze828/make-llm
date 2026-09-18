from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, f = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(d, f, bias=False)
        self.up_proj = nn.Linear(d, f, bias=False)
        self.down_proj = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
