from dataclasses import dataclass
import torch


@dataclass
class KVCache:
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    positions: torch.Tensor
    document_ids: torch.Tensor
    valid: torch.Tensor
    owner: object

    @property
    def length(self):
        return self.positions.shape[1]
