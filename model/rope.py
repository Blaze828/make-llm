import torch


def apply_rope(q, k, positions, theta):
    """Split-half RoPE; positions [B,T], q/k [B,H,T,D]."""
    dim = q.shape[-1]
    with torch.autocast(device_type=q.device.type, enabled=False):
        inv = theta ** (-torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim)
        angles = positions.float().unsqueeze(-1) * inv
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        cos, sin = angles.cos(), angles.sin()

    def rotate(x):
        left, right = x.chunk(2, dim=-1)
        return x * cos.to(x.dtype) + torch.cat((-right, left), -1) * sin.to(x.dtype)

    return rotate(q), rotate(k)
