"""Shared mask-conditioned encoder f_theta: one 6-hour sensor block -> one 32-dimensional latent."""
import torch
from torch import nn

from .mlp import build_mlp


def mask_conditioned_input(values, mask):
    """u = [flat(x * m), flat(m)] for every block.

    values, mask : [..., block_size, num_sensors], e.g. [6, 4], [B, 6, 4] or [B, 4, 6, 4]
    returns      : [..., 2 * block_size * num_sensors], e.g. [..., 48]

    Both halves are flattened with the same row-major order, hour first and sensor second:
    entry h * num_sensors + s is (hour h, sensor s) in the value half, and 24 + h * num_sensors + s
    is the same reading's availability in the mask half.
    """
    if values.shape != mask.shape:
        raise ValueError(f"values {tuple(values.shape)} and mask {tuple(mask.shape)} must have the same shape")
    flat_values = (values * mask).flatten(start_dim=-2)  # [..., 24]; unavailable readings become 0
    flat_mask = mask.flatten(start_dim=-2)               # [..., 24]
    return torch.cat([flat_values, flat_mask], dim=-1)   # [..., 48]


class Encoder(nn.Module):
    """MLP encoder z = f_theta(u): 48 -> 64 -> 64 -> 32 (GELU hidden activations, linear output).

    One instance is shared by the four historical blocks and the future target block.
    """

    def __init__(self, input_dim=48, hidden_dims=(64, 64), latent_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.net = build_mlp(input_dim, hidden_dims, latent_dim)

    def forward(self, u):
        """u: [N, input_dim] mask-conditioned inputs -> [N, latent_dim] latents."""
        if u.shape[-1] != self.input_dim:
            raise ValueError(f"expected {self.input_dim} input features, got {u.shape[-1]}")
        return self.net(u)

    def encode_blocks(self, values, mask):
        """Encode every 6-hour block independently with the same parameters.

        values, mask : [..., block_size, num_sensors], e.g. [B, 4, 6, 4] (history) or [B, 6, 4] (target)
        returns      : [..., latent_dim], e.g. [B, 4, 32] or [B, 32]
        """
        u = mask_conditioned_input(values, mask)  # [..., 48]
        leading = u.shape[:-1]                    # (B, 4) or (B,)
        z = self(u.reshape(-1, self.input_dim))   # [B * 4, 48] -> [B * 4, 32] (or [B, 48] -> [B, 32])
        return z.reshape(*leading, self.latent_dim)
