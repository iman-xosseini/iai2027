"""Temporal predictor g_phi: four historical latents -> predicted latent of the next block."""
from torch import nn

from .mlp import build_mlp


class TemporalPredictor(nn.Module):
    """MLP predictor z_hat_{t+1} = g_phi(z_{t-3}, z_{t-2}, z_{t-1}, z_t): 128 -> 128 -> 64 -> 32.

    The history is concatenated as [z_{t-3} | z_{t-2} | z_{t-1} | z_t], so each time step has its
    own fixed input positions and weights. The predictor never receives the target latent.
    """

    def __init__(self, context_length=4, latent_dim=32, hidden_dims=(128, 64)):
        super().__init__()
        self.context_length = context_length
        self.latent_dim = latent_dim
        self.net = build_mlp(context_length * latent_dim, hidden_dims, latent_dim)

    def forward(self, z_history):
        """z_history: [B, context_length, latent_dim], oldest block first -> [B, latent_dim]."""
        expected = (self.context_length, self.latent_dim)
        if tuple(z_history.shape[1:]) != expected:
            raise ValueError(f"expected history [B, {expected[0]}, {expected[1]}], got {tuple(z_history.shape)}")
        # Row-major flatten keeps chronological order: entries 0-31 are z_{t-3}, ..., 96-127 are z_t.
        return self.net(z_history.reshape(len(z_history), -1))  # [B, 128] -> [B, 32]
