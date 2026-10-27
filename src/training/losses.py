"""Stage 3 objective: latent prediction loss + SIGReg (LeWorldModel, Maes et al. 2026, Eq. 1-3 and Appendix A)."""
import math

import torch
from torch import nn


def prediction_loss(z_pred, z_target):
    """Mean squared error over the batch and all latent dimensions, LeWM's `(pred - tgt).pow(2).mean()`.

    z_target is NOT detached: the target branch is trained through the shared encoder.
    """
    if z_pred.shape != z_target.shape:
        raise ValueError(f"z_pred {tuple(z_pred.shape)} and z_target {tuple(z_target.shape)} must have the same shape")
    return (z_pred - z_target).pow(2).mean()


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer with the Epps-Pulley statistic (LeWM `module.SIGReg`).

    For embeddings Z [T, N, D] (T time positions, N samples, D dimensions) and each of `num_proj` random
    unit directions a_m, the projected samples h = Z a_m are compared with N(0, 1) through

        EP(h) = N * integral_{-inf}^{inf} w(t) |phi_N(t) - phi_0(t)|^2 dt,
        phi_N(t) = (1/N) sum_n exp(i t h_n)   (empirical characteristic function)
        phi_0(t) = exp(-t^2 / 2)              (characteristic function of N(0, 1))
        w(t)     = exp(-t^2 / 2)              (Gaussian weighting window)

    The integrand is even in t, so the integral is 2 * int_0^t_max, evaluated with the trapezoid rule on
    `knots` equally spaced nodes (t_max = 3, where w(3) = 0.011). The result is averaged over the
    projections and the T positions, i.e. SIGReg is computed independently per position and then averaged.

    Scaling: the factor N makes the statistic sample-size calibrated. For an exactly Gaussian batch its
    expectation is int w(t)(1 - e^{-t^2}) dt = sqrt(2 pi) - sqrt(2 pi / 3) ~ 1.06 for every N, while for
    any fixed non-Gaussian distribution it grows proportionally to N. There is no centering: a non-zero
    mean is penalized too, as the target is N(0, I).
    """

    def __init__(self, num_proj=1024, knots=17, t_max=3.0):
        super().__init__()
        self.num_proj, self.knots, self.t_max = int(num_proj), int(knots), float(t_max)
        t = torch.linspace(0, self.t_max, self.knots, dtype=torch.float32)
        dt = self.t_max / (self.knots - 1)
        trapezoid = torch.full((self.knots,), 2 * dt, dtype=torch.float32)  # x2: integral over [-t_max, t_max]
        trapezoid[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)                  # phi_0(t), the N(0, 1) characteristic function
        self.register_buffer("weights", trapezoid * window)  # quadrature weights times w(t)

    def settings(self):
        return {"num_proj": self.num_proj, "knots": self.knots, "t_max": self.t_max,
                "integration": "trapezoid on [0, t_max], doubled by symmetry", "weight": "exp(-t^2/2)",
                "scaling": "N * weighted integral, mean over projections and time positions"}

    def sample_directions(self, dim, generator=None, device=None):
        """[dim, num_proj] Gaussian directions normalized to unit length (uniform on the sphere)."""
        gen_device = generator.device if generator is not None else device
        directions = torch.randn(dim, self.num_proj, generator=generator, device=gen_device)
        return (directions / directions.norm(p=2, dim=0, keepdim=True)).to(device)

    def statistic(self, z, generator=None, directions=None):
        """Epps-Pulley statistic per position and projection: z [T, N, D] (or [N, D]) -> [T, num_proj]."""
        if z.dim() == 2:
            z = z.unsqueeze(0)
        if directions is None:
            directions = self.sample_directions(z.size(-1), generator, z.device)
        x_t = (z @ directions.to(z.dtype)).unsqueeze(-1) * self.t                         # [T, N, P, K]
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()      # |phi_N - phi_0|^2: [T, P, K]
        return (err @ self.weights) * z.size(-2)                                          # [T, P]

    def forward(self, z, generator=None, directions=None):
        """Scalar SIGReg: mean of the statistic over projections and time positions."""
        return self.statistic(z, generator, directions).mean()

    def null_expectation(self):
        """Expected statistic for an exactly N(0, I) batch under this quadrature (any N)."""
        return float((self.weights * (1 - torch.exp(-self.t.square()))).sum())

    @staticmethod
    def null_expectation_exact():
        """The same expectation with the exact integral over the real line: sqrt(2 pi) - sqrt(2 pi / 3)."""
        return math.sqrt(2 * math.pi) - math.sqrt(2 * math.pi / 3)


def stack_embeddings(outputs):
    """[5, B, D]: the four historical positions z_{t-3}..z_t and the target z_{t+1}, position first."""
    return torch.cat([outputs["z_history"].transpose(0, 1), outputs["z_target"].unsqueeze(0)], dim=0)


def world_model_loss(outputs, sigreg, sigreg_weight, generator=None):
    """LeWM objective total = prediction_loss + lambda * SIGReg(Z), with every component returned for logging."""
    pred = prediction_loss(outputs["z_pred"], outputs["z_target"])
    sig = sigreg(stack_embeddings(outputs), generator)
    weighted = sigreg_weight * sig
    return {"prediction_loss": pred, "sigreg_loss": sig, "weighted_sigreg_loss": weighted, "total_loss": pred + weighted}
