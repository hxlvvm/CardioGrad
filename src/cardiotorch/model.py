"""Aliev-Panfilov two-variable model of cardiac excitation (dimensionless units).

    du/dt = div(D grad u) - k u (u - a)(u - 1) - u v
    dv/dt = eps(u, v) * (-v - k u (u - a - 1)),   eps = eps0 + mu1 v / (u + mu2)

u is a normalised transmembrane potential (0 = rest, 1 = excited) and v a recovery variable.
Reference: Aliev & Panfilov, Chaos, Solitons & Fractals 7(3):293-301, 1996.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AlievPanfilov:
    """Reaction terms of the Aliev-Panfilov model. Defaults are the published parameter set."""

    k: float = 8.0
    a: float = 0.15
    eps0: float = 0.002
    mu1: float = 0.2
    mu2: float = 0.3

    def reaction(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (du/dt, dv/dt) without the diffusion term."""
        du = -self.k * u * (u - self.a) * (u - 1.0) - u * v
        eps = self.eps0 + self.mu1 * v / (u + self.mu2)
        dv = eps * (-v - self.k * u * (u - self.a - 1.0))
        return du, dv
