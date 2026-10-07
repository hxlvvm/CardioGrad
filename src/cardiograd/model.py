"""Aliev-Panfilov reaction terms (Aliev & Panfilov 1996)."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AlievPanfilov:
    """Reaction terms of the Aliev-Panfilov model."""

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
