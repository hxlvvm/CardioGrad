"""Recover tissue properties from activation maps by gradient descent through the simulator.

Two parameterisations of the conductivity map c(x) in (c_min, 1]:
  ScarModel   a smooth circular scar: centre, radius and contrast (4 parameters, well posed)
  PixelField  one value per pixel (ill posed: use several pacing sites and total-variation regularisation)
Both keep c inside its bounds with a sigmoid, so the explicit time step stays stable during the fit.
"""
from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn

from .simulate import Tissue, simulate


class ScarModel(nn.Module):
    def __init__(self, shape, centre, radius: float, contrast: float = 0.5, c_min: float = 0.05,
                 edge: float = 1.5):
        super().__init__()
        self.shape, self.c_min, self.edge = tuple(shape), c_min, edge
        self.centre = nn.Parameter(torch.tensor(centre, dtype=torch.float32))
        self.log_radius = nn.Parameter(torch.tensor(math.log(radius), dtype=torch.float32))
        self.contrast_logit = nn.Parameter(torch.tensor(math.log(contrast / (1 - contrast)), dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        h, w = self.shape
        yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32),
                                indexing="ij")
        dist = torch.sqrt((yy - self.centre[0]) ** 2 + (xx - self.centre[1]) ** 2 + 1e-6)
        inside = torch.sigmoid((torch.exp(self.log_radius) - dist) / self.edge)
        return 1.0 - (1.0 - self.c_min) * torch.sigmoid(self.contrast_logit) * inside

    def describe(self) -> dict:
        return {"centre": self.centre.detach().tolist(), "radius": float(torch.exp(self.log_radius)),
                "contrast": float(torch.sigmoid(self.contrast_logit))}


class PixelField(nn.Module):
    def __init__(self, shape, init: float = 0.9, c_min: float = 0.05):
        super().__init__()
        self.c_min = c_min
        p = (init - c_min) / (1 - c_min)
        self.logits = nn.Parameter(torch.full(tuple(shape), math.log(p / (1 - p))))

    def forward(self) -> torch.Tensor:
        return self.c_min + (1 - self.c_min) * torch.sigmoid(self.logits)


def total_variation(c: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Smoothed isotropic total variation (mean over the grid)."""
    dx = c[..., :, 1:] - c[..., :, :-1]
    dy = c[..., 1:, :] - c[..., :-1, :]
    return torch.sqrt(dx[..., :-1, :] ** 2 + dy[..., :, :-1] ** 2 + eps).mean()


def fit(param_model: nn.Module, tissue: Tissue, u0: torch.Tensor, observed: torch.Tensor, *, dt: float,
        t_end: float, steps: int = 100, lr: float = 0.05, tv_weight: float = 0.0, checkpoint_steps: int = 0,
        log_every: int = 0) -> list[float]:
    """Minimise the mean squared activation-time misfit over the parameters of `param_model`.

    u0: (B, H, W) initial conditions, one per pacing site; observed: (B, H, W) activation times.
    Returns the loss history.
    """
    opt = torch.optim.Adam(param_model.parameters(), lr=lr)
    history = []
    for it in range(steps):
        opt.zero_grad()
        c = param_model()
        sim = simulate(dataclasses.replace(tissue, conductivity=c), u0, dt=dt, t_end=t_end,
                       checkpoint_steps=checkpoint_steps)
        loss = ((sim["act_time"] - observed) ** 2).mean()
        if tv_weight > 0:
            loss = loss + tv_weight * total_variation(c)
        loss.backward()
        opt.step()
        history.append(float(loss))
        if log_every and (it % log_every == 0 or it == steps - 1):
            print(f"  iter {it:4d}  loss {history[-1]:.4f}", flush=True)
    return history
