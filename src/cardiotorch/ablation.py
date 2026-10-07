"""Experimental: gradient-based ablation planning for re-entry.

Ablation destroys tissue so that it no longer conducts. Given an established spiral wave, a per-pixel
lesion field is optimised by differentiating through the simulator so that the spiral stops within a time
horizon, while a penalty keeps the ablated area small.

    loss = active_fraction(u at t0 + horizon) + area_weight * mean(lesion) + tv_weight * TV(lesion)

The result is binarised and checked with a plain forward simulation; the classic comparison is a straight
lesion from the spiral core to the nearest boundary, which is known to stop a single spiral.
"""
from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn

from .inverse import total_variation
from .simulate import Tissue, simulate


class LesionField(nn.Module):
    """Per-pixel lesion strength s in (0, 1); conductivity c = 1 - (1 - c_min) * s."""

    def __init__(self, shape, init: float = 0.02, c_min: float = 0.02):
        super().__init__()
        self.c_min = c_min
        self.logits = nn.Parameter(torch.full(tuple(shape), math.log(init / (1 - init))))

    def strength(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def forward(self) -> torch.Tensor:
        return 1.0 - (1.0 - self.c_min) * self.strength()


def lesion_conductivity(mask: torch.Tensor, c_min: float = 0.02) -> torch.Tensor:
    """Conductivity map of a binary lesion mask."""
    return 1.0 - (1.0 - c_min) * mask.float()


def active_fraction(u: torch.Tensor, threshold: float = 0.3, sharpness: float = 0.05) -> torch.Tensor:
    """Smooth fraction of tissue that is excited."""
    return torch.sigmoid((u - threshold) / sharpness).mean()


def spiral_state(tissue: Tissue, dt: float, settle: float = 150.0, s2_times=range(40, 200, 5)):
    """Create a sustained spiral with an S1-S2 protocol and return (u, v) after it has settled."""
    h, w = tissue.shape
    s1 = torch.zeros(h, w)
    s1[:, :3] = 1.0
    s2 = torch.zeros(h, w)
    s2[h // 2:, : w // 2] = 1.0
    with torch.no_grad():
        for t2 in s2_times:
            out = simulate(tissue, s1, dt=dt, t_end=t2 + settle, stimuli=[(float(t2), s2)])
            later = simulate(tissue, out["u"], v0=out["v"], dt=dt, t_end=100.0)
            if float(active_fraction(later["u"])) > 0.05:
                return out["u"], out["v"], t2
    raise RuntimeError("no S2 time produced a sustained spiral")


def terminates(tissue: Tissue, u0, v0, conductivity, dt: float, t_end: float, level: float = 0.01) -> bool:
    """Forward check: is the tissue (almost) quiescent after t_end with this conductivity map?"""
    with torch.no_grad():
        out = simulate(dataclasses.replace(tissue, conductivity=conductivity), u0, v0=v0, dt=dt, t_end=t_end)
    return float((out["u"] > 0.3).float().mean()) < level


def plan_ablation(tissue: Tissue, u0, v0, *, dt: float, horizon: float, steps: int = 60, lr: float = 0.3,
                  area_weight: float = 1.0, tv_weight: float = 0.0, checkpoint_steps: int = 200,
                  log_every: int = 10) -> tuple[LesionField, list[float]]:
    """Optimise a lesion field that stops the spiral (u0, v0) within `horizon`."""
    lesion = LesionField(tissue.shape)
    opt = torch.optim.Adam(lesion.parameters(), lr=lr)
    history = []
    for it in range(steps):
        opt.zero_grad()
        c = lesion()
        out = simulate(dataclasses.replace(tissue, conductivity=c), u0, v0=v0, dt=dt, t_end=horizon,
                       checkpoint_steps=checkpoint_steps)
        s = lesion.strength()
        loss = active_fraction(out["u"]) + area_weight * s.mean()
        if tv_weight > 0:
            loss = loss + tv_weight * total_variation(s)
        loss.backward()
        opt.step()
        history.append(float(loss))
        if log_every and (it % log_every == 0 or it == steps - 1):
            print(f"  iter {it:3d}  loss {history[-1]:.4f}  active {float(active_fraction(out['u'])):.3f}  "
                  f"area {float((s > 0.5).float().mean()) * 100:.2f} %", flush=True)
    return lesion, history
