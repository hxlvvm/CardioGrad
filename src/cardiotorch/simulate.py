"""Explicit time stepping of the 2D monodomain Aliev-Panfilov model, differentiable end to end."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.checkpoint import checkpoint

from .model import AlievPanfilov
from .tissue import diffusion_tensor, divergence

# Explicit Euler on the 2D diffusion operator is stable for dt * D_max / dx^2 <= 1/4 (isotropic).
# The cross terms of an anisotropic tensor tighten this, so a margin is kept.
STABILITY_LIMIT = 0.2


@dataclass(eq=False)
class Tissue:
    """A 2D sheet of tissue: grid spacing, diffusivities, fibre angle and a conductivity map."""

    shape: tuple[int, int]
    dx: float = 1.0
    d_long: float = 1.0
    d_trans: float = 1.0
    fibre_angle: torch.Tensor | float = 0.0
    conductivity: torch.Tensor | None = None
    model: AlievPanfilov = field(default_factory=AlievPanfilov)

    def tensor(self, dtype=torch.float32, device=None):
        """(Dxx, Dyy, Dxy) on the given dtype/device; maps must be scalars or match `shape`."""
        c = torch.ones(self.shape) if self.conductivity is None else torch.as_tensor(self.conductivity)
        theta = torch.as_tensor(self.fibre_angle)
        for name, t in (("conductivity", c), ("fibre_angle", theta)):
            if t.dim() > 0 and tuple(t.shape[-2:]) != tuple(self.shape):
                raise ValueError(f"{name} has shape {tuple(t.shape)}, expected {tuple(self.shape)}")
        c = c.to(dtype=dtype, device=device)
        theta = theta.to(dtype=dtype, device=device)
        return diffusion_tensor(c, theta, self.d_long, self.d_trans)

    def max_stable_dt(self) -> float:
        c_max = 1.0 if self.conductivity is None else float(torch.as_tensor(self.conductivity).max())
        d_max = max(self.d_long, self.d_trans) * c_max
        return STABILITY_LIMIT * self.dx ** 2 / d_max


def point_stimulus(shape, centres, radius: float, dx: float = 1.0, dtype=torch.float32, device=None):
    """Initial potential with u = 1 inside a disc around each centre (one batch entry per centre).

    centres are (row, col) in grid units, radius is in length units; returns (len(centres), H, W).
    """
    h, w = shape
    yy, xx = torch.meshgrid(torch.arange(h, dtype=dtype, device=device),
                            torch.arange(w, dtype=dtype, device=device), indexing="ij")
    out = [((yy - r) ** 2 + (xx - c) ** 2 <= (radius / dx) ** 2).to(dtype) for r, c in centres]
    return torch.stack(out)


def _step(u, v, dxx, dyy, dxy, model: AlievPanfilov, dx: float, dt: float):
    du, dv = model.reaction(u, v)
    u = u + dt * (divergence(u, dxx, dyy, dxy, dx) + du)
    v = v + dt * dv
    return u, v


def simulate(tissue: Tissue, u0: torch.Tensor, *, dt: float, t_end: float, v0: torch.Tensor | None = None,
             stimuli: list[tuple[float, torch.Tensor]] | None = None, threshold: float = 0.5,
             sharpness: float = 0.05, record_every: int = 0, checkpoint_steps: int = 0,
             reaction: bool = True) -> dict:
    """Integrate the model from u0 (shape (B, H, W) or (H, W)) to t_end.

    Returns a dict with
      u, v        final state
      act_time    differentiable activation time per cell: the time spent before the cell first
                  crosses `threshold`, using a running soft maximum of sigmoid((u - threshold) / sharpness).
                  Cells never activated get ~t_end.
      frames      u after every `record_every` steps (stacked on dim 1) if requested. Frames are detached
                  copies for visualisation and analysis (not differentiable); the initial state is not included.
    stimuli: optional list of (time, mask) applied as u = max(u, mask) before the step at that time
    (e.g. S1-S2 protocols); 0 <= time < t_end. Masks at the same step are combined by their maximum.
    checkpoint_steps > 0 recomputes chunks of that many steps in the backward pass to save memory.
    reaction=False switches the reaction terms off (pure diffusion; used to test conservation).
    """
    if dt > tissue.max_stable_dt() * (1 + 1e-9):
        raise ValueError(f"dt={dt} exceeds the stable limit {tissue.max_stable_dt():.4g} for this tissue")
    if tuple(u0.shape[-2:]) != tuple(tissue.shape):
        raise ValueError(f"u0 has spatial shape {tuple(u0.shape[-2:])}, tissue is {tuple(tissue.shape)}")
    squeeze = u0.dim() == 2
    u = u0.unsqueeze(0) if squeeze else u0
    v = torch.zeros_like(u) if v0 is None else (v0.unsqueeze(0) if squeeze else v0)
    dxx, dyy, dxy = tissue.tensor(dtype=u.dtype, device=u.device)
    model = tissue.model if reaction else _NoReaction()
    n_steps = int(round(t_end / dt))
    if abs(t_end / dt - n_steps) > 1e-6:
        raise ValueError(f"t_end={t_end} is not a whole number of steps of dt={dt}")
    events: dict[int, torch.Tensor] = {}
    for t, mask in stimuli or []:
        k = int(round(t / dt))
        if not 0 <= k < n_steps:
            raise ValueError(f"stimulus time {t} is outside [0, t_end)")
        events[k] = mask if k not in events else torch.maximum(events[k], mask)

    m = torch.sigmoid((u - threshold) / sharpness)
    act = torch.zeros_like(u)
    frames = []

    def chunk(u, v, m, act, start, stop):
        for i in range(start, stop):
            if i in events:
                u = torch.maximum(u, events[i].to(u))
            u, v = _step(u, v, dxx, dyy, dxy, model, tissue.dx, dt)
            m = torch.maximum(m, torch.sigmoid((u - threshold) / sharpness))
            act = act + dt * (1 - m)
        return u, v, m, act

    size = checkpoint_steps if checkpoint_steps > 0 else (record_every if record_every > 0 else n_steps)
    i = 0
    while i < n_steps:
        j = min(i + size, n_steps)
        if record_every > 0:                       # never step past a recording point
            j = min(j, (i // record_every + 1) * record_every)
        if checkpoint_steps > 0 and torch.is_grad_enabled():
            u, v, m, act = checkpoint(chunk, u, v, m, act, i, j, use_reentrant=False)
        else:
            u, v, m, act = chunk(u, v, m, act, i, j)
        i = j
        if record_every > 0 and i % record_every == 0:
            frames.append(u.detach().clone())

    out = {"u": u, "v": v, "act_time": act}
    if frames:
        out["frames"] = torch.stack(frames, dim=1)
    if squeeze:
        out = {k: t.squeeze(0) for k, t in out.items()}
    return out


class _NoReaction(AlievPanfilov):
    def reaction(self, u, v):
        return torch.zeros_like(u), torch.zeros_like(v)
