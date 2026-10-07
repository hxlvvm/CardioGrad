"""Pseudo-ECG from the tissue's current dipoles."""
from __future__ import annotations

import torch

from .simulate import Tissue


def pseudo_ecg(tissue: Tissue, frames: torch.Tensor, electrodes, height: float = 10.0) -> torch.Tensor:
    """frames: (B, T, H, W) or (T, H, W) potentials; electrodes: list of (row, col) in grid units."""
    h, w = tissue.shape
    if tuple(frames.shape[-2:]) != (h, w):
        raise ValueError(f"frames have spatial shape {tuple(frames.shape[-2:])}, tissue is {(h, w)}")
    if frames.dim() == 3:
        return pseudo_ecg(tissue, frames.unsqueeze(0), electrodes, height)[0]
    dxx, dyy, dxy = tissue.tensor(dtype=frames.dtype, device=frames.device)
    gx = torch.zeros_like(frames)
    gy = torch.zeros_like(frames)
    gx[..., :, 1:-1] = (frames[..., :, 2:] - frames[..., :, :-2]) / (2 * tissue.dx)
    gy[..., 1:-1, :] = (frames[..., 2:, :] - frames[..., :-2, :]) / (2 * tissue.dx)
    jx = dxx * gx + dxy * gy                 # D grad u
    jy = dxy * gx + dyy * gy
    yy, xx = torch.meshgrid(torch.arange(h, dtype=frames.dtype, device=frames.device),
                            torch.arange(w, dtype=frames.dtype, device=frames.device), indexing="ij")
    out = []
    for r, c in electrodes:
        ry, rx = (yy - r) * tissue.dx, (xx - c) * tissue.dx
        r3 = (rx ** 2 + ry ** 2 + height ** 2) ** 1.5
        # grad(1/r) = -(x - e) / r^3
        out.append((jx * rx / r3 + jy * ry / r3).sum(dim=(-2, -1)) * tissue.dx ** 2)
    return torch.stack(out, dim=1)
