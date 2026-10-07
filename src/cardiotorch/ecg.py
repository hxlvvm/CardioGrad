"""Pseudo-ECG: the extracellular potential of the tissue sheet seen by electrodes above it.

    phi(e, t) = - sum_cells  (D grad u) . grad(1 / r) * dx^2,   r = |x - e| with the electrode at height z0

This is the classic dipole-source approximation (Plonsey); amplitudes are in arbitrary units, which is
fine for waveform shape comparisons.
"""
from __future__ import annotations

import torch

from .simulate import Tissue


def pseudo_ecg(tissue: Tissue, frames: torch.Tensor, electrodes, height: float = 10.0) -> torch.Tensor:
    """frames: (B, T, H, W) or (T, H, W) potentials; electrodes: list of (row, col) in grid units.

    Returns (B, n_electrodes, T), or (n_electrodes, T) for unbatched frames. Gradients of u are taken
    in the interior only (zero on boundary cells). Differentiable if `frames` is.
    """
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
