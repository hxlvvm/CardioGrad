"""Anisotropic diffusion on a 2D grid in conservative (flux) form with no-flux boundaries.

The diffusion tensor at each cell is  D = c * (D_t I + (D_l - D_t) f f^T),  f = (cos theta, sin theta),
where c in (0, 1] is a local conductivity scale (c < 1 models fibrosis or scar), D_l / D_t are the
along- and across-fibre diffusivities and theta is the fibre angle.

Fluxes are evaluated on cell faces and the boundary fluxes are set to zero, so with the reaction term
switched off the total sum of u is conserved exactly (the face fluxes telescope).
Convention: tensors are (..., H, W); x runs along W (columns), y along H (rows).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def diffusion_tensor(conductivity, fibre_angle, d_long: float, d_trans: float):
    """Return the (Dxx, Dyy, Dxy) fields of the anisotropic diffusion tensor."""
    conductivity = torch.as_tensor(conductivity)
    theta = torch.as_tensor(fibre_angle, dtype=conductivity.dtype, device=conductivity.device)
    c, s = torch.cos(theta), torch.sin(theta)
    dl = d_long - d_trans
    dxx = conductivity * (d_trans + dl * c * c)
    dyy = conductivity * (d_trans + dl * s * s)
    dxy = conductivity * (dl * c * s)
    return dxx, dyy, dxy


def _pad_rows(u):
    """Replicate the first and last row (zero normal gradient in y)."""
    return torch.cat([u[..., :1, :], u, u[..., -1:, :]], dim=-2)


def _pad_cols(u):
    """Replicate the first and last column (zero normal gradient in x)."""
    return torch.cat([u[..., :, :1], u, u[..., :, -1:]], dim=-1)


def divergence(u, dxx, dyy, dxy, dx: float):
    """div(D grad u) on a uniform grid with spacing dx and zero flux through the boundary."""
    dxx, dyy, dxy = (torch.broadcast_to(t, u.shape) for t in (dxx, dyy, dxy))
    # central gradients at cell centres (one-sided zero at the edges via replicate padding)
    gx = (_pad_cols(u)[..., :, 2:] - _pad_cols(u)[..., :, :-2]) / (2 * dx)
    gy = (_pad_rows(u)[..., 2:, :] - _pad_rows(u)[..., :-2, :]) / (2 * dx)

    # x-faces: between columns j and j+1  -> (..., H, W-1)
    avg_x = lambda t: 0.5 * (t[..., :, 1:] + t[..., :, :-1])
    jx = avg_x(dxx) * (u[..., :, 1:] - u[..., :, :-1]) / dx + avg_x(dxy) * avg_x(gy)
    # y-faces: between rows i and i+1  -> (..., H-1, W)
    avg_y = lambda t: 0.5 * (t[..., 1:, :] + t[..., :-1, :])
    jy = avg_y(dyy) * (u[..., 1:, :] - u[..., :-1, :]) / dx + avg_y(dxy) * avg_y(gx)

    jx = F.pad(jx, (1, 1))            # zero flux through left / right boundary
    jy = F.pad(jy, (0, 0, 1, 1))      # zero flux through top / bottom boundary
    return (jx[..., :, 1:] - jx[..., :, :-1]) / dx + (jy[..., 1:, :] - jy[..., :-1, :]) / dx
