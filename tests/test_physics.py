"""Physics, autodiff and API tests."""
from __future__ import annotations

import math

import pytest
import torch

from cardiograd import PixelField, ScarModel, Tissue, fit, point_stimulus, pseudo_ecg, simulate


def strip_cv(d_long: float, d_trans: float | None = None, angle: float = 0.0, dx: float = 1.0) -> float:
    """Plane-wave conduction velocity along x (length units per time unit) on a narrow strip of length 200."""
    n = int(round(200 / dx))
    tissue = Tissue((8, n), dx=dx, d_long=d_long, d_trans=d_long if d_trans is None else d_trans,
                    fibre_angle=angle)
    u0 = torch.zeros(8, n)
    u0[:, : max(4, int(4 / dx))] = 1.0
    act = simulate(tissue, u0, dt=0.05, t_end=300)["act_time"][4]
    i50, i150 = int(round(50 / dx)), int(round(150 / dx))
    return 100.0 / float(act[i150] - act[i50])


def test_diffusion_conserves_mass():
    torch.manual_seed(0)
    tissue = Tissue((20, 30), d_long=1.0, d_trans=0.3, fibre_angle=0.7)
    u0 = torch.rand(20, 30)
    out = simulate(tissue, u0, dt=0.1, t_end=20, reaction=False)
    assert torch.isclose(out["u"].sum(), u0.sum(), rtol=1e-5)
    assert out["u"].std() < u0.std()          # and it actually diffuses


def test_conduction_velocity_scales_with_sqrt_diffusivity():
    ratio = strip_cv(4.0) / strip_cv(1.0)
    assert 1.8 < ratio < 2.3                 # theory: sqrt(4) = 2 (coarse grid adds a little)


def test_anisotropy_follows_fibre_direction():
    # dx = 0.5 resolves the slow across-fibre front (at dx = 1 the grid exaggerates the ratio)
    along = strip_cv(1.0, 0.25, angle=0.0, dx=0.5)            # wave travels along the fibres
    across = strip_cv(1.0, 0.25, angle=math.pi / 2, dx=0.5)   # same strip, fibres rotated 90 degrees
    assert 1.8 < along / across < 2.25                        # theory: sqrt(1 / 0.25) = 2
    # rotating the tensor by 90 degrees must reproduce the plain isotropic case D = D_t
    assert abs(across / strip_cv(0.25, dx=0.5) - 1) < 0.02


def test_unstable_time_step_is_rejected():
    with pytest.raises(ValueError):
        simulate(Tissue((10, 10), d_long=4.0, d_trans=4.0), torch.zeros(10, 10), dt=0.1, t_end=1)


def test_gradients_match_finite_differences():
    torch.manual_seed(0)
    tissue = Tissue((6, 6), d_long=1.0, d_trans=0.5, fibre_angle=0.3)
    u0 = torch.zeros(1, 6, 6, dtype=torch.float64)
    u0[:, :2, :2] = 1.0
    c = (0.5 + 0.5 * torch.rand(6, 6, dtype=torch.float64)).requires_grad_()

    def f(c):
        t = Tissue(tissue.shape, d_long=1.0, d_trans=0.5, fibre_angle=0.3, conductivity=c)
        return simulate(t, u0, dt=0.05, t_end=1.0)["u"].sum()

    assert torch.autograd.gradcheck(f, (c,), eps=1e-6, atol=1e-5)


def test_checkpointing_gives_the_same_gradient():
    tissue = Tissue((12, 12))
    u0 = point_stimulus((12, 12), [(2, 2)], radius=2)
    grads = []
    for ck in (0, 10):
        c = torch.full((12, 12), 0.8, requires_grad=True)
        t = Tissue(tissue.shape, conductivity=c)
        simulate(t, u0, dt=0.05, t_end=4.0, checkpoint_steps=ck)["act_time"].sum().backward()
        grads.append(c.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


def test_pseudo_ecg_shape():
    tissue = Tissue((20, 20))
    out = simulate(tissue, point_stimulus((20, 20), [(5, 5)], radius=2), dt=0.1, t_end=10, record_every=10)
    ecg = pseudo_ecg(tissue, out["frames"], electrodes=[(0, 0), (19, 19)])
    assert ecg.shape == (1, 2, 10) and torch.isfinite(ecg).all()


def test_scar_parameters_are_recovered():
    torch.manual_seed(0)
    shape, dt, t_end = (32, 32), 0.1, 40.0
    tissue = Tissue(shape)
    u0 = point_stimulus(shape, [(2, 2), (29, 29)], radius=2)
    truth = ScarModel(shape, centre=(16.0, 16.0), radius=6.0, contrast=0.9)
    with torch.no_grad():
        observed = simulate(Tissue(shape, conductivity=truth()), u0, dt=dt, t_end=t_end)["act_time"]
    guess = ScarModel(shape, centre=(12.0, 19.0), radius=4.0, contrast=0.5)
    hist = fit(guess, tissue, u0, observed, dt=dt, t_end=t_end, steps=60, lr=0.1)
    found = guess.describe()
    assert hist[-1] < 0.1 * hist[0]
    assert math.dist(found["centre"], (16.0, 16.0)) < 1.5


def test_pixel_field_is_bounded():
    c = PixelField((5, 5), init=0.9)()
    assert torch.all((c > 0.05) & (c <= 1.0))


def test_frames_are_identical_with_and_without_checkpointing():
    tissue = Tissue((16, 16))
    u0 = point_stimulus((16, 16), [(3, 3)], radius=2)
    a = simulate(tissue, u0, dt=0.1, t_end=3.0, record_every=5)["frames"]
    b = simulate(tissue, u0, dt=0.1, t_end=3.0, record_every=5, checkpoint_steps=7)["frames"]
    assert a.shape == b.shape == (1, 6, 16, 16) and torch.allclose(a, b)


def test_batch_matches_separate_runs():
    tissue = Tissue((16, 16), d_long=1.0, d_trans=0.5, fibre_angle=0.4)
    u0 = point_stimulus((16, 16), [(3, 3), (12, 10)], radius=2)
    both = simulate(tissue, u0, dt=0.1, t_end=5.0)["u"]
    for k in range(2):
        assert torch.allclose(both[k], simulate(tissue, u0[k], dt=0.1, t_end=5.0)["u"], atol=1e-6)


def test_stimuli_combine_and_are_validated():
    tissue = Tissue((8, 8))
    ones, zeros = torch.ones(8, 8), torch.zeros(8, 8)
    out = simulate(tissue, zeros, dt=0.1, t_end=1.0, stimuli=[(0.5, ones), (0.5, zeros)])
    assert out["u"].max() > 0.5                      # the second mask does not erase the first
    with pytest.raises(ValueError):
        simulate(tissue, zeros, dt=0.1, t_end=1.0, stimuli=[(5.0, ones)])


def test_shapes_are_validated():
    with pytest.raises(ValueError):
        simulate(Tissue((10, 10)), torch.zeros(12, 10), dt=0.1, t_end=1.0)
    with pytest.raises(ValueError):
        simulate(Tissue((10, 10), conductivity=torch.ones(10, 12)), torch.zeros(10, 10), dt=0.1, t_end=1.0)


def test_pseudo_ecg_unbatched_shape():
    tissue = Tissue((20, 20))
    frames = simulate(tissue, point_stimulus((20, 20), [(5, 5)], radius=2)[0], dt=0.1, t_end=10,
                      record_every=10)["frames"]
    assert pseudo_ecg(tissue, frames, electrodes=[(0, 0), (19, 19), (10, 0)]).shape == (3, 10)
