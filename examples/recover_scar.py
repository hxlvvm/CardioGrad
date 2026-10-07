"""Recover a hidden scar from noisy activation maps (writes assets/scar_recovery.png)."""
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from cardiograd import PixelField, ScarModel, Tissue, fit, point_stimulus, simulate

torch.manual_seed(0)
N, DT, T_END, NOISE = 64, 0.1, 110.0, 0.5
SITES = [(3, 3), (3, 60), (60, 3), (60, 60)]
TRUE = dict(centre=(36.0, 28.0), radius=10.0, contrast=0.9)

# ---- synthetic measurement on a 2x finer grid (same physical domain)
fine = (2 * N, 2 * N)
truth_fine = ScarModel(fine, centre=tuple(2 * x + 0.5 for x in TRUE["centre"]), radius=2 * TRUE["radius"],
                       contrast=TRUE["contrast"], edge=3.0)
with torch.no_grad():
    u0_fine = point_stimulus(fine, [(2 * r + 0.5, 2 * c + 0.5) for r, c in SITES], radius=1.5, dx=0.5)
    act_fine = simulate(Tissue(fine, dx=0.5, conductivity=truth_fine()), u0_fine, dt=0.04,
                        t_end=T_END)["act_time"]
    observed = F.avg_pool2d(act_fine.unsqueeze(1), 2).squeeze(1)
    observed = observed + NOISE * torch.randn_like(observed)
    c_true = ScarModel((N, N), **TRUE)()

tissue = Tissue((N, N))
u0 = point_stimulus((N, N), SITES, radius=1.5)


def dice(c, ref, thr=0.5):
    a, b = c < thr, ref < thr
    return float(2 * (a & b).sum() / (a.sum() + b.sum()))


t0 = time.time()
print("low-dimensional scar model")
scar = ScarModel((N, N), centre=(32.0, 32.0), radius=6.0, contrast=0.5)
h_scar = fit(scar, tissue, u0, observed, dt=DT, t_end=T_END, steps=150, lr=0.1, log_every=25)
c_scar = scar().detach()
print("  found", {k: [round(x, 2) for x in v] if isinstance(v, list) else round(v, 3)
                  for k, v in scar.describe().items()}, "| true", TRUE)

print("per-pixel field + total variation (experimental)")
field = PixelField((N, N), init=0.9)
h_pix = fit(field, tissue, u0, observed, dt=DT, t_end=T_END, steps=200, lr=0.05, tv_weight=20.0,
            log_every=25)
c_pix = field().detach()
print(f"Dice vs true scar: scar model {dice(c_scar, c_true):.3f} | per-pixel {dice(c_pix, c_true):.3f} "
      f"| total time {time.time() - t0:.0f} s")

fig, ax = plt.subplots(1, 5, figsize=(13, 2.9))
panels = [(c_true, "true conductivity"), (observed[0], "observed activation\n(noisy, one pacing site)"),
          (c_scar, f"recovered: scar model\nDice {dice(c_scar, c_true):.2f}"),
          (c_pix, f"recovered: per-pixel + TV\nDice {dice(c_pix, c_true):.2f}")]
for a, (img, title) in zip(ax[:4], panels):
    is_act = "activation" in title
    a.imshow(img, cmap="viridis" if is_act else "magma", vmin=None if is_act else 0, vmax=None if is_act else 1,
             origin="lower")
    a.set_title(title, fontsize=9)
    a.axis("off")
ax[4].semilogy(h_scar, label="scar model")
ax[4].semilogy(h_pix, label="per-pixel")
ax[4].set_title("misfit during the fit", fontsize=9)
ax[4].set_xlabel("iteration", fontsize=8)
ax[4].legend(fontsize=7)
ax[4].tick_params(labelsize=7)
fig.tight_layout()
Path("assets").mkdir(exist_ok=True)
fig.savefig("assets/scar_recovery.png", dpi=110)
print("wrote assets/scar_recovery.png")
