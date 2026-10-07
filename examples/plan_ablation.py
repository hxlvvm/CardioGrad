"""Experimental: learn an ablation lesion that stops a spiral wave, by gradient descent through the simulator.

1. Create a sustained spiral (S1-S2) and locate its core (the point whose potential varies least).
2. Baseline: a straight lesion from the core to the nearest boundary (the classic way to stop one spiral).
3. Learned: optimise a per-pixel lesion field (activity at the horizon + area + TV penalties), binarise it,
   and check with a plain forward run whether the spiral stops.
Writes assets/ablation.png and prints the lesion areas.

    python examples/plan_ablation.py
"""
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from cardiotorch import Tissue, simulate
from cardiotorch.ablation import lesion_conductivity, plan_ablation, spiral_state, terminates

torch.manual_seed(0)
N, DT, HORIZON, CHECK = 96, 0.05, 150.0, 400.0
tissue = Tissue((N, N))
t0 = time.time()
u0, v0, t2 = spiral_state(tissue, DT)
print(f"spiral from S2 at t = {t2} ({time.time() - t0:.0f} s)")

with torch.no_grad():
    frames = simulate(tissue, u0, v0=v0, dt=DT, t_end=60.0, record_every=20)["frames"]
std = frames.std(dim=0)
m = 8
core = divmod(int(torch.argmin(std[m:-m, m:-m])), N - 2 * m)
core = (core[0] + m, core[1] + m)
print("spiral core (row, col):", core)


def line_mask(core, frac=1.0, width=3):
    r, c = core
    dists = {"left": c, "right": N - 1 - c, "bottom": r, "top": N - 1 - r}
    side = min(dists, key=dists.get)
    length = int(round(dists[side] * frac)) + 1
    mask = torch.zeros(N, N, dtype=torch.bool)
    lo, hi = max(0, r - width // 2), r + width // 2 + 1
    clo, chi = max(0, c - width // 2), c + width // 2 + 1
    if side == "left":
        mask[lo:hi, c - length + 1: c + 1] = True
    elif side == "right":
        mask[lo:hi, c: c + length] = True
    elif side == "bottom":
        mask[r - length + 1: r + 1, clo:chi] = True
    else:
        mask[r: r + length, clo:chi] = True
    return mask


line = line_mask(core)
line_ok = terminates(tissue, u0, v0, lesion_conductivity(line), DT, CHECK)
print(f"baseline straight lesion core->boundary: {int(line.sum())} px, stops spiral: {line_ok}")

print("learning a lesion")
lesion, hist = plan_ablation(tissue, u0, v0, dt=DT, horizon=HORIZON, steps=100, lr=0.5, area_weight=0.3,
                             tv_weight=0.2, samples=10)
s = lesion.strength().detach()
best = None
for thr in (0.5, 0.4, 0.3, 0.2, 0.1):
    mask = s > thr
    if mask.sum() and terminates(tissue, u0, v0, lesion_conductivity(mask), DT, CHECK):
        best = (thr, mask)
        break
if best:
    print(f"learned lesion (s > {best[0]}): {int(best[1].sum())} px, stops spiral: True "
          f"| baseline {int(line.sum())} px (ratio {int(best[1].sum()) / int(line.sum()):.2f})")
else:
    print("learned lesion did not stop the spiral after binarisation")

with torch.no_grad():
    act = {}
    for name, c in [("no ablation", None), ("straight lesion", lesion_conductivity(line)),
                    ("learned lesion", lesion_conductivity(best[1]) if best else None)]:
        if name == "learned lesion" and best is None:
            continue
        tt = Tissue((N, N), conductivity=c)
        f = simulate(tt, u0, v0=v0, dt=DT, t_end=CHECK, record_every=40)["frames"]
        act[name] = (f > 0.3).float().mean(dim=(1, 2))

fig, ax = plt.subplots(1, 4, figsize=(13, 3.1))
ax[0].imshow(u0, cmap="inferno", origin="lower", vmin=0, vmax=1)
ax[0].plot(core[1], core[0], "c+", ms=10, mew=2)
ax[0].set_title("spiral wave (+ = core)", fontsize=9)
ax[1].imshow(s, cmap="Blues", origin="lower", vmin=0, vmax=1)
ax[1].set_title("learned lesion strength", fontsize=9)
over = torch.zeros(N, N, 3)
over[..., 0] = line.float()
if best:
    over[..., 2] = best[1].float()
ax[2].imshow(over, origin="lower")
ax[2].set_title("lesions: straight (red) vs learned (blue)", fontsize=9)
for a in ax[:3]:
    a.axis("off")
tt = torch.arange(1, len(next(iter(act.values()))) + 1) * 40 * DT
for name, y in act.items():
    ax[3].plot(tt, y * 100, label=name)
ax[3].set_xlabel("time after ablation", fontsize=8)
ax[3].set_ylabel("excited tissue (%)", fontsize=8)
ax[3].legend(fontsize=7)
ax[3].tick_params(labelsize=7)
fig.tight_layout()
Path("assets").mkdir(exist_ok=True)
fig.savefig("assets/ablation.png", dpi=110)
print(f"wrote assets/ablation.png ({time.time() - t0:.0f} s)")
