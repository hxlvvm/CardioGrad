"""Spiral (re-entrant) wave from an S1-S2 protocol, with the pseudo-ECG it produces.

S1: a plane wave from the left edge. S2: a second stimulus over the lower-left quadrant, timed so that it
meets the refractory tail of S1; the wave can only spread one way and curls into a rotating spiral, the
2D analogue of re-entrant arrhythmia. Writes assets/spiral.gif.

    python examples/spiral_wave.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from cardiotorch import Tissue, pseudo_ecg, simulate

N, DT, T_END, EVERY = 160, 0.05, 520.0, 60          # one frame every 3 time units

tissue = Tissue((N, N), d_long=1.0, d_trans=0.5, fibre_angle=0.5)
s1 = torch.zeros(N, N)
s1[:, :4] = 1.0
s2 = torch.zeros(N, N)
s2[N // 2:, : N // 2] = 1.0

# The S2 time must fall in the window where the S1 waveback is still refractory at the quadrant's edge:
# too early and S2 cannot fire, too late and it only launches another plane wave. Scan for it.
with torch.no_grad():
    for s2_time in range(60, 200, 5):
        out = simulate(tissue, s1, dt=DT, t_end=T_END, stimuli=[(float(s2_time), s2)], record_every=EVERY)
        frames = out["frames"]                                # (T, H, W)
        active = (frames[-len(frames) // 4:] > 0.5).float().mean(dim=(1, 2))
        if bool((active > 0.02).all()):                       # still re-entering at the end: a spiral
            break
    else:
        raise RuntimeError("no S2 time produced a sustained spiral")
print(f"S2 at t = {s2_time}: sustained re-entry")
ecg = pseudo_ecg(tissue, frames.unsqueeze(0), electrodes=[(N // 2, N + 20)], height=30.0)[0, 0]
t = torch.arange(1, len(frames) + 1) * EVERY * DT

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1, 1.3]})
im = ax0.imshow(frames[0], cmap="inferno", vmin=0, vmax=1, origin="lower")
ax0.set_title("transmembrane potential", fontsize=9)
ax0.axis("off")
(line,) = ax1.plot([], [], color="#1f3a5f", lw=1.4)
ax1.set_xlim(0, float(t[-1]))
ax1.set_ylim(float(ecg.min()) * 1.1, float(ecg.max()) * 1.1)
ax1.set_title("pseudo-ECG (electrode right of the sheet)", fontsize=9)
ax1.set_xlabel("time (model units)", fontsize=8)
ax1.tick_params(labelsize=7)
ax1.set_yticks([])
fig.tight_layout()


def draw(i):
    im.set_data(frames[i])
    line.set_data(t[: i + 1], ecg[: i + 1])
    return im, line


Path("assets").mkdir(exist_ok=True)
FuncAnimation(fig, draw, frames=len(frames), blit=True).save("assets/spiral.gif", writer=PillowWriter(fps=12), dpi=80)
print(f"wrote assets/spiral.gif ({len(frames)} frames)")
