# Experimental: gradient-based ablation planning (negative result so far)

**Question.** Can a per-pixel lesion field be learned by differentiating through the simulator so that a
re-entrant spiral stops, using less ablated tissue than the classic straight lesion from the spiral core
to the nearest boundary?

**Setup.** `examples/plan_ablation.py`: 96x96 isotropic sheet, S1-S2 spiral, core located as the point
with the smallest potential variance. Baseline is a 3-px-wide straight lesion from the core to the nearest
boundary.

**Result (2026-10-07).**

| | lesion area | stops the spiral? |
|---|---|---|
| straight lesion, core to boundary | 69 px | **yes** |
| learned, attempt 1 (end-time activity, 60 iterations) | ~58 px (s > 0.5) | no |
| learned, attempt 2 (activity averaged over the second half of the horizon, 100 iterations) | ~218 px | no |

The optimiser lowers the excited fraction slowly (0.36 to 0.22), but the gate is not met: no binarised
learned lesion stops the spiral, at any threshold.

**Why it is hard.** Termination is topological: a lesion has to connect the core to a boundary. A gradient
from a smooth activity loss does not see that connection until it is almost complete, and gradients
through several rotations of a spiral are poorly conditioned.

**Ideas not yet tried.** Initialise from tip-trajectory heat maps; use curriculum horizons; parameterise
the lesion as a line (start, end, width) instead of per-pixel; use phase-singularity loss terms.

This code lives on the `experimental-ablation` branch. It is not part of the released package.
