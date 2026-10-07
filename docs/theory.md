# The ideas behind CardioGrad, in plain language

## 1. The heart as an excitable medium

Every heartbeat is an electrical wave. Each heart muscle cell can be in one of three broad states:

- **resting**, ready to fire;
- **excited**, firing;
- **refractory**, recovering and unable to fire again for a while.

An excited cell pushes current into its neighbours. That raises their voltage until they fire too, so a
wave of excitation travels through the tissue. Behind the wave, cells are refractory, which stops the wave
from immediately travelling backwards. Media that behave like this are called *excitable media*. Forest
fires and some chemical reactions behave the same way.

## 2. The Aliev–Panfilov model

Detailed cell models track dozens of ion channels. The Aliev–Panfilov model (1996) keeps only two
numbers per point:

- `u`, the electrical potential, scaled so that 0 is rest and 1 is fully excited;
- `v`, a slow recovery variable. It builds up after a cell fires and holds it refractory.

Two equations describe how they change:

```
du/dt = div(D grad u) - k u (u - a)(u - 1) - u v      (spread + firing - recovery)
dv/dt = eps(u, v) * (-v - k u (u - a - 1))            (slow recovery)
```

The cubic term `u (u - a)(u - 1)` makes the cell "all or nothing". A small push decays back to rest, but
a push past the threshold `a` fires the cell all the way to 1. The model reproduces the shape of the action
potential and its restitution (shorter beats at faster rates), at a tiny fraction of the cost of a detailed
model.

## 3. Diffusion and fibres

The `div(D grad u)` term is how the wave spreads. Current flows from high to low potential, like heat
spreading through metal.

Heart muscle is made of elongated fibres, and current flows faster along a fibre than across it. That is
why `D` is a 2×2 *tensor*: one diffusivity along the fibre direction and a smaller one across it. The wave
speed scales with the square root of `D`, so `D_long = 4 × D_trans` gives a wave that is twice as fast
along the fibres. One of the tests checks exactly this.

Scar tissue conducts poorly. A local conductivity factor `c` between 0.05 and 1 scales `D` down where
tissue is damaged.

**How it is computed.** The tissue is a grid of cells. Currents are computed on the *faces* between
neighbouring cells, and the boundary faces carry no current. Because every face current leaves one cell and
enters its neighbour, nothing is created or lost. With the firing terms switched off, the total `u` stays
exactly constant, and a test checks this. Time advances in small explicit steps. They have to be small
enough (`dt ≤ 0.2 dx² / D_max`) or the simulation blows up, so the code refuses unstable settings.

## 4. Spiral waves and arrhythmia

Sometimes a wave hits tissue that is still refractory and breaks. Its free end then curls around and
becomes a **spiral wave** that keeps re-exciting the tissue, which is called *re-entry*. In the heart this
is the mechanism behind many dangerous arrhythmias, such as ventricular tachycardia.

The `spiral_wave.py` example creates one with the classic S1–S2 protocol:

1. A first stimulus (S1) launches a plane wave.
2. A second stimulus (S2) is placed so that it meets S1's refractory tail.

## 5. The pseudo-ECG

An electrode does not touch the cells. It sees the summed electric field of all the currents in the
tissue, weighted by distance. The pseudo-ECG adds up each cell's current dipole `D grad u`, scaled by how
it points relative to the electrode and by `1/distance²`. During the spiral, the trace becomes a regular
oscillation, much like the ECG of a monomorphic tachycardia.

## 6. The inverse problem: finding hidden scar

The forward question is: *given the tissue, what activation pattern do we see?* Clinically, the opposite
is needed: *given what we measured, what does the tissue look like?* Where is the scar? This is an
**inverse problem**.

CardioGrad is written entirely in PyTorch, so every operation is differentiable. PyTorch can therefore
compute how the activation-time misfit changes when the conductivity at any pixel changes, back through
thousands of time steps. That is *automatic differentiation*, the same machinery used to train neural
networks. With that gradient, an optimiser (Adam) adjusts the conductivity map step by step until the
simulated activation matches the measurement.

A few details make this work:

- **Activation time is not normally differentiable**, because it is the moment a cell crosses a
  threshold. CardioGrad uses a smooth version: a soft "has it fired yet" indicator built from a sigmoid,
  integrated over time.
- **Bounded parameters.** Conductivity passes through a sigmoid, so it can never leave its allowed range.
  That keeps the time step stable during the fit.
- **Memory.** Back-propagating through thousands of steps stores a lot. Gradient checkpointing recomputes
  chunks during the backward pass instead of storing them.
- **Avoiding the "inverse crime".** If the test data come from the same simulator and grid used for
  fitting, the problem looks unrealistically easy. The example generates the "measurement" on a grid twice
  as fine, averages it down and adds noise.
- **Well-posed vs ill-posed.**
  - A scar described by four numbers (centre, radius, contrast) is well determined by a few activation
    maps.
  - One value per pixel is not, because many maps explain the data equally well. Several pacing sites and
    a total-variation penalty, which prefers piecewise-constant maps, are needed. This is why the per-pixel
    fit is labelled experimental.

## 7. What this is not

- It is a **2D sheet** with a phenomenological model. It is not a patient-specific 3D heart, and there is
  no torso.
- Units are dimensionless. Mapping them to milliseconds and millimetres requires calibration.
- It is a teaching and prototyping tool, not a clinical one.

## Glossary

- **Action potential.** The voltage spike of one heart cell when it fires.
- **Conduction velocity.** How fast the excitation wave travels.
- **Refractory period.** The time after firing during which a cell cannot fire again.
- **Re-entry.** A wave that keeps circulating and re-exciting the same tissue.
- **Automatic differentiation.** Computing exact derivatives of a program by tracking every operation.
- **Total variation.** The sum of absolute differences between neighbouring pixels; penalising it favours
  piecewise-constant maps.
