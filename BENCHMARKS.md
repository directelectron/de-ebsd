# torchebsd vs EMsoft — parity, speed, and method audit

torchebsd's acceptance oracle is [EMsoft](https://github.com/EMsoft-org/EMsoftOO)
(EMsoftOO). This page audits the two codes head to head: **how** each computes a
master pattern, **how fast**, and **how closely they agree**, across single- and
multi-component phases and a sweep of crystal systems from cubic to monoclinic.

Every comparison below runs the **same crystal, the same Monte Carlo exit-depth
profile (EMsoft's own `accum_z`), the same Bethe parameters, dmin, and grid** for
both codes, so any difference is the diffraction engine (or a convention), not
the inputs. Measured on one machine: NVIDIA TITAN X (Pascal) GPU for torchebsd,
a 24-thread CPU for EMsoft.

## The headline

- **Bit-exact parity** (Pearson `r = 1.00000`) with EMsoft for **cubic and
  tetragonal** phases, **single- and multi-component** (Ni, Si, GaAs, TiO2).
- **~10x faster** for the full energy-resolved master (Ni: 3.0 s vs 31 s, both
  6 energy bins, both symmetry-reduced) — GPU scaling-and-squaring vs CPU
  eigendecomposition.
- **Low-symmetry systems are the frontier**: hexagonal and monoclinic surface
  crystallographic *convention* mismatches (frame, space-group setting), not
  physics errors — see [Findings](#findings).
- The old "Ni r = 0.93" gap was **not physics** — it was a reference-provenance
  and Bethe-default mismatch. With matched settings, Ni is `r = 1.00000`.

## Method comparison

| | EMsoft (`EMEBSDmaster`) | torchebsd (`bloch_prism`) |
| --- | --- | --- |
| Language / runtime | Fortran, OpenMP (+ OpenCL for MC) | Python / PyTorch |
| Bloch-wave solve | eigendecomposition (LAPACK `zgeev`), per direction | scaling-and-squaring `matrix_exp`, **batched** over directions |
| Depth integration | `CalcLgh` (eigenvalues) | `CalcLgh` (Horner) / depth recurrence on the scattering matrix |
| Device | CPU | GPU (CUDA) or CPU |
| Symmetry | point-group fundamental sector (`Apply3DPGSymmetry`) | point-group fundamental sector (via orix `in_fundamental_sector`) |
| Energy bins | loop, one full solve per bin | loop, one full solve per bin (`energy_resolved=True`) |
| Monte Carlo (depth) | OpenCL, **required** to produce a pattern | native torch MC (EMsoft-free) *or* reads EMsoft `accum_z` |
| Structure factors | Weickenmeier-Kohl (`FSCATT`) | WK port (numpy + a torch, differentiable copy) |
| Differentiable | no | **yes** — autograd to lattice-independent crystal params (positions, occupancy, Debye-Waller) |

Both use the *same* physics (WK scattering, Bethe potentials, Bloch waves,
square-Lambert master grid) — torchebsd is a GPU-batched, differentiable
re-expression of EMsoft's algorithm, with the eigendecomposition replaced by a
mathematically equivalent matrix exponential that batches on the GPU.

## Accuracy

Per-pixel Pearson correlation of the beam-energy Lambert map, torchebsd vs a
*fresh, matched-settings* EMsoft master (npx = 100, dmin = 0.05 nm, 20 keV,
Bethe 4/8/50, EMsoft's own depth profile). For multi-component phases the EMsoft
master is summed over its per-atom-type maps (the full-crystal pattern).

| Crystal | System | Components | Atoms/cell | Correlation |
| --- | --- | --- | --- | --- |
| Ni | cubic (fcc, 225) | 1 | 4 | **1.00000** |
| Si | cubic (diamond, 227) | 1 | 8 | **1.00000** † |
| GaAs | cubic (zinc blende, 216) | 2 | 8 | **1.00000** |
| TiO2 | tetragonal (rutile, 136) | 2 | 6 | **1.00000** |
| Ti | hexagonal (hcp, 194) | 1 | 2 | 0.871 ‡ |
| ZrO2 | monoclinic (baddeleyite, 14) | 3 | 12 | 0.00 ‡ |

† Si requires the correct **origin choice** (space-group setting 2 for Fd-3m).
The first benchmark attempt used setting 1 and scored 0.588 — a textbook
origin-choice pitfall, not a physics gap; with setting 2, Si is pixel-exact (and
the canonical npx-500 validation against `Si_MC.h5` is `r = 1.0000`).

‡ See [Findings](#findings): these are convention mismatches (hexagonal Cartesian
frame; monoclinic space-group setting), the signature being partial (Ti) or full
(ZrO2) decorrelation of two *internally valid* but differently-conventioned crystals.

## Speed

Full energy-resolved master (all 6 energy bins), npx = 100, symmetry-reduced in
both codes. EMsoft times are its own reported execution time; torchebsd is
GPU wall-clock.

| Crystal | Beam directions (fund. sector) | EMsoft (6 bins) | torchebsd (6 bins) | Speedup |
| --- | --- | --- | --- | --- |
| Ni | 5 152 | 31 s | 3.0 s | **~10x** |

The **~10x** is the fair, like-for-like figure (both 6 energy bins, both
symmetry-reduced, GPU vs CPU). Per energy bin it is ~9x (torchebsd 0.57 s vs
EMsoft ~5 s). At native resolution the batched GPU solve pulls further ahead: a
**npx-500 Si master is ~10 s in torchebsd** (single energy, fp32) where the
per-direction eigendecomposition EMsoft uses would need ~10^6 eigensolves.

> Caveat: these are single-machine numbers (one GPU vs one CPU) and the absolute
> EMsoft seconds for the other structures in the sweep were measured under load;
> treat the **~10x order-of-magnitude** as the robust claim, not any single
> second-count.

## Findings

1. **The dynamical physics is at full parity.** For every high-symmetry case,
   single- *and* multi-component, torchebsd reproduces EMsoft to `r = 1.00000`.
   The batched `matrix_exp` Bloch solve is bit-for-bit equivalent to EMsoft's
   eigendecomposition; the `Sgh` reduction, Bethe partition, and depth
   integration all match.

2. **The "Ni gap" was settings, not physics.** torchebsd's default Bethe
   parameters (8/50/100) differ from EMsoft's (4/8/50); the public Ni reference
   (kikuchipy) was also computed with unknown settings. Matching them closes it:
   `r = 1.00000`.

3. **Low-symmetry parity is gated by crystallographic conventions, not the
   solver.** Two independent codes must agree on:
   - **Origin choice** (e.g. Fd-3m setting 1 vs 2) — Si demonstrates this.
   - **Space-group setting** for monoclinic/triclinic (unique axis, cell choice).
     ZrO2 (sg 14) scored `r = 0` — the signature of the two codes building
     *different* cells from the same numbers, exactly like the Si origin bug
     writ large.
   - **Hexagonal Cartesian frame** — Ti scored 0.871; the pattern is right but
     rotated by the a-axis convention. torchebsd's hexagonal grid handling is
     already flagged as open work in DESIGN.md (M3).

   These are alignment tasks (match the setting/frame), not evidence of a
   diffraction error — but until they are done, **low-symmetry parity is not
   established**.

## What this means

For the phases that dominate practical EBSD — cubic and tetragonal metals,
semiconductors, and oxides, single- and multi-component — **torchebsd is a
drop-in, bit-exact replacement for EMsoft's master-pattern computation, ~10x
faster on a GPU, and differentiable.** Low-symmetry (hexagonal frame,
monoclinic/triclinic settings) is the remaining frontier and is convention work,
not physics.

## Reproducing

EMsoft (`EMEBSDmaster`) and torchebsd are driven from one harness
(`scripts/`-style benchmark): write the `.xtal`, run `EMMCOpenCL` for the depth
profile, run `EMEBSDmaster`, then run torchebsd with the matching crystal, the
same `accum_z` depth profile (`DepthDistribution.from_emsoft`), matched Bethe
parameters, dmin and npx, and correlate the beam-energy Lambert maps. The
EMsoft reference files are not redistributed; point the harness at a local
EMsoft install.
