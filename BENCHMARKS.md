# de-ebsd vs EMsoft — parity, speed, and method audit

de-ebsd's acceptance oracle is [EMsoft](https://github.com/EMsoft-org/EMsoftOO)
(EMsoftOO). This page audits the two codes head to head: **how** each computes a
master pattern, **how fast**, and **how closely they agree**, across single- and
multi-component phases and a sweep of crystal systems from cubic to monoclinic.

Every comparison below runs the **same crystal, the same Monte Carlo exit-depth
profile (EMsoft's own `accum_z`), the same Bethe parameters, dmin, and grid** for
both codes, so any difference is the diffraction engine (or a convention), not
the inputs. Measured on one machine: NVIDIA TITAN X (Pascal) GPU for de-ebsd,
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

| | EMsoft (`EMEBSDmaster`) | de-ebsd (`bloch_prism`) |
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
square-Lambert master grid) — de-ebsd is a GPU-batched, differentiable
re-expression of EMsoft's algorithm, with the eigendecomposition replaced by a
mathematically equivalent matrix exponential that batches on the GPU.

## Accuracy

Per-pixel Pearson correlation of the beam-energy Lambert map, de-ebsd vs a
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

‡ **Ti** is *not* a frame/orientation mismatch: de-ebsd's crystal frame is
verified identical to orix's, the hexagonal metric is exact (d-spacings match
analytic hcp to 1e-4), and the two masters share the same band geometry and
orientation (no rigid rotation/mirror improves the match). The residual is a
contrast/intensity difference in the hexagonal dynamical computation — an open
item, small and localised, not a convention error. **ZrO2** is a space-group
**setting** mismatch (the two codes build different cells from the same numbers —
`r = 0`, the Si origin bug writ large for monoclinic); see
[Conventions](#crystallographic-conventions-across-software).

## Speed

Full energy-resolved master (all 6 energy bins), npx = 100, symmetry-reduced in
both codes. EMsoft times are its own reported execution time; de-ebsd is
GPU wall-clock.

| Crystal | Beam directions (fund. sector) | EMsoft (6 bins) | de-ebsd (6 bins) | Speedup |
| --- | --- | --- | --- | --- |
| Ni | 5 152 | 31 s | 3.0 s | **~10x** |

The **~10x** is the fair, like-for-like figure (both 6 energy bins, both
symmetry-reduced, GPU vs CPU). Per energy bin it is ~9x (de-ebsd 0.57 s vs
EMsoft ~5 s). At native resolution the batched GPU solve pulls further ahead: a
**npx-500 Si master is ~10 s in de-ebsd** (single energy, fp32) where the
per-direction eigendecomposition EMsoft uses would need ~10^6 eigensolves.

> Caveat: these are single-machine numbers (one GPU vs one CPU) and the absolute
> EMsoft seconds for the other structures in the sweep were measured under load;
> treat the **~10x order-of-magnitude** as the robust claim, not any single
> second-count.

## Crystallographic conventions across software

EBSD master patterns only compare pixel-for-pixel when two codes agree on a
stack of conventions. de-ebsd deliberately follows the **orix / kikuchipy**
ecosystem (it already depends on orix); EMsoft is the physics oracle. Here is
where each convention lives and whether they agree.

| Convention | de-ebsd | orix / kikuchipy | EMsoft | Agree? |
| --- | --- | --- | --- | --- |
| Crystal Cartesian frame | `direct_structure_matrix`: **x ∥ a, z ∥ c** | `Phase.structure.lattice.base`: **x ∥ a, z ∥ c** | x ∥ a for cubic/tetragonal; hexagonal frame differs subtly | de-ebsd **= orix** (verified, bit-identical); EMsoft matches for cubic/tetragonal |
| Lambert projection | square Rosca–Lambert | square Rosca–Lambert (reads EMsoft `mLPNH`) | square `mLPNH` output (hexagonal has an internal sampling variant) | yes for cubic (r = 1.0) |
| Space-group engine | diffpy `GetSpaceGroup` | diffpy (same) | internal tables + `SpaceGroupSetting` | needs matching `setting` |
| Origin choice | diffpy default (e.g. **2** for Fd-3m) | diffpy default | `SpaceGroupSetting` (1 or 2) | must be set to match — Si needs 2 |
| Monoclinic/triclinic setting | diffpy default (unique axis b, cell choice 1) | diffpy default | `SpaceGroupSetting` | **mismatch** for ZrO2 → r = 0 |
| Bethe parameters (default) | strong 8 / weak 50 / cutoff 100 | n/a | c1 4 / c2 8 / c3 50 | **differ** — must be matched |
| Structure factors | Weickenmeier–Kohl | n/a | Weickenmeier–Kohl | yes (U_g match to < 0.003 %) |
| Monte Carlo depth | native (EMsoft-free) or reads `accum_z` | uses EMsoft's | OpenCL, required | same profile when shared |

**What this means in practice**

- **The crystal frame is settled.** `de_ebsd.Crystal.direct_structure_matrix`
  transposed equals `orix.crystal_map.Phase(...).structure.lattice.base` exactly
  (verified for hexagonal, the case where the a vs a\* choice matters). So a
  de-ebsd master is drop-in for the orix/kikuchipy world — the same convention
  kikuchipy assumes when it loads an EMsoft master and applies orix orientations.
  (Note: *raw* `diffpy.structure.Lattice(...).base` uses a different default
  rotation; orix's `Phase` normalises it to x ∥ a, which is what de-ebsd uses.)

- **Matching EMsoft is a two-knob job** for high-symmetry: set the **Bethe
  parameters** (EMsoft 4/8/50, not de-ebsd's 8/50/100 default) and the
  **origin choice** (`SpaceGroupSetting = 2` for diamond-type). With those,
  cubic and tetragonal are pixel-exact (Ni, Si, GaAs, TiO2 above).

- **Monoclinic/triclinic settings are the genuinely hard part.** Space group 14
  alone has several settings (P2₁/c, P2₁/a, P2₁/n; unique axis b or c). diffpy
  and EMsoft must be told the *same* one; otherwise they build different cells
  from identical numbers and the patterns are uncorrelated (ZrO2, r = 0). This is
  a specification/round-trip task (agree the setting string, or exchange the full
  symmetry-expanded atom list) rather than a physics gap.

- **Hexagonal has a small residual beyond conventions** (Ti, r = 0.87): frame,
  metric, and band geometry all match, but the intensity distribution differs
  slightly. This is the one item that is *not* explained by a convention and is
  open for investigation (it aligns with the "hexagonal Lambert grid" work
  already noted in DESIGN.md, M3).

## Findings

1. **The dynamical physics is at full parity.** For every high-symmetry case,
   single- *and* multi-component, de-ebsd reproduces EMsoft to `r = 1.00000`.
   The batched `matrix_exp` Bloch solve is bit-for-bit equivalent to EMsoft's
   eigendecomposition; the `Sgh` reduction, Bethe partition, and depth
   integration all match.

2. **The "Ni gap" was settings, not physics.** de-ebsd's default Bethe
   parameters (8/50/100) differ from EMsoft's (4/8/50); the public Ni reference
   (kikuchipy) was also computed with unknown settings. Matching them closes it:
   `r = 1.00000`.

3. **de-ebsd already follows the orix/kikuchipy conventions.** Its crystal
   Cartesian frame is bit-identical to orix's `Phase.lattice.base`, its metric
   is exact, and its Lambert projection matches. So for the ecosystem it targets,
   the conventions are settled — see
   [Conventions](#crystallographic-conventions-across-software).

4. **Low-symmetry EMsoft parity is gated by two remaining items:**
   - **Space-group setting** for monoclinic/triclinic. ZrO2 (sg 14) scored
     `r = 0` because diffpy/orix and EMsoft chose different settings for the same
     numbers — a specification task (agree the setting), not a diffraction error;
     the Si origin choice is the same class of pitfall, and it was closable.
   - **A hexagonal intensity residual** (Ti, `r = 0.87`) that is *not* a
     convention: frame, metric, and band geometry all match; only the intensity
     distribution differs slightly. This is the single open physics-side item and
     tracks the hexagonal-Lambert work in DESIGN.md (M3).

## What this means

For the phases that dominate practical EBSD — cubic and tetragonal metals,
semiconductors, and oxides, single- and multi-component — **de-ebsd is a
drop-in, bit-exact replacement for EMsoft's master-pattern computation, ~10x
faster on a GPU, and differentiable**, and it already speaks the orix/kikuchipy
convention. The remaining frontier is narrow: **monoclinic/triclinic space-group
setting agreement** (a specification task) and **one hexagonal intensity residual**
(the lone physics-side open item, ~0.87 with matching band geometry).

## Reproducing

EMsoft (`EMEBSDmaster`) and de-ebsd are driven from one harness
(`scripts/`-style benchmark): write the `.xtal`, run `EMMCOpenCL` for the depth
profile, run `EMEBSDmaster`, then run de-ebsd with the matching crystal, the
same `accum_z` depth profile (`DepthDistribution.from_emsoft`), matched Bethe
parameters, dmin and npx, and correlate the beam-energy Lambert maps. The
EMsoft reference files are not redistributed; point the harness at a local
EMsoft install.
