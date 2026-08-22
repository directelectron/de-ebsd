# Spherical Indexing (`torchebsd.index`) — plan & project structure

**Status:** planning. **Goal:** a fast, GPU-accelerated, *differentiable* spherical
indexing engine for EBSD — the matching counterpart to `torchebsd.sim`'s
differentiable master-pattern *generation* — modelled on **EMSphInx** (Lenthe,
Singh & De Graef, *Ultramicroscopy* 2019), implemented in PyTorch, that emits
**orix** orientations / `CrystalMap`s and plugs into **kikuchipy**. Rich
visualization throughout, matching the simulation side.

---

## 1. What spherical indexing is, and why it is fast

**Dictionary indexing (DI)** — kikuchipy's current workhorse — simulates a large
*dictionary* of patterns (one per sampled orientation) and, for each experimental
pattern, finds the best match by normalized cross-correlation. Cost per pattern is
`O(N_orientations × N_pixels)`; the dictionary is large and the search is brute
force.

**Spherical indexing (EMSphInx)** reframes the match on the sphere. The dynamical
master pattern *is* a function on the unit sphere `M(n)`. Back-project an
experimental pattern to the sphere, `E(n)`. The normalized cross-correlation
between `E` and `M` **as a function of the trial orientation** `g ∈ SO(3)`,

```
C(g) = ⟨ E(n) , M(g·n) ⟩ ,
```

is itself a function on SO(3). Expand both signals in **spherical harmonics** up
to a bandwidth `L`; then `C(g)` is a band-limited function on SO(3) whose
**Wigner-D (SO(3) Fourier) coefficients are the outer product of the two harmonic
spectra**. One inverse SO(3) FFT (the SOFT algorithm) yields `C(g)` on a whole
grid of orientations at once — `O(L³ log L)` instead of a brute-force sweep. The
peak of `C(g)` is the orientation; a local refinement sharpens it.

The primitives are therefore: a **spherical harmonic transform (SHT)**, the
**Wigner-D / SO(3) cross-correlation**, **peak finding**, and **refinement**.

## 2. Why PyTorch (the differentiator vs EMSphInx)

- The SHT and the SO(3) transform are batched linear algebra → **GPU**, and
  naturally batch over an **entire scan** (10⁴–10⁶ patterns).
- The whole path is **differentiable**. That buys three things EMSphInx cannot:
  1. gradient-based **orientation refinement** on the correlation surface;
  2. joint **pattern-center / projection-geometry** refinement — the classic hard
     EBSD calibration — by autograd through the back-projection;
  3. because `torchebsd.sim` master patterns are already differentiable in the
     **crystal parameters** (positions, occupancy, Debye–Waller — see
     `DESIGN.md` M4), forward-model *fitting* of the structure to a real scan
     becomes possible. This is the research frontier the differentiable simulator
     uniquely opens.

## 3. Dependency direction & the groundwork we build on

**torchebsd is a foundational engine.** In the crystallography ecosystem it
depends only on **orix** -- the shared base that kikuchipy also builds on. It does
**not** depend on kikuchipy. The intended direction is the reverse: **kikuchipy
(and other tools) depend on torchebsd** and call it as a GPU-accelerated,
differentiable simulation + indexing backend, consuming its orix-native output.
So `torchebsd.index` reuses only orix and `torchebsd.sim`'s own detector geometry;
it never imports kikuchipy.

| Piece | From | Used for |
| --- | --- | --- |
| Orientations, symmetry, SO(3)/FZ sampling, misorientation, `CrystalMap`, IPF colouring | **orix** (dependency) | trial grid, refinement search, output `Rotation`/`CrystalMap`, viz |
| `EBSDDetector` geometry (kikuchipy-parity PC conventions), master projection, square-Lambert sphere | **`torchebsd.sim`** (in-package) | back-projection geometry, the continuous "dictionary" `M(n)` |
| Differentiable master, differentiable in crystal params (DESIGN.md M4) | **`torchebsd.sim`** | forward model for refinement / fitting |

The novel part we build is the **SH + SO(3) matching engine** on top of orix +
`torchebsd.sim`. Everything geometric (detector, projection, Lambert sphere) is
already in `torchebsd.sim` -- torchebsd is self-contained, no kikuchipy needed.
kikuchipy appears only as an **optional validation baseline** (its dictionary
indexing, behind a test/benchmark extra), never at runtime. Emitting orix-native
objects is exactly what lets kikuchipy depend on torchebsd and call it.

## 4. The pipeline

```
pattern arrays (H x W scan) ─┐
torchebsd EBSDDetector ──────┤   (kikuchipy hands us arrays + geometry; we don't import it)
torchebsd master M(n) ───────┘
        │
  1. preprocess (background / mask / normalize -- basic in-package, or done upstream)
  2. back-project each pattern to the sphere  E(n)           [index.backproject]
  3. forward SHT:  M -> a_lm (once) ;  E -> b_lm (batched)    [index.harmonics]
  4. SO(3) cross-correlation via Wigner-D FFT  ->  C(g)       [index.wigner + index.correlation]
  5. peak on SO(3)  ->  candidate orientation(s)              [index.peaks]
  6. refine (sub-pixel SO(3); gradient on the forward model) [index.refine]
  7. assemble orix Rotation per pixel -> CrystalMap          [index.io]
```

## 5. Project structure

```
src/torchebsd/
  sim/                 # existing simulation code (moved into its own subpackage)
  index/               # NEW spherical-indexing package
    __init__.py
    harmonics.py       # spherical harmonic transform (forward/inverse), GPU, batched, autograd
    wigner.py          # Wigner-d / D matrices, SO(3) grid, the SOFT transform
    correlation.py     # SO(3) cross-correlation: SH spectra product -> inverse SO(3) FFT
    backproject.py     # detector pattern <-> sphere resampling (EBSDDetector geometry)
    peaks.py           # peak detection + sub-pixel refinement on the SO(3) grid
    refine.py          # gradient-based orientation / pattern-center refinement (differentiable forward model)
    indexer.py         # SphericalIndexer: end-to-end driver over a scan -> CrystalMap
    io.py              # orix / kikuchipy adapters (Rotation, CrystalMap, EBSD signal, master import)
    display.py         # rich viz (anyplotlib / anywidget), matching the sim side
tests/
  index/               # unit + integration tests
scripts/
  validate_indexing.py # accuracy/speed vs kikuchipy DI and (where available) EMSphInx
INDEXING_DESIGN.md     # this document
```

`torchebsd.index` sits beside `torchebsd.sim`: **sim generates, index matches**,
and both are differentiable and GPU-native.

## 6. Milestones

| # | Milestone | Acceptance |
| --- | --- | --- |
| I1 | **Spherical harmonic transform** — forward/inverse, GPU-batched, autograd | round-trip to ~1e-6; matches a reference SHT (scipy `sph_harm` sums) |
| I2 | **SO(3) cross-correlation** — Wigner-D FFT (SOFT) | matches a brute-force orientation sweep on a small grid to ~1e-5 |
| I3 | **Back-projection** — detector ↔ sphere via `EBSDDetector` | round-trips kikuchipy's own projection of a master pattern |
| I4 | **Single-pattern indexing** — end to end | recovers a known orientation from a `sim`-generated pattern to < grid resolution; agrees with kikuchipy DI |
| I5 | **Scan indexing** — batched map → `CrystalMap` | full Ni/Si map; **benchmark vs kikuchipy DI** (and EMSphInx if available) |
| I6 | **Refinement** — sub-pixel SO(3) + gradient orientation/PC | orientation error below the DI grid; PC refinement demonstrated |
| I7 | **Visualization** — SO(3) surface, IPF + quality maps, pattern overlay | interactive figures like `sim.show_setup` |
| I8 | **Integration** — orix-native output (`Rotation`/`CrystalMap`), array-based ingest (no kikuchipy import); documented so kikuchipy can *depend on* torchebsd | orix round-trip; a kikuchipy dataset indexes through torchebsd |

Sequencing: **I1 → I2 first** (the maths engine, validated against brute force),
then I3–I4 (geometry + one pattern), then I5 (scale + benchmark), then I6–I8
(refinement, viz, integration). Each is its own focused, validated increment —
the same rhythm as the `sim`/MC/M4 work.

## 7. Validation & benchmarking

Same philosophy as `BENCHMARKS.md`: an external oracle, matched inputs, honest
numbers.

- **Primitives** (I1/I2) against analytic / brute-force references.
- **Round-trip**: `sim` a pattern at a known orientation → index it back →
  disorientation should be below the SO(3) grid resolution.
- **vs kikuchipy dictionary indexing** on a real dataset: same orientations,
  measured speedup (the batched SH/SO(3) path should win at scale).
- **vs EMSphInx (with EMsoft)** where installed — the oracle for *spherical*
  indexing exactly as EMsoft is for the master pattern; report accuracy + speed
  across crystal systems (reusing the `BENCHMARKS.md` harness).

## 8. Open decisions (to settle at I1)

- **SHT kernel**: direct matrix multiply (simple, GPU-friendly, fine to moderate
  `L`) vs a recursive fast SHT for large `L`. Start direct; profile.
- **Real vs complex** spherical-harmonic basis.
- **Bandwidth `L`** and **SO(3) grid resolution** — the accuracy/speed knobs.
- **Refinement search**: reuse orix's SO(3)/fundamental-zone sampling, or refine
  purely on the SOFT grid.
- **In-torch vs external** SH/SO(3) library — favour in-torch for GPU + autograd.

## 9. Dependencies

Runtime: `torch`, `numpy`, **`orix`**, `diffsims`, `h5py`, `anyplotlib`,
`anywidget`, plus **`torchebsd.sim`** as the forward model. **kikuchipy is NOT a
runtime dependency** -- it is an optional validation/benchmark baseline only (the
reverse direction is the goal: kikuchipy depends on torchebsd). No new hard
dependency is anticipated (the SH/SO(3) transforms are implemented in torch for
GPU execution and autograd).
