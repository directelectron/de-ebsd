# M-MC — Native electron Monte Carlo (scope & design)

**Status:** scoping. **Goal:** produce the exit depth/energy statistics that the
Bloch depth integration needs *natively*, removing torchebsd's last runtime
dependency on EMsoft-generated files. EMsoft `accum_z` stays the validation
oracle for the MC itself.

---

## 1. The one interface contract

The master-pattern pipeline consumes exactly one object,
`montecarlo.DepthDistribution`, through three members
(`src/torchebsd/montecarlo.py`):

- fields: `energies_kev (nE,)`, `depth_step` (nm), `accum_z (nE, nz, ny, nx)`,
  `n_total`;
- `integration_depths()` → per-energy `izzmax` (99 % depth);
- `depth_weights(energy, x_gp)` → `lambda_E(z)`, the **lateral-summed** exit
  fraction per depth bin, times `exp(+2 pi z / x_gp)`.

Both derived quantities reduce `accum_z` over the lateral axes, so the native MC
only needs the **1-D exit-depth histogram per energy bin** — it can emit
`accum_z` with `ny = nx = 1`. Everything downstream (`BlochWaveBackend.prepare`,
`BlochPrismBackend`) is untouched.

**Deliverable:** `DepthDistribution.from_simulation(material, microscope, ...)`
(or a `MonteCarloSimulator` producing the same dataclass) — a drop-in for
`DepthDistribution.from_emsoft`. The `single_energy` fallback is retired to a
true last resort.

---

## 2. Physics model (v1 targets parity with EMsoft `EMMCOpenCL`)

Matching EMsoft's MC model makes validation apples-to-apples (correlate against
its `accum_z` directly). The sample is a **homogeneous bulk** with mean
properties — EMsoft's MC does not resolve individual atoms.

- **Target material:** mean atomic number `Z̄`, mean atomic weight `Ā`, density
  `ρ`, atomic number density `n = ρ N_A / Ā`, mean ionization potential `J(Z̄)`.
- **Elastic scattering:** screened relativistic Rutherford. Screening parameter
  `α(Z, E)`; total cross-section `σ(E)` → elastic mean free path `λ = 1/(n σ)`;
  polar angle `θ` sampled from the screened differential cross-section by inverse
  CDF, azimuth `φ` uniform.
- **Inelastic loss:** continuous slowing down (CSDA) with the Joy–Luo modified
  Bethe stopping power `dE/ds`; energy lost over each elastic step length.
- **Geometry:** electrons enter a tilted (`microscope.sample_tilt`, 70°)
  semi-infinite bulk. Step: advance `λ`, lose `dE/ds·λ`, scatter, repeat.
  Terminate on (a) crossing back through the top surface → **backscattered**,
  record exit energy + depth into `accum_z`; (b) `E < E_min` (absorbed);
  (c) max depth / step-count guard.
- **`accum_z` binning:** must reproduce EMsoft's exact definition (which
  trajectory depth is scored, and the energy-bin edges `Ehistmin/Ebinsize`).
  **Action:** read `EMMCOpenCL.cl` + `mod_MCfiles.f90` before coding this —
  correctness-critical, and the one place a subtle mismatch would silently skew
  `lambda_E(z)`.

Screened Rutherford + CSDA is well-behaved down to ~1 keV; EBSD lives at
10–30 keV, comfortably in range.

---

## 3. Inputs — what exists vs. what to add

**Exists:** `Crystal.volume` (nm³), `Crystal.sites` (element, occupancy,
Debye–Waller), `Crystal.positions`, `scattering.atomic_number`,
`constants.C_AVOGADRO`, `Microscope.energy`, `Microscope.sample_tilt`.

**To add (small):**
- `scattering.atomic_weight(element)` — an element→A table (extend the existing
  `ELEMENT_SYMBOLS` / `_ATOMIC_NUMBER` machinery, or borrow from `diffpy`).
- `Crystal.density` and mean `Z̄` / `Ā` helpers: `ρ = Σ_i n_i A_i / (N_A · V)`
  over the occupancy-weighted orbit populations; composition-average `Z`, `A`.
- `J(Z)` mean ionization potential and any MC-only constants in `constants.py`.

---

## 4. Implementation shape

- **torch, vectorized over electrons.** Advance a batch of `N` electrons one
  step at a time (position, direction, energy tensors; a boolean "alive" mask),
  exactly the data-parallel structure of EMsoft's OpenCL kernel. `device`
  selects CPU/CUDA like the backends. `N ~ 1e6–1e7`, chunked to bound memory.
- **Seeded** via `torch.Generator` for reproducible tests.
- **numpy single-trajectory reference** first (readable, unit-testable against
  analytic CSDA / cross-section moments), then the batched torch port validated
  against it — mirrors the `bloch` → `bloch_prism` pattern already in the repo.

### Differentiability — dual-mode (decided: autograd from the start)

A naive trajectory MC is not differentiable: it has discrete scattering counts,
a hard surface-exit boundary, an `E < E_min` cutoff, and histogram binning.
Making `accum_z(φ)` differentiable w.r.t. continuous material parameters
`φ = {Z̄, Ā, ρ, J, E_0, tilt}` (so gradients can flow into the M4 pipeline) uses
four standard differentiable-simulation techniques, and the module runs in two
modes sharing one code path:

- **Reparameterized sampling.** Draw the uniforms `u ~ U(0,1)` once and *detach*
  them; step length `s = -λ(φ,E)·ln u` and the screened-Rutherford polar angle
  (inverse-CDF) are then smooth functions of `φ`. Position, direction, and
  energy evolve as differentiable functions of `φ` for fixed noise (pathwise /
  reparameterization-trick gradients).
- **Fixed step count.** Run all electrons for a constant `S_max` steps (pad
  instead of terminating), carrying a continuous survival weight `w ∈ [0,1]`
  that smooth gates (sigmoids in depth / energy) multiply down — replacing the
  discrete "trajectory ended" branch.
- **Soft exit + soft binning.** The top-surface crossing and the `E_min` cutoff
  become sigmoid gates (temperature `τ`); each electron's contribution is
  scattered into the depth/energy bins of `accum_z` through a normalized kernel
  (triangular/Gaussian) centered on its continuous exit depth and energy — the
  differentiable-rendering "soft rasterization" trick.

**Two modes, one implementation:**
- **hard mode** — `τ → 0`, straight-through / discrete gates: the exact
  EMsoft-style forward used for the validation number (must match EMsoft
  `accum_z`) and for production depth profiles;
- **soft mode** — finite `τ`: forward value converges to hard mode as `τ → 0`
  and provides gradients.

Trade-off to manage: finite `τ` biases the forward value, and pathwise MC
gradients have variance — so temperature annealing and electron count are tuned
against the gradient-validation study (§5), and the *forward* master pattern
always uses hard mode.

New module: `src/torchebsd/montecarlo_sim.py` (keeps the EMsoft *reader* in
`montecarlo.py` clean), exporting the simulator and the `from_simulation`
constructor. Tests in `tests/test_montecarlo_sim.py`.

---

## 5. Validation plan

- **Primary:** correlate native `lambda_E(z)` / `accum_z` against EMsoft
  `accum_z` for Si and Ni at 20 keV (profile-shape match per energy bin).
- **Physics sanity:** backscatter yield `η` vs. Reimer/Joy empirical values
  (Ni η ≈ 0.30, Si η ≈ 0.16 at 20 keV); median exit depth vs. known ranges; BSE
  energy spectrum shape.
- **End-to-end:** swap the native MC in as the depth source and re-run the
  master pattern (hard mode) — Si should stay `r ≈ 1.0000` (native profile ≈
  EMsoft profile). Note: Ni's `r = 0.93` gate gap is tracked to reference
  provenance, not the depth model, so the MC is unlikely to fully close it.
- **Gradient validation (soft mode):** autograd `d lambda_E(z) / dφ` for
  `φ ∈ {Z̄, ρ, E_0}` vs. **common-random-number finite differences** of the
  forward MC (CRN is required for low-variance FD of a stochastic simulator);
  plus a `τ`-annealing study showing soft-forward → hard-forward and the
  bias/variance trade as `τ → 0`.
- **Unit tests:** step energy loss vs. analytic CSDA; sampled elastic-angle
  moments vs. the cross-section; MFP vs. `1/(nσ)`; zero-tilt hemispherical
  symmetry; determinism under a fixed seed; soft-mode `accum_z` → hard-mode as
  `τ → 0`.

---

## 6. Task breakdown

| Sub | Work | Size |
| --- | --- | --- |
| M-MC.1 | `atomic_weight`, `Crystal.density`/`Z̄`/`Ā`, `J(Z)`, MC constants | S — **done** |
| M-MC.2 | numpy single-trajectory reference + physics unit tests | M — **done** |
| M-MC.3 | vectorized torch batch MC (hard mode) → `DepthDistribution.from_simulation` | M–L — **done** |
| M-MC.4 | validation vs EMsoft `accum_z` (Si, Ni) + yield sanity | M — **done** |
| M-MC.5 | differentiable soft mode (reparam draws, soft gates, kernel binning) | L |
| M-MC.6 | gradient validation (autograd vs CRN finite diff) + `τ`-annealing study | M |
| M-MC.7 | wire in as the default depth source; docs; roadmap update | S — **done** |

Sequencing: **M-MC.3 (hard mode) first** — it pins physics correctness against
EMsoft with no differentiability confound — then layer the soft mode (M-MC.5)
on the same code path.

---

## 7. Decisions (resolved)

1. **Elastic model / validation target — screened Rutherford, match EMsoft.**
   Reproduce `EMMCOpenCL`'s model so `accum_z` correlates directly against its
   output; success bar = reproduce EMsoft `accum_z` shape + sane backscatter
   yields. Mott cross-sections deferred as a later fidelity refinement.
2. **Differentiability — autograd from the start**, via the dual-mode design in
   §4 (hard forward for validation/production, soft mode for gradients).

---

## 8. Risks / unknowns

- Exact `accum_z` scoring + energy-bin semantics (resolve by reading EMsoft
  source before M-MC.3).
- Screened-Rutherford/CSDA accuracy at the low-energy tail of the bin range
  (minor for 10–30 keV EBSD).
- **Soft-mode bias/variance:** the `τ` gates bias the forward value and pathwise
  MC gradients have variance; mitigated by annealing, electron count, CRN, and
  keeping the forward pattern on hard mode. The main new risk from the
  differentiability decision.
- Throughput for `1e7` electrons on the Pascal dev GPU (chunk + profile); soft
  mode holds `S_max` steps for *all* electrons, raising the floor cost.

---

## Appendix A. EMsoft bulk-MC model (pinned from `opencl/EMMC.cl`)

Extracted verbatim from EMsoft's OpenCL kernel and ported in `montecarlo_sim.py`
(M-MC.2). Energies in keV, `rho` g/cm^3, `A` g/mol, lengths nm; `Z` may be the
composition mean. Per step, at current energy `E`:

- screening: `alpha = 3.4e-3 * Z^(2/3) / E`
- cross-section (× N_A): `sig_eNA = 5.21*602.2 * Z^2/E^2 * 4pi/(alpha(1+alpha)) * ((E+511)/(E+1024))^2`
- mean free path: `mfp = A * 1e7 / (rho * sig_eNA)` nm
- step (exponential): `step = -mfp * ln(u)`, `u ~ U(0,1]`
- ionization potential: `J = (9.76 Z + 58.5 Z^-0.19) * 1e-3` keV
- stopping power: `de_ds = -0.00785 * Z/(A E) * ln(1.166 E/J + 0.9911)`
- energy update: `E += step * rho * de_ds`
- polar angle: `cos(theta) = 1 - 2 alpha u / (1 + alpha - u)`, `u ~ U(0,1)`
- azimuth: `psi = 2 pi u`
- direction rotation: the EMsoft cosine formula, with a `|c_z| >= 0.99999`
  pole branch
- entry: `r0 = (0,0,0)`, `c0 = (cos(omega) sin(sig), sin(omega) sin(sig), cos(sig))`
  for sample tilt `sig`
- exit / backscatter when `z <= 0`; the depth measure is
  `escape_depth = z / c_z` (its exact role in `accum_z` binning is finalized in
  M-MC.3)

Note the angular distribution is strongly forward-peaked (`alpha` ~ 1e-3), so
large-angle events need `u` within ~1e-9 of 1 -- backscatter accumulates from
many small deflections, not single large ones. There is **no** energy floor in
the kernel; the `Emin` cutoff and `accum_z`/`accum_e` binning live host-side
(`mod_MCOpenCL`).

The host-side binning is now reproduced (M-MC.3): energy
`iE = round((E - Ehistmin)/Ebinsize)`, depth `iz = round(|escape_depth|/depthstep)`,
accept `Ehistmin < E <= EkeV` and `0 <= iz < numzbins`, where
`escape_depth = z_pre / c_z` (pre-step depth over the exit direction's z-cosine).
The `Emin` cutoff doubles as the MC absorption threshold (a sub-`Emin` electron
can only lose more energy, so it can never re-enter a bin).

---

## Appendix B. M-MC.4 validation result

Validated against EMsoft's `Si_MC.h5` and `Ni_MC.h5` (`scripts/validate_montecarlo.py`,
run with each reference's own tilt / energy / depth binning). Per-energy-bin
Pearson correlation of the exit-depth profiles and total backscatter yield:

| Material | min per-bin corr | yield (EMsoft / native) |
| --- | --- | --- |
| Si | 0.9959 | 0.4496 / 0.4499 |
| Ni | 0.9989 | 0.5687 / 0.5655 |

Every energy bin agrees to > 0.995; mean exit depths match within ~0.5 nm (Si
~18 nm, Ni ~4 nm -- Ni is denser, so shallower). Two findings changed the plan:

- **The lateral exit window is not needed** (and hurts). EMsoft stores `accum_z`
  on a central `+/- numsx/10` window; restricting the native MC to near-normal
  exit *lowered* the correlation. Counting every backscattered electron matches
  EMsoft's depth profile, so `from_simulation` deliberately applies no window.
- **Surface reflections must be excluded.** Electrons that scatter back out on
  step 1 (never entering the bulk, `z_pre = 0 -> escape_depth = 0`) piled a spike
  into the top energy bin's `z = 0` and dropped its correlation to 0.59. EMsoft
  drops them; requiring an electron to have entered the bulk before a backscatter
  counts restores the top bin to 0.998. A `penetrated` flag enforces this in both
  the scalar reference and the batched MC.
