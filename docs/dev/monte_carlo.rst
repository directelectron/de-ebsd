==============================
Monte Carlo electron transport
==============================

The depth integration needs the exit-depth statistics of backscattered
electrons. de-ebsd computes them natively (:mod:`de_ebsd.sim.montecarlo`), so a
plain ``compute()`` needs no EMsoft files; EMsoft's ``accum_z`` remains the
validation oracle and can be read with ``DepthDistribution.from_emsoft``.

The interface contract
======================

The master-pattern pipeline consumes one object,
:class:`~de_ebsd.sim.montecarlo.DepthDistribution`, through three members:
``energies_kev``, ``depth_step`` and ``accum_z`` (with ``n_total``);
``integration_depths()``, the per-energy 99 % depth; and
``depth_weights(energy, x_gp)``, the laterally summed exit fraction per depth
bin times ``exp(+2 pi z / x_gp)``. Both derived quantities reduce ``accum_z``
over the lateral axes, so the native Monte Carlo only emits a one-dimensional
exit-depth histogram per energy bin.

Physics model
=============

The sample is a homogeneous bulk of mean atomic number ``Z``, mean atomic weight
``A`` and density ``rho``; individual atoms are not resolved. The model is the
one in EMsoft's ``EMMCOpenCL`` kernel, so the output correlates directly with
``accum_z``. Energies are in keV, ``rho`` in g/cm\ :sup:`3`, ``A`` in g/mol and
lengths in nm. Per step, at energy ``E``:

- screening ``alpha = 3.4e-3 Z^(2/3) / E``;
- cross-section times Avogadro's number
  ``5.21 * 602.2 * Z^2 / E^2 * 4 pi / (alpha (1 + alpha)) * ((E + 511) / (E + 1024))^2``,
  and mean free path ``A * 1e7 / (rho * sigma)`` nm;
- exponential step ``-mfp ln u``;
- mean ionization potential ``J = (9.76 Z + 58.5 Z^-0.19) * 1e-3`` keV and
  stopping power ``-0.00785 Z / (A E) ln(1.166 E / J + 0.9911)``, so the energy
  changes by ``step * rho * dE/ds``;
- polar angle ``cos(theta) = 1 - 2 alpha u / (1 + alpha - u)`` and uniform
  azimuth, rotated into the direction with a pole branch at
  ``|c_z| >= 0.99999``;
- entry at the origin with direction
  ``(cos(omega) sin(sigma), sin(omega) sin(sigma), cos(sigma))`` for sample tilt
  ``sigma``; the electron is backscattered when ``z <= 0``.

The angular distribution is strongly forward-peaked (``alpha`` of order 1e-3),
so backscatter accumulates from many small deflections.

Mott elastic scattering
-----------------------

Screened Rutherford is the first Born approximation. It scatters too much
between about 5 and 60 degrees and too little past 90 (for nickel at 15 keV its
differential cross section is 1.3-1.7 times the Mott value from 5 to 20 degrees
and 0.8 times past 120, and its transport cross section is 17 % high), so it
overestimates the backscatter yield. ``elastic="mott"`` (or
``compute(mc_elastic="mott")``) replaces it with Mott cross sections computed by
ELSEPA :cite:`salvat2005elsepa` for the neutral free atom with ELSEPA's default
model (Dirac-Fock density, Fermi nucleus, Furness-McCarthy exchange, no
polarization or absorption):

- mean free path ``1 / (n sum_i x_i sigma_i(E))`` from ELSEPA's total cross
  sections, interpolated in ``ln sigma`` against ``ln E``;
- the atom of each collision drawn with probability ``x_i sigma_i / sum``, for a
  compound, ``x_i`` its number fraction;
- the polar angle from the inverse cumulative distribution of that atom's
  differential cross section, interpolated between the two bracketing table
  energies in ``ln E``.

The package table (:func:`~de_ebsd.sim.montecarlo.mott_cross_sections`, 41
elements, 50 eV to 30 keV) keeps ELSEPA's differential cross section at 30
energies and 128 of its 606 angles. Against ELSEPA's full output (119 energies,
all angles) the sampled distribution differs by at most 0.0011 in cumulative
probability above 1 keV, and nickel's backscatter coefficient by under 0.001.
``scripts/generate_mott_tables.py`` rebuilds the table, or one with other
elements, from an ELSEPA executable.

========================  ==========  ===================  =====
15 keV, normal incidence  measured    screened Rutherford  Mott
========================  ==========  ===================  =====
Ni                        about 0.30  0.328                0.304
Al                                    0.168                0.155
Au                                    0.558                0.481
========================  ==========  ===================  =====

The Mott model applies to the hard-mode Monte Carlo
(``simulate_depth_distribution``); the scalar reference trajectory and the
differentiable soft mode remain screened Rutherford.

Binning follows EMsoft's host code: energy bin
``round((E - Ehistmin) / Ebinsize)``, depth bin
``round(|escape_depth| / depthstep)`` with ``escape_depth = z_pre / c_z``, and
acceptance of ``Ehistmin < E <= E_beam`` and ``0 <= iz < numzbins``. The ``Emin``
cutoff doubles as the absorption threshold. Two choices matter for agreement:

- **No lateral exit window.** EMsoft stores ``accum_z`` on a central window, but
  restricting the native Monte Carlo to near-normal exit lowers the correlation;
  counting every backscattered electron matches EMsoft's depth profile.
- **Surface reflections are excluded.** An electron that scatters back out on its
  first step never entered the bulk and would pile a spike into the top energy
  bin at ``z = 0``. A ``penetrated`` flag requires an electron to have entered
  the bulk before a backscatter counts.

Validation
==========

Against EMsoft's ``Si_MC.h5`` and ``Ni_MC.h5`` (``scripts/validate_montecarlo.py``,
run with each reference's own tilt, energy and depth binning), the per-energy-bin
Pearson correlation of the exit-depth profiles and the backscatter yield are:

========  =================  =======================
Material  min per-bin corr.  yield (EMsoft / native)
========  =================  =======================
Si        0.9959             0.4496 / 0.4499
Ni        0.9989             0.5687 / 0.5655
========  =================  =======================

Mean exit depths agree within about 0.5 nm (silicon about 18 nm, nickel about
4 nm).

Implementation
==============

The batched torch implementation advances all electrons one step at a time with
an alive mask, the data-parallel structure of EMsoft's kernel, and is seeded
through ``torch.Generator`` for reproducibility. It is validated against a
single-trajectory numpy reference.

A differentiable soft mode, ``soft_depth_histogram``, shares the code path. It
draws the uniforms once and detaches them (reparameterization), runs a fixed
number of steps with sigmoid survival and exit gates, and bins each electron into
the depth and energy histogram through a bilinear kernel. The histogram is then
a smooth function of ``mean_z``, ``mean_a``, ``density`` and
``mean_ionization_kev``. The forward value converges to the hard mode as the gate
temperatures shrink (finite temperatures under-count, since soft gates leak), and
autograd ``d(yield)/d(density)`` matches common-random-number finite differences
to about 2 %. The hard mode is always used for the forward master pattern. The
grid-boundary acceptance is still a hard mask, so gradients dominated by
electrons near the edges of the energy or depth grid (for example ``d/dZ`` of the
total yield) are noisier.
