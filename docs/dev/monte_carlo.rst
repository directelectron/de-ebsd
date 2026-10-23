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
