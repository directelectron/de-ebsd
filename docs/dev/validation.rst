==========
Validation
==========

EMsoft is the ground-truth oracle, used for validation only: de-ebsd computes
the master pattern itself and does not call EMsoft at runtime.

``scripts/validate_master_pattern.py`` builds the same crystal, simulates,
resamples the reference onto the simulated grid through the shared
(kikuchipy-validated) interpolation, and reports the per-pixel Pearson
correlation of the Lambert maps for each hemisphere. The metric is scale
invariant, so normalization differences do not matter, while a frame or symmetry
error collapses it toward zero.

- Nickel, 20 keV, 200 pixels: kikuchipy's bundled
  ``nickel_ebsd_master_pattern_small``, which needs no external files.
- Silicon, 20 keV, 500 pixels, 11 energy bins: a local EMsoft ``Si_MC.h5`` run,
  given by ``--mc`` or ``DE_EBSD_SI_MC``. It also supplies the reference run's
  Monte Carlo statistics, crystal data and Bethe parameters. The file is not
  redistributed.

Parity with EMsoft
==================

Every comparison below uses the same crystal, the same exit-depth profile
(EMsoft's own ``accum_z``), the same Bethe parameters, ``dmin`` and grid for
both codes, so a difference is the diffraction engine or a convention, not the
inputs. Per-pixel Pearson correlation of the beam-energy Lambert map against a
fresh EMsoft master (100 pixels, ``dmin`` 0.05 nm, 20 keV, Bethe 4/8/50), summed
over EMsoft's per-atom-type maps for multi-component phases:

=======  =========================  ==========  =====  ===========
Crystal  System                     Components  Atoms  Correlation
=======  =========================  ==========  =====  ===========
Ni       cubic (fcc, 225)           1           4      1.00000
Si       cubic (diamond, 227)       1           8      1.00000
GaAs     cubic (zincblende, 216)    2           8      1.00000
TiO2     tetragonal (rutile, 136)   2           6      1.00000
Ti       hexagonal (hcp, 194)       1           2      0.871
ZrO2     monoclinic (14)            3           12     0.00
=======  =========================  ==========  =====  ===========

A native 500-pixel silicon master scores 1.0000 against ``Si_MC.h5``. The
agreement was closed by three things:

- ``U_g`` matches EMsoft's ``CalcUcg`` to better than 0.003 %.
- Systematically absent reflections must never be strong beams.
- The Bethe parameters must match: de-ebsd defaults to strong 8, weak 50,
  cutoff 100, EMsoft to 4/8/50, and an earlier nickel disagreement of 0.93 came
  from that default and an unknown reference provenance, not from the physics.

Silicon also needs the correct space-group origin choice (setting 2 for Fd-3m);
setting 1 scores 0.588.

Open items
----------

- **Monoclinic and triclinic settings.** Space group 14 alone has several
  settings (unique axis b or c, three cell choices). diffpy and EMsoft must be
  told the same one, otherwise they build different cells from identical numbers
  and the patterns are uncorrelated, as with ZrO2.
- **Hexagonal intensity.** For titanium the crystal frame (identical to orix's),
  the metric (hcp d-spacings to 1e-4) and the band geometry all agree, yet the
  intensity distribution differs slightly (0.871). This is not a convention
  error and is open; hexagonal Lambert sampling is related.

Speed
=====

A full energy-resolved nickel master (6 energy bins, symmetry-reduced in both
codes, 5152 directions) takes 3.0 s in de-ebsd on a Pascal TITAN X and 31 s in
EMsoft on 24 CPU threads, about 10 times faster; per energy bin 0.57 s against
about 5 s. At native resolution the gap widens, since a 500-pixel silicon master
is about 10 s while a per-direction eigendecomposition needs about a million
solves. These are single-machine numbers, so treat the order of magnitude as the
robust claim.

Conventions across software
===========================

=========================  ============================  ==========================
Convention                 de-ebsd and orix / kikuchipy  EMsoft
=========================  ============================  ==========================
Crystal Cartesian frame    x parallel to a, z to c*      same for cubic/tetragonal
Lambert projection         square Rosca-Lambert          square ``mLPNH``
Space-group engine         diffpy ``GetSpaceGroup``      internal tables and
                                                         ``SpaceGroupSetting``
Bethe defaults             8 / 50 / 100                  4 / 8 / 50
Structure factors          Weickenmeier-Kohl             Weickenmeier-Kohl
=========================  ============================  ==========================

The crystal frame is settled: ``Crystal.direct_structure_matrix`` transposed
equals ``orix.crystal_map.Phase(...).structure.lattice.base`` exactly, verified
for hexagonal cells where the a versus a* choice matters. (A raw
``diffpy.structure.Lattice(...).base`` uses a different default rotation; orix's
``Phase`` normalizes it.) Matching EMsoft for high symmetry is therefore a
two-setting job, the Bethe parameters and the origin choice.

Reproducing a comparison
========================

Write the ``.xtal`` file, run ``EMMCOpenCL`` for the depth profile and
``EMEBSDmaster``, then run de-ebsd with the matching crystal, the same
``accum_z`` (``DepthDistribution.from_emsoft``), matched Bethe parameters,
``dmin`` and ``npx``, and correlate the beam-energy Lambert maps. EMsoft files
are not redistributed.
