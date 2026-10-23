=======================
The dynamical algorithm
=======================

The master pattern is computed as in EMsoft's ``EMEBSDmaster``
(``mod_EBSDmaster.f90``), per energy bin ``E`` and incident direction ``k``.

1. **Structure factors.** For every ``hkl`` in the lookup box (out to twice the
   ``dmin`` sphere), Weickenmeier-Kohl complex atomic amplitudes with phonon and
   core absorption (``FSCATT``, ported with EMsoft's exact constants) are summed
   over the orbit with phases ``exp(-2 pi i g.r)`` and scaled by
   ``pref = 0.04787801 / (4 pi V_cell)``, then ``U = preg V`` with
   ``preg = 0.664840340614319`` (``CalcUcg``). ``V_0``, ``U'_0`` and the
   refraction-corrected relativistic wavelength come from the ``g = 0`` entry.

2. **Incident directions.** The square Rosca-Lambert grid, both hemispheres, as
   unit vectors in the Cartesian crystal frame (``k = k_hat / lambda``,
   ``k_n = 1 / lambda``). Only the point group's fundamental sector is computed
   and the result replicated by symmetry.

3. **Reflection list.** All centering-allowed ``hkl`` with ``d >= dmin`` and
   ``|s_g| / (lambda |U_g|) <= c3``. Forbidden reflections are kept within
   ``|s_g| <= sg_double`` as double-diffraction channels. Excitation errors come
   from the Ewald-sphere relation ``s_g = -g.(2k + g) / (2 (k + g).n)`` with the
   foil normal along the beam.

4. **Bethe partition.** For each candidate ``g``,
   ``m = min_h |s_g| / (lambda |U_{g-h}|)`` over all candidates ``h``. A
   reflection is strong when ``m <= c1``, weak when ``c1 < m <= c2`` and dropped
   otherwise. Systematically absent reflections carry no structure factor and are
   never strong beams; promoting them was the one thing that held silicon below
   exact agreement with EMsoft.

5. **Dynamical matrix** (nm\ :sup:`-2`, ``GetDynMat``):
   ``A_gh = U_{g-h} - (lambda/2) sum_w U_{g-w} U_{w-h} / s_w`` off the diagonal,
   ``A_gg = 2 s_g / lambda - (lambda/2) sum_w |U_{g-w}|^2 / s_w + i U'_0``.

6. **Bloch solve and depth integration** (``CalcLgh``). Eigendecompose ``A``,
   scale the eigenvalues ``W -> W / (2 k_n)``, invert the eigenvector matrix
   ``C`` and take ``alpha = C^-1[:, 0]``. Then
   ``I_jk = conj(alpha_j) alpha_k sum_z lambda_E(z) exp(-q_jk z) dz / t`` with
   ``q_jk = 2 pi [(Im W_j + Im W_k) + i (Re W_j - Re W_k)]`` (real part folded
   positive), and ``L_gh = conj(C) I C^T``. ``lambda_E(z)`` is the Monte Carlo
   exit-depth fraction times ``exp(+2 pi z / x_gp)``, which cancels the normal
   absorption already in the eigenvalues; the depth limit is the 99 % histogram
   depth. ``bloch_prism`` evaluates the same integral through the scattering
   matrix instead of an eigendecomposition.

7. **S_gh weighting.** Per asymmetric-unit site,
   ``S_gh = sum_atoms Z^2 exp(-B s^2) exp(+2 pi i (h - g).r)``, and the
   master-pattern intensity is ``I(k) = Re sum_gh L_gh S_gh / n_atoms``.

8. **Accumulation.** Intensities land at their Lambert pixel on the correct
   hemisphere, each computed pixel written to all its symmetry equivalents.

9. **Energy loop.** Steps 2-8 repeat per energy bin, since the wavelength,
   reflection lists and matrices change with ``E``; an energy-resolved master
   has shape ``(nE, 2, n, n)``. Trigonal and hexagonal systems would use the
   hexagonal Lambert grid with a final square resampling, which is not
   implemented.
