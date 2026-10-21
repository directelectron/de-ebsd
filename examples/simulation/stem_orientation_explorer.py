"""
Explore STEM zone axes with lines and spots
==========================================

Load a crystal, rotate it, and follow geometric Kikuchi lines toward a zone
axis while watching the kinematic spot pattern. Both layers share a forward
gnomonic projection and can be switched on or off independently.

No master pattern or transport calculation is needed. Intensities are relative
elastic kinematic values, not a quantitative simulation of STEM Kikuchi
contrast, thickness-dependent dynamical diffraction, or convergent-beam discs.

Use a live Jupyter notebook for the controls and linked rotation. A standalone
HTML figure is a snapshot: its camera can move locally, but recalculating the
linked patterns requires a Python kernel or application bridge.
"""

import de_ebsd as de

# %%
# Silicon uses a symmetry-expanded diamond structure. Lengths are in nm;
# the explorer's beam-energy control is explicitly in keV.
crystal = de.Crystal.from_parameters(
    a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))]
)
explorer = crystal.explore(energy_kev=200, dmin=0.1)
explorer.set_zone_axis((0, 0, 1))
explorer

# %%
# Hosts such as SpyDE can mount this same native anyplotlib figure and retain
# the explorer object to keep its Python callbacks alive.
figure = explorer.figure
figure

# %%
# The same controls are available from Python. These small tilts move away
# from [001]; reset returns to the starting orientation. Setting roll rotates
# the screen basis about the beam, and the two layers stay registered.
explorer.set_fine_tilt(tilt_x_deg=0.2, tilt_y_deg=-0.1)
explorer.set_roll(15)
explorer.set_show_lines(False)
explorer.set_show_lines(True)
explorer.reset()
