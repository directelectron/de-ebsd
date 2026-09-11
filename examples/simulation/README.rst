Master-pattern simulation
-------------------------

The forward chain: crystal -> structure factors -> reflections -> Bloch-wave
dynamical solve -> master pattern -> detector projection, with every
intermediate object inspectable. Start with the CIF-to-detector workflow, then
explore energy-resolved Monte Carlo statistics and numerical convergence.

``compare_emsoft.py`` compares against kikuchipy's bundled nickel reference
using its stored crystal, Bethe settings, reflection cutoff and Monte Carlo
depth profile. It reports correlations and normalized residuals for both
hemispheres and a detector projection, and displays both 3D spheres. The
reference is 8-bit, so the example does not claim bitwise intensity parity.
An unquantized local EMsoft file can be selected by editing ``reference_path``.
