"""The EBSD pipeline Autopilot runs: indexing, pattern-center calibration,
refinement, crystal maps, the set-up, spots and unit cells, the phases and the
one accelerator lock.

Moved here verbatim from Autopilot's ``pyxem_base`` so the code lives in one
package. Parts of it overlap what the rest of ``de_ebsd`` does; they are
reconciled one behavior at a time, with this pipeline as the reference.
"""
