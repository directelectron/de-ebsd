Orientation indexing
--------------------

The inverse problem: recover the orientation that produced a measured pattern.
Both examples index a synthetic two-grain scan projected from a de-ebsd master
pattern, so the answer is known and the error can be quoted -- a closed round
trip that validates the conventions and the workflow, not experimental accuracy.

``dictionary_indexing.py`` is the brute-force baseline: project the master at
every orientation of a fundamental-zone sample and take the best normalized
cross-correlation. ``spherical_indexing.py`` replaces that library with a
spherical-harmonic cross-correlation over all of SO(3), whose cost does not
depend on the crystal symmetry. Both end in gradient refinement, which is what
takes a coarse grid hit to sub-degree accuracy.
