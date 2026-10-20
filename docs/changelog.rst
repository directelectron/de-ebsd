=========
Changelog
=========

All notable changes to de-ebsd. This project follows
`Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_ and
`semantic versioning <https://semver.org/>`_.

Unreleased
==========

Added
-----
- Dictionary and spherical indexing examples, each a closed round trip on a
  synthetic two-grain scan with known orientations.
- This documentation site.

Changed
-------
- **The package is renamed** from ``torchebsd`` to ``de_ebsd`` (distribution
  ``de-ebsd``). No legacy import alias is installed; see
  :doc:`user/installation`.
- The viz widgets' embedded JavaScript moved into standalone ``.js`` files
  shipped as package data.
- Gallery examples lost their ``plot_##`` prefixes and regrouped into
  ``examples/simulation`` and ``examples/indexing``.

Fixed
-----
- A simulated master pattern recorded ``repr(crystal)`` in
  ``metadata["crystal"]`` where an EMsoft-imported one recorded the
  :class:`~de_ebsd.crystal.Crystal`, so
  :class:`~de_ebsd.index.DictionaryIndexer` could not derive the point group
  from a master de-ebsd had just produced.

0.1.0
=====

Initial development version.
