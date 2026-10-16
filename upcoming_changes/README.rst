Filing Change Log Entries
=========================

de-ebsd uses `towncrier <https://towncrier.readthedocs.io/>`_ to assemble
``CHANGELOG.rst``. When you open a pull request that should appear in the next
release notes, add a short news **fragment file** to this directory as part of
that PR.

Writing the entry with the change — rather than deriving notes from commit
subjects at release time — is the whole point. A commit subject says what the
author was doing; a release note has to say what the change means to someone
upgrading, and only the author knows that.

Naming convention
-----------------

Each fragment is a plain ``.rst`` file named::

    {PR_number}.{type}.rst

If the change has no natural PR number, name it ``+{slug}.{type}.rst`` — the
leading ``+`` marks it an "orphan" so towncrier omits the PR link. Without it
the slug renders as a broken link.

=================  ==============================================================
Type               Use when …
=================  ==============================================================
``api_change``     Existing behaviour changed in a way a user has to act on — a
                   signature, a default, a result that now comes out different.
                   Use this even when the change is a *fix*: what matters to
                   someone upgrading is that the old behaviour is gone.
``new_feature``    A user-visible capability has been added.
``bugfix``         A bug has been fixed.
``performance``    Something measurably got faster or lighter. Quote the number.
``deprecation``    Something is deprecated and will be removed later.
``removal``        A previously deprecated API has been removed.
``doc``            Documentation improved with no code change.
``maintenance``    Internal / infrastructure change invisible to users.
=================  ==============================================================

Content guidelines
------------------

* **One sentence per file**, in the **past tense**, from the *user's*
  perspective — not the implementer's.
* Say what changed for them, not which function you edited.
* Cross-reference a class or function with a Sphinx role where it earns its
  place.
* Do **not** put the PR number in the sentence; towncrier appends the link.

Example — ``12.bugfix.rst``::

    :func:`~de_ebsd.index.determine_pattern_center` recovered the pattern
    center from a start 0.03 off on a real dynamical master, where it used to
    see no slope and stay put.

Building
--------

The **Prepare Release** workflow runs ``towncrier build`` for you, so the
release PR carries the assembled changelog. To preview locally without
consuming the fragments::

    uvx towncrier build --draft --version 0.2.0
