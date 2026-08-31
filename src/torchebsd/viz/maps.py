"""Indexing result figures: pattern match, orientation map, SO(3) correlation."""

from __future__ import annotations

import numpy as np

from torchebsd.viz.figures import apl, label

__all__ = ["show_correlation", "show_map", "show_match", "show_refinement"]

_EPS = 1e-12


def _to_numpy(x) -> np.ndarray:
    detach = getattr(x, "detach", None)
    if detach is not None:
        x = detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _to_image(x) -> np.ndarray:
    image = np.squeeze(_to_numpy(x))
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D pattern, got shape {image.shape}")
    return image


def _standardize(image: np.ndarray) -> np.ndarray:
    image = image - image.mean()
    std = float(image.std())
    return image / std if std > _EPS else image


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    na = float(np.linalg.norm(a0))
    nb = float(np.linalg.norm(b0))
    if na < _EPS or nb < _EPS:
        return 0.0
    return float(np.vdot(a0, b0) / (na * nb))


def show_match(experimental, master, detector, orientation, energy_kev=None):
    """The indexing sanity check: experimental vs simulated pattern at the fit.

    Renders three panels side by side -- the experimental pattern, the pattern
    :meth:`torchebsd.sim.MasterPattern.project` produces at the indexed
    ``orientation``, and their (contrast-standardized) difference -- so a viewer
    can see at a glance whether the orientation is right. The middle panel is
    titled with the normalized cross-correlation (:func:`_ncc`) between the
    experimental and simulated patterns: a contrast-invariant number in
    ``[-1, 1]`` that quantifies the match (``~1`` is a good index).

    Parameters
    ----------
    experimental
        The measured detector image ``(H, W)`` (numpy array or torch tensor;
        a leading length-1 axis is squeezed).
    master
        The :class:`torchebsd.sim.MasterPattern` used for indexing.
    detector
        The :class:`torchebsd.sim.detector.EBSDDetector` geometry.
    orientation
        The indexed orientation as a length-1 :class:`orix.quaternion.Rotation`
        (e.g. a single element of :meth:`SphericalIndexer.index`'s output).
    energy_kev
        Energy bin to project (default: the master's highest).

    Returns
    -------
    anyplotlib.Figure
        A 1x3 figure (experimental | simulated | difference).
    """
    exp = _to_image(experimental)
    sim = _to_image(master.project(detector, orientation, energy_kev=energy_kev))
    if sim.shape != exp.shape:
        raise ValueError(
            f"simulated pattern {sim.shape} does not match experimental {exp.shape}; "
            "check the detector shape and orientation length (must be 1)"
        )
    diff = _standardize(exp) - _standardize(sim)
    score = _ncc(exp, sim)

    fig, axes = apl().subplots(1, 3, figsize=(1140, 400))
    ax_e, ax_s, ax_d = np.ravel(axes)[:3]
    label(ax_e.imshow(exp, cmap="gray", origin="lower"), "experimental")
    label(ax_s.imshow(sim, cmap="gray", origin="lower"), f"simulated  |  NCC = {score:.3f}")
    label(ax_d.imshow(diff, cmap="coolwarm", origin="lower"), "difference")
    fig.match_ncc = score
    return fig


def _ipf_rgb(crystal_map) -> np.ndarray:
    from orix.plot import IPFColorKeyTSL
    from orix.vector import Vector3d

    point_group = crystal_map.phases[crystal_map.phase_id[0]].point_group
    key = IPFColorKeyTSL(point_group, direction=Vector3d.zvector())
    rgb = key.orientation2color(crystal_map.rotations)  # (N, 3) floats in [0, 1]
    rgb = np.asarray(rgb, dtype=np.float64).reshape(*crystal_map.shape, 3)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _quality_image(crystal_map) -> np.ndarray:
    if "quality" not in crystal_map.prop:
        raise KeyError("crystal_map.prop has no 'quality'; index_scan stores it there")
    return np.asarray(crystal_map.prop["quality"], dtype=np.float64).reshape(crystal_map.shape)


def show_map(crystal_map, value: str = "ipf"):
    """Show an indexed scan (orix ``CrystalMap``) as an IPF and/or quality map.

    Parameters
    ----------
    crystal_map
        The :class:`orix.crystal_map.CrystalMap` from
        :meth:`SphericalIndexer.index_scan` (single phase, ``prop['quality']``
        holding the per-point score).
    value
        Which map(s) to draw:

        * ``"ipf"`` (default) -- the IPF-z orientation map (orix TSL coloring);
        * ``"quality"`` -- the ``prop['quality']`` field, sequential colormap;
        * ``"both"`` -- the two side by side in a 1x2 figure.

    Returns
    -------
    anyplotlib.Figure
        A single-panel figure (``"ipf"`` / ``"quality"``) or a 1x2 figure
        (``"both"``). Row 0 of the scan is drawn at the top (``origin="upper"``).
    """
    if value not in ("ipf", "quality", "both"):
        raise ValueError("value must be 'ipf', 'quality', or 'both'")

    if value == "both":
        fig, axes = apl().subplots(1, 2, figsize=(1000, 480))
        ax_ipf, ax_q = np.ravel(axes)[:2]
        label(ax_ipf.imshow(_ipf_rgb(crystal_map), origin="upper"), "IPF-z")
        panel_q = ax_q.imshow(_quality_image(crystal_map), cmap="viridis", origin="upper")
        label(panel_q, "quality")
        _colorbar_label(panel_q, "quality")
        return fig

    fig, ax = apl().subplots(1, 1, figsize=(560, 520))
    if value == "ipf":
        label(ax.imshow(_ipf_rgb(crystal_map), origin="upper"), "IPF-z orientation map")
    else:
        panel = ax.imshow(_quality_image(crystal_map), cmap="viridis", origin="upper")
        label(panel, "quality map")
        _colorbar_label(panel, "quality")
    return fig


def _colorbar_label(panel, text: str) -> None:
    setter = getattr(panel, "set_colorbar_label", None)
    if callable(setter):
        try:
            setter(text)
        except Exception:  # noqa: BLE001 - a missing colorbar must never break .show()
            pass


def show_correlation(correlation):
    """Render the SO(3) cross-correlation volume as a legible 2-D heatmap.

    The cross-correlation from
    :func:`torchebsd.index.spherical.correlation.so3_cross_correlation` is a function on
    all of SO(3), sampled on the ``(alpha, beta, gamma)`` grid (shape
    ``(N, N, N)``). This reduces it to an ``(alpha, gamma)`` heatmap by taking
    the maximum over the ``beta`` axis (so the global peak survives the
    projection), marks the peak cell, and titles the panel with the peak value
    and its ``(alpha, beta, gamma)`` grid indices.

    Parameters
    ----------
    correlation
        The correlation grid, shape ``(N, N, N)`` for a single pattern, or a
        stack ``(Np, N, N, N)`` whose **first** pattern is shown (numpy array or
        torch tensor). Obtain it from
        :meth:`~torchebsd.index.spherical.indexer.SphericalIndexer.index` with
        ``return_surface=True`` (the third returned element).

    Returns
    -------
    anyplotlib.Figure
        A single heatmap panel with the peak marked. The peak grid indices and
        value are also attached to the figure as ``fig.peak_index`` (an
        ``(alpha, beta, gamma)`` tuple) and ``fig.peak_value`` for programmatic
        access.
    """
    corr = _to_numpy(correlation)
    if corr.ndim == 4:
        corr = corr[0]
    if corr.ndim != 3:
        raise ValueError(
            f"correlation must be (N, N, N) or (Np, N, N, N), got shape {corr.shape}"
        )

    a_idx, b_idx, g_idx = (int(i) for i in np.unravel_index(int(np.argmax(corr)), corr.shape))
    peak = float(corr[a_idx, b_idx, g_idx])
    projection = corr.max(axis=1)  # (alpha, gamma), max over beta -- keeps the global peak

    fig, ax = apl().subplots(1, 1, figsize=(620, 560))
    panel = ax.imshow(projection, cmap="magma", origin="lower")
    label(
        panel,
        f"SO(3) correlation (max over beta)  |  peak {peak:.4g} "
        f"at (a, b, g) = ({a_idx}, {b_idx}, {g_idx})",
    )
    _ax_setter(panel, "set_xlabel", "gamma index")
    _ax_setter(panel, "set_ylabel", "alpha index")
    # mark the peak: heatmap x is the gamma index, y is the alpha index
    try:
        panel.add_points([(g_idx, a_idx)], color="#4fc3f7", sizes=14, alpha=0.9)
    except Exception:  # noqa: BLE001 - the title already reports the peak location
        pass

    fig.peak_index = (a_idx, b_idx, g_idx)
    fig.peak_value = peak
    return fig


def _ax_setter(panel, method: str, value: str) -> None:
    setter = getattr(panel, method, None)
    if callable(setter):
        try:
            setter(value)
        except Exception:  # noqa: BLE001 - a missing axis label must never break .show()
            pass


def show_refinement(history):
    """Plot the refinement NCC trajectory (convergence) vs iteration.

    Parameters
    ----------
    history
        The per-iteration NCC trajectory from
        :func:`torchebsd.index.refine.orientation.refine_orientation` (``return_history=True``),
        shape ``(n_iters, Np)`` -- one column per pattern -- or ``(n_iters,)`` for a
        single pattern (numpy array or torch tensor). ``history[0]`` is the seed's
        score at ``w = 0``.

    Returns
    -------
    anyplotlib.Figure
        A line plot of NCC against iteration: one thin curve per pattern, plus a
        thick mean curve when more than one pattern is refined.
    """
    hist = _to_numpy(history)
    if hist.ndim == 1:
        hist = hist[:, None]
    if hist.ndim != 2:
        raise ValueError(f"history must be (n_iters, Np) or (n_iters,), got shape {hist.shape}")
    n_iters, n_p = hist.shape
    iters = np.arange(n_iters, dtype=np.float64)

    fig, ax = apl().subplots(1, 1, figsize=(680, 420))
    per_pattern_label = "per pattern" if n_p > 1 else "NCC"
    plot = ax.plot(
        hist[:, 0],
        axes=[iters],
        units="iteration",
        y_units="NCC",
        color="#4fc3f7",
        linewidth=1.0 if n_p > 1 else 2.0,
        alpha=0.5 if n_p > 1 else 1.0,
        label=per_pattern_label,
    )
    for p in range(1, n_p):
        plot.add_line(hist[:, p], x_axis=iters, color="#4fc3f7", linewidth=1.0, alpha=0.5)
    if n_p > 1:
        plot.add_line(
            hist.mean(axis=1), x_axis=iters, color="#ff7043", linewidth=2.5, label="mean"
        )
    _ax_setter(plot, "set_title", "Refinement convergence")
    _ax_setter(plot, "set_xlabel", "iteration")
    _ax_setter(plot, "set_ylabel", "NCC")
    return fig
