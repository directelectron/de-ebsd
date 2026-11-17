"""How a detector sees the master pattern: fitted on a scan's own patterns.

A simulated master is a model of the electrons a pattern is made of. A real
detector weighs their energies by its own response, blurs them by its point
spread, maps their intensity through its contrast, and its patterns keep a smooth
background that background correction did not take away. Each mismatch lowers the
correct orientation's correlation and pulls a strain fit off. This module fits
that appearance from patterns whose orientations are already known (indexed), and
builds the master an indexer then matches against:

.. math::

    S(g) = c\\big(\\mathrm{PSF} * \\textstyle\\sum_e w_e\\, M_e\\big)(g\\,d) + B(d)

with ``M_e`` the master of one electron at energy ``e``
(:meth:`~de_ebsd.sim.MasterPattern.per_electron`), ``w_e`` the energy weights,
the PSF a Gaussian (or a core plus a halo) on the sphere, ``c`` a gamma on the
master's intensity range, and ``B`` a low-order Legendre background on the
detector. The weights, the PSF and the contrast act on the sphere, so they are
baked into the master (:meth:`DetectorAppearance.master`) at no cost per pattern;
the background is fixed on the detector, so the indexer removes it from the real
patterns instead (:class:`~de_ebsd.index.live.background.ResidualBackground`).

:func:`fit_appearance` fits the shared parameters together with a small rotation of
every pattern's orientation, maximizing the mean normalized cross-correlation
(NCC), with torch autograd and L-BFGS.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn.functional as F

from de_ebsd.core.constants import SINGLE_PRECISION_FLOOR
from de_ebsd.core.geometry import sample_to_detector_matrix
from de_ebsd.core.lambert import sample_texture
from de_ebsd.core.rotation import quaternion_to_matrix, rotation_vector_to_matrix
from de_ebsd.index.live.background import ResidualBackground, ScanBackground, legendre_terms
from de_ebsd.index.live.common import average_to
from de_ebsd.index.preprocess import normalize_rows

__all__ = [
    "AppearanceFit",
    "DetectorAppearance",
    "fit_appearance",
    "load_appearance",
    "select_patterns",
]

#: The largest core PSF the fit reaches, Gaussian sigma in degrees on the sphere.
MAX_PSF_DEG = 4.0
#: The largest halo the fit reaches, sigma in degrees, and its largest share.
MAX_HALO_DEG = 10.0
MAX_HALO_FRACTION = 0.5
#: The range of the high pass's sigma, degrees on the sphere.
HIGH_PASS_DEG = (2.0, 30.0)
#: The largest share of the high pass: the indexer takes the whole of its blur from a
#: real pattern, so the fit is free to reach one and a little past it.
MAX_HIGH_PASS = 1.5
#: The gamma's range: ``exp(log(MAX_GAMMA) * tanh(raw))``.
MAX_GAMMA = 5.0
#: Largest pattern-center shift (in the Bruker units) and radial distortion.
MAX_CENTER_SHIFT = 0.05
MAX_DISTORTION = 0.2
#: A tiny ridge on the energy curve and the background, so that a direction the
#: patterns cannot see stays at its default rather than wandering.
RIDGE = 1e-4
#: Correlations between fitted parameters above this are flagged.
CORRELATED = 0.9
#: A fitted parameter is flagged as unconstrained where a unit of its raw value moves
#: the mean NCC by less than this.
FLAT_NCC = 1e-5
#: The orientation unknowns are rotation vectors in units of this many radians,
#: about half a degree, so they are scaled like the shared parameters for L-BFGS.
ROTATION_UNIT = 0.01


# --- the appearance ----------------------------------------------------------------


@dataclass
class DetectorAppearance:
    """How a detector sees a master pattern: the parameters of the appearance model.

    Attributes
    ----------
    energies_kev
        The resolved master's energy bins ``(E,)``.
    prior_weights
        The weights the resolved master's maps carry ``(E,)`` (the Monte Carlo's
        exit shares), normalized to sum to one.
    energy_curve
        Coefficients ``a_k`` of Legendre polynomials ``P_1 .. P_K`` over the energy
        range: ``w_e = prior_e * exp(sum_k a_k P_k(e))``, normalized to sum to one.
        Empty, or zeros, for the Monte Carlo's weights. A low-order curve times the
        prior, so a few bins cannot be fitted one by one.
    psf_sigma_deg
        The point spread's Gaussian sigma, degrees on the sphere (the Lambert
        grid's ``npx / 90`` pixels a degree, as the indexers blur).
    halo_fraction, halo_sigma_deg
        A second, wide Gaussian's share of the point spread and its sigma.
    gamma
        The contrast: the master's intensity, mapped onto ``[0, 1]`` by
        *contrast_range*, raised to this power.
    high_pass_fraction, high_pass_sigma_deg
        The share of a wide Gaussian blur of the master (sigma in degrees) taken
        from it after the contrast: the indexer removes each real pattern's dynamic
        background (a wide blur of it) and not the simulation's, and this puts the
        same high pass on the simulation, on the sphere.
    contrast_range
        ``(low, high)`` of the default master (the Monte Carlo's weights,
        unblurred), which the gamma is taken over.
    background
        Coefficients of the residual background over
        :func:`~de_ebsd.index.live.background.legendre_terms` of *background_order*,
        in the units of the contrast's ``[0, 1]``.
    background_share
        ``|B| / |S|`` over the fit's patterns, the share of a normalized pattern the
        background is; see :class:`~de_ebsd.index.live.background.ResidualBackground`.
    background_high_pass
        Whether the patterns the background was fitted on had their dynamic
        background removed, so the background is taken through the same.
    pattern_center_shift
        A refined pattern center less the detector's, Bruker units.
    distortion
        Radial distortion ``k``: a pixel at ``r`` from the detector's center (in
        detector heights) is seen at ``r (1 + k r^2)``.
    """

    energies_kev: np.ndarray
    prior_weights: np.ndarray
    energy_curve: np.ndarray = field(default_factory=lambda: np.zeros(0))
    psf_sigma_deg: float = 0.0
    halo_fraction: float = 0.0
    halo_sigma_deg: float = 0.0
    gamma: float = 1.0
    high_pass_fraction: float = 0.0
    high_pass_sigma_deg: float = 0.0
    contrast_range: tuple[float, float] = (0.0, 1.0)
    background: np.ndarray = field(default_factory=lambda: np.zeros(0))
    background_order: int = 0
    background_share: float = 0.0
    background_high_pass: bool = False
    pattern_center_shift: np.ndarray = field(default_factory=lambda: np.zeros(3))
    distortion: float = 0.0

    def __post_init__(self) -> None:
        self.energies_kev = np.atleast_1d(np.asarray(self.energies_kev, dtype=np.float64))
        prior = np.atleast_1d(np.asarray(self.prior_weights, dtype=np.float64))
        if prior.shape != self.energies_kev.shape:
            raise ValueError("one prior weight per energy bin")
        self.prior_weights = prior / max(prior.sum(), 1e-300)
        self.energy_curve = np.atleast_1d(np.asarray(self.energy_curve, dtype=np.float64))
        self.background = np.atleast_1d(np.asarray(self.background, dtype=np.float64))
        self.pattern_center_shift = np.asarray(self.pattern_center_shift, dtype=np.float64)
        self.contrast_range = tuple(float(v) for v in self.contrast_range)

    @classmethod
    def default(cls, resolved) -> DetectorAppearance:
        """The appearance that reproduces today's master: the Monte Carlo's weights,
        no blur, linear contrast, no background. *resolved* is an energy-resolved
        :class:`~de_ebsd.sim.MasterPattern` (one bin works too)."""
        prior = _prior_weights(resolved)
        layers = resolved.per_electron()
        weights = torch.as_tensor(prior, device=layers.device, dtype=layers.dtype)
        collapsed = torch.einsum("e,ehij->hij", weights, layers)
        low, high = float(collapsed.min()), float(collapsed.max())
        return cls(resolved.energies_kev, prior, contrast_range=(low, high))

    @property
    def energy_weights(self) -> np.ndarray:
        """The fitted weight of each energy bin ``(E,)``, summing to one."""
        return _energy_weights_np(self.prior_weights, self.energy_curve, self.energies_kev)

    def master(self, resolved):
        """The single-layer :class:`~de_ebsd.sim.MasterPattern` an indexer matches
        against, from the energy-resolved master *resolved* the fit was made on: its
        energies collapsed with :attr:`energy_weights`, blurred by the point spread
        and mapped through the contrast, all on the sphere. Give it to
        :class:`~de_ebsd.index.live.LiveIndexer` as any master, with
        :meth:`residual_background` for the background."""
        from de_ebsd.sim.masterpattern import MasterPattern

        if len(resolved.energies_kev) != len(self.energies_kev) or not np.allclose(
            resolved.energies_kev, self.energies_kev
        ):
            raise ValueError("the master's energy bins are not the ones the fit was made on")
        layers = resolved.per_electron()
        with torch.no_grad():
            texture = _texture(
                layers,
                torch.as_tensor(self.energy_weights, device=layers.device, dtype=layers.dtype),
                resolved.npx,
                self.psf_sigma_deg,
                self.halo_fraction,
                self.halo_sigma_deg,
                self.gamma,
                self.contrast_range,
                self.high_pass_fraction,
                self.high_pass_sigma_deg,
            )
        metadata = dict(
            resolved.metadata,
            energy_resolved=False,
            energy_range_kev=(float(self.energies_kev.min()), float(self.energies_kev.max())),
            appearance=json.dumps(self.to_dict()),
        )
        return MasterPattern(
            texture[None].contiguous(),
            np.array([float(self.energies_kev.max())]),
            metadata,
        )

    def residual_background(self) -> ResidualBackground | None:
        """The background the indexer removes from the real patterns, or None where
        the fit has none (:class:`~de_ebsd.index.live.background.ResidualBackground`)."""
        if not self.background.size or not np.any(self.background):
            return None
        return ResidualBackground(
            tuple(float(v) for v in self.background),
            int(self.background_order),
            float(self.background_share),
            bool(self.background_high_pass),
        )

    def detector(self, detector):
        """*detector* at the refined pattern center (the shift added)."""
        center = np.asarray(detector.pattern_center, dtype=np.float64) + self.pattern_center_shift
        return replace(detector, pattern_center=tuple(float(v) for v in center))

    # --- persistence -----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "energies_kev": self.energies_kev.tolist(),
            "prior_weights": self.prior_weights.tolist(),
            "energy_weights": self.energy_weights.tolist(),
            "energy_curve": self.energy_curve.tolist(),
            "psf_sigma_deg": float(self.psf_sigma_deg),
            "halo_fraction": float(self.halo_fraction),
            "halo_sigma_deg": float(self.halo_sigma_deg),
            "gamma": float(self.gamma),
            "high_pass_fraction": float(self.high_pass_fraction),
            "high_pass_sigma_deg": float(self.high_pass_sigma_deg),
            "contrast_range": list(self.contrast_range),
            "background": self.background.tolist(),
            "background_order": int(self.background_order),
            "background_share": float(self.background_share),
            "background_high_pass": bool(self.background_high_pass),
            "pattern_center_shift": self.pattern_center_shift.tolist(),
            "distortion": float(self.distortion),
        }

    @classmethod
    def from_dict(cls, values: dict) -> DetectorAppearance:
        values = {k: v for k, v in values.items() if k != "energy_weights"}
        return cls(**values)

    def save(self, path) -> None:
        """Write the parameters as JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"appearance": self.to_dict()}, f, indent=2)

    @classmethod
    def load(cls, path) -> DetectorAppearance:
        """Read parameters written by :meth:`save` or :meth:`AppearanceFit.save`."""
        return load_appearance(path)


def load_appearance(path) -> DetectorAppearance:
    """The :class:`DetectorAppearance` in a JSON file written by
    :meth:`DetectorAppearance.save` or :meth:`AppearanceFit.save`."""
    with open(path, encoding="utf-8") as f:
        values = json.load(f)
    return DetectorAppearance.from_dict(values["appearance"])


# --- the result --------------------------------------------------------------------


@dataclass
class AppearanceFit:
    """What :func:`fit_appearance` found.

    Attributes
    ----------
    appearance
        The fitted :class:`DetectorAppearance`.
    default
        The appearance the fit started from (the Monte Carlo's), for comparison.
    train, held_out
        The indices, into the patterns given, fitted on and held out.
    ncc_before, ncc_after
        Each pattern's NCC ``(N,)`` as the indexer scores it, against the default
        master and against the fitted one (:meth:`DetectorAppearance.master`, the
        residual background removed from the pattern), every orientation refined
        under each (the held-out ones with the shared parameters fixed), so neither
        side is charged for orientations fitted under the other.
    ncc_model
        Each pattern's NCC ``(N,)`` under the fitted model itself: the simulation
        with the background added and, for patterns whose dynamic background was
        removed, the same removed from it.
    orientations
        The fitted orientations ``(N, 3, 3)``, under the fitted appearance.
    margin_before, margin_after
        Each held-out pattern's rough-search margin: the best dictionary score less
        the best more than 5 degrees from it, under each appearance; None where no
        point group was given.
    log
        The convergence log: ``(evaluation, seconds, mean training NCC)``.
    restarts
        One entry per random restart: its fitted appearance's parameters and mean
        training and held-out NCC.
    correlations
        The fitted parameters' correlation matrix, the orientations profiled out
        (from the curvature of the mean NCC), and *parameter_names* its order.
    warnings
        Parameters at a bound, strongly correlated, or not constrained by the data.
    seconds
        The fit's wall time.
    """

    appearance: DetectorAppearance
    default: DetectorAppearance
    train: np.ndarray
    held_out: np.ndarray
    ncc_before: np.ndarray
    ncc_after: np.ndarray
    orientations: np.ndarray
    ncc_model: np.ndarray | None = None
    margin_before: np.ndarray | None = None
    margin_after: np.ndarray | None = None
    log: list = field(default_factory=list)
    restarts: list = field(default_factory=list)
    correlations: np.ndarray | None = None
    parameter_names: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> dict:
        """Mean NCC before and after, on the patterns fitted and held out, and the
        held-out margins' medians."""

        def mean(values, rows):
            return float(np.mean(values[rows])) if len(rows) else float("nan")

        out = {
            "train_before": mean(self.ncc_before, self.train),
            "train_after": mean(self.ncc_after, self.train),
            "held_out_before": mean(self.ncc_before, self.held_out),
            "held_out_after": mean(self.ncc_after, self.held_out),
        }
        if self.ncc_model is not None:
            out["train_model"] = mean(self.ncc_model, self.train)
            out["held_out_model"] = mean(self.ncc_model, self.held_out)
        if self.margin_before is not None:
            out["margin_before"] = float(np.median(self.margin_before))
            out["margin_after"] = float(np.median(self.margin_after))
        return out

    def report(self) -> str:
        """The fit as a few lines of text."""
        s = self.summary()
        a = self.appearance
        lines = [
            f"mean NCC, fitted on {len(self.train)}: {s['train_before']:.4f} -> "
            f"{s['train_after']:.4f}",
            f"mean NCC, held out {len(self.held_out)}: {s['held_out_before']:.4f} -> "
            f"{s['held_out_after']:.4f}",
        ]
        if "train_model" in s:
            lines.append(
                f"mean NCC of the model itself: {s['train_model']:.4f} fitted on, "
                f"{s['held_out_model']:.4f} held out"
            )
        if "margin_before" in s:
            lines.append(
                f"median search margin, held out: {s['margin_before']:.4f} -> "
                f"{s['margin_after']:.4f}"
            )
        lines += [
            "energy weights (keV: Monte Carlo -> fitted): "
            + ", ".join(
                f"{e:g}: {p:.3f}->{w:.3f}"
                for e, p, w in zip(a.energies_kev, a.prior_weights, a.energy_weights, strict=True)
            ),
            f"PSF sigma {a.psf_sigma_deg:.3f} deg, halo {a.halo_fraction:.3f} x "
            f"{a.halo_sigma_deg:.2f} deg, gamma {a.gamma:.3f}, high pass "
            f"{a.high_pass_fraction:.3f} x {a.high_pass_sigma_deg:.1f} deg, background share "
            f"{a.background_share:.3f}",
        ]
        if np.any(a.pattern_center_shift) or a.distortion:
            lines.append(
                f"pattern center shift {np.round(a.pattern_center_shift, 4).tolist()}, "
                f"distortion {a.distortion:.4f}"
            )
        lines.append(f"{self.seconds:.1f} s")
        lines += [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)

    def save(self, path) -> None:
        """Write the fitted appearance and the fit's numbers as JSON; read the
        appearance back with :func:`load_appearance`."""

        def listed(values):
            return None if values is None else np.asarray(values).tolist()

        record = {
            "appearance": self.appearance.to_dict(),
            "default": self.default.to_dict(),
            "summary": self.summary(),
            "train": listed(self.train),
            "held_out": listed(self.held_out),
            "ncc_before": listed(self.ncc_before),
            "ncc_after": listed(self.ncc_after),
            "ncc_model": listed(self.ncc_model),
            "margin_before": listed(self.margin_before),
            "margin_after": listed(self.margin_after),
            "log": [list(entry) for entry in self.log],
            "restarts": self.restarts,
            "parameter_names": self.parameter_names,
            "correlations": listed(self.correlations),
            "warnings": self.warnings,
            "seconds": self.seconds,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1)


# --- choosing the patterns -----------------------------------------------------------


def select_patterns(
    orientations,
    scores,
    point_group,
    *,
    count: int = 300,
    per_grain: int = 8,
    minimum: int = 200,
    positions=None,
    grain_deg: float = 5.0,
    min_score: float | None = None,
    held_out: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Which indexed patterns to fit on and which to hold out: ``(train, held_out)``,
    indices into *orientations*.

    The well-scored patterns, spread over many orientations: those scored at least
    *min_score* are grouped into grains (orientations within *grain_deg* of a grain's
    best-scored pattern, symmetry reduced), and at most *per_grain* are taken from
    each, *count* in all. A scan of few grains would give few patterns that way, so
    the cap is raised, evenly over the grains, until *minimum* patterns are taken or
    the grains have none left. Within a grain the patterns are the best-scored, or,
    given *positions*, spread over the grain's area: the best-scored first, then
    each the farthest from those already taken.

    Whole grains are held out (a share *held_out* of them), so the held-out patterns
    test the fit on orientations it has not seen; with fewer than five grains the
    patterns are split instead.

    Parameters
    ----------
    orientations
        ``(N, 3, 3)`` matrices (sample to crystal, as the indexers give them).
    scores
        ``(N,)`` the indexer's scores.
    point_group
        The phase's orix point group, or its operators ``(K, 3, 3)``.
    positions
        ``(N, 2)`` each pattern's place in the scan (any units), to spread the
        patterns taken from one grain over its area.
    min_score
        Patterns scored below this are left out; default the median score.
    """
    g = np.asarray(_matrices(orientations), dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if min_score is None:
        min_score = float(np.median(scores))
    operators = point_group.to_matrix() if hasattr(point_group, "to_matrix") else point_group
    operators = np.asarray(operators, dtype=np.float64)
    threshold = math.cos(math.radians(grain_deg))

    # every candidate into a grain, in order of score
    seeds, members = [], []
    for i in np.argsort(-scores, kind="stable"):
        if scores[i] < min_score:
            break
        grain = -1
        if seeds:
            turn = np.asarray(seeds) @ g[i].T  # (S, 3, 3)
            traces = np.einsum("kij,sij->sk", operators, turn).max(-1)
            close = np.flatnonzero((traces - 1.0) / 2.0 >= threshold)
            if len(close):
                grain = int(close[0])
        if grain < 0:
            seeds.append(g[i])
            members.append([])
            grain = len(seeds) - 1
        members[grain].append(int(i))
    if not members:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)

    # the cap, raised evenly while too few patterns would be taken
    sizes = np.array([len(m) for m in members])
    total = max(int(count), int(minimum))
    cap = int(per_grain)
    while np.minimum(sizes, cap).sum() < min(minimum, sizes.sum()):
        cap += 1
    picks = [_spread(m, cap, positions) for m in members]
    # round-robin over the grains, the first of each first, up to the total
    chosen, grain_of = [], []
    for rank in range(cap):
        for grain, pick in enumerate(picks):
            if rank < len(pick) and len(chosen) < total:
                chosen.append(pick[rank])
                grain_of.append(grain)
    chosen, grain_of = np.asarray(chosen, dtype=np.int64), np.asarray(grain_of)
    rng = np.random.default_rng(seed)
    grains = np.unique(grain_of)
    if len(grains) >= 5:
        held = rng.choice(grains, max(1, round(held_out * len(grains))), replace=False)
        mask = np.isin(grain_of, held)
    else:
        mask = np.zeros(len(chosen), dtype=bool)
        mask[rng.choice(len(chosen), round(held_out * len(chosen)), replace=False)] = True
    return chosen[~mask], chosen[mask]


def _spread(members, cap, positions) -> list:
    """Up to *cap* of a grain's *members* (indices, best-scored first): the first
    ones, or, given *positions*, the best-scored then each the farthest from those
    taken."""
    if positions is None or len(members) <= cap:
        return members[:cap]
    where = np.asarray(positions, dtype=np.float64)[members]
    taken = [0]
    distance = np.linalg.norm(where - where[0], axis=1)
    while len(taken) < cap:
        far = int(np.argmax(distance))
        taken.append(far)
        distance = np.minimum(distance, np.linalg.norm(where - where[far], axis=1))
    return [members[k] for k in taken]


# --- the fit -----------------------------------------------------------------------------


def fit_appearance(
    resolved,
    patterns,
    orientations,
    detector,
    *,
    train=None,
    held_out=None,
    point_group=None,
    fit_energy: bool = True,
    energy_order: int = 2,
    fit_psf: bool = True,
    halo: bool = False,
    fit_contrast: bool = True,
    fit_high_pass: bool = True,
    dynamic_background: bool = True,
    background_order: int = 3,
    fit_pattern_center: bool = False,
    fit_distortion: bool = False,
    iterations: int = 150,
    restarts: int = 0,
    margin_resolution_deg: float | None = None,
    search_shape=(32, 32),
    device=None,
    seed: int = 0,
    progress=None,
) -> AppearanceFit:
    """Fit how a detector sees *resolved* from patterns whose orientations are known.

    Parameters
    ----------
    resolved
        The phase's energy-resolved :class:`~de_ebsd.sim.MasterPattern` (with its
        :attr:`~de_ebsd.sim.MasterPattern.energy_weights`); one energy bin works, the
        energy weights then left out.
    patterns
        ``(N, rows, cols)`` real patterns, background corrected as the indexer
        corrects them (``LiveIndexer.prepare``'s refine rows, reshaped), at the
        resolution the indexer refines at; or ``(N, rows * cols)`` rows.
    orientations
        ``(N, 3, 3)`` matrices the patterns were indexed at (sample to crystal), or an
        orix ``Rotation``.
    detector
        The :class:`~de_ebsd.sim.EBSDDetector` the patterns were taken with; its shape
        is replaced by the patterns'.
    train, held_out
        Indices of the patterns to fit on and to hold out; default all of them, and
        none (:func:`select_patterns` chooses both).
    point_group
        The phase's orix point group: given, the held-out rough-search margin is
        measured too (with *margin_resolution_deg*, default 3).
    fit_energy, energy_order
        Fit the energy weights, as a curve of this many Legendre terms times the
        Monte Carlo's (2: a tilt and a bend).
    fit_psf, halo
        Fit the point spread; with *halo*, a wide second Gaussian too.
    fit_contrast
        Fit the gamma.
    dynamic_background
        Whether *patterns* had their dynamic background removed, as
        ``LiveIndexer.prepare`` removes it (a wide blur of each subtracted). The
        model then removes it from its simulations the same way, so the fitted
        parameters are the detector's rather than ones bent to make up for it.
    fit_high_pass
        With *dynamic_background*: after the model, fit the high pass on the sphere
        (:attr:`DetectorAppearance.high_pass_fraction`) that best stands in, in the
        indexer's master, for the dynamic background removal the indexer does to
        real patterns and not to its simulations.
    background_order
        The residual background's Legendre order (0 for none).
    fit_pattern_center, fit_distortion
        Refine the pattern center (one shift for all patterns) and a radial
        distortion. Off by default: a small center shift is nearly a rotation of
        every pattern, which the orientations take up, and the center's distance
        trades against the distortion; the fit flags both when they correlate.
    iterations
        L-BFGS iterations of the joint fit.
    restarts
        Further fits from random starting points; the best on the training patterns
        is kept, and each is reported (:attr:`AppearanceFit.restarts`).
    device
        Default: the master's.
    progress
        Called as ``progress(stage, evaluations, mean_ncc)``.

    Returns
    -------
    AppearanceFit
    """
    start = time.perf_counter()
    device = torch.device(device) if device is not None else resolved.device
    rows = torch.as_tensor(np.asarray(patterns) if not torch.is_tensor(patterns) else patterns)
    rows = rows.to(device, torch.float32)
    if rows.ndim == 3:
        shape = tuple(rows.shape[-2:])
        rows = rows.flatten(1)
    else:
        shape = tuple(detector.shape)
    if rows.shape[1] != shape[0] * shape[1]:
        raise ValueError("the patterns are not at the detector's shape; give them (N, rows, cols)")
    rows = normalize_rows(rows)
    g = torch.as_tensor(_matrices(orientations), device=device, dtype=torch.float32)
    n = len(rows)
    train = np.arange(n) if train is None else np.asarray(train, dtype=np.int64)
    held_out = np.zeros(0, np.int64) if held_out is None else np.asarray(held_out, np.int64)

    model = _Model(
        resolved,
        replace(detector, shape=shape),
        device,
        energy_order=energy_order if fit_energy else 0,
        halo=halo,
        background_order=background_order,
        dynamic_background=dynamic_background,
    )
    flags = {
        "energy": fit_energy and model.energy_terms > 0,
        "psf": fit_psf,
        "halo": fit_psf and halo,
        "gamma": fit_contrast,
        "high_pass": False,
        "background": background_order > 0,
        "center": fit_pattern_center,
        "distortion": fit_distortion,
    }
    nothing = dict.fromkeys(flags, False)
    log = []

    def report(stage):
        def call(evaluations, ncc):
            log.append((len(log), time.perf_counter() - start, float(ncc)))
            if progress is not None:
                progress(stage, evaluations, ncc)

        return call

    train_t = torch.as_tensor(train, device=device)
    held_t = torch.as_tensor(held_out, device=device)

    # the default master as the indexer scores it, every orientation refined under it
    default_params = model.initial(nothing, jitter=None)
    omega_before = model.refine_orientations(default_params, rows, g, 40, indexer=True)
    ncc_before = model.ncc(default_params, rows, g, omega_before, indexer=True)

    # the model: the shared parameters and every orientation, from the default, and
    # from random starting points. Each start's orientations are first refined under
    # it alone: started from orientations a different model preferred, the joint fit
    # can settle on a sharper master that makes up for them.
    fits = []
    generator = torch.Generator().manual_seed(seed)
    for attempt in range(restarts + 1):
        params = model.initial(flags, jitter=None if attempt == 0 else generator)
        omega = model.refine_orientations(params, rows[train_t], g[train_t], 40)
        params, omega = model.fit(
            params,
            flags,
            rows[train_t],
            g[train_t],
            omega,
            iterations=iterations,
            callback=report(f"fit {attempt}"),
        )
        train_ncc = float(model.ncc(params, rows[train_t], g[train_t], omega).mean())
        held_ncc = float("nan")
        if len(held_out):
            omega_held = model.refine_orientations(params, rows[held_t], g[held_t], 40)
            held_ncc = float(model.ncc(params, rows[held_t], g[held_t], omega_held).mean())
        fits.append((train_ncc, held_ncc, params, omega))
    best = max(range(len(fits)), key=lambda i: fits[i][0])
    _, _, params, omega_train = fits[best]
    omega_model = omega_before.clone()
    omega_model[train_t] = omega_train
    if len(held_out):
        omega_model[held_t] = model.refine_orientations(params, rows[held_t], g[held_t], 40)
    ncc_model = model.ncc(params, rows, g, omega_model)

    appearance = model.appearance(params, flags, rows[train_t], g[train_t], omega_train)
    names, correlations, flat = model.correlations(
        params, flags, rows[train_t], g[train_t], omega_train
    )
    warnings = model.warnings(appearance, flags, names, correlations, flat)

    # the indexer's master: the real patterns less the residual background, matched
    # against the master with, where the indexer high-passes the real patterns only,
    # the high pass on the sphere that best stands in for it
    corrected = _without_residual(rows, appearance.residual_background(), shape)
    if dynamic_background and fit_high_pass:
        stage = {**nothing, "high_pass": True}
        params = {**params, **_only(model.initial(stage, jitter=None), "high_pass")}
        params, _ = model.fit(
            params,
            stage,
            corrected[train_t],
            g[train_t],
            omega_train,
            iterations=max(20, iterations // 3),
            callback=report("high pass"),
            indexer=True,
        )
        high = model.appearance(params, stage, None, None, None)
        appearance = replace(
            appearance,
            high_pass_fraction=high.high_pass_fraction,
            high_pass_sigma_deg=high.high_pass_sigma_deg,
        )
        warnings += model.warnings(appearance, stage, [], None, [])
    omega_after = model.refine_orientations(params, corrected, g, 40, indexer=True)
    ncc_after = model.ncc(params, corrected, g, omega_after, indexer=True)
    default = model.appearance(default_params, nothing, None, None, None)

    margin_before = margin_after = None
    if point_group is not None and len(held_out):
        resolution = 3.0 if margin_resolution_deg is None else margin_resolution_deg
        margin_before = search_margin(
            default.master(resolved.to(device)),
            None,
            rows[held_t],
            shape,
            detector,
            point_group,
            resolution,
            search_shape,
        )
        margin_after = search_margin(
            appearance.master(resolved.to(device)),
            appearance.residual_background(),
            rows[held_t],
            shape,
            appearance.detector(detector),
            point_group,
            resolution,
            search_shape,
        )

    g_after = (g @ rotation_vector_to_matrix(omega_after * ROTATION_UNIT)).detach()
    return AppearanceFit(
        appearance=appearance,
        default=default,
        train=train,
        held_out=held_out,
        ncc_before=ncc_before.cpu().numpy(),
        ncc_after=ncc_after.cpu().numpy(),
        orientations=g_after.cpu().numpy(),
        ncc_model=ncc_model.cpu().numpy(),
        margin_before=margin_before,
        margin_after=margin_after,
        log=log,
        restarts=[
            {
                "train_ncc": train_ncc,
                "held_out_ncc": held_ncc,
                "appearance": model.appearance(p, flags, None, None, None).to_dict(),
            }
            for train_ncc, held_ncc, p, _ in fits
        ],
        correlations=correlations,
        parameter_names=names,
        warnings=warnings,
        seconds=time.perf_counter() - start,
    )


def search_margin(
    master,
    residual,
    refine_rows,
    refine_shape,
    detector,
    point_group,
    resolution_deg,
    search_shape=(32, 32),
    separation_deg: float = 5.0,
) -> np.ndarray:
    """Each pattern's rough-search margin against *master*: its best dictionary score
    less its best more than *separation_deg* from that entry (symmetry reduced), a
    measure of how clearly the search picks the right basin. *refine_rows*
    ``(N, pixels)`` are corrected and normalized at *refine_shape*; *residual* (a
    :class:`~de_ebsd.index.live.background.ResidualBackground` or None) is removed
    from them first, as the indexer would."""
    from de_ebsd.index.live import LiveIndexer

    indexer = LiveIndexer(
        [master],
        [point_group],
        replace(detector, shape=tuple(refine_shape)),
        resolution=resolution_deg,
        search_shape=search_shape,
        refine_shape=tuple(refine_shape),
        refine_steps_deg=(),
        search_blur_deg=resolution_deg / 2.0,
        device=master.device,
        dtype=torch.float32,
        residual_background=residual,
    )
    rows = normalize_rows(refine_rows.to(indexer.device, torch.float32))
    if indexer._residual_rows is not None:
        search, _ = indexer._remove_residual(rows)
    else:
        grid = rows.view(len(rows), *indexer.refine_shape)
        search = normalize_rows(average_to(grid, indexer.search_shape).flatten(1))
    operators = point_group.to_matrix() if hasattr(point_group, "to_matrix") else point_group
    operators = torch.as_tensor(np.asarray(operators), device=indexer.device, dtype=torch.float32)
    cosine = math.cos(math.radians(separation_deg))
    margins = []
    for row in search:
        scores = row @ indexer.dictionary
        best = int(scores.argmax())
        turn = indexer.rotations @ indexer.rotations[best].T  # (E, 3, 3)
        traces = torch.einsum("kij,eij->ek", operators, turn).amax(-1)
        far = (traces - 1.0) / 2.0 < cosine
        margins.append(float(scores[best] - scores[far].max()))
    return np.asarray(margins)


# --- the model ---------------------------------------------------------------------------


class _Model:
    """The appearance model's forward and its fits, on one master and detector."""

    def __init__(
        self,
        resolved,
        detector,
        device,
        *,
        energy_order,
        halo,
        background_order,
        dynamic_background=False,
    ):
        self.detector = detector
        self.device = device
        self.npx = resolved.npx
        self.layers = resolved.per_electron().to(device, torch.float32)
        self.energies = np.asarray(resolved.energies_kev, dtype=np.float64)
        self.prior_np = _prior_weights(resolved)
        self.prior = torch.as_tensor(self.prior_np, device=device, dtype=torch.float32)
        self.energy_terms = min(int(energy_order), max(len(self.energies) - 2, 0))
        self.energy_basis = torch.as_tensor(
            _energy_basis(self.energies, self.energy_terms), device=device, dtype=torch.float32
        )
        default = torch.einsum("e,ehij->hij", self.prior, self.layers)
        self.contrast_range = (float(default.min()), float(default.max()))
        self.halo = halo
        self.background_order = int(background_order)
        self.background_terms = legendre_terms(detector.shape, background_order, device)
        self.pixels = _pixel_positions(detector.shape, device)
        self.detector_to_sample = torch.as_tensor(
            sample_to_detector_matrix(
                detector.sample_tilt, detector.tilt, detector.azimuthal, detector.twist
            ),
            device=device,
            dtype=torch.float32,
        )
        self.center = torch.as_tensor(detector.pattern_center, device=device, dtype=torch.float32)
        self.aspect = detector.ncols / detector.nrows
        #: the indexer's dynamic background removal, put on the model's simulations
        #: where the real patterns had it
        self.dynamic = ScanBackground(detector.shape, device) if dynamic_background else None

    # parameters: raw, unconstrained tensors; the physical values are functions of them

    def initial(self, flags, jitter):
        """The raw parameters at the default appearance (no blur, linear contrast),
        or, with a torch Generator *jitter*, at a random point around it."""

        def draw(low, high, size=()):
            if jitter is None:
                return None
            return low + (high - low) * torch.rand(size, generator=jitter)

        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float32).to(self.device)

        params = {
            "energy": tensor(torch.zeros(self.energy_terms)),
            "background": tensor(torch.zeros(len(self.background_terms))),
        }
        if flags["energy"] and jitter is not None:
            params["energy"] = tensor(draw(-1.0, 1.0, (self.energy_terms,)))
        # the PSF starts at a third of a Lambert pixel when fitted: at exactly zero
        # its gradient vanishes
        start_px = 1.0 / 3.0 if flags["psf"] else 0.0
        sigma = start_px * 90.0 / self.npx
        if jitter is not None and flags["psf"]:
            sigma = float(draw(0.05, 0.8 * MAX_PSF_DEG))
        params["psf"] = tensor(_logit(sigma / MAX_PSF_DEG))
        halo = 0.05 if flags["halo"] else 0.0
        params["halo_fraction"] = tensor(_logit(halo / MAX_HALO_FRACTION))
        params["halo_sigma"] = tensor(_logit(3.0 / MAX_HALO_DEG))
        gamma = 1.0 if jitter is None or not flags["gamma"] else float(draw(0.5, 2.0))
        params["gamma"] = tensor(math.atanh(math.log(gamma) / math.log(MAX_GAMMA)))
        # the high pass starts at a tenth (at zero its sigmoid's slope vanishes) and
        # at the indexer's width: an eighth of the detector's, in degrees at its center
        fraction = 0.1 if flags["high_pass"] else 0.0
        if jitter is not None and flags["high_pass"]:
            fraction = float(draw(0.05, 0.9))
        params["high_pass"] = tensor(_logit(fraction / MAX_HIGH_PASS))
        width = math.degrees(self.aspect / (8.0 * float(self.center[2])))
        low, high = HIGH_PASS_DEG
        width = min(max(width, low + 0.5), high - 0.5)
        if jitter is not None and flags["high_pass"]:
            width = float(draw(0.5 * width, min(1.5 * width, high - 0.5)))
        params["high_pass_sigma"] = tensor(_logit((width - low) / (high - low)))
        params["center"] = tensor(torch.zeros(3))
        params["distortion"] = tensor(0.0)
        return params

    def physical(self, params, flags=None):
        """The physical parameters (tensors) of *params*; those not fitted at their
        defaults where *flags* is given."""
        values = {
            "weights": self.weights(params["energy"]),
            "sigma": MAX_PSF_DEG * torch.sigmoid(params["psf"]),
            "halo_fraction": MAX_HALO_FRACTION * torch.sigmoid(params["halo_fraction"]),
            "halo_sigma": MAX_HALO_DEG * torch.sigmoid(params["halo_sigma"]),
            "gamma": torch.exp(math.log(MAX_GAMMA) * torch.tanh(params["gamma"])),
            "high_pass": MAX_HIGH_PASS * torch.sigmoid(params["high_pass"]),
            "high_pass_sigma": HIGH_PASS_DEG[0]
            + (HIGH_PASS_DEG[1] - HIGH_PASS_DEG[0]) * torch.sigmoid(params["high_pass_sigma"]),
            "background": params["background"],
            "center": MAX_CENTER_SHIFT * torch.tanh(params["center"]),
            "distortion": MAX_DISTORTION * torch.tanh(params["distortion"]),
        }
        if not self.halo:
            values["halo_fraction"] = values["halo_fraction"] * 0
        return values

    def weights(self, curve):
        if not len(curve):
            return self.prior
        w = self.prior * torch.exp(curve @ self.energy_basis)
        return w / w.sum()

    def texture(self, physical):
        return _texture(
            self.layers,
            physical["weights"],
            self.npx,
            physical["sigma"],
            physical["halo_fraction"],
            physical["halo_sigma"],
            physical["gamma"],
            self.contrast_range,
            physical["high_pass"],
            physical["high_pass_sigma"],
        )[None]

    def directions(self, physical):
        """Sample-frame unit vectors to every pixel ``(P, 3)``, at the shifted center
        and through the distortion."""
        center = self.center + physical["center"]
        position = self.pixels * (1.0 + physical["distortion"] * (self.pixels**2).sum(-1))[:, None]
        detector = torch.stack(
            [
                position[:, 0] + self.aspect * (0.5 - center[0]),
                position[:, 1] + (center[1] - 0.5),
                center[2].expand(len(position)),
            ],
            -1,
        )
        rays = detector @ self.detector_to_sample
        return rays / rays.norm(dim=-1, keepdim=True)

    def simulate(self, params, g, omega, texture=None, directions=None, indexer=False):
        """Normalized simulated rows ``(N, P)`` at orientations ``g @ R(omega)``: the
        model's (with the background, and high-passed as the real patterns were), or
        with *indexer* the master's projection alone, as the indexer scores it."""
        physical = self.physical(params)
        texture = self.texture(physical) if texture is None else texture
        directions = self.directions(physical) if directions is None else directions
        turned = g @ rotation_vector_to_matrix(omega * ROTATION_UNIT)
        v = torch.einsum("nij,pj->npi", turned, directions)
        values = sample_texture(texture, v)
        if not indexer:
            if len(self.background_terms):
                values = values + physical["background"] @ self.background_terms
            if self.dynamic is not None:
                grid = values.view(len(values), *self.detector.shape)
                values = self.dynamic.remove(grid).flatten(1)
        return normalize_rows(values)

    def ncc(self, params, rows, g, omega, indexer=False):
        out = []
        with torch.no_grad():
            physical = self.physical(params)
            texture, directions = self.texture(physical), self.directions(physical)
            for lo in range(0, len(rows), 256):
                hi = lo + 256
                sim = self.simulate(
                    params, g[lo:hi], omega[lo:hi], texture, directions, indexer=indexer
                )
                out.append((sim * rows[lo:hi]).sum(-1))
        return torch.cat(out) if out else torch.zeros(0, device=self.device)

    def loss(self, params, rows, g, omega, indexer=False):
        sim = self.simulate(params, g, omega, indexer=indexer)
        ncc = (sim * rows).sum(-1).mean()
        ridge = RIDGE * ((params["energy"] ** 2).sum() + (params["background"] ** 2).sum())
        return -ncc + ridge, ncc

    def fit(self, params, flags, rows, g, omega, iterations, callback, indexer=False):
        """Every fitted shared parameter and every orientation together, by L-BFGS."""
        fitted = _fitted_keys(flags)
        params = {k: v.detach().clone().requires_grad_(k in fitted) for k, v in params.items()}
        omega = omega.detach().clone().requires_grad_(True)
        variables = [params[k] for k in fitted] + [omega]
        optimizer = torch.optim.LBFGS(
            variables,
            lr=1.0,
            max_iter=iterations,
            history_size=20,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )
        evaluations = [0]

        def closure():
            optimizer.zero_grad()
            loss, ncc = self.loss(params, rows, g, omega, indexer=indexer)
            loss.backward()
            evaluations[0] += 1
            callback(evaluations[0], float(ncc))
            return loss

        optimizer.step(closure)
        return {k: v.detach() for k, v in params.items()}, omega.detach()

    def refine_orientations(self, params, rows, g, iterations, indexer=False):
        """Each orientation refined with the shared parameters fixed: ``omega (N, 3)``."""
        params = {k: v.detach() for k, v in params.items()}
        physical = self.physical(params)
        with torch.no_grad():
            texture, directions = self.texture(physical), self.directions(physical)
        out = []
        for lo in range(0, len(rows), 256):
            hi = lo + 256
            omega = torch.zeros(len(rows[lo:hi]), 3, device=self.device, requires_grad=True)
            optimizer = torch.optim.LBFGS(
                [omega], max_iter=iterations, history_size=10, line_search_fn="strong_wolfe"
            )

            def closure(omega=omega, lo=lo, hi=hi, optimizer=optimizer):
                optimizer.zero_grad()
                sim = self.simulate(params, g[lo:hi], omega, texture, directions, indexer)
                # summed, not averaged: each pattern's own step is then its own
                loss = -(sim * rows[lo:hi]).sum()
                loss.backward()
                return loss

            optimizer.step(closure)
            out.append(omega.detach())
        return torch.cat(out) if out else torch.zeros(0, 3, device=self.device)

    def appearance(self, params, flags, rows, g, omega) -> DetectorAppearance:
        """*params* as a :class:`DetectorAppearance`, those *flags* do not fit at their
        defaults, the background's share measured on the patterns *rows* at
        ``g R(omega)`` where they are given."""
        with torch.no_grad():
            p = self.physical(params)
            share = 0.0
            if rows is not None and len(self.background_terms) and len(rows):
                texture, directions = self.texture(p), self.directions(p)
                turned = g @ rotation_vector_to_matrix(omega * ROTATION_UNIT)
                v = torch.einsum("nij,pj->npi", turned, directions)
                background = (p["background"] @ self.background_terms)[None]
                values = sample_texture(texture, v) + background
                if self.dynamic is not None:
                    shape = self.detector.shape
                    values = self.dynamic.remove(values.view(-1, *shape)).flatten(1)
                    background = self.dynamic.remove(background.view(1, *shape)).flatten(1)
                values = values - values.mean(-1, keepdim=True)
                b = background - background.mean()
                share = float(
                    (b.norm() / values.norm(dim=-1).clamp_min(SINGLE_PRECISION_FLOOR)).mean()
                )
            halo = float(p["halo_fraction"]) if flags["halo"] else 0.0
            return DetectorAppearance(
                energies_kev=self.energies,
                prior_weights=self.prior_np,
                energy_curve=params["energy"].detach().cpu().double().numpy(),
                psf_sigma_deg=float(p["sigma"]) if flags["psf"] else 0.0,
                halo_fraction=halo,
                halo_sigma_deg=float(p["halo_sigma"]) if halo else 0.0,
                gamma=float(p["gamma"]) if flags["gamma"] else 1.0,
                high_pass_fraction=float(p["high_pass"]) if flags["high_pass"] else 0.0,
                high_pass_sigma_deg=float(p["high_pass_sigma"]) if flags["high_pass"] else 0.0,
                contrast_range=self.contrast_range,
                background=p["background"].cpu().double().numpy(),
                background_order=self.background_order,
                background_share=share,
                background_high_pass=self.dynamic is not None,
                pattern_center_shift=p["center"].cpu().double().numpy(),
                distortion=float(p["distortion"]),
            )

    def correlations(self, params, flags, rows, g, omega, step=1e-2):
        """The fitted parameters' correlations with the orientations profiled out:
        ``(names, correlations, flat)``, *flat* the names of directions the mean NCC
        does not curve along.

        The curvature (Hessian of the loss) is taken by central differences of its
        gradient: a column per shared parameter, and each pattern's own 3 x 3
        orientation block from three more pairs (the patterns' orientations do not
        couple to each other). The orientations are then eliminated (a Schur
        complement), so a parameter a rotation could stand in for shows as weak."""
        fitted = _fitted_keys(flags)
        names, slots = [], []
        for key in fitted:
            for i in range(params[key].numel()):
                names.append(_parameter_name(key, i, self))
                slots.append((key, i))
        if not slots:
            return [], None, []
        n = len(rows)

        def gradient(shift_key=None, shift_index=None, shift=0.0, omega_axis=None):
            p = {k: v.detach().clone() for k, v in params.items()}
            o = omega.detach().clone()
            if shift_key is not None:
                p[shift_key].view(-1)[shift_index] += shift
            if omega_axis is not None:
                o[:, omega_axis] += shift
            for k in fitted:
                p[k].requires_grad_(True)
            o.requires_grad_(True)
            # summed over patterns, so each orientation block is one pattern's
            loss = -(self.simulate(p, g, o) * rows).sum()
            loss.backward()
            shared = torch.cat([p[k].grad.reshape(-1) for k in fitted])
            return shared.double(), o.grad.double()

        m = len(slots)
        h_ss = torch.zeros(m, m, dtype=torch.float64, device=self.device)
        h_so = torch.zeros(m, n, 3, dtype=torch.float64, device=self.device)
        for column, (key, i) in enumerate(slots):
            plus = gradient(key, i, step)
            minus = gradient(key, i, -step)
            h_ss[:, column] = (plus[0] - minus[0]) / (2 * step)
            h_so[column] = (plus[1] - minus[1]) / (2 * step)
        h_oo = torch.zeros(n, 3, 3, dtype=torch.float64, device=self.device)
        for axis in range(3):
            plus = gradient(omega_axis=axis, shift=step)
            minus = gradient(omega_axis=axis, shift=-step)
            h_oo[:, :, axis] = (plus[1] - minus[1]) / (2 * step)
        h_ss = 0.5 * (h_ss + h_ss.T)
        h_oo = 0.5 * (h_oo + h_oo.transpose(1, 2))
        h_oo = h_oo + 1e-9 * torch.eye(3, dtype=torch.float64, device=self.device)
        solved = torch.linalg.solve(h_oo, h_so.permute(1, 2, 0))  # (n, 3, m)
        schur = h_ss - torch.einsum("mnk,nkj->mj", h_so, solved)
        schur = 0.5 * (schur + schur.T).cpu()
        # a parameter is flat where a whole unit of it (its raw, unbounded value)
        # moves the mean NCC by less than FLAT_NCC: the patterns do not constrain it,
        # and it is left out of the correlations, which it would only make noise in
        diagonal = torch.diagonal(schur)
        steep = (0.5 * diagonal / n) > FLAT_NCC
        flat = [names[i] for i in range(m) if not steep[i]]
        correlations = torch.full((m, m), float("nan"), dtype=torch.float64)
        kept = torch.nonzero(steep).flatten()
        if len(kept):
            covariance = torch.linalg.pinv(schur[kept][:, kept], hermitian=True)
            spread = torch.diagonal(covariance).clamp_min(1e-300).sqrt()
            inner = (covariance / spread[:, None] / spread[None, :]).clamp(-1, 1)
            correlations[kept[:, None], kept[None, :]] = inner
        return names, correlations.numpy(), flat

    def warnings(self, appearance, flags, names, correlations, flat) -> list:
        out = []
        if flags["psf"] and appearance.psf_sigma_deg > 0.95 * MAX_PSF_DEG:
            out.append(f"PSF sigma at its bound ({MAX_PSF_DEG} deg)")
        if flags["halo"]:
            if appearance.halo_fraction > 0.95 * MAX_HALO_FRACTION:
                out.append(f"halo fraction at its bound ({MAX_HALO_FRACTION})")
            if appearance.halo_sigma_deg > 0.95 * MAX_HALO_DEG:
                out.append(f"halo sigma at its bound ({MAX_HALO_DEG} deg)")
        if flags["gamma"] and not (1.05 / MAX_GAMMA < appearance.gamma < 0.95 * MAX_GAMMA):
            out.append(f"gamma at its bound ({appearance.gamma:.3g})")
        if flags["high_pass"] and appearance.high_pass_fraction > 0.97 * MAX_HIGH_PASS:
            out.append(f"high-pass fraction at its bound ({MAX_HIGH_PASS})")
        if flags["high_pass"] and not (
            1.02 * HIGH_PASS_DEG[0] < appearance.high_pass_sigma_deg < 0.98 * HIGH_PASS_DEG[1]
        ):
            out.append(f"high-pass sigma at its bound ({appearance.high_pass_sigma_deg:.3g} deg)")
        if flags["energy"] and np.any(np.abs(appearance.energy_curve) > 3.0):
            out.append(
                "energy weights far from the Monte Carlo's (a curve coefficient past 3): "
                "check the beam energy and the energy range simulated"
            )
        if flags["center"] and np.any(np.abs(appearance.pattern_center_shift) > MAX_CENTER_SHIFT):
            out.append("pattern center shift at its bound")
        if flags["distortion"] and abs(appearance.distortion) > 0.95 * MAX_DISTORTION:
            out.append("distortion at its bound")
        for name in flat:
            out.append(f"{name} is not constrained by these patterns")
        if correlations is not None:
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    if np.abs(correlations[i, j]) > CORRELATED:
                        out.append(
                            f"{names[i]} and {names[j]} are strongly correlated "
                            f"({correlations[i, j]:+.2f})"
                        )
        return out


# --- pieces --------------------------------------------------------------------------------


def _only(params, *prefixes) -> dict:
    """The entries of *params* whose keys start with one of *prefixes*."""
    return {k: v for k, v in params.items() if k.startswith(prefixes)}


def _without_residual(rows, residual, shape):
    """Normalized *rows* less the *residual* background, as the indexer removes it."""
    if residual is None or not residual.share:
        return rows
    direction = residual.direction(shape, rows.device)
    return normalize_rows(rows - residual.share * direction)


def _fitted_keys(flags) -> list:
    keys = []
    if flags["energy"]:
        keys.append("energy")
    if flags["psf"]:
        keys.append("psf")
    if flags["halo"]:
        keys += ["halo_fraction", "halo_sigma"]
    if flags["gamma"]:
        keys.append("gamma")
    if flags["high_pass"]:
        keys += ["high_pass", "high_pass_sigma"]
    if flags["background"]:
        keys.append("background")
    if flags["center"]:
        keys.append("center")
    if flags["distortion"]:
        keys.append("distortion")
    return keys


def _parameter_name(key, index, model) -> str:
    if key == "energy":
        return f"energy curve P{index + 1}"
    if key == "background":
        terms = [
            (total - j, j)
            for total in range(1, model.background_order + 1)
            for j in range(total + 1)
        ]
        i, j = terms[index]
        return f"background x^{i} y^{j}"
    if key == "center":
        return f"pattern center {'xyz'[index]}"
    return {
        "psf": "PSF sigma",
        "halo_fraction": "halo fraction",
        "halo_sigma": "halo sigma",
        "gamma": "gamma",
        "high_pass": "high-pass fraction",
        "high_pass_sigma": "high-pass sigma",
        "distortion": "distortion",
    }[key]


def _prior_weights(resolved) -> np.ndarray:
    if resolved.energy_weights is None:
        prior = np.ones(len(resolved.energies_kev))
    else:
        prior = np.asarray(resolved.energy_weights, dtype=np.float64)
    if prior.sum() <= 0:
        raise ValueError("the master's energy weights are all zero")
    return prior / prior.sum()


def _energy_basis(energies, terms) -> np.ndarray:
    """Legendre polynomials ``P_1 .. P_terms`` ``(terms, E)`` over the energy range."""
    if terms == 0:
        return np.zeros((0, len(energies)))
    low, high = float(np.min(energies)), float(np.max(energies))
    t = (2.0 * (energies - low) / max(high - low, 1e-12)) - 1.0
    return np.stack(
        [np.polynomial.legendre.legval(t, np.eye(terms + 1)[k]) for k in range(1, terms + 1)]
    )


def _energy_weights_np(prior, curve, energies) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    if not curve.size:
        return np.asarray(prior, dtype=np.float64)
    w = prior * np.exp(curve @ _energy_basis(energies, len(curve)))
    return w / w.sum()


def _pixel_positions(shape, device) -> torch.Tensor:
    """Every pixel center's ``(x, y)`` from the detector's center ``(P, 2)``, in
    detector heights, x right and y up, row-major."""
    rows, cols = int(shape[0]), int(shape[1])
    x = ((torch.arange(cols, device=device, dtype=torch.float32) + 0.5) / cols - 0.5) * (
        cols / rows
    )
    y = 0.5 - (torch.arange(rows, device=device, dtype=torch.float32) + 0.5) / rows
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _matrices(orientations) -> np.ndarray | torch.Tensor:
    """Orientations as ``(N, 3, 3)`` matrices: matrices as they are, an orix
    ``Rotation`` or ``(N, 4)`` quaternions converted."""
    if torch.is_tensor(orientations):
        if orientations.shape[-2:] == (3, 3):
            return orientations.reshape(-1, 3, 3)
        return quaternion_to_matrix(orientations.reshape(-1, 4))
    data = np.asarray(getattr(orientations, "data", orientations), dtype=np.float64)
    if data.shape[-2:] == (3, 3):
        return data.reshape(-1, 3, 3)
    return quaternion_to_matrix(torch.as_tensor(data.reshape(-1, 4))).numpy()


def _sphere_blur(texture, sigma_px, radius: int):
    """*texture* ``(2, n, n)`` (north, south) blurred by a Gaussian of *sigma_px*
    Lambert pixels (a tensor, so the blur is differentiable in it), out to *radius*.

    Past an edge of the square a hemisphere continues into the other one, mirrored,
    so each is padded with the other's rows and columns rather than its own."""
    radius = min(int(radius), texture.shape[-1] - 1)
    if radius < 1:
        return texture
    x = torch.arange(-radius, radius + 1, device=texture.device, dtype=texture.dtype)
    kernel = torch.exp(-0.5 * (x / sigma_px.clamp_min(1e-3)) ** 2)
    kernel = kernel / kernel.sum()
    other = texture.flip(0)

    def pad_rows(t, o):
        top = o[:, 1 : radius + 1].flip(1)
        bottom = o[:, -radius - 1 : -1].flip(1)
        return torch.cat([top, t, bottom], 1)

    t = pad_rows(texture, other)
    o = pad_rows(other, texture)
    t = torch.cat([o[:, :, 1 : radius + 1].flip(2), t, o[:, :, -radius - 1 : -1].flip(2)], 2)
    t = t[:, None]
    t = F.conv2d(t, kernel.view(1, 1, 1, -1))
    t = F.conv2d(t, kernel.view(1, 1, -1, 1))
    return t[:, 0]


def _texture(
    layers,
    weights,
    npx,
    sigma,
    halo_fraction,
    halo_sigma,
    gamma,
    contrast_range,
    high_pass=0.0,
    high_pass_sigma=0.0,
):
    """The appearance's master ``(2, n, n)``: the energy layers weighted and summed,
    blurred on the sphere, mapped through the contrast, high-passed. Every parameter
    may be a tensor with a gradient or a plain number."""

    def as_tensor(value):
        return torch.as_tensor(value, device=layers.device, dtype=layers.dtype)

    master = torch.einsum("e,ehij->hij", as_tensor(weights), layers)
    per_degree = npx / 90.0
    sigma, halo_fraction, halo_sigma = (as_tensor(v) for v in (sigma, halo_fraction, halo_sigma))
    blurred = master
    if float(sigma) > 0:
        radius = math.ceil(4.0 * min(float(sigma), MAX_PSF_DEG) * per_degree)
        blurred = _sphere_blur(master, sigma * per_degree, radius)
    if float(halo_fraction) > 0:
        radius = math.ceil(4.0 * min(float(halo_sigma), MAX_HALO_DEG) * per_degree)
        wide = _sphere_blur(master, halo_sigma * per_degree, radius)
        blurred = (1 - halo_fraction) * blurred + halo_fraction * wide
    low, high = contrast_range
    scaled = ((blurred - low) / max(high - low, 1e-12)).clamp_min(1e-4)
    texture = scaled ** as_tensor(gamma)
    high_pass = as_tensor(high_pass)
    if float(high_pass) > 1e-4:
        # a wide blur is as well found at a quarter of the resolution, and drawn back up
        coarse = F.avg_pool2d(texture[:, None], 4, ceil_mode=True)[:, 0]
        sigma_coarse = as_tensor(high_pass_sigma) * per_degree / 4.0
        radius = math.ceil(4.0 * float(sigma_coarse))
        wide = _sphere_blur(coarse, sigma_coarse, radius)
        wide = F.interpolate(wide[:, None], size=texture.shape[-2:], mode="bilinear")[:, 0]
        texture = texture - high_pass * wide
    return texture
