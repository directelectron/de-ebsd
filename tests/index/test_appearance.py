"""The appearance fit: its model, its fit on patterns rendered through a known
appearance, and the indexer's use of what it finds.

The master is an energy-resolved band master (``appearance_scan.py``), whose energies
differ in band widths and strengths, so the tests run in seconds on a CPU; the
dynamical validation, on a GPU, is ``scripts/validate_appearance.py``.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch
from appearance_scan import detector, nickel, random_orientations, render, resolved_band_master
from orix.quaternion import Rotation

from de_ebsd.core.rotation import misorientation_deg, rotation_vector_to_matrix
from de_ebsd.index.appearance import (
    DetectorAppearance,
    _Model,
    _sphere_blur,
    fit_appearance,
    load_appearance,
    select_patterns,
)
from de_ebsd.index.io import phase_from_crystal
from de_ebsd.index.live import LiveIndexer
from de_ebsd.index.live.background import ResidualBackground, ScanBackground, legendre_terms
from de_ebsd.index.preprocess import normalize_rows


@pytest.fixture(scope="module")
def resolved():
    return resolved_band_master(npx=64)


@pytest.fixture(scope="module")
def point_group():
    return phase_from_crystal(nickel()).point_group


@pytest.fixture(scope="module")
def truth(resolved):
    return replace(
        DetectorAppearance.default(resolved),
        energy_curve=np.array([-0.8, 0.3]),
        psf_sigma_deg=2.5,
        gamma=0.6,
        background=np.array([0.05, -0.08, 0.03, 0.0, 0.02]),
        background_order=2,
    )


def _corrected(patterns):
    """Patterns with their dynamic background removed, as the indexer's are."""
    return ScanBackground(tuple(patterns.shape[-2:]), patterns.device).remove(patterns)


def _perturbed(g, degrees, seed=0):
    generator = torch.Generator().manual_seed(seed)
    turn = torch.randn(len(g), 3, generator=generator) * np.radians(degrees) / np.sqrt(3)
    return (torch.as_tensor(g, dtype=torch.float32) @ rotation_vector_to_matrix(turn)).numpy()


# --- the model ------------------------------------------------------------------------


def test_default_appearance_is_the_monte_carlo_master(resolved):
    """The default appearance's master is today's (the energies summed with the Monte
    Carlo's weights), up to the contrast's linear map onto [0, 1]."""
    default = DetectorAppearance.default(resolved)
    master = default.master(resolved)
    summed = resolved.collapse().data[0] / resolved.energy_weights.sum()
    low, high = default.contrast_range
    expected = ((summed - low) / (high - low)).clamp_min(1e-4)
    assert master.data.shape == (1, *resolved.data.shape[1:])
    assert torch.allclose(master.data[0], expected, atol=1e-5)
    np.testing.assert_allclose(default.energy_weights, default.prior_weights)


def test_model_directions_are_the_detectors(resolved):
    """With no center shift and no distortion, the model sees the detector's pixels."""
    det = detector((24, 30))
    model = _Model(resolved, det, "cpu", energy_order=2, halo=False, background_order=0)
    flags = dict.fromkeys(
        ("energy", "psf", "halo", "gamma", "high_pass", "background", "center", "distortion"),
        False,
    )
    directions = model.directions(model.physical(model.initial(flags, None)))
    assert torch.allclose(directions, det.direction_cosines(), atol=1e-5)


def test_sphere_blur_keeps_a_constant_and_smooths():
    flat = torch.full((2, 33, 33), 3.0)
    assert torch.allclose(_sphere_blur(flat, torch.tensor(2.0), 8), flat, atol=1e-5)
    spike = torch.zeros(2, 33, 33)
    spike[0, 16, 16] = 1.0
    blurred = _sphere_blur(spike, torch.tensor(2.0), 8)
    assert blurred[0, 16, 16] < 0.1
    assert blurred.sum() == pytest.approx(1.0, abs=1e-4)
    # past an edge the north hemisphere continues into the south
    edge = torch.zeros(2, 33, 33)
    edge[0, 1, 16] = 1.0  # beside the equator, which both hemispheres share
    assert _sphere_blur(edge, torch.tensor(2.0), 8)[1, 1, 16] > 0


def test_legendre_terms_are_smooth_and_free_of_the_constant():
    terms = legendre_terms((20, 30), 3)
    assert terms.shape == (9, 600)
    assert torch.allclose(terms.mean(-1), torch.zeros(9), atol=0.05)


# --- the fit ------------------------------------------------------------------------------


def test_fit_recovers_a_known_appearance(resolved, truth):
    """Patterns rendered through a known appearance, Poisson noise, orientations half a
    degree off: the fit finds the PSF, the gamma and the background, moves the energy
    weights the true way, and the held-out patterns match better.

    The energy weights are only loosely fixed beside a PSF: this master's low energies
    are mostly its high ones blurred, so a blur can stand in for part of them (see
    the next test for the weights alone)."""
    det = detector((40, 40))
    g = random_orientations(100)
    patterns = render(resolved, truth, g, det, counts=200)
    fit = fit_appearance(
        resolved,
        patterns,
        _perturbed(g, 0.5),
        det,
        train=np.arange(75),
        held_out=np.arange(75, 100),
        background_order=2,
        dynamic_background=False,
    )
    a = fit.appearance
    prior = fit.default.energy_weights
    assert a.energy_weights[0] > 1.5 * prior[0] and a.energy_weights[-1] < prior[-1]
    assert a.psf_sigma_deg == pytest.approx(truth.psf_sigma_deg, rel=0.15)
    assert a.gamma == pytest.approx(truth.gamma, rel=0.1)
    np.testing.assert_allclose(a.background, truth.background, atol=0.01)
    s = fit.summary()
    assert s["held_out_after"] > s["held_out_before"] + 0.05
    assert s["train_after"] > s["train_before"] + 0.05
    assert a.residual_background() is not None and a.background_share > 0
    assert fit.log and fit.log[-1][2] > fit.log[0][2]  # the convergence log rises
    assert len(fit.parameter_names) == 2 + 1 + 1 + 5
    assert fit.correlations.shape == (9, 9)


def test_fit_recovers_energy_weights(resolved):
    """Only the energy weights differ: the fit finds them."""
    det = detector((40, 40))
    truth = replace(DetectorAppearance.default(resolved), energy_curve=np.array([-0.8, 0.3]))
    g = random_orientations(100)
    patterns = render(resolved, truth, g, det, counts=200)
    fit = fit_appearance(
        resolved,
        patterns,
        _perturbed(g, 0.5),
        det,
        train=np.arange(75),
        held_out=np.arange(75, 100),
        fit_psf=False,
        fit_contrast=False,
        background_order=0,
        dynamic_background=False,
    )
    np.testing.assert_allclose(fit.appearance.energy_weights, truth.energy_weights, atol=0.03)
    assert fit.summary()["held_out_after"] > fit.summary()["held_out_before"]


def test_fit_stays_near_the_default_without_a_mismatch(resolved):
    """Patterns rendered from the default appearance: nothing to fit, and the fit
    neither moves far nor loses held-out NCC."""
    det = detector((40, 40))
    default = DetectorAppearance.default(resolved)
    g = random_orientations(80, seed=3)
    patterns = render(resolved, default, g, det, counts=100, seed=3)
    fit = fit_appearance(
        resolved,
        patterns,
        g,
        det,
        train=np.arange(60),
        held_out=np.arange(60, 80),
        background_order=2,
        dynamic_background=False,
        restarts=1,
    )
    a = fit.appearance
    np.testing.assert_allclose(a.energy_weights, default.energy_weights, atol=0.03)
    assert a.psf_sigma_deg < 0.6  # under a Lambert pixel (1.4 degrees here)
    assert a.gamma == pytest.approx(1.0, abs=0.1)
    assert np.abs(a.background).max() < 0.01
    s = fit.summary()
    assert s["held_out_after"] >= s["held_out_before"] - 1e-3
    assert len(fit.restarts) == 2


def test_fit_on_the_indexers_patterns_improves_indexing(resolved, truth, point_group):
    """End to end: a scan indexed against the default master, fitted on the indexer's
    own corrected patterns (whose dynamic background the fit puts on its simulations
    too), indexed again against the fitted master with the residual background
    removed. The scores rise, the margin with them, and the orientations hold."""
    det = detector((40, 40))
    g = random_orientations(120, seed=5)
    patterns = render(resolved, truth, g, det, counts=50, seed=5)
    operators = point_group.to_matrix()

    def index(appearance):
        indexer = LiveIndexer(
            [appearance.master(resolved)],
            [point_group],
            det,
            resolution=6.0,
            search_shape=(20, 20),
            refine_shape=(40, 40),
            search_blur_deg=3.0,
            device="cpu",
            residual_background=appearance.residual_background(),
        )
        _, found, score, _ = indexer.index(patterns)
        error = np.array(
            [misorientation_deg(found[i : i + 1], g[i], operators)[0] for i in range(len(g))]
        )
        return indexer, found, score, error

    indexer, found, score, error = index(DetectorAppearance.default(resolved))
    train, held = select_patterns(found, score, point_group, count=90, per_grain=5, min_score=-1)
    chosen = np.concatenate([train, held])
    rows = indexer.prepare(patterns[chosen])[1].view(len(chosen), 40, 40)
    fit = fit_appearance(
        resolved,
        rows,
        found[chosen],
        det,
        train=np.arange(len(train)),
        held_out=np.arange(len(train), len(chosen)),
        background_order=2,
        point_group=point_group,
        margin_resolution_deg=6.0,
    )
    s = fit.summary()
    assert s["held_out_after"] > s["held_out_before"]
    assert s["held_out_model"] > s["held_out_after"]  # the model matches best of all
    assert fit.appearance.background_high_pass
    assert fit.appearance.high_pass_fraction > 0
    # the fitted model's physics, from the indexer's patterns: weights tilted the
    # true way, a blur and a gamma near the truth's
    assert fit.appearance.energy_weights[0] > fit.default.energy_weights[0]
    assert fit.appearance.psf_sigma_deg == pytest.approx(truth.psf_sigma_deg, rel=0.3)
    assert fit.appearance.gamma == pytest.approx(truth.gamma, rel=0.2)
    _, _, fitted_score, fitted_error = index(fit.appearance)
    assert float(fitted_score.mean()) > float(score.mean())
    assert np.mean(fitted_error > 5) <= np.mean(error > 5)
    assert np.median(fitted_error) < 1.0


def test_select_patterns_spreads_over_grains(point_group):
    rng = np.random.default_rng(0)
    centers = random_orientations(12, seed=1)
    g = np.repeat(centers, 20, axis=0)
    g = _perturbed(g, 0.5)
    score = rng.uniform(0.2, 0.9, len(g))
    train, held = select_patterns(g, score, point_group, count=60, per_grain=4, min_score=0.0)
    chosen = np.concatenate([train, held])
    grain = chosen // 20
    assert len(chosen) == 12 * 4  # every grain capped at four
    assert np.bincount(grain).max() <= 4
    assert not set(train // 20) & set(held // 20)  # whole grains held out
    assert 0.1 < len(held) / len(chosen) < 0.35


def test_appearance_saves_and_loads(resolved, truth, tmp_path):
    path = tmp_path / "appearance.json"
    truth.save(path)
    loaded = load_appearance(path)
    np.testing.assert_allclose(loaded.energy_weights, truth.energy_weights)
    assert loaded.psf_sigma_deg == truth.psf_sigma_deg
    assert torch.allclose(loaded.master(resolved).data, truth.master(resolved).data)


def test_fit_saves_and_its_appearance_loads(resolved, truth, tmp_path):
    det = detector((32, 32))
    g = random_orientations(30, seed=7)
    patterns = render(resolved, truth, g, det)
    fit = fit_appearance(
        resolved, patterns, g, det, background_order=2, dynamic_background=False, iterations=20
    )
    path = tmp_path / "fit.json"
    fit.save(path)
    loaded = DetectorAppearance.load(path)
    np.testing.assert_allclose(loaded.energy_weights, fit.appearance.energy_weights)
    assert "mean NCC" in fit.report()


def test_master_rejects_other_energy_bins(resolved, truth):
    from de_ebsd import MasterPattern

    other = MasterPattern(
        resolved.data[:3], resolved.energies_kev[:3], {}, resolved.energy_weights[:3]
    )
    with pytest.raises(ValueError, match="energy bins"):
        truth.master(other)


# --- the indexer's residual background --------------------------------------------------


def test_indexer_removes_the_residual_background(resolved, point_group):
    """A pattern carrying a known residual background matches the master better once
    the indexer removes it, and the search rows are the corrected refine rows averaged.

    The indexer removes each pattern's dynamic background first, so the residual is
    taken through the same (``high_pass``), as a fit on its patterns finds it."""
    det = detector((40, 40))
    default = DetectorAppearance.default(resolved)
    master = default.master(resolved)
    g = torch.as_tensor(random_orientations(16, seed=2), dtype=torch.float32)
    clean = master.project(det, Rotation.from_matrix(g.numpy()), rescale=False)
    coefficients = (0.4, -0.6, 0.3, 0.0, 0.2, 0.2, -0.1, 0.0, 0.1)
    background = (torch.tensor(coefficients) @ legendre_terms((40, 40), 3)).view(1, 40, 40)
    carrying = clean + background
    dynamic = ScanBackground((40, 40), "cpu").remove
    b = normalize_rows(dynamic(background).flatten(1))
    centered = dynamic(carrying).flatten(1)
    centered = centered - centered.mean(1, keepdim=True)
    share = float((dynamic(background).flatten(1).std(1) / centered.std(1)).mean())
    residual = ResidualBackground(coefficients, 3, share, high_pass=True)
    assert torch.allclose(residual.direction((40, 40)), b[0], atol=1e-5)

    kwargs = dict(resolution=10.0, search_shape=(20, 20), refine_shape=(40, 40), device="cpu")
    plain = LiveIndexer([master], [point_group], det, **kwargs)
    removing = LiveIndexer([master], [point_group], det, residual_background=residual, **kwargs)
    reference = normalize_rows(dynamic(clean).flatten(1))
    _, before = plain.prepare(carrying)
    search, after = removing.prepare(carrying)
    assert (after * reference).sum(1).mean() > (before * reference).sum(1).mean()
    pooled = torch.nn.functional.avg_pool2d(after.view(-1, 1, 40, 40), 2).flatten(1)
    assert torch.allclose(search, normalize_rows(pooled), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fit_runs_on_cuda(resolved, truth):
    det = detector((40, 40))
    g = random_orientations(60)
    patterns = render(resolved, truth, g, det, counts=200).cuda()
    fit = fit_appearance(
        resolved.to("cuda"),
        patterns,
        g,
        det,
        train=np.arange(45),
        held_out=np.arange(45, 60),
        background_order=2,
        dynamic_background=False,
    )
    np.testing.assert_allclose(fit.appearance.energy_weights, truth.energy_weights, atol=0.02)
    assert fit.summary()["held_out_after"] > fit.summary()["held_out_before"]
