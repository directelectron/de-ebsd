"""GPU background subtraction (preprocessing) -- shapes, semantics, and the blur.

No kikuchipy: the Gaussian blur is checked against :func:`scipy.ndimage.gaussian_filter`
(the spatial-domain filter this module implements), and the subtract/divide + rescale
semantics against a hand-computed reference.
"""

import numpy as np
import pytest
import torch

from de_ebsd.index import (
    remove_background,
    remove_dynamic_background,
    remove_static_background,
)
from de_ebsd.index.preprocess import _gaussian_blur


def _rng(shape, seed=0):
    g = np.random.default_rng(seed)
    return g.random(shape, dtype=np.float64).astype(np.float32)


def test_gaussian_blur_matches_scipy():
    """The separable reflected-padding blur matches scipy's spatial ``gaussian_filter``.

    torch's ``F.pad(mode="reflect")`` mirrors about the edge pixel *without* repeating it,
    which is scipy's ``mode="mirror"`` (not its default ``"reflect"``, which repeats the
    edge); compared against the matching mode the two agree to fp32 precision.
    """
    ndi = pytest.importorskip("scipy.ndimage")
    imgs = _rng((3, 40, 44))
    std = 44 / 8.0
    out = _gaussian_blur(torch.as_tensor(imgs), std, truncate=4.0).numpy()
    for i in range(imgs.shape[0]):
        ref = ndi.gaussian_filter(imgs[i], sigma=std, truncate=4.0, mode="mirror")
        assert np.allclose(out[i], ref, atol=1e-5), f"max {np.abs(out[i] - ref).max():.2e}"


def test_static_subtract_semantics():
    """Static subtract removes the fixed background and rescales each pattern to [0, 1]."""
    bg = _rng((16, 16), seed=1)
    signal = _rng((5, 16, 16), seed=2)
    patterns = signal + bg[None]  # each pattern = its own signal + the shared background
    out = remove_static_background(patterns, bg).numpy()
    # after subtracting bg we recover `signal`; rescale is affine, so out == rescaled signal
    for i in range(5):
        s = signal[i]
        ref = (s - s.min()) / (s.max() - s.min())
        assert np.allclose(out[i], ref, atol=1e-5)
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6


def test_shapes_roundtrip_all_layouts():
    """(H,W), (N,H,W) and (rows,cols,H,W) inputs come back with the same leading shape."""
    bg = _rng((12, 12))
    single = remove_dynamic_background(_rng((12, 12)))
    assert tuple(single.shape) == (12, 12)
    stack = remove_dynamic_background(_rng((7, 12, 12)))
    assert tuple(stack.shape) == (7, 12, 12)
    grid = remove_background(_rng((3, 4, 12, 12)), static_bg=bg)
    assert tuple(grid.shape) == (3, 4, 12, 12)


def test_numpy_and_torch_inputs_agree():
    """A numpy array and the equivalent torch tensor preprocess identically."""
    arr = _rng((6, 20, 20), seed=7)
    bg = _rng((20, 20), seed=8)
    from_np = remove_background(arr, static_bg=bg).numpy()
    from_torch = remove_background(torch.as_tensor(arr), static_bg=torch.as_tensor(bg)).numpy()
    assert np.allclose(from_np, from_torch, atol=1e-6)


def test_remove_background_composes_static_then_dynamic():
    """``remove_background`` equals static-then-dynamic applied in sequence."""
    arr = _rng((4, 24, 24), seed=3)
    bg = _rng((24, 24), seed=4)
    combined = remove_background(arr, static_bg=bg).numpy()
    stepwise = remove_dynamic_background(remove_static_background(arr, bg)).numpy()
    assert np.allclose(combined, stepwise, atol=1e-6)
    # skipping the static step is honoured
    dyn_only = remove_background(arr, static_bg=None).numpy()
    assert np.allclose(dyn_only, remove_dynamic_background(arr).numpy(), atol=1e-6)


def test_divide_operation_runs_and_is_bounded():
    """The divide operation is supported and stays finite/bounded in [0, 1]."""
    arr = _rng((3, 16, 16), seed=5) + 1.0  # strictly positive so division is well posed
    out = remove_dynamic_background(arr, operation="divide").numpy()
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6


def test_default_std_is_width_over_eight():
    """The default dynamic ``std`` is width/8 (matches kikuchipy)."""
    arr = _rng((2, 32, 48), seed=6)
    default = remove_dynamic_background(arr).numpy()
    explicit = remove_dynamic_background(arr, std=48 / 8.0).numpy()
    assert np.allclose(default, explicit, atol=1e-6)


def test_bad_shape_and_bad_operation_raise():
    with pytest.raises(ValueError, match="patterns must be"):
        remove_dynamic_background(_rng((2, 3, 4, 5, 6)))
    with pytest.raises(ValueError, match="operation must be"):
        remove_static_background(_rng((8, 8)), _rng((8, 8)), operation="multiply")
    with pytest.raises(ValueError, match="static_bg shape"):
        remove_static_background(_rng((8, 8)), _rng((4, 4)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu():
    """The GPU path matches the CPU path (bit-for-bit up to fp32 tolerance)."""
    arr = _rng((10, 24, 24), seed=9)
    bg = _rng((24, 24), seed=10)
    cpu = remove_background(arr, static_bg=bg, device="cpu").numpy()
    gpu = remove_background(arr, static_bg=bg, device="cuda").cpu().numpy()
    assert np.allclose(cpu, gpu, atol=1e-4)
