"""Fused CUDA kernels for live indexing: a master pattern sampled at rotated
detector directions in one pass, compiled at run time by CuPy.

Projecting through torch operations writes every intermediate -- the rotated
directions, their Lambert coordinates, the sampled values -- to GPU memory and
reads it back, about 390 M samples/s on a Titan X. Here each thread rotates its
direction, maps it to the square Lambert grid and interpolates the master in
registers, reading only the master (small enough to stay in L2) and writing one
value; :func:`score` does not even write that, reducing each candidate's samples
straight to its normalized cross-correlation with its pattern.

The arithmetic is :func:`de_ebsd.core.lambert.sphere_to_square` followed by
``grid_sample(mode="bilinear", align_corners=True, padding_mode="border")``, as
:meth:`MasterPattern.sample` does it; ``tests/index/test_live.py`` compares them.
CuPy is optional: without it (or off CUDA) the torch path is used.
"""

from __future__ import annotations

import functools
import importlib.util

import torch

_SOURCE = r"""
#define SQRT_PI_HALF 1.2533141373155003f
#define SQRT_PI_OVER_2 0.886226925452758f
#define TWO_OVER_SQRT_PI 1.1283791670955126f

__device__ __forceinline__ float sample_master(
    const float* __restrict__ texture, int base, int n, float vx, float vy, float vz)
{
    float inverse = rsqrtf(fmaxf(vx * vx + vy * vy + vz * vz, 1e-24f));
    vx *= inverse; vy *= inverse; vz *= inverse;
    int channel = base + (vz < 0.f ? 1 : 0);
    float abs_z = fabsf(vz);
    float lx = 0.f, ly = 0.f;
    if (abs_z < 1.f) {
        float root = sqrtf(fmaxf(2.f * (1.f - abs_z), 1e-30f));
        if (fabsf(vy) <= fabsf(vx)) {
            float sign = vx >= 0.f ? 1.f : -1.f;
            float safe = fabsf(vx) < 1e-12f ? 1e-12f : vx;
            lx = sign * root * SQRT_PI_OVER_2;
            ly = sign * root * TWO_OVER_SQRT_PI * atanf(vy / safe);
        } else {
            float sign = vy >= 0.f ? 1.f : -1.f;
            float safe = fabsf(vy) < 1e-12f ? 1e-12f : vy;
            lx = sign * root * TWO_OVER_SQRT_PI * atanf(vx / safe);
            ly = sign * root * SQRT_PI_OVER_2;
        }
    }
    float last = (float)(n - 1);
    float px = fminf(fmaxf((lx / SQRT_PI_HALF + 1.f) * 0.5f * last, 0.f), last);
    float py = fminf(fmaxf((ly / SQRT_PI_HALF + 1.f) * 0.5f * last, 0.f), last);
    int x0 = (int)floorf(px), y0 = (int)floorf(py);
    int x1 = min(x0 + 1, n - 1), y1 = min(y0 + 1, n - 1);
    float fx = px - x0, fy = py - y0;
    const float* t = texture + (size_t)channel * n * n;
    float top = t[y0 * n + x0] * (1.f - fx) + t[y0 * n + x1] * fx;
    float bottom = t[y1 * n + x0] * (1.f - fx) + t[y1 * n + x1] * fx;
    return top * (1.f - fy) + bottom * fy;
}

__device__ __forceinline__ float sample_rotated(
    const float* __restrict__ texture, int base, int n,
    const float* __restrict__ r, const float* __restrict__ d)
{
    float dx = d[0], dy = d[1], dz = d[2];
    return sample_master(texture, base, n,
                         r[0] * dx + r[1] * dy + r[2] * dz,
                         r[3] * dx + r[4] * dy + r[5] * dz,
                         r[6] * dx + r[7] * dy + r[8] * dz);
}

// out[m, p]: rotation m's pattern at direction p, from channel pair bases[m].
extern "C" __global__ void project(
    const float* __restrict__ texture, int n,
    const float* __restrict__ rotations, const int* __restrict__ bases,
    const float* __restrict__ directions, int pixels, long long count,
    float* __restrict__ out)
{
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    long long m = index / pixels;
    int p = (int)(index - m * pixels);
    out[index] = sample_rotated(texture, bases[m], n, rotations + m * 9, directions + p * 3);
}

// scores[m]: the normalized cross-correlation of rotation m's pattern with
// rows[m / per_row], a row already zero-mean and unit-norm. One block per m.
extern "C" __global__ void score(
    const float* __restrict__ texture, int n,
    const float* __restrict__ rotations, const int* __restrict__ bases,
    const float* __restrict__ directions, int pixels,
    const float* __restrict__ rows, int per_row,
    float* __restrict__ scores)
{
    extern __shared__ float shared[];
    long long m = blockIdx.x;
    const float* r = rotations + m * 9;
    const float* row = rows + (m / per_row) * (long long)pixels;
    int base = bases[m];
    float sum = 0.f, squares = 0.f, product = 0.f;
    for (int p = threadIdx.x; p < pixels; p += blockDim.x) {
        float s = sample_rotated(texture, base, n, r, directions + p * 3);
        sum += s; squares += s * s; product += s * row[p];
    }
    float* a = shared; float* b = shared + blockDim.x; float* c = shared + 2 * blockDim.x;
    a[threadIdx.x] = sum; b[threadIdx.x] = squares; c[threadIdx.x] = product;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half) {
            a[threadIdx.x] += a[threadIdx.x + half];
            b[threadIdx.x] += b[threadIdx.x + half];
            c[threadIdx.x] += c[threadIdx.x + half];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float variance = b[0] - a[0] * a[0] / pixels;
        scores[m] = c[0] * rsqrtf(fmaxf(variance, 1e-12f));
    }
}

// The block-wide sum of each of count per-thread partial sums in `values`
// (count * blockDim.x floats, value-major); the result lands in values[i * blockDim.x].
__device__ __forceinline__ void block_sums(float* values, int count)
{
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half)
            for (int i = 0; i < count; i++)
                values[i * blockDim.x + threadIdx.x] +=
                    values[i * blockDim.x + threadIdx.x + half];
        __syncthreads();
    }
}

// scores[m], along[m]: rotation m's normalized pattern dotted with rows[m / per_row]
// and with ghost (both zero-mean, unit-norm; ghost may be null). One block per m.
extern "C" __global__ void score_ghost(
    const float* __restrict__ texture, int n,
    const float* __restrict__ rotations, const int* __restrict__ bases,
    const float* __restrict__ directions, int pixels,
    const float* __restrict__ rows, int per_row, const float* __restrict__ ghost,
    float* __restrict__ scores, float* __restrict__ along)
{
    extern __shared__ float shared[];
    long long m = blockIdx.x;
    const float* r = rotations + m * 9;
    const float* row = rows + (m / per_row) * (long long)pixels;
    int base = bases[m];
    float sum = 0.f, squares = 0.f, product = 0.f, on_ghost = 0.f;
    for (int p = threadIdx.x; p < pixels; p += blockDim.x) {
        float s = sample_rotated(texture, base, n, r, directions + p * 3);
        sum += s; squares += s * s; product += s * row[p];
        if (ghost) on_ghost += s * ghost[p];
    }
    int t = threadIdx.x, b = blockDim.x;
    shared[t] = sum; shared[b + t] = squares;
    shared[2 * b + t] = product; shared[3 * b + t] = on_ghost;
    block_sums(shared, 4);
    if (t == 0) {
        float inverse = rsqrtf(fmaxf(shared[b] - shared[0] * shared[0] / pixels, 1e-12f));
        scores[m] = shared[2 * b] * inverse;
        along[m] = shared[3 * b] * inverse;
    }
}

// The raw sums of the orientation-only fit, for pattern i's seven candidates
// rotations[7 i .. 7 i + 6] (the fit, then +step about x, y and z, then -step about
// each), in one pass over the pixels and one block per pattern. With s the fit's
// sample less `offset` (the master's mean, to keep the float32 sums small) and j
// its central differences: out[i] = (sum s, sum s^2, sum j [3], sum s j [3],
// sum j j [00 01 02 11 12 22], sum s row, sum j row [3], sum s ghost, sum j ghost [3]).
// Normalizing is left to the caller: the derivative of a normalized pattern is
// a closed form in these, so no candidate is normalized and none is stored.
#define FIT_CANDIDATES 7
#define FIT_SUMS 22
extern "C" __global__ void fit_moments(
    const float* __restrict__ texture, int n,
    const float* __restrict__ rotations, const int* __restrict__ bases,
    const float* __restrict__ directions, int pixels,
    const float* __restrict__ rows, const float* __restrict__ ghost, float step,
    float offset, float* __restrict__ out)
{
    extern __shared__ float shared[];
    __shared__ float rotation[FIT_CANDIDATES * 9];
    long long i = blockIdx.x;
    int t = threadIdx.x, b = blockDim.x;
    for (int k = t; k < FIT_CANDIDATES * 9; k += b)
        rotation[k] = rotations[i * FIT_CANDIDATES * 9 + k];
    int base = bases[i * FIT_CANDIDATES];
    const float* row = rows + i * (long long)pixels;
    __syncthreads();

    float acc[FIT_SUMS];
    for (int k = 0; k < FIT_SUMS; k++) acc[k] = 0.f;
    float half_step = 0.5f / step;
    for (int p = t; p < pixels; p += b) {
        const float* d = directions + p * 3;
        float s = sample_rotated(texture, base, n, rotation, d) - offset;
        float j[3];
        for (int c = 0; c < 3; c++)
            j[c] = (sample_rotated(texture, base, n, rotation + (1 + c) * 9, d)
                    - sample_rotated(texture, base, n, rotation + (4 + c) * 9, d)) * half_step;
        float r = row[p], g = ghost ? ghost[p] : 0.f;
        acc[0] += s; acc[1] += s * s;
        for (int c = 0; c < 3; c++) {
            acc[2 + c] += j[c]; acc[5 + c] += s * j[c];
            acc[15 + c] += j[c] * r; acc[19 + c] += j[c] * g;
        }
        acc[8] += j[0] * j[0]; acc[9] += j[0] * j[1]; acc[10] += j[0] * j[2];
        acc[11] += j[1] * j[1]; acc[12] += j[1] * j[2]; acc[13] += j[2] * j[2];
        acc[14] += s * r; acc[18] += s * g;
    }
    __syncthreads();
    for (int k = 0; k < FIT_SUMS; k++) shared[k * b + t] = acc[k];
    block_sums(shared, FIT_SUMS);
    if (t < FIT_SUMS) out[i * FIT_SUMS + t] = shared[t * b];
}
"""

#: Threads per block of :func:`score`; a power of two for its reduction.
SCORE_THREADS = 256
#: Threads per block of :func:`fit_moments`: a power of two, and fewer than
#: :data:`SCORE_THREADS` because each holds 22 running sums.
FIT_THREADS = 128
#: Candidates per pattern, and sums per pattern, of :func:`fit_moments`.
FIT_CANDIDATES, FIT_SUMS = 7, 22


def available(device) -> bool:
    """Whether the fused kernels can run on *device*."""
    return torch.device(device).type == "cuda" and importlib.util.find_spec("cupy") is not None


@functools.cache
def _module(device_index: int):
    import cupy

    with cupy.cuda.Device(device_index):
        return cupy.RawModule(code=_SOURCE, options=("--use_fast_math",))


def _cupy(tensor: torch.Tensor):
    import cupy

    return cupy.from_dlpack(tensor.contiguous())


def _launch(name, grid, block, arguments, shared_mem=0):
    import cupy

    device = torch.cuda.current_device()
    stream = cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    with cupy.cuda.Device(device), stream:
        _module(device).get_function(name)(grid, block, arguments, shared_mem=shared_mem)


def project(
    texture: torch.Tensor, rotations: torch.Tensor, bases: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    """Patterns ``(M, pixels)``: *texture* ``(1, channels, n, n)`` sampled at
    *directions* ``(pixels, 3)`` turned by *rotations* ``(M, 3, 3)``, each from the
    hemisphere pair starting at its channel ``bases[m]``."""
    import cupy

    m, pixels = len(rotations), len(directions)
    out = torch.empty(m, pixels, device=rotations.device, dtype=torch.float32)
    count = m * pixels
    threads = 256
    texture = texture.contiguous()
    _launch(
        "project",
        ((count + threads - 1) // threads,),
        (threads,),
        (
            _cupy(texture),
            cupy.int32(texture.shape[-1]),
            _cupy(rotations.float()),
            _cupy(bases.to(torch.int32)),
            _cupy(directions.float()),
            cupy.int32(pixels),
            cupy.int64(count),
            _cupy(out),
        ),
    )
    return out


def score(
    texture: torch.Tensor,
    rotations: torch.Tensor,
    bases: torch.Tensor,
    directions: torch.Tensor,
    rows: torch.Tensor,
    per_row: int,
) -> torch.Tensor:
    """Scores ``(M,)``: the normalized cross-correlation of each rotation's
    pattern (as :func:`project`) with ``rows[m // per_row]``, rows being
    zero-mean, unit-norm ``(N, pixels)``."""
    import cupy

    m = len(rotations)
    out = torch.empty(m, device=rotations.device, dtype=torch.float32)
    texture = texture.contiguous()
    _launch(
        "score",
        (m,),
        (SCORE_THREADS,),
        (
            _cupy(texture),
            cupy.int32(texture.shape[-1]),
            _cupy(rotations.float()),
            _cupy(bases.to(torch.int32)),
            _cupy(directions.float()),
            cupy.int32(len(directions)),
            _cupy(rows.float()),
            cupy.int32(per_row),
            _cupy(out),
        ),
        shared_mem=3 * SCORE_THREADS * 4,
    )
    return out


def score_ghost(
    texture: torch.Tensor,
    rotations: torch.Tensor,
    bases: torch.Tensor,
    directions: torch.Tensor,
    rows: torch.Tensor,
    per_row: int,
    ghost: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`score`, and each normalized pattern's component along *ghost*
    ``(pixels,)`` (zero-mean, unit-norm; None for no ghost, then zero)."""
    import cupy

    m = len(rotations)
    scores = torch.empty(m, device=rotations.device, dtype=torch.float32)
    along = torch.zeros(m, device=rotations.device, dtype=torch.float32)
    texture = texture.contiguous()
    _launch(
        "score_ghost",
        (m,),
        (SCORE_THREADS,),
        (
            _cupy(texture),
            cupy.int32(texture.shape[-1]),
            _cupy(rotations.float()),
            _cupy(bases.to(torch.int32)),
            _cupy(directions.float()),
            cupy.int32(len(directions)),
            _cupy(rows.float()),
            cupy.int32(per_row),
            _cupy(ghost.float()) if ghost is not None else cupy.uint64(0),
            _cupy(scores),
            _cupy(along),
        ),
        shared_mem=4 * SCORE_THREADS * 4,
    )
    return scores, along


def fit_moments(
    texture: torch.Tensor,
    rotations: torch.Tensor,
    bases: torch.Tensor,
    directions: torch.Tensor,
    rows: torch.Tensor,
    ghost: torch.Tensor | None,
    step: float,
    offset: float = 0.0,
) -> torch.Tensor:
    """The orientation-only fit's raw sums ``(N, 22)`` for *rotations* ``(N * 7, 3,
    3)`` (each pattern's fit, then it turned by +*step* about x, y and z, then by
    -*step*), *rows* ``(N, pixels)`` and *ghost* ``(pixels,)`` or None, the fit's
    samples taken less *offset*; see the kernel."""
    import cupy

    n = len(rows)
    out = torch.empty(n, FIT_SUMS, device=rows.device, dtype=torch.float32)
    texture = texture.contiguous()
    _launch(
        "fit_moments",
        (n,),
        (FIT_THREADS,),
        (
            _cupy(texture),
            cupy.int32(texture.shape[-1]),
            _cupy(rotations.float()),
            _cupy(bases.to(torch.int32)),
            _cupy(directions.float()),
            cupy.int32(len(directions)),
            _cupy(rows.float()),
            _cupy(ghost.float()) if ghost is not None else cupy.uint64(0),
            cupy.float32(step),
            cupy.float32(offset),
            _cupy(out),
        ),
        shared_mem=FIT_SUMS * FIT_THREADS * 4,
    )
    return out
