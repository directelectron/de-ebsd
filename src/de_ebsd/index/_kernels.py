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

// sample_master and its gradient with respect to the direction v (before it is
// normalized), into dv[3]: through the bilinear lookup of the master's own x and
// y derivatives gx, gy (per texel, smooth, not the bilinear facets' slope), the
// Lambert square's map and the normalization. Clamped at the square's edge, the
// gradient along that axis is zero there.
__device__ __forceinline__ float sample_gradient(
    const float* __restrict__ texture, const float* __restrict__ gx,
    const float* __restrict__ gy, int base, int n, float vx, float vy, float vz, float* dv)
{
    float inverse = rsqrtf(fmaxf(vx * vx + vy * vy + vz * vz, 1e-24f));
    float x = vx * inverse, y = vy * inverse, z = vz * inverse;
    int channel = base + (z < 0.f ? 1 : 0);
    float abs_z = fabsf(z);
    float lx = 0.f, ly = 0.f;
    float dlx[3] = {0.f, 0.f, 0.f}, dly[3] = {0.f, 0.f, 0.f};
    if (abs_z < 1.f) {
        float root = sqrtf(fmaxf(2.f * (1.f - abs_z), 1e-30f));
        // d root / d z, for the hemisphere this z is on
        float droot = -(z >= 0.f ? 1.f : -1.f) / fmaxf(root, 1e-6f);
        float planar = fmaxf(x * x + y * y, 1e-24f);
        if (fabsf(y) <= fabsf(x)) {
            float sign = x >= 0.f ? 1.f : -1.f;
            float safe = fabsf(x) < 1e-12f ? 1e-12f : x;
            float angle = atanf(y / safe);
            lx = sign * root * SQRT_PI_OVER_2;
            ly = sign * root * TWO_OVER_SQRT_PI * angle;
            dlx[2] = sign * SQRT_PI_OVER_2 * droot;
            dly[0] = -sign * TWO_OVER_SQRT_PI * root * y / planar;
            dly[1] = sign * TWO_OVER_SQRT_PI * root * x / planar;
            dly[2] = sign * TWO_OVER_SQRT_PI * angle * droot;
        } else {
            float sign = y >= 0.f ? 1.f : -1.f;
            float safe = fabsf(y) < 1e-12f ? 1e-12f : y;
            float angle = atanf(x / safe);
            lx = sign * root * TWO_OVER_SQRT_PI * angle;
            ly = sign * root * SQRT_PI_OVER_2;
            dlx[0] = sign * TWO_OVER_SQRT_PI * root * y / planar;
            dlx[1] = -sign * TWO_OVER_SQRT_PI * root * x / planar;
            dlx[2] = sign * TWO_OVER_SQRT_PI * angle * droot;
            dly[2] = sign * SQRT_PI_OVER_2 * droot;
        }
    }
    float last = (float)(n - 1);
    float scale = 0.5f * last / SQRT_PI_HALF;
    float ux = (lx / SQRT_PI_HALF + 1.f) * 0.5f * last;
    float uy = (ly / SQRT_PI_HALF + 1.f) * 0.5f * last;
    float px = fminf(fmaxf(ux, 0.f), last), py = fminf(fmaxf(uy, 0.f), last);
    float kx = (ux > 0.f && ux < last) ? scale : 0.f;
    float ky = (uy > 0.f && uy < last) ? scale : 0.f;
    int x0 = (int)floorf(px), y0 = (int)floorf(py);
    int x1 = min(x0 + 1, n - 1), y1 = min(y0 + 1, n - 1);
    float fx = px - x0, fy = py - y0;
    size_t offset = (size_t)channel * n * n;
    int i00 = y0 * n + x0, i01 = y0 * n + x1, i10 = y1 * n + x0, i11 = y1 * n + x1;
    float w00 = (1.f - fx) * (1.f - fy), w01 = fx * (1.f - fy);
    float w10 = (1.f - fx) * fy, w11 = fx * fy;
    const float* t = texture + offset;
    const float* tx = gx + offset;
    const float* ty = gy + offset;
    float s = t[i00] * w00 + t[i01] * w01 + t[i10] * w10 + t[i11] * w11;
    float sx = (tx[i00] * w00 + tx[i01] * w01 + tx[i10] * w10 + tx[i11] * w11) * kx;
    float sy = (ty[i00] * w00 + ty[i01] * w01 + ty[i10] * w10 + ty[i11] * w11) * ky;
    float du[3];
    for (int c = 0; c < 3; c++) du[c] = sx * dlx[c] + sy * dly[c];
    float along = x * du[0] + y * du[1] + z * du[2];
    dv[0] = (du[0] - x * along) * inverse;
    dv[1] = (du[1] - y * along) * inverse;
    dv[2] = (du[2] - z * along) * inverse;
    return s;
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

// ---- one block per pattern: the stepped refine and the orientation fit -------
// Each runs every level or iteration of its pattern in one launch: the block's
// threads sample and reduce over the pixels, thread 0 does the pattern's own
// arithmetic between the reductions (in double), so nothing per pattern goes
// through the host or through any other kernel.

#define FIT_THREADS 128
#define MAX_LEVELS 8
#define MAX_MULTIPLES 4

// m = a @ b, 3x3 row-major
__device__ void mat_mul(const float* a, const float* b, float* m)
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            m[i * 3 + j] =
                a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
}

// The rotation by vector (x, y, z) radians, Rodrigues with the series limits of
// rotation_vector_to_matrix at small angles.
__device__ void rotation(double x, double y, double z, float* r)
{
    double t2 = x * x + y * y + z * z, t = sqrt(fmax(t2, 1e-12));
    double a = t2 < 1e-12 ? 1.0 : sin(t) / t;
    double b = t2 < 1e-12 ? 0.5 : (1.0 - cos(t)) / fmax(t2, 1e-12);
    double k[9] = {0, -z, y, z, 0, -x, -y, x, 0};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double kk = k[i * 3] * k[j] + k[i * 3 + 1] * k[3 + j] + k[i * 3 + 2] * k[6 + j];
            r[i * 3 + j] = (float)((i == j ? 1.0 : 0.0) + a * k[i * 3 + j] + b * kk);
        }
}

// m = g @ (I - skew(q / sqrt 2)), then @ map if one is given: the fit's candidate
// for rotation numbers q (the trace-free basis's three rotations).
__device__ void turned(const float* g, double q0, double q1, double q2, const float* map, float* m)
{
    const double h = 0.70710678118654752;
    double wx = q0 * h, wy = q1 * h, wz = q2 * h;
    float e[9] = {1.f, (float)wz, (float)-wy,
                  (float)-wz, 1.f, (float)wx,
                  (float)wy, (float)-wx, 1.f};
    if (map) {
        float t[9];
        mat_mul(g, e, t);
        mat_mul(t, map, m);
    } else {
        mat_mul(g, e, m);
    }
}

// refine_stepped: LiveIndexer.refine on the fused path. For each step of steps
// (radians): the seven stencil candidates g @ R(stencil * step), each's normalized
// cross-correlation with its row, and g moved to the parabola's vertex per axis
// (parabola_offset). g_out, score_out: the result and its last level's score.
extern "C" __global__ void refine_stepped(
    const float* __restrict__ texture, int n,
    const float* __restrict__ g_in, const int* __restrict__ bases,
    const float* __restrict__ directions, int pixels,
    const float* __restrict__ rows, const float* __restrict__ steps, int levels, float floor_,
    float* __restrict__ g_out, float* __restrict__ score_out)
{
    __shared__ float shared[21 * FIT_THREADS];
    __shared__ float g[9], candidates[7 * 9];
    long long i = blockIdx.x;
    int t = threadIdx.x, b = blockDim.x;
    if (t < 9) g[t] = g_in[i * 9 + t];
    int base = bases[i];
    const float* row = rows + i * (long long)pixels;
    float score = 0.f;
    const int stencil[7][3] = {
        {0, 0, 0}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (int level = 0; level < levels; level++) {
        double step = steps[level];
        __syncthreads();
        if (t == 0) {
            for (int c = 0; c < 7; c++) {
                float r[9];
                rotation(stencil[c][0] * step, stencil[c][1] * step, stencil[c][2] * step, r);
                mat_mul(g, r, candidates + c * 9);
            }
        }
        __syncthreads();
        float acc[21];
        for (int k = 0; k < 21; k++) acc[k] = 0.f;
        for (int p = t; p < pixels; p += b) {
            const float* d = directions + p * 3;
            float r = row[p];
            for (int c = 0; c < 7; c++) {
                float s = sample_rotated(texture, base, n, candidates + c * 9, d);
                acc[c] += s; acc[7 + c] += s * s; acc[14 + c] += s * r;
            }
        }
        for (int k = 0; k < 21; k++) shared[k * b + t] = acc[k];
        block_sums(shared, 21);
        if (t == 0) {
            double sc[7];
            for (int c = 0; c < 7; c++) {
                double sum = shared[c * b], squares = shared[(7 + c) * b];
                double product = shared[(14 + c) * b];
                sc[c] = product / sqrt(fmax(squares - sum * sum / pixels, 1e-12));
            }
            double offset[3];
            for (int axis = 0; axis < 3; axis++) {
                double center = sc[0], plus = sc[1 + 2 * axis], minus = sc[2 + 2 * axis];
                double curvature = plus - 2 * center + minus;
                if (curvature < 0) {
                    double vertex = 0.5 * (minus - plus) / fmin(curvature, -(double)floor_);
                    offset[axis] = fmin(fmax(vertex, -1.0), 1.0);
                } else {
                    offset[axis] = (fmax(plus, minus) > center) ? (plus > minus ? 1.0 : -1.0) : 0.0;
                }
            }
            float r[9], moved[9];
            rotation(offset[0] * step, offset[1] * step, offset[2] * step, r);
            mat_mul(g, r, moved);
            for (int k = 0; k < 9; k++) g[k] = moved[k];
            score = (float)sc[0];
        }
    }
    __syncthreads();
    if (t < 9) g_out[i * 9 + t] = g[t];
    if (t == 0) score_out[i] = score;
}

// fit_orientations: refine_deformation without the strains on the fused path:
// Gauss-Newton over the three rotations, its Jacobian by central differences of
// `step`, each step capped at max_step and tried at each of multiples, keeping
// the best scored. The row is taken off the ghost as it is read; every score is
// the off-ghost one. The fit's rotation is the polar part of (I - beta)^-1: for a
// rotation beta = skew(w), the turn about w by atan|w|. A fit ending below its
// seed's score less margin is refused (g and score as given). maps: per-pattern
// center maps, or null.
extern "C" __global__ void fit_orientations(
    const float* __restrict__ texture, int n,
    const float* __restrict__ g_in, const int* __restrict__ bases, const float* __restrict__ maps,
    const float* __restrict__ directions, int pixels,
    const float* __restrict__ rows, const float* __restrict__ ghost,
    int iterations, float step, float max_step, const float* __restrict__ multiples, int count,
    float damping, float margin, float offset, float floor_,
    const float* __restrict__ grad_x, const float* __restrict__ grad_y,
    float* __restrict__ g_out, float* __restrict__ score_out, int* __restrict__ refused_out)
{
    __shared__ float shared[22 * FIT_THREADS];
    __shared__ float g[9], map[9], candidates[7 * 9];
    __shared__ double state[8];  // p[3], seed, row . ghost, 1 / its length off it, current
    long long i = blockIdx.x;
    int t = threadIdx.x, b = blockDim.x;
    if (t < 9) { g[t] = g_in[i * 9 + t]; if (maps) map[t] = maps[i * 9 + t]; }
    const float* mp = maps ? map : (const float*)0;
    int base = bases[i];
    const float* row = rows + i * (long long)pixels;

    // the row off the ghost: (row - a ghost) / |row - a ghost|, |...| = sqrt(1 - a^2)
    float a_sum = 0.f;
    if (ghost)
        for (int p = t; p < pixels; p += b) a_sum += row[p] * ghost[p];
    shared[t] = a_sum;
    block_sums(shared, 1);
    if (t == 0) {
        double a = ghost ? (double)shared[0] : 0.0;
        state[0] = state[1] = state[2] = 0.0;
        state[3] = -1e30;
        state[4] = a;
        state[5] = 1.0 / sqrt(fmax(1.0 - a * a, 1e-12));
    }
    __syncthreads();
    const float a = (float)state[4], inverse_length = (float)state[5];

    for (int iteration = 0; iteration < iterations; iteration++) {
        if (t == 0) {
            const double o[7][3] = {
                {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}};
            for (int c = 0; c < (grad_x ? 1 : 7); c++)
                turned(g, state[0] + step * o[c][0], state[1] + step * o[c][1],
                       state[2] + step * o[c][2], mp, candidates + c * 9);
            if (grad_x) {
                // dv/dq_k = -g skew(e_k) / sqrt 2 (map d): q's derivative of the
                // candidate's direction, the same for any q
                const float h = 0.70710678118654752f;
                for (int k = 0; k < 3; k++) {
                    float e[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                    int a1 = (k + 1) % 3, a2 = (k + 2) % 3;  // skew(e_k)[a1][a2] = -1
                    e[a1 * 3 + a2] = h;
                    e[a2 * 3 + a1] = -h;
                    mat_mul(g, e, candidates + (1 + k) * 9);
                }
            }
        }
        __syncthreads();
        float acc[22];
        for (int k = 0; k < 22; k++) acc[k] = 0.f;
        float half_step = 0.5f / step;
        for (int p = t; p < pixels; p += b) {
            const float* d = directions + p * 3;
            float gh = ghost ? ghost[p] : 0.f;
            float r = (row[p] - a * gh) * inverse_length;
            float s, j[3];
            if (grad_x) {
                const float* m = candidates;
                float dv[3];
                s = sample_gradient(texture, grad_x, grad_y, base, n,
                                    m[0] * d[0] + m[1] * d[1] + m[2] * d[2],
                                    m[3] * d[0] + m[4] * d[1] + m[5] * d[2],
                                    m[6] * d[0] + m[7] * d[1] + m[8] * d[2], dv) - offset;
                float u[3] = {d[0], d[1], d[2]};
                if (mp) {
                    for (int c = 0; c < 3; c++)
                        u[c] = mp[c * 3] * d[0] + mp[c * 3 + 1] * d[1] + mp[c * 3 + 2] * d[2];
                }
                for (int k = 0; k < 3; k++) {
                    const float* w = candidates + (1 + k) * 9;
                    j[k] = dv[0] * (w[0] * u[0] + w[1] * u[1] + w[2] * u[2])
                         + dv[1] * (w[3] * u[0] + w[4] * u[1] + w[5] * u[2])
                         + dv[2] * (w[6] * u[0] + w[7] * u[1] + w[8] * u[2]);
                }
            } else {
                s = sample_rotated(texture, base, n, candidates, d) - offset;
                for (int c = 0; c < 3; c++) {
                    float plus = sample_rotated(texture, base, n, candidates + (1 + c) * 9, d);
                    float minus = sample_rotated(texture, base, n, candidates + (4 + c) * 9, d);
                    j[c] = (plus - minus) * half_step;
                }
            }
            acc[0] += s; acc[1] += s * s;
            for (int c = 0; c < 3; c++) {
                acc[2 + c] += j[c]; acc[5 + c] += s * j[c];
                acc[15 + c] += j[c] * r; acc[19 + c] += j[c] * gh;
            }
            acc[8] += j[0] * j[0]; acc[9] += j[0] * j[1]; acc[10] += j[0] * j[2];
            acc[11] += j[1] * j[1]; acc[12] += j[1] * j[2]; acc[13] += j[2] * j[2];
            acc[14] += s * r; acc[18] += s * gh;
        }
        __syncthreads();
        for (int k = 0; k < 22; k++) shared[k * b + t] = acc[k];
        block_sums(shared, 22);
        if (t == 0) {
            double S[22];
            for (int k = 0; k < 22; k++) S[k] = shared[k * b];
            double P = pixels;
            double sigma2 = fmax(S[1] - S[0] * S[0] / P, (double)floor_), sigma = sqrt(sigma2);
            double uj[3], jr[3], jg[3];
            for (int c = 0; c < 3; c++) uj[c] = S[5 + c] - S[0] * S[2 + c] / P;
            const int jidx[3][3] = {{8, 9, 10}, {9, 11, 12}, {10, 12, 13}};
            double normal[3][3];
            for (int x = 0; x < 3; x++)
                for (int y = 0; y < 3; y++)
                    normal[x][y] = (S[jidx[x][y]] - S[2 + x] * S[2 + y] / P) / sigma2
                                   - uj[x] * uj[y] / (sigma2 * sigma2);
            double fit_row = S[14] / sigma, fit_ghost = S[18] / sigma;
            for (int c = 0; c < 3; c++) {
                jr[c] = S[15 + c] / sigma - S[14] * uj[c] / (sigma2 * sigma);
                jg[c] = S[19 + c] / sigma - S[18] * uj[c] / (sigma2 * sigma);
            }
            double scale = fit_row / (1.0 - fit_ghost * fit_ghost);
            double current = scale * sqrt(fmax(1.0 - fit_ghost * fit_ghost, 0.0));
            if (iteration == 0) state[3] = current;
            double trace = 0.0;
            for (int x = 0; x < 3; x++)
                for (int y = 0; y < 3; y++) normal[x][y] -= jg[x] * jg[y];
            for (int x = 0; x < 3; x++) trace += normal[x][x];
            for (int x = 0; x < 3; x++) normal[x][x] += damping * trace / 3.0;
            double gradient[3];
            double safe = fmax(scale, (double)floor_);
            for (int c = 0; c < 3; c++)
                gradient[c] = (jr[c] + scale * jg[c] * fit_ghost) / safe;
            // the 3x3 solve by the adjugate; a singular system makes no step
            double (*A)[3] = normal;
            double c0[3] = {A[1][1] * A[2][2] - A[1][2] * A[2][1],
                            A[1][2] * A[2][0] - A[1][0] * A[2][2],
                            A[1][0] * A[2][1] - A[1][1] * A[2][0]};
            double c1[3] = {A[2][1] * A[0][2] - A[2][2] * A[0][1],
                            A[2][2] * A[0][0] - A[2][0] * A[0][2],
                            A[2][0] * A[0][1] - A[2][1] * A[0][0]};
            double c2[3] = {A[0][1] * A[1][2] - A[0][2] * A[1][1],
                            A[0][2] * A[1][0] - A[0][0] * A[1][2],
                            A[0][0] * A[1][1] - A[0][1] * A[1][0]};
            double det = A[0][0] * c0[0] + A[0][1] * c0[1] + A[0][2] * c0[2];
            double largest = fmax(fabs(A[0][0]), fmax(fabs(A[1][1]), fabs(A[2][2])));
            double change[3] = {0, 0, 0};
            if (fabs(det) > 1e-12 * largest * largest * largest) {
                for (int x = 0; x < 3; x++)
                    change[x] =
                        (c0[x] * gradient[0] + c1[x] * gradient[1] + c2[x] * gradient[2]) / det;
                if (!(isfinite(change[0]) && isfinite(change[1]) && isfinite(change[2])))
                    change[0] = change[1] = change[2] = 0.0;
            }
            double reach = fmax(fabs(change[0]), fmax(fabs(change[1]), fabs(change[2])));
            double cap = max_step / fmax(reach, (double)max_step);
            for (int x = 0; x < 3; x++) change[x] *= cap;
            state[6] = current;
            for (int m = 0; m < count; m++)
                turned(g, state[0] + multiples[m] * change[0], state[1] + multiples[m] * change[1],
                       state[2] + multiples[m] * change[2], mp, candidates + m * 9);
            // keep the step in shared for the choice below
            shared[22 * b - 3] = (float)change[0];
            shared[22 * b - 2] = (float)change[1];
            shared[22 * b - 1] = (float)change[2];
        }
        __syncthreads();
        double ch0 = shared[22 * b - 3], ch1 = shared[22 * b - 2], ch2 = shared[22 * b - 1];
        __syncthreads();
        float sums[4 * MAX_MULTIPLES];
        for (int k = 0; k < 4 * MAX_MULTIPLES; k++) sums[k] = 0.f;
        for (int p = t; p < pixels; p += b) {
            const float* d = directions + p * 3;
            float gh = ghost ? ghost[p] : 0.f;
            float r = (row[p] - a * gh) * inverse_length;
            for (int m = 0; m < count; m++) {
                float s = sample_rotated(texture, base, n, candidates + m * 9, d);
                sums[m] += s; sums[MAX_MULTIPLES + m] += s * s;
                sums[2 * MAX_MULTIPLES + m] += s * r; sums[3 * MAX_MULTIPLES + m] += s * gh;
            }
        }
        for (int k = 0; k < 4 * MAX_MULTIPLES; k++) shared[k * b + t] = sums[k];
        block_sums(shared, 4 * MAX_MULTIPLES);
        if (t == 0) {
            double best = state[6], chosen = 0.0;
            if (!isfinite(best)) best = -1e30;
            for (int m = 0; m < count; m++) {
                double sum = shared[m * b], squares = shared[(MAX_MULTIPLES + m) * b];
                double inv = 1.0 / sqrt(fmax(squares - sum * sum / pixels, 1e-12));
                double sc = shared[(2 * MAX_MULTIPLES + m) * b] * inv;
                double along = shared[(3 * MAX_MULTIPLES + m) * b] * inv;
                double score = sc / sqrt(fmax(1.0 - along * along, 1e-12));
                if (isfinite(score) && score > best) { best = score; chosen = multiples[m]; }
            }
            state[0] += chosen * ch0; state[1] += chosen * ch1; state[2] += chosen * ch2;
        }
        __syncthreads();
    }

    // the final score, and the rotation
    if (t == 0) turned(g, state[0], state[1], state[2], mp, candidates);
    __syncthreads();
    float sums[4] = {0.f, 0.f, 0.f, 0.f};
    for (int p = t; p < pixels; p += b) {
        float gh = ghost ? ghost[p] : 0.f;
        float r = (row[p] - a * gh) * inverse_length;
        float s = sample_rotated(texture, base, n, candidates, directions + p * 3);
        sums[0] += s; sums[1] += s * s; sums[2] += s * r; sums[3] += s * gh;
    }
    for (int k = 0; k < 4; k++) shared[k * b + t] = sums[k];
    block_sums(shared, 4);
    if (t == 0) {
        double variance = shared[b] - (double)shared[0] * shared[0] / pixels;
        double inv = 1.0 / sqrt(fmax(variance, 1e-12));
        double along = shared[3 * b] * inv;
        double score = shared[2 * b] * inv / sqrt(fmax(1.0 - along * along, 1e-12));
        double seed = state[3];
        bool refused = iterations > 0 && !(score >= seed - margin);
        refused_out[i] = refused ? 1 : 0;
        if (refused) {
            for (int k = 0; k < 9; k++) g_out[i * 9 + k] = g[k];
            score_out[i] = (float)seed;
        } else {
            const double h = 0.70710678118654752;
            double w[3] = {state[0] * h, state[1] * h, state[2] * h};
            double length = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
            double angle = atan(length), k = length > 1e-12 ? angle / length : 1.0;
            float r[9], fitted[9];
            rotation(w[0] * k, w[1] * k, w[2] * k, r);  // the polar part of (I - beta)^-1
            float rt[9] = {r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]};
            mat_mul(g, rt, fitted);
            for (int q = 0; q < 9; q++) g_out[i * 9 + q] = fitted[q];
            score_out[i] = (float)score;
        }
    }
}
// fit_passes: the orientation fit as a schedule of Gauss-Newton passes, the
// Jacobian from the master's gradient, no line search and no final pass. Pass k
// is scored at full resolution or at the coarse one (coarse[k]) and moves by
// alpha[k] times its step: every pass's moments give the score where it stands
// for nothing, so the fit keeps the best-scored full-resolution point it visited
// (the first pass is at full resolution, from the given orientation: the seed).
extern "C" __global__ void fit_passes(
    const float* __restrict__ texture, int n,
    const float* __restrict__ g_in, const int* __restrict__ bases,
    const float* __restrict__ maps,
    const float* __restrict__ directions, int pixels,
    const float* __restrict__ rows, const float* __restrict__ ghost,
    const float* __restrict__ coarse_directions, int coarse_pixels,
    const float* __restrict__ coarse_rows, const float* __restrict__ coarse_ghost,
    const int* __restrict__ coarse, const float* __restrict__ alpha, int passes,
    float max_step, float damping, float margin, float offset, float floor_,
    const float* __restrict__ grad_x, const float* __restrict__ grad_y,
    float* __restrict__ g_out, float* __restrict__ score_out, int* __restrict__ refused_out)
{
    __shared__ float shared[22 * FIT_THREADS];
    __shared__ float g[9], map[9], candidates[4 * 9];
    // p[3], best p[3], seed, best, along full, 1/length full, along coarse, 1/length coarse
    __shared__ double state[12];
    long long i = blockIdx.x;
    int t = threadIdx.x, b = blockDim.x;
    if (t < 9) {
        g[t] = g_in[i * 9 + t];
        if (maps) map[t] = maps[i * 9 + t];
    }
    const float* mp = maps ? map : (const float*)0;
    int base = bases[i];
    const float* full_row = rows + i * (long long)pixels;
    const float* coarse_row = coarse_rows + i * (long long)coarse_pixels;

    // each row off its ghost: (row - a ghost) / sqrt(1 - a^2)
    float a_full = 0.f, a_coarse = 0.f;
    if (ghost)
        for (int p = t; p < pixels; p += b) a_full += full_row[p] * ghost[p];
    if (coarse_ghost)
        for (int p = t; p < coarse_pixels; p += b) a_coarse += coarse_row[p] * coarse_ghost[p];
    shared[t] = a_full;
    shared[b + t] = a_coarse;
    block_sums(shared, 2);
    if (t == 0) {
        for (int k = 0; k < 6; k++) state[k] = 0.0;
        state[6] = state[7] = -1e30;
        double af = ghost ? (double)shared[0] : 0.0, ac = coarse_ghost ? (double)shared[b] : 0.0;
        state[8] = af;
        state[9] = 1.0 / sqrt(fmax(1.0 - af * af, 1e-12));
        state[10] = ac;
        state[11] = 1.0 / sqrt(fmax(1.0 - ac * ac, 1e-12));
        // dv/dq_k = -g skew(e_k) / sqrt 2 (map d), the same for any q
        const float h = 0.70710678118654752f;
        for (int k = 0; k < 3; k++) {
            float e[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            int a1 = (k + 1) % 3, a2 = (k + 2) % 3;
            e[a1 * 3 + a2] = h;
            e[a2 * 3 + a1] = -h;
            mat_mul(g, e, candidates + (1 + k) * 9);
        }
    }
    __syncthreads();

    for (int pass = 0; pass < passes; pass++) {
        bool low = coarse[pass] != 0;
        const float* d_all = low ? coarse_directions : directions;
        const float* row = low ? coarse_row : full_row;
        const float* gh_all = low ? coarse_ghost : ghost;
        int count = low ? coarse_pixels : pixels;
        float a = (float)state[low ? 10 : 8], inverse_length = (float)state[low ? 11 : 9];
        if (t == 0) turned(g, state[0], state[1], state[2], mp, candidates);
        __syncthreads();
        float acc[22];
        for (int k = 0; k < 22; k++) acc[k] = 0.f;
        for (int p = t; p < count; p += b) {
            const float* d = d_all + p * 3;
            float gh = gh_all ? gh_all[p] : 0.f;
            float r = (row[p] - a * gh) * inverse_length;
            const float* m = candidates;
            float dv[3];
            float s = sample_gradient(texture, grad_x, grad_y, base, n,
                                      m[0] * d[0] + m[1] * d[1] + m[2] * d[2],
                                      m[3] * d[0] + m[4] * d[1] + m[5] * d[2],
                                      m[6] * d[0] + m[7] * d[1] + m[8] * d[2], dv) - offset;
            float u[3] = {d[0], d[1], d[2]};
            if (mp) {
                for (int c = 0; c < 3; c++)
                    u[c] = mp[c * 3] * d[0] + mp[c * 3 + 1] * d[1] + mp[c * 3 + 2] * d[2];
            }
            float j[3];
            for (int k = 0; k < 3; k++) {
                const float* w = candidates + (1 + k) * 9;
                j[k] = dv[0] * (w[0] * u[0] + w[1] * u[1] + w[2] * u[2])
                     + dv[1] * (w[3] * u[0] + w[4] * u[1] + w[5] * u[2])
                     + dv[2] * (w[6] * u[0] + w[7] * u[1] + w[8] * u[2]);
            }
            acc[0] += s; acc[1] += s * s;
            for (int c = 0; c < 3; c++) {
                acc[2 + c] += j[c]; acc[5 + c] += s * j[c];
                acc[15 + c] += j[c] * r; acc[19 + c] += j[c] * gh;
            }
            acc[8] += j[0] * j[0]; acc[9] += j[0] * j[1]; acc[10] += j[0] * j[2];
            acc[11] += j[1] * j[1]; acc[12] += j[1] * j[2]; acc[13] += j[2] * j[2];
            acc[14] += s * r; acc[18] += s * gh;
        }
        __syncthreads();
        for (int k = 0; k < 22; k++) shared[k * b + t] = acc[k];
        block_sums(shared, 22);
        if (t == 0) {
            double S[22];
            for (int k = 0; k < 22; k++) S[k] = shared[k * b];
            double P = count;
            double sigma2 = fmax(S[1] - S[0] * S[0] / P, (double)floor_), sigma = sqrt(sigma2);
            double uj[3], jr[3], jg[3];
            for (int c = 0; c < 3; c++) uj[c] = S[5 + c] - S[0] * S[2 + c] / P;
            const int jidx[3][3] = {{8, 9, 10}, {9, 11, 12}, {10, 12, 13}};
            double A[3][3];
            for (int x = 0; x < 3; x++)
                for (int y = 0; y < 3; y++)
                    A[x][y] = (S[jidx[x][y]] - S[2 + x] * S[2 + y] / P) / sigma2
                              - uj[x] * uj[y] / (sigma2 * sigma2);
            double fit_row = S[14] / sigma, fit_ghost = S[18] / sigma;
            for (int c = 0; c < 3; c++) {
                jr[c] = S[15 + c] / sigma - S[14] * uj[c] / (sigma2 * sigma);
                jg[c] = S[19 + c] / sigma - S[18] * uj[c] / (sigma2 * sigma);
            }
            double scale = fit_row / (1.0 - fit_ghost * fit_ghost);
            double current = scale * sqrt(fmax(1.0 - fit_ghost * fit_ghost, 0.0));
            if (!low) {
                if (pass == 0) state[6] = current;
                if (isfinite(current) && current > state[7]) {
                    state[7] = current;
                    state[3] = state[0]; state[4] = state[1]; state[5] = state[2];
                }
            }
            double trace = 0.0;
            for (int x = 0; x < 3; x++)
                for (int y = 0; y < 3; y++) A[x][y] -= jg[x] * jg[y];
            for (int x = 0; x < 3; x++) trace += A[x][x];
            for (int x = 0; x < 3; x++) A[x][x] += damping * trace / 3.0;
            double gradient[3];
            double safe = fmax(scale, (double)floor_);
            for (int c = 0; c < 3; c++)
                gradient[c] = (jr[c] + scale * jg[c] * fit_ghost) / safe;
            double c0[3] = {A[1][1] * A[2][2] - A[1][2] * A[2][1],
                            A[1][2] * A[2][0] - A[1][0] * A[2][2],
                            A[1][0] * A[2][1] - A[1][1] * A[2][0]};
            double c1[3] = {A[2][1] * A[0][2] - A[2][2] * A[0][1],
                            A[2][2] * A[0][0] - A[2][0] * A[0][2],
                            A[2][0] * A[0][1] - A[2][1] * A[0][0]};
            double c2[3] = {A[0][1] * A[1][2] - A[0][2] * A[1][1],
                            A[0][2] * A[1][0] - A[0][0] * A[1][2],
                            A[0][0] * A[1][1] - A[0][1] * A[1][0]};
            double det = A[0][0] * c0[0] + A[0][1] * c0[1] + A[0][2] * c0[2];
            double largest = fmax(fabs(A[0][0]), fmax(fabs(A[1][1]), fabs(A[2][2])));
            double change[3] = {0, 0, 0};
            if (fabs(det) > 1e-12 * largest * largest * largest) {
                for (int x = 0; x < 3; x++)
                    change[x] =
                        (c0[x] * gradient[0] + c1[x] * gradient[1] + c2[x] * gradient[2]) / det;
                if (!(isfinite(change[0]) && isfinite(change[1]) && isfinite(change[2])))
                    change[0] = change[1] = change[2] = 0.0;
            }
            double reach = fmax(fabs(change[0]), fmax(fabs(change[1]), fabs(change[2])));
            double cap = alpha[pass] * max_step / fmax(reach, (double)max_step);
            for (int x = 0; x < 3; x++) state[x] += change[x] * cap;
        }
        __syncthreads();
    }

    if (t == 0) {
        double seed = state[6], best = state[7];
        bool refused = !(best >= seed - margin);
        refused_out[i] = refused ? 1 : 0;
        if (refused) {
            for (int k = 0; k < 9; k++) g_out[i * 9 + k] = g[k];
            score_out[i] = (float)seed;
        } else {
            const double h = 0.70710678118654752;
            double w[3] = {state[3] * h, state[4] * h, state[5] * h};
            double length = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
            double k = length > 1e-12 ? atan(length) / length : 1.0;
            float r[9], fitted[9];
            rotation(w[0] * k, w[1] * k, w[2] * k, r);  // the polar part of (I - beta)^-1
            float rt[9] = {r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]};
            mat_mul(g, rt, fitted);
            for (int q = 0; q < 9; q++) g_out[i * 9 + q] = fitted[q];
            score_out[i] = (float)best;
        }
    }
}
"""

#: Threads per block of :func:`score`; a power of two for its reduction.
SCORE_THREADS = 256
#: Threads per block of :func:`refine_stepped` and :func:`fit_orientations`,
#: one block per pattern (FIT_THREADS in the source).
FIT_THREADS = 128
#: The most refine levels and line-search multiples the kernels take.
MAX_LEVELS, MAX_MULTIPLES = 8, 4


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


def refine_stepped(texture, g, bases, directions, rows, steps, floor):
    """:meth:`LiveIndexer.refine` in one launch, one block per pattern: ``(g,
    score)`` for seeds *g* ``(N, 3, 3)``, *rows* ``(N, pixels)`` zero-mean and
    unit-norm, and *steps* (radians), at most :data:`MAX_LEVELS`."""
    import cupy

    n = len(g)
    if len(steps) > MAX_LEVELS:
        raise ValueError(f"at most {MAX_LEVELS} refine levels")
    out = torch.empty(n, 3, 3, device=g.device, dtype=torch.float32)
    score = torch.empty(n, device=g.device, dtype=torch.float32)
    texture = texture.contiguous()
    steps = torch.as_tensor(steps, dtype=torch.float32, device=g.device)
    _launch(
        "refine_stepped",
        (n,),
        (FIT_THREADS,),
        (
            _cupy(texture),
            cupy.int32(texture.shape[-1]),
            _cupy(g.float()),
            _cupy(bases.to(torch.int32)),
            _cupy(directions.float()),
            cupy.int32(len(directions)),
            _cupy(rows.float()),
            _cupy(steps),
            cupy.int32(len(steps)),
            cupy.float32(floor),
            _cupy(out),
            _cupy(score),
        ),
    )
    return out, score


def fit_orientations(
    texture,
    g,
    bases,
    directions,
    rows,
    ghost,
    maps,
    *,
    iterations,
    step,
    max_step,
    multiples,
    damping,
    margin,
    offset,
    floor,
    gradients=None,
):
    """:func:`~de_ebsd.index.live.deformation.refine_deformation` without the
    strains in one launch, one block per pattern: ``(g, score, refused)``. *ghost*
    ``(pixels,)`` and *maps* ``(N, 3, 3)`` may be None; *offset* is the master's
    mean, taken off the fit's samples to keep its float32 sums small. With
    *gradients* ``(grad_x, grad_y)``, the master's derivatives along its columns
    and rows, the Jacobian is the analytic one; without, central differences of
    *step*."""
    import cupy

    n = len(g)
    if len(multiples) > MAX_MULTIPLES:
        raise ValueError(f"at most {MAX_MULTIPLES} line-search multiples")
    out = torch.empty(n, 3, 3, device=g.device, dtype=torch.float32)
    score = torch.empty(n, device=g.device, dtype=torch.float32)
    texture = texture.contiguous()
    multiples = torch.as_tensor(multiples, dtype=torch.float32, device=g.device)
    refused = torch.empty(n, device=g.device, dtype=torch.int32)
    _launch(
        "fit_orientations",
        (n,),
        (FIT_THREADS,),
        (
            _cupy(texture),
            cupy.int32(texture.shape[-1]),
            _cupy(g.float()),
            _cupy(bases.to(torch.int32)),
            _cupy(maps.float()) if maps is not None else cupy.uint64(0),
            _cupy(directions.float()),
            cupy.int32(len(directions)),
            _cupy(rows.float()),
            _cupy(ghost.float()) if ghost is not None else cupy.uint64(0),
            cupy.int32(iterations),
            cupy.float32(step),
            cupy.float32(max_step),
            _cupy(multiples),
            cupy.int32(len(multiples)),
            cupy.float32(damping),
            cupy.float32(margin),
            cupy.float32(offset),
            cupy.float32(floor),
            _cupy(gradients[0]) if gradients is not None else cupy.uint64(0),
            _cupy(gradients[1]) if gradients is not None else cupy.uint64(0),
            _cupy(out),
            _cupy(score),
            _cupy(refused),
        ),
    )
    return out, score, refused.bool()


def fit_passes(
    texture,
    g,
    bases,
    maps,
    full,
    coarse,
    schedule,
    *,
    max_step,
    damping,
    margin,
    offset,
    floor,
    gradients,
):
    """The orientation fit as a *schedule* of ``(coarse, alpha)`` passes, in one
    launch, one block per pattern: ``(g, score, refused)``. *full* and *coarse*
    are each ``(directions, rows, ghost)`` at the refine's resolution and the
    coarse one (*ghost* may be None); the first pass must be at full resolution.
    *gradients* are the master's ``(grad_x, grad_y)``."""
    import cupy

    if not schedule or schedule[0][0]:
        raise ValueError("the first pass is at full resolution")
    n = len(g)
    out = torch.empty(n, 3, 3, device=g.device, dtype=torch.float32)
    score = torch.empty(n, device=g.device, dtype=torch.float32)
    refused = torch.empty(n, device=g.device, dtype=torch.int32)
    texture = texture.contiguous()
    lows = torch.tensor([int(c) for c, _ in schedule], dtype=torch.int32, device=g.device)
    alphas = torch.tensor([float(a) for _, a in schedule], dtype=torch.float32, device=g.device)

    def optional(tensor):
        return _cupy(tensor.float()) if tensor is not None else cupy.uint64(0)

    (full_directions, full_rows, full_ghost), (low_directions, low_rows, low_ghost) = full, coarse
    _launch(
        "fit_passes",
        (n,),
        (FIT_THREADS,),
        (
            _cupy(texture),
            cupy.int32(texture.shape[-1]),
            _cupy(g.float()),
            _cupy(bases.to(torch.int32)),
            optional(maps),
            _cupy(full_directions.float()),
            cupy.int32(len(full_directions)),
            _cupy(full_rows.float()),
            optional(full_ghost),
            _cupy(low_directions.float()),
            cupy.int32(len(low_directions)),
            _cupy(low_rows.float()),
            optional(low_ghost),
            _cupy(lows),
            _cupy(alphas),
            cupy.int32(len(schedule)),
            cupy.float32(max_step),
            cupy.float32(damping),
            cupy.float32(margin),
            cupy.float32(offset),
            cupy.float32(floor),
            _cupy(gradients[0]),
            _cupy(gradients[1]),
            _cupy(out),
            _cupy(score),
            _cupy(refused),
        ),
    )
    return out, score, refused.bool()
