// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// GeometryLossOps kernels: ports of depth_loss.cu, normal_loss.cu and
// normal_consistency_loss.cu. MSL has no double, so the per-block statistics
// are float sums stored at the start of the CUDA double region, and the depth
// prior variance is accumulated around a per-image shift instead of as raw
// moments. The float prefix (finals) keeps the CUDA slots.

// Fast math may fold isfinite to true; test the exponent bits instead.
static bool geom_finite(const float x) {
    return (as_type<uint>(x) & 0x7f800000u) != 0x7f800000u;
}

struct GeomParams {
    device const float* normal; // rendered normal [3,H,W]; the prior normal for prior_depth
    device const float* depth;  // accumulated depth [H,W]
    device const float* alpha;
    device const float* target; // depth prior [H,W] or normal target [3,H,W]
    device const float* weight; // optional pixel weight
    device float* grad_normal;
    device float* grad_depth;
    device float* grad_alpha;
    device float* loss;
    device float* finals;
    device float* blocks; // stat k of block b at k * num_blocks + b
    int width, height;
    uint num_blocks;
    uint has_weight;
    float fx, fy, cx, cy;
    float loss_weight;
    float lambda_grad;
    float quantization_step;
    float floor_override;
    int anchor_model;
    float anchor_scale, anchor_shift, anchor_floor;
    float min_count, min_weight;
    uint prior; // depth-normal kernels: 0 consistency, 1 prior depth
};

constant constexpr int kGeomMaxStats = 3;

// Depth slots (depth_loss_slots).
constant constexpr int kDepthValid = 0, kDepthModel = 1, kDepthScale = 2, kDepthShift = 3, kDepthFloor = 4,
                       kDepthInvNorm = 5, kDepthSumAlpha = 6, kDepthCount = 7, kDepthMeanE = 8, kDepthSigmaP = 9;
// Normal and consistency slots (normal_loss_slots, normal_consistency_slots).
constant constexpr int kNormalValid = 0, kNormalSumAlpha = 1, kNormalCount = 2, kNormalMeanCos = 3,
                       kNormalInvNorm = 4;

constant constexpr float kDepthFloorFraction = 0.05f;
constant constexpr float kDepthMinAlpha = 1.0e-3f;
constant constexpr float kDepthResidualScale = 2.0f;
constant constexpr float kDepthMinVariance = 1.0e-20f;
constant constexpr float kDepthMinFloor = 1.0e-8f;
constant constexpr float kNormalMinAlpha = 1.0e-3f;
constant constexpr float kNormalMinPriorNorm = 0.5f;
constant constexpr float kNormalMinRenderNorm = 0.1f;
constant constexpr float kConsistencyMinAlpha = 0.5f;
constant constexpr float kConsistencyMaxRelJump = 0.05f;
constant constexpr float kConsistencyMinDepth = 1.0e-6f;
constant constexpr float kConsistencyMinCrossSq = 1.0e-24f;

struct GeomThread {
    uint rank, width, group, groups, simd_lane, simd_group, simd_groups;
};

#define GEOM_THREAD_ARGS                                                                                    \
    uint rank [[thread_index_in_threadgroup]], uint width [[threads_per_threadgroup]],                    \
        uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],               \
        uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]], \
        uint simd_groups [[simdgroups_per_threadgroup]]
#define GEOM_THREAD GeomThread{rank, width, group, groups, simd_lane, simd_group, simd_groups}

static void geom_store_block(constant GeomParams& p, thread float* sums, const int count, const GeomThread t,
                             threadgroup float* scratch) {
    for (int k = 0; k < count; ++k) {
        const float total = loss_group_sum(sums[k], scratch, t.rank, t.simd_lane, t.simd_group, t.simd_groups);
        if (t.rank == 0)
            p.blocks[uint(k) * p.num_blocks + t.group] = total;
    }
}

// Sums every block's statistics; valid in thread 0.
static void geom_sum_blocks(constant GeomParams& p, thread float* sums, const int count, const GeomThread t,
                            threadgroup float* scratch) {
    for (int k = 0; k < count; ++k)
        sums[k] = 0.0f;
    for (uint i = t.rank; i < p.num_blocks; i += t.width) {
        for (int k = 0; k < count; ++k)
            sums[k] += p.blocks[uint(k) * p.num_blocks + i];
    }
    for (int k = 0; k < count; ++k)
        sums[k] = loss_group_sum(sums[k], scratch, t.rank, t.simd_lane, t.simd_group, t.simd_groups);
}

static float geom_pixel_weight(constant GeomParams& p, const uint i) {
    if (p.has_weight == 0)
        return 1.0f;
    const float w = p.weight[i];
    return geom_finite(w) && w > 0.0f ? w : 0.0f;
}

// ---- Depth ----------------------------------------------------------------

static bool depth_active(const float target, const float depth, const float alpha) {
    return target > 0.0f && alpha > kDepthMinAlpha && geom_finite(target) && geom_finite(depth) &&
           geom_finite(alpha);
}

kernel void geom_depth_primary(constant GeomParams& p [[buffer(0)]], GEOM_THREAD_ARGS) {
    threadgroup float scratch[32];
    float sums[3] = {0.0f, 0.0f, 0.0f};
    const uint pixels = uint(p.width) * uint(p.height);
    for (uint i = group * width + rank; i < pixels; i += groups * width) {
        const float t = p.target[i], d = p.depth[i], a = p.alpha[i];
        if (!depth_active(t, d, a))
            continue;
        const float w = geom_pixel_weight(p, i);
        if (w == 0.0f)
            continue;
        const float aw = a * w;
        sums[0] += aw;
        sums[1] += aw * (fmax(d, 0.0f) / a);
        sums[2] += 1.0f;
    }
    geom_store_block(p, sums, 3, GEOM_THREAD, scratch);
}

kernel void geom_depth_finalize_primary(constant GeomParams& p [[buffer(0)]], GEOM_THREAD_ARGS) {
    threadgroup float scratch[32];
    float sums[3];
    geom_sum_blocks(p, sums, 3, GEOM_THREAD, scratch);
    if (rank != 0)
        return;
    const bool valid = sums[0] > 0.0f;
    const float mean_e = valid ? sums[1] / sums[0] : 0.0f;
    p.finals[kDepthValid] = valid ? 1.0f : 0.0f;
    p.finals[kDepthFloor] = fmax(kDepthMinFloor, kDepthFloorFraction * mean_e);
    p.finals[kDepthSumAlpha] = sums[0];
    p.finals[kDepthCount] = sums[2];
    p.finals[kDepthMeanE] = mean_e;
}

static float depth_floor(constant GeomParams& p) {
    return p.floor_override > 0.0f ? p.floor_override : p.finals[kDepthFloor];
}

// Moments of p = 1/(e + floor) around shift = 1/(mean_e + floor), so the
// variance does not cancel in float.
kernel void geom_depth_inverse(constant GeomParams& p [[buffer(0)]], GEOM_THREAD_ARGS) {
    threadgroup float scratch[32];
    float sums[2] = {0.0f, 0.0f};
    const float floor_f = depth_floor(p);
    const float shift = 1.0f / (p.finals[kDepthMeanE] + floor_f);
    const uint pixels = uint(p.width) * uint(p.height);
    if (p.finals[kDepthValid] > 0.5f) {
        for (uint i = group * width + rank; i < pixels; i += groups * width) {
            const float t = p.target[i], d = p.depth[i], a = p.alpha[i];
            if (!depth_active(t, d, a))
                continue;
            const float w = geom_pixel_weight(p, i);
            if (w == 0.0f)
                continue;
            const float e = fmax(d, 0.0f) / a;
            const float dp = 1.0f / (e + floor_f) - shift;
            const float aw = a * w;
            sums[0] += aw * dp;
            sums[1] += aw * dp * dp;
        }
    }
    geom_store_block(p, sums, 2, GEOM_THREAD, scratch);
}

kernel void geom_depth_finalize_alignment(constant GeomParams& p [[buffer(0)]], GEOM_THREAD_ARGS) {
    threadgroup float scratch[32];
    float sums[2];
    geom_sum_blocks(p, sums, 2, GEOM_THREAD, scratch);
    if (rank != 0)
        return;
    const bool valid = p.finals[kDepthValid] > 0.5f && p.anchor_floor > 0.0f && geom_finite(p.anchor_scale) &&
                       geom_finite(p.anchor_shift);
    const float sum_alpha = p.finals[kDepthSumAlpha];
    float sigma_p = 0.0f;
    if (valid) {
        const float mean = sums[0] / sum_alpha;
        const float var = fmax(sums[1] / sum_alpha - mean * mean, 0.0f);
        sigma_p = sqrt(fmax(var, kDepthMinVariance));
        p.finals[kDepthFloor] = p.anchor_floor;
    }
    p.finals[kDepthValid] = valid ? 1.0f : 0.0f;
    p.finals[kDepthModel] = float(p.anchor_model);
    p.finals[kDepthScale] = p.anchor_scale;
    p.finals[kDepthShift] = p.anchor_shift;
    p.finals[kDepthSigmaP] = sigma_p;
    p.finals[kDepthInvNorm] = valid ? 1.0f / sum_alpha : 0.0f;
}

struct DepthSample {
    bool ok;
    float alpha, weighted_alpha, e, p, d, delta;
};

static DepthSample depth_sample(constant GeomParams& p, const uint i, const int model, const float a, const float b,
                                const float floor_f, const float p_max, const float half_step) {
    DepthSample s = {false, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const float t = p.target[i], d_raw = p.depth[i], alpha = p.alpha[i];
    if (!depth_active(t, d_raw, alpha))
        return s;
    const float w = geom_pixel_weight(p, i);
    if (w == 0.0f)
        return s;
    s.alpha = alpha;
    s.weighted_alpha = alpha * w;
    s.e = fmax(d_raw, 0.0f) / alpha;
    s.p = 1.0f / (s.e + floor_f);
    const float fit = a * t + b;
    if (!(fit > 0.0f))
        return s;
    if (model == 0) {
        s.d = fmin(fit, p_max);
        s.delta = half_step;
    } else {
        s.d = 1.0f / (fit + floor_f);
        s.delta = half_step * s.d * s.d;
    }
    s.ok = true;
    return s;
}

static float depth_rho(const float x) {
    const float x2 = x * x;
    return 0.5f * x2 / (1.0f + x2);
}

static float depth_psi(const float x) {
    const float x2 = x * x;
    const float den = 1.0f + x2;
    return x / (den * den);
}

static float depth_deadband(const float r, const float delta) {
    const float excess = fabs(r) - delta;
    return excess > 0.0f ? (r > 0.0f ? excess : (r < 0.0f ? -excess : 0.0f)) : 0.0f;
}

kernel void geom_depth_grad(constant GeomParams& p [[buffer(0)]], GEOM_THREAD_ARGS) {
    threadgroup float scratch[32];
    float sums[2] = {0.0f, 0.0f};
    const bool valid = p.finals[kDepthValid] > 0.5f;
    const int model = int(p.finals[kDepthModel]);
    const float a = p.finals[kDepthScale];
    const float b = p.finals[kDepthShift];
    const float floor_f = p.finals[kDepthFloor];
    const float inv_norm = p.finals[kDepthInvNorm];
    const float sigma_p = p.finals[kDepthSigmaP];
    const float inv_sigma = valid && sigma_p > 0.0f ? 1.0f / (kDepthResidualScale * sigma_p) : 0.0f;
    const float p_max = 1.0f / floor_f;
    const float half_step = 0.5f * p.quantization_step * fabs(a);
    const uint W = uint(p.width), H = uint(p.height);
    const float lambda = p.lambda_grad;

    for (uint i = group * width + rank; i < W * H; i += groups * width) {
        if (!valid) {
            p.grad_depth[i] = 0.0f;
            p.grad_alpha[i] = 0.0f;
            continue;
        }
        const uint x = i % W, y = i / W;
        const DepthSample c = depth_sample(p, i, model, a, b, floor_f, p_max, half_step);
        float gp = 0.0f;
        if (c.ok) {
            const float x_r = depth_deadband(c.p - c.d, c.delta) * inv_sigma;
            if (x_r != 0.0f) {
                sums[0] += c.weighted_alpha * depth_rho(x_r);
                gp += c.weighted_alpha * depth_psi(x_r) * inv_sigma;
            }
            // Forward differences own the loss term; backward ones only add gradient.
            if (x + 1 < W) {
                const DepthSample n = depth_sample(p, i + 1, model, a, b, floor_f, p_max, half_step);
                if (n.ok) {
                    const float x_h = depth_deadband((n.p - c.p) - (n.d - c.d), c.delta + n.delta) * inv_sigma;
                    if (x_h != 0.0f) {
                        const float w2 = fmin(c.weighted_alpha, n.weighted_alpha);
                        sums[1] += w2 * depth_rho(x_h);
                        gp -= lambda * w2 * depth_psi(x_h) * inv_sigma;
                    }
                }
            }
            if (y + 1 < H) {
                const DepthSample n = depth_sample(p, i + W, model, a, b, floor_f, p_max, half_step);
                if (n.ok) {
                    const float x_v = depth_deadband((n.p - c.p) - (n.d - c.d), c.delta + n.delta) * inv_sigma;
                    if (x_v != 0.0f) {
                        const float w2 = fmin(c.weighted_alpha, n.weighted_alpha);
                        sums[1] += w2 * depth_rho(x_v);
                        gp -= lambda * w2 * depth_psi(x_v) * inv_sigma;
                    }
                }
            }
            if (x > 0) {
                const DepthSample n = depth_sample(p, i - 1, model, a, b, floor_f, p_max, half_step);
                if (n.ok) {
                    const float x_h = depth_deadband((c.p - n.p) - (c.d - n.d), c.delta + n.delta) * inv_sigma;
                    if (x_h != 0.0f)
                        gp += lambda * fmin(n.weighted_alpha, c.weighted_alpha) * depth_psi(x_h) * inv_sigma;
                }
            }
            if (y > 0) {
                const DepthSample n = depth_sample(p, i - W, model, a, b, floor_f, p_max, half_step);
                if (n.ok) {
                    const float x_v = depth_deadband((c.p - n.p) - (c.d - n.d), c.delta + n.delta) * inv_sigma;
                    if (x_v != 0.0f)
                        gp += lambda * fmin(n.weighted_alpha, c.weighted_alpha) * depth_psi(x_v) * inv_sigma;
                }
            }
        }
        if (c.ok && gp != 0.0f) {
            const float g = p.loss_weight * inv_norm * gp * (-c.p * c.p);
            p.grad_depth[i] = g / c.alpha;
            p.grad_alpha[i] = -g * c.e / c.alpha;
        } else {
            p.grad_depth[i] = 0.0f;
            p.grad_alpha[i] = 0.0f;
        }
    }
    geom_store_block(p, sums, 2, GEOM_THREAD, scratch);
}

kernel void geom_depth_finalize_loss(constant GeomParams& p [[buffer(0)]], GEOM_THREAD_ARGS) {
    threadgroup float scratch[32];
    float sums[2];
    geom_sum_blocks(p, sums, 2, GEOM_THREAD, scratch);
    if (rank != 0)
        return;
    const bool valid = p.finals[kDepthValid] > 0.5f;
    p.loss[0] = valid ? p.loss_weight * p.finals[kDepthInvNorm] * (sums[0] + p.lambda_grad * sums[1]) : 0.0f;
}

// ---- Normal, consistency and prior depth ----------------------------------

// Stats (sum weight, count, sum weight*cos) to finals, shared by the three losses.
kernel void geom_normal_finalize_stats(constant GeomParams& p [[buffer(0)]], GEOM_THREAD_ARGS) {
    threadgroup float scratch[32];
    float sums[kGeomMaxStats];
    geom_sum_blocks(p, sums, 3, GEOM_THREAD, scratch);
    if (rank != 0)
        return;
    const float sum_alpha = sums[0], count = sums[1];
    const bool valid = count >= p.min_count && sum_alpha >= p.min_weight;
    p.finals[kNormalValid] = valid ? 1.0f : 0.0f;
    p.finals[kNormalSumAlpha] = sum_alpha;
    p.finals[kNormalCount] = count;
    p.finals[kNormalMeanCos] = sum_alpha > 0.0f ? sums[2] / sum_alpha : 0.0f;
    p.finals[kNormalInvNorm] = valid ? p.loss_weight / fmax(sum_alpha, 1.0f) : 0.0f;
}

kernel void geom_normal_finalize_loss(constant GeomParams& p [[buffer(0)]], GEOM_THREAD_ARGS) {
    threadgroup float scratch[32];
    float sums[1];
    geom_sum_blocks(p, sums, 1, GEOM_THREAD, scratch);
    if (rank == 0)
        p.loss[0] = p.finals[kNormalInvNorm] * sums[0];
}

struct NormalSample {
    bool active;
    float alpha, cos, render_norm;
    float3 target_hat, render_hat;
};

static NormalSample normal_sample(constant GeomParams& p, const uint i, const uint pixels) {
    NormalSample s = {false, 0.0f, 0.0f, 0.0f, float3(0.0f), float3(0.0f)};
    const float a = p.alpha[i];
    const float3 t = float3(p.target[i], p.target[pixels + i], p.target[2 * pixels + i]);
    const float3 n = float3(p.normal[i], p.normal[pixels + i], p.normal[2 * pixels + i]);
    if (!geom_finite(a) || !geom_finite(t.x) || !geom_finite(t.y) || !geom_finite(t.z) || !geom_finite(n.x) ||
        !geom_finite(n.y) || !geom_finite(n.z) || a <= kNormalMinAlpha)
        return s;
    float effective = a;
    if (p.has_weight != 0) {
        const float w = p.weight[i];
        if (!geom_finite(w) || w <= 0.0f)
            return s;
        effective *= w;
    }
    const float t_norm = sqrt(t.x * t.x + t.y * t.y + t.z * t.z);
    if (t_norm < kNormalMinPriorNorm)
        return s;
    const float n_norm = sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if (n_norm < kNormalMinRenderNorm)
        return s;
    s.target_hat = t * (1.0f / t_norm);
    s.render_hat = n * (1.0f / n_norm);
    s.cos = s.render_hat.x * s.target_hat.x + s.render_hat.y * s.target_hat.y + s.render_hat.z * s.target_hat.z;
    s.alpha = effective;
    s.render_norm = n_norm;
    s.active = true;
    return s;
}

kernel void geom_normal_stats(constant GeomParams& p [[buffer(0)]], GEOM_THREAD_ARGS) {
    threadgroup float scratch[32];
    float sums[3] = {0.0f, 0.0f, 0.0f};
    const uint pixels = uint(p.width) * uint(p.height);
    for (uint i = group * width + rank; i < pixels; i += groups * width) {
        const NormalSample s = normal_sample(p, i, pixels);
        if (!s.active)
            continue;
        sums[0] += s.alpha;
        sums[1] += 1.0f;
        sums[2] += s.alpha * s.cos;
    }
    geom_store_block(p, sums, 3, GEOM_THREAD, scratch);
}

kernel void geom_normal_grad(constant GeomParams& p [[buffer(0)]], GEOM_THREAD_ARGS) {
    threadgroup float scratch[32];
    float sums[1] = {0.0f};
    const float inv_norm = p.finals[kNormalInvNorm];
    const uint pixels = uint(p.width) * uint(p.height);
    for (uint i = group * width + rank; i < pixels; i += groups * width) {
        const NormalSample s = normal_sample(p, i, pixels);
        if (!s.active || inv_norm == 0.0f) {
            p.grad_normal[i] = 0.0f;
            p.grad_normal[pixels + i] = 0.0f;
            p.grad_normal[2 * pixels + i] = 0.0f;
            continue;
        }
        sums[0] += s.alpha * (1.0f - s.cos);
        // d(1 - cos)/dn = -(t_hat - cos * n_hat) / |n|
        const float scale = -inv_norm * s.alpha / s.render_norm;
        const float3 g = scale * (s.target_hat - s.cos * s.render_hat);
        p.grad_normal[i] = g.x;
        p.grad_normal[pixels + i] = g.y;
        p.grad_normal[2 * pixels + i] = g.z;
    }
    geom_store_block(p, sums, 1, GEOM_THREAD, scratch);
}

static float3 geom_ray(constant GeomParams& p, const int x, const int y) {
    return float3((float(x) + 0.5f - p.cx) / p.fx, (float(y) + 0.5f - p.cy) / p.fy, 1.0f);
}

static bool geom_expected_depth(constant GeomParams& p, const uint i, thread float& e, thread float& a) {
    const float alpha = p.alpha[i];
    const float d = p.depth[i];
    if (!geom_finite(alpha) || !geom_finite(d) || alpha < kConsistencyMinAlpha)
        return false;
    const float value = fmax(d, 0.0f) / alpha;
    if (value < kConsistencyMinDepth)
        return false;
    e = value;
    a = alpha;
    return true;
}

// The normal of the depth surface from central differences. Neighbor order:
// +x, -x, +y, -y.
struct DepthNormal {
    bool active;
    float alpha, sign, nraw_norm;
    float3 nd, tx, ty;
    float neighbor_e[4];
    float neighbor_alpha[4];
};

static DepthNormal depth_normal(constant GeomParams& p, const int x, const int y) {
    DepthNormal s;
    s.active = false;
    if (x <= 0 || y <= 0 || x >= p.width - 1 || y >= p.height - 1)
        return s;
    const uint W = uint(p.width);
    const uint i = uint(y) * W + uint(x);
    float e_c, a_c;
    if (!geom_expected_depth(p, i, e_c, a_c))
        return s;
    const uint neighbor[4] = {i + 1, i - 1, i + W, i - W};
    const float jump = kConsistencyMaxRelJump * e_c;
    for (int k = 0; k < 4; ++k) {
        if (!geom_expected_depth(p, neighbor[k], s.neighbor_e[k], s.neighbor_alpha[k]))
            return s;
        if (fabs(s.neighbor_e[k] - e_c) > jump)
            return s;
    }
    const float3 tx = s.neighbor_e[0] * geom_ray(p, x + 1, y) - s.neighbor_e[1] * geom_ray(p, x - 1, y);
    const float3 ty = s.neighbor_e[2] * geom_ray(p, x, y + 1) - s.neighbor_e[3] * geom_ray(p, x, y - 1);
    const float3 n_raw = float3(tx.y * ty.z - tx.z * ty.y, tx.z * ty.x - tx.x * ty.z, tx.x * ty.y - tx.y * ty.x);
    const float norm_sq = n_raw.x * n_raw.x + n_raw.y * n_raw.y + n_raw.z * n_raw.z;
    if (norm_sq < kConsistencyMinCrossSq)
        return s;
    // Camera-facing orientation, detached like the rasterizer's own flip.
    const float3 ray = geom_ray(p, x, y);
    const float facing = n_raw.x * ray.x + n_raw.y * ray.y + n_raw.z * ray.z;
    s.sign = facing > 0.0f ? -1.0f : 1.0f;
    s.nraw_norm = sqrt(norm_sq);
    s.nd = n_raw * (s.sign / s.nraw_norm);
    s.tx = tx;
    s.ty = ty;
    s.alpha = a_c;
    s.active = true;
    return s;
}

static void geom_atomic_add(device float* base, const uint i, const float value) {
    atomic_fetch_add_explicit((device atomic_float*)(base + i), value, memory_order_relaxed);
}

// Scatters d(1 - cos)/d(depth, alpha) of a depth normal to its four neighbors.
static void depth_normal_backward(constant GeomParams& p, thread const DepthNormal& s, const float3 target_hat,
                                  const float cos, const float g_w, const int x, const int y) {
    const float scale = -g_w * s.sign / s.nraw_norm;
    const float3 g_raw = scale * (target_hat - cos * s.nd);
    // n_raw = tx x ty: grad_tx = ty x g_raw, grad_ty = g_raw x tx
    const float3 g_tx = float3(s.ty.y * g_raw.z - s.ty.z * g_raw.y, s.ty.z * g_raw.x - s.ty.x * g_raw.z,
                               s.ty.x * g_raw.y - s.ty.y * g_raw.x);
    const float3 g_ty = float3(g_raw.y * s.tx.z - g_raw.z * s.tx.y, g_raw.z * s.tx.x - g_raw.x * s.tx.z,
                               g_raw.x * s.tx.y - g_raw.y * s.tx.x);
    const float3 r_xp = geom_ray(p, x + 1, y), r_xm = geom_ray(p, x - 1, y);
    const float3 r_yp = geom_ray(p, x, y + 1), r_ym = geom_ray(p, x, y - 1);
    const float g_e[4] = {g_tx.x * r_xp.x + g_tx.y * r_xp.y + g_tx.z * r_xp.z,
                          -(g_tx.x * r_xm.x + g_tx.y * r_xm.y + g_tx.z * r_xm.z),
                          g_ty.x * r_yp.x + g_ty.y * r_yp.y + g_ty.z * r_yp.z,
                          -(g_ty.x * r_ym.x + g_ty.y * r_ym.y + g_ty.z * r_ym.z)};
    const uint W = uint(p.width);
    const uint i = uint(y) * W + uint(x);
    const uint neighbor[4] = {i + 1, i - 1, i + W, i - W};
    for (int k = 0; k < 4; ++k) {
        // E = max(accum, 0)/alpha: dE/daccum = 1/alpha, dE/dalpha = -E/alpha
        const float inv_a = 1.0f / s.neighbor_alpha[k];
        geom_atomic_add(p.grad_depth, neighbor[k], g_e[k] * inv_a);
        geom_atomic_add(p.grad_alpha, neighbor[k], -g_e[k] * s.neighbor_e[k] * inv_a);
    }
}

// The rendered normal at a pixel with a valid depth normal.
static bool rendered_hat(constant GeomParams& p, const uint i, const uint pixels, thread float3& hat,
                         thread float& norm) {
    const float3 n = float3(p.normal[i], p.normal[pixels + i], p.normal[2 * pixels + i]);
    if (!geom_finite(n.x) || !geom_finite(n.y) || !geom_finite(n.z))
        return false;
    norm = sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if (norm < kNormalMinRenderNorm)
        return false;
    hat = n * (1.0f / norm);
    return true;
}

static bool prior_hat(constant GeomParams& p, const uint i, const uint pixels, thread float3& hat) {
    const float3 n = float3(p.normal[i], p.normal[pixels + i], p.normal[2 * pixels + i]);
    if (!geom_finite(n.x) || !geom_finite(n.y) || !geom_finite(n.z))
        return false;
    const float norm = sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if (norm < kNormalMinPriorNorm)
        return false;
    hat = n * (1.0f / norm);
    return true;
}

// The depth-normal pixel sample of the consistency (prior = 0) or prior-depth
// (prior = 1) loss.
static bool depth_normal_sample(constant GeomParams& p, const uint i, const uint prior, thread DepthNormal& s,
                                thread float3& hat, thread float& render_norm, thread float& cos, thread float& aw) {
    const uint W = uint(p.width);
    const uint pixels = W * uint(p.height);
    s = depth_normal(p, int(i % W), int(i / W));
    if (!s.active)
        return false;
    if (prior != 0) {
        if (!prior_hat(p, i, pixels, hat))
            return false;
    } else if (!rendered_hat(p, i, pixels, hat, render_norm)) {
        return false;
    }
    cos = s.nd.x * hat.x + s.nd.y * hat.y + s.nd.z * hat.z;
    aw = s.alpha * geom_pixel_weight(p, i);
    return aw != 0.0f;
}

kernel void geom_depth_normal_stats(constant GeomParams& p [[buffer(0)]],
                                    GEOM_THREAD_ARGS) {
    threadgroup float scratch[32];
    float sums[3] = {0.0f, 0.0f, 0.0f};
    const uint pixels = uint(p.width) * uint(p.height);
    for (uint i = group * width + rank; i < pixels; i += groups * width) {
        DepthNormal s;
        float3 hat;
        float render_norm = 0.0f, cos = 0.0f, aw = 0.0f;
        if (!depth_normal_sample(p, i, p.prior, s, hat, render_norm, cos, aw))
            continue;
        sums[0] += aw;
        sums[1] += 1.0f;
        sums[2] += aw * cos;
    }
    geom_store_block(p, sums, 3, GEOM_THREAD, scratch);
}

// Adds into grad_normal (consistency) and scatters into grad_depth/grad_alpha.
kernel void geom_depth_normal_grad(constant GeomParams& p [[buffer(0)]],
                                   GEOM_THREAD_ARGS) {
    threadgroup float scratch[32];
    float sums[1] = {0.0f};
    const float inv_norm = p.finals[kNormalInvNorm];
    const uint W = uint(p.width);
    const uint pixels = W * uint(p.height);
    for (uint i = group * width + rank; i < pixels; i += groups * width) {
        DepthNormal s;
        float3 hat;
        float render_norm = 0.0f, cos = 0.0f, aw = 0.0f;
        if (inv_norm == 0.0f || !depth_normal_sample(p, i, p.prior, s, hat, render_norm, cos, aw))
            continue;
        sums[0] += aw * (1.0f - cos);
        const float g_w = inv_norm * aw;
        if (p.prior == 0) {
            // d(1 - cos)/dn_render = -(n_d - cos * n_hat) / |n_render|
            const float3 g = (-g_w / render_norm) * (s.nd - cos * hat);
            p.grad_normal[i] += g.x;
            p.grad_normal[pixels + i] += g.y;
            p.grad_normal[2 * pixels + i] += g.z;
        }
        depth_normal_backward(p, s, hat, cos, g_w, int(i % W), int(i / W));
    }
    geom_store_block(p, sums, 1, GEOM_THREAD, scratch);
}

// ---- Anchor samples ---------------------------------------------------------

struct GeomAnchorParams {
    device const float* points;
    device const float* w2c;
    device const float* prior;
    device float2* pairs;
    device atomic_int* count;
    float4 aabb_lo, aabb_hi;
    uint samples, stride;
    int width, height, capacity;
    float fx, fy, cx, cy, near_plane;
};

// Projects every stride-th point, samples the prior at the landing pixel and
// appends (prior value, camera-space depth).
kernel void geom_anchor_collect(constant GeomAnchorParams& p [[buffer(0)]], uint s [[thread_position_in_grid]]) {
    if (s >= p.samples)
        return;
    const uint i = s * p.stride;
    const float x = p.points[i * 3], y = p.points[i * 3 + 1], zw = p.points[i * 3 + 2];
    if (!geom_finite(x) || !geom_finite(y) || !geom_finite(zw))
        return;
    if (x < p.aabb_lo.x || x > p.aabb_hi.x || y < p.aabb_lo.y || y > p.aabb_hi.y || zw < p.aabb_lo.z ||
        zw > p.aabb_hi.z)
        return;
    device const float* m = p.w2c;
    const float z = m[8] * x + m[9] * y + m[10] * zw + m[11];
    if (!(z > p.near_plane) || !geom_finite(z))
        return;
    const float xc = m[0] * x + m[1] * y + m[2] * zw + m[3];
    const float yc = m[4] * x + m[5] * y + m[6] * zw + m[7];
    const int u = int(floor(p.fx * xc / z + p.cx));
    const int v = int(floor(p.fy * yc / z + p.cy));
    if (u < 0 || u >= p.width || v < 0 || v >= p.height)
        return;
    const float t = p.prior[uint(v) * uint(p.width) + uint(u)];
    if (!(t > 0.0f) || !geom_finite(t))
        return;
    const int slot = atomic_fetch_add_explicit(p.count, 1, memory_order_relaxed);
    if (slot < p.capacity)
        p.pairs[slot] = float2(t, z);
}
