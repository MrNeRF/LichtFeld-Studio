// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// MrnfOps kernels: ports of mrnf_kernels.cu, plus the curand Philox stream
// and the radix select of densification_kernels.cu that Refine and Mcmc use.

// The library builds with fast math, which assumes finite values, so NaN and
// infinity tests read the bits.
static bool mrnf_is_nan(const float v) { return (as_type<uint>(v) & 0x7fffffffu) > 0x7f800000u; }
static bool mrnf_is_inf(const float v) { return (as_type<uint>(v) & 0x7fffffffu) == 0x7f800000u; }
static bool mrnf_is_finite(const float v) { return (as_type<uint>(v) & 0x7fffffffu) < 0x7f800000u; }
// v > 0 without NaNs: +0 < bits <= +inf.
static bool mrnf_is_positive(const float v) { return as_type<uint>(v) - 1u < 0x7f800000u; }

// curandStatePhilox4_32_10_t after curand_init(seed, subsequence, 0): the
// first output block of Philox4x32-10 at counter (0, 0, subsequence).
static uint4 mrnf_philox_block(const ulong seed, const ulong subsequence) {
    uint4 c = uint4(0u, 0u, uint(subsequence), uint(subsequence >> 32));
    uint2 k = uint2(uint(seed), uint(seed >> 32));
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            k.x += 0x9E3779B9u;
            k.y += 0xBB67AE85u;
        }
        const uint hi0 = mulhi(0xD2511F53u, c.x);
        const uint lo0 = 0xD2511F53u * c.x;
        const uint hi1 = mulhi(0xCD9E8D57u, c.z);
        const uint lo1 = 0xCD9E8D57u * c.z;
        c = uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    }
    return c;
}

// curand_uniform on the stream's first word: (0, 1].
static float mrnf_curand_uniform(const uint x) {
    return fma(float(x), 2.3283064e-10f, 2.3283064e-10f / 2.0f);
}

// curand_normal4: Box-Muller on the words (x, y) and (z, w).
static float4 mrnf_curand_normal4(const uint4 x) {
    constexpr float kInv2Pi = 2.3283064e-10f * 6.2831855f;
    const float u0 = mrnf_curand_uniform(x.x);
    const float v0 = fma(float(x.y), kInv2Pi, kInv2Pi / 2.0f);
    const float u1 = mrnf_curand_uniform(x.z);
    const float v1 = fma(float(x.w), kInv2Pi, kInv2Pi / 2.0f);
    const float s0 = sqrt(-2.0f * log(u0));
    const float s1 = sqrt(-2.0f * log(u1));
    return float4(sin(v0) * s0, cos(v0) * s0, sin(v1) * s1, cos(v1) * s1);
}

static bool mrnf_flag(device const uchar* mask, const uint count, const uint i) {
    return mask != nullptr && i < count && mask[i] != 0;
}

static float mrnf_sigmoid(const float x) { return 1.0f / (1.0f + exp(-x)); }

static float mrnf_logit(float p) {
    if (mrnf_is_nan(p))
        return p;
    p = fmin(fmax(p, 1e-12f), as_type<float>(0x3f7fffffu));
    return log(p / (1.0f - p));
}

// Radix select: the element sorted[rank] of cub::DeviceRadixSort's float
// order, in three passes of 11/11/10 bits. Selection s keeps its state
// (total, rank, prefix, value bits) at workspace[4 s] and its histogram at
// workspace[4 selections + 2048 s]; element i is values[offsets[s] + i * stride].
constant constexpr uint kMrnfSelectBins = 2048;
constant constexpr uint kMrnfSelectMedian = 0xffffffffu;

struct MrnfSelectParams {
    device const float* values;
    device uint* workspace;
    uint count;
    uint stride;
    uint positive_only;
    uint pass;
    uint selections;
    uint offsets[8];
    uint ranks[8];
};

static uint mrnf_select_shift(const uint pass) { return pass == 0 ? 21u : (pass == 1 ? 10u : 0u); }
static uint mrnf_select_bits(const uint pass) { return pass == 2 ? 10u : 11u; }

static uint mrnf_select_key(const float v) {
    const uint bits = as_type<uint>(v);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

kernel void mrnf_select_histogram(constant MrnfSelectParams& p [[buffer(0)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint2 groups [[threadgroups_per_grid]],
                                 uint lane [[thread_index_in_threadgroup]],
                                 uint simd_lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint bins[kMrnfSelectBins];
    for (uint b = lane; b < kMrnfSelectBins; b += kThreadgroupWidth)
        atomic_store_explicit(&bins[b], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint s = group.y;
    device atomic_uint* state = (device atomic_uint*)(p.workspace + 4 * s);
    device atomic_uint* hist = (device atomic_uint*)(p.workspace + 4 * p.selections + kMrnfSelectBins * s);
    const uint shift = mrnf_select_shift(p.pass);
    const uint digit_mask = (1u << mrnf_select_bits(p.pass)) - 1u;
    const uint fixed_shift = shift + mrnf_select_bits(p.pass);
    const uint fixed = p.pass == 0 ? 0u : atomic_load_explicit(&state[2], memory_order_relaxed) >> fixed_shift;
    uint counted = 0u;
    const uint grid_stride = groups.x * kThreadgroupWidth;
    for (uint i = group.x * kThreadgroupWidth + lane; i < p.count; i += grid_stride) {
        const float v = p.values[p.offsets[s] + ulong(i) * p.stride];
        if (p.positive_only != 0 && !mrnf_is_positive(v))
            continue;
        const uint key = mrnf_select_key(v);
        if (p.pass > 0 && (key >> fixed_shift) != fixed)
            continue;
        ++counted;
        atomic_fetch_add_explicit(&bins[(key >> shift) & digit_mask], 1u, memory_order_relaxed);
    }
    if (p.pass == 0 && p.positive_only != 0) {
        counted = simd_sum(counted);
        if (simd_lane == 0 && counted != 0u)
            atomic_fetch_add_explicit(&state[0], counted, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint b = lane; b < kMrnfSelectBins; b += kThreadgroupWidth) {
        const uint c = atomic_load_explicit(&bins[b], memory_order_relaxed);
        if (c != 0u)
            atomic_fetch_add_explicit(&hist[b], c, memory_order_relaxed);
    }
}

// One thread per selection walks the digit histogram.
kernel void mrnf_select_pick(constant MrnfSelectParams& p [[buffer(0)]], uint s [[thread_position_in_grid]]) {
    if (s >= p.selections)
        return;
    device uint* state = p.workspace + 4 * s;
    device uint* hist = p.workspace + 4 * p.selections + kMrnfSelectBins * s;
    if (p.pass == 0) {
        if (p.positive_only == 0)
            state[0] = p.count;
        state[1] = p.ranks[s] == kMrnfSelectMedian ? state[0] / 2u : p.ranks[s];
        state[2] = 0u;
    }
    const uint bins = 1u << mrnf_select_bits(p.pass);
    uint below = 0u;
    uint digit = bins - 1u;
    for (uint b = 0; b < bins; ++b) {
        const uint c = hist[b];
        if (state[1] < below + c) {
            digit = b;
            break;
        }
        below += c;
    }
    for (uint b = 0; b < kMrnfSelectBins; ++b)
        hist[b] = 0u;
    state[2] |= digit << mrnf_select_shift(p.pass);
    state[1] -= below;
    if (p.pass == 2) {
        const uint key = state[2];
        state[3] = state[0] == 0u ? 0u : ((key & 0x80000000u) != 0u ? (key & 0x7fffffffu) : ~key);
    }
}

struct MrnfNoiseParams {
    device float* means;
    device const float* raw_opacity;
    device const float* visibility;
    device const uchar* frozen;
    ulong seed;
    uint frozen_count;
    uint count;
    float lr_mean;
    float noise_weight;
    float median_scale;
};

kernel void mrnf_noise(constant MrnfNoiseParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count || mrnf_flag(p.frozen, p.frozen_count, i) || p.visibility[i] <= 0.0f)
        return;
    const float inv_opacity = fmax(1.0f - mrnf_sigmoid(p.raw_opacity[i]), 0.0f);
    float weight = inv_opacity > 0.0f ? pow(inv_opacity, 150.0f) : 0.0f;
    weight *= p.lr_mean * p.noise_weight;
    if (weight < 1e-12f)
        return;
    const float4 n = mrnf_curand_normal4(mrnf_philox_block(p.seed, i));
    const float noise[3] = {n.x, n.y, n.z};
    for (uint d = 0; d < 3; ++d)
        p.means[i * 3 + d] += fmin(fmax(noise[d] * weight, -p.median_scale), p.median_scale);
}

struct MrnfDecayParams {
    device float* raw_opacity;
    device float* log_scales;
    device const uchar* frozen;
    device const uchar* far_mask;
    uint frozen_count;
    uint far_count;
    uint count;
    float opacity_decay;
    float scale_decay;
    float far_decay_scale;
    float train_t;
};

kernel void mrnf_decay(constant MrnfDecayParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count || mrnf_flag(p.frozen, p.frozen_count, i))
        return;
    float opacity_decay = p.opacity_decay;
    float scale_decay = p.scale_decay;
    if (mrnf_flag(p.far_mask, p.far_count, i)) {
        opacity_decay *= p.far_decay_scale;
        scale_decay *= p.far_decay_scale;
    }
    const float t_shrink = 1.0f - p.train_t;
    const float opacity_delta = opacity_decay * t_shrink;
    const float raw = p.raw_opacity[i];
    if (opacity_delta != 0.0f || mrnf_is_inf(raw))
        p.raw_opacity[i] = mrnf_logit(mrnf_sigmoid(raw) - opacity_delta);
    const float factor = 1.0f - scale_decay * t_shrink;
    for (uint d = 0; d < 3; ++d) {
        const float scale = exp(p.log_scales[i * 3 + d]) * factor;
        p.log_scales[i * 3 + d] = log(fmax(scale, 1e-12f));
    }
}

struct MrnfFoldParams {
    device float* visibility;
    device float* weight_max;
    device float* densification;
    device float* ratio_max;
    uint count;
    float ratio_power;
};

kernel void mrnf_fold(constant MrnfFoldParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const float vis = p.densification[i];
    const float err = p.densification[p.count + i];
    p.visibility[i] += vis;
    p.weight_max[i] = fmax(p.weight_max[i], err);
    if (p.ratio_max != nullptr) {
        const float ratio = vis >= 0.05f ? (p.ratio_power > 0.0f ? err / pow(vis, p.ratio_power) : err / vis) : 0.0f;
        p.ratio_max[i] = fmax(p.ratio_max[i], ratio);
    }
    p.densification[i] = 0.0f;
    p.densification[p.count + i] = 0.0f;
}

kernel void mrnf_fold_error(constant MrnfFoldParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    p.weight_max[i] = fmax(p.weight_max[i], p.densification[p.count + i]);
    p.densification[p.count + i] = 0.0f;
}

struct MrnfGeomeanParams {
    device const float* raw_scales;
    device float* extents;
    uint count;
};

kernel void mrnf_geomean_extent(constant MrnfGeomeanParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const float g = exp((p.raw_scales[i * 3] + p.raw_scales[i * 3 + 1] + p.raw_scales[i * 3 + 2]) * (1.0f / 3.0f));
    p.extents[i] = mrnf_is_finite(g) && g > 0.0f ? g : 0.0f;
}

struct MrnfGumbelParams {
    device const float* weights;
    device const long* sources;
    device float* keys;
    ulong seed;
    uint count;
};

// Keys for every weight, or for the positive weights listed in sources.
kernel void mrnf_gumbel_keys(constant MrnfGumbelParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const float w = p.weights[p.sources != nullptr ? uint(p.sources[i]) : i];
    if (p.sources == nullptr && w <= 0.0f) {
        p.keys[i] = -1e30f;
        return;
    }
    float u = mrnf_curand_uniform(mrnf_philox_block(p.seed, i).x);
    u = fmin(fmax(u, 1e-10f), 1.0f - 1e-7f);
    p.keys[i] = -log(-log(u)) + log(w);
}

struct MrnfGatherIndicesParams {
    device const long* order;
    device const long* sources;
    device long* output;
    uint count;
};

// output[i] = sources[order[i]], either table absent meaning the identity.
kernel void mrnf_gather_indices(constant MrnfGatherIndicesParams& p [[buffer(0)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const long position = p.order != nullptr ? p.order[i] : long(i);
    p.output[i] = p.sources != nullptr ? p.sources[position] : position;
}

struct MrnfProjectParams {
    device const float* means;
    device const float* view;
    device float* means2d;
    device float* radii;
    uint count;
    int width;
    int height;
    float fx, fy, cx, cy;
    float near_plane;
};

kernel void mrnf_project_centers(constant MrnfProjectParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    device const float* m = p.view;
    const float x = p.means[i * 3], y = p.means[i * 3 + 1], z = p.means[i * 3 + 2];
    const float cam_x = m[0] * x + m[1] * y + m[2] * z + m[3];
    const float cam_y = m[4] * x + m[5] * y + m[6] * z + m[7];
    const float cam_z = m[8] * x + m[9] * y + m[10] * z + m[11];
    if (!mrnf_is_finite(cam_x) || !mrnf_is_finite(cam_y) || !mrnf_is_finite(cam_z) || !(cam_z > p.near_plane)) {
        p.means2d[i * 2] = 0.0f;
        p.means2d[i * 2 + 1] = 0.0f;
        p.radii[i] = 0.0f;
        return;
    }
    const float px = p.fx * (cam_x / cam_z) + p.cx;
    const float py = p.fy * (cam_y / cam_z) + p.cy;
    p.means2d[i * 2] = px;
    p.means2d[i * 2 + 1] = py;
    p.radii[i] = px >= 0.0f && py >= 0.0f && px < float(p.width) && py < float(p.height) ? 1.0f : 0.0f;
}

struct MrnfCenterErrorParams {
    device const float* means2d;
    device const float* radii;
    device const float* error;
    device float* scores;
    uint count;
    int width;
    int height;
};

kernel void mrnf_gather_center_error(constant MrnfCenterErrorParams& p [[buffer(0)]],
                                     uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    if (p.radii[i] <= 0.0f) {
        p.scores[i] = 0.0f;
        return;
    }
    const int x = clamp(int(floor(p.means2d[i * 2])), 0, p.width - 1);
    const int y = clamp(int(floor(p.means2d[i * 2 + 1])), 0, p.height - 1);
    p.scores[i] = p.error[uint(y) * uint(p.width) + uint(x)];
}

struct MrnfFarMaskParams {
    device const float* means;
    device uchar* mask;
    float3 center;
    uint count;
    float radius_sq;
};

kernel void mrnf_far_mask(constant MrnfFarMaskParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const float dx = p.means[i * 3] - p.center.x;
    const float dy = p.means[i * 3 + 1] - p.center.y;
    const float dz = p.means[i * 3 + 2] - p.center.z;
    p.mask[i] = (dx * dx + dy * dy + dz * dz) > p.radius_sq ? 1 : 0;
}

struct MrnfMeanAbsErrorParams {
    device const float* predicted;
    device const float* target;
    device float* error;
    uint pixels;
    uint channels;
};

kernel void mrnf_mean_abs_error(constant MrnfMeanAbsErrorParams& p [[buffer(0)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= p.pixels)
        return;
    float sum = 0.0f;
    for (uint c = 0; c < p.channels; ++c)
        sum += fabs(p.predicted[c * p.pixels + i] - p.target[c * p.pixels + i]);
    p.error[i] = sum / float(p.channels);
}

struct MrnfSeedWeightsParams {
    device const float* error;
    device const float* alpha;
    device float* weights;
    uint count;
};

kernel void mrnf_seed_weights(constant MrnfSeedWeightsParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count)
        p.weights[i] = p.error[i] * (1.0f - p.alpha[i]);
}

struct MrnfGatherSeedsParams {
    device const long* indices;
    device const float* target;
    device const float* alpha;
    device const float* depth;
    device float* rgb;
    device float* sampled_alpha;
    device float* sampled_depth;
    uint count;
    uint pixels;
    int channels;
};

kernel void mrnf_gather_seeds(constant MrnfGatherSeedsParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const long index = p.indices[i];
    const uint pixel = index >= 0 && ulong(index) < p.pixels ? uint(index) : 0u;
    const int channels = p.channels > 0 ? p.channels : 1;
    for (int c = 0; c < 3; ++c)
        p.rgb[i * 3 + uint(c)] = p.target[uint(min(c, channels - 1)) * p.pixels + pixel];
    p.sampled_alpha[i] = p.alpha[pixel];
    p.sampled_depth[i] = p.depth != nullptr ? p.depth[pixel] : 0.0f;
}

struct MrnfStarvationParams {
    device float* weights;
    device const float* visibility;
    uint count;
    float median;
};

kernel void mrnf_starvation_weights(constant MrnfStarvationParams& p [[buffer(0)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const float vis = p.visibility[i];
    if (vis == 0.0f) {
        p.weights[i] = 0.0f;
        return;
    }
    const float starved = fmin(fmax(1.0f - vis / fmax(p.median, 1.19209290e-07f), 0.0f), 1.0f);
    // kStarvEps and kStarvGamma of mrnf.hpp.
    const float term = starved > 0.0f ? pow(starved, 1.72f) : 0.0f;
    p.weights[i] *= 0.0026f + term;
}

struct MrnfPruneBoundsParams {
    device const float* means;
    device const float* scale_max;
    device uchar* mask;
    float3 center;
    uint count;
    float maximum;
    float log_maximum;
};

kernel void mrnf_prune_bounds(constant MrnfPruneBoundsParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const float dx = fabs(p.means[i * 3] - p.center.x);
    const float dy = fabs(p.means[i * 3 + 1] - p.center.y);
    const float dz = fabs(p.means[i * 3 + 2] - p.center.z);
    const bool far = !mrnf_is_nan(dx) && !mrnf_is_nan(dy) && !mrnf_is_nan(dz) && fmax(dx, fmax(dy, dz)) > p.maximum;
    p.mask[i] = (p.mask[i] != 0 || p.scale_max[i] > p.log_maximum || far) ? 1 : 0;
}

struct MrnfParentWeightsParams {
    device const float* opacity;
    device const float* visibility;
    device const uchar* active;
    device const uchar* trainable;
    device const float* edge;
    device float* weights;
    uint count;
};

kernel void mrnf_replace_parent_weights(constant MrnfParentWeightsParams& p [[buffer(0)]],
                                        uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    float weight = p.opacity[i] * (p.visibility[i] > 0.0f ? 1.0f : 0.0f);
    if (p.active != nullptr)
        weight *= p.active[i] != 0 ? 1.0f : 0.0f;
    if (p.trainable != nullptr)
        weight *= p.trainable[i] != 0 ? 1.0f : 0.0f;
    if (p.edge != nullptr)
        weight *= p.edge[i];
    p.weights[i] = weight;
}
