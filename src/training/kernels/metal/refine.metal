// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// RefineOps kernels: ports of densification_kernels.cu and pruning_kernels.cu.

// Largest axis first, then the other two in order; ties prefer the lower axis.
static uint3 refine_axis_order(const float3 v) {
    const float largest = fmax(v.x, fmax(v.y, v.z));
    if (largest == v.x)
        return uint3(0, 1, 2);
    if (largest == v.y)
        return uint3(1, 0, 2);
    return uint3(2, 0, 1);
}

struct RefineSplitParams {
    device float* means;
    device const float* rotations;
    device float* scales;
    device const float* sh0;
    device float* opacity;
    device float* child_means;
    device float* child_rotations;
    device float* child_scales;
    device float* child_sh0;
    device float* child_opacity;
    device const long* indices;
    uint count;
};

// Long-axis split: the parent moves half its long axis forward in place,
// the child half backward.
kernel void refine_split(constant RefineSplitParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const int src = int(p.indices[i]);
    const float3 pos = float3(p.means[src * 3], p.means[src * 3 + 1], p.means[src * 3 + 2]);
    const float4 q = float4(p.rotations[src * 4], p.rotations[src * 4 + 1], p.rotations[src * 4 + 2],
                            p.rotations[src * 4 + 3]);
    const float3 scale = float3(p.scales[src * 3], p.scales[src * 3 + 1], p.scales[src * 3 + 2]);
    // Trained quaternions drift from unit norm; projection normalizes them.
    const float inverse_norm = fmin(rsqrt(dot(q, q)), 1e12f);
    const float w = q.x * inverse_norm, x = q.y * inverse_norm, y = q.z * inverse_norm, z = q.w * inverse_norm;
    // Columns of the row-major rotation matrix of quat_to_rotmat.
    const float3 column[3] = {
        float3(1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y + w * z), 2.0f * (x * z - w * y)),
        float3(2.0f * (x * y - w * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z + w * x)),
        float3(2.0f * (x * z + w * y), 2.0f * (y * z - w * x), 1.0f - 2.0f * (x * x + y * y))};
    const uint3 axes = refine_axis_order(scale);
    const float offset_magnitude = exp(scale[axes.x]) * 0.5f;
    float3 new_scale;
    new_scale[axes.x] = scale[axes.x] + as_type<float>(0xbf317218u); // logf(0.5f)
    new_scale[axes.y] = scale[axes.y] + as_type<float>(0xbe266b5bu); // logf(0.85f)
    new_scale[axes.z] = scale[axes.z] + as_type<float>(0xbe266b5bu);
    const float sig = 1.0f / (1.0f + exp(-p.opacity[src]));
    const float raw_sig = fmax(1e-7f, fmin(1.0f - 1e-7f, sig * 0.6f));
    const float new_opacity = log(raw_sig / (1.0f - raw_sig));
    const float3 offset = column[axes.x] * offset_magnitude;

    for (uint d = 0; d < 3; ++d) {
        p.means[src * 3 + int(d)] = pos[d] + offset[d];
        p.scales[src * 3 + int(d)] = new_scale[d];
        p.child_means[i * 3 + d] = pos[d] - offset[d];
        p.child_scales[i * 3 + d] = new_scale[d];
        p.child_sh0[i * 3 + d] = p.sh0[src * 3 + int(d)];
    }
    for (uint d = 0; d < 4; ++d)
        p.child_rotations[i * 4 + d] = q[d];
    p.opacity[src] = new_opacity;
    p.child_opacity[i] = new_opacity;
}

struct RefineFillParams {
    device const long* indices;
    device const float* means;
    device const float* rotations;
    device const float* scales;
    device const float* sh0;
    device const float* opacity;
    device float* dst_means;
    device float* dst_rotations;
    device float* dst_scales;
    device float* dst_sh0;
    device float* dst_opacity;
    device uchar* free_mask;
    uint count;
    uint rows;
};

kernel void refine_fill_slots(constant RefineFillParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const long target = p.indices[i];
    if (target < 0 || ulong(target) >= p.rows)
        return;
    const uint dst = uint(target);
    for (uint d = 0; d < 3; ++d) {
        p.dst_means[dst * 3 + d] = p.means[i * 3 + d];
        p.dst_scales[dst * 3 + d] = p.scales[i * 3 + d];
        p.dst_sh0[dst * 3 + d] = p.sh0[i * 3 + d];
    }
    for (uint d = 0; d < 4; ++d)
        p.dst_rotations[dst * 4 + d] = p.rotations[i * 4 + d];
    p.dst_opacity[dst] = p.opacity[i];
    if (p.free_mask != nullptr)
        p.free_mask[dst] = 0;
}

struct RefineCountsParams {
    device const uchar* bool0;
    device const uchar* bool1;
    device const float* float0;
    device const float* float1;
    device long* counts;
    uint count_bool0;
    uint count_bool1;
    uint count_float0;
    uint count_float1;
};

// One threadgroup counts set bytes in bool0/bool1 and values > 0 in float0/float1.
kernel void refine_counts(constant RefineCountsParams& p [[buffer(0)]],
                          uint lane [[thread_index_in_threadgroup]],
                          uint width [[threads_per_threadgroup]],
                          uint simd_lane [[thread_index_in_simdgroup]],
                          uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint4 partial[32];
    uint4 local = uint4(0u);
    if (p.bool0 != nullptr)
        for (uint i = lane; i < p.count_bool0; i += width)
            local.x += p.bool0[i] != 0 ? 1u : 0u;
    if (p.bool1 != nullptr)
        for (uint i = lane; i < p.count_bool1; i += width)
            local.y += p.bool1[i] != 0 ? 1u : 0u;
    if (p.float0 != nullptr)
        for (uint i = lane; i < p.count_float0; i += width)
            local.z += p.float0[i] > 0.0f ? 1u : 0u;
    if (p.float1 != nullptr)
        for (uint i = lane; i < p.count_float1; i += width)
            local.w += p.float1[i] > 0.0f ? 1u : 0u;
    local = uint4(simd_sum(local.x), simd_sum(local.y), simd_sum(local.z), simd_sum(local.w));
    if (simd_lane == 0)
        partial[simd_group] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        uint4 total = uint4(0u);
        for (uint g = 0; g < (width + 31) / 32; ++g)
            total += partial[g];
        p.counts[0] = long(total.x);
        p.counts[1] = long(total.y);
        p.counts[2] = long(total.z);
        p.counts[3] = long(total.w);
    }
}

struct RefineValuesParams {
    device float* values;
    device const uint* select_state;
    uint count;
};

kernel void refine_zero_nan(constant RefineValuesParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count && mrnf_is_nan(p.values[i]))
        p.values[i] = 0.0f;
}

// Divides by the positive median of mrnf_select_*, or zeros when no value is positive.
kernel void refine_divide_by_median(constant RefineValuesParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    if (p.select_state[0] == 0u) {
        p.values[i] = 0.0f;
        return;
    }
    p.values[i] /= fmax(as_type<float>(p.select_state[3]), 1e-9f);
}

struct RefineShareParams {
    device float* log_scales;
    device const float* error;
    device const float* shares;
    device const uchar* frozen;
    device float* scores;
    uint frozen_count;
    uint count;
    float limit;
};

kernel void refine_clip_scales(constant RefineShareParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count || mrnf_flag(p.frozen, p.frozen_count, i))
        return;
    const float share = p.shares[i];
    if (!(share > p.limit))
        return;
    const float delta = fmin(log(share / p.limit), as_type<float>(0x3ecf991fu)); // logf(1.5f)
    const float3 scale = float3(p.log_scales[i * 3], p.log_scales[i * 3 + 1], p.log_scales[i * 3 + 2]);
    p.log_scales[i * 3 + refine_axis_order(scale).x] -= delta;
}

kernel void refine_oversize_scores(constant RefineShareParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    p.scores[i] = mrnf_flag(p.frozen, p.frozen_count, i) ? 0.0f : oversize_split_score(p.error[i], p.shares[i], p.limit);
}

struct RefineMaskParams {
    device const float* opacity;
    device const float* rotations;
    device uchar* mask;
    uint count;
    float minimum_opacity;
};

static bool refine_zero_rotation(device const float* q) {
    return q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3] < 1e-8f;
}

kernel void refine_dead_mask(constant RefineMaskParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count)
        p.mask[i] = p.opacity[i] <= p.minimum_opacity || refine_zero_rotation(p.rotations + i * 4) ? 1 : 0;
}

kernel void refine_rotation_mask(constant RefineMaskParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count)
        p.mask[i] = refine_zero_rotation(p.rotations + i * 4) ? 1 : 0;
}
