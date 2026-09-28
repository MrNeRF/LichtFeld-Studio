// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// MaskOps kernels: port of mask_preprocess.cu. Masks are Float32, or UInt8/Bool
// when mask_bytes is set. Reductions end in loss_reduce_final.

struct MaskParams {
    device const float* alpha;
    device const uchar* mask;
    device const float* roi; // optional
    device float* out;       // weight, or grad_alpha
    device float* partials;
    uint count;
    uint mask_bytes;
    uint has_roi;
    uint mode;        // MaskPhotoMode / MaskOpacityMode: 0 BinaryGt0, 1 SegmentAndIgnore
    float power;
    float grad_scale;
    float keep_min;    // kMaskKeepMin
    float segment_min; // kMaskSegmentMin
};

// UInt8 band masks are normalized by 1/255; Float32 masks already are.
static float mask_normalized(constant MaskParams& p, const uint i) {
    return p.mask_bytes != 0 ? float(p.mask[i]) / 255.0f : ((device const float*)p.mask)[i];
}

// UInt8/Bool: nonzero is 1; Float32 passes through.
static float mask_as_float(constant MaskParams& p, const uint i) {
    return p.mask_bytes != 0 ? (p.mask[i] != 0 ? 1.0f : 0.0f) : ((device const float*)p.mask)[i];
}

kernel void mask_photometric_weight(constant MaskParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    float w = p.mode == 1 ? (mask_normalized(p, i) > p.keep_min ? 1.0f : 0.0f) : mask_as_float(p, i);
    if (p.has_roi != 0)
        w *= p.roi[i];
    p.out[i] = w;
}

kernel void mask_opacity_penalty(constant MaskParams& p [[buffer(0)]],
                                 uint rank [[thread_index_in_threadgroup]],
                                 uint width [[threads_per_threadgroup]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint groups [[threadgroups_per_grid]],
                                 uint simd_lane [[thread_index_in_simdgroup]],
                                 uint simd_group [[simdgroup_index_in_threadgroup]],
                                 uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[32];
    float sum = 0.0f;
    for (uint i = group * width + rank; i < p.count; i += groups * width) {
        float bg;
        if (p.mode == 1) {
            const float v = mask_normalized(p, i);
            bg = v >= p.segment_min && v <= p.keep_min ? 1.0f : 0.0f;
        } else {
            bg = 1.0f - mask_as_float(p, i);
        }
        float penalty = bg <= 0.0f ? 0.0f : (bg >= 1.0f && p.power > 0.0f ? 1.0f : pow(bg, p.power));
        if (p.has_roi != 0)
            penalty *= p.roi[i];
        p.out[i] = penalty * p.grad_scale;
        sum += p.alpha[i] * penalty;
    }
    sum = loss_group_sum(sum, scratch, rank, simd_lane, simd_group, simd_groups);
    if (rank == 0)
        p.partials[group] = sum;
}

kernel void mask_alpha_consistency(constant MaskParams& p [[buffer(0)]],
                                   uint rank [[thread_index_in_threadgroup]],
                                   uint width [[threads_per_threadgroup]],
                                   uint group [[threadgroup_position_in_grid]],
                                   uint groups [[threadgroups_per_grid]],
                                   uint simd_lane [[thread_index_in_simdgroup]],
                                   uint simd_group [[simdgroup_index_in_threadgroup]],
                                   uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[32];
    float sum = 0.0f;
    for (uint i = group * width + rank; i < p.count; i += groups * width) {
        const float diff = p.alpha[i] - mask_as_float(p, i);
        const float sign = diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f);
        const float roi = p.has_roi != 0 ? p.roi[i] : 1.0f;
        p.out[i] = sign * roi * p.grad_scale;
        sum += fabs(diff) * roi;
    }
    sum = loss_group_sum(sum, scratch, rank, simd_lane, simd_group, simd_groups);
    if (rank == 0)
        p.partials[group] = sum;
}
