// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// ExtraLossOps kernels: ports of regularization.cu. Reductions end in
// loss_reduce_final.

struct ExtraRegularizeParams {
    device const float* raw;
    device float* gradient; // accumulated into when has_gradient
    device float* partials;
    uint count;
    uint has_gradient;
    uint opacity; // 0: exp(x) scale term, 1: sigmoid(x) opacity term
    float grad_scale;
};

// Each element has one owner, so the CUDA atomicAdd is a plain add here.
kernel void extra_regularize(constant ExtraRegularizeParams& p [[buffer(0)]],
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
        const float x = p.raw[i];
        float value, derivative;
        if (p.opacity != 0) {
            value = 1.0f / (1.0f + exp(-x));
            const float e = exp(-fabs(x));
            const float d = 1.0f + e;
            derivative = e / (d * d);
        } else {
            value = exp(x);
            derivative = value;
        }
        sum += value;
        if (p.has_gradient != 0)
            p.gradient[i] += p.grad_scale * derivative;
    }
    sum = loss_group_sum(sum, scratch, rank, simd_lane, simd_group, simd_groups);
    if (rank == 0)
        p.partials[group] = sum;
}

