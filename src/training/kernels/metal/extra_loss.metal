// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// ExtraLossOps kernels: ports of regularization.cu and
// sparsity_optimizer_kernels.cu. Reductions end in loss_reduce_final.

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
            derivative = value * (1.0f - value);
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

struct ExtraAdmmParams {
    device float* gradient;
    device const float* sigmoid;
    device const float* z;
    device const float* u;
    uint count;
    uint accumulate;
    float rho;
    float grad_loss;
};

// grad = rho * (opa - z + u) * opa * (1 - opa) * grad_loss
kernel void extra_admm(constant ExtraAdmmParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const float opa = p.sigmoid[i];
    const float grad = p.rho * (opa - p.z[i] + p.u[i]) * (opa * (1.0f - opa)) * p.grad_loss;
    p.gradient[i] = p.accumulate != 0 ? p.gradient[i] + grad : grad;
}
