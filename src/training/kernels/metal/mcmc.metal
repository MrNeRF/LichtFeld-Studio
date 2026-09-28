// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// McmcOps kernels: ports of mcmc_kernels.cu.

constant constexpr int kMcmcRelocationMax = 51;

struct McmcRelocateParams {
    device const float* opacity;
    device const float* scales;
    device const int* ratios;
    device const float* coefficients;
    device float* new_opacity;
    device float* new_scales;
    uint count;
    float min_opacity;
};

// Equation (9) of "3D Gaussian Splatting as Markov Chain Monte Carlo".
kernel void mcmc_relocate(constant McmcRelocateParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const int n = p.ratios[i];
    const float opacity = fmin(fmax(p.opacity[i], 1e-6f), 1.0f - 1e-6f);
    const float new_opacity =
        fmin(fmax(1.0f - pow(1.0f - opacity, 1.0f / float(n)), fmax(1e-6f, p.min_opacity)), 1.0f - 1e-6f);
    p.new_opacity[i] = new_opacity;
    float denom_sum = 0.0f;
    for (int row = 1; row <= n; ++row)
        for (int k = 0; k <= row - 1; ++k)
            denom_sum += p.coefficients[(row - 1) * kMcmcRelocationMax + k] * pow(new_opacity, float(k + 1));
    float safe_denom = fmax(fabs(denom_sum), 1e-8f);
    if (denom_sum < 0.0f)
        safe_denom = -safe_denom;
    const float coeff = fmin(fmax(opacity / safe_denom, -1e6f), 1e6f);
    for (uint d = 0; d < 3; ++d)
        p.new_scales[i * 3 + d] = fmax(fabs(coeff * p.scales[i * 3 + d]), 1e-10f);
}

// glm's column-major product: column c of a * b.
static float3x3 mcmc_mul(const float3x3 a, const float3x3 b) {
    float3x3 r;
    for (int c = 0; c < 3; ++c)
        r[c] = a[0] * b[c][0] + a[1] * b[c][1] + a[2] * b[c][2];
    return r;
}

struct McmcNoiseParams {
    device const float* raw_opacity;
    device const float* raw_scales;
    device const float* raw_quats;
    device const uchar* frozen;
    device float* means;
    ulong seed;
    uint frozen_count;
    uint count;
    float lr;
};

kernel void mcmc_noise(constant McmcNoiseParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count || mrnf_flag(p.frozen, p.frozen_count, i))
        return;
    const float4 n = mrnf_curand_normal4(mrnf_philox_block(p.seed, i));

    const float3x3 s2 = float3x3(float3(exp(2.0f * p.raw_scales[i * 3]), 0.0f, 0.0f),
                                 float3(0.0f, exp(2.0f * p.raw_scales[i * 3 + 1]), 0.0f),
                                 float3(0.0f, 0.0f, exp(2.0f * p.raw_scales[i * 3 + 2])));
    float w = p.raw_quats[i * 4], x = p.raw_quats[i * 4 + 1], y = p.raw_quats[i * 4 + 2], z = p.raw_quats[i * 4 + 3];
    const float inv_norm = fmin(rsqrt(x * x + y * y + z * z + w * w), 1e+12f);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    const float x2 = x * x, y2 = y * y, z2 = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;
    const float3x3 r = float3x3(float3(1.f - 2.f * (y2 + z2), 2.f * (xy + wz), 2.f * (xz - wy)),
                                float3(2.f * (xy - wz), 1.f - 2.f * (x2 + z2), 2.f * (yz + wx)),
                                float3(2.f * (xz + wy), 2.f * (yz - wx), 1.f - 2.f * (x2 + y2)));
    const float3x3 covariance = mcmc_mul(mcmc_mul(r, s2), transpose(r));
    const float3 noise = float3(n.x, n.y, n.z);
    const float3 transformed = float3(
        covariance[0][0] * noise.x + covariance[1][0] * noise.y + covariance[2][0] * noise.z,
        covariance[0][1] * noise.x + covariance[1][1] * noise.y + covariance[2][1] * noise.z,
        covariance[0][2] * noise.x + covariance[1][2] * noise.y + covariance[2][2] * noise.z);

    // __frcp_rn is a correctly rounded reciprocal.
    const float opacity = precise::divide(1.0f, 1.0f + exp(-p.raw_opacity[i]));
    const float op_sigmoid = precise::divide(1.0f, 1.0f + exp(100.0f * opacity - 0.5f));
    const float factor = p.lr * op_sigmoid;
    p.means[i * 3] += factor * transformed.x;
    p.means[i * 3 + 1] += factor * transformed.y;
    p.means[i * 3 + 2] += factor * transformed.z;
}

struct McmcRowsParams {
    device const long* source;
    device const long* destination;
    device float* means;
    device float* sh0;
    device float* scales;
    device float* quats;
    device float* opacity;
    device const float* new_scales;
    device const float* new_opacity;
    uint count;
    uint rows;
};

kernel void mcmc_copy_rows(constant McmcRowsParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const long src = p.source[i];
    const long dst = p.destination[i];
    if (src < 0 || ulong(src) >= p.rows || dst < 0 || ulong(dst) >= p.rows)
        return;
    for (uint d = 0; d < 3; ++d) {
        p.means[dst * 3 + d] = p.means[src * 3 + d];
        p.sh0[dst * 3 + d] = p.sh0[src * 3 + d];
        p.scales[dst * 3 + d] = p.scales[src * 3 + d];
    }
    for (uint d = 0; d < 4; ++d)
        p.quats[dst * 4 + d] = p.quats[src * 4 + d];
    p.opacity[dst] = p.opacity[src];
}

kernel void mcmc_update_rows(constant McmcRowsParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const long target = p.destination[i];
    if (target < 0 || ulong(target) >= p.rows)
        return;
    for (uint d = 0; d < 3; ++d)
        p.scales[target * 3 + d] = p.new_scales[i * 3 + d];
    p.opacity[target] = p.new_opacity[i];
}

struct McmcSampleParams {
    device const float* cumsum;
    device const long* alive;
    device const float* opacity;
    device const float* raw_scales;
    device long* indices;
    device float* sampled_opacity;
    device float* sampled_scales;
    ulong seed;
    uint categories;
    uint samples;
};

// Inverse-CDF draw over the inclusive cumsum; alive maps a category to its row.
kernel void mcmc_sample(constant McmcSampleParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.samples)
        return;
    const float total = p.cumsum[p.categories - 1];
    if (total <= 0.0f) {
        p.indices[i] = 0;
        p.sampled_opacity[i] = 0.0f;
        for (uint d = 0; d < 3; ++d)
            p.sampled_scales[i * 3 + d] = 0.0f;
        return;
    }
    const float u = mrnf_curand_uniform(mrnf_philox_block(p.seed, i).x) * total;
    int left = 0;
    int right = int(p.categories) - 1;
    int selected = int(p.categories) - 1;
    while (left <= right) {
        const int mid = (left + right) / 2;
        if (p.cumsum[mid] >= u) {
            selected = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    const long row = p.alive != nullptr ? p.alive[selected] : long(selected);
    p.indices[i] = row;
    p.sampled_opacity[i] = p.opacity[row];
    for (uint d = 0; d < 3; ++d)
        p.sampled_scales[i * 3 + d] = exp(p.raw_scales[row * 3 + d]);
}

struct McmcFoldParams {
    device float* error_max;
    device float* densification;
    uint count;
};

kernel void mcmc_fold_error(constant McmcFoldParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    p.error_max[i] = fmax(p.error_max[i], p.densification[p.count + i]);
    p.densification[i] = 0.0f;
    p.densification[p.count + i] = 0.0f;
}
