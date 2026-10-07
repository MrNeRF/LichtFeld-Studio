// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// PhotometricOps kernels: ports of ssim.cu, ssim_reduction.cu and l1_loss.cu.
// Also holds the two-stage loss reduction the mask, extra-loss and geometry
// kernels share: per-group partial sums, then loss_reduce_final.

constant uint kPhotoTargetBytes [[function_constant(40)]];
constant uint kPhotoMaskBytes [[function_constant(41)]];
// Forward partial sums: 0 none, 1 SSIM mean, 2 fused L1+SSIM, 3 masked fused.
constant uint kPhotoLossMode [[function_constant(42)]];
// Forward maps: 1 writes the channel mean [N,1,H,W], 0 one map per channel.
constant uint kPhotoChannelMean [[function_constant(43)]];

constant float kSsimC1 = 0.01f * 0.01f;
constant float kSsimC2 = 0.03f * 0.03f;
constant float kSsimMaskEpsilon = 1e-8f;
constant float kSsimGauss[11] = {
    0.001028380123898387f, 0.0075987582094967365f, 0.036000773310661316f, 0.10936068743467331f,
    0.21300552785396576f,  0.26601171493530273f,   0.21300552785396576f,  0.10936068743467331f,
    0.036000773310661316f, 0.0075987582094967365f, 0.001028380123898387f};

// 16x16 output tiles with a 5-pixel halo, as the CUDA kernels.
constant constexpr int kPhotoTile = 16;
constant constexpr int kPhotoHalo = 5;
constant constexpr int kPhotoSpan = kPhotoTile + 2 * kPhotoHalo;
constant constexpr uint kPhotoSpanCells = uint(kPhotoSpan * kPhotoSpan);
constant constexpr uint kPhotoConvCells = uint(kPhotoSpan * kPhotoTile);
constant constexpr uint kLossMaxGroups = 1024;

// Sum over the threadgroup; the result is valid in thread 0. scratch holds one
// float per SIMD group.
static float loss_group_sum(float value, threadgroup float* scratch, const uint rank, const uint simd_lane,
                            const uint simd_group, const uint simd_groups) {
    value = simd_sum(value);
    if (simd_lane == 0)
        scratch[simd_group] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    if (rank == 0) {
        for (uint i = 0; i < simd_groups; ++i)
            total += scratch[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

struct LossReduceParams {
    device const float* partials;
    device float* result;
    device float* mask_sum;
    uint count;
    uint masked;
    float channels;
    float scale;
    float divisor;
    float offset;
    float denominator;
};

// result = offset + scale * (sum / divisor). Masked: partials hold the loss sums
// then the mask sums at kLossMaxGroups; result = loss / (mask * channels + eps).
kernel void loss_reduce_final(constant LossReduceParams& p [[buffer(0)]],
                              uint rank [[thread_index_in_threadgroup]],
                              uint width [[threads_per_threadgroup]],
                              uint simd_lane [[thread_index_in_simdgroup]],
                              uint simd_group [[simdgroup_index_in_threadgroup]],
                              uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[32];
    float sum = 0.0f;
    float mask = 0.0f;
    for (uint i = rank; i < p.count; i += width) {
        sum += p.partials[i];
        if (p.masked != 0)
            mask += p.partials[kLossMaxGroups + i];
    }
    sum = loss_group_sum(sum, scratch, rank, simd_lane, simd_group, simd_groups);
    if (p.masked != 0)
        mask = loss_group_sum(mask, scratch, rank, simd_lane, simd_group, simd_groups);
    if (rank != 0)
        return;
    if (p.masked != 0) {
        const float normalized = p.denominator > 0.0f ? p.denominator : mask * p.channels + kSsimMaskEpsilon;
        p.result[0] = sum / normalized;
        p.mask_sum[0] = normalized;
    } else {
        p.result[0] = p.offset + p.scale * (sum / p.divisor);
    }
}

static float photo_target(device const uchar* target, const uint index) {
    if (kPhotoTargetBytes != 0)
        return float(target[index]) * (1.0f / 255.0f);
    return ((device const float*)target)[index];
}

static float photo_mask(device const uchar* mask, const uint index) {
    if (kPhotoMaskBytes != 0)
        return mask[index] != 0 ? 1.0f : 0.0f;
    return ((device const float*)mask)[index];
}

struct PhotoL1Params {
    device const float* prediction;
    device const uchar* target;
    device float* grad;
    device float* partials;
    uint count;
    float grad_scale;
};

kernel void photo_l1(constant PhotoL1Params& p [[buffer(0)]],
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
        const float diff = p.prediction[i] - photo_target(p.target, i);
        sum += fabs(diff);
        p.grad[i] = diff > 0.0f ? p.grad_scale : (diff < 0.0f ? -p.grad_scale : 0.0f);
    }
    sum = loss_group_sum(sum, scratch, rank, simd_lane, simd_group, simd_groups);
    if (rank == 0)
        p.partials[group] = sum;
}

struct PhotoForwardParams {
    device const float* prediction; // the corrected image when decoupled
    device const float* raw;        // decoupled only
    device const uchar* target;
    device const uchar* mask;
    device float* ssim_map;
    device float* cs_map;
    device half* dm_mu;       // decoupled: d(ssim)/d mu(corrected)
    device half* dm_sigma1;   // decoupled: raw sigma^2 partial
    device half* dm_sigma12;  // decoupled: raw sigma12 partial
    device half* raw_dm_mu;   // decoupled only
    device float* error;      // optional [H,W] error map, single image
    device float* partials;   // per group loss sums, then mask sums at kLossMaxGroups
    int height, width, channels, batch;
    uint tiles_x, tiles_y;
    float ssim_weight;
    uint valid_padding;
    uint write_map, write_cs, write_partials, write_error, error_from_cs;
};

// Adds one tap pair (or the center tap when pair is false) of the horizontal
// window: x, x^2, y, y^2, xy; decoupled: corrected, raw, raw^2, gt, gt^2, raw*gt.
template <bool Decoupled>
static void photo_window(thread float* q, threadgroup const float* s0, threadgroup const float* s1,
                         threadgroup const float* s2, const uint l, const uint r, const bool pair, const float w) {
    const float a0 = s0[l], a1 = s1[l];
    const float b0 = pair ? s0[r] : 0.0f, b1 = pair ? s1[r] : 0.0f;
    if (Decoupled) {
        const float a2 = s2[l], b2 = pair ? s2[r] : 0.0f;
        q[0] += (a0 + b0) * w;
        q[1] += (a1 + b1) * w;
        q[2] += (a1 * a1 + b1 * b1) * w;
        q[3] += (a2 + b2) * w;
        q[4] += (a2 * a2 + b2 * b2) * w;
        q[5] += (a1 * a2 + b1 * b2) * w;
    } else {
        q[0] += (a0 + b0) * w;
        q[1] += (a0 * a0 + b0 * b0) * w;
        q[2] += (a1 + b1) * w;
        q[3] += (a1 * a1 + b1 * b1) * w;
        q[4] += (a0 * a1 + b0 * b1) * w;
    }
}

// One output tile of the separable 11x11 gaussian SSIM. Sums the loss terms of
// the tile's pixels into loss_sum (and the mask into mask_sum) per kPhotoLossMode.
template <bool Decoupled>
static void photo_forward_tile(constant PhotoForwardParams& p, const uint tile, const uint2 t, const uint rank,
                               threadgroup float* s0, threadgroup float* s1, threadgroup float* s2,
                               threadgroup float* conv, thread float& loss_sum, thread float& mask_sum) {
    constexpr int kQuantities = Decoupled ? 6 : 5;
    const int H = p.height, W = p.width, C = p.channels;
    const uint per_image = p.tiles_x * p.tiles_y;
    const uint b = tile / per_image;
    const uint within = tile % per_image;
    const int y0 = int(within / p.tiles_x) * kPhotoTile;
    const int x0 = int(within % p.tiles_x) * kPhotoTile;
    const int px = x0 + int(t.x), py = y0 + int(t.y);
    const bool inside = px < W && py < H;
    const uint plane = uint(H) * uint(W);
    const uint pixel = inside ? uint(py) * uint(W) + uint(px) : 0;
    const uint center = (t.y + kPhotoHalo) * kPhotoSpan + t.x + kPhotoHalo;

    float ssim_sum = 0.0f, cs_sum = 0.0f, l1_sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        const uint base = (b * uint(C) + uint(c)) * plane;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = rank; i < kPhotoSpanCells; i += uint(kPhotoTile * kPhotoTile)) {
            const int gy = y0 + int(i / kPhotoSpan) - kPhotoHalo;
            const int gx = x0 + int(i % kPhotoSpan) - kPhotoHalo;
            const bool in = gx >= 0 && gx < W && gy >= 0 && gy < H;
            const uint g = in ? base + uint(gy) * uint(W) + uint(gx) : 0;
            s0[i] = in ? p.prediction[g] : 0.0f;
            if (Decoupled) {
                s1[i] = in ? p.raw[g] : 0.0f;
                s2[i] = in ? photo_target(p.target, g) : 0.0f;
            } else {
                s1[i] = in ? photo_target(p.target, g) : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint row = t.y; row < uint(kPhotoSpan); row += kPhotoTile) {
            const uint o = row * kPhotoSpan + t.x + kPhotoHalo;
            float q[6];
            for (int k = 0; k < kQuantities; ++k)
                q[k] = 0.0f;
            for (int d = 1; d <= kPhotoHalo; ++d)
                photo_window<Decoupled>(q, s0, s1, s2, o - uint(d), o + uint(d), true, kSsimGauss[kPhotoHalo - d]);
            photo_window<Decoupled>(q, s0, s1, s2, o, o, false, kSsimGauss[kPhotoHalo]);
            for (int k = 0; k < kQuantities; ++k)
                conv[uint(k) * kPhotoConvCells + row * kPhotoTile + t.x] = q[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float v[6];
        for (int k = 0; k < kQuantities; ++k)
            v[k] = 0.0f;
        const uint column = t.y + kPhotoHalo;
        for (int d = 1; d <= kPhotoHalo; ++d) {
            const float w = kSsimGauss[kPhotoHalo - d];
            const uint top = (column - uint(d)) * kPhotoTile + t.x;
            const uint bottom = (column + uint(d)) * kPhotoTile + t.x;
            for (int k = 0; k < kQuantities; ++k)
                v[k] += (conv[uint(k) * kPhotoConvCells + top] + conv[uint(k) * kPhotoConvCells + bottom]) * w;
        }
        for (int k = 0; k < kQuantities; ++k)
            v[k] += conv[uint(k) * kPhotoConvCells + column * kPhotoTile + t.x] * kSsimGauss[kPhotoHalo];
        if (!inside)
            continue;

        const uint index = base + pixel;
        float ssim, cs;
        if (Decoupled) {
            const float mu_c = v[0], mu_r = v[1], mu_g = v[3];
            const float sigma_r = v[2] - mu_r * mu_r;
            const float sigma_g = v[4] - mu_g * mu_g;
            const float sigma_rg = v[5] - mu_r * mu_g;
            const float A = mu_c * mu_c + mu_g * mu_g + kSsimC1;
            const float Cn = 2.0f * mu_c * mu_g + kSsimC1;
            const float luminance = Cn / A;
            const float B = sigma_r + sigma_g + kSsimC2;
            const float D = 2.0f * sigma_rg + kSsimC2;
            cs = D / B;
            ssim = luminance * cs;
            if (p.write_partials != 0) {
                const float dl_dmu = 2.0f * (mu_g * A - mu_c * Cn) / (A * A);
                p.dm_mu[index] = half(cs * dl_dmu);
                const float ds1 = p.ssim_weight * (-(luminance * D) / (B * B));
                const float ds12 = p.ssim_weight * ((2.0f * luminance) / B);
                p.dm_sigma1[index] = half(ds1);
                p.dm_sigma12[index] = half(ds12);
                p.raw_dm_mu[index] = half(ds1 * (-2.0f * mu_r) + ds12 * (-mu_g));
            }
        } else {
            const float mu1 = v[0], mu2 = v[2];
            const float mu1_sq = mu1 * mu1, mu2_sq = mu2 * mu2;
            const float sigma1 = v[1] - mu1_sq;
            const float sigma2 = v[3] - mu2_sq;
            const float sigma12 = v[4] - mu1 * mu2;
            const float A = mu1_sq + mu2_sq + kSsimC1;
            const float B = sigma1 + sigma2 + kSsimC2;
            const float Cn = 2.0f * mu1 * mu2 + kSsimC1;
            const float D = 2.0f * sigma12 + kSsimC2;
            ssim = (Cn * D) / (A * B);
            cs = D / B;
            if (p.write_partials != 0) {
                const float dm_mu = (mu2 * 2.0f * D) / (A * B) - (mu2 * 2.0f * Cn) / (A * B) -
                                    (mu1 * 2.0f * Cn * D) / (A * A * B) + (mu1 * 2.0f * Cn * D) / (A * B * B);
                p.dm_mu[index] = half(dm_mu);
                p.dm_sigma1[index] = half((-Cn * D) / (A * B * B));
                p.dm_sigma12[index] = half((2.0f * Cn) / (A * B));
            }
        }
        if (kPhotoChannelMean == 0) {
            if (p.write_map != 0)
                p.ssim_map[index] = ssim;
            if (p.write_cs != 0)
                p.cs_map[index] = cs;
        }
        ssim_sum += ssim;
        cs_sum += cs;
        if (kPhotoLossMode >= 2)
            l1_sum += fabs(s0[center] - (Decoupled ? s2[center] : s1[center]));
    }
    if (!inside)
        return;

    const float inv_c = 1.0f / float(C);
    const float ssim_mean = ssim_sum * inv_c;
    if (kPhotoChannelMean != 0) {
        if (p.write_map != 0)
            p.ssim_map[b * plane + pixel] = ssim_mean;
        if (p.write_cs != 0)
            p.cs_map[b * plane + pixel] = cs_sum * inv_c;
    }
    if (p.write_error != 0)
        p.error[pixel] = fmax(1.0f - (p.error_from_cs != 0 ? cs_sum : ssim_sum) * inv_c, 0.0f);

    if (kPhotoLossMode == 0)
        return;
    const float per_pixel = (1.0f - p.ssim_weight) * l1_sum + float(C) * p.ssim_weight * (1.0f - ssim_mean);
    if (kPhotoLossMode == 3) {
        const float m = photo_mask(p.mask, pixel);
        loss_sum += per_pixel * m;
        if (b == 0)
            mask_sum += m;
        return;
    }
    // The reduction crops each axis on its own, as ssim_reduction.cu does.
    const bool crop_y = p.valid_padding != 0 && H > 10;
    const bool crop_x = p.valid_padding != 0 && W > 10;
    if ((crop_y && (py < 5 || py >= H - 5)) || (crop_x && (px < 5 || px >= W - 5)))
        return;
    loss_sum += kPhotoLossMode == 1 ? ssim_sum : per_pixel;
}

template <bool Decoupled>
static void photo_forward(constant PhotoForwardParams& p, const uint2 t, const uint rank, const uint group,
                          const uint groups, const uint simd_lane, const uint simd_group, const uint simd_groups,
                          threadgroup float* s0, threadgroup float* s1, threadgroup float* s2,
                          threadgroup float* conv, threadgroup float* scratch) {
    float loss_sum = 0.0f, mask_sum = 0.0f;
    const uint tiles = p.tiles_x * p.tiles_y * uint(p.batch);
    for (uint tile = group; tile < tiles; tile += groups)
        photo_forward_tile<Decoupled>(p, tile, t, rank, s0, s1, s2, conv, loss_sum, mask_sum);
    if (kPhotoLossMode == 0)
        return;
    loss_sum = loss_group_sum(loss_sum, scratch, rank, simd_lane, simd_group, simd_groups);
    if (rank == 0)
        p.partials[group] = loss_sum;
    if (kPhotoLossMode == 3) {
        mask_sum = loss_group_sum(mask_sum, scratch, rank, simd_lane, simd_group, simd_groups);
        if (rank == 0)
            p.partials[kLossMaxGroups + group] = mask_sum;
    }
}

// Groups loop over tiles so at most kLossMaxGroups partial sums come out.
kernel void photo_ssim_forward(constant PhotoForwardParams& p [[buffer(0)]],
                               uint2 t [[thread_position_in_threadgroup]],
                               uint rank [[thread_index_in_threadgroup]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint2 groups [[threadgroups_per_grid]],
                               uint simd_lane [[thread_index_in_simdgroup]],
                               uint simd_group [[simdgroup_index_in_threadgroup]],
                               uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup float s0[kPhotoSpanCells], s1[kPhotoSpanCells];
    threadgroup float conv[5 * kPhotoConvCells];
    threadgroup float scratch[32];
    photo_forward<false>(p, t, rank, group.x, groups.x, simd_lane, simd_group, simd_groups, s0, s1, s1, conv, scratch);
}

kernel void photo_ssim_forward_decoupled(constant PhotoForwardParams& p [[buffer(0)]],
                                         uint2 t [[thread_position_in_threadgroup]],
                                         uint rank [[thread_index_in_threadgroup]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         uint2 groups [[threadgroups_per_grid]],
                                         uint simd_lane [[thread_index_in_simdgroup]],
                                         uint simd_group [[simdgroup_index_in_threadgroup]],
                                         uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup float s0[kPhotoSpanCells], s1[kPhotoSpanCells], s2[kPhotoSpanCells];
    threadgroup float conv[6 * kPhotoConvCells];
    threadgroup float scratch[32];
    photo_forward<true>(p, t, rank, group.x, groups.x, simd_lane, simd_group, simd_groups, s0, s1, s2, conv, scratch);
}

struct PhotoBackwardParams {
    device const float* prediction;
    device const uchar* target;
    device const uchar* mask;
    device const float* mask_sum; // normalized mask sum from the forward reduction
    device const half* dm_mu;
    device const half* dm_sigma1;
    device const half* dm_sigma12;
    device float* grad;
    int height, width, channels;
    uint tiles_x, tiles_y;
    float ssim_weight;
    float grad_per_pixel;
    uint valid_padding;
    uint masked;
    uint has_sigma; // 0: the sigma partials are identically zero
};

// Chain rule through the gaussian window plus the L1 sign term (ssim.cu
// fusedL1SSIMBackwardCUDA / maskedFusedL1SSIMBackwardCUDA). ssim_weight 1 with
// grad_per_pixel 1/n is the pure SSIM backward for a loss of 1 - SSIM.
kernel void photo_ssim_backward(constant PhotoBackwardParams& p [[buffer(0)]],
                                uint2 t [[thread_position_in_threadgroup]],
                                uint2 g [[threadgroup_position_in_grid]],
                                uint rank [[thread_index_in_threadgroup]]) {
    threadgroup float d0[kPhotoSpanCells], d1[kPhotoSpanCells], d2[kPhotoSpanCells];
    threadgroup float conv[3 * kPhotoConvCells];
    const int H = p.height, W = p.width, C = p.channels;
    const uint b = g.y / p.tiles_y;
    const int y0 = int(g.y % p.tiles_y) * kPhotoTile;
    const int x0 = int(g.x) * kPhotoTile;
    const int px = x0 + int(t.x), py = y0 + int(t.y);
    const bool inside = px < W && py < H;
    const uint plane = uint(H) * uint(W);
    const uint pixel = inside ? uint(py) * uint(W) + uint(px) : 0;
    const float l1_weight = 1.0f - p.ssim_weight;
    const float inv_mask_sum = p.masked != 0 ? 1.0f / p.mask_sum[0] : 0.0f;
    const bool crop_y = p.valid_padding != 0 && H > 10, crop_x = p.valid_padding != 0 && W > 10;

    float chain_center = 0.0f;
    if (inside) {
        if (p.masked != 0)
            chain_center = photo_mask(p.mask, pixel) * inv_mask_sum;
        else if ((!crop_x || (px >= 5 && px < W - 5)) && (!crop_y || (py >= 5 && py < H - 5)))
            chain_center = p.grad_per_pixel;
    }

    for (int c = 0; c < C; ++c) {
        const uint base = (b * uint(C) + uint(c)) * plane;
        for (uint i = rank; i < kPhotoSpanCells; i += uint(kPhotoTile * kPhotoTile)) {
            const int gy = y0 + int(i / kPhotoSpan) - kPhotoHalo;
            const int gx = x0 + int(i % kPhotoSpan) - kPhotoHalo;
            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f;
            if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
                const uint local = uint(gy) * uint(W) + uint(gx);
                float chain;
                if (p.masked != 0)
                    chain = photo_mask(p.mask, local) * inv_mask_sum;
                else
                    chain = (!crop_x || (gx >= 5 && gx < W - 5)) && (!crop_y || (gy >= 5 && gy < H - 5)) ? p.grad_per_pixel : 0.0f;
                const float scale = -p.ssim_weight * chain;
                v0 = scale * float(p.dm_mu[base + local]);
                if (p.has_sigma != 0) {
                    v1 = scale * float(p.dm_sigma1[base + local]);
                    v2 = scale * float(p.dm_sigma12[base + local]);
                }
            }
            d0[i] = v0;
            d1[i] = v1;
            d2[i] = v2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint row = t.y; row < uint(kPhotoSpan); row += kPhotoTile) {
            const uint o = row * kPhotoSpan + t.x + kPhotoHalo;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
            for (int d = 1; d <= kPhotoHalo; ++d) {
                const float w = kSsimGauss[kPhotoHalo - d];
                const uint l = o - uint(d), r = o + uint(d);
                a0 += (d0[l] + d0[r]) * w;
                a1 += (d1[l] + d1[r]) * w;
                a2 += (d2[l] + d2[r]) * w;
            }
            a0 += d0[o] * kSsimGauss[kPhotoHalo];
            a1 += d1[o] * kSsimGauss[kPhotoHalo];
            a2 += d2[o] * kSsimGauss[kPhotoHalo];
            const uint cell = row * kPhotoTile + t.x;
            conv[cell] = a0;
            conv[kPhotoConvCells + cell] = a1;
            conv[2 * kPhotoConvCells + cell] = a2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (inside) {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
            const uint column = t.y + kPhotoHalo;
            for (int d = 1; d <= kPhotoHalo; ++d) {
                const float w = kSsimGauss[kPhotoHalo - d];
                const uint top = (column - uint(d)) * kPhotoTile + t.x;
                const uint bottom = (column + uint(d)) * kPhotoTile + t.x;
                s0 += (conv[top] + conv[bottom]) * w;
                s1 += (conv[kPhotoConvCells + top] + conv[kPhotoConvCells + bottom]) * w;
                s2 += (conv[2 * kPhotoConvCells + top] + conv[2 * kPhotoConvCells + bottom]) * w;
            }
            const uint mid = column * kPhotoTile + t.x;
            s0 += conv[mid] * kSsimGauss[kPhotoHalo];
            s1 += conv[kPhotoConvCells + mid] * kSsimGauss[kPhotoHalo];
            s2 += conv[2 * kPhotoConvCells + mid] * kSsimGauss[kPhotoHalo];
            const float p1 = p.prediction[base + pixel];
            const float p2 = photo_target(p.target, base + pixel);
            const float sign = p1 == p2 ? 0.0f : copysign(1.0f, p1 - p2);
            p.grad[base + pixel] = s0 + 2.0f * p1 * s1 + p2 * s2 + l1_weight * sign * chain_center;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

struct PhotoErrorParams {
    device const float* map;
    device float* error;
    uint plane;
    uint channels;
};

// error = max(0, 1 - mean over channels); map may alias error for one channel.
kernel void photo_map_to_error(constant PhotoErrorParams& p [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index >= p.plane)
        return;
    float sum = 0.0f;
    for (uint c = 0; c < p.channels; ++c)
        sum += p.map[c * p.plane + index];
    p.error[index] = fmax(1.0f - sum * (1.0f / float(p.channels)), 0.0f);
}
