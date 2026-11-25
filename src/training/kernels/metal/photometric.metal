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
constant bool kPhotoMaskedWindow = kPhotoLossMode == 3 || kPhotoLossMode == 4;

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
// Weighted windows scale each tap by its mask weight and add the window mass.
template <bool Decoupled, bool Weighted>
static void photo_window(thread float* q, threadgroup const float* s0, threadgroup const float* s1,
                         threadgroup const float* s2, threadgroup const float* sm,
                         const uint l, const uint r, const bool pair, const float w) {
    const float a0 = s0[l], a1 = s1[l], a2 = Decoupled ? s2[l] : 0.0f;
    const float b0 = pair ? s0[r] : 0.0f, b1 = pair ? s1[r] : 0.0f;
    const float b2 = Decoupled && pair ? s2[r] : 0.0f;
    const float ml = Weighted ? sm[l] : 1.0f;
    const float mr = Weighted && pair ? sm[r] : pair ? 1.0f : 0.0f;
    q[0] += (a0 * ml + b0 * mr) * w;
    q[1] += (Decoupled ? a1 * ml + b1 * mr : a0 * a0 * ml + b0 * b0 * mr) * w;
    q[2] += (Decoupled ? a1 * a1 * ml + b1 * b1 * mr : a1 * ml + b1 * mr) * w;
    if (Decoupled) {
        q[3] += (a2 * ml + b2 * mr) * w;
        q[4] += (a2 * a2 * ml + b2 * b2 * mr) * w;
        q[5] += (a1 * a2 * ml + b1 * b2 * mr) * w;
        if (Weighted) q[6] += (ml + mr) * w;
    } else {
        q[3] += (a1 * a1 * ml + b1 * b1 * mr) * w;
        q[4] += (a0 * a1 * ml + b0 * b1 * mr) * w;
        // The standard path uses five moments; the weighted path adds mass.
        if (Weighted) q[5] += (ml + mr) * w;
    }
}

// Horizontal window sums of the tile's kPhotoSpan rows into conv.
template <bool Decoupled, bool Weighted>
static void photo_horizontal(const uint2 t, threadgroup const float* s0, threadgroup const float* s1,
                             threadgroup const float* s2, threadgroup const float* sm, threadgroup float* conv) {
    constexpr int quantities = (Decoupled ? 6 : 5) + (Weighted ? 1 : 0);
    for (uint row = t.y; row < uint(kPhotoSpan); row += kPhotoTile) {
        const uint o = row * kPhotoSpan + t.x + kPhotoHalo;
        float q[7];
        for (int k = 0; k < quantities; ++k)
            q[k] = 0.0f;
        for (int d = 1; d <= kPhotoHalo; ++d)
            photo_window<Decoupled, Weighted>(q, s0, s1, s2, sm, o - uint(d), o + uint(d), true, kSsimGauss[kPhotoHalo - d]);
        photo_window<Decoupled, Weighted>(q, s0, s1, s2, sm, o, o, false, kSsimGauss[kPhotoHalo]);
        for (int k = 0; k < quantities; ++k)
            conv[uint(k) * kPhotoConvCells + row * kPhotoTile + t.x] = q[k];
    }
}

// Vertical window sums of conv at this thread's output pixel.
template <int Quantities>
static void photo_vertical(const uint2 t, threadgroup const float* conv, thread float* v) {
    for (int k = 0; k < Quantities; ++k)
        v[k] = 0.0f;
    const uint column = t.y + kPhotoHalo;
    for (int d = 1; d <= kPhotoHalo; ++d) {
        const float w = kSsimGauss[kPhotoHalo - d];
        const uint top = (column - uint(d)) * kPhotoTile + t.x;
        const uint bottom = (column + uint(d)) * kPhotoTile + t.x;
        for (int k = 0; k < Quantities; ++k)
            v[k] += (conv[uint(k) * kPhotoConvCells + top] + conv[uint(k) * kPhotoConvCells + bottom]) * w;
    }
    for (int k = 0; k < Quantities; ++k)
        v[k] += conv[uint(k) * kPhotoConvCells + column * kPhotoTile + t.x] * kSsimGauss[kPhotoHalo];
}

// The SSIM terms of one pixel from its window moments v. Safe math: inlined
// under fast math, these sums fuse into the window sums differently in the
// masked and unmasked kernels, which then round equal moments apart.
struct PhotoSsimTerms {
    float A, B, Cn, D, luminance, cs, ssim;
};

template <bool Decoupled>
static PhotoSsimTerms photo_ssim_terms(thread const float* v) {
#pragma METAL fp math_mode(safe)
    PhotoSsimTerms s;
    if (Decoupled) {
        const float mu_c = v[0], mu_r = v[1], mu_g = v[3];
        const float sigma_r = fma(-mu_r, mu_r, v[2]);
        const float sigma_g = fma(-mu_g, mu_g, v[4]);
        const float sigma_rg = fma(-mu_r, mu_g, v[5]);
        s.A = mu_c * mu_c + mu_g * mu_g + kSsimC1;
        s.Cn = 2.0f * mu_c * mu_g + kSsimC1;
        s.luminance = s.Cn / s.A;
        s.B = sigma_r + sigma_g + kSsimC2;
        s.D = 2.0f * sigma_rg + kSsimC2;
        s.cs = s.D / s.B;
        s.ssim = s.luminance * s.cs;
    } else {
        const float mu1 = v[0], mu2 = v[2];
        const float mu1_sq = mu1 * mu1, mu2_sq = mu2 * mu2;
        const float sigma1 = v[1] - mu1_sq;
        const float sigma2 = v[3] - mu2_sq;
        const float sigma12 = v[4] - mu1 * mu2;
        s.A = mu1_sq + mu2_sq + kSsimC1;
        s.B = sigma1 + sigma2 + kSsimC2;
        s.Cn = 2.0f * mu1 * mu2 + kSsimC1;
        s.D = 2.0f * sigma12 + kSsimC2;
        s.luminance = 0.0f;
        s.ssim = (s.Cn * s.D) / (s.A * s.B);
        s.cs = s.D / s.B;
    }
    return s;
}

// SSIM and contrast-structure of one pixel from its window moments v. Writes the
// backward partials, scaled by derivative_weight, when the forward keeps them.
template <bool Decoupled>
static void photo_ssim(constant PhotoForwardParams& p, thread const float* v, const uint index,
                       const float derivative_weight, thread float& ssim, thread float& cs) {
    const PhotoSsimTerms s = photo_ssim_terms<Decoupled>(v);
    const float A = s.A, B = s.B, Cn = s.Cn, D = s.D;
    ssim = s.ssim;
    cs = s.cs;
    if (p.write_partials == 0)
        return;
    if (Decoupled) {
        const float mu_c = v[0], mu_r = v[1], mu_g = v[3];
        const float luminance = s.luminance;
        const float dl_dmu = 2.0f * (mu_g * A - mu_c * Cn) / (A * A);
        p.dm_mu[index] = half(cs * dl_dmu * derivative_weight);
        const float ds1 = p.ssim_weight * (-(luminance * D) / (B * B));
        const float ds12 = p.ssim_weight * ((2.0f * luminance) / B);
        p.dm_sigma1[index] = half(ds1 * derivative_weight);
        p.dm_sigma12[index] = half(ds12 * derivative_weight);
        p.raw_dm_mu[index] = half((ds1 * (-2.0f * mu_r) + ds12 * (-mu_g)) * derivative_weight);
    } else {
        const float mu1 = v[0], mu2 = v[2];
        const float dm_mu = (mu2 * 2.0f * D) / (A * B) - (mu2 * 2.0f * Cn) / (A * B) -
                            (mu1 * 2.0f * Cn * D) / (A * A * B) + (mu1 * 2.0f * Cn * D) / (A * B * B);
        p.dm_mu[index] = half(dm_mu * derivative_weight);
        p.dm_sigma1[index] = half((-Cn * D) / (A * B * B) * derivative_weight);
        p.dm_sigma12[index] = half((2.0f * Cn) / (A * B) * derivative_weight);
    }
}

// Loads one channel of the tile's span into s0 and s1 (decoupled: s0, s1, s2).
// With mask, also loads the span's mask weights into sm and returns whether any
// of them is not one.
template <bool Decoupled>
static bool photo_load_channel(constant PhotoForwardParams& p, const uint base, const int y0, const int x0,
                               const uint rank, threadgroup float* s0, threadgroup float* s1, threadgroup float* s2,
                               threadgroup float* sm, const bool mask, const float inv_mask_scale) {
    const int H = p.height, W = p.width;
    bool partial_support = false;
    for (uint i = rank; i < kPhotoSpanCells; i += uint(kPhotoTile * kPhotoTile)) {
        const int gy = y0 + int(i / kPhotoSpan) - kPhotoHalo;
        const int gx = x0 + int(i % kPhotoSpan) - kPhotoHalo;
        const bool in = gx >= 0 && gx < W && gy >= 0 && gy < H;
        const uint g = in ? base + uint(gy) * uint(W) + uint(gx) : 0;
        s0[i] = in ? p.prediction[g] : 0.0f;
        if (mask) {
            const float weight = (in ? photo_mask(p.mask, uint(gy) * uint(W) + uint(gx)) : 1.0f) * inv_mask_scale;
            sm[i] = weight;
            partial_support = partial_support || weight != 1.0f;
        }
        if (Decoupled) {
            s1[i] = in ? p.raw[g] : 0.0f;
            s2[i] = in ? photo_target(p.target, g) : 0.0f;
        } else {
            s1[i] = in ? photo_target(p.target, g) : 0.0f;
        }
    }
    return partial_support;
}

// The channels of one output tile; the first is already loaded. Weighted tiles
// normalize each window by its mass, the others keep the unmasked window sums.
// Each kind gets its own loop: picking the window code per channel inside one
// loop made the masked kernel about 15% slower on an M3 Ultra.
template <bool Decoupled, bool Weighted>
static void photo_tile_channels(constant PhotoForwardParams& p, const uint b, const int y0, const int x0, const uint2 t,
                                const uint rank, threadgroup float* s0, threadgroup float* s1, threadgroup float* s2,
                                threadgroup float* sm, threadgroup float* conv,
                                thread float& ssim_sum, thread float& cs_sum, thread float& l1_sum) {
    constexpr int kQuantities = Decoupled ? 6 : 5;
    const int H = p.height, W = p.width, C = p.channels;
    const int px = x0 + int(t.x), py = y0 + int(t.y);
    const bool inside = px < W && py < H;
    const uint plane = uint(H) * uint(W);
    const uint pixel = inside ? uint(py) * uint(W) + uint(px) : 0;
    const uint center = (t.y + kPhotoHalo) * kPhotoSpan + t.x + kPhotoHalo;
    for (int c = 0; c < C; ++c) {
        const uint base = (b * uint(C) + uint(c)) * plane;
        if (c > 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            photo_load_channel<Decoupled>(p, base, y0, x0, rank, s0, s1, s2, sm, false, 1.0f);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        photo_horizontal<Decoupled, Weighted>(t, s0, s1, s2, sm, conv);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float v[7];
        photo_vertical<kQuantities + (Weighted ? 1 : 0)>(t, conv, v);
        if (!inside)
            continue;
        const float inv_window = Weighted ? (v[kQuantities] >= 1.0e-30f ? 1.0f / v[kQuantities] : 0.0f) : 1.0f;
        const float derivative_weight = kPhotoMaskedWindow ? sm[center] * inv_window : 1.0f;
        if (Weighted)
            for (int k = 0; k < kQuantities; ++k) v[k] *= inv_window;

        const uint index = base + pixel;
        float ssim, cs;
        photo_ssim<Decoupled>(p, v, index, derivative_weight, ssim, cs);
        if (kPhotoChannelMean == 0) {
            if (p.write_map != 0)
                p.ssim_map[index] = kPhotoMaskedWindow && sm[center] <= 0.0f ? 1.0f : ssim;
            if (p.write_cs != 0)
                p.cs_map[index] = kPhotoMaskedWindow && sm[center] <= 0.0f ? 1.0f : cs;
        }
        ssim_sum += ssim;
        cs_sum += cs;
        if (kPhotoLossMode >= 2)
            l1_sum += fabs(s0[center] - (Decoupled ? s2[center] : s1[center]));
    }
}

// One output tile of the separable 11x11 gaussian SSIM. Sums the loss terms of
// the tile's pixels into loss_sum (and the mask into mask_sum) per kPhotoLossMode.
// weighted_end is one past the last tile whose mask window held a weight other
// than one; tiles only increase within a group, so it needs no reset per tile.
template <bool Decoupled>
static void photo_forward_tile(constant PhotoForwardParams& p, const uint tile, const uint2 t, const uint rank,
                               threadgroup float* s0, threadgroup float* s1, threadgroup float* s2,
                               threadgroup float* sm, threadgroup float* conv, threadgroup atomic_uint* weighted_end,
                               thread float& loss_sum, thread float& mask_sum) {
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
    const int anchor_x = min(x0 + kPhotoTile / 2, W - 1);
    const int anchor_y = min(y0 + kPhotoTile / 2, H - 1);
    const float inv_mask_scale = kPhotoMaskedWindow ? 1.0f / fmax(1.0f, photo_mask(p.mask, uint(anchor_y) * uint(W) + uint(anchor_x))) : 1.0f;

    // The first channel's loads also bring in the mask weights, which every channel shares.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool partial_support = photo_load_channel<Decoupled>(p, b * uint(C) * plane, y0, x0, rank, s0, s1, s2, sm,
                                                               kPhotoMaskedWindow, inv_mask_scale);
    if (kPhotoMaskedWindow && simd_any(partial_support) && simd_is_first())
        atomic_fetch_max_explicit(weighted_end, tile + 1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // A tile whose whole window has weight one keeps the unmasked rounding and mass one.
    const bool weighted = kPhotoMaskedWindow && atomic_load_explicit(weighted_end, memory_order_relaxed) == tile + 1;
    float ssim_sum = 0.0f, cs_sum = 0.0f, l1_sum = 0.0f;
    if (weighted)
        photo_tile_channels<Decoupled, true>(p, b, y0, x0, t, rank, s0, s1, s2, sm, conv, ssim_sum, cs_sum, l1_sum);
    else
        photo_tile_channels<Decoupled, false>(p, b, y0, x0, t, rank, s0, s1, s2, sm, conv, ssim_sum, cs_sum, l1_sum);
    if (!inside)
        return;

    const float inv_c = 1.0f / float(C);
    const float ssim_mean = ssim_sum * inv_c;
    if (kPhotoChannelMean != 0) {
        const bool ignored_center = kPhotoMaskedWindow && sm[center] <= 0.0f;
        if (p.write_map != 0)
            p.ssim_map[b * plane + pixel] = ignored_center ? 1.0f : ssim_mean;
        if (p.write_cs != 0)
            p.cs_map[b * plane + pixel] = ignored_center ? 1.0f : cs_sum * inv_c;
    }
    if (p.write_error != 0)
        p.error[pixel] = fmax(1.0f - (p.error_from_cs != 0 ? cs_sum : ssim_sum) * inv_c, 0.0f);

    if (kPhotoLossMode == 0 || kPhotoLossMode == 4)
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
                          threadgroup float* sm, threadgroup float* conv, threadgroup float* scratch,
                          threadgroup atomic_uint* weighted_end) {
    float loss_sum = 0.0f, mask_sum = 0.0f;
    // The barrier that starts each tile's first channel publishes the reset.
    if (kPhotoMaskedWindow && rank == 0)
        atomic_store_explicit(weighted_end, 0u, memory_order_relaxed);
    const uint tiles = p.tiles_x * p.tiles_y * uint(p.batch);
    for (uint tile = group; tile < tiles; tile += groups)
        photo_forward_tile<Decoupled>(p, tile, t, rank, s0, s1, s2, sm, conv, weighted_end, loss_sum, mask_sum);
    if (kPhotoLossMode == 0 || kPhotoLossMode == 4)
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
    threadgroup float sm[kPhotoSpanCells];
    threadgroup float conv[6 * kPhotoConvCells];
    threadgroup float scratch[32];
    threadgroup atomic_uint weighted_end;
    photo_forward<false>(p, t, rank, group.x, groups.x, simd_lane, simd_group, simd_groups, s0, s1, s1, sm, conv, scratch,
                         &weighted_end);
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
    threadgroup float sm[kPhotoSpanCells];
    threadgroup float conv[7 * kPhotoConvCells];
    threadgroup float scratch[32];
    threadgroup atomic_uint weighted_end;
    photo_forward<true>(p, t, rank, group.x, groups.x, simd_lane, simd_group, simd_groups, s0, s1, s2, sm, conv, scratch,
                        &weighted_end);
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
    const float input_mask = p.masked != 0 && inside ? photo_mask(p.mask, pixel) : 1.0f;
    const float inv_mask_sum = p.masked != 0 ? 1.0f / p.mask_sum[0] : 0.0f;
    const bool crop_y = p.valid_padding != 0 && H > 10, crop_x = p.valid_padding != 0 && W > 10;

    float chain_center = 0.0f;
    if (inside) {
        if (p.masked != 0)
            chain_center = input_mask * inv_mask_sum;
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
                    chain = inv_mask_sum;
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
            p.grad[base + pixel] = input_mask * (s0 + 2.0f * p1 * s1 + p2 * s2) + l1_weight * sign * chain_center;
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

// ---- Blur-tolerant SSIM (MRNF): ssim.cu blurTolerantForwardCUDA / blurTolerantBackwardCUDA ----
// G is a normalised Gaussian blur over the kept pixels (weight above zero): G v = n * T(keep v), n = 1 / T(keep),
// T the separable taps of the photo's blur sigma (radius R <= 5). The forward scores each channel and window with
// two decoupled SSIM branches from the same photo statistics, against the render and against G of the render, and
// takes their soft maximum (temperature 50); a flat photo window keeps the sharp branch alone. Three passes over
// 16x16 tiles: the forward (G inside the tile, the loss summed per group), the blurred branch's backward at G raw
// into fp16 (v, u), and the gradient pass: the sharp backward plus the adjoint blur T(v), T(u).
constant uint kBlurRadius [[function_constant(44)]];
// photo_blur_backward: 0 the blurred branch's pass, 1 the gradient pass.
constant uint kBlurGradient [[function_constant(45)]];
constant constexpr int kBlurMaxRadius = 5;
// The blur's reach around the SSIM span, as a pitch: an odd row length.
constant constexpr int kBlurReachPitch = kPhotoSpan + 2 * kBlurMaxRadius + 1;
// The adjoint blur's reach around a tile, as a pitch.
constant constexpr int kBlurFootPitch = kPhotoTile + 2 * kBlurMaxRadius + 1;
constant constexpr int kBlurDataPitch = kPhotoSpan + 1;
constant float kBlurTemperature = 50.0f;
constant float kBlurFlatVariance = 0.0005f;
constant float kBlurLn2 = 0.69314718055994531f;

struct PhotoBlurForwardParams {
    device const float* corrected;
    device const float* raw; // the corrected image again when not decoupled
    device const uchar* target;
    device const uchar* mask;
    device float* ssim_map; // [N,1,H,W]: the sharp SSIM
    device float* cs_map;   // [N,1,H,W]: the blend's contrast-structure
    device half4* sharp;    // [N,C,H,W]: raw mean, raw sigma^2, raw sigma12, corrected mean partials
    device half4* blurred;  // the blurred branch's partials in the same order
    device float* partials; // per group loss sums, then mask sums at kLossMaxGroups
    int height, width, channels, batch;
    uint tiles_x, tiles_y;
    float ssim_weight;
    float taps[kBlurMaxRadius + 1];
};

// The loss weight a pixel gives the blur: its mask (the masked loss), or 1 inside the image.
static float blur_weight(device const uchar* mask, const int y, const int x, const int H, const int W) {
    if (x < 0 || y < 0 || x >= W || y >= H)
        return 0.0f;
    return kPhotoMaskedWindow ? photo_mask(mask, uint(y) * uint(W) + uint(x)) : 1.0f;
}

// Four mask weights from index (packed: any alignment).
static float4 blur_mask4(device const uchar* mask, const uint index) {
    if (kPhotoMaskBytes != 0)
        return select(float4(0.0f), float4(1.0f), uchar4(*(device const packed_uchar4*)(mask + index)) != uchar4(0));
    return float4(*(device const packed_float4*)((device const float*)mask + index));
}

// Four photo values from index (packed: any alignment).
static float4 blur_target4(device const uchar* target, const uint index) {
    if (kPhotoTargetBytes != 0)
        return float4(uchar4(*(device const packed_uchar4*)(target + index))) * (1.0f / 255.0f);
    return float4(*(device const packed_float4*)((device const float*)target + index));
}

// One channel's loads: keep corrected and keep raw over the blur's reach (E x E) into src, the photo over the SSIM
// span into photo. Four columns per item; groups that leave the image or the row load element by element.
static void blur_load_channel(constant PhotoBlurForwardParams& p, const uint base, const int y0, const int x0,
                              const int R, const uint rank, threadgroup float2* src, threadgroup float* photo) {
    const int H = p.height, W = p.width, E = kPhotoSpan + 2 * R;
    const int reach_quads = (E + 3) / 4;
    for (uint i = rank; i < uint(E * reach_quads); i += uint(kPhotoTile * kPhotoTile)) {
        const int yy = int(i) / reach_quads, xx = 4 * (int(i) % reach_quads);
        const int gy = y0 + yy - kPhotoHalo - R, gx = x0 + xx - kPhotoHalo - R;
        threadgroup float2* out = src + yy * kBlurReachPitch + xx;
        if (xx + 4 <= E && gy >= 0 && gy < H && gx >= 0 && gx + 4 <= W) {
            const uint g = base + uint(gy) * uint(W) + uint(gx);
            const float4 c = float4(*(device const packed_float4*)(p.corrected + g));
            const float4 r = float4(*(device const packed_float4*)(p.raw + g));
            const bool4 keep = kPhotoMaskedWindow ? blur_mask4(p.mask, uint(gy) * uint(W) + uint(gx)) > 0.0f : bool4(true);
            out[0] = keep.x ? float2(c.x, r.x) : float2(0.0f);
            out[1] = keep.y ? float2(c.y, r.y) : float2(0.0f);
            out[2] = keep.z ? float2(c.z, r.z) : float2(0.0f);
            out[3] = keep.w ? float2(c.w, r.w) : float2(0.0f);
        } else {
            for (int k = 0; k < 4 && xx + k < E; ++k) {
                float2 value = float2(0.0f);
                if (blur_weight(p.mask, gy, gx + k, H, W) > 0.0f) {
                    const uint g = base + uint(gy) * uint(W) + uint(gx + k);
                    value = float2(p.corrected[g], p.raw[g]);
                }
                out[k] = value;
            }
        }
    }
    constexpr int kSpanQuads = (kPhotoSpan + 3) / 4;
    for (uint i = rank; i < uint(kPhotoSpan * kSpanQuads); i += uint(kPhotoTile * kPhotoTile)) {
        const int yy = int(i) / kSpanQuads, xx = 4 * (int(i) % kSpanQuads);
        const int gy = y0 + yy - kPhotoHalo, gx = x0 + xx - kPhotoHalo;
        threadgroup float* out = photo + yy * kPhotoSpan + xx;
        if (xx + 4 <= kPhotoSpan && gy >= 0 && gy < H && gx >= 0 && gx + 4 <= W) {
            const float4 y = blur_target4(p.target, base + uint(gy) * uint(W) + uint(gx));
            out[0] = y.x;
            out[1] = y.y;
            out[2] = y.z;
            out[3] = y.w;
        } else {
            for (int k = 0; k < 4 && xx + k < kPhotoSpan; ++k) {
                const bool in = gx + k >= 0 && gx + k < W && gy >= 0 && gy < H;
                out[k] = in ? photo_target(p.target, base + uint(gy) * uint(W) + uint(gx + k)) : 0.0f;
            }
        }
    }
}

// One decoupled SSIM branch from window means (bt_branch): returns S, writes cs and the partials in the order of
// the half4 stores.
static float blur_branch(const float mc, const float mr, const float mrr, const float my, const float myy,
                         const float mry, const float ssim_weight, thread float& cs, thread float4& partial) {
    const float A = mc * mc + my * my + kSsimC1;
    const float Cn = 2.0f * mc * my + kSsimC1;
    const float lum = Cn / A;
    const float B = (mrr - mr * mr) + (myy - my * my) + kSsimC2;
    const float D = 2.0f * (mry - mr * my) + kSsimC2;
    cs = D / B;
    const float ds1 = ssim_weight * (-(lum * D) / (B * B));
    const float ds12 = ssim_weight * ((2.0f * lum) / B);
    partial = float4(ds1 * (-2.0f * mr) + ds12 * (-my), ds1, ds12, cs * (2.0f * (my * A - mc * Cn) / (A * A)));
    return lum * cs;
}

// Five window moments at this thread's pixel. Horizontal sums of the SSIM span's 26 rows go to rows (five planes
// of 26x16), then the vertical sums at the pixel, times inv_window. Quantity selects the samples: 0 the sharp
// moments (corrected, raw, raw^2, raw photo, photo), 1 the blurred ones (photo^2, G corrected, G raw, (G raw)^2,
// G raw photo); src holds the values at reach position (span + R).
template <int Quantity>
static void blur_moments(const uint2 t, const uint rank, const int R, threadgroup const float2* src,
                         threadgroup const float* photo, threadgroup const float* weights, threadgroup float* rows,
                         const float inv_window, thread float* m) {
    constexpr uint kCells = uint(kPhotoSpan * kPhotoTile);
    for (uint o = rank; o < kCells; o += uint(kPhotoTile * kPhotoTile)) {
        const int r = int(o) / kPhotoTile, q0 = int(o) % kPhotoTile;
        float a[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int d = -kPhotoHalo; d <= kPhotoHalo; ++d) {
            const int q = q0 + kPhotoHalo + d;
            const float2 x = src[(r + R) * kBlurReachPitch + q + R];
            const float py = photo[r * kPhotoSpan + q];
            const float w = kPhotoMaskedWindow ? weights[r * kPhotoSpan + q] : 1.0f;
            const float tap = kSsimGauss[kPhotoHalo + d];
            float v[5];
            if (Quantity == 0) {
                v[0] = w * x.x, v[1] = w * x.y, v[2] = w * (x.y * x.y), v[3] = w * (x.y * py), v[4] = w * py;
            } else {
                v[0] = w * (py * py), v[1] = w * x.x, v[2] = w * x.y, v[3] = w * (x.y * x.y), v[4] = w * (x.y * py);
            }
            for (int k = 0; k < 5; ++k)
                a[k] += tap * v[k];
        }
        for (int k = 0; k < 5; ++k)
            rows[uint(k) * kCells + o] = a[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float a[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int d = -kPhotoHalo; d <= kPhotoHalo; ++d) {
        const uint cell = uint(int(t.y) + kPhotoHalo + d) * uint(kPhotoTile) + t.x;
        const float tap = kSsimGauss[kPhotoHalo + d];
        for (int k = 0; k < 5; ++k)
            a[k] += tap * rows[uint(k) * kCells + cell];
    }
    for (int k = 0; k < 5; ++k)
        m[k] = a[k] * inv_window;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The blur T of the two planes of src's reach (rows 0..E, columns 0..E) over the span: horizontal sums into rows
// (E x 26 float2), then vertical ones, times the support n, written back over src at the span's reach position.
// all_kept: n is 1 (the blur's whole reach is kept). With Support the plane is keep and the result goes to support.
template <bool Support>
static void blur_span(constant PhotoBlurForwardParams& p, const uint rank, const int R, const int y0, const int x0,
                      threadgroup float2* src, threadgroup float* support, threadgroup float2* rows,
                      const bool all_kept) {
    const int E = kPhotoSpan + 2 * R;
    for (uint o = rank; o < uint(E * kPhotoSpan); o += uint(kPhotoTile * kPhotoTile)) {
        const int r = int(o) / kPhotoSpan, q = int(o) % kPhotoSpan;
        float2 a = float2(0.0f);
        for (int d = -R; d <= R; ++d)
            a += p.taps[abs(d)] * src[r * kBlurReachPitch + q + R + d];
        rows[o] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = rank; o < uint(kPhotoSpan * kPhotoSpan); o += uint(kPhotoTile * kPhotoTile)) {
        const int r = int(o) / kPhotoSpan, q = int(o) % kPhotoSpan;
        float2 a = float2(0.0f);
        for (int d = -R; d <= R; ++d)
            a += p.taps[abs(d)] * rows[(r + R + d) * kPhotoSpan + q];
        if (Support) {
            const int gy = y0 + r - kPhotoHalo, gx = x0 + q - kPhotoHalo;
            const bool inside = gx >= 0 && gx < p.width && gy >= 0 && gy < p.height;
            support[o] = inside && a.x > 1.0e-30f ? 1.0f / a.x : 0.0f;
        } else {
            const float n = all_kept ? 1.0f : support[o];
            src[(r + R) * kBlurReachPitch + q + R] = float2(a.x * n, a.y * n);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// One 16x16 output tile of the forward. Sums the tile's loss terms into loss_sum (masked: times the pixel weight,
// and the weights into mask_sum).
static void blur_forward_tile(constant PhotoBlurForwardParams& p, const uint tile, const uint2 t, const uint rank,
                              threadgroup float2* src, threadgroup float* photo, threadgroup float* weights,
                              threadgroup float* support, threadgroup float2* rows2, threadgroup atomic_uint* flags,
                              thread float& loss_sum, thread float& mask_sum) {
    threadgroup float* rows = (threadgroup float*)rows2;
    const int H = p.height, W = p.width, C = p.channels, R = int(kBlurRadius);
    const int E = kPhotoSpan + 2 * R;
    const uint per_image = p.tiles_x * p.tiles_y;
    const uint b = tile / per_image;
    const uint within = tile % per_image;
    const int y0 = int(within / p.tiles_x) * kPhotoTile;
    const int x0 = int(within % p.tiles_x) * kPhotoTile;
    const int px = x0 + int(t.x), py = y0 + int(t.y);
    const bool inside = px < W && py < H;
    const uint plane = uint(H) * uint(W);
    const uint pixel = inside ? uint(py) * uint(W) + uint(px) : 0;
    const uint center_cell = (t.y + kPhotoHalo) * kPhotoSpan + t.x + kPhotoHalo;

    // flags: 0 a window weight other than one, 1 a pixel of the reach not kept, 2 a weighted pixel in the tile.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rank < 3)
        atomic_store_explicit(&flags[rank], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool partial_support = false, dropped = false;
    if (kPhotoMaskedWindow) {
        // A common scale cancels from normalised moments and centre/mass (load_mask_window).
        const int anchor_x = min(x0 + kPhotoTile / 2, W - 1), anchor_y = min(y0 + kPhotoTile / 2, H - 1);
        const float scale = 1.0f / fmax(1.0f, photo_mask(p.mask, uint(anchor_y) * uint(W) + uint(anchor_x)));
        for (uint i = rank; i < kPhotoSpanCells; i += uint(kPhotoTile * kPhotoTile)) {
            const int gy = y0 + int(i / kPhotoSpan) - kPhotoHalo, gx = x0 + int(i % kPhotoSpan) - kPhotoHalo;
            const bool in = gx >= 0 && gx < W && gy >= 0 && gy < H;
            const float value = (in ? photo_mask(p.mask, uint(gy) * uint(W) + uint(gx)) : 1.0f) * scale;
            weights[i] = value;
            partial_support = partial_support || value != 1.0f;
        }
        if (simd_any(inside && photo_mask(p.mask, pixel) != 0.0f) && simd_is_first())
            atomic_fetch_or_explicit(&flags[2], 1u, memory_order_relaxed);
        if (simd_any(partial_support) && simd_is_first())
            atomic_fetch_or_explicit(&flags[0], 1u, memory_order_relaxed);
    }
    for (uint i = rank; i < uint(E * E); i += uint(kPhotoTile * kPhotoTile)) {
        const int yy = int(i) / E, xx = int(i) % E;
        const bool keep = blur_weight(p.mask, y0 + yy - kPhotoHalo - R, x0 + xx - kPhotoHalo - R, H, W) > 0.0f;
        src[yy * kBlurReachPitch + xx] = float2(keep ? 1.0f : 0.0f, 0.0f);
        dropped = dropped || !keep;
    }
    if (simd_any(dropped) && simd_is_first())
        atomic_fetch_or_explicit(&flags[1], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool weighted_window = kPhotoMaskedWindow && atomic_load_explicit(&flags[0], memory_order_relaxed) != 0;
    const bool all_kept = atomic_load_explicit(&flags[1], memory_order_relaxed) == 0;
    if (kPhotoMaskedWindow && atomic_load_explicit(&flags[2], memory_order_relaxed) == 0) {
        // No loss term in the tile: zero partials and unit maps, as the CUDA forward.
        if (!inside)
            return;
        for (int c = 0; c < C; ++c) {
            const uint index = (b * uint(C) + uint(c)) * plane + pixel;
            p.sharp[index] = half4(0.0h);
            p.blurred[index] = half4(0.0h);
        }
        p.ssim_map[b * plane + pixel] = 1.0f;
        p.cs_map[b * plane + pixel] = 1.0f;
        return;
    }

    float inv_window = 1.0f, center = 1.0f;
    if (kPhotoMaskedWindow) {
        center = weights[center_cell];
        if (weighted_window) {
            // The window mass (load_mask_window's order: the centre tap, then the pairs outwards).
            for (uint o = rank; o < uint(kPhotoSpan * kPhotoTile); o += uint(kPhotoTile * kPhotoTile)) {
                const uint r = o / uint(kPhotoTile), x = o % uint(kPhotoTile) + uint(kPhotoHalo);
                float sum = weights[r * kPhotoSpan + x] * kSsimGauss[kPhotoHalo];
                for (int d = 1; d <= kPhotoHalo; ++d)
                    sum += (weights[r * kPhotoSpan + x - uint(d)] + weights[r * kPhotoSpan + x + uint(d)]) *
                           kSsimGauss[kPhotoHalo - d];
                rows[o] = sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint y = t.y + kPhotoHalo;
            float sum = rows[y * kPhotoTile + t.x] * kSsimGauss[kPhotoHalo];
            for (int d = 1; d <= kPhotoHalo; ++d)
                sum += (rows[(y - uint(d)) * kPhotoTile + t.x] + rows[(y + uint(d)) * kPhotoTile + t.x]) *
                       kSsimGauss[kPhotoHalo - d];
            inv_window = sum >= 1.0e-30f ? 1.0f / sum : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (!all_kept)
        blur_span<true>(p, rank, R, y0, x0, src, support, rows2, false);

    float ssim_sum = 0.0f, blend_sum = 0.0f, cs_sum = 0.0f, l1_sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        const uint base = (b * uint(C) + uint(c)) * plane;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        blur_load_channel(p, base, y0, x0, R, rank, src, photo);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ms[5], mb[5];
        blur_moments<0>(t, rank, R, src, photo, weights, rows, inv_window, ms);
        blur_span<false>(p, rank, R, y0, x0, src, support, rows2, all_kept);
        blur_moments<1>(t, rank, R, src, photo, weights, rows, inv_window, mb);
        if (!inside)
            continue;
        float cs0, cs1;
        float4 p0, p1;
        const float s0 = blur_branch(ms[0], ms[1], ms[2], ms[4], mb[0], ms[3], p.ssim_weight, cs0, p0);
        const float s1 = blur_branch(mb[1], mb[2], mb[3], ms[4], mb[0], mb[4], p.ssim_weight, cs1, p1);
        const float d = kBlurTemperature * (s0 - s1), e = exp(-fabs(d));
        const float high = 1.0f / (1.0f + e), low = e * high; // the soft-max weights of the larger, smaller branch
        const bool flat = mb[0] - ms[4] * ms[4] < kBlurFlatVariance;
        const float w0 = flat ? 1.0f : (d >= 0.0f ? high : low), w1 = flat ? 0.0f : (d >= 0.0f ? low : high);
        ssim_sum += s0;
        blend_sum += flat ? s0 : fmax(s0, s1) + (log(1.0f + e) - kBlurLn2) / kBlurTemperature;
        cs_sum += w0 * cs0 + w1 * cs1;
        // Partial scaling as the masked decoupled kernels: the masked raw mean partial is stored / 16.
        const float dw = center * inv_window;
        const float raw_mean_scale = kPhotoMaskedWindow ? dw * (1.0f / 16.0f) : dw;
        const uint index = base + pixel;
        p.sharp[index] = half4(half(p0.x * (w0 * raw_mean_scale)), half(p0.y * (w0 * dw)), half(p0.z * (w0 * dw)),
                               half(p0.w * (w0 * dw)));
        p.blurred[index] = half4(half(p1.x * (w1 * raw_mean_scale)), half(p1.y * (w1 * dw)), half(p1.z * (w1 * dw)),
                                 half(p1.w * (w1 * dw * p.ssim_weight)));
        l1_sum += fabs(p.corrected[index] - photo[center_cell]);
    }
    if (!inside)
        return;
    const float inv_c = 1.0f / float(C);
    const bool kept = !kPhotoMaskedWindow || center > 0.0f;
    p.ssim_map[b * plane + pixel] = kept ? ssim_sum * inv_c : 1.0f;
    p.cs_map[b * plane + pixel] = kept ? cs_sum * inv_c : 1.0f;
    const float blend = kept ? blend_sum * inv_c : 1.0f;
    const float per_pixel = (1.0f - p.ssim_weight) * l1_sum + float(C) * p.ssim_weight * (1.0f - blend);
    if (kPhotoMaskedWindow) {
        const float m = photo_mask(p.mask, pixel);
        loss_sum += per_pixel * m;
        if (b == 0)
            mask_sum += m;
        return;
    }
    // The valid padding crops each axis on its own.
    if ((H > 10 && (py < 5 || py >= H - 5)) || (W > 10 && (px < 5 || px >= W - 5)))
        return;
    loss_sum += per_pixel;
}

// Groups loop over tiles so at most kLossMaxGroups partial sums come out.
kernel void photo_blur_forward(constant PhotoBlurForwardParams& p [[buffer(0)]],
                               uint2 t [[thread_position_in_threadgroup]],
                               uint rank [[thread_index_in_threadgroup]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint2 groups [[threadgroups_per_grid]],
                               uint simd_lane [[thread_index_in_simdgroup]],
                               uint simd_group [[simdgroup_index_in_threadgroup]],
                               uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup float2 src[(kPhotoSpan + 2 * kBlurMaxRadius) * kBlurReachPitch];
    threadgroup float photo[kPhotoSpanCells];
    threadgroup float weights[kPhotoSpanCells];
    threadgroup float support[kPhotoSpanCells];
    threadgroup float2 rows[5 * kPhotoConvCells / 2]; // five planes of floats, or the blur's float2 rows
    threadgroup float scratch[32];
    threadgroup atomic_uint flags[3];
    float loss_sum = 0.0f, mask_sum = 0.0f;
    const uint tiles = p.tiles_x * p.tiles_y * uint(p.batch);
    for (uint tile = group.x; tile < tiles; tile += groups.x)
        blur_forward_tile(p, tile, t, rank, src, photo, weights, support, rows, flags, loss_sum, mask_sum);
    loss_sum = loss_group_sum(loss_sum, scratch, rank, simd_lane, simd_group, simd_groups);
    if (rank == 0)
        p.partials[group.x] = loss_sum;
    if (kPhotoMaskedWindow) {
        mask_sum = loss_group_sum(mask_sum, scratch, rank, simd_lane, simd_group, simd_groups);
        if (rank == 0)
            p.partials[kLossMaxGroups + group.x] = mask_sum;
    }
}

struct PhotoBlurBackwardParams {
    device const float* corrected;
    device const float* raw; // the corrected image again when not decoupled
    device const uchar* target;
    device const uchar* mask;
    device const float* mask_sum;   // masked: the loss denominator from the reduction
    device const half4* partials;   // the branch pass: blurred; the gradient pass: sharp
    device half2* vu;               // written by the branch pass, read by the gradient pass
    device float* grad;             // corrected image, or both images' sum without grad_raw
    device float* grad_raw;         // decoupled only
    int height, width, channels;
    uint tiles_y;
    float ssim_weight;
    float chain; // unmasked: 1 / the cropped element count
    uint decoupled;
    float taps[kBlurMaxRadius + 1];
};

// One 16x16 tile per group. Per channel: the window adjoint W' of the partials (window centres with partials only),
// T of two planes, then per pixel
//   branch: the blurred branch's backward at G raw = T(keep raw) / T(keep) and the photo, times weight * n: v, u;
//   gradient: the sharp backward (SSIM weight sw, L1 1 - sw) plus keep * T(v) and keep * T(u).
kernel void photo_blur_backward(constant PhotoBlurBackwardParams& p [[buffer(0)]],
                                uint2 t [[thread_position_in_threadgroup]],
                                uint2 g [[threadgroup_position_in_grid]],
                                uint rank [[thread_index_in_threadgroup]]) {
    threadgroup half4 data[kPhotoSpan * kBlurDataPitch];
    threadgroup float4 conv[kPhotoSpan * kPhotoTile];
    threadgroup float2 source[(kPhotoTile + 2 * kBlurMaxRadius) * kBlurFootPitch];
    threadgroup float2 rows[(kPhotoTile + 2 * kBlurMaxRadius) * kPhotoTile];
    threadgroup atomic_uint weighted;
    const bool gradient = kBlurGradient != 0;
    const int H = p.height, W = p.width, C = p.channels, R = int(kBlurRadius);
    const int F = kPhotoTile + 2 * R;
    const uint b = g.y / p.tiles_y;
    const int y0 = int(g.y % p.tiles_y) * kPhotoTile;
    const int x0 = int(g.x) * kPhotoTile;
    const int px = x0 + int(t.x), py = y0 + int(t.y);
    const bool inside = px < W && py < H;
    const uint plane = uint(H) * uint(W);
    const uint pixel = inside ? uint(py) * uint(W) + uint(px) : 0;
    const float center = blur_weight(p.mask, py, px, H, W);
    // Window centres with partials: masked partials carry their own window weights; unmasked, the valid padding
    // counts only centres 5 pixels or more inside the border.
    const auto valid = [&](const int y, const int x) {
        return x >= 0 && x < W && y >= 0 && y < H &&
               (kPhotoMaskedWindow || ((H <= 10 || (y >= 5 && y < H - 5)) && (W <= 10 || (x >= 5 && x < W - 5))));
    };
    if (kPhotoMaskedWindow) {
        // The masked rule: no gradient crosses into a tile without a weighted pixel.
        if (rank == 0)
            atomic_store_explicit(&weighted, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_any(center != 0.0f) && simd_is_first())
            atomic_fetch_or_explicit(&weighted, 1u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atomic_load_explicit(&weighted, memory_order_relaxed) == 0) {
            if (!inside)
                return;
            for (int c = 0; c < C; ++c) {
                const uint q = (b * uint(C) + uint(c)) * plane + pixel;
                if (!gradient)
                    p.vu[q] = half2(0.0h);
                else if (p.decoupled != 0)
                    p.grad[q] = p.grad_raw[q] = 0.0f;
                else
                    p.grad[q] = 0.0f;
            }
            return;
        }
    }
    const float chain = kPhotoMaskedWindow ? 1.0f / p.mask_sum[0] : p.chain;
    for (int c = 0; c < C; ++c) {
        const uint base = (b * uint(C) + uint(c)) * plane;
        for (uint i = rank; i < kPhotoSpanCells; i += uint(kPhotoTile * kPhotoTile)) {
            const int yy = int(i) / kPhotoSpan, xx = int(i) % kPhotoSpan;
            const int gy = y0 + yy - kPhotoHalo, gx = x0 + xx - kPhotoHalo;
            data[yy * kBlurDataPitch + xx] = valid(gy, gx) ? p.partials[base + uint(gy) * uint(W) + uint(gx)] : half4(0.0h);
        }
        for (uint i = rank; i < uint(F * F); i += uint(kPhotoTile * kPhotoTile)) {
            const int yy = int(i) / F, xx = int(i) % F;
            const int gy = y0 + yy - R, gx = x0 + xx - R;
            float2 value = float2(0.0f);
            if (gradient) {
                if (gx >= 0 && gx < W && gy >= 0 && gy < H)
                    value = float2(p.vu[base + uint(gy) * uint(W) + uint(gx)]);
            } else if (blur_weight(p.mask, gy, gx, H, W) > 0.0f) {
                value = float2(p.raw[base + uint(gy) * uint(W) + uint(gx)], 1.0f);
            }
            source[yy * kBlurFootPitch + xx] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint o = rank; o < uint(kPhotoSpan * kPhotoTile); o += uint(kPhotoTile * kPhotoTile)) {
            const int r = int(o) / kPhotoTile, q = int(o) % kPhotoTile;
            float4 a = float4(0.0f);
            for (int d = -kPhotoHalo; d <= kPhotoHalo; ++d)
                a += kSsimGauss[kPhotoHalo + d] * float4(data[r * kBlurDataPitch + q + kPhotoHalo + d]);
            conv[o] = a;
        }
        for (uint o = rank; o < uint(F * kPhotoTile); o += uint(kPhotoTile * kPhotoTile)) {
            const int r = int(o) / kPhotoTile, q = int(o) % kPhotoTile;
            float2 a = float2(0.0f);
            for (int d = -R; d <= R; ++d)
                a += p.taps[abs(d)] * source[r * kBlurFootPitch + q + R + d];
            rows[o] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (inside) {
            float4 m = float4(0.0f);
            for (int d = -kPhotoHalo; d <= kPhotoHalo; ++d)
                m += kSsimGauss[kPhotoHalo + d] * conv[uint(int(t.y) + kPhotoHalo + d) * uint(kPhotoTile) + t.x];
            float2 s = float2(0.0f);
            for (int d = -R; d <= R; ++d)
                s += p.taps[abs(d)] * rows[uint(int(t.y) + R + d) * uint(kPhotoTile) + t.x];
            const uint q = base + pixel;
            const float p2 = photo_target(p.target, q);
            const float s0 = (kPhotoMaskedWindow ? -16.0f : -1.0f) * m.x; // the masked mean partial is / 16
            if (gradient) {
                const float pc = p.corrected[q], pr = p.raw[q];
                const float keep = center > 0.0f ? 1.0f : 0.0f; // the adjoint lands on kept pixels
                const float sign = pc == p2 ? 0.0f : copysign(1.0f, pc - p2);
                const float l1 = (1.0f - p.ssim_weight) * sign * (kPhotoMaskedWindow ? center : (valid(py, px) ? 1.0f : 0.0f));
                const float gc = chain * (center * -p.ssim_weight * m.w + l1 + keep * s.x);
                const float gr = chain * (center * (s0 - 2.0f * pr * m.y - p2 * m.z) + keep * s.y);
                if (p.decoupled != 0) {
                    p.grad[q] = gc;
                    p.grad_raw[q] = gr;
                } else {
                    p.grad[q] = gc + gr;
                }
            } else {
                const float n = s.y > 1.0e-30f ? 1.0f / s.y : 0.0f, kn = center * n;
                p.vu[q] = half2(half(-kn * m.w), half(kn * (s0 - 2.0f * (s.x * n) * m.y - p2 * m.z)));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
