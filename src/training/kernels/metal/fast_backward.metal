// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// FastRasterOps backward: ports of blend_backward_cu, preprocess_backward_cu
// and the fused joint Adam of kernel_utils.cuh. preprocess_backward is split
// in two kernels: SH (sh0/shN Adam and the SH mean gradient) and geometry
// (means, rotation, scaling, opacity). Both keep the 256-primitive threadgroup
// that owns one joint-bounds row.

// Per-primitive gradient helper row: mean2d (0, 1), conic (2..4), opacity
// (5), color (6..8; replaced by the SH mean gradient), depth (9).
constant constexpr uint kFastGradStride = 12u;
constant constexpr float kFastGradClamp = 1e4f;

static float fast_clamp_grad(const float g) { return fmin(fmax(g, -kFastGradClamp), kFastGradClamp); }

struct FastBlendBackwardParams {
    device const uint2* ranges;
    device const uint* values;
    device const FastMeanBox* mean_box;
    device const float4* conic_opacity;
    device const float4* color_depth;
    device const float4* normals;
    device const float* grad_image;
    device const float* grad_alpha;
    device const float* grad_depth;
    device const float* grad_normal;
    device const float* bg_color;
    device const float* bg_image;
    device const uint* n_contrib;
    device const float* final_transmittance;
    device atomic_float* grads;
    device atomic_float* normal_grads;
    device atomic_float* densification;
    device const float* error_map;
    device const float* edge_weight;
    device atomic_float* edge_score;
    uint n_instances, n_primitives, width, height;
    uint grid_w, unused0, unused1, unused2;
};

// Reverse-walk state of one pixel.
struct FastBackwardPixel {
    float2 pixel;
    float3 grad_color;
    float3 grad_normal;
    float transmittance;
    float grad_transmittance;
    float grad_depth;
    float error;
    float edge_weight;
    uint last;
    bool inside;
};

static FastBackwardPixel fast_backward_pixel(constant FastBlendBackwardParams& p, const uint2 pix, const bool inside,
                                             const uint final_index) {
    FastBackwardPixel s;
    s.pixel = float2(pix) + 0.5f;
    s.inside = inside;
    s.grad_color = float3(0.0f);
    s.grad_normal = float3(0.0f);
    s.transmittance = 1.0f;
    s.grad_transmittance = 0.0f;
    s.grad_depth = 0.0f;
    s.error = 1.0f;
    s.edge_weight = 0.0f;
    s.last = 0u;
    if (!inside)
        return s;
    const uint n_pixels = p.width * p.height;
    const uint pixel = p.width * pix.y + pix.x;
    s.last = p.n_contrib[pixel];
    s.transmittance = p.final_transmittance[final_index];
    s.grad_color = float3(p.grad_image[pixel], p.grad_image[n_pixels + pixel], p.grad_image[2u * n_pixels + pixel]);
    // dL/dT of output = color + T * background; alpha = 1 - T.
    const float3 bg = fast_background(p.bg_color, p.bg_image, pixel, n_pixels);
    float grad_alpha = -dot(s.grad_color, bg);
    if (p.grad_alpha != nullptr)
        grad_alpha += p.grad_alpha[pixel];
    s.grad_transmittance = -grad_alpha;
    if (kFastDepthGrad != 0u)
        s.grad_depth = p.grad_depth[pixel];
    if (kFastNormalChannel != 0u && p.grad_normal != nullptr)
        s.grad_normal =
            float3(p.grad_normal[pixel], p.grad_normal[n_pixels + pixel], p.grad_normal[2u * n_pixels + pixel]);
    if (kFastDensification != 0u && p.error_map != nullptr)
        s.error = p.error_map[pixel];
    if (kFastEdgeWeight != 0u)
        s.edge_weight = p.edge_weight[pixel];
    return s;
}

struct FastBlendAccum {
    float2 mean;
    float3 conic;
    float opacity;
    float3 color;
    float depth;
    float3 normal;
    float densification_weight;
    float densification_error;
    float edge;
};

// One pixel's contribution of one splat, walking back to front.
static bool fast_accumulate_pixel(thread FastBackwardPixel& s, thread FastBlendAccum& a, const uint tile_primitive,
                                  const float2 mean, const float4 conic_opacity, const float3 color,
                                  const float3 color_factor, const float depth, const float3 normal) {
    if (!s.inside || tile_primitive >= s.last)
        return false;
    const float2 d = mean - s.pixel;
    const float sigma_over_2 =
        0.5f * (conic_opacity.x * d.x * d.x + conic_opacity.z * d.y * d.y) + conic_opacity.y * d.x * d.y;
    if (!(sigma_over_2 >= 0.0f))
        return false;
    const float gaussian = exp(-sigma_over_2);
    const float unclamped_alpha = conic_opacity.w * gaussian;
    const float alpha = fmin(unclamped_alpha, kFastMaxFragmentAlpha);
    if (!(alpha >= kFastMinAlpha))
        return false;
    const bool saturated = unclamped_alpha >= kFastMaxFragmentAlpha;
    const float one_minus_alpha = 1.0f - alpha;
    const float transmittance_before = s.transmittance / fmax(one_minus_alpha, 1e-4f);
    const float weight = transmittance_before * alpha;
    a.edge += weight * s.edge_weight;
    const float normal_dot = kFastNormalChannel != 0u ? dot(normal, s.grad_normal) : 0.0f;
    a.densification_weight += weight;
    a.densification_error += weight * s.error;
    a.color += weight * s.grad_color * color_factor;
    const float dL_dalpha = dot(transmittance_before * color, s.grad_color) -
                            s.grad_transmittance * transmittance_before + transmittance_before * depth * s.grad_depth +
                            transmittance_before * normal_dot;
    a.opacity += saturated ? 0.0f : gaussian * dL_dalpha;
    a.depth += weight * s.grad_depth;
    if (kFastNormalChannel != 0u)
        a.normal += weight * s.grad_normal;
    const float helper = saturated ? 0.0f : -alpha * dL_dalpha;
    a.conic += 0.5f * helper * float3(d.x * d.x, d.x * d.y, d.y * d.y);
    a.mean += helper * float2(conic_opacity.x * d.x + conic_opacity.y * d.y, conic_opacity.y * d.x + conic_opacity.z * d.y);
    s.grad_transmittance = dot(s.grad_color, alpha * color) + alpha * depth * s.grad_depth + alpha * normal_dot +
                           s.grad_transmittance * one_minus_alpha;
    s.transmittance = transmittance_before;
    return true;
}

static void fast_atomic_add(device atomic_float* target, const float value) {
    atomic_fetch_add_explicit(target, value, memory_order_relaxed);
}

// One halving step of fast_transpose_sum16 with compile-time indices, so the
// values stay in registers.
template <uint W>
static void fast_transpose_step(thread float (&v)[16], const uint lane) {
    const bool upper = (lane & (2u * W)) != 0u;
#pragma unroll
    for (uint i = 0u; i < W; ++i) {
        const float send = upper ? v[i] : v[i + W];
        const float keep = upper ? v[i + W] : v[i];
        v[i] = keep + simd_shuffle_xor(send, ushort(2u * W));
    }
}

// Sums 16 values over the SIMD group in 16 shuffles instead of 16 separate
// reductions: each step halves the values a lane keeps. Lane l ends with the
// total of value (l >> 1) & 15, so the 16 atomics that follow issue at once.
static float fast_transpose_sum16(thread float (&v)[16], const uint lane) {
    fast_transpose_step<8u>(v, lane);
    fast_transpose_step<4u>(v, lane);
    fast_transpose_step<2u>(v, lane);
    fast_transpose_step<1u>(v, lane);
    return v[0] + simd_shuffle_xor(v[0], ushort(1));
}

// SIMD groups per tile in the backward blend. Each owns kFastBwdSubtiles of
// the tile's eight 8x4 sub-tiles, one pixel of each per lane. Fewer groups
// mean fewer reductions and atomics per splat and tile, which dominate on
// Apple GPUs; the per-pixel math does not change.
constant constexpr uint kFastBwdWarps = 2u;
constant constexpr uint kFastBwdSubtiles = 8u / kFastBwdWarps;
constant constexpr uint kFastBwdThreads = 32u * kFastBwdWarps;

// Port of blend_backward_cu: reverse walk over [0, T_eff) with the exact
// ellipse sub-tile cull; one atomic per splat per SIMD group.
kernel void fast_blend_backward(constant FastBlendBackwardParams& p [[buffer(0)]],
                                const uint tile_idx [[threadgroup_position_in_grid]],
                                const uint rank [[thread_index_in_threadgroup]],
                                const uint lane [[thread_index_in_simdgroup]],
                                const uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint s_prim[kFastBlendThreads];
    threadgroup float2 s_mean[kFastBlendThreads];
    threadgroup ushort4 s_bbox[kFastBlendThreads];
    threadgroup float4 s_conic[kFastBlendThreads];
    threadgroup float4 s_color[kFastBlendThreads];
    threadgroup float s_depth[kFastBlendThreads];
    threadgroup float3 s_normal[kFastBlendThreads];
    threadgroup uint s_max[kFastBwdWarps];

    const uint2 range = p.ranges[tile_idx];
    if (range.x >= range.y || range.y > p.n_instances)
        return;
    const int tile_n = int(range.y - range.x);

    const uint2 origin = uint2(tile_idx % p.grid_w, tile_idx / p.grid_w) * kFastTileSize;
    const uint2 local = uint2(lane & 7u, lane >> 3);
    uint2 sub[kFastBwdSubtiles];
    FastBackwardPixel px[kFastBwdSubtiles];
    uint last = 0u;
#pragma unroll
    for (uint k = 0u; k < kFastBwdSubtiles; ++k) {
        const uint st = warp + kFastBwdWarps * k;
        const uint2 offset = uint2((st & 1u) * 8u, (st >> 1) * 4u);
        sub[k] = origin + offset;
        const uint2 pixel = sub[k] + local;
        const bool inside = pixel.x < p.width && pixel.y < p.height;
        px[k] = fast_backward_pixel(p, pixel, inside,
                                    tile_idx * kFastTilePixels + (offset.y + local.y) * kFastTileSize + offset.x + local.x);
        last = max(last, px[k].last);
    }

    const uint warp_max = simd_max(last);
    if (lane == 0u)
        s_max[warp] = warp_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tile_max = 0u;
    for (uint w = 0u; w < kFastBwdWarps; ++w)
        tile_max = max(tile_max, s_max[w]);
    const int t_eff = min(tile_n, int(tile_max));
    if (t_eff <= 0)
        return;
    const int batch_size = t_eff <= 4 ? 32 : (t_eff <= 16 ? 64 : (t_eff <= 36 ? 96 : int(kFastBlendThreads)));
    const bool edge = kFastEdgeWeight != 0u && p.edge_score != nullptr;

    for (int batch_base = 0; batch_base < t_eff; batch_base += batch_size) {
        const int n_batch = min(batch_size, t_eff - batch_base);
        for (int slot = int(rank); slot < n_batch; slot += int(kFastBwdThreads)) {
            const uint prim = p.values[range.x + uint(t_eff - (batch_base + slot) - 1)];
            const bool valid = prim < p.n_primitives;
            s_prim[slot] = valid ? prim : kFastInvalid;
            if (valid) {
                const FastMeanBox geom = p.mean_box[prim];
                const float4 cd = p.color_depth[prim];
                s_mean[slot] = geom.mean;
                s_bbox[slot] = geom.bbox;
                s_conic[slot] = p.conic_opacity[prim];
                // Clamped color; w holds which channels stayed below the upper clamp.
                const uint factor_bits = (cd.x <= kFastMaxBlendColor ? 1u : 0u) |
                                         (cd.y <= kFastMaxBlendColor ? 2u : 0u) |
                                         (cd.z <= kFastMaxBlendColor ? 4u : 0u);
                s_color[slot] = float4(fmin(fmax(cd.xyz, 0.0f), kFastMaxBlendColor), as_type<float>(factor_bits));
                s_depth[slot] = cd.w;
                if (kFastNormalChannel != 0u)
                    s_normal[slot] = p.normals[prim].xyz;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int j_base = 0; j_base < n_batch; j_base += 32) {
            const int j_test = j_base + int(lane);
            uint mask[kFastBwdSubtiles];
            uint live = 0u;
            const bool testable = j_test < n_batch && s_prim[j_test] != kFastInvalid;
            const ushort4 bb = testable ? s_bbox[j_test] : ushort4(0);
            const float4 co_test = testable ? s_conic[j_test] : float4(0.0f);
            const float2 mean_test = testable ? s_mean[j_test] : float2(0.0f);
            const float power_test = log(co_test.w * kFastMinAlphaRcp);
#pragma unroll
            for (uint k = 0u; k < kFastBwdSubtiles; ++k) {
                const bool hit = testable && fast_bbox_hits(bb, sub[k]) &&
                                 fast_overlaps_subtile(mean_test, co_test.xyz, co_test.w, power_test, float2(sub[k]));
                mask[k] = uint(static_cast<simd_vote::vote_t>(simd_ballot(hit)));
                live |= mask[k];
            }
            while (live != 0u) {
                const uint bit = ctz(live);
                live &= live - 1u;
                const int j = j_base + int(bit);
                const uint tile_primitive = uint(t_eff - (batch_base + j) - 1);
                const uint prim = s_prim[j];
                const float2 mean = s_mean[j];
                const float4 co = s_conic[j];
                const float4 c = s_color[j];
                const uint bits = as_type<uint>(c.w);
                const float3 factor = float3((bits & 1u) ? 1.0f : 0.0f, (bits & 2u) ? 1.0f : 0.0f, (bits & 4u) ? 1.0f : 0.0f);
                const float depth = s_depth[j];
                const float3 normal = kFastNormalChannel != 0u ? s_normal[j] : float3(0.0f);

                FastBlendAccum a = {float2(0.0f), float3(0.0f), 0.0f, float3(0.0f), 0.0f, float3(0.0f), 0.0f, 0.0f, 0.0f};
                bool contributed = false;
#pragma unroll
                for (uint k = 0u; k < kFastBwdSubtiles; ++k) {
                    if ((mask[k] >> bit) & 1u)
                        contributed |= fast_accumulate_pixel(px[k], a, tile_primitive, mean, co, c.xyz, factor, depth, normal);
                }
                if (!simd_any(contributed))
                    continue;
                float v[16] = {a.mean.x, a.mean.y, a.conic.x, a.conic.y, a.conic.z, a.opacity,
                               a.color.x, a.color.y, a.color.z, a.depth, a.normal.x, a.normal.y,
                               a.normal.z, a.densification_weight, a.densification_error, a.edge};
                const float total = fast_transpose_sum16(v, lane);
                const uint component = (lane >> 1u) & 15u;
                // Zero totals leave the +0-initialized accumulators unchanged.
                if ((lane & 1u) == 0u && total != 0.0f) {
                    if (component < 10u) {
                        fast_atomic_add(p.grads + prim * kFastGradStride + component, fast_clamp_grad(total));
                    } else if (component < 13u) {
                        if (kFastNormalChannel != 0u)
                            fast_atomic_add(p.normal_grads + prim * 4u + (component - 10u), fast_clamp_grad(total));
                    } else if (component < 15u) {
                        if (kFastDensification != 0u)
                            fast_atomic_add(p.densification + (component - 13u) * p.n_primitives + prim, total);
                    } else if (edge) {
                        fast_atomic_add(p.edge_score + prim, total);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// The fused Adam update and the SH value re-encode compile in safe math mode
// (see joint_adam.metal): fast math would reassociate the moment updates. The
// step's sqrt and division stay approximate, as with CUDA's -use_fast_math.
#pragma METAL fp math_mode(safe)

// Inverse code ranges of new joint bounds, hoisted out of the per-cell encode.
static float2 fast_inverse_ranges(const float4 mm) {
    return float2(1.0f / fmax(mm.y - mm.x, kJointEps), 1.0f / fmax(mm.w - mm.z, kJointEps));
}

static float fast_adam_delta(const float step, const float m, const float v, const float bc2_sqrt_rcp,
                             const float eps) {
    return fast::divide(step * m, fast::sqrt(v) * bc2_sqrt_rcp + eps);
}

// One Adam binding, mirroring FusedAdamParam. `param` holds float values, or
// for shN IEEE half values or Q16 codes (value_bits == 16).
struct FastAdamGroup {
    device uchar* param;
    device uchar* packed;
    device float4* bounds;
    device float2* value_bounds;
    device const uchar* frozen;
    device const uchar* crop;
    device const float* screen_share;
    int frozen_n, crop_n, screen_share_n;
    int joint_bits, value_bits, value_cells;
    int primitives, elements, attributes;
    float step_size, bc2_sqrt_rcp, frozen_lr_scale, cropbox_lr_scale;
    float screen_share_limit, screen_share_penalty;
    uint enabled;
};

struct FastLane {
    uint lane, simd_lane, simd_group, simd_groups, block;
};

// New joint bounds of the threadgroup's 256-primitive block from per-thread
// (u_min, -u_max, s_min, -s_max); an untouched block gets zero bounds.
static float4 fast_block_bounds(const float4 local, threadgroup float4* scratch, const FastLane t) {
    const float4 r = threadgroup_min4(local, scratch, t.lane, t.simd_lane, t.simd_group, t.simd_groups);
    return r.x > -r.y ? float4(0.0f) : float4(r.x, -r.y, r.z, -r.w);
}

struct FastRowStep {
    float step;
    bool touch;
    bool apply;
};

static FastRowStep fast_row_step(constant FastAdamGroup& g, const uint p, const float step_scale) {
    FastRowStep r = {g.step_size * step_scale, g.enabled != 0u && g.packed != nullptr && g.bounds != nullptr, true};
    if (r.touch && g.frozen != nullptr && p < uint(g.frozen_n) && g.frozen[p] != 0u) {
        if (g.frozen_lr_scale == 0.0f)
            r.apply = false;
        else
            r.step *= g.frozen_lr_scale;
    }
    if (r.touch && g.crop != nullptr && p < uint(g.crop_n) && g.crop[p] != 0u) {
        if (g.cropbox_lr_scale == 0.0f)
            r.apply = false;
        else
            r.step *= g.cropbox_lr_scale;
    }
    return r;
}

// Port of adam_step_row_joint over a contiguous row of one primitive. Every
// thread of the threadgroup must call it.
template <int BITS>
static void fast_adam_row(constant FastAdamGroup& g, thread const float* grads, const uint p, const uint row_elements,
                          const float step_scale, const float beta1, const float beta2, const float eps,
                          threadgroup float4* scratch, const FastLane t, const bool owner) {
    const FastRowStep r = fast_row_step(g, p, step_scale);
    const uint n_attr = uint(max(g.attributes, 1));
    const uint base = p * n_attr;
    const bool touch = owner && r.touch && g.attributes > 0 && base < uint(g.elements);
    const uint row = touch ? min(n_attr, uint(g.elements) - base) : 0u;
    const uint active = min(row_elements, row);
    const float4 old_mm = touch ? g.bounds[t.block] : float4(0.0f);
    device float* param = reinterpret_cast<device float*>(g.param);
    float us_u[4];
    float us_s[4];
    float4 local = float4(1e30f);
    for (uint i = 0; i < row; ++i) {
        const float2 mv = JointCodec<BITS>::decode_g1g2(g.packed, long(base + i), old_mm);
        float m = mv.x;
        float v = mv.y;
        if (r.apply) {
            const float grad = i < active ? grads[i] : 0.0f;
            float hinge = 0.0f;
            if (i < active && g.screen_share != nullptr && p < uint(g.screen_share_n))
                hinge = screen_share_hinge_extra_grad(g.screen_share[p], g.screen_share_limit,
                                                      g.screen_share_penalty, mv.y, g.bc2_sqrt_rcp, eps);
            m = beta1 * mv.x + (1.0f - beta1) * (grad + hinge);
            v = beta2 * mv.y + (1.0f - beta2) * grad * grad;
            if (i < active)
                param[base + i] -= fast_adam_delta(r.step, m, v, g.bc2_sqrt_rcp, eps);
        }
        const float2 us = JointCodec<BITS>::g1g2_to_us(m, v);
        us_u[i] = us.x;
        us_s[i] = us.y;
        local = fmin(local, float4(us.x, -us.x, us.y, -us.y));
    }
    const float4 mm = fast_block_bounds(local, scratch, t);
    if (g.enabled != 0u && g.bounds != nullptr && t.lane == 0u)
        g.bounds[t.block] = mm;
    const float2 inv = fast_inverse_ranges(mm);
    for (uint i = 0; i < row; ++i)
        JointCodec<BITS>::encode_us(g.packed, long(base + i), us_u[i], us_s[i], mm.x, mm.z, inv.x, inv.y);
}

// Every thread of the threadgroup must call it; only owners update a row.
static void fast_adam_step(constant FastAdamGroup& g, thread const float* grads, const uint p, const uint row_elements,
                           const float step_scale, const float beta1, const float beta2, const float eps,
                           threadgroup float4* scratch, const FastLane t, const bool owner = true) {
    if (g.joint_bits == 16)
        fast_adam_row<16>(g, grads, p, row_elements, step_scale, beta1, beta2, eps, scratch, t, owner);
    else if (g.joint_bits == 8)
        fast_adam_row<8>(g, grads, p, row_elements, step_scale, beta1, beta2, eps, scratch, t, owner);
}

// One cell's moment update from its codes (u_q, s_q).
static float2 fast_shN_moment(const float grad, const float2 codes, const float4 old_mm,
                              const bool apply, const bool update, const float beta1, const float beta2,
                              const float step, const float eps, const float bc2_sqrt_rcp, thread float& value) {
    const float2 mv = JointCodec<8>::us_to_g1g2(JointCodec<8>::decode_codes(codes, old_mm));
    float m = mv.x;
    float v = mv.y;
    if (apply) {
        m = beta1 * mv.x + (1.0f - beta1) * grad;
        v = beta2 * mv.y + (1.0f - beta2) * grad * grad;
        if (update)
            value -= fast_adam_delta(step, m, v, bc2_sqrt_rcp, eps);
    }
    return JointCodec<8>::g1g2_to_us(m, v);
}

// The 8-bit moment codes of a float4 slot: 4 cells of (u_q, s_q), 8 bytes.
static float2 fast_slot_codes(const uint2 packed, const uint c) {
    const uint word = c < 2u ? packed.x : packed.y;
    const uint shift = (c & 1u) * 16u;
    return float2((word >> shift) & 0xffu, (word >> (shift + 8u)) & 0xffu);
}

static float4 fast_shN_load(constant FastAdamGroup& g, const bool q16, const bool f16, const uint p, const uint k,
                            const uint slot, const uint cells, const float2 vmm) {
    if (q16) {
        device const ushort* codes = reinterpret_cast<device const ushort*>(g.param);
        float4 v = float4(0.0f);
        for (uint c = 0; c < 4u; ++c) {
            const uint cell = k * 4u + c;
            if (cell < cells)
                v[c] = sh_q16_decode(codes[sh_q16_index(p, cell, cells)], vmm.x, vmm.y);
        }
        return v;
    }
    if (f16)
        return float4(reinterpret_cast<device const half4*>(g.param)[slot]);
    return reinterpret_cast<device const float4*>(g.param)[slot];
}

// Gradient of float4 slot k from dL/dcolor and the SH basis; rest
// coefficients beyond the active degree get zero.
static float4 fast_shN_slot_grad(const uint k, const bool compute, thread const float* basis, const float3 grad_color) {
    float4 g = float4(0.0f);
    if (!compute)
        return g;
    for (uint c = 0; c < 4u; ++c) {
        const uint linear = k * 4u + c;
        if (linear / 3u < kFastShBases - 1u)
            g[c] = basis[linear / 3u] * grad_color[linear % 3u];
    }
    return g;
}

// SH rest slots per thread in fast_backward_sh: a primitive's twelve float4
// slots spread over kFastShParts threads, so each keeps its updated moments in
// registers between the bound reduction and the encode.
constant constexpr uint kFastShSlotsPerThread = 3u;
constant constexpr uint kFastShParts = kShMaxSlots / kFastShSlotsPerThread;

// Port of apply_shN_grads_packed_joint (8-bit moments on float4-slot cells) for
// the slots [part * kFastShSlotsPerThread, ...) of primitive p. Every thread of
// the threadgroup must call it.
static void fast_adam_shN(constant FastAdamGroup& g, const uint p, const uint part, const uint layout_rest,
                          const float3 grad_color, const float3 direction, const bool compute, const float beta1,
                          const float beta2, const float eps, threadgroup float4* scratch, const FastLane t) {
    const bool q16 = g.value_bits == 16 && g.value_bounds != nullptr && g.value_cells > 0;
    const bool f16 = g.value_bits == 16 && !q16;
    const uint cells = q16 ? uint(g.value_cells) : 0u;
    const uint layout_slots = sh_float4_slots(layout_rest);
    const FastRowStep r = fast_row_step(g, p, 1.0f);
    const bool touch = r.touch && layout_slots > 0u && !(g.primitives > 0 && p >= uint(g.primitives));
    const uint active_slots = kFastShBases > 9u ? 12u : (kFastShBases > 4u ? 6u : 3u);
    const float4 old_mm = touch ? g.bounds[t.block] : float4(0.0f);
    const float2 old_vmm = q16 ? g.value_bounds[t.block] : float2(0.0f);
    float basis[15];
    if (compute)
        fast_sh_basis(direction, basis);

    float4 us_u[kFastShSlotsPerThread];
    float4 us_s[kFastShSlotsPerThread];
    float4 updated[kFastShSlotsPerThread];
    float4 local = float4(1e30f);
    float2 value_local = float2(1e30f);
#pragma unroll
    for (uint j = 0; j < kFastShSlotsPerThread; ++j) {
        const uint k = part * kFastShSlotsPerThread + j;
        if (!touch || k >= layout_slots)
            continue;
        const uint slot = sh_swizzled_index(p, k, layout_rest);
        const bool active_slot = k < active_slots;
        const float4 grad = fast_shN_slot_grad(k, compute && active_slot, basis, grad_color);
        float4 values = fast_shN_load(g, q16, f16, p, k, slot, cells, old_vmm);
        const uint2 packed = reinterpret_cast<device const uint2*>(g.packed)[slot];
        for (uint c = 0; c < 4u; ++c) {
            float value = values[c];
            const float2 us = fast_shN_moment(grad[c], fast_slot_codes(packed, c), old_mm, r.apply, active_slot,
                                              beta1, beta2, r.step, eps, g.bc2_sqrt_rcp, value);
            values[c] = value;
            us_u[j][c] = us.x;
            us_s[j][c] = us.y;
            if (q16 && k * 4u + c < cells)
                value_local = fmin(value_local, float2(value, -value));
            local = fmin(local, float4(us.x, -us.x, us.y, -us.y));
        }
        updated[j] = values;
        if (r.apply && active_slot) {
            if (f16)
                reinterpret_cast<device half4*>(g.param)[slot] = half4(values);
            else if (!q16)
                reinterpret_cast<device float4*>(g.param)[slot] = values;
        }
    }

    const float4 mm = fast_block_bounds(local, scratch, t);
    const float2 vmm = q16 ? threadgroup_q16_bounds(value_local.x, -value_local.y, scratch, t.lane, t.simd_lane,
                                                    t.simd_group, t.simd_groups)
                           : float2(0.0f);
    if (t.lane == 0u) {
        if (g.enabled != 0u && g.bounds != nullptr)
            g.bounds[t.block] = mm;
        if (q16)
            g.value_bounds[t.block] = vmm;
    }

    if (!touch)
        return;
    const float2 inv = fast_inverse_ranges(mm);
#pragma unroll
    for (uint j = 0; j < kFastShSlotsPerThread; ++j) {
        const uint k = part * kFastShSlotsPerThread + j;
        if (k >= layout_slots)
            continue;
        const uint slot = sh_swizzled_index(p, k, layout_rest);
        uint2 encoded = uint2(0u);
        for (uint c = 0; c < 4u; ++c) {
            const float2 q = JointCodec<8>::encode_codes(us_u[j][c], us_s[j][c], mm.x, mm.z, inv.x, inv.y);
            const uint cell_bits = (uint(q.x) | (uint(q.y) << 8u)) << ((c & 1u) * 16u);
            if (c < 2u)
                encoded.x |= cell_bits;
            else
                encoded.y |= cell_bits;
            if (q16 && k * 4u + c < cells)
                reinterpret_cast<device ushort*>(g.param)[sh_q16_index(p, k * 4u + c, cells)] =
                    sh_q16_encode(updated[j][c], vmm.x, vmm.y);
        }
        reinterpret_cast<device uint2*>(g.packed)[slot] = encoded;
    }
}

#pragma METAL fp math_mode(fast)

// dL/dmean through the SH view direction (convert_sh_to_color_backward_grads).
static float3 fast_sh_mean_grad(device const uchar* shN, device const float2* bounds, const uint storage, const uint p,
                                const float3 mean, const float3 camera, const uint layout_rest, const uint q16_cells,
                                const float3 grad_color) {
    if (kFastShBases <= 1u)
        return float3(0.0f);
    const float3 raw = mean - camera;
    const float3 c0 = fast_sh_rest(shN, bounds, storage, p, 0u, layout_rest, q16_cells);
    const float3 c1 = fast_sh_rest(shN, bounds, storage, p, 1u, layout_rest, q16_cells);
    const float3 c2 = fast_sh_rest(shN, bounds, storage, p, 2u, layout_rest, q16_cells);
    float3 gx = -0.48860251190291987f * c2;
    float3 gy = -0.48860251190291987f * c0;
    float3 gz = 0.48860251190291987f * c1;
    if (kFastShBases > 4u) {
        const float3 d = fast_safe_normalize(raw);
        const float x = d.x, y = d.y, z = d.z;
        const float xx = x * x, yy = y * y, zz = z * z, xy = x * y, xz = x * z, yz = y * z;
        const float3 c3 = fast_sh_rest(shN, bounds, storage, p, 3u, layout_rest, q16_cells);
        const float3 c4 = fast_sh_rest(shN, bounds, storage, p, 4u, layout_rest, q16_cells);
        const float3 c5 = fast_sh_rest(shN, bounds, storage, p, 5u, layout_rest, q16_cells);
        const float3 c6 = fast_sh_rest(shN, bounds, storage, p, 6u, layout_rest, q16_cells);
        const float3 c7 = fast_sh_rest(shN, bounds, storage, p, 7u, layout_rest, q16_cells);
        gx = gx + (1.0925484305920792f * y) * c3 + (-1.0925484305920792f * z) * c6 + (1.0925484305920792f * x) * c7;
        gy = gy + (1.0925484305920792f * x) * c3 + (-1.0925484305920792f * z) * c4 + (-1.0925484305920792f * y) * c7;
        gz = gz + (-1.0925484305920792f * y) * c4 + (1.8923493915151202f * z) * c5 + (-1.0925484305920792f * x) * c6;
        if (kFastShBases > 9u) {
            const float3 c8 = fast_sh_rest(shN, bounds, storage, p, 8u, layout_rest, q16_cells);
            const float3 c9 = fast_sh_rest(shN, bounds, storage, p, 9u, layout_rest, q16_cells);
            const float3 c10 = fast_sh_rest(shN, bounds, storage, p, 10u, layout_rest, q16_cells);
            const float3 c11 = fast_sh_rest(shN, bounds, storage, p, 11u, layout_rest, q16_cells);
            const float3 c12 = fast_sh_rest(shN, bounds, storage, p, 12u, layout_rest, q16_cells);
            const float3 c13 = fast_sh_rest(shN, bounds, storage, p, 13u, layout_rest, q16_cells);
            const float3 c14 = fast_sh_rest(shN, bounds, storage, p, 14u, layout_rest, q16_cells);
            gx = gx + (-3.5402615395598609f * xy) * c8 + (2.8906114426405538f * yz) * c9 +
                 (0.45704579946446572f - 2.2852289973223288f * zz) * c12 + (2.8906114426405538f * xz) * c13 +
                 (-1.7701307697799304f * xx + 1.7701307697799304f * yy) * c14;
            gy = gy + (-1.7701307697799304f * xx + 1.7701307697799304f * yy) * c8 + (2.8906114426405538f * xz) * c9 +
                 (0.45704579946446572f - 2.2852289973223288f * zz) * c10 + (-2.8906114426405538f * yz) * c13 +
                 (3.5402615395598609f * xy) * c14;
            gz = gz + (2.8906114426405538f * xy) * c9 + (-4.5704579946446566f * yz) * c10 +
                 (5.597644988851731f * zz - 1.1195289977703462f) * c11 + (-4.5704579946446566f * xz) * c12 +
                 (1.4453057213202769f * xx - 1.4453057213202769f * yy) * c13;
        }
    }
    const float3 gd = float3(dot(gx, grad_color), dot(gy, grad_color), dot(gz, grad_color));
    const float xx = raw.x * raw.x, yy = raw.y * raw.y, zz = raw.z * raw.z;
    const float xy = raw.x * raw.y, xz = raw.x * raw.z, yz = raw.y * raw.z;
    const float norm_sq = fmax(xx + yy + zz, 1e-6f);
    const float inv_norm_cubed = fmin(rsqrt(norm_sq * norm_sq * norm_sq), 1e6f);
    return float3((yy + zz) * gd.x - xy * gd.y - xz * gd.z, -xy * gd.x + (xx + zz) * gd.y - yz * gd.z,
                  -xz * gd.x - yz * gd.y + (xx + yy) * gd.z) *
           inv_norm_cubed;
}

struct FastBackwardShParams {
    device const packed_float3* means;
    device const float* camera;
    device const uchar* shN;
    device const float2* sh_bounds;
    device const uint* n_touched;
    device const float4* color_depth;
    device float* grads;
    FastAdamGroup sh0;
    FastAdamGroup shN_adam;
    float beta1, beta2, eps;
    uint n;
};

kernel void fast_backward_sh(constant FastBackwardShParams& p [[buffer(0)]],
                             const uint group [[threadgroup_position_in_grid]],
                             const uint lane [[thread_index_in_threadgroup]],
                             const uint simd_lane [[thread_index_in_simdgroup]],
                             const uint simd_group [[simdgroup_index_in_threadgroup]],
                             const uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup float4 scratch[32];
    const FastLane t = {lane, simd_lane, simd_group, simd_groups, group};
    // kFastShParts threads per primitive; part 0 also owns sh0 and the mean gradient.
    const uint part = lane / uint(kJointBlock);
    const uint idx = group * uint(kJointBlock) + lane % uint(kJointBlock);
    const bool visible = idx < p.n && p.n_touched[idx] > 0u;
    float sh0_grads[3] = {0.0f, 0.0f, 0.0f};
    float3 grad_color = float3(0.0f);
    float3 direction = float3(0.0f);
    float3 mean_grad = float3(0.0f);
    if (visible) {
        device const float* g = p.grads + idx * kFastGradStride;
        grad_color = float3(g[6], g[7], g[8]);
        // The blend clamps colour at zero. Below it, keep only the image
        // gradients that brighten the splat; the forward never saw the rest.
        const float3 colour = p.color_depth[idx].xyz;
        for (uint c = 0; c < 3u; ++c) {
            if (colour[c] < 0.0f && grad_color[c] >= 0.0f)
                grad_color[c] = 0.0f;
        }
        const float3 mean = float3(p.means[idx]);
        const float3 camera = float3(p.camera[0], p.camera[1], p.camera[2]);
        direction = fast_safe_normalize(mean - camera);
        if (part == 0u) {
            const float3 d0 = 0.28209479177387814f * grad_color;
            sh0_grads[0] = d0.x;
            sh0_grads[1] = d0.y;
            sh0_grads[2] = d0.z;
            mean_grad = fast_sh_mean_grad(p.shN, p.sh_bounds, kFastShStorage, idx, mean, camera, kFastShLayoutRest,
                                          fast_q16_cells(), grad_color);
        }
    }
    // The mean gradient reads the rest coefficients before any part updates them,
    // and every part reads the color gradient before part 0 overwrites it.
    threadgroup_barrier(mem_flags::mem_device);
    fast_adam_step(p.sh0, sh0_grads, idx, 3u, 1.0f, p.beta1, p.beta2, p.eps, scratch, t, part == 0u);
    if (kFastShBases > 1u && p.shN_adam.joint_bits == 8)
        fast_adam_shN(p.shN_adam, idx, part, kFastShLayoutRest, grad_color, direction, visible, p.beta1, p.beta2,
                      p.eps, scratch, t);
    if (visible && part == 0u) {
        device float* g = p.grads + idx * kFastGradStride;
        g[6] = mean_grad.x;
        g[7] = mean_grad.y;
        g[8] = mean_grad.z;
    }
}

struct FastBackwardGeometryParams {
    device const packed_float3* means;
    device const packed_float3* scales;
    device const float4* rotations;
    device const float* opacities;
    device const float4* view;
    device const float* camera;
    device const uint* n_touched;
    device const float* grads;
    device const float4* normal_grads;
    device float* densification;
    device const uchar* far_mask;
    device atomic_float* scale_loss;
    device atomic_float* opacity_loss;
    device const float* sparsity_sigmoid;
    device const float* sparsity_z;
    device const float* sparsity_u;
    FastAdamGroup means_adam;
    FastAdamGroup rotation_adam;
    FastAdamGroup scaling_adam;
    FastAdamGroup opacity_adam;
    float beta1, beta2, eps;
    float scale_reg_weight, flatten_reg_weight, opacity_reg_weight, sparsity_rho, sparsity_grad_loss;
    float median_extent, r_min, r_max;
    float width, height, fx, fy;
    float clip_left, clip_right, clip_top, clip_bottom;
    uint n, far_mask_n, sparsity_n, per_splat_mean_step;
};

static float fast_sigmoid(const float x) { return 1.0f / (1.0f + exp(-x)); }

static float fast_scale_reg_grad(constant FastBackwardGeometryParams& p, const uint element) {
    constant FastAdamGroup& g = p.scaling_adam;
    if (p.scale_reg_weight <= 0.0f || g.elements <= 0)
        return 0.0f;
    return p.scale_reg_weight * exp(reinterpret_cast<device const float*>(g.param)[element]) / float(g.elements);
}

// L = weight * mean(exp(min raw scale)); the argmin is held constant.
static void fast_flatten_reg_grads(constant FastBackwardGeometryParams& p, const uint base, thread float* grads) {
    constant FastAdamGroup& g = p.scaling_adam;
    if (p.flatten_reg_weight <= 0.0f || g.elements <= 0)
        return;
    device const float* s = reinterpret_cast<device const float*>(g.param) + base;
    const uint axis = (s[0] <= s[1] && s[0] <= s[2]) ? 0u : (s[1] <= s[2] ? 1u : 2u);
    grads[axis] += 3.0f * p.flatten_reg_weight * exp(s[axis]) / float(g.elements);
}

static float fast_opacity_extra_grad(constant FastBackwardGeometryParams& p, const uint idx) {
    constant FastAdamGroup& g = p.opacity_adam;
    float grad = 0.0f;
    if (p.opacity_reg_weight > 0.0f && g.elements > 0) {
        const float opacity = fast_sigmoid(reinterpret_cast<device const float*>(g.param)[idx]);
        grad += p.opacity_reg_weight * opacity * (1.0f - opacity) / float(g.elements);
    }
    if (p.sparsity_sigmoid != nullptr && p.sparsity_z != nullptr && p.sparsity_u != nullptr && idx < p.sparsity_n) {
        const float opacity = p.sparsity_sigmoid[idx];
        grad += p.sparsity_rho * (opacity - p.sparsity_z[idx] + p.sparsity_u[idx]) * opacity * (1.0f - opacity) *
                p.sparsity_grad_loss;
    }
    return grad;
}

// Sum over the threadgroup, valid on lane 0.
static float fast_block_sum(const float value, threadgroup float* scratch, const FastLane t) {
    const float partial = simd_sum(value);
    if (t.simd_lane == 0u)
        scratch[t.simd_group] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    if (t.lane == 0u)
        for (uint s = 0; s < t.simd_groups; ++s)
            total += scratch[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

kernel void fast_backward_geometry(constant FastBackwardGeometryParams& p [[buffer(0)]],
                                   const uint group [[threadgroup_position_in_grid]],
                                   const uint lane [[thread_index_in_threadgroup]],
                                   const uint simd_lane [[thread_index_in_simdgroup]],
                                   const uint simd_group [[simdgroup_index_in_threadgroup]],
                                   const uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup float4 scratch[32];
    threadgroup float sum_scratch[32];
    const FastLane t = {lane, simd_lane, simd_group, simd_groups, group};
    const uint idx = group * 256u + lane;
    const bool in_range = idx < p.n;

    // Regularizer loss scalars, from the parameters before this step.
    constant FastAdamGroup& scaling = p.scaling_adam;
    constant FastAdamGroup& opacity_group = p.opacity_adam;
    if (p.scale_loss != nullptr) {
        float local = 0.0f;
        if (in_range && p.scale_reg_weight > 0.0f && scaling.param != nullptr && scaling.elements > 0) {
            device const float* s = reinterpret_cast<device const float*>(scaling.param) + idx * 3u;
            local = p.scale_reg_weight / float(scaling.elements) * (exp(s[0]) + exp(s[1]) + exp(s[2]));
        }
        const float total = fast_block_sum(local, sum_scratch, t);
        if (lane == 0u && total != 0.0f)
            atomic_fetch_add_explicit(p.scale_loss, total, memory_order_relaxed);
    }
    if (p.opacity_loss != nullptr) {
        float local = 0.0f;
        if (in_range && p.opacity_reg_weight > 0.0f && opacity_group.param != nullptr && opacity_group.elements > 0)
            local = p.opacity_reg_weight *
                    fast_sigmoid(reinterpret_cast<device const float*>(opacity_group.param)[idx]) /
                    float(opacity_group.elements);
        const float total = fast_block_sum(local, sum_scratch, t);
        if (lane == 0u && total != 0.0f)
            atomic_fetch_add_explicit(p.opacity_loss, total, memory_order_relaxed);
    }

    float mean_grads[3] = {0.0f, 0.0f, 0.0f};
    float rotation_grads[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float scale_grads[3] = {0.0f, 0.0f, 0.0f};
    float opacity_grads[1] = {0.0f};
    const bool visible = in_range && p.n_touched[idx] > 0u;
    if (in_range && !visible) {
        // Regularizers only; the other groups decay their moments.
        for (uint i = 0; i < 3u; ++i)
            scale_grads[i] = fast_scale_reg_grad(p, idx * 3u + i);
        fast_flatten_reg_grads(p, idx * 3u, scale_grads);
        opacity_grads[0] = fast_opacity_extra_grad(p, idx);
    } else if (visible) {
        device const float* g = p.grads + idx * kFastGradStride;
        const float3 mean3d = float3(p.means[idx]);
        const float4 w1 = p.view[0], w2 = p.view[1], w3 = p.view[2];
        const float depth = w3.x * mean3d.x + w3.y * mean3d.y + w3.z * mean3d.z + w3.w;
        const float inv_depth = 1.0f / fmax(depth, 1e-4f);
        const float x = (w1.x * mean3d.x + w1.y * mean3d.y + w1.z * mean3d.z + w1.w) * inv_depth;
        const float y = (w2.x * mean3d.x + w2.y * mean3d.y + w2.z * mean3d.z + w2.w) * inv_depth;

        const float3 raw_scale = float3(p.scales[idx]);
        const float3 variance = exp(2.0f * fmin(raw_scale, kFastMaxRawScale));
        const float4 q = p.rotations[idx];
        const FastRotation rot = fast_rotation(q);
        const FastCov3 cov3d = fast_cov3d(rot, variance);

        const float tx = clamp(x, p.clip_left, p.clip_right);
        const float ty = clamp(y, p.clip_top, p.clip_bottom);
        const float j11 = p.fx * inv_depth;
        const float j13 = -j11 * tx;
        const float j22 = p.fy * inv_depth;
        const float j23 = -j22 * ty;
        const float3 jw1 = j11 * w1.xyz + j13 * w3.xyz;
        const float3 jw2 = j22 * w2.xyz + j23 * w3.xyz;
        const float3 jwc1 = fast_cov_mul(cov3d, jw1);
        const float3 jwc2 = fast_cov_mul(cov3d, jw2);

        const float kernel_size = kFastMipFilter != 0u ? kFastDilationMip : kFastDilation;
        const float raw_a = dot(jwc1, jw1), raw_b = dot(jwc1, jw2), raw_c = dot(jwc2, jw2);
        const float a = raw_a + kernel_size, b = raw_b, c = raw_c + kernel_size;
        const float aa = a * a, bb = b * b, cc = c * c, ac = a * c, ab = a * b, bc = b * c;
        const float determinant = ac - bb;
        const float det_rcp = 1.0f / fmax(determinant, kFastMinDeterminant);
        const float det_rcp_sq = det_rcp * det_rcp;
        const float3 dconic = float3(g[2], g[3], g[4]);
        const float3 dcov2d = det_rcp_sq * float3(2.0f * bc * dconic.y - cc * dconic.x - bb * dconic.z,
                                                  bc * dconic.x - (ac + bb) * dconic.y + ab * dconic.z,
                                                  2.0f * ab * dconic.y - bb * dconic.x - aa * dconic.z);

        const float original_opacity = fast_sigmoid(p.opacities[idx]);
        float compensation = 1.0f;
        if (kFastMipFilter != 0u) {
            const float det_raw = raw_a * raw_c - raw_b * raw_b;
            compensation = det_raw > kFastMinDeterminant && determinant > kFastMinDeterminant ? sqrt(det_raw * det_rcp)
                                                                                              : 0.0f;
        }
        opacity_grads[0] = g[5] * compensation * original_opacity * (1.0f - original_opacity) +
                           fast_opacity_extra_grad(p, idx);

        const FastCov3 dcov3d = {
            (jw1.x * jw1.x) * dcov2d.x + 2.0f * (jw1.x * jw2.x) * dcov2d.y + (jw2.x * jw2.x) * dcov2d.z,
            (jw1.x * jw1.y) * dcov2d.x + (jw1.x * jw2.y + jw1.y * jw2.x) * dcov2d.y + (jw2.x * jw2.y) * dcov2d.z,
            (jw1.x * jw1.z) * dcov2d.x + (jw1.x * jw2.z + jw1.z * jw2.x) * dcov2d.y + (jw2.x * jw2.z) * dcov2d.z,
            (jw1.y * jw1.y) * dcov2d.x + 2.0f * (jw1.y * jw2.y) * dcov2d.y + (jw2.y * jw2.y) * dcov2d.z,
            (jw1.y * jw1.z) * dcov2d.x + (jw1.y * jw2.z + jw1.z * jw2.y) * dcov2d.y + (jw2.y * jw2.z) * dcov2d.z,
            (jw1.z * jw1.z) * dcov2d.x + 2.0f * (jw1.z * jw2.z) * dcov2d.y + (jw2.z * jw2.z) * dcov2d.z,
        };

        const float3 djw1 = 2.0f * (jwc1 * dcov2d.x + jwc2 * dcov2d.y);
        const float3 djw2 = 2.0f * (jwc1 * dcov2d.y + jwc2 * dcov2d.z);
        const float dj11 = dot(w1.xyz, djw1);
        const float dj22 = dot(w2.xyz, djw2);
        const float dj13 = dot(w3.xyz, djw1);
        const float dj23 = dot(w3.xyz, djw2);

        // Camera-space mean gradient through mean2d, depth and J (with tx/ty clipping).
        const float2 dmean2d = float2(g[0], g[1]);
        float3 dcam = float3(j11 * dmean2d.x, j22 * dmean2d.y, -j11 * x * dmean2d.x - j22 * y * dmean2d.y + g[9]);
        const bool valid_x = x >= p.clip_left && x <= p.clip_right;
        const bool valid_y = y >= p.clip_top && y <= p.clip_bottom;
        if (valid_x)
            dcam.x -= j11 * dj13 * inv_depth;
        if (valid_y)
            dcam.y -= j22 * dj23 * inv_depth;
        const float factor_x = valid_x ? 2.0f : 1.0f;
        const float factor_y = valid_y ? 2.0f : 1.0f;
        dcam.z += (j11 * (factor_x * tx * dj13 - dj11) + j22 * (factor_y * ty * dj23 - dj22)) * inv_depth;
        const float3 dmean = w1.xyz * dcam.x + w2.xyz * dcam.y + w3.xyz * dcam.z + float3(g[6], g[7], g[8]);
        mean_grads[0] = fast_clamp_grad(dmean.x);
        mean_grads[1] = fast_clamp_grad(dmean.y);
        mean_grads[2] = fast_clamp_grad(dmean.z);

        // Raw scale gradient; zero where the scale is clamped.
        const float3 col0 = float3(rot.r0.x, rot.r1.x, rot.r2.x);
        const float3 col1 = float3(rot.r0.y, rot.r1.y, rot.r2.y);
        const float3 col2 = float3(rot.r0.z, rot.r1.z, rot.r2.z);
        const float3 dvariance = float3(
            col0.x * col0.x * dcov3d.m11 + col0.y * col0.y * dcov3d.m22 + col0.z * col0.z * dcov3d.m33 +
                2.0f * (col0.x * col0.y * dcov3d.m12 + col0.x * col0.z * dcov3d.m13 + col0.y * col0.z * dcov3d.m23),
            col1.x * col1.x * dcov3d.m11 + col1.y * col1.y * dcov3d.m22 + col1.z * col1.z * dcov3d.m33 +
                2.0f * (col1.x * col1.y * dcov3d.m12 + col1.x * col1.z * dcov3d.m13 + col1.y * col1.z * dcov3d.m23),
            col2.x * col2.x * dcov3d.m11 + col2.y * col2.y * dcov3d.m22 + col2.z * col2.z * dcov3d.m33 +
                2.0f * (col2.x * col2.y * dcov3d.m12 + col2.x * col2.z * dcov3d.m13 + col2.y * col2.z * dcov3d.m23));
        for (uint i = 0; i < 3u; ++i) {
            const float draw = raw_scale[i] < kFastMaxRawScale ? 2.0f * variance[i] * dvariance[i] : 0.0f;
            scale_grads[i] = fast_clamp_grad(draw) + fast_scale_reg_grad(p, idx * 3u + i);
        }
        fast_flatten_reg_grads(p, idx * 3u, scale_grads);

        // dL/dR, rows; rs = R diag(variance).
        const float3 rs0 = rot.r0 * variance, rs1 = rot.r1 * variance, rs2 = rot.r2 * variance;
        float3 dr0 = 2.0f * (rs0 * dcov3d.m11 + rs1 * dcov3d.m12 + rs2 * dcov3d.m13);
        float3 dr1 = 2.0f * (rs0 * dcov3d.m12 + rs1 * dcov3d.m22 + rs2 * dcov3d.m23);
        float3 dr2 = 2.0f * (rs0 * dcov3d.m13 + rs1 * dcov3d.m23 + rs2 * dcov3d.m33);
        if (p.normal_grads != nullptr) {
            const float3 gc = p.normal_grads[idx].xyz;
            const float3 gw = w1.xyz * gc.x + w2.xyz * gc.y + w3.xyz * gc.z;
            const float3 view_dir = mean3d - float3(p.camera[0], p.camera[1], p.camera[2]);
            const uint axis = (variance.x <= variance.y && variance.x <= variance.z) ? 0u
                              : (variance.y <= variance.z)                           ? 1u
                                                                                     : 2u;
            const float3 column = axis == 0u ? col0 : (axis == 1u ? col1 : col2);
            const float sign = dot(column, view_dir) > 0.0f ? -1.0f : 1.0f;
            dr0[axis] += sign * gw.x;
            dr1[axis] += sign * gw.y;
            dr2[axis] += sign * gw.z;
        }
        const float dqxx = -dr1.y - dr2.z;
        const float dqyy = -dr0.x - dr2.z;
        const float dqzz = -dr0.x - dr1.y;
        const float dqxy = dr0.y + dr1.x;
        const float dqxz = dr0.z + dr2.x;
        const float dqyz = dr1.z + dr2.y;
        const float dqrx = dr2.y - dr1.z;
        const float dqry = dr0.z - dr2.x;
        const float dqrz = dr1.x - dr0.y;
        const float norm_helper = rot.qxx * dqxx + rot.qyy * dqyy + rot.qzz * dqzz + rot.qxy * dqxy + rot.qxz * dqxz +
                                  rot.qyz * dqyz + rot.qrx * dqrx + rot.qry * dqry + rot.qrz * dqrz;
        const float qr = q.x, qx = q.y, qy = q.z, qz = q.w;
        const float4 drot = float4(qx * dqrx + qy * dqry + qz * dqrz - qr * norm_helper,
                                   2.0f * qx * dqxx + qy * dqxy + qz * dqxz + qr * dqrx - qx * norm_helper,
                                   2.0f * qy * dqyy + qx * dqxy + qz * dqyz + qr * dqry - qy * norm_helper,
                                   2.0f * qz * dqzz + qx * dqxz + qy * dqyz + qr * dqrz - qz * norm_helper) *
                            rot.inv_q_norm_sq;
        for (uint i = 0; i < 4u; ++i)
            rotation_grads[i] = fast_clamp_grad(drot[i]);

        if (p.densification != nullptr) {
            p.densification[idx] += 1.0f;
            p.densification[p.n + idx] += length(dmean2d * float2(0.5f * p.width, 0.5f * p.height));
        }
    }

    // The per-splat factor scales the applied mean step, not the gradient.
    float mean_step_scale = 1.0f;
    if (in_range && p.per_splat_mean_step != 0u && p.far_mask != nullptr && idx < p.far_mask_n &&
        p.far_mask[idx] != 0u && scaling.param != nullptr && idx * 3u + 2u < uint(scaling.elements)) {
        device const float* s = reinterpret_cast<device const float*>(scaling.param) + idx * 3u;
        mean_step_scale = per_splat_mean_step_ratio(s[0], s[1], s[2], p.median_extent, p.r_min, p.r_max);
    }
    fast_adam_step(p.means_adam, mean_grads, idx, 3u, mean_step_scale, p.beta1, p.beta2, p.eps, scratch, t);
    fast_adam_step(p.rotation_adam, rotation_grads, idx, 4u, 1.0f, p.beta1, p.beta2, p.eps, scratch, t);
    fast_adam_step(p.scaling_adam, scale_grads, idx, 3u, 1.0f, p.beta1, p.beta2, p.eps, scratch, t);
    fast_adam_step(p.opacity_adam, opacity_grads, idx, 1u, 1.0f, p.beta1, p.beta2, p.eps, scratch, t);
}
