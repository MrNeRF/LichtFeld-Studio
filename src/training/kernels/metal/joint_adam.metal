// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Shared by every family that reads or writes Adam state: ports of
// joint_adam_codec.cuh, screen_share.cuh and the block
// reduction in warp_reduce.cuh.
//
// This file, sh_storage.metal and the Adam, Sh and Morton kernels compile in
// safe math mode: Metal's fast math reassociates, e.g. (1 - beta2) * g * g into
// g * g - beta2 * g * g, a 1e-4 relative error the CUDA build (which contracts
// but never reassociates) does not make. Transcendentals stay fast:: as CUDA's
// __expf and __logf. Each such file restores fast math at its end.
#pragma METAL fp math_mode(safe)

// Joint (u, log_s) moment codec. Cells hold two 8- or 16-bit codes; every
// 256-primitive block keeps float4 bounds (u_min, u_max, s_min, s_max).
constant constexpr int kJointBlock = 256;
constant constexpr float kJointEps = 1e-15f;

// MSL has no log1p or expm1; Kahan's compensated forms keep small arguments
// exact, which the codec needs for its 0 <-> 0 fixed point.
static float log1p_precise(const float x) {
    const float u = 1.0f + x;
    return u == 1.0f ? x : precise::log(u) * (x / (u - 1.0f));
}

static float expm1_precise(const float x) {
    const float u = precise::exp(x);
    if (u == 1.0f)
        return x;
    const float um1 = u - 1.0f;
    return um1 == -1.0f ? -1.0f : um1 * (x / precise::log(u));
}

template <int BITS>
struct JointCodec {
    static float q_max() { return float((1 << BITS) - 1); }

    // The bulk of the range uses CUDA's __logf and __expf, which scale the
    // base-2 hardware functions.
    static float forward_sqrt_g2(const float sqrt_g2) {
        const float x = fmax(sqrt_g2, 0.0f) * (1.0f / kJointEps);
        return x > 0.125f ? fast::log2(1.0f + x) * M_LN2_F : log1p_precise(x);
    }

    static float inverse_sqrt_g2(const float log_s) {
        const float e1 = log_s > 0.118f ? fast::exp2(log_s * M_LOG2E_F) - 1.0f : expm1_precise(log_s);
        return kJointEps * e1;
    }

    static float2 g1g2_to_us(const float g1, const float g2) {
        const float sqrt_g2 = sqrt(fmax(g2, 0.0f));
        return float2(g1 / (sqrt_g2 + kJointEps), forward_sqrt_g2(sqrt_g2));
    }

    // Codes (u_q, s_q) to (u, log_s). The CUDA codec scales by the rounded
    // reciprocal kInvQMax, and nvcc contracts the sum into an fma.
    static float2 decode_codes(const float2 q, const float4 mm) {
        const float inv_q_max = 1.0f / q_max();
        return float2(fma(mm.y - mm.x, q.x * inv_q_max, mm.x), fma(mm.w - mm.z, q.y * inv_q_max, mm.z));
    }

    static float2 decode_us(device const uchar* packed, const long cell, const float4 mm) {
        if (BITS == 16) {
            device const ushort* codes = reinterpret_cast<device const ushort*>(packed);
            return decode_codes(float2(codes[cell * 2], codes[cell * 2 + 1]), mm);
        }
        return decode_codes(float2(packed[cell * 2], packed[cell * 2 + 1]), mm);
    }

    static float2 us_to_g1g2(const float2 us) {
        const float sqrt_g2 = inverse_sqrt_g2(us.y);
        const float g2 = sqrt_g2 * sqrt_g2;
        // Zero variance must not turn a quantized u residue into momentum.
        const float g1 = g2 == 0.0f ? 0.0f : us.x * (sqrt_g2 + kJointEps);
        return float2(g1, g2);
    }

    static float2 decode_g1g2(device const uchar* packed, const long cell, const float4 mm) {
        return us_to_g1g2(decode_us(packed, cell, mm));
    }

    static float2 encode_codes(const float u, const float log_s, const float u_min, const float s_min,
                               const float inv_u_range, const float inv_s_range) {
        return float2(fmin(fmax(round(q_max() * (u - u_min) * inv_u_range), 0.0f), q_max()),
                      fmin(fmax(round(q_max() * (log_s - s_min) * inv_s_range), 0.0f), q_max()));
    }

    static void encode_us(device uchar* packed, const long cell, const float u, const float log_s,
                          const float u_min, const float s_min, const float inv_u_range, const float inv_s_range) {
        const float2 q = encode_codes(u, log_s, u_min, s_min, inv_u_range, inv_s_range);
        if (BITS == 16) {
            device ushort* codes = reinterpret_cast<device ushort*>(packed);
            codes[cell * 2] = ushort(q.x);
            codes[cell * 2 + 1] = ushort(q.y);
        } else {
            packed[cell * 2] = uchar(q.x);
            packed[cell * 2 + 1] = uchar(q.y);
        }
    }

    static void encode_us(device uchar* packed, const long cell, const float u, const float log_s, const float4 mm) {
        encode_us(packed, cell, u, log_s, mm.x, mm.z, 1.0f / fmax(mm.y - mm.x, kJointEps),
                  1.0f / fmax(mm.w - mm.z, kJointEps));
    }
};

// Bounds widened to contain (u, log_s) = (0, 0), so zero moments encode exactly.
static float4 joint_bounds_with_zero(const float4 mm) {
    return float4(fmin(mm.x, 0.0f), fmax(mm.y, 0.0f), fmin(mm.z, 0.0f), fmax(mm.w, 0.0f));
}

static float gaussian_screen_share(const float3 mean, const float3 camera, const float3 log_scale,
                                   const float opacity_raw) {
    const float max_log = fmax(log_scale.x, fmax(log_scale.y, log_scale.z));
    const float opacity = 1.0f / (1.0f + fast::exp(-opacity_raw));
    const float extend = sqrt(2.0f * fast::log(fmax(255.0f * opacity, 1.0f)));
    const float r = fast::exp(max_log) * extend;
    const float d = length(mean - camera);
    const float denom = fmax(d, r) + sqrt(fmax(d * d - r * r, 0.0f));
    if (!(denom > 0.0f) || !(r > 0.0f))
        return 0.0f;
    return fmin(fmax(r / denom, 0.0f), 1.0f);
}

static bool screen_share_cap_active(const float max_screen_share) {
    return max_screen_share > 0.0f && max_screen_share < 1.0f;
}

static float oversize_split_score(const float error_score, const float max_share, const float limit) {
    if (!(limit > 0.0f) || !(limit < 1.0f) || !(max_share > limit) || !(error_score > 0.0f))
        return 0.0f;
    return sqrt(error_score) * (max_share / limit);
}

// First-moment-only term (see screen_share.cuh): feeding it into v overflows v.
static float screen_share_hinge_extra_grad(const float share, const float limit, const float penalty,
                                           const float old_v, const float bias_correction2_sqrt_rcp,
                                           const float eps) {
    if (!(limit > 0.0f) || !(limit < 1.0f) || !(share > limit) || !(penalty > 0.0f))
        return 0.0f;
    const float hinge = penalty * fast::log2(share / limit);
    return hinge * (sqrt(old_v) * bias_correction2_sqrt_rcp + eps);
}

// Metal has no float atomic max; compare-and-swap on the bits.
static void atomic_max_float(device atomic_uint* address, const float value) {
    uint old = atomic_load_explicit(address, memory_order_relaxed);
    while (as_type<float>(old) < value) {
        if (atomic_compare_exchange_weak_explicit(address, &old, as_type<uint>(value), memory_order_relaxed,
                                                  memory_order_relaxed))
            return;
    }
}

// Minimum of float4 lanes over a threadgroup of up to 1024 threads. scratch
// holds one float4 per SIMD group; every thread receives the result.
static float4 threadgroup_min4(float4 value, threadgroup float4* scratch, const uint lane, const uint simd_lane,
                               const uint simd_group, const uint simd_groups) {
    value = float4(simd_min(value.x), simd_min(value.y), simd_min(value.z), simd_min(value.w));
    if (simd_lane == 0)
        scratch[simd_group] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float4 result = scratch[0];
        for (uint group = 1; group < simd_groups; ++group)
            result = min(result, scratch[group]);
        scratch[0] = result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float4 result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

// Joint block bounds from each lane's (u_min, u_max, s_min, s_max): zero when
// no lane held a cell (lanes without cells pass +-1e30).
static float4 threadgroup_joint_bounds(const float4 local, threadgroup float4* scratch, const uint lane,
                                       const uint simd_lane, const uint simd_group, const uint simd_groups) {
    const float4 red = threadgroup_min4(float4(local.x, -local.y, local.z, -local.w), scratch, lane, simd_lane,
                                        simd_group, simd_groups);
    return red.x > -red.y ? float4(0.0f) : float4(red.x, -red.y, red.z, -red.w);
}

#pragma METAL fp math_mode(fast)
