// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// AdamOps: ports of adam_kernels.cuh (contiguous batch step, encode-zero) and
// of apply_shN_grads_packed_joint over a swizzled gradient (adam_shN_joint.cu).
// Every threadgroup is one 256-primitive bounds block.
// Safe math mode, as joint_adam.metal explains.
#pragma METAL fp math_mode(safe)

constant constexpr int kAdamMaxSteps = 6;
constant constexpr int kAdamMaxAttributes = 16;
constant constexpr float kAdamInf = 1e30f;
constant constexpr uint kAdamSimdGroups = kThreadgroupWidth / 32;

struct AdamRowMasks {
    device const bool* frozen;
    device const bool* crop;
    int frozen_count;
    int crop_count;
    float frozen_lr_scale;
    float cropbox_lr_scale;
};

// Applies the frozen and crop-box rate factors to `rate`; false when a zero
// factor skips the Adam update of this row.
static bool adam_row_rate(constant AdamRowMasks& m, const int prim, thread float& rate) {
    bool apply = true;
    if (m.frozen != nullptr && prim < m.frozen_count && m.frozen[prim]) {
        if (m.frozen_lr_scale == 0.0f)
            apply = false;
        else
            rate *= m.frozen_lr_scale;
    }
    if (m.crop != nullptr && prim < m.crop_count && m.crop[prim]) {
        if (m.cropbox_lr_scale == 0.0f)
            apply = false;
        else
            rate *= m.cropbox_lr_scale;
    }
    return apply;
}

struct AdamStep {
    device float* parameter;
    device uchar* packed;
    device float4* bounds;
    device const float* gradient;
    int primitives;
    int attributes;
    float lr;
    float bc1_rcp;
    float bc2_sqrt_rcp;
    uint apply_mean_step;
    uint apply_screen_share;
};

struct AdamBatchParams {
    AdamStep steps[kAdamMaxSteps];
    AdamRowMasks rows;
    device const float* raw_scales;
    device const bool* far_mask;
    device const float* screen_share;
    int raw_scales_count;
    int far_count;
    int screen_share_count;
    float median_extent;
    float r_min;
    float r_max;
    float screen_share_limit;
    float screen_share_penalty;
    float beta1;
    float beta2;
    float eps;
};

// Grid (blocks of the largest step, steps). Rows whose update is skipped still
// re-encode under the block's new bounds.
kernel void adam_step_batch(constant AdamBatchParams& p [[buffer(0)]], uint2 group [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_threadgroup]],
                            uint simd_lane [[thread_index_in_simdgroup]],
                            uint simd_group [[simdgroup_index_in_threadgroup]],
                            uint simd_groups [[simdgroups_per_threadgroup]]) {
    using C = JointCodec<16>;
    threadgroup float4 scratch[kAdamSimdGroups];
    constant AdamStep& step = p.steps[group.y];
    const int prim = int(group.x) * kJointBlock + int(lane);
    const bool in_range = prim < step.primitives && step.attributes > 0;

    float row_lr = step.lr;
    const bool apply_step = in_range && adam_row_rate(p.rows, prim, row_lr);
    if (step.apply_mean_step != 0u && p.raw_scales != nullptr && p.far_mask != nullptr && prim < p.far_count &&
        p.far_mask[prim]) {
        const int sb = prim * 3;
        if (sb + 2 < p.raw_scales_count)
            row_lr *= per_splat_mean_step_ratio(p.raw_scales[sb], p.raw_scales[sb + 1], p.raw_scales[sb + 2],
                                                p.median_extent, p.r_min, p.r_max);
    }

    const float4 old_mm = step.bounds[group.x];
    float us_u[kAdamMaxAttributes];
    float us_s[kAdamMaxAttributes];
    float4 local = float4(kAdamInf, -kAdamInf, kAdamInf, -kAdamInf);
    const int row = in_range ? min(step.attributes, kAdamMaxAttributes) : 0;
    const float step_size = row_lr * step.bc1_rcp;
    for (int i = 0; i < row; ++i) {
        const long cell = long(prim) * step.attributes + i;
        const float2 mv = C::decode_g1g2(step.packed, cell, old_mm);
        float m = mv.x;
        float v = mv.y;
        if (apply_step) {
            const float grad = step.gradient[cell];
            float hinge = 0.0f;
            if (step.apply_screen_share != 0u && p.screen_share != nullptr && prim < p.screen_share_count)
                hinge = screen_share_hinge_extra_grad(p.screen_share[prim], p.screen_share_limit,
                                                      p.screen_share_penalty, mv.y, step.bc2_sqrt_rcp, p.eps);
            m = p.beta1 * mv.x + (1.0f - p.beta1) * (grad + hinge);
            v = p.beta2 * mv.y + (1.0f - p.beta2) * grad * grad;
            const float denom = sqrt(v) * step.bc2_sqrt_rcp + p.eps;
            step.parameter[cell] -= step_size * m / denom;
        }
        const float2 us = C::g1g2_to_us(m, v);
        us_u[i] = us.x;
        us_s[i] = us.y;
        local = float4(fmin(local.x, us.x), fmax(local.y, us.x), fmin(local.z, us.y), fmax(local.w, us.y));
    }

    const float4 new_mm = threadgroup_joint_bounds(local, scratch, lane, simd_lane, simd_group, simd_groups);
    if (lane == 0)
        step.bounds[group.x] = new_mm;
    const float inv_u = 1.0f / fmax(new_mm.y - new_mm.x, kJointEps);
    const float inv_s = 1.0f / fmax(new_mm.w - new_mm.z, kJointEps);
    for (int i = 0; i < row; ++i)
        C::encode_us(step.packed, long(prim) * step.attributes + i, us_u[i], us_s[i], new_mm.x, new_mm.z, inv_u,
                     inv_s);
}

// parameter holds float4 slots, half slots (value_mode 1) or Q16 codes with
// per-block value bounds (value_mode 2).
struct AdamShParams {
    device void* parameter;
    device uchar* packed;
    device float4* bounds;
    device float2* value_bounds;
    device const float4* gradient;
    AdamRowMasks rows;
    int primitives;
    uint layout_slots;
    uint active_slots;
    uint value_mode;
    uint value_cells;
    float step_size;
    float beta1;
    float beta2;
    float eps;
    float bc2_sqrt_rcp;
};

constant constexpr uint kShValueHalf = 1u;
constant constexpr uint kShValueQ16 = 2u;

static float4 adam_sh_load_slot(constant AdamShParams& p, const uint prim, const uint k, const uint slot,
                                const float2 vmm) {
    if (p.value_mode == kShValueQ16) {
        device const ushort* codes = static_cast<device const ushort*>(p.parameter);
        float4 value = float4(0.0f);
        for (uint c = 0; c < 4u; ++c) {
            if (k * 4u + c < p.value_cells)
                value[c] = sh_q16_decode(codes[sh_q16_index(prim, k * 4u + c, p.value_cells)], vmm.x, vmm.y);
        }
        return value;
    }
    if (p.value_mode == kShValueHalf)
        return float4(static_cast<device const half4*>(p.parameter)[slot]);
    return static_cast<device const float4*>(p.parameter)[slot];
}

// One cell's Adam step from its stored moments; returns the new (u, log_s).
static float2 adam_sh_moment(constant AdamShParams& p, const float grad, const long cell, const float4 old_mm,
                             const bool apply_step, const bool update_param, const float row_step,
                             thread float& value) {
    using C = JointCodec<8>;
    const float2 mv = C::decode_g1g2(p.packed, cell, old_mm);
    float m = mv.x;
    float v = mv.y;
    if (apply_step) {
        m = p.beta1 * mv.x + (1.0f - p.beta1) * grad;
        v = p.beta2 * mv.y + (1.0f - p.beta2) * grad * grad;
        if (update_param)
            value -= row_step * m / (sqrt(v) * p.bc2_sqrt_rcp + p.eps);
    }
    return C::g1g2_to_us(m, v);
}

// Walks every layout slot, so inactive bands re-encode under the new bounds.
// Pass 1 steps and reduces the bounds; pass 2 repeats the step from the
// still-original moments and values and encodes.
kernel void adam_step_sh(constant AdamShParams& p [[buffer(0)]], uint group [[threadgroup_position_in_grid]],
                         uint lane [[thread_position_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
                         uint simd_group [[simdgroup_index_in_threadgroup]],
                         uint simd_groups [[simdgroups_per_threadgroup]]) {
    using C = JointCodec<8>;
    threadgroup float4 scratch[kAdamSimdGroups];
    const uint prim = group * kJointBlock + lane;
    const bool touch = prim < uint(p.primitives);
    const bool q16 = p.value_mode == kShValueQ16;
    const uint slots = min(p.layout_slots, kShMaxSlots);

    float row_step = p.step_size;
    const bool apply_step = adam_row_rate(p.rows, int(prim), row_step);
    const float4 old_mm = touch ? p.bounds[group] : float4(0.0f);
    const float2 old_vmm = q16 ? p.value_bounds[group] : float2(0.0f);

    float4 local = float4(kAdamInf, -kAdamInf, kAdamInf, -kAdamInf);
    float v_lo = kAdamInf;
    float v_hi = -kAdamInf;
    if (touch) {
        for (uint k = 0; k < slots; ++k) {
            const uint slot = sh_slot_index(prim, k, p.layout_slots);
            const bool active = k < p.active_slots;
            const float4 grad = active ? p.gradient[slot] : float4(0.0f);
            float4 value = adam_sh_load_slot(p, prim, k, slot, old_vmm);
            for (uint c = 0; c < 4u; ++c) {
                float cell_value = value[c];
                const float2 us = adam_sh_moment(p, grad[c], long(slot) * 4 + c, old_mm, apply_step, active,
                                                 row_step, cell_value);
                value[c] = cell_value;
                if (q16 && k * 4u + c < p.value_cells) {
                    v_lo = fmin(v_lo, cell_value);
                    v_hi = fmax(v_hi, cell_value);
                }
                local = float4(fmin(local.x, us.x), fmax(local.y, us.x), fmin(local.z, us.y), fmax(local.w, us.y));
            }
            if (apply_step && active) {
                if (p.value_mode == kShValueHalf)
                    static_cast<device half4*>(p.parameter)[slot] = half4(value);
                else if (!q16)
                    static_cast<device float4*>(p.parameter)[slot] = value;
            }
        }
    }

    const float4 new_mm = threadgroup_joint_bounds(local, scratch, lane, simd_lane, simd_group, simd_groups);
    const float2 new_vmm =
        q16 ? threadgroup_q16_bounds(v_lo, v_hi, scratch, lane, simd_lane, simd_group, simd_groups) : float2(0.0f);
    if (lane == 0) {
        p.bounds[group] = new_mm;
        if (q16)
            p.value_bounds[group] = new_vmm;
    }
    if (!touch)
        return;
    const float inv_u = 1.0f / fmax(new_mm.y - new_mm.x, kJointEps);
    const float inv_s = 1.0f / fmax(new_mm.w - new_mm.z, kJointEps);
    device ushort* codes = static_cast<device ushort*>(p.parameter);
    for (uint k = 0; k < slots; ++k) {
        const uint slot = sh_slot_index(prim, k, p.layout_slots);
        const bool active = k < p.active_slots;
        const float4 grad = active ? p.gradient[slot] : float4(0.0f);
        const float4 value = q16 ? adam_sh_load_slot(p, prim, k, slot, old_vmm) : float4(0.0f);
        for (uint c = 0; c < 4u; ++c) {
            float cell_value = value[c];
            const long cell = long(slot) * 4 + c;
            const float2 us = adam_sh_moment(p, grad[c], cell, old_mm, apply_step, active, row_step, cell_value);
            C::encode_us(p.packed, cell, us.x, us.y, new_mm.x, new_mm.z, inv_u, inv_s);
            if (q16 && k * 4u + c < p.value_cells)
                codes[sh_q16_index(prim, k * 4u + c, p.value_cells)] = sh_q16_encode(cell_value, new_vmm.x, new_vmm.y);
        }
    }
}

struct AdamZeroMarkParams {
    device const long* indices;
    device uchar* flags;
    device uint* touched;
    int count;
    int primitives;
};

kernel void adam_encode_zero_mark(constant AdamZeroMarkParams& p [[buffer(0)]],
                                  uint index [[thread_position_in_grid]]) {
    if (index >= uint(p.count))
        return;
    const long prim = p.indices[index];
    if (prim < 0 || prim >= long(p.primitives))
        return;
    p.flags[prim] = 1;
    p.touched[prim / kJointBlock] = 1u;
}

// `width` is attributes per row, or float4 slots per primitive when swizzled.
struct AdamZeroParams {
    device uchar* packed;
    device float4* bounds;
    device const uchar* flags;
    device const uint* touched;
    int primitives;
    uint width;
    uint swizzled;
    int bits;
};

template <int BITS>
static void adam_encode_zero_row(constant AdamZeroParams& p, const uint prim, const float4 old_mm, const float4 new_mm) {
    using C = JointCodec<BITS>;
    const uint cells = p.swizzled != 0u ? p.width * 4u : p.width;
    const auto cell_of = [&](const uint i) {
        return p.swizzled != 0u ? long(sh_slot_index(prim, i / 4u, p.width)) * 4 + i % 4u : long(prim) * p.width + i;
    };
    if (any(old_mm != new_mm)) {
        for (uint i = 0; i < cells; ++i) {
            const float2 us = C::decode_us(p.packed, cell_of(i), old_mm);
            C::encode_us(p.packed, cell_of(i), us.x, us.y, new_mm);
        }
    }
    if (p.flags[prim] != 0) {
        for (uint i = 0; i < cells; ++i)
            C::encode_us(p.packed, cell_of(i), 0.0f, 0.0f, new_mm);
    }
}

// Widens a touched block's bounds to contain (0, 0), re-encodes it and writes
// zero moments for the marked primitives.
kernel void adam_encode_zero(constant AdamZeroParams& p [[buffer(0)]], uint group [[threadgroup_position_in_grid]],
                             uint lane [[thread_position_in_threadgroup]]) {
    if (p.touched[group] == 0u)
        return;
    const uint prim = group * kJointBlock + lane;
    const float4 old_mm = p.bounds[group];
    const float4 new_mm = joint_bounds_with_zero(old_mm);
    threadgroup_barrier(mem_flags::mem_none);
    if (prim < uint(p.primitives)) {
        if (p.bits == 16)
            adam_encode_zero_row<16>(p, prim, old_mm, new_mm);
        else
            adam_encode_zero_row<8>(p, prim, old_mm, new_mm);
    }
    if (lane == 0)
        p.bounds[group] = new_mm;
}

#pragma METAL fp math_mode(fast)
