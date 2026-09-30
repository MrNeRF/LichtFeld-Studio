/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::training {

    struct EffectiveRankPenalty {
        float value = 0.0f;
        float gradient[3] = {0.0f, 0.0f, 0.0f};
    };

    __device__ __forceinline__ EffectiveRankPenalty effective_rank_penalty(
        const float x, const float y, const float z) {
        constexpr float kMinShare = 1.0e-30f;
        constexpr float kOffset = 0.99999f;
        const float raw[3] = {x, y, z};
        const float peak = fmaxf(x, fmaxf(y, z));
        const float q[3] = {expf(2.0f * (x - peak)), expf(2.0f * (y - peak)), expf(2.0f * (z - peak))};
        const float mass = q[0] + q[1] + q[2];
        const float p[3] = {q[0] / mass, q[1] / mass, q[2] / mass};
        int hi = p[1] > p[0] ? 1 : 0;
        if (p[2] > p[hi])
            hi = 2;
        int lo = p[1] < p[0] ? 1 : 0;
        if (p[2] < p[lo])
            lo = 2;
        const float high = p[hi];
        const float low_raw = p[lo];
        const float low = fmaxf(low_raw, kMinShare);
        const float mid_raw = 1.0f - high - low;
        const float mid = fmaxf(mid_raw, kMinShare);
        const float entropy = -(high * logf(high) + mid * logf(mid) + low * logf(low));
        const float rank = expf(entropy);
        const float gap = rank - kOffset;
        EffectiveRankPenalty out;
        if (!(gap > 0.0f))
            return out;
        out.value = fmaxf(-logf(gap), 0.0f);
        if (!(out.value > 0.0f))
            return out;
        const bool mid_active = mid_raw > kMinShare;
        const bool low_active = low_raw > kMinShare;
        const float dH_high = mid_active ? logf(mid) - logf(high) : -(logf(high) + 1.0f);
        const float dH_low = !low_active ? 0.0f : (mid_active ? logf(mid) - logf(low) : -(logf(low) + 1.0f));
        const float dpen_dH = -rank / gap;
        for (int axis = 0; axis < 3; ++axis) {
            const float dp_high = 2.0f * p[hi] * ((axis == hi ? 1.0f : 0.0f) - p[axis]);
            const float dp_low = 2.0f * p[lo] * ((axis == lo ? 1.0f : 0.0f) - p[axis]);
            out.gradient[axis] = raw[axis] < -1.0e4f ? 0.0f : dpen_dH * (dH_high * dp_high + dH_low * dp_low);
        }
        return out;
    }

} // namespace lfs::training
