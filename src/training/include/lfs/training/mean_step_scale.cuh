/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Per-splat mean-step ratio. Applied to the Adam step, never the raw gradient:
// after Adam's second-moment normalize, scaling the gradient is a no-op.
// r_i = clamp(geomean(exp(scaling_raw)) / median_splat_extent, r_min, r_max).
// Callers apply this only to far-field rows (camera-hull mask). A null mask,
// invalid hull, or inside-hull row must keep r_i = 1 (the global-mode step).
namespace lfs::training {

    inline constexpr float kPerSplatMeanStepRatioMin = 1.0f;
    inline constexpr float kPerSplatMeanStepRatioMax = 300.0f;

#ifdef __CUDACC__
    __device__ __forceinline__ float per_splat_mean_step_ratio(
        const float s0,
        const float s1,
        const float s2,
        const float median_splat_extent,
        const float r_min,
        const float r_max) {
        if (!(median_splat_extent > 0.0f)) {
            return 1.0f;
        }
        const float s_i = expf((s0 + s1 + s2) * (1.0f / 3.0f));
        return fminf(fmaxf(s_i / median_splat_extent, r_min), r_max);
    }
#endif

} // namespace lfs::training
