/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

// Per-splat Adam bias correction. Each splat keeps the number of optimizer steps since its
// moments last restarted (densify reset, growth append, compaction). A row on its t-th step
// since that restart is corrected with 1 - beta^t; a parameter group's own step count stays
// the bound, so rows that never restarted use exactly the group's global correction.
namespace lfs::training {

    // Counters saturate here; a saturated row (or one whose history predates the counters)
    // uses the global correction, which is 1 in float long before this many steps.
    inline constexpr std::int32_t kRowStepSaturation = 65535;

#ifdef __CUDACC__
    // Rewrites step_size and bias_correction2_sqrt_rcp for a row whose own step t = steps + 1 is
    // below the group's step `step`. step_size carries the group's 1 / (1 - beta1^step), so it
    // is rescaled by (1 - beta1^step) / (1 - beta1^t). Returns false (inputs untouched) when the
    // global correction applies.
    __device__ __forceinline__ bool apply_row_bias_correction(
        const std::int32_t steps,
        const int step,
        const float bias_correction1_rcp,
        const float beta1,
        const float beta2,
        float& step_size,
        float& bias_correction2_sqrt_rcp) {
        if (steps < 0 || steps >= kRowStepSaturation || steps + 1 >= step)
            return false;
        const float t = static_cast<float>(steps + 1);
        const float bias_correction1 = 1.0f - powf(beta1, t);
        const float bias_correction2 = 1.0f - powf(beta2, t);
        step_size *= 1.0f / (bias_correction1 * bias_correction1_rcp);
        bias_correction2_sqrt_rcp = 1.0f / sqrtf(bias_correction2);
        return true;
    }
#endif

} // namespace lfs::training
