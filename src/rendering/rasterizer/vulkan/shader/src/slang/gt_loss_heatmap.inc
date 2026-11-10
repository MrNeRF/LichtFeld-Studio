/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Shared by the CPU capture path and the GLSL split-view compositor.
LFS_LOSS_INLINE LFS_LOSS_VEC3 lfsGtLossHeatmapColor(LFS_LOSS_VEC3 ground_truth, LFS_LOSS_VEC3 rendered) {
    const LFS_LOSS_VEC3 difference = abs(ground_truth - rendered);
    const float mean_absolute_error = (difference.r + difference.g + difference.b) / 3.0f;
    const float intensity = LFS_LOSS_ISFINITE(mean_absolute_error)
                                ? clamp(1.0f - exp(-8.0f * max(mean_absolute_error, 0.0f)), 0.0f, 1.0f)
                                : 1.0f;

    const LFS_LOSS_VEC3 black = LFS_LOSS_VEC3(0.0f, 0.0f, 0.0f);
    const LFS_LOSS_VEC3 purple = LFS_LOSS_VEC3(0.22f, 0.02f, 0.47f);
    const LFS_LOSS_VEC3 red = LFS_LOSS_VEC3(0.72f, 0.12f, 0.29f);
    const LFS_LOSS_VEC3 yellow = LFS_LOSS_VEC3(0.99f, 0.65f, 0.04f);
    const LFS_LOSS_VEC3 white = LFS_LOSS_VEC3(1.0f, 1.0f, 0.75f);
    if (intensity < 0.25f) {
        return mix(black, purple, intensity * 4.0f);
    }
    if (intensity < 0.5f) {
        return mix(purple, red, (intensity - 0.25f) * 4.0f);
    }
    if (intensity < 0.75f) {
        return mix(red, yellow, (intensity - 0.5f) * 4.0f);
    }
    return mix(yellow, white, (intensity - 0.75f) * 4.0f);
}
