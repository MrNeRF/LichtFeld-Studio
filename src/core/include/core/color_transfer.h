// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
// Shared C++/Slang SDR curves. Values outside [0,1] are deliberately retained.
#ifdef __cplusplus
#include <cmath>
namespace lfs::core::color {
    using std::pow;
#define LFS_COLOR_INLINE inline
#else
#define LFS_COLOR_INLINE
#endif
    LFS_COLOR_INLINE float srgbToLinear(float v) {
        return v <= 0.04045f ? v / 12.92f : pow((v + 0.055f) / 1.055f, 2.4f);
    }
    LFS_COLOR_INLINE float bt709ToLinear(float v) {
        return v < 0.081f ? v / 4.5f : pow((v + 0.099f) / 1.099f, 1.0f / 0.45f);
    }
#undef LFS_COLOR_INLINE
#ifdef __cplusplus
} // namespace lfs::core::color
#endif
