// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
// Shared C++/Slang curves. SDR retains values outside [0,1].
#ifdef __cplusplus
#include <cmath>
namespace lfs::core::color {
    using std::exp;
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
    // ITU-R BT.2100 PQ EOTF normalized to the 10,000 cd/m2 reference peak.
    // PQ is defined on [0,1]; callers clamp encoded RGB before this operation.
    LFS_COLOR_INLINE float pqToLinear(float v) {
        float p = pow(v > 0.0f ? v : 0.0f, 32.0f / 2523.0f);
        float numerator = p - 3424.0f / 4096.0f;
        numerator = numerator > 0.0f ? numerator : 0.0f;
        return pow(numerator / (2413.0f / 128.0f - 2392.0f / 128.0f * p), 16384.0f / 2610.0f);
    }
    // Absolute cd/m2, without mastering-peak rescale.
    LFS_COLOR_INLINE float pqToNits(float v) {
        return 10000.0f * pqToLinear(v);
    }
    // BT.2100 inverse HLG OETF: relative scene light, normalized to [0,1].
    // No display OOTF, system gamma, black lift or absolute luminance is applied.
    LFS_COLOR_INLINE float hlgToSceneLinear(float v) {
        return v <= 0.5f ? v * v / 3.0f : (exp((v - 0.55991073f) / 0.17883277f) + 0.28466892f) / 12.0f;
    }
#undef LFS_COLOR_INLINE
#ifdef __cplusplus
} // namespace lfs::core::color
#endif
