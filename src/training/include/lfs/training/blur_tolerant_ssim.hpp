// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <algorithm>
namespace lfs::training::kernels {
    inline constexpr float BLUR_TOLERANT_SSIM_TEMPERATURE = 50.0f;
    inline constexpr float BLUR_TOLERANT_SSIM_FLAT_VARIANCE = 0.0005f;
    inline constexpr int BLUR_TOLERANT_SSIM_MIN_SHORT_SIDE = 1200;
    inline constexpr float BLUR_TOLERANT_SSIM_REFERENCE_SHORT_SIDE = 3000.0f;
    inline constexpr float BLUR_TOLERANT_SSIM_MIN_SIGMA = 0.5f;
    inline constexpr float BLUR_TOLERANT_SSIM_MAX_SIGMA = 1.5f;

    // The blur sigma of a photo of this size; 0 means plain SSIM.
    [[nodiscard]] inline float blur_tolerant_ssim_sigma(const int height, const int width) {
        const int short_side = std::min(height, width);
        if (short_side < BLUR_TOLERANT_SSIM_MIN_SHORT_SIDE)
            return 0.0f;
        return std::clamp(BLUR_TOLERANT_SSIM_MAX_SIGMA * static_cast<float>(short_side) /
                              BLUR_TOLERANT_SSIM_REFERENCE_SHORT_SIDE,
                          BLUR_TOLERANT_SSIM_MIN_SIGMA, BLUR_TOLERANT_SSIM_MAX_SIGMA);
    }

} // namespace lfs::training::kernels
