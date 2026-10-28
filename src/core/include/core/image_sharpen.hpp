/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace lfs::core {

    enum class SharpenKernel : std::uint8_t {
        Usm,
        Laplacian4,
        Laplacian8,
    };

    [[nodiscard]] inline constexpr std::string_view sharpen_kernel_name(const SharpenKernel kernel) noexcept {
        switch (kernel) {
        case SharpenKernel::Usm:
            return "usm";
        case SharpenKernel::Laplacian4:
            return "laplacian4";
        case SharpenKernel::Laplacian8:
            return "laplacian8";
        }
        return "usm";
    }

    [[nodiscard]] inline constexpr std::optional<SharpenKernel> sharpen_kernel_from_string(
        const std::string_view value) noexcept {
        if (value == "usm")
            return SharpenKernel::Usm;
        if (value == "laplacian4")
            return SharpenKernel::Laplacian4;
        if (value == "laplacian8")
            return SharpenKernel::Laplacian8;
        return std::nullopt;
    }

    inline constexpr float kMaxSharpenAmount = 5.0f;
    inline constexpr float kMinSharpenRadius = 0.3f;
    inline constexpr float kMaxSharpenRadius = 10.0f;

    // Luma sharpening of training images after resize and undistortion; amount 0 turns it off.
    struct SharpenParams {
        SharpenKernel kernel = SharpenKernel::Usm;
        float amount = 0.0f;
        float radius = 1.0f;    // Gaussian sigma in output pixels; usm only
        float threshold = 0.0f; // smallest luma difference that is sharpened, on the 0..1 scale

        [[nodiscard]] bool enabled() const noexcept { return amount > 0.0f; }
        bool operator==(const SharpenParams&) const = default;
    };

} // namespace lfs::core
