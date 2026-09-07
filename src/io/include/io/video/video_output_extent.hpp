/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <climits>
#include <cstdint>
#include <optional>
#include <string_view>

namespace lfs::io::video {

    // Shared by encoder validation and reconstruction preflight. Packed RGB
    // conversion uses signed int indexing, independently of the selected backend.
    [[nodiscard]] inline std::optional<std::string_view> videoOutputExtentError(
        const int width, const int height) {
        if (width <= 0 || height <= 0)
            return "Video width and height must be positive";
        if ((width & 1) != 0 || (height & 1) != 0)
            return "YUV420 video width and height must be even";
        if (static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) >
            static_cast<std::uint64_t>(INT_MAX / 3))
            return "Video dimensions exceed the supported pixel budget";
        return std::nullopt;
    }

} // namespace lfs::io::video
