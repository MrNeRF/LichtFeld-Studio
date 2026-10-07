/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <climits>
#include <cstdint>
#include <format>
#include <optional>
#include <string>

namespace lfs::io::video {

    [[nodiscard]] inline std::optional<std::string> videoEncodingRangeError(int framerate, int crf) {
        if (framerate <= 0 || framerate > 1000)
            return std::format("Video framerate must be between 1 and 1000 (got {})", framerate);
        if (crf < 0 || crf > 51)
            return std::format("Video CRF must be between 0 and 51 (got {})", crf);
        return std::nullopt;
    }

    // Shared by encoder validation and reconstruction preflight. Packed RGB
    // conversion uses signed int indexing, independently of the selected backend.
    [[nodiscard]] inline std::optional<std::string> videoOutputExtentError(
        const int width, const int height) {
        if (width <= 0 || height <= 0)
            return std::format("Video width and height must be positive (got {}x{})", width, height);
        if ((width & 1) != 0 || (height & 1) != 0)
            return std::format("YUV420 video width and height must be even (got {}x{})", width, height);
        if (static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) >
            static_cast<std::uint64_t>(INT_MAX / 3))
            return std::format("Video dimensions exceed the supported pixel budget (got {}x{}, maximum_pixels={})", width, height, INT_MAX / 3);
        return std::nullopt;
    }

} // namespace lfs::io::video
