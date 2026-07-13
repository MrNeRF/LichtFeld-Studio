/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <ranges>
#include <string>
#include <string_view>

namespace lfs::io::video {

    inline constexpr std::array<std::string_view, 7> kSupportedVideoExtensions{
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".flv",
        ".wmv",
    };

    [[nodiscard]] inline std::string normalize_video_extension(std::string_view ext) {
        std::string normalized(ext);
        std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](const unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return normalized;
    }

    [[nodiscard]] inline bool is_supported_video_extension(std::string_view ext) {
        const std::string normalized = normalize_video_extension(ext);
        return std::ranges::find(kSupportedVideoExtensions, normalized) != kSupportedVideoExtensions.end();
    }

    [[nodiscard]] inline std::string supported_video_extensions_display() {
        std::string result;
        for (const std::string_view ext : kSupportedVideoExtensions) {
            if (!result.empty())
                result += ", ";
            result += ext;
        }
        return result;
    }

} // namespace lfs::io::video
