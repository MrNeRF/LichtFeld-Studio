// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <optional>
#include <string_view>

namespace lfs::media {
    enum class VideoEncodeCodec { H264,
                                  HEVC };
    [[nodiscard]] inline constexpr std::optional<VideoEncodeCodec> videoEncodeCodecFromId(std::string_view id) {
        if (id == "h264")
            return VideoEncodeCodec::H264;
        if (id == "hevc")
            return VideoEncodeCodec::HEVC;
        return std::nullopt;
    }
} // namespace lfs::media
