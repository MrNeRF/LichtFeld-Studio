/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

namespace lfs::io::video {

    enum class VideoPreset : uint8_t {
        YOUTUBE_1080P,      // 1920x1080, 30fps
        YOUTUBE_4K,         // 3840x2160, 30fps
        HD_720P,            // 1280x720, 30fps
        TIKTOK,             // 1080x1920, 30fps (9:16)
        TIKTOK_HD,          // 1080x1920, 60fps (9:16)
        INSTAGRAM_SQUARE,   // 1080x1080, 30fps (1:1)
        INSTAGRAM_PORTRAIT, // 1080x1350, 30fps (4:5)
        CUSTOM
    };

    struct PresetInfo {
        int width;
        int height;
        int framerate;
        int crf;
        const char* name;
        const char* description;
    };

    [[nodiscard]] inline constexpr PresetInfo getPresetInfo(const VideoPreset preset) {
        switch (preset) {
        case VideoPreset::YOUTUBE_1080P:
            return {1920, 1080, 30, 18, "YouTube 1080p", "1920x1080 @ 30fps (16:9)"};
        case VideoPreset::YOUTUBE_4K:
            return {3840, 2160, 30, 18, "YouTube 4K", "3840x2160 @ 30fps (16:9)"};
        case VideoPreset::HD_720P:
            return {1280, 720, 30, 20, "HD 720p", "1280x720 @ 30fps (16:9)"};
        case VideoPreset::TIKTOK:
            return {1080, 1920, 30, 20, "TikTok/Reels", "1080x1920 @ 30fps (9:16)"};
        case VideoPreset::TIKTOK_HD:
            return {1080, 1920, 60, 18, "TikTok HD", "1080x1920 @ 60fps (9:16)"};
        case VideoPreset::INSTAGRAM_SQUARE:
            return {1080, 1080, 30, 20, "Instagram Square", "1080x1080 @ 30fps (1:1)"};
        case VideoPreset::INSTAGRAM_PORTRAIT:
            return {1080, 1350, 30, 20, "Instagram Portrait", "1080x1350 @ 30fps (4:5)"};
        case VideoPreset::CUSTOM:
            return {1920, 1080, 30, 18, "Custom", "Custom resolution"};
        }
        return {1920, 1080, 30, 18, "YouTube 1080p", "1920x1080 @ 30fps"};
    }

    [[nodiscard]] inline constexpr int getPresetCount() {
        return static_cast<int>(VideoPreset::CUSTOM) + 1;
    }

    struct VideoExportOptions {
        VideoPreset preset = VideoPreset::YOUTUBE_1080P;
        int width = 1920;
        int height = 1080;
        int framerate = 30;
        int crf = 18;
    };

} // namespace lfs::io::video
