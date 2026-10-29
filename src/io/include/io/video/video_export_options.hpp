/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/provenance.hpp"
#include "io/image_sequence_options.hpp"
#include "io/video/video_frame_geometry.hpp"
#include "io/video/video_reconstruction.hpp"
#include "media/video_encode_codec.hpp"
#include "media/video_output_extent.hpp"

#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

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
        std::optional<core::ProvenanceStamp> provenance{}; // always written to the format's metadata slot; caller chooses full vs minimal, writers fall back to minimal
        VideoReconstructionSelection reconstruction{};
        bool spherical_video = false;
        media::VideoEncodeCodec codec = media::VideoEncodeCodec::H264;
        std::optional<TensorImageSequenceOptions> image_sequence;
        // FPS is framerate / framerate_denominator, e.g. 30000 / 1001.
        int framerate_denominator = 1;
        // Optional camera aspect terms. Zero retains legacy output-sized input.
        // Otherwise writeFrame receives videoCameraRenderSize(...), and the
        // encoder fits that camera image into width/height without changing FOV.
        int camera_width = 0;
        int camera_height = 0;
        VideoFrameFit framing = VideoFrameFit::FIT;
    };

    [[nodiscard]] inline std::expected<void, std::string> validateVideoEncodingOptions(
        const VideoExportOptions& options) {
        if ((options.camera_width != 0 || options.camera_height != 0) &&
            (options.camera_width < 1 || options.camera_width > 8192 || options.camera_height < 1 || options.camera_height > 8192))
            return std::unexpected("Invalid camera aspect ratio");
        if (options.framing < VideoFrameFit::FIT || options.framing > VideoFrameFit::STRETCH)
            return std::unexpected("Invalid video framing options");
        if (options.image_sequence) {
            if (options.width <= 0 || options.height <= 0 || videoFrameRateError(options.framerate, options.framerate_denominator) ||
                static_cast<uint64_t>(options.width) * static_cast<uint64_t>(options.height) > static_cast<uint64_t>(std::numeric_limits<int>::max() / 4))
                return std::unexpected("Invalid image sequence dimensions or framerate");
            // Sequence writers take frames at the output size; only the video encoder reframes another camera aspect.
            const auto camera = videoCameraRenderSize(options.width, options.height, options.camera_width,
                                                      options.camera_height, options.framing);
            if (!options.spherical_video && (camera.width != options.width || camera.height != options.height))
                return std::unexpected("Image sequences need a camera aspect ratio matching their size");
            return {};
        }
        if (const auto error = videoOutputExtentError(options.width, options.height))
            return std::unexpected(std::string(*error));
        if (auto error = videoEncodingRangeError(options.framerate, options.crf, options.framerate_denominator))
            return std::unexpected(std::move(*error));
        if (options.codec != media::VideoEncodeCodec::H264 && options.codec != media::VideoEncodeCodec::HEVC && options.codec != media::VideoEncodeCodec::AV1)
            return std::unexpected("Unknown video codec (got " + std::to_string(static_cast<int>(options.codec)) + ")");

        return {};
    }

    // UI-side bounds for manual WxH entry. The encoder additionally requires even
    // sides (YUV420) and a packed-RGB pixel budget; 8192x8192 is well inside that.
    inline constexpr int MIN_VIDEO_SIDE = 16;
    inline constexpr int MAX_VIDEO_SIDE = 8192;

    struct VideoResolution {
        int width = 0;
        int height = 0;
    };

    [[nodiscard]] inline int clampEvenVideoDimension(int value) {
        if (value < MIN_VIDEO_SIDE)
            value = MIN_VIDEO_SIDE;
        if (value > MAX_VIDEO_SIDE)
            value = MAX_VIDEO_SIDE;
        if ((value & 1) != 0)
            --value;
        return value;
    }

    // Accepts "WxH" or "W x H" (x/X). Rejects empty, missing sides, signs, and trailing junk.
    // Positive dimensions must fit in int; out-of-range and odd values are clamped
    // to even [MIN_VIDEO_SIDE, MAX_VIDEO_SIDE].
    [[nodiscard]] inline std::optional<VideoResolution> parseVideoResolution(const std::string_view text) {
        const char* p = text.data();
        const char* const end = p + text.size();
        const auto skip_ws = [&]() {
            while (p < end && (*p == ' ' || *p == '\t'))
                ++p;
        };
        const auto parse_uint = [&](int& out) -> bool {
            if (p >= end || *p < '0' || *p > '9')
                return false;
            int value = 0;
            while (p < end && *p >= '0' && *p <= '9') {
                const int digit = *p - '0';
                if (value > (std::numeric_limits<int>::max() - digit) / 10)
                    return false;
                value = value * 10 + digit;
                ++p;
            }
            out = value;
            return true;
        };

        skip_ws();
        int width = 0;
        int height = 0;
        if (!parse_uint(width))
            return std::nullopt;
        skip_ws();
        if (p >= end || (*p != 'x' && *p != 'X'))
            return std::nullopt;
        ++p;
        skip_ws();
        if (!parse_uint(height))
            return std::nullopt;
        skip_ws();
        if (p != end)
            return std::nullopt;
        if (width <= 0 || height <= 0)
            return std::nullopt;
        return VideoResolution{clampEvenVideoDimension(width), clampEvenVideoDimension(height)};
    }

} // namespace lfs::io::video
