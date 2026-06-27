/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <string_view>

namespace lfs::io {

    // Supports %d, %0Nd zero-padding, %% escaping, and legacy %000 zero-padding.
    [[nodiscard]] std::string formatFrameFilenameStem(std::string_view pattern, int frame_number);

    enum class ExtractionMode {
        FPS,     // Extract at specific FPS
        INTERVAL // Extract every N frames
    };

    enum class ImageFormat {
        PNG,
        JPG
    };

    enum class ResolutionMode {
        Original,
        Scale,
        Custom
    };

    class VideoFrameExtractor {
    public:
        VideoFrameExtractor();
        ~VideoFrameExtractor();

        struct Params {
            std::filesystem::path video_path;
            std::filesystem::path output_dir;
            ExtractionMode mode = ExtractionMode::FPS;
            double fps = 1.0;
            int frame_interval = 1;
            ImageFormat format = ImageFormat::PNG;
            int jpg_quality = 95;
            std::function<void(int, int)> progress_callback; // (current, total)
            std::function<bool()> cancel_requested;

            // Trim range
            double start_time = 0.0;
            double end_time = -1.0; // -1 means end of video

            // Resolution
            ResolutionMode resolution_mode = ResolutionMode::Original;
            float scale = 1.0f;
            int custom_width = 0;
            int custom_height = 0;

            // Output naming
            std::string filename_pattern = "frame_%d"; // %d = frame number
        };

        bool extract(const Params& params, std::string& error);

    private:
        class Impl;
        Impl* impl_;
    };

} // namespace lfs::io
