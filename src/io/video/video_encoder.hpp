/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/video/video_encoder_interface.hpp"
#include "io/video/video_export_options.hpp"

#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace lfs::io::video {

    class VideoEncoderImpl;

    class VideoEncoder : public IVideoEncoder {
    public:
        VideoEncoder();
        ~VideoEncoder();

        VideoEncoder(const VideoEncoder&) = delete;
        VideoEncoder& operator=(const VideoEncoder&) = delete;
        VideoEncoder(VideoEncoder&&) noexcept;
        VideoEncoder& operator=(VideoEncoder&&) noexcept;

        [[nodiscard]] std::expected<void, std::string> open(
            const std::filesystem::path& output_path,
            const VideoExportOptions& options) override;

        // Write RGBA frame from CPU memory
        [[nodiscard]] std::expected<void, std::string> writeFrame(
            std::span<const uint8_t> rgba_data,
            int width,
            int height);

        [[nodiscard]] std::expected<void, std::string> writeFrameGpu(
            const void* rgba_gpu_ptr,
            int width,
            int height,
            void* cuda_stream = nullptr) override;

        [[nodiscard]] std::expected<void, std::string> close() override;

        [[nodiscard]] bool isOpen() const;

    private:
        std::unique_ptr<VideoEncoderImpl> impl_;
    };

} // namespace lfs::io::video
