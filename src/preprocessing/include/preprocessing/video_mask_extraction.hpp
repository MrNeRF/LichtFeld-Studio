/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "io/video_frame_extractor.hpp"
#include "preprocessing/video_masking.hpp"

namespace lfs::preprocessing {

    struct VideoMaskExtractionOptions {
        bool enabled = false;
        ImageCircle circle;
        VideoMaskOptions masking;
    };

    [[nodiscard]] std::filesystem::path video_frames_directory(
        const std::filesystem::path& output, bool masking);
    [[nodiscard]] std::filesystem::path video_mask_path(
        const std::filesystem::path& dataset, const std::filesystem::path& image_name);

    // Masks one sequence on a private CUDA stream.
    [[nodiscard]] std::expected<VideoMaskReport, Error> mask_video_frames(
        std::span<const VideoMaskFrame> frames, const ImageCircle& circle, const VideoMaskOptions& masking);

    [[nodiscard]] io::ExtractionOutcome extract_video_frames(
        const io::VideoFrameExtractor::Params& params,
        const VideoMaskExtractionOptions& options, std::string& error, std::string* warning = nullptr);

} // namespace lfs::preprocessing
