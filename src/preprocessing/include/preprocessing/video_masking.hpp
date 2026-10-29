/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/tensor.hpp"
#include "preprocessing/operator_view.hpp"
#include <functional>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace lfs::preprocessing {

    struct ImageCircle {
        double center_x = 0.5, center_y = 0.5;
        double radius_x = 0, radius_y = 0;
    };

    struct VideoMaskFrame {
        std::filesystem::path image_path;
        std::filesystem::path mask_path;
        int width = 0, height = 0;
        double time_seconds = 0;
    };

    // Motion pairs in which a grid cell kept its content (fixed) or changed (moving), with the thirds of the
    // sequence they fell into and the number of frames with fixed evidence.
    struct VideoPromptCellEvidence {
        size_t fixed_pairs = 0, moving_pairs = 0, support_frames = 0;
        std::uint8_t fixed_time_bins = 0, moving_time_bins = 0;
    };

    struct VideoMaskDetectionReport {
        std::string status;
        std::vector<std::uint8_t> prompt_cells;
        std::vector<VideoPromptCellEvidence> evidence;
        std::vector<std::array<int, 2>> positive_cells;
        std::vector<std::array<double, 2>> negative_points;
        size_t images = 0, pairs_tested = 0, motion_pairs = 0;
    };

    struct VideoMaskReport {
        VideoMaskDetectionReport detection;
        std::string warning;
        double mask_wall_seconds = 0;
    };

    enum class VideoMaskStage { Patches,
                                Motion,
                                Seed,
                                Forward,
                                Reverse,
                                Writing };

    struct VideoMaskProgress {
        VideoMaskStage stage;
        size_t completed = 0, total = 0;
    };

    struct VideoMaskSeedPrompt {
        int frame = 0; // Index in timestamp order.
        // Normalized x, y and label (1 foreground, 0 background).
        std::vector<std::array<float, 3>> points;
        // Optional normalized x0, y0, x1, y1 bounds for small or partially visible operators.
        std::optional<std::array<float, 4>> box;
    };

    struct VideoMaskOptions {
        // Circle masks are still written when SAM2 inference is disabled.
        bool enabled = true;
        std::filesystem::path weights;
        bool allow_download = true;
        // Stop before propagation when the operator cannot be localized.
        bool require_operator = false;
        // Zero uses available VRAM. A cap limits decode buffering.
        size_t gpu_memory_budget_bytes = 0;
        bool recenter_fisheye = false;
        float fisheye_fov_degrees = 200;
        float view_fov_degrees = 160;
        int dilation_radius = 12;
        std::optional<VideoMaskSeedPrompt> prompt;
        std::function<bool()> cancel_requested;
        std::function<void(const VideoMaskProgress&)> progress;
    };

    struct VideoMaskResourcePlan {
        size_t budget_bytes = 0, reserve_bytes = 0, worker_bytes = 0;
    };

    // Conservative admission estimate, independent of sequence length. This does not
    // impose a CUDA allocator quota or reserve memory against external GPU consumers.
    std::expected<VideoMaskResourcePlan, Error> plan_video_mask_resources(
        std::span<const VideoMaskFrame> frames, const VideoMaskOptions& options,
        size_t available_bytes);

    // One video sequence, ordered by timestamp, with a single temporal memory.
    // Masks use 0 for ignored pixels and 255 for retained pixels. Missing operators produce circle masks only.
    std::expected<VideoMaskReport, Error> mask_video_sequence(
        std::span<const VideoMaskFrame> frames, const ImageCircle& circle,
        const VideoMaskOptions& options, cudaStream_t stream);

} // namespace lfs::preprocessing
