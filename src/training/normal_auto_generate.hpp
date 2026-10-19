/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/error.hpp"
#include "core/parameters.hpp"

#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::training {

    // An empty output path means the camera already has that map.
    struct PriorMapJob {
        std::filesystem::path image_path;
        std::filesystem::path depth_output_path;
        std::filesystem::path normal_output_path;
    };

    using PriorMapProgress = std::function<void(
        std::size_t done, std::size_t total, std::string_view filename)>;

    using PriorMapEstimator = std::function<std::expected<void, lfs::Error>(
        std::span<const PriorMapJob> jobs,
        const PriorMapProgress& progress)>;

    [[nodiscard]] bool training_depth_priors_enabled(
        const lfs::core::param::OptimizationParameters& opt);

    // Prior loading and generation require a backend with a normal channel.
    [[nodiscard]] bool training_normal_priors_enabled(
        const lfs::core::param::OptimizationParameters& opt);

    [[nodiscard]] bool normal_auto_generate_needed(
        bool use_normal_loss,
        bool normal_auto_generate,
        float normal_loss_weight,
        std::span<const std::shared_ptr<lfs::core::Camera>> cameras);

    struct PriorMapCounts {
        std::size_t existing = 0;
        std::size_t missing = 0;
    };

    struct PriorAutoGenerateOutcome {
        bool attempted = false;
        bool generated = false;
        bool failed = false;
        PriorMapCounts depth;
        PriorMapCounts normal;
        std::string warning;
    };

    // Generates the missing maps of every enabled prior whose auto-generate flag
    // is on, with one MoGe-2 pass per image. Never fails training: on
    // estimator/download/ONNX errors, logs one warning and leaves cameras
    // without maps so the prior stays inactive.
    PriorAutoGenerateOutcome ensure_training_prior_maps(
        const lfs::core::param::TrainingParameters& params,
        std::span<const std::shared_ptr<lfs::core::Camera>> cameras,
        const PriorMapEstimator& estimator = {});

} // namespace lfs::training
