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

    struct NormalAutoGenerateJob {
        std::filesystem::path image_path;
        std::filesystem::path output_path;
    };

    using NormalGenerateProgress = std::function<void(
        std::size_t done, std::size_t total, std::string_view filename)>;

    using NormalEstimator = std::function<std::expected<void, lfs::Error>(
        std::span<const NormalAutoGenerateJob> jobs,
        const NormalGenerateProgress& progress)>;

    [[nodiscard]] bool normal_auto_generate_needed(
        bool use_normal_loss,
        bool normal_auto_generate,
        float normal_loss_weight,
        std::span<const std::shared_ptr<lfs::core::Camera>> cameras);

    struct NormalAutoGenerateOutcome {
        bool attempted = false;
        bool generated = false;
        bool failed = false;
        std::size_t existing_count = 0;
        std::size_t missing_count = 0;
        std::string warning;
    };

    // Never fails training. On estimator/download/ONNX errors, logs one warning
    // and leaves cameras without maps so the prior stays inactive.
    NormalAutoGenerateOutcome ensure_training_normal_maps(
        const lfs::core::param::TrainingParameters& params,
        std::span<const std::shared_ptr<lfs::core::Camera>> cameras,
        const NormalEstimator& estimator = {});

} // namespace lfs::training
