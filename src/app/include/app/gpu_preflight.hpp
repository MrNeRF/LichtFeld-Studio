/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"

namespace lfs::app {

    enum class GpuPreflightDecision {
        UseCuda,
        UseVulkan,
        Fatal,
    };

    [[nodiscard]] inline GpuPreflightDecision decide_gpu_preflight(
        [[maybe_unused]] const bool viewer_only,
        const bool cuda_usable,
        const bool vulkan_usable,
        const bool explicit_cuda = false,
        const bool cuda_hardware_problem = false) {
        if (cuda_usable) {
            return GpuPreflightDecision::UseCuda;
        }
        if (explicit_cuda || cuda_hardware_problem)
            return GpuPreflightDecision::Fatal;
        if (vulkan_usable) {
            return GpuPreflightDecision::UseVulkan;
        }
        return GpuPreflightDecision::Fatal;
    }

    [[nodiscard]] inline bool training_params_are_viewer_only(
        const lfs::core::param::TrainingParameters& params) {
        return params.render_path.has_value() ||
               (!params.optimization.headless &&
                params.dataset.data_path.empty() &&
                !params.resume_checkpoint.has_value() &&
                !params.resume_project.has_value());
    }

} // namespace lfs::app
