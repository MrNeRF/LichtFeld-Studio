/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"

namespace lfs::app {

    enum class GpuPreflightDecision {
        UseCuda,
        UseVulkanViewer,
        Fatal,
    };

    [[nodiscard]] inline GpuPreflightDecision decide_gpu_preflight(
        const bool viewer_only,
        const bool cuda_usable,
        const bool vulkan_usable) {
        if (cuda_usable) {
            return GpuPreflightDecision::UseCuda;
        }
        if (viewer_only && vulkan_usable) {
            return GpuPreflightDecision::UseVulkanViewer;
        }
        return GpuPreflightDecision::Fatal;
    }

    [[nodiscard]] inline bool training_params_are_viewer_only(
        const lfs::core::param::TrainingParameters& params) {
        return !params.optimization.headless &&
               !params.render_path.has_value() &&
               params.dataset.data_path.empty() &&
               !params.resume_checkpoint.has_value() &&
               !params.resume_project.has_value();
    }

} // namespace lfs::app
