/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/tensor.hpp"

#include <expected>
#include <string>

namespace lfs::training {

    struct MetricsMaskLoadConfig {
        int resize_factor = -1;
        int max_width = 0;
        bool invert_masks = false;
        float mask_threshold = 0.5f;
        lfs::core::param::MaskMode mask_mode = lfs::core::param::MaskMode::None;
    };

    struct LoadedMetricsMask {
        lfs::core::Tensor gt_image;
        lfs::core::Tensor mask;
    };

    [[nodiscard]] inline MetricsMaskLoadConfig metrics_mask_config_from(
        const lfs::core::param::TrainingParameters& params) {
        return {
            .resize_factor = params.dataset.resize_factor,
            .max_width = params.dataset.max_width,
            .invert_masks = params.optimization.invert_masks,
            .mask_threshold = params.optimization.mask_threshold,
            .mask_mode = params.optimization.mask_mode,
        };
    }

    /// Classify a decoded mask to a UInt8 {0,1} keep mask.
    /// UInt8/Bool carry authored 0..255 samples (keep = value > 250).
    /// Float32 is the pipelined-loader domain [0,1] (keep = value > kMaskKeepMin).
    /// Never compare normalized floats against 250.
    [[nodiscard]] lfs::core::Tensor classify_keep_mask_for_metrics(const lfs::core::Tensor& mask);

    /// Batch-eval mask loader used by MetricsEvaluator::load_eval_mask.
    /// Sidecar (file or in-memory) wins over RGBA alpha.
    [[nodiscard]] lfs::core::Tensor load_eval_mask(
        lfs::core::Camera* cam,
        lfs::core::Tensor& gt_image,
        bool alpha_as_mask,
        const MetricsMaskLoadConfig& config);

    /// Interactive sidecar loader. Uses Camera::load_and_get_mask (lossless PNG /
    /// in-memory), not RGB JPEG recode via load_image_immediate.
    [[nodiscard]] std::expected<lfs::core::Tensor, std::string> load_external_mask_for_metrics(
        const lfs::core::Camera& camera,
        const MetricsMaskLoadConfig& config);

    /// Interactive / eval RGBA-alpha loader. Skips mask_threshold in SegmentAndIgnore.
    [[nodiscard]] std::expected<LoadedMetricsMask, std::string> load_alpha_masked_metrics_inputs(
        const lfs::core::Camera& camera,
        const MetricsMaskLoadConfig& config);

} // namespace lfs::training
