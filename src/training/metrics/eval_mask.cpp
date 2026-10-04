/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "eval_mask.hpp"

#include "core/image_io.hpp"
#include "core/nn/ops.hpp"
#include "core/shared_image_ops.hpp"
#include "core/tensor_image.hpp"
#include "io/cache_image_loader.hpp"
#include "lfs/training/ops/masks.hpp"
#include "lfs/training/ops/registry.hpp"

#include <cassert>
#include <stdexcept>
#include <utility>

namespace lfs::training {
    namespace {

        [[nodiscard]] bool is_segment_and_ignore(const lfs::core::param::MaskMode mode) {
            return mode == lfs::core::param::MaskMode::SegmentAndIgnore;
        }

        [[nodiscard]] lfs::core::Tensor finalize_binary_metrics_mask(
            lfs::core::Tensor mask, const float threshold = 0.5f) {
            return mask.ge(threshold).to(lfs::core::DataType::UInt8).contiguous();
        }

        [[nodiscard]] std::expected<LoadedMetricsMask, std::string> load_rgba_metrics_inputs(
            const lfs::core::Camera& camera,
            const MetricsMaskLoadConfig& config) {
            try {
                const bool undistort = camera.is_undistort_prepared() && config.apply_undistortion;
                const auto rgba = lfs::io::load_rgba_image_cpu_decoded(
                    camera.image_path(),
                    undistort ? 1 : config.resize_factor,
                    undistort ? 0 : config.max_width);
                const auto H = rgba.shape()[1];
                const auto W = rgba.shape()[2];
                const auto rgb_float = rgba.slice(0, 0, 3).contiguous();
                auto mask = rgba.slice(0, 3, 4).squeeze(0).contiguous();
                auto rgb = lfs::core::Tensor::empty(
                    lfs::core::TensorShape({3, H, W}),
                    lfs::core::Device::GPU, lfs::core::DataType::UInt8);
                lfs::core::shared_image_ops(lfs::core::default_gpu_backend())->convert(rgb_float, rgb, lfs::gpu_ops::ImageConversion::F32CHWToU8CHW, H, W, 3, {});

                const bool sai = is_segment_and_ignore(config.mask_mode);
                if (config.invert_masks) {
                    lfs::training::training_ops(lfs::core::default_gpu_backend()).shared_image->mask(mask, lfs::gpu_ops::MaskTransform::Invert, 0.f);
                }
                // SegmentAndIgnore must keep authored bands through undistort.
                // Binary modes threshold the final area values after the warp.
                if (!sai && !undistort && config.mask_threshold > 0.0f) {
                    lfs::training::training_ops(lfs::core::default_gpu_backend()).shared_image->mask(mask, lfs::gpu_ops::MaskTransform::Threshold, config.mask_threshold);
                }

                if (undistort) {
                    const auto scaled = lfs::core::prepare_undistort_params(
                        camera.undistort_params(),
                        static_cast<int>(W), static_cast<int>(H),
                        config.resize_factor,
                        config.max_width);
                    auto rgb_float = rgb.to(lfs::core::DataType::Float32) / 255.0f;
                    rgb_float = lfs::core::undistort_image(
                        rgb_float, scaled, nullptr);
                    auto rgb_uint8 = lfs::core::Tensor::empty(
                        rgb_float.shape(), lfs::core::Device::GPU, lfs::core::DataType::UInt8);
                    lfs::training::training_ops(lfs::core::default_gpu_backend()).shared_image->convert(rgb_float, rgb_uint8, lfs::gpu_ops::ImageConversion::F32CHWToU8CHW, rgb_float.shape()[1], rgb_float.shape()[2], rgb_float.shape()[0], {});
                    rgb = std::move(rgb_uint8);
                    mask = lfs::core::undistort_mask_area(mask, scaled, nullptr);
                }

                if (sai) {
                    mask = classify_keep_mask_for_metrics(mask);
                } else {
                    const float threshold = undistort && config.mask_threshold > 0.0f
                                                ? config.mask_threshold
                                                : 0.5f;
                    mask = finalize_binary_metrics_mask(std::move(mask), threshold);
                }
                return LoadedMetricsMask{.gt_image = std::move(rgb), .mask = std::move(mask)};
            } catch (const std::exception& e) {
                return std::unexpected(e.what());
            }
        }

        [[nodiscard]] lfs::core::Tensor load_sidecar_keep_mask(
            lfs::core::Camera& camera,
            const MetricsMaskLoadConfig& config) {
            const bool sai = is_segment_and_ignore(config.mask_mode);
            auto mask = camera.load_and_get_mask(
                config.resize_factor,
                config.max_width,
                config.invert_masks,
                config.mask_threshold,
                !sai,
                config.apply_undistortion);
            if (!mask.is_valid()) {
                return {};
            }
            if (sai) {
                return classify_keep_mask_for_metrics(mask);
            }
            return mask;
        }

    } // namespace

    lfs::core::Tensor classify_keep_mask_for_metrics(const lfs::core::Tensor& mask) {
        if (!mask.is_valid()) {
            return {};
        }
        if (mask.dtype() == lfs::core::DataType::UInt8 ||
            mask.dtype() == lfs::core::DataType::Bool) {
            return mask.gt(250).to(lfs::core::DataType::UInt8).contiguous();
        }
        return mask.gt(lfs::training::kernels::kMaskKeepMin)
            .to(lfs::core::DataType::UInt8)
            .contiguous();
    }

    lfs::core::Tensor load_eval_mask(
        lfs::core::Camera* cam,
        lfs::core::Tensor& gt_image,
        const bool alpha_as_mask,
        const MetricsMaskLoadConfig& config) {
        if (cam == nullptr) {
            return {};
        }
        if (cam->has_mask()) {
            return load_sidecar_keep_mask(*cam, config);
        }
        if (!alpha_as_mask) {
            return {};
        }
        auto loaded = load_rgba_metrics_inputs(*cam, config);
        if (!loaded) {
            return {};
        }
        if (config.replace_gt_image)
            gt_image = std::move(loaded->gt_image);
        return std::move(loaded->mask);
    }

    std::expected<lfs::core::Tensor, std::string> load_external_mask_for_metrics(
        const lfs::core::Camera& camera,
        const MetricsMaskLoadConfig& config) {
        try {
            // Cache population is a lazy decode; Camera::load_and_get_mask is
            // non-const only because it writes the keyed mask cache.
            auto mask = load_sidecar_keep_mask(
                const_cast<lfs::core::Camera&>(camera), config);
            if (!mask.is_valid()) {
                return std::unexpected("failed to decode mask");
            }
            return mask;
        } catch (const std::exception& e) {
            return std::unexpected(e.what());
        }
    }

    std::expected<LoadedMetricsMask, std::string> load_alpha_masked_metrics_inputs(
        const lfs::core::Camera& camera,
        const MetricsMaskLoadConfig& config) {
        return load_rgba_metrics_inputs(camera, config);
    }

    lfs::core::Tensor erode_metrics_mask(const lfs::core::Tensor& mask, const int radius) {
        const int side = 2 * radius + 1;
        const auto input = mask.to(lfs::core::DataType::Float32).unsqueeze(0).unsqueeze(0);
        const auto weight = lfs::core::Tensor::ones({1, 1, size_t(side), size_t(side)}, mask.device());
        lfs::core::nn::Conv2dParams params;
        params.pad_h = params.pad_w = radius;
        return lfs::core::nn::conv2d(input, weight, nullptr, params).squeeze(0).squeeze(0).ge(float(side * side) - 0.5f).to(lfs::core::DataType::UInt8).contiguous();
    }
    lfs::core::Tensor load_eval_alpha(
        const lfs::core::Camera& camera,
        const MetricsMaskLoadConfig& config) {
        const bool undistort = camera.is_undistort_prepared() && config.apply_undistortion;
        const auto rgba = lfs::io::load_rgba_image_cpu_decoded(
            camera.image_path(),
            undistort ? 1 : config.resize_factor,
            undistort ? 0 : config.max_width);
        assert(rgba.ndim() == 3 && rgba.shape()[0] == 4 && rgba.dtype() == lfs::core::DataType::Float32);
        auto alpha = rgba.slice(0, 3, 4).squeeze(0).contiguous();
        if (undistort) {
            const auto scaled = lfs::core::prepare_undistort_params(
                camera.undistort_params(),
                static_cast<int>(rgba.shape()[2]), static_cast<int>(rgba.shape()[1]),
                config.resize_factor,
                config.max_width);
            alpha = lfs::core::undistort_mask_area(alpha, scaled, nullptr);
        }
        return alpha;
    }

    lfs::core::Tensor composite_over_background(const lfs::core::Tensor& rgb,
                                                const lfs::core::Tensor& alpha,
                                                const lfs::core::Tensor& background) {
        using lfs::core::DataType;
        assert(rgb.ndim() == 3 && rgb.shape()[0] == 3);
        assert(rgb.dtype() == DataType::Float32 || rgb.dtype() == DataType::UInt8);
        const size_t height = rgb.shape()[1], width = rgb.shape()[2];
        assert(alpha.dtype() == DataType::Float32 && alpha.numel() == height * width);
        assert(background.dtype() == DataType::Float32 &&
               (background.numel() == 3 || background.numel() == 3 * height * width));
        const auto color = rgb.dtype() == DataType::UInt8 ? rgb.to(DataType::Float32) / 255.0f : rgb;
        const auto coverage = alpha.reshape({1, static_cast<int>(height), static_cast<int>(width)}).clamp(0.0f, 1.0f);
        const auto backdrop = background.numel() == 3
                                  ? background.reshape({3, 1, 1})
                                  : background.reshape({3, static_cast<int>(height), static_cast<int>(width)});
        return (color * coverage + backdrop * (coverage.neg() + 1.0f)).contiguous();
    }

} // namespace lfs::training
