/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "eval_mask.hpp"

#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "io/cuda/image_format_kernels.cuh"
#include "training/kernels/mask_preprocess.hpp"

#include <stdexcept>
#include <utility>

namespace lfs::training {
    namespace {

        [[nodiscard]] bool is_segment_and_ignore(const lfs::core::param::MaskMode mode) {
            return mode == lfs::core::param::MaskMode::SegmentAndIgnore;
        }

        [[nodiscard]] lfs::core::Tensor finalize_binary_metrics_mask(lfs::core::Tensor mask) {
            return mask.ge(0.5f).to(lfs::core::DataType::UInt8).contiguous();
        }

        [[nodiscard]] std::expected<LoadedMetricsMask, std::string> load_rgba_metrics_inputs(
            const lfs::core::Camera& camera,
            const MetricsMaskLoadConfig& config) {
            try {
                const auto stream = lfs::core::getCurrentCUDAStream();
                auto [img_data, width, height, channels] = lfs::core::load_image_with_alpha(
                    camera.image_path(), config.resize_factor, config.max_width);

                if (!img_data || channels != 4) {
                    if (img_data) {
                        lfs::core::free_image(img_data);
                    }
                    return std::unexpected("failed to decode RGBA image");
                }

                const auto H = static_cast<size_t>(height);
                const auto W = static_cast<size_t>(width);

                auto cpu_tensor = lfs::core::Tensor::from_blob(
                    img_data, lfs::core::TensorShape({H, W, 4}),
                    lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                // The implicit transfer stream overload completes the upload before returning.
                auto gpu_uint8 = cpu_tensor.to(lfs::core::Device::CUDA);
                lfs::core::free_image(img_data);

                auto rgb = lfs::core::Tensor::zeros(
                    lfs::core::TensorShape({3, H, W}),
                    lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
                auto mask = lfs::core::Tensor::zeros(
                    lfs::core::TensorShape({H, W}),
                    lfs::core::Device::CUDA, lfs::core::DataType::Float32);

                lfs::io::cuda::launch_uint8_rgba_split_to_uint8_rgb_and_float32_alpha(
                    gpu_uint8.ptr<uint8_t>(), rgb.ptr<uint8_t>(), mask.ptr<float>(),
                    H, W, stream);
                gpu_uint8 = lfs::core::Tensor();

                const bool sai = is_segment_and_ignore(config.mask_mode);
                if (config.invert_masks) {
                    lfs::io::cuda::launch_mask_invert(mask.ptr<float>(), H, W, stream);
                }
                // SegmentAndIgnore must keep authored bands through undistort.
                // Binary modes still snap before the warp, then re-binarize after.
                if (!sai && config.mask_threshold > 0.0f) {
                    lfs::io::cuda::launch_mask_threshold(
                        mask.ptr<float>(), H, W, config.mask_threshold, stream);
                }

                if (camera.is_undistort_prepared()) {
                    const auto scaled = lfs::core::scale_undistort_params(
                        camera.undistort_params(),
                        static_cast<int>(W), static_cast<int>(H),
                        config.max_width);
                    auto rgb_float = rgb.to(lfs::core::DataType::Float32) / 255.0f;
                    rgb_float = lfs::core::undistort_image(rgb_float, scaled, stream);
                    auto rgb_uint8 = lfs::core::Tensor::empty(
                        rgb_float.shape(), lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
                    lfs::io::cuda::launch_float32_chw_to_uint8_chw(
                        rgb_float.ptr<float>(),
                        rgb_uint8.ptr<uint8_t>(),
                        rgb_float.shape()[1],
                        rgb_float.shape()[2],
                        rgb_float.shape()[0],
                        stream);
                    rgb = std::move(rgb_uint8);
                    mask = lfs::core::undistort_mask(mask, scaled, stream);
                }

                if (sai) {
                    mask = classify_keep_mask_for_metrics(mask);
                } else {
                    mask = finalize_binary_metrics_mask(std::move(mask));
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
                !sai);
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

} // namespace lfs::training
