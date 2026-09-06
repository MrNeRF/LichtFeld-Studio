/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "bilateral_grid.hpp"
#include "config_serialization.hpp"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <nvtx3/nvToolsExt.h>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace lfs::training {

    namespace {
        constexpr uint32_t CHECKPOINT_MAGIC = 0x4C464247; // "LFBG"
        constexpr uint32_t CHECKPOINT_MIN_VERSION = 1;
        constexpr uint32_t CHECKPOINT_VERSION = 3;
        constexpr uint32_t CONFIG_SCHEMA_VERSION = 2;
        constexpr uint32_t CONFIG_SCHEMA_V1_BYTES = 52;
        constexpr uint32_t CONFIG_SCHEMA_V2_BYTES = 56;

        [[nodiscard]] bool is_known_parameterization(const int32_t value) {
            return value == static_cast<int32_t>(BilateralGridParameterization::Affine) ||
                   value == static_cast<int32_t>(BilateralGridParameterization::ExposureChroma);
        }

        struct LegacyConfigV1 {
            double lr;
            double beta1;
            double beta2;
            double eps;
            int warmup_steps;
            double warmup_start_factor;
            double final_lr_factor;
        };

        struct DeserializedGridConfig {
            BilateralGrid::Config config{};
            BilateralGridParameterization parameterization = BilateralGridParameterization::Affine;
        };

        void serialize_config(std::ostream& os,
                              const BilateralGrid::Config& config,
                              const BilateralGridParameterization parameterization) {
            using config_serialization_detail::write_little_endian;

            // Schema is append-only. Bump the schema version and append fields;
            // payload size lets older readers skip the suffix and retain defaults.
            write_little_endian(os, CONFIG_SCHEMA_VERSION, "bilateral-grid config schema");
            write_little_endian(os, CONFIG_SCHEMA_V2_BYTES, "bilateral-grid config size");
            write_little_endian(os, config.lr, "bilateral-grid config lr");
            write_little_endian(os, config.beta1, "bilateral-grid config beta1");
            write_little_endian(os, config.beta2, "bilateral-grid config beta2");
            write_little_endian(os, config.eps, "bilateral-grid config eps");
            write_little_endian(
                os, static_cast<int32_t>(config.warmup_steps), "bilateral-grid config warmup_steps");
            write_little_endian(
                os, config.warmup_start_factor, "bilateral-grid config warmup_start_factor");
            write_little_endian(os, config.final_lr_factor, "bilateral-grid config final_lr_factor");
            write_little_endian(os, static_cast<int32_t>(parameterization),
                                "bilateral-grid config parameterization");
        }

        [[nodiscard]] DeserializedGridConfig deserialize_config(std::istream& is) {
            using config_serialization_detail::read_little_endian;

            const uint32_t schema_version =
                read_little_endian<uint32_t>(is, "bilateral-grid config schema");
            const uint32_t payload_bytes =
                read_little_endian<uint32_t>(is, "bilateral-grid config size");
            if (schema_version == 0) {
                config_serialization_detail::throw_config_data_loss(
                    "bilateral-grid config schema", "version must be positive");
            }
            if (payload_bytes < CONFIG_SCHEMA_V1_BYTES ||
                payload_bytes > config_serialization_detail::MAX_CONFIG_PAYLOAD_BYTES) {
                config_serialization_detail::throw_config_data_loss(
                    "bilateral-grid config size", "payload size is out of bounds");
            }

            DeserializedGridConfig parsed{};
            parsed.config.lr = read_little_endian<double>(is, "bilateral-grid config lr");
            parsed.config.beta1 = read_little_endian<double>(is, "bilateral-grid config beta1");
            parsed.config.beta2 = read_little_endian<double>(is, "bilateral-grid config beta2");
            parsed.config.eps = read_little_endian<double>(is, "bilateral-grid config eps");
            parsed.config.warmup_steps =
                read_little_endian<int32_t>(is, "bilateral-grid config warmup_steps");
            parsed.config.warmup_start_factor =
                read_little_endian<double>(is, "bilateral-grid config warmup_start_factor");
            parsed.config.final_lr_factor =
                read_little_endian<double>(is, "bilateral-grid config final_lr_factor");
            uint32_t consumed = CONFIG_SCHEMA_V1_BYTES;
            if (payload_bytes >= CONFIG_SCHEMA_V2_BYTES) {
                const int32_t parameterization =
                    read_little_endian<int32_t>(is, "bilateral-grid config parameterization");
                if (!is_known_parameterization(parameterization)) {
                    config_serialization_detail::throw_config_data_loss(
                        "bilateral-grid config parameterization", "unknown parameterization");
                }
                parsed.parameterization = static_cast<BilateralGridParameterization>(parameterization);
                consumed = CONFIG_SCHEMA_V2_BYTES;
            }
            config_serialization_detail::skip_bytes(
                is, payload_bytes - consumed, "bilateral-grid config");
            return parsed;
        }

        [[nodiscard]] BilateralGrid::Config deserialize_legacy_config(std::istream& is) {
            static_assert(std::is_standard_layout_v<LegacyConfigV1>);
            static_assert(sizeof(LegacyConfigV1) == 56);
            static_assert(offsetof(LegacyConfigV1, lr) == 0);
            static_assert(offsetof(LegacyConfigV1, beta1) == 8);
            static_assert(offsetof(LegacyConfigV1, beta2) == 16);
            static_assert(offsetof(LegacyConfigV1, eps) == 24);
            static_assert(offsetof(LegacyConfigV1, warmup_steps) == 32);
            static_assert(offsetof(LegacyConfigV1, warmup_start_factor) == 40);
            static_assert(offsetof(LegacyConfigV1, final_lr_factor) == 48);

            LegacyConfigV1 legacy{};
            lfs::core::serialization_detail::read_exact(
                is, &legacy, sizeof(legacy), "legacy bilateral-grid configuration");
            return {
                .lr = legacy.lr,
                .beta1 = legacy.beta1,
                .beta2 = legacy.beta2,
                .eps = legacy.eps,
                .warmup_steps = legacy.warmup_steps,
                .warmup_start_factor = legacy.warmup_start_factor,
                .final_lr_factor = legacy.final_lr_factor,
            };
        }

        struct ImageLayout {
            bool chw = false;
            int height = 0;
            int width = 0;
        };

        [[nodiscard]] size_t validated_grid_elements(
            const int num_images,
            const int grid_width,
            const int grid_height,
            const int grid_guidance,
            const int channels,
            const int total_iterations,
            const BilateralGrid::Config& config) {
            if (num_images <= 0 || grid_width <= 0 || grid_height <= 0 || grid_guidance <= 0)
                throw std::invalid_argument("BilateralGrid dimensions and image count must be positive");
            if (channels <= 0)
                throw std::invalid_argument("BilateralGrid channel count must be positive");
            if (total_iterations <= 0)
                throw std::invalid_argument("BilateralGrid total_iterations must be positive");
            if (!std::isfinite(config.lr) || config.lr < 0.0 ||
                !std::isfinite(config.beta1) || config.beta1 < 0.0 || config.beta1 >= 1.0 ||
                !std::isfinite(config.beta2) || config.beta2 < 0.0 || config.beta2 >= 1.0 ||
                !std::isfinite(config.eps) || config.eps <= 0.0 || config.warmup_steps < 0 ||
                !std::isfinite(config.warmup_start_factor) || config.warmup_start_factor < 0.0 ||
                !std::isfinite(config.final_lr_factor) || config.final_lr_factor <= 0.0) {
                throw std::invalid_argument("Invalid BilateralGrid optimizer configuration");
            }

            uint64_t elements = static_cast<uint64_t>(num_images);
            for (const uint64_t factor : {
                     static_cast<uint64_t>(channels),
                     static_cast<uint64_t>(grid_guidance),
                     static_cast<uint64_t>(grid_height),
                     static_cast<uint64_t>(grid_width)}) {
                if (elements > std::numeric_limits<uint64_t>::max() / factor)
                    throw std::length_error("BilateralGrid allocation size overflows");
                elements *= factor;
            }
            if (elements > lfs::core::MAX_SERIALIZED_TENSOR_BYTES / sizeof(float) ||
                elements > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                throw std::length_error("BilateralGrid allocation exceeds the CUDA kernel index budget");
            }
            return static_cast<size_t>(elements);
        }

        [[nodiscard]] ImageLayout validate_image_tensor(
            const lfs::core::Tensor& tensor,
            const std::string_view operation) {
            if (!tensor.is_valid())
                throw std::invalid_argument(std::string(operation) + ": image tensor is invalid");
            if (tensor.device() != lfs::core::Device::GPU || tensor.dtype() != lfs::core::DataType::Float32)
                throw std::invalid_argument(std::string(operation) + ": image tensor must be CUDA Float32");
            if (tensor.ndim() != 3)
                throw std::invalid_argument(std::string(operation) + ": image tensor must have rank 3");

            const auto& shape = tensor.shape();
            const bool chw = shape[0] == 3;
            const bool hwc = shape[2] == 3;
            if (!chw && !hwc)
                throw std::invalid_argument(std::string(operation) + ": expected CHW or HWC image with 3 channels");

            const size_t height = chw ? shape[1] : shape[0];
            const size_t width = chw ? shape[2] : shape[1];
            if (height == 0 || width == 0 ||
                height > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                width > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                height > static_cast<size_t>(std::numeric_limits<int>::max()) / width) {
                throw std::invalid_argument(std::string(operation) + ": image dimensions must be positive signed ints");
            }
            return {
                .chw = chw,
                .height = static_cast<int>(height),
                .width = static_cast<int>(width),
            };
        }
    } // namespace

    BilateralGrid::BilateralGrid(int num_images, int grid_W, int grid_H, int grid_L,
                                 int total_iterations, Config config,
                                 BilateralGridParameterization parameterization)
        : config_(config),
          current_lr_(config.warmup_steps > 0 ? config.lr * config.warmup_start_factor : config.lr),
          initial_lr_(config.lr),
          total_iterations_(total_iterations),
          num_images_(num_images),
          grid_width_(grid_W),
          grid_height_(grid_H),
          grid_guidance_(grid_L),
          channels_(bilateral_grid_channel_count(parameterization)),
          parameterization_(parameterization) {

        const size_t grid_elements = validated_grid_elements(
            num_images, grid_W, grid_H, grid_L, channels_, total_iterations, config);

        const size_t n = static_cast<size_t>(num_images);
        const size_t c = static_cast<size_t>(channels_);
        const size_t guidance = static_cast<size_t>(grid_L);
        const size_t height = static_cast<size_t>(grid_H);
        const size_t width = static_cast<size_t>(grid_W);
        grids_ = lfs::core::Tensor::empty({n, c, guidance, height, width}, lfs::core::Device::GPU);

        if (parameterization_ == BilateralGridParameterization::Affine) {
            kernels::launch_bilateral_grid_init_identity(
                grids_.ptr<float>(), num_images, grid_L, grid_H, grid_W, nullptr);
        } else {
            grids_.zero_();
        }

        exp_avg_ = lfs::core::Tensor::zeros(grids_.shape(), lfs::core::Device::GPU);
        exp_avg_sq_ = lfs::core::Tensor::zeros(grids_.shape(), lfs::core::Device::GPU);
        slice_grad_ = lfs::core::Tensor::zeros(
            {c, guidance, height, width}, lfs::core::Device::GPU);

        const size_t spatial = guidance * height * width;
        const size_t temp_size = std::max(size_t(2048), (spatial + 255) / 256);
        tv_temp_buffer_ = lfs::core::Tensor::empty({temp_size}, lfs::core::Device::GPU);
        tv_loss_scalar_ = lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);

        last_step_.assign(n, 0);

        rebuild_identity_mean();
        rebuild_projection_state();

        LOG_DEBUG("BilateralGrid: {}x{}x{} for {} images, C={}, lr={:.2e}",
                  grid_W, grid_H, grid_L, num_images, channels_, config.lr);
    }

    lfs::core::Tensor BilateralGrid::apply(const lfs::core::Tensor& rgb, int image_idx) {
        if (image_idx < 0 || image_idx >= num_images_) {
            throw std::out_of_range("BilateralGrid::apply: image_idx out of range");
        }

        const ImageLayout layout = validate_image_tensor(rgb, "BilateralGrid::apply");
        const auto& shape = rgb.shape();
        const auto rgb_cont = rgb.contiguous();
        const float* grid_ptr = slice_ptr(grids_, image_idx);
        const float* offset_ptr = shared_offset_.ptr<float>();
        assert(static_cast<int>(grids_.shape()[1]) == channels_);

        if (layout.chw) {
            auto output = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::GPU);
            if (parameterization_ == BilateralGridParameterization::ExposureChroma) {
                kernels::launch_bilateral_grid_slice_forward_exposure_chroma_chw(
                    grid_ptr, rgb_cont.ptr<float>(), output.ptr<float>(),
                    grid_guidance_, grid_height_, grid_width_, layout.height, layout.width,
                    offset_ptr, nullptr);
            } else {
                kernels::launch_bilateral_grid_slice_forward_chw(
                    grid_ptr, rgb_cont.ptr<float>(), output.ptr<float>(),
                    grid_guidance_, grid_height_, grid_width_, layout.height, layout.width,
                    offset_ptr, nullptr);
            }
            return output;
        }

        auto output = lfs::core::Tensor::empty({shape[0], shape[1], 3}, lfs::core::Device::GPU);
        if (parameterization_ == BilateralGridParameterization::ExposureChroma) {
            kernels::launch_bilateral_grid_slice_forward_exposure_chroma(
                grid_ptr, rgb_cont.ptr<float>(), output.ptr<float>(),
                grid_guidance_, grid_height_, grid_width_, layout.height, layout.width,
                offset_ptr, nullptr);
        } else {
            kernels::launch_bilateral_grid_slice_forward(
                grid_ptr, rgb_cont.ptr<float>(), output.ptr<float>(),
                grid_guidance_, grid_height_, grid_width_, layout.height, layout.width,
                offset_ptr, nullptr);
        }
        return output;
    }

    lfs::core::Tensor BilateralGrid::backward(const lfs::core::Tensor& rgb,
                                              const lfs::core::Tensor& grad_output,
                                              int image_idx) {
        if (image_idx < 0 || image_idx >= num_images_) {
            throw std::out_of_range("BilateralGrid::backward: image_idx out of range");
        }

        const ImageLayout layout = validate_image_tensor(rgb, "BilateralGrid::backward");
        const ImageLayout grad_layout = validate_image_tensor(grad_output, "BilateralGrid::backward");
        if (rgb.shape() != grad_output.shape() || layout.chw != grad_layout.chw)
            throw std::invalid_argument("BilateralGrid::backward: rgb and grad_output shapes must match");

        const auto& shape = rgb.shape();
        const auto rgb_cont = rgb.contiguous();
        const auto grad_cont = grad_output.contiguous();
        const float* grid_ptr = slice_ptr(grids_, image_idx);
        const float* offset_ptr = shared_offset_.ptr<float>();
        float* grad_grid_ptr = slice_grad_.ptr<float>();
        assert(static_cast<int>(grids_.shape()[1]) == channels_);

        LFS_CUDA_CHECK(cudaMemsetAsync(
            grad_grid_ptr, 0, slice_elements() * sizeof(float), nullptr));

        if (layout.chw) {
            auto grad_rgb = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::GPU);
            if (parameterization_ == BilateralGridParameterization::ExposureChroma) {
                kernels::launch_bilateral_grid_slice_backward_exposure_chroma_chw(
                    grid_ptr, rgb_cont.ptr<float>(), grad_cont.ptr<float>(),
                    grad_grid_ptr, grad_rgb.ptr<float>(),
                    grid_guidance_, grid_height_, grid_width_, layout.height, layout.width,
                    offset_ptr, nullptr);
            } else {
                kernels::launch_bilateral_grid_slice_backward_chw(
                    grid_ptr, rgb_cont.ptr<float>(), grad_cont.ptr<float>(),
                    grad_grid_ptr, grad_rgb.ptr<float>(),
                    grid_guidance_, grid_height_, grid_width_, layout.height, layout.width,
                    offset_ptr, nullptr);
            }
            return grad_rgb;
        }

        auto grad_rgb = lfs::core::Tensor::empty({shape[0], shape[1], 3}, lfs::core::Device::GPU);
        if (parameterization_ == BilateralGridParameterization::ExposureChroma) {
            kernels::launch_bilateral_grid_slice_backward_exposure_chroma(
                grid_ptr, rgb_cont.ptr<float>(), grad_cont.ptr<float>(),
                grad_grid_ptr, grad_rgb.ptr<float>(),
                grid_guidance_, grid_height_, grid_width_, layout.height, layout.width,
                offset_ptr, nullptr);
        } else {
            kernels::launch_bilateral_grid_slice_backward(
                grid_ptr, rgb_cont.ptr<float>(), grad_cont.ptr<float>(),
                grad_grid_ptr, grad_rgb.ptr<float>(),
                grid_guidance_, grid_height_, grid_width_, layout.height, layout.width,
                offset_ptr, nullptr);
        }
        return grad_rgb;
    }

    lfs::core::Tensor BilateralGrid::tv_loss_gpu() {
        assert(static_cast<int>(grids_.shape()[1]) == channels_);
        LFS_CUDA_CHECK(cudaMemsetAsync(
            tv_loss_scalar_.ptr<float>(), 0, sizeof(float), nullptr));
        kernels::launch_bilateral_grid_tv_forward(
            grids_.ptr<float>(), tv_loss_scalar_.ptr<float>(), tv_temp_buffer_.ptr<float>(),
            num_images_, channels_, grid_guidance_, grid_height_, grid_width_,
            num_images_, nullptr);
        return tv_loss_scalar_;
    }

    lfs::core::Tensor BilateralGrid::tv_loss_gpu(int image_idx) {
        if (image_idx < 0 || image_idx >= num_images_) {
            throw std::out_of_range("BilateralGrid::tv_loss_gpu: image_idx out of range");
        }
        LFS_CUDA_CHECK(cudaMemsetAsync(
            tv_loss_scalar_.ptr<float>(), 0, sizeof(float), nullptr));
        kernels::launch_bilateral_grid_tv_forward(
            slice_ptr(grids_, image_idx), tv_loss_scalar_.ptr<float>(), tv_temp_buffer_.ptr<float>(),
            1, channels_, grid_guidance_, grid_height_, grid_width_,
            num_images_, nullptr);
        return tv_loss_scalar_;
    }

    void BilateralGrid::tv_backward(float tv_weight) {
        for (int i = 0; i < num_images_; ++i) {
            LFS_CUDA_CHECK(cudaMemsetAsync(
                slice_grad_.ptr<float>(), 0, slice_elements() * sizeof(float), nullptr));
            tv_backward(tv_weight, i);
        }
    }

    void BilateralGrid::tv_backward(float tv_weight, int image_idx) {
        if (image_idx < 0 || image_idx >= num_images_) {
            throw std::out_of_range("BilateralGrid::tv_backward: image_idx out of range");
        }
        kernels::launch_bilateral_grid_tv_backward(
            slice_ptr(grids_, image_idx), tv_weight, slice_grad_.ptr<float>(),
            1, channels_, grid_guidance_, grid_height_, grid_width_,
            num_images_, nullptr);
    }

    void BilateralGrid::optimizer_step() {
        for (int i = 0; i < num_images_; ++i) {
            optimizer_step(i);
        }
    }

    void BilateralGrid::optimizer_step(int image_idx) {
        if (image_idx < 0 || image_idx >= num_images_) {
            throw std::out_of_range("BilateralGrid::optimizer_step: image_idx out of range");
        }
        const int64_t K = step_ - last_step_[static_cast<size_t>(image_idx)];
        if (K > 1) {
            const double skipped = static_cast<double>(K - 1);
            const float scale_avg = static_cast<float>(std::pow(config_.beta1, skipped));
            const float scale_avg_sq = static_cast<float>(std::pow(config_.beta2, skipped));
            kernels::launch_bilateral_grid_scale_moments(
                slice_ptr(exp_avg_, image_idx), slice_ptr(exp_avg_sq_, image_idx),
                static_cast<int>(slice_elements()), scale_avg, scale_avg_sq, nullptr);
        }
        float bc1_rcp, bc2_sqrt_rcp;
        compute_bias_corrections(bc1_rcp, bc2_sqrt_rcp);
        kernels::launch_bilateral_grid_adam_update(
            slice_ptr(grids_, image_idx), slice_ptr(exp_avg_, image_idx),
            slice_ptr(exp_avg_sq_, image_idx), slice_grad_.ptr<float>(),
            static_cast<int>(slice_elements()),
            static_cast<float>(current_lr_),
            static_cast<float>(config_.beta1), static_cast<float>(config_.beta2),
            bc1_rcp, bc2_sqrt_rcp, static_cast<float>(config_.eps), nullptr);
        last_step_[static_cast<size_t>(image_idx)] = step_;
    }

    void BilateralGrid::step_image(int image_idx, float tv_weight) {
        nvtxRangePush("bilateral_grid_tv_backward");
        tv_backward(tv_weight, image_idx);
        nvtxRangePop();

        const auto mean_old = channel_mean_of_image(image_idx);

        nvtxRangePush("bilateral_grid_adam");
        optimizer_step(image_idx);
        nvtxRangePop();

        nvtxRangePush("bilateral_grid_project_mean");
        if (parameterization_ == BilateralGridParameterization::ExposureChroma) {
            project_image(image_idx);
        } else {
            const auto mean_new = channel_mean_of_image(image_idx);
            const float spatial = static_cast<float>(
                grid_guidance_ * grid_height_ * grid_width_);
            const float inv_n_spatial = 1.0f / (static_cast<float>(num_images_) * spatial);
            kernels::launch_bilateral_grid_update_shared_offset(
                channel_sum_.ptr<float>(), shared_offset_.ptr<float>(),
                identity_mean_.ptr<float>(), mean_old.ptr<float>(), mean_new.ptr<float>(),
                channels_, spatial, inv_n_spatial, nullptr);
        }
        nvtxRangePop();

        zero_grad();
        scheduler_step();
    }

    void BilateralGrid::zero_grad() {
        LFS_CUDA_CHECK(cudaMemsetAsync(slice_grad_.ptr<float>(), 0,
                                       slice_grad_.numel() * sizeof(float), nullptr));
    }

    void BilateralGrid::scheduler_step() {
        ++step_;

        if (step_ <= config_.warmup_steps) {
            const double progress = static_cast<double>(step_) / config_.warmup_steps;
            const double scale = config_.warmup_start_factor + (1.0 - config_.warmup_start_factor) * progress;
            current_lr_ = initial_lr_ * scale;
        } else {
            const double gamma = std::pow(config_.final_lr_factor,
                                          1.0 / (total_iterations_ - config_.warmup_steps));
            current_lr_ = initial_lr_ * std::pow(gamma, step_ - config_.warmup_steps);
        }
    }

    void BilateralGrid::rebuild_identity_mean() {
        std::vector<float> identity(static_cast<size_t>(channels_), 0.0f);
        if (parameterization_ == BilateralGridParameterization::Affine) {
            identity[0] = 1.0f;
            identity[5] = 1.0f;
            identity[10] = 1.0f;
        }
        identity_mean_ = lfs::core::Tensor::from_vector(
            identity,
            {static_cast<size_t>(channels_)},
            lfs::core::Device::GPU);
    }

    void BilateralGrid::rebuild_projection_state() {
        const int dataset_axes[] = {0, 2, 3, 4};
        const auto mean = grids_.mean(std::span<const int>(dataset_axes), false);
        const float spatial = static_cast<float>(grid_guidance_ * grid_height_ * grid_width_);
        // Sum over every image's cells: N * L * H * W, not the per-image spatial count.
        const float n_spatial = spatial * static_cast<float>(num_images_);
        channel_sum_ = mean * n_spatial;
        shared_offset_ = identity_mean_.sub(mean);
    }

    size_t BilateralGrid::slice_elements() const {
        return static_cast<size_t>(channels_) *
               static_cast<size_t>(grid_guidance_) *
               static_cast<size_t>(grid_height_) *
               static_cast<size_t>(grid_width_);
    }

    float* BilateralGrid::slice_ptr(lfs::core::Tensor& tensor, int image_idx) {
        return tensor.ptr<float>() + static_cast<size_t>(image_idx) * slice_elements();
    }

    const float* BilateralGrid::slice_ptr(const lfs::core::Tensor& tensor, int image_idx) const {
        return tensor.ptr<float>() + static_cast<size_t>(image_idx) * slice_elements();
    }

    lfs::core::Tensor BilateralGrid::channel_mean_of_image(int image_idx) const {
        auto slice = grids_.slice(0, static_cast<size_t>(image_idx), static_cast<size_t>(image_idx) + 1)
                         .squeeze(0)
                         .flatten(1);
        return slice.mean(1);
    }

    void BilateralGrid::project_image(int image_idx) {
        if (image_idx < 0 || image_idx >= num_images_) {
            throw std::out_of_range("BilateralGrid::project_image: image_idx out of range");
        }
        const auto mean = channel_mean_of_image(image_idx);
        kernels::launch_bilateral_grid_project_mean(
            slice_ptr(grids_, image_idx), mean.ptr<float>(), identity_mean_.ptr<float>(),
            1, channels_, grid_guidance_, grid_height_, grid_width_, 1, nullptr);
        const float spatial = static_cast<float>(grid_guidance_ * grid_height_ * grid_width_);
        const float inv_n_spatial = 1.0f / (static_cast<float>(num_images_) * spatial);
        auto zeros = lfs::core::Tensor::zeros({static_cast<size_t>(channels_)}, lfs::core::Device::GPU);
        kernels::launch_bilateral_grid_update_shared_offset(
            channel_sum_.ptr<float>(), shared_offset_.ptr<float>(),
            identity_mean_.ptr<float>(), mean.ptr<float>(), zeros.ptr<float>(),
            channels_, spatial, inv_n_spatial, nullptr);
    }

    void BilateralGrid::project_mean(bool per_image) {
        assert(grids_.is_valid());
        assert(grids_.ndim() == 5);
        assert(static_cast<int>(grids_.shape()[0]) == num_images_);
        assert(static_cast<int>(grids_.shape()[1]) == channels_);
        assert(identity_mean_.is_valid());

        if (per_image) {
            for (int i = 0; i < num_images_; ++i) {
                project_image(i);
            }
            return;
        }
        rebuild_projection_state();
    }

    float BilateralGrid::max_abs_channel_mean_deviation() const {
        auto sum_cpu = channel_sum_.cpu();
        auto off_cpu = shared_offset_.cpu();
        auto id_cpu = identity_mean_.cpu();
        const float* sum = sum_cpu.ptr<float>();
        const float* off = off_cpu.ptr<float>();
        const float* id = id_cpu.ptr<float>();
        const float denom = static_cast<float>(num_images_) *
                            static_cast<float>(grid_guidance_ * grid_height_ * grid_width_);
        float max_dev = 0.0f;
        for (int c = 0; c < channels_; ++c) {
            const float mean = sum[c] / denom + off[c];
            max_dev = std::max(max_dev, std::abs(mean - id[c]));
        }
        return max_dev;
    }

    void BilateralGrid::log_eval_diagnostics() const {
        LOG_INFO("BilateralGrid drift: max_abs_channel_mean_deviation={:.6e} parameterization={}",
                 max_abs_channel_mean_deviation(),
                 bilateral_grid_parameterization_name(parameterization_));
    }

    void BilateralGrid::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_MAGIC), sizeof(CHECKPOINT_MAGIC));
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_VERSION), sizeof(CHECKPOINT_VERSION));

        os.write(reinterpret_cast<const char*>(&num_images_), sizeof(num_images_));
        os.write(reinterpret_cast<const char*>(&grid_width_), sizeof(grid_width_));
        os.write(reinterpret_cast<const char*>(&grid_height_), sizeof(grid_height_));
        os.write(reinterpret_cast<const char*>(&grid_guidance_), sizeof(grid_guidance_));

        serialize_config(os, config_, parameterization_);
        os.write(reinterpret_cast<const char*>(&step_), sizeof(step_));
        os.write(reinterpret_cast<const char*>(&current_lr_), sizeof(current_lr_));
        os.write(reinterpret_cast<const char*>(&initial_lr_), sizeof(initial_lr_));
        os.write(reinterpret_cast<const char*>(&total_iterations_), sizeof(total_iterations_));

        os << grids_ << exp_avg_ << exp_avg_sq_;
        assert(last_step_.size() == static_cast<size_t>(num_images_));
        const size_t last_step_bytes = last_step_.size() * sizeof(int64_t);
        os.write(reinterpret_cast<const char*>(last_step_.data()),
                 static_cast<std::streamsize>(last_step_bytes));
    }

    void BilateralGrid::deserialize(std::istream& is) {
        uint32_t magic = 0, version = 0;
        lfs::core::serialization_detail::read_exact(is, &magic, sizeof(magic), "bilateral-grid magic");
        lfs::core::serialization_detail::read_exact(is, &version, sizeof(version), "bilateral-grid version");

        if (magic != CHECKPOINT_MAGIC) {
            throw std::runtime_error("Invalid BilateralGrid checkpoint");
        }
        if (version < CHECKPOINT_MIN_VERSION || version > CHECKPOINT_VERSION) {
            config_serialization_detail::throw_unsupported_component_version(
                "BilateralGrid", version, CHECKPOINT_MIN_VERSION, CHECKPOINT_VERSION);
        }

        int num_images = 0;
        int grid_width = 0;
        int grid_height = 0;
        int grid_guidance = 0;
        Config config{};
        BilateralGridParameterization parameterization = BilateralGridParameterization::Affine;
        int64_t step = 0;
        double current_lr = 0.0;
        double initial_lr = 0.0;
        int total_iterations = 0;
        lfs::core::serialization_detail::read_exact(is, &num_images, sizeof(num_images), "bilateral-grid image count");
        lfs::core::serialization_detail::read_exact(is, &grid_width, sizeof(grid_width), "bilateral-grid width");
        lfs::core::serialization_detail::read_exact(is, &grid_height, sizeof(grid_height), "bilateral-grid height");
        lfs::core::serialization_detail::read_exact(is, &grid_guidance, sizeof(grid_guidance), "bilateral-grid guidance size");
        if (version == CHECKPOINT_MIN_VERSION) {
            config = deserialize_legacy_config(is);
        } else {
            const auto parsed = deserialize_config(is);
            config = parsed.config;
            parameterization = parsed.parameterization;
        }
        lfs::core::serialization_detail::read_exact(is, &step, sizeof(step), "bilateral-grid step");
        lfs::core::serialization_detail::read_exact(is, &current_lr, sizeof(current_lr), "bilateral-grid learning rate");
        lfs::core::serialization_detail::read_exact(is, &initial_lr, sizeof(initial_lr), "bilateral-grid initial learning rate");
        lfs::core::serialization_detail::read_exact(
            is, &total_iterations, sizeof(total_iterations), "bilateral-grid iteration count");

        if (num_images <= 0 || grid_width <= 0 || grid_height <= 0 || grid_guidance <= 0 ||
            num_images > 10'000'000 || grid_width > 4096 || grid_height > 4096 || grid_guidance > 4096 ||
            step < 0 || total_iterations <= 0 ||
            !std::isfinite(current_lr) || current_lr < 0.0 ||
            !std::isfinite(initial_lr) || initial_lr < 0.0 ||
            !std::isfinite(config.lr) || config.lr < 0.0 ||
            !std::isfinite(config.beta1) || config.beta1 < 0.0 || config.beta1 >= 1.0 ||
            !std::isfinite(config.beta2) || config.beta2 < 0.0 || config.beta2 >= 1.0 ||
            !std::isfinite(config.eps) || config.eps <= 0.0 || config.warmup_steps < 0 ||
            !std::isfinite(config.warmup_start_factor) || config.warmup_start_factor < 0.0 ||
            !std::isfinite(config.final_lr_factor) || config.final_lr_factor <= 0.0) {
            throw std::runtime_error("Invalid BilateralGrid checkpoint state");
        }

        const int channels = bilateral_grid_channel_count(parameterization);
        uint64_t grid_elements = static_cast<uint64_t>(num_images);
        for (const auto factor : {static_cast<size_t>(channels),
                                  static_cast<size_t>(grid_guidance),
                                  static_cast<size_t>(grid_height),
                                  static_cast<size_t>(grid_width)}) {
            if (grid_elements > std::numeric_limits<uint64_t>::max() / factor)
                throw std::runtime_error("Invalid BilateralGrid checkpoint: grid size overflows");
            grid_elements *= factor;
        }
        if (grid_elements > lfs::core::MAX_SERIALIZED_TENSOR_BYTES / sizeof(float) ||
            grid_elements > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("Invalid BilateralGrid checkpoint: grid exceeds CUDA kernel index budget");
        }

        lfs::core::Tensor grids, exp_avg, exp_avg_sq;
        is >> grids >> exp_avg >> exp_avg_sq;

        std::vector<int64_t> last_step(static_cast<size_t>(num_images), 0);
        if (version >= 3) {
            lfs::core::serialization_detail::read_exact(
                is, last_step.data(), last_step.size() * sizeof(int64_t),
                "bilateral-grid last_step");
            for (const int64_t last : last_step) {
                if (last < 0) {
                    throw std::runtime_error("Invalid BilateralGrid checkpoint last_step");
                }
            }
        }
        const lfs::core::TensorShape expected_shape{
            static_cast<size_t>(num_images), static_cast<size_t>(channels),
            static_cast<size_t>(grid_guidance), static_cast<size_t>(grid_height), static_cast<size_t>(grid_width)};
        if (grids.dtype() != lfs::core::DataType::Float32 || grids.shape() != expected_shape ||
            exp_avg.dtype() != lfs::core::DataType::Float32 || exp_avg.shape() != expected_shape ||
            exp_avg_sq.dtype() != lfs::core::DataType::Float32 || exp_avg_sq.shape() != expected_shape) {
            throw std::runtime_error("Invalid BilateralGrid checkpoint tensor schema");
        }

        grids = grids.gpu();
        exp_avg = exp_avg.gpu();
        exp_avg_sq = exp_avg_sq.gpu();

        const size_t spatial = static_cast<size_t>(grid_guidance) * static_cast<size_t>(grid_height) *
                               static_cast<size_t>(grid_width);
        const size_t temp_size = std::max(size_t(2048), (spatial + 255) / 256);
        auto tv_temp_buffer = lfs::core::Tensor::empty({temp_size}, lfs::core::Device::GPU);
        auto tv_loss_scalar = lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
        auto slice_grad = lfs::core::Tensor::zeros(
            {static_cast<size_t>(channels), static_cast<size_t>(grid_guidance),
             static_cast<size_t>(grid_height), static_cast<size_t>(grid_width)},
            lfs::core::Device::GPU);

        num_images_ = num_images;
        grid_width_ = grid_width;
        grid_height_ = grid_height;
        grid_guidance_ = grid_guidance;
        channels_ = channels;
        parameterization_ = parameterization;
        config_ = config;
        step_ = step;
        current_lr_ = current_lr;
        initial_lr_ = initial_lr;
        total_iterations_ = total_iterations;
        grids_ = std::move(grids);
        exp_avg_ = std::move(exp_avg);
        exp_avg_sq_ = std::move(exp_avg_sq);
        slice_grad_ = std::move(slice_grad);
        tv_temp_buffer_ = std::move(tv_temp_buffer);
        tv_loss_scalar_ = std::move(tv_loss_scalar);
        last_step_ = std::move(last_step);
        rebuild_identity_mean();
        rebuild_projection_state();
    }

    void BilateralGrid::adopt_checkpoint_state(BilateralGrid& loaded) {
        if (parameterization_ != loaded.parameterization_) {
            throw std::runtime_error(
                std::string("BilateralGrid parameterization mismatch: checkpoint is ") +
                bilateral_grid_parameterization_name(loaded.parameterization_) +
                ", current is " +
                bilateral_grid_parameterization_name(parameterization_));
        }
        std::swap(grids_, loaded.grids_);
        std::swap(exp_avg_, loaded.exp_avg_);
        std::swap(exp_avg_sq_, loaded.exp_avg_sq_);
        std::swap(slice_grad_, loaded.slice_grad_);
        std::swap(tv_temp_buffer_, loaded.tv_temp_buffer_);
        std::swap(tv_loss_scalar_, loaded.tv_loss_scalar_);
        std::swap(identity_mean_, loaded.identity_mean_);
        std::swap(shared_offset_, loaded.shared_offset_);
        std::swap(channel_sum_, loaded.channel_sum_);
        std::swap(config_, loaded.config_);
        std::swap(step_, loaded.step_);
        std::swap(last_step_, loaded.last_step_);
        std::swap(current_lr_, loaded.current_lr_);
        std::swap(initial_lr_, loaded.initial_lr_);
        std::swap(total_iterations_, loaded.total_iterations_);
        std::swap(num_images_, loaded.num_images_);
        std::swap(grid_width_, loaded.grid_width_);
        std::swap(grid_height_, loaded.grid_height_);
        std::swap(grid_guidance_, loaded.grid_guidance_);
        std::swap(channels_, loaded.channels_);
        std::swap(parameterization_, loaded.parameterization_);
    }

} // namespace lfs::training
