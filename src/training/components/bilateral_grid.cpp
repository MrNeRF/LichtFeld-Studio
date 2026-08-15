/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "bilateral_grid.hpp"
#include "config_serialization.hpp"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace lfs::training {

    namespace {
        constexpr uint32_t CHECKPOINT_MAGIC = 0x4C464247; // "LFBG"
        constexpr uint32_t CHECKPOINT_MIN_VERSION = 1;
        constexpr uint32_t CHECKPOINT_VERSION = 2;
        constexpr uint32_t CONFIG_SCHEMA_VERSION = 1;
        constexpr uint32_t CONFIG_SCHEMA_V1_BYTES = 52;
        constexpr size_t GRID_CHANNELS = 12;

        struct LegacyConfigV1 {
            double lr;
            double beta1;
            double beta2;
            double eps;
            int warmup_steps;
            double warmup_start_factor;
            double final_lr_factor;
        };

        void serialize_config(std::ostream& os, const BilateralGrid::Config& config) {
            using config_serialization_detail::write_little_endian;

            // Schema v1 is append-only. Bump the schema version and append fields;
            // payload size lets older readers skip the suffix and retain defaults.
            write_little_endian(os, CONFIG_SCHEMA_VERSION, "bilateral-grid config schema");
            write_little_endian(os, CONFIG_SCHEMA_V1_BYTES, "bilateral-grid config size");
            write_little_endian(os, config.lr, "bilateral-grid config lr");
            write_little_endian(os, config.beta1, "bilateral-grid config beta1");
            write_little_endian(os, config.beta2, "bilateral-grid config beta2");
            write_little_endian(os, config.eps, "bilateral-grid config eps");
            write_little_endian(
                os, static_cast<int32_t>(config.warmup_steps), "bilateral-grid config warmup_steps");
            write_little_endian(
                os, config.warmup_start_factor, "bilateral-grid config warmup_start_factor");
            write_little_endian(os, config.final_lr_factor, "bilateral-grid config final_lr_factor");
        }

        [[nodiscard]] BilateralGrid::Config deserialize_config(std::istream& is) {
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

            BilateralGrid::Config config{};
            config.lr = read_little_endian<double>(is, "bilateral-grid config lr");
            config.beta1 = read_little_endian<double>(is, "bilateral-grid config beta1");
            config.beta2 = read_little_endian<double>(is, "bilateral-grid config beta2");
            config.eps = read_little_endian<double>(is, "bilateral-grid config eps");
            config.warmup_steps =
                read_little_endian<int32_t>(is, "bilateral-grid config warmup_steps");
            config.warmup_start_factor =
                read_little_endian<double>(is, "bilateral-grid config warmup_start_factor");
            config.final_lr_factor =
                read_little_endian<double>(is, "bilateral-grid config final_lr_factor");
            config_serialization_detail::skip_bytes(
                is, payload_bytes - CONFIG_SCHEMA_V1_BYTES, "bilateral-grid config");
            return config;
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
            const int total_iterations,
            const BilateralGrid::Config& config) {
            if (num_images <= 0 || grid_width <= 0 || grid_height <= 0 || grid_guidance <= 0)
                throw std::invalid_argument("BilateralGrid dimensions and image count must be positive");
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
                     static_cast<uint64_t>(GRID_CHANNELS),
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
            if (tensor.device() != lfs::core::Device::CUDA || tensor.dtype() != lfs::core::DataType::Float32)
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
                                 int total_iterations, Config config)
        : config_(config),
          current_lr_(config.warmup_steps > 0 ? config.lr * config.warmup_start_factor : config.lr),
          initial_lr_(config.lr),
          total_iterations_(total_iterations),
          num_images_(num_images),
          grid_width_(grid_W),
          grid_height_(grid_H),
          grid_guidance_(grid_L) {

        const size_t grid_elements = validated_grid_elements(
            num_images, grid_W, grid_H, grid_L, total_iterations, config);

        // All allocations and initialization on GPU - no CPU allocation
        grids_ = lfs::core::Tensor::empty(
            {static_cast<size_t>(num_images), GRID_CHANNELS,
             static_cast<size_t>(grid_L), static_cast<size_t>(grid_H), static_cast<size_t>(grid_W)},
            lfs::core::Device::CUDA);

        // Initialize identity transform directly on GPU
        kernels::launch_bilateral_grid_init_identity(
            grids_.ptr<float>(), num_images, grid_L, grid_H, grid_W, nullptr);

        exp_avg_ = lfs::core::Tensor::zeros(grids_.shape(), lfs::core::Device::CUDA);
        exp_avg_sq_ = lfs::core::Tensor::zeros(grids_.shape(), lfs::core::Device::CUDA);
        accumulated_grads_ = lfs::core::Tensor::zeros(grids_.shape(), lfs::core::Device::CUDA);

        const size_t total_elements = grid_elements / GRID_CHANNELS;
        const size_t temp_size = std::max(size_t(2048), (total_elements + 255) / 256);
        tv_temp_buffer_ = lfs::core::Tensor::empty({temp_size}, lfs::core::Device::CUDA);

        const size_t grid_slice_size = grid_elements / static_cast<size_t>(num_images);
        grad_buffer_ = lfs::core::Tensor::empty({grid_slice_size}, lfs::core::Device::CUDA);

        LOG_DEBUG("BilateralGrid: {}x{}x{} for {} images, lr={:.2e}",
                  grid_W, grid_H, grid_L, num_images, config.lr);
    }

    lfs::core::Tensor BilateralGrid::apply(const lfs::core::Tensor& rgb, int image_idx) {
        if (image_idx < 0 || image_idx >= num_images_) {
            throw std::out_of_range("BilateralGrid::apply: image_idx out of range");
        }

        const ImageLayout layout = validate_image_tensor(rgb, "BilateralGrid::apply");
        const auto& shape = rgb.shape();
        const auto rgb_cont = rgb.contiguous();
        const size_t grid_slice_size = grids_.numel() / static_cast<size_t>(num_images_);
        const float* grid_ptr = grids_.ptr<float>() + static_cast<size_t>(image_idx) * grid_slice_size;

        if (layout.chw) {
            auto output = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::CUDA);
            kernels::launch_bilateral_grid_slice_forward_chw(
                grid_ptr, rgb_cont.ptr<float>(), output.ptr<float>(),
                grid_guidance_, grid_height_, grid_width_, layout.height, layout.width, nullptr);
            return output;
        }

        auto output = lfs::core::Tensor::empty({shape[0], shape[1], 3}, lfs::core::Device::CUDA);
        kernels::launch_bilateral_grid_slice_forward(
            grid_ptr, rgb_cont.ptr<float>(), output.ptr<float>(),
            grid_guidance_, grid_height_, grid_width_, layout.height, layout.width, nullptr);
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
        const size_t grid_slice_size = grids_.numel() / static_cast<size_t>(num_images_);
        const size_t grid_offset = static_cast<size_t>(image_idx) * grid_slice_size;
        const float* grid_ptr = grids_.ptr<float>() + grid_offset;
        float* grad_grid_ptr = accumulated_grads_.ptr<float>() + grid_offset;

        if (layout.chw) {
            auto grad_rgb = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::CUDA);

            LFS_CUDA_CHECK(cudaMemsetAsync(
                grad_buffer_.ptr<float>(), 0, grid_slice_size * sizeof(float), nullptr));
            kernels::launch_bilateral_grid_slice_backward_chw(
                grid_ptr, rgb_cont.ptr<float>(), grad_cont.ptr<float>(),
                grad_buffer_.ptr<float>(), grad_rgb.ptr<float>(),
                grid_guidance_, grid_height_, grid_width_, layout.height, layout.width, nullptr);
            kernels::launch_bilateral_grid_accumulate_grad(
                grad_grid_ptr, grad_buffer_.ptr<float>(),
                static_cast<int>(grid_slice_size), nullptr);
            return grad_rgb;
        }

        auto grad_rgb = lfs::core::Tensor::empty({shape[0], shape[1], 3}, lfs::core::Device::CUDA);

        kernels::launch_bilateral_grid_slice_backward(
            grid_ptr, rgb_cont.ptr<float>(), grad_cont.ptr<float>(),
            grad_grid_ptr, grad_rgb.ptr<float>(),
            grid_guidance_, grid_height_, grid_width_, layout.height, layout.width, nullptr);
        return grad_rgb;
    }

    lfs::core::Tensor BilateralGrid::tv_loss_gpu() {
        auto tv_device = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
        kernels::launch_bilateral_grid_tv_forward(
            grids_.ptr<float>(), tv_device.ptr<float>(), tv_temp_buffer_.ptr<float>(),
            num_images_, grid_guidance_, grid_height_, grid_width_, nullptr);
        return tv_device;
    }

    void BilateralGrid::tv_backward(float tv_weight) {
        kernels::launch_bilateral_grid_tv_backward(
            grids_.ptr<float>(), tv_weight, accumulated_grads_.ptr<float>(),
            num_images_, grid_guidance_, grid_height_, grid_width_, nullptr);
    }

    void BilateralGrid::optimizer_step() {
        float bc1_rcp, bc2_sqrt_rcp;
        compute_bias_corrections(bc1_rcp, bc2_sqrt_rcp);

        kernels::launch_bilateral_grid_adam_update(
            grids_.ptr<float>(), exp_avg_.ptr<float>(), exp_avg_sq_.ptr<float>(),
            accumulated_grads_.ptr<float>(), static_cast<int>(grids_.numel()),
            static_cast<float>(current_lr_),
            static_cast<float>(config_.beta1), static_cast<float>(config_.beta2),
            bc1_rcp, bc2_sqrt_rcp, static_cast<float>(config_.eps), nullptr);
    }

    void BilateralGrid::zero_grad() {
        LFS_CUDA_CHECK(cudaMemsetAsync(accumulated_grads_.ptr<float>(), 0,
                                       accumulated_grads_.numel() * sizeof(float), nullptr));
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

    void BilateralGrid::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_MAGIC), sizeof(CHECKPOINT_MAGIC));
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_VERSION), sizeof(CHECKPOINT_VERSION));

        os.write(reinterpret_cast<const char*>(&num_images_), sizeof(num_images_));
        os.write(reinterpret_cast<const char*>(&grid_width_), sizeof(grid_width_));
        os.write(reinterpret_cast<const char*>(&grid_height_), sizeof(grid_height_));
        os.write(reinterpret_cast<const char*>(&grid_guidance_), sizeof(grid_guidance_));

        serialize_config(os, config_);
        os.write(reinterpret_cast<const char*>(&step_), sizeof(step_));
        os.write(reinterpret_cast<const char*>(&current_lr_), sizeof(current_lr_));
        os.write(reinterpret_cast<const char*>(&initial_lr_), sizeof(initial_lr_));
        os.write(reinterpret_cast<const char*>(&total_iterations_), sizeof(total_iterations_));

        os << grids_ << exp_avg_ << exp_avg_sq_;
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
        int64_t step = 0;
        double current_lr = 0.0;
        double initial_lr = 0.0;
        int total_iterations = 0;
        lfs::core::serialization_detail::read_exact(is, &num_images, sizeof(num_images), "bilateral-grid image count");
        lfs::core::serialization_detail::read_exact(is, &grid_width, sizeof(grid_width), "bilateral-grid width");
        lfs::core::serialization_detail::read_exact(is, &grid_height, sizeof(grid_height), "bilateral-grid height");
        lfs::core::serialization_detail::read_exact(is, &grid_guidance, sizeof(grid_guidance), "bilateral-grid guidance size");
        config = version == CHECKPOINT_MIN_VERSION
                     ? deserialize_legacy_config(is)
                     : deserialize_config(is);
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

        uint64_t grid_elements = static_cast<uint64_t>(num_images);
        for (const auto factor : {GRID_CHANNELS,
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
        const lfs::core::TensorShape expected_shape{
            static_cast<size_t>(num_images), GRID_CHANNELS,
            static_cast<size_t>(grid_guidance), static_cast<size_t>(grid_height), static_cast<size_t>(grid_width)};
        if (grids.dtype() != lfs::core::DataType::Float32 || grids.shape() != expected_shape ||
            exp_avg.dtype() != lfs::core::DataType::Float32 || exp_avg.shape() != expected_shape ||
            exp_avg_sq.dtype() != lfs::core::DataType::Float32 || exp_avg_sq.shape() != expected_shape) {
            throw std::runtime_error("Invalid BilateralGrid checkpoint tensor schema");
        }

        grids = grids.cuda();
        exp_avg = exp_avg.cuda();
        exp_avg_sq = exp_avg_sq.cuda();
        auto accumulated_grads = lfs::core::Tensor::zeros(expected_shape, lfs::core::Device::CUDA);

        const size_t total_elements = static_cast<size_t>(grid_elements / GRID_CHANNELS);
        const size_t temp_size = std::max(size_t(2048), (total_elements + 255) / 256);
        auto tv_temp_buffer = lfs::core::Tensor::empty({temp_size}, lfs::core::Device::CUDA);

        const size_t grid_slice_size = static_cast<size_t>(grid_elements / static_cast<uint64_t>(num_images));
        auto grad_buffer = lfs::core::Tensor::empty({grid_slice_size}, lfs::core::Device::CUDA);

        num_images_ = num_images;
        grid_width_ = grid_width;
        grid_height_ = grid_height;
        grid_guidance_ = grid_guidance;
        config_ = config;
        step_ = step;
        current_lr_ = current_lr;
        initial_lr_ = initial_lr;
        total_iterations_ = total_iterations;
        grids_ = std::move(grids);
        exp_avg_ = std::move(exp_avg);
        exp_avg_sq_ = std::move(exp_avg_sq);
        accumulated_grads_ = std::move(accumulated_grads);
        tv_temp_buffer_ = std::move(tv_temp_buffer);
        grad_buffer_ = std::move(grad_buffer);
    }

    void BilateralGrid::adopt_checkpoint_state(BilateralGrid& loaded) noexcept {
        std::swap(grids_, loaded.grids_);
        std::swap(exp_avg_, loaded.exp_avg_);
        std::swap(exp_avg_sq_, loaded.exp_avg_sq_);
        std::swap(accumulated_grads_, loaded.accumulated_grads_);
        std::swap(grad_buffer_, loaded.grad_buffer_);
        std::swap(tv_temp_buffer_, loaded.tv_temp_buffer_);
        std::swap(config_, loaded.config_);
        std::swap(step_, loaded.step_);
        std::swap(current_lr_, loaded.current_lr_);
        std::swap(initial_lr_, loaded.initial_lr_);
        std::swap(total_iterations_, loaded.total_iterations_);
        std::swap(num_images_, loaded.num_images_);
        std::swap(grid_width_, loaded.grid_width_);
        std::swap(grid_height_, loaded.grid_height_);
        std::swap(grid_guidance_, loaded.grid_guidance_);
    }

} // namespace lfs::training
