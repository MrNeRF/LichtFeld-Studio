/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "bilateral_grid.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include <cmath>
#include <stdexcept>

namespace lfs::training {

    namespace {
        constexpr uint32_t CHECKPOINT_MAGIC = 0x4C464247; // "LFBG"
        constexpr uint32_t CHECKPOINT_VERSION = 1;
        constexpr size_t GRID_CHANNELS = 12;
    } // namespace

    BilateralGrid::BilateralGrid(int num_images, int grid_W, int grid_H, int grid_L,
                                 int total_iterations, Config config)
        : config_(config),
          current_lr_(config.lr),
          initial_lr_(config.lr),
          total_iterations_(total_iterations),
          num_images_(num_images),
          grid_width_(grid_W),
          grid_height_(grid_H),
          grid_guidance_(grid_L) {

        // All allocations and initialization on GPU - no CPU allocation
        grids_ = lfs::core::Tensor::empty(
            {static_cast<size_t>(num_images), GRID_CHANNELS,
             static_cast<size_t>(grid_L), static_cast<size_t>(grid_H), static_cast<size_t>(grid_W)},
            lfs::core::Device::CUDA);

        // Initialize identity transform directly on GPU
        kernels::launch_bilateral_grid_init_identity(
            grids_.ptr<float>(), num_images, grid_L, grid_H, grid_W, lfs::core::getCurrentCUDAStream());

        exp_avg_ = lfs::core::Tensor::zeros(grids_.shape(), lfs::core::Device::CUDA);
        exp_avg_sq_ = lfs::core::Tensor::zeros(grids_.shape(), lfs::core::Device::CUDA);
        accumulated_grads_ = lfs::core::Tensor::zeros(grids_.shape(), lfs::core::Device::CUDA);

        const size_t total_elements = num_images * grid_L * grid_H * grid_W;
        const size_t temp_size = std::max(size_t(2048), (total_elements + 255) / 256);
        tv_temp_buffer_ = lfs::core::Tensor::empty({temp_size}, lfs::core::Device::CUDA);

        const size_t grid_slice_size = GRID_CHANNELS * grid_L * grid_H * grid_W;
        grad_buffer_ = lfs::core::Tensor::empty({grid_slice_size}, lfs::core::Device::CUDA);

        LOG_DEBUG("BilateralGrid: {}x{}x{} for {} images, lr={:.2e}",
                  grid_W, grid_H, grid_L, num_images, config.lr);
    }

    lfs::core::Tensor BilateralGrid::apply(const lfs::core::Tensor& rgb, int image_idx) {
        if (image_idx < 0 || image_idx >= num_images_) {
            throw std::out_of_range("BilateralGrid::apply: image_idx out of range");
        }

        const auto& shape = rgb.shape();
        const bool is_chw = (shape.rank() == 3 && shape[0] == 3);
        const size_t grid_slice_size = GRID_CHANNELS * grid_guidance_ * grid_height_ * grid_width_;
        const float* grid_ptr = grids_.ptr<float>() + (image_idx * grid_slice_size);

        if (is_chw) {
            const int h = static_cast<int>(shape[1]);
            const int w = static_cast<int>(shape[2]);
            auto output = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::CUDA);
            kernels::launch_bilateral_grid_slice_forward_chw(
                grid_ptr, rgb.ptr<float>(), output.ptr<float>(),
                grid_guidance_, grid_height_, grid_width_, h, w, lfs::core::getCurrentCUDAStream());
            return output;
        }

        const int h = static_cast<int>(shape[0]);
        const int w = static_cast<int>(shape[1]);
        const auto rgb_cont = rgb.contiguous();
        auto output = lfs::core::Tensor::empty({shape[0], shape[1], 3}, lfs::core::Device::CUDA);
        kernels::launch_bilateral_grid_slice_forward(
            grid_ptr, rgb_cont.ptr<float>(), output.ptr<float>(),
            grid_guidance_, grid_height_, grid_width_, h, w, lfs::core::getCurrentCUDAStream());
        return output;
    }

    lfs::core::Tensor BilateralGrid::backward(const lfs::core::Tensor& rgb,
                                              const lfs::core::Tensor& grad_output,
                                              int image_idx) {
        if (image_idx < 0 || image_idx >= num_images_) {
            throw std::out_of_range("BilateralGrid::backward: image_idx out of range");
        }

        const auto& shape = rgb.shape();
        const bool is_chw = (shape.rank() == 3 && shape[0] == 3);
        const size_t grid_slice_size = GRID_CHANNELS * grid_guidance_ * grid_height_ * grid_width_;
        const float* grid_ptr = grids_.ptr<float>() + (image_idx * grid_slice_size);
        float* grad_grid_ptr = accumulated_grads_.ptr<float>() + (image_idx * grid_slice_size);

        if (is_chw) {
            const int h = static_cast<int>(shape[1]);
            const int w = static_cast<int>(shape[2]);
            auto grad_rgb = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::CUDA);

            cudaMemsetAsync(grad_buffer_.ptr<float>(), 0, grid_slice_size * sizeof(float), lfs::core::getCurrentCUDAStream());
            kernels::launch_bilateral_grid_slice_backward_chw(
                grid_ptr, rgb.ptr<float>(), grad_output.ptr<float>(),
                grad_buffer_.ptr<float>(), grad_rgb.ptr<float>(),
                grid_guidance_, grid_height_, grid_width_, h, w, lfs::core::getCurrentCUDAStream());
            kernels::launch_bilateral_grid_accumulate_grad(
                grad_grid_ptr, grad_buffer_.ptr<float>(),
                static_cast<int>(grid_slice_size), lfs::core::getCurrentCUDAStream());
            return grad_rgb;
        }

        const int h = static_cast<int>(shape[0]);
        const int w = static_cast<int>(shape[1]);
        const auto rgb_cont = rgb.contiguous();
        const auto grad_cont = grad_output.contiguous();
        auto grad_rgb = lfs::core::Tensor::empty({shape[0], shape[1], 3}, lfs::core::Device::CUDA);

        kernels::launch_bilateral_grid_slice_backward(
            grid_ptr, rgb_cont.ptr<float>(), grad_cont.ptr<float>(),
            grad_grid_ptr, grad_rgb.ptr<float>(),
            grid_guidance_, grid_height_, grid_width_, h, w, lfs::core::getCurrentCUDAStream());
        return grad_rgb;
    }

    lfs::core::Tensor BilateralGrid::tv_loss_gpu() {
        auto tv_device = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
        kernels::launch_bilateral_grid_tv_forward(
            grids_.ptr<float>(), tv_device.ptr<float>(), tv_temp_buffer_.ptr<float>(),
            num_images_, grid_guidance_, grid_height_, grid_width_, lfs::core::getCurrentCUDAStream());
        return tv_device;
    }

    void BilateralGrid::tv_backward(float tv_weight) {
        kernels::launch_bilateral_grid_tv_backward(
            grids_.ptr<float>(), tv_weight, accumulated_grads_.ptr<float>(),
            num_images_, grid_guidance_, grid_height_, grid_width_, lfs::core::getCurrentCUDAStream());
    }

    void BilateralGrid::optimizer_step() {
        float bc1_rcp, bc2_sqrt_rcp;
        compute_bias_corrections(bc1_rcp, bc2_sqrt_rcp);

        kernels::launch_bilateral_grid_adam_update(
            grids_.ptr<float>(), exp_avg_.ptr<float>(), exp_avg_sq_.ptr<float>(),
            accumulated_grads_.ptr<float>(), static_cast<int>(grids_.numel()),
            static_cast<float>(current_lr_),
            static_cast<float>(config_.beta1), static_cast<float>(config_.beta2),
            bc1_rcp, bc2_sqrt_rcp, static_cast<float>(config_.eps), lfs::core::getCurrentCUDAStream());
    }

    void BilateralGrid::zero_grad() {
        cudaMemsetAsync(accumulated_grads_.ptr<float>(), 0,
                        accumulated_grads_.numel() * sizeof(float), lfs::core::getCurrentCUDAStream());
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

        os.write(reinterpret_cast<const char*>(&config_), sizeof(config_));
        os.write(reinterpret_cast<const char*>(&step_), sizeof(step_));
        os.write(reinterpret_cast<const char*>(&current_lr_), sizeof(current_lr_));
        os.write(reinterpret_cast<const char*>(&initial_lr_), sizeof(initial_lr_));
        os.write(reinterpret_cast<const char*>(&total_iterations_), sizeof(total_iterations_));

        os << grids_ << exp_avg_ << exp_avg_sq_;
    }

    void BilateralGrid::deserialize(std::istream& is) {
        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != CHECKPOINT_MAGIC) {
            throw std::runtime_error("Invalid BilateralGrid checkpoint");
        }
        if (version != CHECKPOINT_VERSION) {
            throw std::runtime_error("Unsupported BilateralGrid checkpoint version");
        }

        is.read(reinterpret_cast<char*>(&num_images_), sizeof(num_images_));
        is.read(reinterpret_cast<char*>(&grid_width_), sizeof(grid_width_));
        is.read(reinterpret_cast<char*>(&grid_height_), sizeof(grid_height_));
        is.read(reinterpret_cast<char*>(&grid_guidance_), sizeof(grid_guidance_));

        is.read(reinterpret_cast<char*>(&config_), sizeof(config_));
        is.read(reinterpret_cast<char*>(&step_), sizeof(step_));
        is.read(reinterpret_cast<char*>(&current_lr_), sizeof(current_lr_));
        is.read(reinterpret_cast<char*>(&initial_lr_), sizeof(initial_lr_));
        is.read(reinterpret_cast<char*>(&total_iterations_), sizeof(total_iterations_));

        is >> grids_ >> exp_avg_ >> exp_avg_sq_;
        grids_ = grids_.cuda();
        exp_avg_ = exp_avg_.cuda();
        exp_avg_sq_ = exp_avg_sq_.cuda();

        accumulated_grads_ = lfs::core::Tensor::zeros(grids_.shape(), lfs::core::Device::CUDA);

        const size_t total_elements = num_images_ * grid_guidance_ * grid_height_ * grid_width_;
        const size_t temp_size = std::max(size_t(2048), (total_elements + 255) / 256);
        tv_temp_buffer_ = lfs::core::Tensor::empty({temp_size}, lfs::core::Device::CUDA);

        const size_t grid_slice_size = GRID_CHANNELS * grid_guidance_ * grid_height_ * grid_width_;
        grad_buffer_ = lfs::core::Tensor::empty({grid_slice_size}, lfs::core::Device::CUDA);
    }

} // namespace lfs::training
