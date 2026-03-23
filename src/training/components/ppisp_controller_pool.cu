/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core/tensor/internal/tensor_ops.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "lfs/kernels/ppisp.cuh"
#include "ppisp_controller_pool.hpp"
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

namespace lfs::training {

    namespace {
        constexpr uint32_t CHECKPOINT_MAGIC = 0x4C465043;
        constexpr uint32_t CHECKPOINT_VERSION = 1;
        constexpr uint32_t INFERENCE_MAGIC = 0x4C464349;
        constexpr uint32_t INFERENCE_VERSION = 1;

        constexpr int BLOCK_SIZE = 256;
        constexpr int TILE_SIZE = 16;
        constexpr int CNN_CH1 = 16;
        constexpr int CNN_CH2 = 32;
        constexpr int CNN_CH3 = 64;
        constexpr int FC1_INPUT_DIM = 1601;
        constexpr int FC_HIDDEN_DIM = 128;
        constexpr int FC_OUTPUT_DIM = 9;
        constexpr int CNN_FLAT_DIM = 1600;
        constexpr int POOL2_SIZE = 5;
        constexpr int POOL_STRIDE = 3;

        lfs::core::Tensor kaiming_uniform(const size_t fan_in, const size_t fan_out) {
            const float bound = std::sqrt(6.0f / static_cast<float>(fan_in));
            return lfs::core::Tensor::uniform({fan_out, fan_in}, -bound, bound, lfs::core::Device::CUDA);
        }

        lfs::core::Tensor zeros_bias(const size_t size) {
            return lfs::core::Tensor::zeros({size}, lfs::core::Device::CUDA);
        }

        __global__ void relu_backward_kernel(const float* grad, const float* input, float* out, const int n) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                out[idx] = (input[idx] > 0.0f) ? grad[idx] : 0.0f;
            }
        }

        void launch_relu_backward(const float* grad, const float* input, float* out, const int n,
                                  cudaStream_t stream) {
            const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            relu_backward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(grad, input, out, n);
        }

        __global__ void outer_product_accumulate_kernel(const float* a, const float* b, float* c,
                                                        const int m, const int n, const float scale) {
            const int i = blockIdx.y * blockDim.y + threadIdx.y;
            const int j = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < m && j < n) {
                c[i * n + j] += scale * a[i] * b[j];
            }
        }

        void launch_outer_product_accumulate(const float* a, const float* b, float* c, const int m, const int n,
                                             const float scale, cudaStream_t stream) {
            const dim3 block(TILE_SIZE, TILE_SIZE);
            const dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
            outer_product_accumulate_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, scale);
        }

        __global__ void bias_grad_accumulate_kernel(const float* grad, float* bias_grad, const int n) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                bias_grad[idx] += grad[idx];
            }
        }

        void launch_bias_grad_accumulate(const float* grad, float* bias_grad, const int n, cudaStream_t stream) {
            const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            bias_grad_accumulate_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(grad, bias_grad, n);
        }

    } // namespace

    PPISPControllerPool::PPISPControllerPool(const int num_cameras, const int total_iterations, Config config)
        : num_cameras_(num_cameras),
          total_iterations_(total_iterations),
          config_(config),
          current_lr_(config.lr),
          initial_lr_(config.lr) {

        conv1_w_ = kaiming_uniform(3, CNN_CH1);
        conv1_b_ = zeros_bias(CNN_CH1);
        conv2_w_ = kaiming_uniform(CNN_CH1, CNN_CH2);
        conv2_b_ = zeros_bias(CNN_CH2);
        conv3_w_ = kaiming_uniform(CNN_CH2, CNN_CH3);
        conv3_b_ = zeros_bias(CNN_CH3);

        fc1_w_.reserve(num_cameras);
        fc1_b_.reserve(num_cameras);
        fc2_w_.reserve(num_cameras);
        fc2_b_.reserve(num_cameras);
        fc3_w_.reserve(num_cameras);
        fc3_b_.reserve(num_cameras);
        fc4_w_.reserve(num_cameras);
        fc4_b_.reserve(num_cameras);

        for (int i = 0; i < num_cameras; ++i) {
            fc1_w_.push_back(kaiming_uniform(FC1_INPUT_DIM, FC_HIDDEN_DIM));
            fc1_b_.push_back(zeros_bias(FC_HIDDEN_DIM));
            fc2_w_.push_back(kaiming_uniform(FC_HIDDEN_DIM, FC_HIDDEN_DIM));
            fc2_b_.push_back(zeros_bias(FC_HIDDEN_DIM));
            fc3_w_.push_back(kaiming_uniform(FC_HIDDEN_DIM, FC_HIDDEN_DIM));
            fc3_b_.push_back(zeros_bias(FC_HIDDEN_DIM));
            fc4_w_.push_back(kaiming_uniform(FC_HIDDEN_DIM, FC_OUTPUT_DIM));
            fc4_b_.push_back(zeros_bias(FC_OUTPUT_DIM));
        }

        fc1_w_grad_ = lfs::core::Tensor::zeros({FC_HIDDEN_DIM, FC1_INPUT_DIM}, lfs::core::Device::CUDA);
        fc1_b_grad_ = zeros_bias(FC_HIDDEN_DIM);
        fc2_w_grad_ = lfs::core::Tensor::zeros({FC_HIDDEN_DIM, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        fc2_b_grad_ = zeros_bias(FC_HIDDEN_DIM);
        fc3_w_grad_ = lfs::core::Tensor::zeros({FC_HIDDEN_DIM, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        fc3_b_grad_ = zeros_bias(FC_HIDDEN_DIM);
        fc4_w_grad_ = lfs::core::Tensor::zeros({FC_OUTPUT_DIM, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        fc4_b_grad_ = zeros_bias(FC_OUTPUT_DIM);

        fc1_w_m_ = lfs::core::Tensor::zeros({FC_HIDDEN_DIM, FC1_INPUT_DIM}, lfs::core::Device::CUDA);
        fc1_w_v_ = lfs::core::Tensor::zeros({FC_HIDDEN_DIM, FC1_INPUT_DIM}, lfs::core::Device::CUDA);
        fc1_b_m_ = zeros_bias(FC_HIDDEN_DIM);
        fc1_b_v_ = zeros_bias(FC_HIDDEN_DIM);
        fc2_w_m_ = lfs::core::Tensor::zeros({FC_HIDDEN_DIM, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        fc2_w_v_ = lfs::core::Tensor::zeros({FC_HIDDEN_DIM, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        fc2_b_m_ = zeros_bias(FC_HIDDEN_DIM);
        fc2_b_v_ = zeros_bias(FC_HIDDEN_DIM);
        fc3_w_m_ = lfs::core::Tensor::zeros({FC_HIDDEN_DIM, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        fc3_w_v_ = lfs::core::Tensor::zeros({FC_HIDDEN_DIM, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        fc3_b_m_ = zeros_bias(FC_HIDDEN_DIM);
        fc3_b_v_ = zeros_bias(FC_HIDDEN_DIM);
        fc4_w_m_ = lfs::core::Tensor::zeros({FC_OUTPUT_DIM, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        fc4_w_v_ = lfs::core::Tensor::zeros({FC_OUTPUT_DIM, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        fc4_b_m_ = zeros_bias(FC_OUTPUT_DIM);
        fc4_b_v_ = zeros_bias(FC_OUTPUT_DIM);

        buf_fc1_ = lfs::core::Tensor::empty({1, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        buf_fc2_ = lfs::core::Tensor::empty({1, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        buf_fc3_ = lfs::core::Tensor::empty({1, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        buf_output_ = lfs::core::Tensor::empty({1, FC_OUTPUT_DIM}, lfs::core::Device::CUDA);
        fc_input_buffer_ = lfs::core::Tensor::zeros({1, FC1_INPUT_DIM}, lfs::core::Device::CUDA);
        constexpr float DEFAULT_PRIOR = 1.0f;
        cudaMemcpy(fc_input_buffer_.ptr<float>() + CNN_FLAT_DIM, &DEFAULT_PRIOR, sizeof(float), cudaMemcpyHostToDevice);

        grad_fc3_out_ = lfs::core::Tensor::empty({1, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        grad_fc2_out_ = lfs::core::Tensor::empty({1, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        grad_fc1_out_ = lfs::core::Tensor::empty({1, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);

        const size_t per_cam_bytes =
            (FC_HIDDEN_DIM * FC1_INPUT_DIM + FC_HIDDEN_DIM + FC_HIDDEN_DIM * FC_HIDDEN_DIM + FC_HIDDEN_DIM +
             FC_HIDDEN_DIM * FC_HIDDEN_DIM + FC_HIDDEN_DIM + FC_OUTPUT_DIM * FC_HIDDEN_DIM + FC_OUTPUT_DIM) *
            sizeof(float);
        const size_t shared_bytes = (fc1_w_grad_.numel() + fc1_b_grad_.numel() + fc2_w_grad_.numel() +
                                     fc2_b_grad_.numel() + fc3_w_grad_.numel() + fc3_b_grad_.numel() +
                                     fc4_w_grad_.numel() + fc4_b_grad_.numel() + fc1_w_m_.numel() * 2 +
                                     fc1_b_m_.numel() * 2 + fc2_w_m_.numel() * 2 + fc2_b_m_.numel() * 2 +
                                     fc3_w_m_.numel() * 2 + fc3_b_m_.numel() * 2 + fc4_w_m_.numel() * 2 +
                                     fc4_b_m_.numel() * 2) *
                                    sizeof(float);

        LOG_INFO("[PPISPControllerPool] Created for %d cameras: %.1f MB/camera, %.1f MB shared", num_cameras,
                 per_cam_bytes / (1024.0 * 1024.0), shared_bytes / (1024.0 * 1024.0));
    }

    void PPISPControllerPool::allocate_buffers(const size_t max_h, const size_t max_w) {
        if (buf_h_ >= max_h && buf_w_ >= max_w)
            return;

        const size_t pool_h = max_h / POOL_STRIDE;
        const size_t pool_w = max_w / POOL_STRIDE;

        buf_conv1_ = lfs::core::Tensor::empty({1, CNN_CH1, max_h, max_w}, lfs::core::Device::CUDA);
        buf_pool_ = lfs::core::Tensor::empty({1, CNN_CH1, pool_h, pool_w}, lfs::core::Device::CUDA);
        buf_conv2_ = lfs::core::Tensor::empty({1, CNN_CH2, pool_h, pool_w}, lfs::core::Device::CUDA);
        buf_conv3_ = lfs::core::Tensor::empty({1, CNN_CH3, pool_h, pool_w}, lfs::core::Device::CUDA);
        buf_pool2_ = lfs::core::Tensor::empty({1, CNN_CH3, POOL2_SIZE, POOL2_SIZE}, lfs::core::Device::CUDA);

        buf_h_ = max_h;
        buf_w_ = max_w;

        const size_t buf_bytes = (buf_conv1_.numel() + buf_pool_.numel() + buf_conv2_.numel() + buf_conv3_.numel() +
                                  buf_pool2_.numel()) *
                                 sizeof(float);
        LOG_INFO("[PPISPControllerPool] CNN buffers %zux%zu: %.1f MB", max_h, max_w, buf_bytes / (1024.0 * 1024.0));
    }

    lfs::core::Tensor PPISPControllerPool::predict(const int camera_idx, const lfs::core::Tensor& rendered_rgb,
                                                   const float exposure_prior) {
        assert(camera_idx >= 0 && camera_idx < num_cameras_);
        assert(rendered_rgb.shape().rank() == 4);
        assert(rendered_rgb.shape()[0] == 1);
        assert(rendered_rgb.shape()[1] == 3);
        assert(buf_h_ > 0 && buf_w_ > 0);

        const size_t H = rendered_rgb.shape()[2];
        const size_t W = rendered_rgb.shape()[3];
        assert(H <= buf_h_ && W <= buf_w_);

        last_predict_camera_ = camera_idx;

        // Allocate correctly-sized intermediate tensors (cheap GPU allocations)
        const size_t pool_h = H / POOL_STRIDE;
        const size_t pool_w = W / POOL_STRIDE;
        auto conv1_out = lfs::core::Tensor::empty({1, CNN_CH1, H, W}, lfs::core::Device::CUDA);
        auto pool_out = lfs::core::Tensor::empty({1, CNN_CH1, pool_h, pool_w}, lfs::core::Device::CUDA);
        auto conv2_out = lfs::core::Tensor::empty({1, CNN_CH2, pool_h, pool_w}, lfs::core::Device::CUDA);
        auto conv3_out = lfs::core::Tensor::empty({1, CNN_CH3, pool_h, pool_w}, lfs::core::Device::CUDA);

        constexpr int POOL_KERNEL = 3;
        rendered_rgb.conv1x1_bias_out(conv1_w_, conv1_b_, conv1_out);
        conv1_out.max_pool2d_out(POOL_KERNEL, POOL_KERNEL, 0, pool_out);
        pool_out.relu_out(pool_out);
        pool_out.conv1x1_bias_relu_out(conv2_w_, conv2_b_, conv2_out);
        conv2_out.conv1x1_bias_out(conv3_w_, conv3_b_, conv3_out);
        conv3_out.adaptive_avg_pool2d_out(POOL2_SIZE, POOL2_SIZE, buf_pool2_);

        auto flat = buf_pool2_.flatten(1);
        cudaMemcpyAsync(fc_input_buffer_.ptr<float>(), flat.ptr<float>(), CNN_FLAT_DIM * sizeof(float),
                        cudaMemcpyDeviceToDevice, nullptr);
        if (exposure_prior != 1.0f) {
            cudaMemcpyAsync(fc_input_buffer_.ptr<float>() + CNN_FLAT_DIM, &exposure_prior, sizeof(float),
                            cudaMemcpyHostToDevice, nullptr);
        }
        cached_flat_ = fc_input_buffer_;

        cached_flat_.linear_bias_relu_out(fc1_w_[camera_idx], fc1_b_[camera_idx], buf_fc1_);
        buf_fc1_.linear_bias_relu_out(fc2_w_[camera_idx], fc2_b_[camera_idx], buf_fc2_);
        buf_fc2_.linear_bias_relu_out(fc3_w_[camera_idx], fc3_b_[camera_idx], buf_fc3_);
        buf_fc3_.linear_out(fc4_w_[camera_idx], fc4_b_[camera_idx], buf_output_);

        return buf_output_;
    }

    void PPISPControllerPool::backward(const int camera_idx, const lfs::core::Tensor& grad_output) {
        assert(camera_idx >= 0 && camera_idx < num_cameras_);
        assert(camera_idx == last_predict_camera_);
        assert(grad_output.shape().rank() == 2);
        assert(grad_output.shape()[0] == 1 && grad_output.shape()[1] == FC_OUTPUT_DIM);

        const float* const grad_fc4 = grad_output.ptr<float>();
        cudaStream_t stream = nullptr;

        launch_outer_product_accumulate(grad_fc4, buf_fc3_.ptr<float>(), fc4_w_grad_.ptr<float>(), FC_OUTPUT_DIM,
                                        FC_HIDDEN_DIM, 1.0f, stream);
        launch_bias_grad_accumulate(grad_fc4, fc4_b_grad_.ptr<float>(), FC_OUTPUT_DIM, stream);
        core::tensor_ops::launch_sgemm(grad_fc4, fc4_w_[camera_idx].ptr<float>(), grad_fc3_out_.ptr<float>(), 1,
                                       FC_HIDDEN_DIM, FC_OUTPUT_DIM, stream);

        launch_relu_backward(grad_fc3_out_.ptr<float>(), buf_fc3_.ptr<float>(), grad_fc3_out_.ptr<float>(),
                             FC_HIDDEN_DIM, stream);

        const float* const grad_fc3 = grad_fc3_out_.ptr<float>();
        launch_outer_product_accumulate(grad_fc3, buf_fc2_.ptr<float>(), fc3_w_grad_.ptr<float>(), FC_HIDDEN_DIM,
                                        FC_HIDDEN_DIM, 1.0f, stream);
        launch_bias_grad_accumulate(grad_fc3, fc3_b_grad_.ptr<float>(), FC_HIDDEN_DIM, stream);
        core::tensor_ops::launch_sgemm(grad_fc3, fc3_w_[camera_idx].ptr<float>(), grad_fc2_out_.ptr<float>(), 1,
                                       FC_HIDDEN_DIM, FC_HIDDEN_DIM, stream);

        launch_relu_backward(grad_fc2_out_.ptr<float>(), buf_fc2_.ptr<float>(), grad_fc2_out_.ptr<float>(),
                             FC_HIDDEN_DIM, stream);

        const float* const grad_fc2 = grad_fc2_out_.ptr<float>();
        launch_outer_product_accumulate(grad_fc2, buf_fc1_.ptr<float>(), fc2_w_grad_.ptr<float>(), FC_HIDDEN_DIM,
                                        FC_HIDDEN_DIM, 1.0f, stream);
        launch_bias_grad_accumulate(grad_fc2, fc2_b_grad_.ptr<float>(), FC_HIDDEN_DIM, stream);
        core::tensor_ops::launch_sgemm(grad_fc2, fc2_w_[camera_idx].ptr<float>(), grad_fc1_out_.ptr<float>(), 1,
                                       FC_HIDDEN_DIM, FC_HIDDEN_DIM, stream);

        launch_relu_backward(grad_fc1_out_.ptr<float>(), buf_fc1_.ptr<float>(), grad_fc1_out_.ptr<float>(),
                             FC_HIDDEN_DIM, stream);

        const float* const grad_fc1 = grad_fc1_out_.ptr<float>();
        launch_outer_product_accumulate(grad_fc1, cached_flat_.ptr<float>(), fc1_w_grad_.ptr<float>(), FC_HIDDEN_DIM,
                                        FC1_INPUT_DIM, 1.0f, stream);
        launch_bias_grad_accumulate(grad_fc1, fc1_b_grad_.ptr<float>(), FC_HIDDEN_DIM, stream);
    }

    void PPISPControllerPool::compute_bias_corrections(float& bc1_rcp, float& bc2_sqrt_rcp) const {
        const double t = static_cast<double>(step_ + 1);
        const double bc1 = 1.0 - std::pow(config_.beta1, t);
        const double bc2 = 1.0 - std::pow(config_.beta2, t);
        bc1_rcp = static_cast<float>(1.0 / bc1);
        bc2_sqrt_rcp = static_cast<float>(1.0 / std::sqrt(bc2));
    }

    void PPISPControllerPool::adam_update(lfs::core::Tensor& param, lfs::core::Tensor& exp_avg,
                                          lfs::core::Tensor& exp_avg_sq, const lfs::core::Tensor& grad) {
        float bc1_rcp, bc2_sqrt_rcp;
        compute_bias_corrections(bc1_rcp, bc2_sqrt_rcp);

        const float lr = static_cast<float>(current_lr_);
        const float beta1 = static_cast<float>(config_.beta1);
        const float beta2 = static_cast<float>(config_.beta2);
        const float eps = static_cast<float>(config_.eps);

        kernels::launch_ppisp_adam_update(param.ptr<float>(), exp_avg.ptr<float>(), exp_avg_sq.ptr<float>(),
                                          grad.ptr<float>(), static_cast<int>(param.numel()), lr, beta1, beta2,
                                          bc1_rcp, bc2_sqrt_rcp, eps, nullptr);
    }

    void PPISPControllerPool::optimizer_step(const int camera_idx) {
        assert(camera_idx >= 0 && camera_idx < num_cameras_);

        adam_update(fc1_w_[camera_idx], fc1_w_m_, fc1_w_v_, fc1_w_grad_);
        adam_update(fc1_b_[camera_idx], fc1_b_m_, fc1_b_v_, fc1_b_grad_);
        adam_update(fc2_w_[camera_idx], fc2_w_m_, fc2_w_v_, fc2_w_grad_);
        adam_update(fc2_b_[camera_idx], fc2_b_m_, fc2_b_v_, fc2_b_grad_);
        adam_update(fc3_w_[camera_idx], fc3_w_m_, fc3_w_v_, fc3_w_grad_);
        adam_update(fc3_b_[camera_idx], fc3_b_m_, fc3_b_v_, fc3_b_grad_);
        adam_update(fc4_w_[camera_idx], fc4_w_m_, fc4_w_v_, fc4_w_grad_);
        adam_update(fc4_b_[camera_idx], fc4_b_m_, fc4_b_v_, fc4_b_grad_);
    }

    void PPISPControllerPool::zero_grad() {
        fc1_w_grad_.zero_();
        fc1_b_grad_.zero_();
        fc2_w_grad_.zero_();
        fc2_b_grad_.zero_();
        fc3_w_grad_.zero_();
        fc3_b_grad_.zero_();
        fc4_w_grad_.zero_();
        fc4_b_grad_.zero_();
    }

    void PPISPControllerPool::scheduler_step(const int camera_idx) {
        (void)camera_idx;
        ++step_;

        if (step_ <= config_.warmup_steps) {
            const double progress = static_cast<double>(step_) / config_.warmup_steps;
            const double scale = config_.warmup_start_factor + (1.0 - config_.warmup_start_factor) * progress;
            current_lr_ = initial_lr_ * scale;
        } else {
            const double gamma = std::pow(config_.final_lr_factor, 1.0 / (total_iterations_ - config_.warmup_steps));
            current_lr_ = initial_lr_ * std::pow(gamma, step_ - config_.warmup_steps);
        }
    }

    void PPISPControllerPool::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_MAGIC), sizeof(CHECKPOINT_MAGIC));
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_VERSION), sizeof(CHECKPOINT_VERSION));

        os.write(reinterpret_cast<const char*>(&num_cameras_), sizeof(num_cameras_));
        os.write(reinterpret_cast<const char*>(&total_iterations_), sizeof(total_iterations_));
        os.write(reinterpret_cast<const char*>(&config_), sizeof(config_));
        os.write(reinterpret_cast<const char*>(&step_), sizeof(step_));
        os.write(reinterpret_cast<const char*>(&current_lr_), sizeof(current_lr_));
        os.write(reinterpret_cast<const char*>(&initial_lr_), sizeof(initial_lr_));

        // Shared CNN
        os << conv1_w_ << conv1_b_ << conv2_w_ << conv2_b_ << conv3_w_ << conv3_b_;

        // Per-camera FC weights
        for (int i = 0; i < num_cameras_; ++i) {
            os << fc1_w_[i] << fc1_b_[i];
            os << fc2_w_[i] << fc2_b_[i];
            os << fc3_w_[i] << fc3_b_[i];
            os << fc4_w_[i] << fc4_b_[i];
        }

        // Shared Adam state
        os << fc1_w_m_ << fc1_w_v_ << fc1_b_m_ << fc1_b_v_;
        os << fc2_w_m_ << fc2_w_v_ << fc2_b_m_ << fc2_b_v_;
        os << fc3_w_m_ << fc3_w_v_ << fc3_b_m_ << fc3_b_v_;
        os << fc4_w_m_ << fc4_w_v_ << fc4_b_m_ << fc4_b_v_;
    }

    void PPISPControllerPool::deserialize(std::istream& is) {
        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != CHECKPOINT_MAGIC)
            throw std::runtime_error("Invalid PPISPControllerPool checkpoint");
        if (version != CHECKPOINT_VERSION)
            throw std::runtime_error("Unsupported PPISPControllerPool checkpoint version");

        int saved_num_cameras;
        is.read(reinterpret_cast<char*>(&saved_num_cameras), sizeof(saved_num_cameras));
        if (saved_num_cameras != num_cameras_)
            throw std::runtime_error("Camera count mismatch in checkpoint");

        is.read(reinterpret_cast<char*>(&total_iterations_), sizeof(total_iterations_));
        is.read(reinterpret_cast<char*>(&config_), sizeof(config_));
        is.read(reinterpret_cast<char*>(&step_), sizeof(step_));
        is.read(reinterpret_cast<char*>(&current_lr_), sizeof(current_lr_));
        is.read(reinterpret_cast<char*>(&initial_lr_), sizeof(initial_lr_));

        // Shared CNN
        is >> conv1_w_ >> conv1_b_ >> conv2_w_ >> conv2_b_ >> conv3_w_ >> conv3_b_;
        conv1_w_ = conv1_w_.cuda();
        conv1_b_ = conv1_b_.cuda();
        conv2_w_ = conv2_w_.cuda();
        conv2_b_ = conv2_b_.cuda();
        conv3_w_ = conv3_w_.cuda();
        conv3_b_ = conv3_b_.cuda();

        // Per-camera FC weights
        for (int i = 0; i < num_cameras_; ++i) {
            is >> fc1_w_[i] >> fc1_b_[i];
            is >> fc2_w_[i] >> fc2_b_[i];
            is >> fc3_w_[i] >> fc3_b_[i];
            is >> fc4_w_[i] >> fc4_b_[i];

            fc1_w_[i] = fc1_w_[i].cuda();
            fc1_b_[i] = fc1_b_[i].cuda();
            fc2_w_[i] = fc2_w_[i].cuda();
            fc2_b_[i] = fc2_b_[i].cuda();
            fc3_w_[i] = fc3_w_[i].cuda();
            fc3_b_[i] = fc3_b_[i].cuda();
            fc4_w_[i] = fc4_w_[i].cuda();
            fc4_b_[i] = fc4_b_[i].cuda();
        }

        // Shared Adam state
        is >> fc1_w_m_ >> fc1_w_v_ >> fc1_b_m_ >> fc1_b_v_;
        is >> fc2_w_m_ >> fc2_w_v_ >> fc2_b_m_ >> fc2_b_v_;
        is >> fc3_w_m_ >> fc3_w_v_ >> fc3_b_m_ >> fc3_b_v_;
        is >> fc4_w_m_ >> fc4_w_v_ >> fc4_b_m_ >> fc4_b_v_;

        fc1_w_m_ = fc1_w_m_.cuda();
        fc1_w_v_ = fc1_w_v_.cuda();
        fc1_b_m_ = fc1_b_m_.cuda();
        fc1_b_v_ = fc1_b_v_.cuda();
        fc2_w_m_ = fc2_w_m_.cuda();
        fc2_w_v_ = fc2_w_v_.cuda();
        fc2_b_m_ = fc2_b_m_.cuda();
        fc2_b_v_ = fc2_b_v_.cuda();
        fc3_w_m_ = fc3_w_m_.cuda();
        fc3_w_v_ = fc3_w_v_.cuda();
        fc3_b_m_ = fc3_b_m_.cuda();
        fc3_b_v_ = fc3_b_v_.cuda();
        fc4_w_m_ = fc4_w_m_.cuda();
        fc4_w_v_ = fc4_w_v_.cuda();
        fc4_b_m_ = fc4_b_m_.cuda();
        fc4_b_v_ = fc4_b_v_.cuda();
    }

    void PPISPControllerPool::serialize_inference(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&INFERENCE_MAGIC), sizeof(INFERENCE_MAGIC));
        os.write(reinterpret_cast<const char*>(&INFERENCE_VERSION), sizeof(INFERENCE_VERSION));
        os.write(reinterpret_cast<const char*>(&num_cameras_), sizeof(num_cameras_));

        os << conv1_w_ << conv1_b_ << conv2_w_ << conv2_b_ << conv3_w_ << conv3_b_;

        for (int i = 0; i < num_cameras_; ++i) {
            os << fc1_w_[i] << fc1_b_[i];
            os << fc2_w_[i] << fc2_b_[i];
            os << fc3_w_[i] << fc3_b_[i];
            os << fc4_w_[i] << fc4_b_[i];
        }
    }

    void PPISPControllerPool::deserialize_inference(std::istream& is) {
        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != INFERENCE_MAGIC)
            throw std::runtime_error("Invalid PPISPControllerPool inference file");
        if (version != INFERENCE_VERSION)
            throw std::runtime_error("Unsupported inference version");

        int saved_num_cameras;
        is.read(reinterpret_cast<char*>(&saved_num_cameras), sizeof(saved_num_cameras));
        if (saved_num_cameras != num_cameras_)
            throw std::runtime_error("Camera count mismatch");

        is >> conv1_w_ >> conv1_b_ >> conv2_w_ >> conv2_b_ >> conv3_w_ >> conv3_b_;
        conv1_w_ = conv1_w_.cuda();
        conv1_b_ = conv1_b_.cuda();
        conv2_w_ = conv2_w_.cuda();
        conv2_b_ = conv2_b_.cuda();
        conv3_w_ = conv3_w_.cuda();
        conv3_b_ = conv3_b_.cuda();

        for (int i = 0; i < num_cameras_; ++i) {
            is >> fc1_w_[i] >> fc1_b_[i];
            is >> fc2_w_[i] >> fc2_b_[i];
            is >> fc3_w_[i] >> fc3_b_[i];
            is >> fc4_w_[i] >> fc4_b_[i];

            fc1_w_[i] = fc1_w_[i].cuda();
            fc1_b_[i] = fc1_b_[i].cuda();
            fc2_w_[i] = fc2_w_[i].cuda();
            fc2_b_[i] = fc2_b_[i].cuda();
            fc3_w_[i] = fc3_w_[i].cuda();
            fc3_b_[i] = fc3_b_[i].cuda();
            fc4_w_[i] = fc4_w_[i].cuda();
            fc4_b_[i] = fc4_b_[i].cuda();
        }
    }

    std::string PPISPControllerPool::copy_inference_weights_from(
        const PPISPControllerPool& source,
        const std::vector<int>& source_camera_indices) {

        if (static_cast<int>(source_camera_indices.size()) != num_cameras_) {
            return "Camera mapping size does not match target controller pool camera count";
        }

        for (size_t i = 0; i < source_camera_indices.size(); ++i) {
            if (source_camera_indices[i] < 0 || source_camera_indices[i] >= source.num_cameras_) {
                return "Camera mapping contains out-of-range source index at position " + std::to_string(i);
            }
        }

        conv1_w_ = source.conv1_w_.clone().contiguous();
        conv1_b_ = source.conv1_b_.clone().contiguous();
        conv2_w_ = source.conv2_w_.clone().contiguous();
        conv2_b_ = source.conv2_b_.clone().contiguous();
        conv3_w_ = source.conv3_w_.clone().contiguous();
        conv3_b_ = source.conv3_b_.clone().contiguous();

        for (int target_idx = 0; target_idx < num_cameras_; ++target_idx) {
            const int source_idx = source_camera_indices[static_cast<size_t>(target_idx)];
            fc1_w_[target_idx] = source.fc1_w_[source_idx].clone().contiguous();
            fc1_b_[target_idx] = source.fc1_b_[source_idx].clone().contiguous();
            fc2_w_[target_idx] = source.fc2_w_[source_idx].clone().contiguous();
            fc2_b_[target_idx] = source.fc2_b_[source_idx].clone().contiguous();
            fc3_w_[target_idx] = source.fc3_w_[source_idx].clone().contiguous();
            fc3_b_[target_idx] = source.fc3_b_[source_idx].clone().contiguous();
            fc4_w_[target_idx] = source.fc4_w_[source_idx].clone().contiguous();
            fc4_b_[target_idx] = source.fc4_b_[source_idx].clone().contiguous();
        }

        zero_grad();
        fc1_w_m_.zero_();
        fc1_w_v_.zero_();
        fc1_b_m_.zero_();
        fc1_b_v_.zero_();
        fc2_w_m_.zero_();
        fc2_w_v_.zero_();
        fc2_b_m_.zero_();
        fc2_b_v_.zero_();
        fc3_w_m_.zero_();
        fc3_w_v_.zero_();
        fc3_b_m_.zero_();
        fc3_b_v_.zero_();
        fc4_w_m_.zero_();
        fc4_w_v_.zero_();
        fc4_b_m_.zero_();
        fc4_b_v_.zero_();
        step_ = 0;
        current_lr_ = initial_lr_;
        last_predict_camera_ = -1;

        return {};
    }

} // namespace lfs::training
