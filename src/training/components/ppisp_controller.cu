/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core/tensor/internal/tensor_ops.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "lfs/kernels/ppisp.cuh"
#include "ppisp_controller.hpp"
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

        lfs::core::Tensor zeros_like(const lfs::core::Tensor& t) {
            return lfs::core::Tensor::zeros(t.shape(), t.device());
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

    // Static member definitions for shared forward buffers
    size_t PPISPController::shared_buf_h_ = 0;
    size_t PPISPController::shared_buf_w_ = 0;
    lfs::core::Tensor PPISPController::shared_buf_conv1_;
    lfs::core::Tensor PPISPController::shared_buf_pool_;
    lfs::core::Tensor PPISPController::shared_buf_conv2_;
    lfs::core::Tensor PPISPController::shared_buf_conv3_;
    lfs::core::Tensor PPISPController::shared_buf_pool2_;

    PPISPController::PPISPController(const int total_iterations, Config config)
        : config_(config),
          current_lr_(config.warmup_steps > 0 ? config.lr * config.warmup_start_factor : config.lr),
          initial_lr_(config.lr),
          total_iterations_(total_iterations) {

        conv1_w_ = kaiming_uniform(3, CNN_CH1);
        conv1_b_ = zeros_bias(CNN_CH1);
        conv2_w_ = kaiming_uniform(CNN_CH1, CNN_CH2);
        conv2_b_ = zeros_bias(CNN_CH2);
        conv3_w_ = kaiming_uniform(CNN_CH2, CNN_CH3);
        conv3_b_ = zeros_bias(CNN_CH3);

        fc1_w_ = kaiming_uniform(FC1_INPUT_DIM, FC_HIDDEN_DIM);
        fc1_b_ = zeros_bias(FC_HIDDEN_DIM);
        fc2_w_ = kaiming_uniform(FC_HIDDEN_DIM, FC_HIDDEN_DIM);
        fc2_b_ = zeros_bias(FC_HIDDEN_DIM);
        fc3_w_ = kaiming_uniform(FC_HIDDEN_DIM, FC_HIDDEN_DIM);
        fc3_b_ = zeros_bias(FC_HIDDEN_DIM);
        fc4_w_ = kaiming_uniform(FC_HIDDEN_DIM, FC_OUTPUT_DIM);
        fc4_b_ = zeros_bias(FC_OUTPUT_DIM);

        fc1_w_grad_ = zeros_like(fc1_w_);
        fc1_b_grad_ = zeros_like(fc1_b_);
        fc2_w_grad_ = zeros_like(fc2_w_);
        fc2_b_grad_ = zeros_like(fc2_b_);
        fc3_w_grad_ = zeros_like(fc3_w_);
        fc3_b_grad_ = zeros_like(fc3_b_);
        fc4_w_grad_ = zeros_like(fc4_w_);
        fc4_b_grad_ = zeros_like(fc4_b_);

        fc1_w_m_ = zeros_like(fc1_w_);
        fc1_w_v_ = zeros_like(fc1_w_);
        fc1_b_m_ = zeros_like(fc1_b_);
        fc1_b_v_ = zeros_like(fc1_b_);
        fc2_w_m_ = zeros_like(fc2_w_);
        fc2_w_v_ = zeros_like(fc2_w_);
        fc2_b_m_ = zeros_like(fc2_b_);
        fc2_b_v_ = zeros_like(fc2_b_);
        fc3_w_m_ = zeros_like(fc3_w_);
        fc3_w_v_ = zeros_like(fc3_w_);
        fc3_b_m_ = zeros_like(fc3_b_);
        fc3_b_v_ = zeros_like(fc3_b_);
        fc4_w_m_ = zeros_like(fc4_w_);
        fc4_w_v_ = zeros_like(fc4_w_);
        fc4_b_m_ = zeros_like(fc4_b_);
        fc4_b_v_ = zeros_like(fc4_b_);

        buf_fc1_ = lfs::core::Tensor::empty({1, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        buf_fc2_ = lfs::core::Tensor::empty({1, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        buf_fc3_ = lfs::core::Tensor::empty({1, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        buf_output_ = lfs::core::Tensor::empty({1, FC_OUTPUT_DIM}, lfs::core::Device::CUDA);
        fc_input_buffer_ = lfs::core::Tensor::zeros({1, FC1_INPUT_DIM}, lfs::core::Device::CUDA);
        constexpr float DEFAULT_PRIOR = 1.0f;
        cudaMemcpy(fc_input_buffer_.ptr<float>() + CNN_FLAT_DIM, &DEFAULT_PRIOR, sizeof(float), cudaMemcpyHostToDevice);

        bwd_grad_fc3_out_ = lfs::core::Tensor::empty({1, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        bwd_grad_fc2_out_ = lfs::core::Tensor::empty({1, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);
        bwd_grad_fc1_out_ = lfs::core::Tensor::empty({1, FC_HIDDEN_DIM}, lfs::core::Device::CUDA);

        LOG_DEBUG("PPISPController: iterations={}, lr={:.2e}", total_iterations, config.lr);
    }

    void PPISPController::preallocate_shared_buffers(const size_t max_H, const size_t max_W) {
        if (shared_buf_h_ >= max_H && shared_buf_w_ >= max_W)
            return;

        const size_t pool_h = max_H / POOL_STRIDE;
        const size_t pool_w = max_W / POOL_STRIDE;

        shared_buf_conv1_ = lfs::core::Tensor::empty({1, CNN_CH1, max_H, max_W}, lfs::core::Device::CUDA);
        shared_buf_pool_ = lfs::core::Tensor::empty({1, CNN_CH1, pool_h, pool_w}, lfs::core::Device::CUDA);
        shared_buf_conv2_ = lfs::core::Tensor::empty({1, CNN_CH2, pool_h, pool_w}, lfs::core::Device::CUDA);
        shared_buf_conv3_ = lfs::core::Tensor::empty({1, CNN_CH3, pool_h, pool_w}, lfs::core::Device::CUDA);
        shared_buf_pool2_ = lfs::core::Tensor::empty({1, CNN_CH3, POOL2_SIZE, POOL2_SIZE}, lfs::core::Device::CUDA);

        shared_buf_h_ = max_H;
        shared_buf_w_ = max_W;

        LOG_INFO("[PPISPController] Shared buffers: {}x{}", max_H, max_W);
    }

    lfs::core::Tensor PPISPController::predict(const lfs::core::Tensor& rendered_rgb, const float exposure_prior) {
        assert(rendered_rgb.shape().rank() == 4);
        assert(rendered_rgb.shape()[0] == 1);
        assert(rendered_rgb.shape()[1] == 3);
        assert(shared_buf_h_ > 0 && shared_buf_w_ > 0);

        const size_t H = rendered_rgb.shape()[2];
        const size_t W = rendered_rgb.shape()[3];
        assert(H <= shared_buf_h_ && W <= shared_buf_w_);

        // Allocate correctly-sized intermediate tensors (cheap GPU allocations)
        const size_t pool_h = H / POOL_STRIDE;
        const size_t pool_w = W / POOL_STRIDE;
        auto conv1_out = lfs::core::Tensor::empty({1, CNN_CH1, H, W}, lfs::core::Device::CUDA);
        auto pool_out = lfs::core::Tensor::empty({1, CNN_CH1, pool_h, pool_w}, lfs::core::Device::CUDA);
        auto conv2_out = lfs::core::Tensor::empty({1, CNN_CH2, pool_h, pool_w}, lfs::core::Device::CUDA);
        auto conv3_out = lfs::core::Tensor::empty({1, CNN_CH3, pool_h, pool_w}, lfs::core::Device::CUDA);

        rendered_rgb.conv1x1_bias_out(conv1_w_, conv1_b_, conv1_out);
        conv1_out.max_pool2d_out(POOL_STRIDE, POOL_STRIDE, 0, pool_out);
        pool_out.relu_out(pool_out);
        pool_out.conv1x1_bias_relu_out(conv2_w_, conv2_b_, conv2_out);
        conv2_out.conv1x1_bias_out(conv3_w_, conv3_b_, conv3_out);
        conv3_out.adaptive_avg_pool2d_out(POOL2_SIZE, POOL2_SIZE, shared_buf_pool2_);

        auto flat = shared_buf_pool2_.flatten(1);
        cudaMemcpyAsync(fc_input_buffer_.ptr<float>(), flat.ptr<float>(), CNN_FLAT_DIM * sizeof(float),
                        cudaMemcpyDeviceToDevice, nullptr);
        if (exposure_prior != 1.0f) {
            cudaMemcpyAsync(fc_input_buffer_.ptr<float>() + CNN_FLAT_DIM, &exposure_prior, sizeof(float),
                            cudaMemcpyHostToDevice, nullptr);
        }
        cached_flat_ = fc_input_buffer_;

        cached_flat_.linear_bias_relu_out(fc1_w_, fc1_b_, buf_fc1_);
        buf_fc1_.linear_bias_relu_out(fc2_w_, fc2_b_, buf_fc2_);
        buf_fc2_.linear_bias_relu_out(fc3_w_, fc3_b_, buf_fc3_);
        buf_fc3_.linear_out(fc4_w_, fc4_b_, buf_output_);

        return buf_output_;
    }

    void PPISPController::backward(const lfs::core::Tensor& grad_output) {
        assert(grad_output.shape().rank() == 2);
        assert(grad_output.shape()[0] == 1 && grad_output.shape()[1] == FC_OUTPUT_DIM);

        const float* const grad_fc4 = grad_output.ptr<float>();
        cudaStream_t stream = nullptr;

        launch_outer_product_accumulate(grad_fc4, buf_fc3_.ptr<float>(), fc4_w_grad_.ptr<float>(), FC_OUTPUT_DIM,
                                        FC_HIDDEN_DIM, 1.0f, stream);
        launch_bias_grad_accumulate(grad_fc4, fc4_b_grad_.ptr<float>(), FC_OUTPUT_DIM, stream);
        core::tensor_ops::launch_sgemm(grad_fc4, fc4_w_.ptr<float>(), bwd_grad_fc3_out_.ptr<float>(), 1, FC_HIDDEN_DIM,
                                       FC_OUTPUT_DIM, stream);

        launch_relu_backward(bwd_grad_fc3_out_.ptr<float>(), buf_fc3_.ptr<float>(), bwd_grad_fc3_out_.ptr<float>(),
                             FC_HIDDEN_DIM, stream);

        const float* const grad_fc3 = bwd_grad_fc3_out_.ptr<float>();
        launch_outer_product_accumulate(grad_fc3, buf_fc2_.ptr<float>(), fc3_w_grad_.ptr<float>(), FC_HIDDEN_DIM,
                                        FC_HIDDEN_DIM, 1.0f, stream);
        launch_bias_grad_accumulate(grad_fc3, fc3_b_grad_.ptr<float>(), FC_HIDDEN_DIM, stream);
        core::tensor_ops::launch_sgemm(grad_fc3, fc3_w_.ptr<float>(), bwd_grad_fc2_out_.ptr<float>(), 1, FC_HIDDEN_DIM,
                                       FC_HIDDEN_DIM, stream);

        launch_relu_backward(bwd_grad_fc2_out_.ptr<float>(), buf_fc2_.ptr<float>(), bwd_grad_fc2_out_.ptr<float>(),
                             FC_HIDDEN_DIM, stream);

        const float* const grad_fc2 = bwd_grad_fc2_out_.ptr<float>();
        launch_outer_product_accumulate(grad_fc2, buf_fc1_.ptr<float>(), fc2_w_grad_.ptr<float>(), FC_HIDDEN_DIM,
                                        FC_HIDDEN_DIM, 1.0f, stream);
        launch_bias_grad_accumulate(grad_fc2, fc2_b_grad_.ptr<float>(), FC_HIDDEN_DIM, stream);
        core::tensor_ops::launch_sgemm(grad_fc2, fc2_w_.ptr<float>(), bwd_grad_fc1_out_.ptr<float>(), 1, FC_HIDDEN_DIM,
                                       FC_HIDDEN_DIM, stream);

        launch_relu_backward(bwd_grad_fc1_out_.ptr<float>(), buf_fc1_.ptr<float>(), bwd_grad_fc1_out_.ptr<float>(),
                             FC_HIDDEN_DIM, stream);

        const float* const grad_fc1 = bwd_grad_fc1_out_.ptr<float>();
        launch_outer_product_accumulate(grad_fc1, cached_flat_.ptr<float>(), fc1_w_grad_.ptr<float>(), FC_HIDDEN_DIM,
                                        FC1_INPUT_DIM, 1.0f, stream);
        launch_bias_grad_accumulate(grad_fc1, fc1_b_grad_.ptr<float>(), FC_HIDDEN_DIM, stream);
    }

    void PPISPController::adam_update(lfs::core::Tensor& param, lfs::core::Tensor& exp_avg,
                                      lfs::core::Tensor& exp_avg_sq, const lfs::core::Tensor& grad) {
        float bc1_rcp, bc2_sqrt_rcp;
        compute_bias_corrections(bc1_rcp, bc2_sqrt_rcp);

        const float lr = static_cast<float>(current_lr_);
        const float beta1 = static_cast<float>(config_.beta1);
        const float beta2 = static_cast<float>(config_.beta2);
        const float eps = static_cast<float>(config_.eps);

        kernels::launch_ppisp_adam_update(param.ptr<float>(), exp_avg.ptr<float>(), exp_avg_sq.ptr<float>(),
                                          grad.ptr<float>(), static_cast<int>(param.numel()), lr, beta1, beta2, bc1_rcp,
                                          bc2_sqrt_rcp, eps, nullptr);
    }

    void PPISPController::optimizer_step() {
        adam_update(fc1_w_, fc1_w_m_, fc1_w_v_, fc1_w_grad_);
        adam_update(fc1_b_, fc1_b_m_, fc1_b_v_, fc1_b_grad_);
        adam_update(fc2_w_, fc2_w_m_, fc2_w_v_, fc2_w_grad_);
        adam_update(fc2_b_, fc2_b_m_, fc2_b_v_, fc2_b_grad_);
        adam_update(fc3_w_, fc3_w_m_, fc3_w_v_, fc3_w_grad_);
        adam_update(fc3_b_, fc3_b_m_, fc3_b_v_, fc3_b_grad_);
        adam_update(fc4_w_, fc4_w_m_, fc4_w_v_, fc4_w_grad_);
        adam_update(fc4_b_, fc4_b_m_, fc4_b_v_, fc4_b_grad_);
    }

    void PPISPController::zero_grad() {
        fc1_w_grad_.zero_();
        fc1_b_grad_.zero_();
        fc2_w_grad_.zero_();
        fc2_b_grad_.zero_();
        fc3_w_grad_.zero_();
        fc3_b_grad_.zero_();
        fc4_w_grad_.zero_();
        fc4_b_grad_.zero_();
    }

    void PPISPController::scheduler_step() {
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

    void PPISPController::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_MAGIC), sizeof(CHECKPOINT_MAGIC));
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_VERSION), sizeof(CHECKPOINT_VERSION));

        os.write(reinterpret_cast<const char*>(&config_), sizeof(config_));
        os.write(reinterpret_cast<const char*>(&step_), sizeof(step_));
        os.write(reinterpret_cast<const char*>(&current_lr_), sizeof(current_lr_));
        os.write(reinterpret_cast<const char*>(&initial_lr_), sizeof(initial_lr_));
        os.write(reinterpret_cast<const char*>(&total_iterations_), sizeof(total_iterations_));

        os << conv1_w_ << conv1_b_;
        os << conv2_w_ << conv2_b_;
        os << conv3_w_ << conv3_b_;

        os << fc1_w_ << fc1_b_ << fc1_w_m_ << fc1_w_v_ << fc1_b_m_ << fc1_b_v_;
        os << fc2_w_ << fc2_b_ << fc2_w_m_ << fc2_w_v_ << fc2_b_m_ << fc2_b_v_;
        os << fc3_w_ << fc3_b_ << fc3_w_m_ << fc3_w_v_ << fc3_b_m_ << fc3_b_v_;
        os << fc4_w_ << fc4_b_ << fc4_w_m_ << fc4_w_v_ << fc4_b_m_ << fc4_b_v_;
    }

    void PPISPController::deserialize(std::istream& is) {
        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != CHECKPOINT_MAGIC)
            throw std::runtime_error("Invalid PPISPController checkpoint");
        if (version != CHECKPOINT_VERSION)
            throw std::runtime_error("Unsupported PPISPController checkpoint version");

        is.read(reinterpret_cast<char*>(&config_), sizeof(config_));
        is.read(reinterpret_cast<char*>(&step_), sizeof(step_));
        is.read(reinterpret_cast<char*>(&current_lr_), sizeof(current_lr_));
        is.read(reinterpret_cast<char*>(&initial_lr_), sizeof(initial_lr_));
        is.read(reinterpret_cast<char*>(&total_iterations_), sizeof(total_iterations_));

        is >> conv1_w_ >> conv1_b_;
        is >> conv2_w_ >> conv2_b_;
        is >> conv3_w_ >> conv3_b_;

        is >> fc1_w_ >> fc1_b_ >> fc1_w_m_ >> fc1_w_v_ >> fc1_b_m_ >> fc1_b_v_;
        is >> fc2_w_ >> fc2_b_ >> fc2_w_m_ >> fc2_w_v_ >> fc2_b_m_ >> fc2_b_v_;
        is >> fc3_w_ >> fc3_b_ >> fc3_w_m_ >> fc3_w_v_ >> fc3_b_m_ >> fc3_b_v_;
        is >> fc4_w_ >> fc4_b_ >> fc4_w_m_ >> fc4_w_v_ >> fc4_b_m_ >> fc4_b_v_;

        conv1_w_ = conv1_w_.cuda();
        conv1_b_ = conv1_b_.cuda();
        conv2_w_ = conv2_w_.cuda();
        conv2_b_ = conv2_b_.cuda();
        conv3_w_ = conv3_w_.cuda();
        conv3_b_ = conv3_b_.cuda();

        fc1_w_ = fc1_w_.cuda();
        fc1_b_ = fc1_b_.cuda();
        fc1_w_m_ = fc1_w_m_.cuda();
        fc1_w_v_ = fc1_w_v_.cuda();
        fc1_b_m_ = fc1_b_m_.cuda();
        fc1_b_v_ = fc1_b_v_.cuda();

        fc2_w_ = fc2_w_.cuda();
        fc2_b_ = fc2_b_.cuda();
        fc2_w_m_ = fc2_w_m_.cuda();
        fc2_w_v_ = fc2_w_v_.cuda();
        fc2_b_m_ = fc2_b_m_.cuda();
        fc2_b_v_ = fc2_b_v_.cuda();

        fc3_w_ = fc3_w_.cuda();
        fc3_b_ = fc3_b_.cuda();
        fc3_w_m_ = fc3_w_m_.cuda();
        fc3_w_v_ = fc3_w_v_.cuda();
        fc3_b_m_ = fc3_b_m_.cuda();
        fc3_b_v_ = fc3_b_v_.cuda();

        fc4_w_ = fc4_w_.cuda();
        fc4_b_ = fc4_b_.cuda();
        fc4_w_m_ = fc4_w_m_.cuda();
        fc4_w_v_ = fc4_w_v_.cuda();
        fc4_b_m_ = fc4_b_m_.cuda();
        fc4_b_v_ = fc4_b_v_.cuda();

        fc1_w_grad_ = zeros_like(fc1_w_);
        fc1_b_grad_ = zeros_like(fc1_b_);
        fc2_w_grad_ = zeros_like(fc2_w_);
        fc2_b_grad_ = zeros_like(fc2_b_);
        fc3_w_grad_ = zeros_like(fc3_w_);
        fc3_b_grad_ = zeros_like(fc3_b_);
        fc4_w_grad_ = zeros_like(fc4_w_);
        fc4_b_grad_ = zeros_like(fc4_b_);
    }

    void PPISPController::serialize_inference(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&INFERENCE_MAGIC), sizeof(INFERENCE_MAGIC));
        os.write(reinterpret_cast<const char*>(&INFERENCE_VERSION), sizeof(INFERENCE_VERSION));

        os << conv1_w_ << conv1_b_;
        os << conv2_w_ << conv2_b_;
        os << conv3_w_ << conv3_b_;

        os << fc1_w_ << fc1_b_;
        os << fc2_w_ << fc2_b_;
        os << fc3_w_ << fc3_b_;
        os << fc4_w_ << fc4_b_;
    }

    void PPISPController::deserialize_inference(std::istream& is) {
        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != INFERENCE_MAGIC)
            throw std::runtime_error("Invalid PPISPController inference file");
        if (version != INFERENCE_VERSION)
            throw std::runtime_error("Unsupported inference version");

        is >> conv1_w_ >> conv1_b_;
        is >> conv2_w_ >> conv2_b_;
        is >> conv3_w_ >> conv3_b_;

        is >> fc1_w_ >> fc1_b_;
        is >> fc2_w_ >> fc2_b_;
        is >> fc3_w_ >> fc3_b_;
        is >> fc4_w_ >> fc4_b_;

        conv1_w_ = conv1_w_.cuda();
        conv1_b_ = conv1_b_.cuda();
        conv2_w_ = conv2_w_.cuda();
        conv2_b_ = conv2_b_.cuda();
        conv3_w_ = conv3_w_.cuda();
        conv3_b_ = conv3_b_.cuda();

        fc1_w_ = fc1_w_.cuda();
        fc1_b_ = fc1_b_.cuda();
        fc2_w_ = fc2_w_.cuda();
        fc2_b_ = fc2_b_.cuda();
        fc3_w_ = fc3_w_.cuda();
        fc3_b_ = fc3_b_.cuda();
        fc4_w_ = fc4_w_.cuda();
        fc4_b_ = fc4_b_.cuda();
    }

} // namespace lfs::training
