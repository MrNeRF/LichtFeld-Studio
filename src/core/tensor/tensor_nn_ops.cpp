/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_nn_ops.hpp"
#include "core/logger.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"

#include <format>

namespace lfs::core {

    namespace {

        struct NamedTensorOperand {
            std::string_view name;
            const Tensor* tensor;
        };

        void assert_float32_same_device(const Tensor& reference,
                                        const std::string_view operation,
                                        const std::initializer_list<NamedTensorOperand> tensors = {}) {
            tensor_contract::require_valid(
                reference, operation, "input", LFS_SOURCE_SITE_CURRENT());
            tensor_contract::require_dtype(
                reference, DataType::Float32, operation, "input",
                LFS_SOURCE_SITE_CURRENT());
            for (const auto& [name, tensor] : tensors) {
                LFS_ASSERT_MSG(tensor != nullptr,
                               std::format("{} requires a non-null {} tensor pointer "
                                           "(operand={}, pointer=null)",
                                           operation, name, name));
                tensor_contract::require_valid(
                    *tensor, operation, name, LFS_SOURCE_SITE_CURRENT());
                tensor_contract::require_dtype(
                    *tensor, DataType::Float32, operation, name,
                    LFS_SOURCE_SITE_CURRENT());
                tensor_contract::require_same_device(
                    reference, *tensor, operation, "input", name,
                    LFS_SOURCE_SITE_CURRENT());
            }
        }

        void assert_bias_shape(const Tensor& bias,
                               const size_t expected_channels,
                               const std::string_view operation) {
            LFS_ASSERT_MSG(bias.shape().rank() == 1 && bias.shape()[0] == expected_channels,
                           std::format("{} bias must be rank 1 with one value per output channel "
                                       "(bias_shape={}, expected_shape=[{}])",
                                       operation, bias.shape().str(), expected_channels));
        }

        void cpu_max_pool2d(const float* input, float* output,
                            int N, int C, int H_in, int W_in,
                            int H_out, int W_out,
                            int kernel, int stride, int padding) {
            for (int n = 0; n < N; ++n) {
                for (int c = 0; c < C; ++c) {
                    for (int h_out = 0; h_out < H_out; ++h_out) {
                        for (int w_out = 0; w_out < W_out; ++w_out) {
                            const int h_start = h_out * stride - padding;
                            const int w_start = w_out * stride - padding;

                            float max_val = -std::numeric_limits<float>::max();

                            for (int kh = 0; kh < kernel; ++kh) {
                                const int h_in = h_start + kh;
                                if (h_in < 0 || h_in >= H_in)
                                    continue;

                                for (int kw = 0; kw < kernel; ++kw) {
                                    const int w_in = w_start + kw;
                                    if (w_in < 0 || w_in >= W_in)
                                        continue;

                                    const int idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                                    max_val = std::max(max_val, input[idx]);
                                }
                            }

                            const int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            output[out_idx] = max_val;
                        }
                    }
                }
            }
        }

        void cpu_adaptive_avg_pool2d(const float* input, float* output,
                                     int N, int C, int H_in, int W_in,
                                     int H_out, int W_out) {
            for (int n = 0; n < N; ++n) {
                for (int c = 0; c < C; ++c) {
                    for (int h_out = 0; h_out < H_out; ++h_out) {
                        for (int w_out = 0; w_out < W_out; ++w_out) {
                            const int h_start = (h_out * H_in) / H_out;
                            const int h_end = ((h_out + 1) * H_in + H_out - 1) / H_out;
                            const int w_start = (w_out * W_in) / W_out;
                            const int w_end = ((w_out + 1) * W_in + W_out - 1) / W_out;

                            float sum = 0.0f;
                            int count = 0;

                            for (int h = h_start; h < h_end; ++h) {
                                for (int w = w_start; w < w_end; ++w) {
                                    const int idx = ((n * C + c) * H_in + h) * W_in + w;
                                    sum += input[idx];
                                    ++count;
                                }
                            }

                            const int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            output[out_idx] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
                        }
                    }
                }
            }
        }

        void cpu_bias_relu(const float* input, const float* bias, float* output,
                           int total, int channels, int spatial) {
            for (int i = 0; i < total; ++i) {
                const int c = (i / spatial) % channels;
                const float val = input[i] + bias[c];
                output[i] = val > 0.0f ? val : 0.0f;
            }
        }

    } // namespace

    Tensor Tensor::conv1x1(const Tensor& weight) const {
        return conv1x1(weight, Tensor{});
    }

    Tensor Tensor::conv1x1(const Tensor& weight, const Tensor& bias) const {
        assert_float32_same_device(*this, "conv1x1", {{"weight", &weight}});
        if (bias.is_valid()) {
            assert_float32_same_device(*this, "conv1x1", {{"bias", &bias}});
        }
        LFS_ASSERT_MSG(shape_.rank() == 4,
                       std::format("conv1x1 input must be 4D [N,C,H,W] "
                                   "(input_shape={}, input_rank={})",
                                   shape_.str(), shape_.rank()));
        LFS_ASSERT_MSG(weight.shape_.rank() == 2,
                       std::format("conv1x1 weight must be 2D [C_out,C_in] "
                                   "(weight_shape={}, weight_rank={})",
                                   weight.shape_.str(), weight.shape_.rank()));
        LFS_ASSERT_MSG(shape_[1] == weight.shape_[1],
                       std::format("conv1x1 input channels must match weight input channels "
                                   "(input_channels={}, weight_input_channels={}, "
                                   "input_shape={}, weight_shape={})",
                                   shape_[1], weight.shape_[1], shape_.str(), weight.shape_.str()));

        const size_t N = shape_[0];
        const size_t C_in = shape_[1];
        const size_t H = shape_[2];
        const size_t W = shape_[3];
        const size_t C_out = weight.shape_[0];
        const size_t S = H * W;

        if (bias.is_valid()) {
            assert_bias_shape(bias, C_out, "conv1x1");
        }

        const Tensor& input_cont = is_contiguous() ? *this : contiguous();
        const Tensor& weight_cont = weight.is_contiguous() ? weight : weight.contiguous();

        // GPU path: Output[C_out, S] = Weight[C_out, C_in] @ Input[C_in, S]
        if (device_ == Device::CUDA && N == 1) {
            auto output = empty({N, C_out, H, W}, Device::CUDA, dtype_);

            tensor_ops::launch_sgemm(weight_cont.ptr<float>(), input_cont.ptr<float>(),
                                     output.ptr<float>(), C_out, S, C_in, stream());

            if (bias.is_valid()) {
                const int total = static_cast<int>(N * C_out * H * W);
                tensor_ops::launch_bias_add(output.ptr<float>(), bias.ptr<float>(),
                                            output.ptr<float>(), total,
                                            static_cast<int>(C_out), static_cast<int>(S),
                                            stream());
            }

            return output;
        }

        // Batched GPU path for N > 1: process each batch
        if (device_ == Device::CUDA) {
            auto output = empty({N, C_out, H, W}, Device::CUDA, dtype_);

            for (size_t n = 0; n < N; ++n) {
                const float* in_ptr = input_cont.ptr<float>() + n * C_in * S;
                float* out_ptr = output.ptr<float>() + n * C_out * S;
                tensor_ops::launch_sgemm(weight_cont.ptr<float>(), in_ptr, out_ptr,
                                         C_out, S, C_in, stream());
            }

            if (bias.is_valid()) {
                const int total = static_cast<int>(N * C_out * H * W);
                tensor_ops::launch_bias_add(output.ptr<float>(), bias.ptr<float>(),
                                            output.ptr<float>(), total,
                                            static_cast<int>(C_out), static_cast<int>(S),
                                            stream());
            }

            return output;
        }

        // CPU fallback: use original permute-based implementation
        auto input_nhwc = input_cont.permute({0, 2, 3, 1}).contiguous();
        auto input_2d = input_nhwc.reshape({static_cast<int>(N * H * W), static_cast<int>(C_in)});
        auto weight_t = weight_cont.transpose(0, 1).contiguous();
        auto output_2d = input_2d.matmul(weight_t);

        if (bias.is_valid()) {
            output_2d = output_2d + bias;
        }

        auto output_nhwc = output_2d.reshape({static_cast<int>(N), static_cast<int>(H),
                                              static_cast<int>(W), static_cast<int>(C_out)});
        return output_nhwc.permute({0, 3, 1, 2}).contiguous();
    }

    Tensor Tensor::max_pool2d(int kernel_size, int stride, int padding) const {
        assert_float32_same_device(*this, "max_pool2d");
        LFS_ASSERT_MSG(shape_.rank() == 4,
                       std::format("max_pool2d input must be 4D [N,C,H,W] "
                                   "(input_shape={}, input_rank={})",
                                   shape_.str(), shape_.rank()));
        LFS_ASSERT_MSG(kernel_size > 0,
                       std::format("max_pool2d kernel size must be positive "
                                   "(kernel_size={})",
                                   kernel_size));

        if (stride <= 0)
            stride = kernel_size;

        const int N = static_cast<int>(shape_[0]);
        const int C = static_cast<int>(shape_[1]);
        const int H_in = static_cast<int>(shape_[2]);
        const int W_in = static_cast<int>(shape_[3]);

        const int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
        const int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

        LFS_ASSERT_MSG(H_out > 0 && W_out > 0,
                       std::format("max_pool2d output dimensions must be positive "
                                   "(output_height={}, output_width={}, input_shape={}, "
                                   "kernel_size={}, stride={}, padding={})",
                                   H_out, W_out, shape_.str(), kernel_size, stride, padding));

        const Tensor& input_cont = is_contiguous() ? *this : contiguous();
        auto output = empty({static_cast<size_t>(N), static_cast<size_t>(C),
                             static_cast<size_t>(H_out), static_cast<size_t>(W_out)},
                            device_, dtype_);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_max_pool2d(input_cont.ptr<float>(), output.ptr<float>(),
                                          N, C, H_in, W_in, H_out, W_out,
                                          kernel_size, stride, padding, stream());
        } else {
            cpu_max_pool2d(input_cont.ptr<float>(), output.ptr<float>(),
                           N, C, H_in, W_in, H_out, W_out,
                           kernel_size, stride, padding);
        }

        return output;
    }

    Tensor Tensor::adaptive_avg_pool2d(int output_h, int output_w) const {
        assert_float32_same_device(*this, "adaptive_avg_pool2d");
        LFS_ASSERT_MSG(shape_.rank() == 4,
                       std::format("adaptive_avg_pool2d input must be 4D [N,C,H,W] "
                                   "(input_shape={}, input_rank={})",
                                   shape_.str(), shape_.rank()));
        LFS_ASSERT_MSG(output_h > 0 && output_w > 0,
                       std::format("adaptive_avg_pool2d output dimensions must be positive "
                                   "(output_height={}, output_width={})",
                                   output_h, output_w));

        const int N = static_cast<int>(shape_[0]);
        const int C = static_cast<int>(shape_[1]);
        const int H_in = static_cast<int>(shape_[2]);
        const int W_in = static_cast<int>(shape_[3]);

        const Tensor& input_cont = is_contiguous() ? *this : contiguous();
        auto output = empty({static_cast<size_t>(N), static_cast<size_t>(C),
                             static_cast<size_t>(output_h), static_cast<size_t>(output_w)},
                            device_, dtype_);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_adaptive_avg_pool2d(input_cont.ptr<float>(), output.ptr<float>(),
                                                   N, C, H_in, W_in, output_h, output_w, stream());
        } else {
            cpu_adaptive_avg_pool2d(input_cont.ptr<float>(), output.ptr<float>(),
                                    N, C, H_in, W_in, output_h, output_w);
        }

        return output;
    }

    Tensor Tensor::linear(const Tensor& weight) const {
        return linear(weight, Tensor{});
    }

    Tensor Tensor::linear(const Tensor& weight, const Tensor& bias) const {
        assert_float32_same_device(*this, "linear", {{"weight", &weight}});
        if (bias.is_valid()) {
            assert_float32_same_device(*this, "linear", {{"bias", &bias}});
        }
        LFS_ASSERT_MSG(weight.shape_.rank() == 2,
                       std::format("linear weight must be 2D [out_features,in_features] "
                                   "(weight_shape={}, weight_rank={})",
                                   weight.shape_.str(), weight.shape_.rank()));
        LFS_ASSERT_MSG(shape_.rank() >= 1,
                       std::format("linear input must have at least one dimension "
                                   "(input_shape={}, input_rank={})",
                                   shape_.str(), shape_.rank()));
        LFS_ASSERT_MSG(shape_[shape_.rank() - 1] == weight.shape_[1],
                       std::format("linear input features must match weight input features "
                                   "(input_features={}, weight_input_features={}, "
                                   "input_shape={}, weight_shape={})",
                                   shape_[shape_.rank() - 1], weight.shape_[1],
                                   shape_.str(), weight.shape_.str()));

        const size_t in_features = weight.shape_[1];
        const size_t out_features = weight.shape_[0];

        if (bias.is_valid()) {
            assert_bias_shape(bias, out_features, "linear");
        }

        size_t batch_size = 1;
        for (size_t i = 0; i < shape_.rank() - 1; ++i) {
            batch_size *= shape_[i];
        }

        const Tensor& input_cont = is_contiguous() ? *this : contiguous();
        const Tensor& weight_cont = weight.is_contiguous() ? weight : weight.contiguous();

        // GPU path: Output[batch, out] = Input[batch, in] @ Weight^T[in, out]
        if (device_ == Device::CUDA) {
            auto output = empty({batch_size, out_features}, Device::CUDA, dtype_);

            tensor_ops::launch_sgemm_tn(input_cont.ptr<float>(), weight_cont.ptr<float>(),
                                        output.ptr<float>(), batch_size, out_features, in_features,
                                        stream());

            if (bias.is_valid()) {
                const int total = static_cast<int>(batch_size * out_features);
                tensor_ops::launch_bias_add(output.ptr<float>(), bias.ptr<float>(),
                                            output.ptr<float>(), total,
                                            static_cast<int>(out_features), 1,
                                            stream());
            }

            std::vector<int> output_shape;
            for (size_t i = 0; i < shape_.rank() - 1; ++i) {
                output_shape.push_back(static_cast<int>(shape_[i]));
            }
            output_shape.push_back(static_cast<int>(out_features));
            return output.reshape(output_shape);
        }

        // CPU fallback
        auto input_2d = input_cont.reshape({static_cast<int>(batch_size), static_cast<int>(in_features)});
        auto weight_t = weight_cont.transpose(0, 1).contiguous();
        auto output_2d = input_2d.matmul(weight_t);

        if (bias.is_valid()) {
            output_2d = output_2d + bias;
        }

        std::vector<int> output_shape;
        for (size_t i = 0; i < shape_.rank() - 1; ++i) {
            output_shape.push_back(static_cast<int>(shape_[i]));
        }
        output_shape.push_back(static_cast<int>(out_features));

        return output_2d.reshape(output_shape);
    }

    Tensor Tensor::conv1x1_bias_relu(const Tensor& weight, const Tensor& bias) const {
        assert_float32_same_device(*this, "conv1x1_bias_relu",
                                   {{"weight", &weight}, {"bias", &bias}});
        LFS_ASSERT_MSG(shape_.rank() == 4,
                       std::format("conv1x1_bias_relu input must be 4D [N,C,H,W] "
                                   "(input_shape={}, input_rank={})",
                                   shape_.str(), shape_.rank()));
        LFS_ASSERT_MSG(weight.shape_.rank() == 2,
                       std::format("conv1x1_bias_relu weight must be 2D [C_out,C_in] "
                                   "(weight_shape={}, weight_rank={})",
                                   weight.shape_.str(), weight.shape_.rank()));
        LFS_ASSERT_MSG(shape_[1] == weight.shape_[1],
                       std::format("conv1x1_bias_relu input channels must match weight input channels "
                                   "(input_channels={}, weight_input_channels={}, "
                                   "input_shape={}, weight_shape={})",
                                   shape_[1], weight.shape_[1], shape_.str(), weight.shape_.str()));

        const size_t N = shape_[0];
        const size_t C_in = shape_[1];
        const size_t H = shape_[2];
        const size_t W = shape_[3];
        const size_t C_out = weight.shape_[0];
        const size_t S = H * W;

        assert_bias_shape(bias, C_out, "conv1x1_bias_relu");

        const Tensor& input_cont = is_contiguous() ? *this : contiguous();
        const Tensor& weight_cont = weight.is_contiguous() ? weight : weight.contiguous();

        if (device_ == Device::CUDA && N == 1) {
            auto output = empty({N, C_out, H, W}, Device::CUDA, dtype_);

            tensor_ops::launch_sgemm(weight_cont.ptr<float>(), input_cont.ptr<float>(),
                                     output.ptr<float>(), C_out, S, C_in, stream());

            const int total = static_cast<int>(N * C_out * H * W);
            tensor_ops::launch_bias_relu(output.ptr<float>(), bias.ptr<float>(),
                                         output.ptr<float>(), total,
                                         static_cast<int>(C_out), static_cast<int>(S),
                                         stream());
            return output;
        }

        // Fallback: use separate operations
        return conv1x1(weight, bias).relu();
    }

    Tensor Tensor::linear_bias_relu(const Tensor& weight, const Tensor& bias) const {
        assert_float32_same_device(*this, "linear_bias_relu",
                                   {{"weight", &weight}, {"bias", &bias}});
        LFS_ASSERT_MSG(weight.shape_.rank() == 2,
                       std::format("linear_bias_relu weight must be 2D "
                                   "[out_features,in_features] (weight_shape={}, weight_rank={})",
                                   weight.shape_.str(), weight.shape_.rank()));
        LFS_ASSERT_MSG(shape_.rank() >= 1,
                       std::format("linear_bias_relu input must have at least one dimension "
                                   "(input_shape={}, input_rank={})",
                                   shape_.str(), shape_.rank()));
        LFS_ASSERT_MSG(shape_[shape_.rank() - 1] == weight.shape_[1],
                       std::format("linear_bias_relu input features must match weight input features "
                                   "(input_features={}, weight_input_features={}, "
                                   "input_shape={}, weight_shape={})",
                                   shape_[shape_.rank() - 1], weight.shape_[1],
                                   shape_.str(), weight.shape_.str()));

        const size_t in_features = weight.shape_[1];
        const size_t out_features = weight.shape_[0];

        assert_bias_shape(bias, out_features, "linear_bias_relu");

        size_t batch_size = 1;
        for (size_t i = 0; i < shape_.rank() - 1; ++i) {
            batch_size *= shape_[i];
        }

        const Tensor& input_cont = is_contiguous() ? *this : contiguous();
        const Tensor& weight_cont = weight.is_contiguous() ? weight : weight.contiguous();

        if (device_ == Device::CUDA) {
            auto output = empty({batch_size, out_features}, Device::CUDA, dtype_);

            tensor_ops::launch_sgemm_tn(input_cont.ptr<float>(), weight_cont.ptr<float>(),
                                        output.ptr<float>(), batch_size, out_features, in_features,
                                        stream());

            const int total = static_cast<int>(batch_size * out_features);
            tensor_ops::launch_bias_relu(output.ptr<float>(), bias.ptr<float>(),
                                         output.ptr<float>(), total,
                                         static_cast<int>(out_features), 1,
                                         stream());

            std::vector<int> output_shape;
            for (size_t i = 0; i < shape_.rank() - 1; ++i) {
                output_shape.push_back(static_cast<int>(shape_[i]));
            }
            output_shape.push_back(static_cast<int>(out_features));
            return output.reshape(output_shape);
        }

        // Fallback: use separate operations
        return linear(weight, bias).relu();
    }

    // ========== _out variants that write into pre-allocated output tensors ==========

    void Tensor::conv1x1_bias_out(const Tensor& weight, const Tensor& bias, Tensor& output) const {
        assert_float32_same_device(*this, "conv1x1_bias_out",
                                   {{"weight", &weight}, {"bias", &bias}, {"output", &output}});
        LFS_ASSERT_MSG(shape_.rank() == 4,
                       std::format("conv1x1_bias_out input must be 4D [N,C,H,W] "
                                   "(input_shape={}, input_rank={})",
                                   shape_.str(), shape_.rank()));
        LFS_ASSERT_MSG(weight.shape_.rank() == 2,
                       std::format("conv1x1_bias_out weight must be 2D [C_out,C_in] "
                                   "(weight_shape={}, weight_rank={})",
                                   weight.shape_.str(), weight.shape_.rank()));
        LFS_ASSERT_MSG(shape_[1] == weight.shape_[1],
                       std::format("conv1x1_bias_out input channels must match weight input channels "
                                   "(input_channels={}, weight_input_channels={}, "
                                   "input_shape={}, weight_shape={})",
                                   shape_[1], weight.shape_[1], shape_.str(), weight.shape_.str()));
        LFS_ASSERT_MSG(device_ == Device::CUDA,
                       std::format("conv1x1_bias_out requires CUDA tensors "
                                   "(input_device={}, input_shape={})",
                                   device_name(device_), shape_.str()));

        const size_t N = shape_[0];
        const size_t C_in = shape_[1];
        const size_t H = shape_[2];
        const size_t W = shape_[3];
        const size_t C_out = weight.shape_[0];
        const size_t S = H * W;

        LFS_ASSERT_MSG(N == 1,
                       std::format("conv1x1_bias_out currently requires batch size 1 "
                                   "because its output kernel records one image "
                                   "(batch_size={}, input_shape={})",
                                   N, shape_.str()));
        LFS_ASSERT_MSG(output.shape().rank() == 4,
                       std::format("conv1x1_bias_out output must be rank 4 "
                                   "(output_shape={}, output_rank={})",
                                   output.shape().str(), output.shape().rank()));
        LFS_ASSERT_MSG(output.shape()[0] == N && output.shape()[1] == C_out &&
                           output.shape()[2] == H && output.shape()[3] == W,
                       std::format("conv1x1_bias_out output shape must match [N,C_out,H,W] "
                                   "(output_shape={}, expected_shape=[{},{},{},{}])",
                                   output.shape().str(), N, C_out, H, W));
        assert_bias_shape(bias, C_out, "conv1x1_bias_out");

        const Tensor& input_cont = is_contiguous() ? *this : contiguous();
        const Tensor& weight_cont = weight.is_contiguous() ? weight : weight.contiguous();

        tensor_ops::launch_sgemm(weight_cont.ptr<float>(), input_cont.ptr<float>(),
                                 output.ptr<float>(), C_out, S, C_in, stream());

        const int total = static_cast<int>(N * C_out * H * W);
        tensor_ops::launch_bias_add(output.ptr<float>(), bias.ptr<float>(),
                                    output.ptr<float>(), total,
                                    static_cast<int>(C_out), static_cast<int>(S),
                                    stream());
    }

    void Tensor::relu_out(Tensor& output) const {
        assert_float32_same_device(*this, "relu_out", {{"output", &output}});
        LFS_ASSERT_MSG(numel() == output.numel(),
                       std::format("relu_out output element count must match input "
                                   "(input_numel={}, output_numel={}, input_shape={}, output_shape={})",
                                   numel(), output.numel(), shape_.str(), output.shape().str()));
        LFS_ASSERT_MSG(device_ == Device::CUDA,
                       std::format("relu_out requires CUDA tensors "
                                   "(input_device={}, input_shape={})",
                                   device_name(device_), shape_.str()));

        const Tensor& input_cont = is_contiguous() ? *this : contiguous();
        tensor_ops::launch_relu(input_cont.ptr<float>(), output.ptr<float>(),
                                static_cast<int>(numel()), stream());
    }

    void Tensor::conv1x1_bias_relu_out(const Tensor& weight, const Tensor& bias, Tensor& output) const {
        assert_float32_same_device(*this, "conv1x1_bias_relu_out",
                                   {{"weight", &weight}, {"bias", &bias}, {"output", &output}});
        LFS_ASSERT_MSG(shape_.rank() == 4,
                       std::format("conv1x1_bias_relu_out input must be 4D [N,C,H,W] "
                                   "(input_shape={}, input_rank={})",
                                   shape_.str(), shape_.rank()));
        LFS_ASSERT_MSG(weight.shape_.rank() == 2,
                       std::format("conv1x1_bias_relu_out weight must be 2D [C_out,C_in] "
                                   "(weight_shape={}, weight_rank={})",
                                   weight.shape_.str(), weight.shape_.rank()));
        LFS_ASSERT_MSG(shape_[1] == weight.shape_[1],
                       std::format("conv1x1_bias_relu_out input channels must match weight input channels "
                                   "(input_channels={}, weight_input_channels={}, "
                                   "input_shape={}, weight_shape={})",
                                   shape_[1], weight.shape_[1], shape_.str(), weight.shape_.str()));
        LFS_ASSERT_MSG(device_ == Device::CUDA,
                       std::format("conv1x1_bias_relu_out requires CUDA tensors "
                                   "(input_device={}, input_shape={})",
                                   device_name(device_), shape_.str()));

        const size_t N = shape_[0];
        const size_t C_in = shape_[1];
        const size_t H = shape_[2];
        const size_t W = shape_[3];
        const size_t C_out = weight.shape_[0];
        const size_t S = H * W;

        LFS_ASSERT_MSG(N == 1,
                       std::format("conv1x1_bias_relu_out currently requires batch size 1 "
                                   "because its fused output kernel records one image "
                                   "(batch_size={}, input_shape={})",
                                   N, shape_.str()));
        LFS_ASSERT_MSG(output.shape().rank() == 4,
                       std::format("conv1x1_bias_relu_out output must be rank 4 "
                                   "(output_shape={}, output_rank={})",
                                   output.shape().str(), output.shape().rank()));
        LFS_ASSERT_MSG(output.shape()[0] == N && output.shape()[1] == C_out &&
                           output.shape()[2] == H && output.shape()[3] == W,
                       std::format("conv1x1_bias_relu_out output shape must match "
                                   "[N,C_out,H,W] (output_shape={}, expected_shape=[{},{},{},{}])",
                                   output.shape().str(), N, C_out, H, W));
        assert_bias_shape(bias, C_out, "conv1x1_bias_relu_out");

        const Tensor& input_cont = is_contiguous() ? *this : contiguous();
        const Tensor& weight_cont = weight.is_contiguous() ? weight : weight.contiguous();

        const size_t output_size = C_out * S;
        if (output_size >= 500000) {
            // Large outputs: use fused kernel to save memory bandwidth
            tensor_ops::launch_sgemm_bias_relu(weight_cont.ptr<float>(), input_cont.ptr<float>(),
                                               bias.ptr<float>(), output.ptr<float>(),
                                               C_out, S, C_in, stream());
        } else {
            // Small outputs: separate kernels have less overhead
            tensor_ops::launch_sgemm(weight_cont.ptr<float>(), input_cont.ptr<float>(),
                                     output.ptr<float>(), C_out, S, C_in, stream());
            const int total = static_cast<int>(N * C_out * H * W);
            tensor_ops::launch_bias_relu(output.ptr<float>(), bias.ptr<float>(),
                                         output.ptr<float>(), total,
                                         static_cast<int>(C_out), static_cast<int>(S),
                                         stream());
        }
    }

    void Tensor::max_pool2d_out(int kernel_size, int stride, int padding, Tensor& output) const {
        assert_float32_same_device(*this, "max_pool2d_out", {{"output", &output}});
        LFS_ASSERT_MSG(shape_.rank() == 4,
                       std::format("max_pool2d_out input must be 4D [N,C,H,W] "
                                   "(input_shape={}, input_rank={})",
                                   shape_.str(), shape_.rank()));
        LFS_ASSERT_MSG(device_ == Device::CUDA,
                       std::format("max_pool2d_out requires CUDA tensors "
                                   "(input_device={}, input_shape={})",
                                   device_name(device_), shape_.str()));
        LFS_ASSERT_MSG(kernel_size > 0,
                       std::format("max_pool2d_out kernel size must be positive "
                                   "(kernel_size={})",
                                   kernel_size));

        if (stride <= 0)
            stride = kernel_size;

        const int N = static_cast<int>(shape_[0]);
        const int C = static_cast<int>(shape_[1]);
        const int H_in = static_cast<int>(shape_[2]);
        const int W_in = static_cast<int>(shape_[3]);
        const int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
        const int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

        LFS_ASSERT_MSG(H_out > 0 && W_out > 0,
                       std::format("max_pool2d_out computed output dimensions must be positive "
                                   "(output_height={}, output_width={}, input_shape={}, "
                                   "kernel_size={}, stride={}, padding={})",
                                   H_out, W_out, shape_.str(), kernel_size, stride, padding));
        LFS_ASSERT_MSG(output.shape().rank() == 4,
                       std::format("max_pool2d_out output must be rank 4 "
                                   "(output_shape={}, output_rank={})",
                                   output.shape().str(), output.shape().rank()));
        LFS_ASSERT_MSG(output.shape()[0] == static_cast<size_t>(N) &&
                           output.shape()[1] == static_cast<size_t>(C) &&
                           output.shape()[2] == static_cast<size_t>(H_out) &&
                           output.shape()[3] == static_cast<size_t>(W_out),
                       std::format("max_pool2d_out output shape must match the computed shape "
                                   "(output_shape={}, expected_shape=[{},{},{},{}])",
                                   output.shape().str(), N, C, H_out, W_out));

        const Tensor& input_cont = is_contiguous() ? *this : contiguous();
        tensor_ops::launch_max_pool2d(input_cont.ptr<float>(), output.ptr<float>(),
                                      N, C, H_in, W_in, H_out, W_out,
                                      kernel_size, stride, padding, stream());
    }

    void Tensor::adaptive_avg_pool2d_out(int output_h, int output_w, Tensor& output) const {
        assert_float32_same_device(*this, "adaptive_avg_pool2d_out", {{"output", &output}});
        LFS_ASSERT_MSG(shape_.rank() == 4,
                       std::format("adaptive_avg_pool2d_out input must be 4D [N,C,H,W] "
                                   "(input_shape={}, input_rank={})",
                                   shape_.str(), shape_.rank()));
        LFS_ASSERT_MSG(device_ == Device::CUDA,
                       std::format("adaptive_avg_pool2d_out requires CUDA tensors "
                                   "(input_device={}, input_shape={})",
                                   device_name(device_), shape_.str()));
        LFS_ASSERT_MSG(output_h > 0 && output_w > 0,
                       std::format("adaptive_avg_pool2d_out output dimensions must be positive "
                                   "(output_height={}, output_width={})",
                                   output_h, output_w));

        const int N = static_cast<int>(shape_[0]);
        const int C = static_cast<int>(shape_[1]);
        const int H_in = static_cast<int>(shape_[2]);
        const int W_in = static_cast<int>(shape_[3]);

        LFS_ASSERT_MSG(output.shape().rank() == 4,
                       std::format("adaptive_avg_pool2d_out output must be rank 4 "
                                   "(output_shape={}, output_rank={})",
                                   output.shape().str(), output.shape().rank()));
        LFS_ASSERT_MSG(output.shape()[0] == static_cast<size_t>(N) &&
                           output.shape()[1] == static_cast<size_t>(C) &&
                           output.shape()[2] == static_cast<size_t>(output_h) &&
                           output.shape()[3] == static_cast<size_t>(output_w),
                       std::format("adaptive_avg_pool2d_out output shape must match the requested shape "
                                   "(output_shape={}, expected_shape=[{},{},{},{}])",
                                   output.shape().str(), N, C, output_h, output_w));

        const Tensor& input_cont = is_contiguous() ? *this : contiguous();
        tensor_ops::launch_adaptive_avg_pool2d(input_cont.ptr<float>(), output.ptr<float>(),
                                               N, C, H_in, W_in, output_h, output_w, stream());
    }

    void Tensor::linear_bias_relu_out(const Tensor& weight, const Tensor& bias, Tensor& output) const {
        assert_float32_same_device(*this, "linear_bias_relu_out",
                                   {{"weight", &weight}, {"bias", &bias}, {"output", &output}});
        LFS_ASSERT_MSG(shape_.rank() >= 1,
                       std::format("linear_bias_relu_out input must have at least one dimension "
                                   "(input_shape={}, input_rank={})",
                                   shape_.str(), shape_.rank()));
        LFS_ASSERT_MSG(weight.shape().rank() == 2,
                       std::format("linear_bias_relu_out weight must be 2D "
                                   "[out_features,in_features] (weight_shape={}, weight_rank={})",
                                   weight.shape().str(), weight.shape().rank()));
        LFS_ASSERT_MSG(shape_[shape_.rank() - 1] == weight.shape()[1],
                       std::format("linear_bias_relu_out input features must match weight input features "
                                   "(input_features={}, weight_input_features={}, "
                                   "input_shape={}, weight_shape={})",
                                   shape_[shape_.rank() - 1], weight.shape()[1],
                                   shape_.str(), weight.shape().str()));
        LFS_ASSERT_MSG(device_ == Device::CUDA,
                       std::format("linear_bias_relu_out requires CUDA tensors "
                                   "(input_device={}, input_shape={})",
                                   device_name(device_), shape_.str()));

        const size_t in_features = weight.shape_[1];
        const size_t out_features = weight.shape_[0];

        assert_bias_shape(bias, out_features, "linear_bias_relu_out");

        size_t batch_size = 1;
        for (size_t i = 0; i < shape_.rank() - 1; ++i) {
            batch_size *= shape_[i];
        }

        LFS_ASSERT_MSG(output.numel() == batch_size * out_features,
                       std::format("linear_bias_relu_out output element count must match "
                                   "batch_size * out_features "
                                   "(output_numel={}, expected_numel={}, batch_size={}, "
                                   "out_features={}, output_shape={})",
                                   output.numel(), batch_size * out_features, batch_size,
                                   out_features, output.shape().str()));

        const Tensor& input_cont = is_contiguous() ? *this : contiguous();
        const Tensor& weight_cont = weight.is_contiguous() ? weight : weight.contiguous();

        tensor_ops::launch_sgemm_tn(input_cont.ptr<float>(), weight_cont.ptr<float>(),
                                    output.ptr<float>(), batch_size, out_features, in_features,
                                    stream());

        const int total = static_cast<int>(batch_size * out_features);
        tensor_ops::launch_bias_relu(output.ptr<float>(), bias.ptr<float>(),
                                     output.ptr<float>(), total,
                                     static_cast<int>(out_features), 1,
                                     stream());
    }

    void Tensor::linear_out(const Tensor& weight, const Tensor& bias, Tensor& output) const {
        assert_float32_same_device(*this, "linear_out",
                                   {{"weight", &weight}, {"output", &output}});
        if (bias.is_valid()) {
            assert_float32_same_device(*this, "linear_out", {{"bias", &bias}});
        }
        LFS_ASSERT_MSG(shape_.rank() >= 1,
                       std::format("linear_out input must have at least one dimension "
                                   "(input_shape={}, input_rank={})",
                                   shape_.str(), shape_.rank()));
        LFS_ASSERT_MSG(weight.shape().rank() == 2,
                       std::format("linear_out weight must be 2D [out_features,in_features] "
                                   "(weight_shape={}, weight_rank={})",
                                   weight.shape().str(), weight.shape().rank()));
        LFS_ASSERT_MSG(shape_[shape_.rank() - 1] == weight.shape()[1],
                       std::format("linear_out input features must match weight input features "
                                   "(input_features={}, weight_input_features={}, "
                                   "input_shape={}, weight_shape={})",
                                   shape_[shape_.rank() - 1], weight.shape()[1],
                                   shape_.str(), weight.shape().str()));
        LFS_ASSERT_MSG(device_ == Device::CUDA,
                       std::format("linear_out requires CUDA tensors "
                                   "(input_device={}, input_shape={})",
                                   device_name(device_), shape_.str()));

        const size_t in_features = weight.shape_[1];
        const size_t out_features = weight.shape_[0];

        if (bias.is_valid()) {
            assert_bias_shape(bias, out_features, "linear_out");
        }

        size_t batch_size = 1;
        for (size_t i = 0; i < shape_.rank() - 1; ++i) {
            batch_size *= shape_[i];
        }

        LFS_ASSERT_MSG(output.numel() == batch_size * out_features,
                       std::format("linear_out output element count must match "
                                   "batch_size * out_features "
                                   "(output_numel={}, expected_numel={}, batch_size={}, "
                                   "out_features={}, output_shape={})",
                                   output.numel(), batch_size * out_features, batch_size,
                                   out_features, output.shape().str()));

        const Tensor& input_cont = is_contiguous() ? *this : contiguous();
        const Tensor& weight_cont = weight.is_contiguous() ? weight : weight.contiguous();

        tensor_ops::launch_sgemm_tn(input_cont.ptr<float>(), weight_cont.ptr<float>(),
                                    output.ptr<float>(), batch_size, out_features, in_features,
                                    stream());

        if (bias.is_valid()) {
            const int total = static_cast<int>(batch_size * out_features);
            tensor_ops::launch_bias_add(output.ptr<float>(), bias.ptr<float>(),
                                        output.ptr<float>(), total,
                                        static_cast<int>(out_features), 1,
                                        stream());
        }
    }

} // namespace lfs::core
