/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_nn_ops.hpp"
#include "core/logger.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"

#include <cassert>

namespace lfs::core {

    namespace {

        void assert_float32_same_device(const Tensor& reference,
                                        const std::string_view operation,
                                        const std::initializer_list<const Tensor*> tensors = {}) {
            LFS_ASSERT_MSG(reference.is_valid(), std::string(operation) + ": invalid input tensor");
            LFS_ASSERT_MSG(reference.dtype() == DataType::Float32,
                           std::string(operation) + ": input must be Float32");
            for (const Tensor* tensor : tensors) {
                LFS_ASSERT_MSG(tensor != nullptr && tensor->is_valid(),
                               std::string(operation) + ": invalid operand tensor");
                LFS_ASSERT_MSG(tensor->dtype() == DataType::Float32,
                               std::string(operation) + ": all tensors must be Float32");
                LFS_ASSERT_MSG(tensor->device() == reference.device(),
                               std::string(operation) + ": all tensors must be on the same device");
            }
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
        assert_float32_same_device(*this, "conv1x1", {&weight});
        if (bias.is_valid()) {
            assert_float32_same_device(*this, "conv1x1", {&bias});
        }
        LFS_ASSERT(is_valid() && "conv1x1: invalid input tensor");
        LFS_ASSERT(weight.is_valid() && "conv1x1: invalid weight tensor");
        LFS_ASSERT(shape_.rank() == 4 && "conv1x1: input must be 4D [N,C,H,W]");
        LFS_ASSERT(weight.shape_.rank() == 2 && "conv1x1: weight must be 2D [C_out,C_in]");
        LFS_ASSERT(shape_[1] == weight.shape_[1] && "conv1x1: channel mismatch");
        LFS_ASSERT(device_ == weight.device_ && "conv1x1: tensors must be on same device");

        const size_t N = shape_[0];
        const size_t C_in = shape_[1];
        const size_t H = shape_[2];
        const size_t W = shape_[3];
        const size_t C_out = weight.shape_[0];
        const size_t S = H * W;

        const Tensor& input_cont = is_contiguous() ? *this : contiguous();
        const Tensor& weight_cont = weight.is_contiguous() ? weight : weight.contiguous();

        // GPU path: Output[C_out, S] = Weight[C_out, C_in] @ Input[C_in, S]
        if (device_ == Device::CUDA && N == 1) {
            auto output = empty({N, C_out, H, W}, Device::CUDA, dtype_);

            tensor_ops::launch_sgemm(weight_cont.ptr<float>(), input_cont.ptr<float>(),
                                     output.ptr<float>(), C_out, S, C_in, stream());

            if (bias.is_valid()) {
                LFS_ASSERT(bias.shape_.rank() == 1 && bias.shape_[0] == C_out);
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
                LFS_ASSERT(bias.shape_.rank() == 1 && bias.shape_[0] == C_out);
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
            LFS_ASSERT(bias.shape_.rank() == 1 && bias.shape_[0] == C_out);
            output_2d = output_2d + bias;
        }

        auto output_nhwc = output_2d.reshape({static_cast<int>(N), static_cast<int>(H),
                                              static_cast<int>(W), static_cast<int>(C_out)});
        return output_nhwc.permute({0, 3, 1, 2}).contiguous();
    }

    Tensor Tensor::max_pool2d(int kernel_size, int stride, int padding) const {
        assert_float32_same_device(*this, "max_pool2d");
        LFS_ASSERT(is_valid() && "max_pool2d: invalid input tensor");
        LFS_ASSERT(shape_.rank() == 4 && "max_pool2d: input must be 4D [N,C,H,W]");
        LFS_ASSERT(kernel_size > 0 && "max_pool2d: kernel_size must be positive");

        if (stride <= 0)
            stride = kernel_size;

        const int N = static_cast<int>(shape_[0]);
        const int C = static_cast<int>(shape_[1]);
        const int H_in = static_cast<int>(shape_[2]);
        const int W_in = static_cast<int>(shape_[3]);

        const int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
        const int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

        LFS_ASSERT(H_out > 0 && W_out > 0 && "max_pool2d: output dimensions must be positive");

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
        LFS_ASSERT(is_valid() && "adaptive_avg_pool2d: invalid input tensor");
        LFS_ASSERT(shape_.rank() == 4 && "adaptive_avg_pool2d: input must be 4D [N,C,H,W]");
        LFS_ASSERT(output_h > 0 && output_w > 0 && "adaptive_avg_pool2d: output dims must be positive");

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
        assert_float32_same_device(*this, "linear", {&weight});
        if (bias.is_valid()) {
            assert_float32_same_device(*this, "linear", {&bias});
        }
        LFS_ASSERT(is_valid() && "linear: invalid input tensor");
        LFS_ASSERT(weight.is_valid() && "linear: invalid weight tensor");
        LFS_ASSERT(weight.shape_.rank() == 2 && "linear: weight must be 2D [out_features, in_features]");
        LFS_ASSERT(shape_.rank() >= 1 && "linear: input must have at least 1 dimension");
        LFS_ASSERT(shape_[shape_.rank() - 1] == weight.shape_[1] && "linear: feature dimension mismatch");
        LFS_ASSERT(device_ == weight.device_ && "linear: tensors must be on same device");

        const size_t in_features = weight.shape_[1];
        const size_t out_features = weight.shape_[0];

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
                LFS_ASSERT(bias.shape_.rank() == 1 && bias.shape_[0] == out_features);
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
            LFS_ASSERT(bias.shape_.rank() == 1 && bias.shape_[0] == out_features);
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
        assert_float32_same_device(*this, "conv1x1_bias_relu", {&weight, &bias});
        LFS_ASSERT(is_valid() && "conv1x1_bias_relu: invalid input tensor");
        LFS_ASSERT(weight.is_valid() && "conv1x1_bias_relu: invalid weight tensor");
        LFS_ASSERT(bias.is_valid() && "conv1x1_bias_relu: invalid bias tensor");
        LFS_ASSERT(shape_.rank() == 4 && "conv1x1_bias_relu: input must be 4D [N,C,H,W]");
        LFS_ASSERT(weight.shape_.rank() == 2 && "conv1x1_bias_relu: weight must be 2D [C_out,C_in]");
        LFS_ASSERT(shape_[1] == weight.shape_[1] && "conv1x1_bias_relu: channel mismatch");
        LFS_ASSERT(device_ == weight.device_ && "conv1x1_bias_relu: tensors must be on same device");

        const size_t N = shape_[0];
        const size_t C_in = shape_[1];
        const size_t H = shape_[2];
        const size_t W = shape_[3];
        const size_t C_out = weight.shape_[0];
        const size_t S = H * W;

        LFS_ASSERT(bias.shape_.rank() == 1 && bias.shape_[0] == C_out && "conv1x1_bias_relu: bias mismatch");

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
        assert_float32_same_device(*this, "linear_bias_relu", {&weight, &bias});
        LFS_ASSERT(is_valid() && "linear_bias_relu: invalid input tensor");
        LFS_ASSERT(weight.is_valid() && "linear_bias_relu: invalid weight tensor");
        LFS_ASSERT(bias.is_valid() && "linear_bias_relu: invalid bias tensor");
        LFS_ASSERT(weight.shape_.rank() == 2 && "linear_bias_relu: weight must be 2D");
        LFS_ASSERT(shape_.rank() >= 1 && "linear_bias_relu: input must have at least 1 dimension");
        LFS_ASSERT(shape_[shape_.rank() - 1] == weight.shape_[1] && "linear_bias_relu: feature dimension mismatch");
        LFS_ASSERT(device_ == weight.device_ && "linear_bias_relu: tensors must be on same device");

        const size_t in_features = weight.shape_[1];
        const size_t out_features = weight.shape_[0];

        LFS_ASSERT(bias.shape_.rank() == 1 && bias.shape_[0] == out_features && "linear_bias_relu: bias mismatch");

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
        assert_float32_same_device(*this, "conv1x1_bias_out", {&weight, &bias, &output});
        LFS_ASSERT(is_valid() && "conv1x1_bias_out: invalid input tensor");
        LFS_ASSERT(weight.is_valid() && "conv1x1_bias_out: invalid weight tensor");
        LFS_ASSERT(bias.is_valid() && "conv1x1_bias_out: invalid bias tensor");
        LFS_ASSERT(output.is_valid() && "conv1x1_bias_out: invalid output tensor");
        LFS_ASSERT(shape_.rank() == 4 && "conv1x1_bias_out: input must be 4D [N,C,H,W]");
        LFS_ASSERT(weight.shape_.rank() == 2 && "conv1x1_bias_out: weight must be 2D [C_out,C_in]");
        LFS_ASSERT(shape_[1] == weight.shape_[1] && "conv1x1_bias_out: channel mismatch");
        LFS_ASSERT(device_ == Device::CUDA && "conv1x1_bias_out: CUDA only");

        const size_t N = shape_[0];
        const size_t C_in = shape_[1];
        const size_t H = shape_[2];
        const size_t W = shape_[3];
        const size_t C_out = weight.shape_[0];
        const size_t S = H * W;

        LFS_ASSERT(output.shape()[0] == N && output.shape()[1] == C_out &&
                   output.shape()[2] == H && output.shape()[3] == W &&
                   "conv1x1_bias_out: output shape mismatch");
        LFS_ASSERT(bias.shape_[0] == C_out && "conv1x1_bias_out: bias mismatch");

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
        assert_float32_same_device(*this, "relu_out", {&output});
        LFS_ASSERT(is_valid() && "relu_out: invalid input tensor");
        LFS_ASSERT(output.is_valid() && "relu_out: invalid output tensor");
        LFS_ASSERT(numel() == output.numel() && "relu_out: size mismatch");
        LFS_ASSERT(device_ == Device::CUDA && "relu_out: CUDA only");

        const Tensor& input_cont = is_contiguous() ? *this : contiguous();
        tensor_ops::launch_relu(input_cont.ptr<float>(), output.ptr<float>(),
                                static_cast<int>(numel()), stream());
    }

    void Tensor::conv1x1_bias_relu_out(const Tensor& weight, const Tensor& bias, Tensor& output) const {
        assert_float32_same_device(*this, "conv1x1_bias_relu_out", {&weight, &bias, &output});
        LFS_ASSERT(is_valid() && "conv1x1_bias_relu_out: invalid input tensor");
        LFS_ASSERT(weight.is_valid() && "conv1x1_bias_relu_out: invalid weight tensor");
        LFS_ASSERT(bias.is_valid() && "conv1x1_bias_relu_out: invalid bias tensor");
        LFS_ASSERT(output.is_valid() && "conv1x1_bias_relu_out: invalid output tensor");
        LFS_ASSERT(shape_.rank() == 4 && "conv1x1_bias_relu_out: input must be 4D [N,C,H,W]");
        LFS_ASSERT(weight.shape_.rank() == 2 && "conv1x1_bias_relu_out: weight must be 2D [C_out,C_in]");
        LFS_ASSERT(shape_[1] == weight.shape_[1] && "conv1x1_bias_relu_out: channel mismatch");
        LFS_ASSERT(device_ == Device::CUDA && "conv1x1_bias_relu_out: CUDA only");

        const size_t N = shape_[0];
        const size_t C_in = shape_[1];
        const size_t H = shape_[2];
        const size_t W = shape_[3];
        const size_t C_out = weight.shape_[0];
        const size_t S = H * W;

        LFS_ASSERT(output.shape()[0] == N && output.shape()[1] == C_out &&
                   output.shape()[2] == H && output.shape()[3] == W &&
                   "conv1x1_bias_relu_out: output shape mismatch");
        LFS_ASSERT(bias.shape_[0] == C_out && "conv1x1_bias_relu_out: bias mismatch");

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
        assert_float32_same_device(*this, "max_pool2d_out", {&output});
        LFS_ASSERT(is_valid() && "max_pool2d_out: invalid input tensor");
        LFS_ASSERT(output.is_valid() && "max_pool2d_out: invalid output tensor");
        LFS_ASSERT(shape_.rank() == 4 && "max_pool2d_out: input must be 4D [N,C,H,W]");
        LFS_ASSERT(device_ == Device::CUDA && "max_pool2d_out: CUDA only");

        if (stride <= 0)
            stride = kernel_size;

        const int N = static_cast<int>(shape_[0]);
        const int C = static_cast<int>(shape_[1]);
        const int H_in = static_cast<int>(shape_[2]);
        const int W_in = static_cast<int>(shape_[3]);
        const int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
        const int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

        LFS_ASSERT(output.shape()[0] == static_cast<size_t>(N) &&
                   output.shape()[1] == static_cast<size_t>(C) &&
                   output.shape()[2] == static_cast<size_t>(H_out) &&
                   output.shape()[3] == static_cast<size_t>(W_out) &&
                   "max_pool2d_out: output shape mismatch");

        const Tensor& input_cont = is_contiguous() ? *this : contiguous();
        tensor_ops::launch_max_pool2d(input_cont.ptr<float>(), output.ptr<float>(),
                                      N, C, H_in, W_in, H_out, W_out,
                                      kernel_size, stride, padding, stream());
    }

    void Tensor::adaptive_avg_pool2d_out(int output_h, int output_w, Tensor& output) const {
        assert_float32_same_device(*this, "adaptive_avg_pool2d_out", {&output});
        LFS_ASSERT(is_valid() && "adaptive_avg_pool2d_out: invalid input tensor");
        LFS_ASSERT(output.is_valid() && "adaptive_avg_pool2d_out: invalid output tensor");
        LFS_ASSERT(shape_.rank() == 4 && "adaptive_avg_pool2d_out: input must be 4D [N,C,H,W]");
        LFS_ASSERT(device_ == Device::CUDA && "adaptive_avg_pool2d_out: CUDA only");

        const int N = static_cast<int>(shape_[0]);
        const int C = static_cast<int>(shape_[1]);
        const int H_in = static_cast<int>(shape_[2]);
        const int W_in = static_cast<int>(shape_[3]);

        LFS_ASSERT(output.shape()[0] == static_cast<size_t>(N) &&
                   output.shape()[1] == static_cast<size_t>(C) &&
                   output.shape()[2] == static_cast<size_t>(output_h) &&
                   output.shape()[3] == static_cast<size_t>(output_w) &&
                   "adaptive_avg_pool2d_out: output shape mismatch");

        const Tensor& input_cont = is_contiguous() ? *this : contiguous();
        tensor_ops::launch_adaptive_avg_pool2d(input_cont.ptr<float>(), output.ptr<float>(),
                                               N, C, H_in, W_in, output_h, output_w, stream());
    }

    void Tensor::linear_bias_relu_out(const Tensor& weight, const Tensor& bias, Tensor& output) const {
        assert_float32_same_device(*this, "linear_bias_relu_out", {&weight, &bias, &output});
        LFS_ASSERT(is_valid() && "linear_bias_relu_out: invalid input tensor");
        LFS_ASSERT(weight.is_valid() && "linear_bias_relu_out: invalid weight tensor");
        LFS_ASSERT(bias.is_valid() && "linear_bias_relu_out: invalid bias tensor");
        LFS_ASSERT(output.is_valid() && "linear_bias_relu_out: invalid output tensor");
        LFS_ASSERT(device_ == Device::CUDA && "linear_bias_relu_out: CUDA only");

        const size_t in_features = weight.shape_[1];
        const size_t out_features = weight.shape_[0];

        size_t batch_size = 1;
        for (size_t i = 0; i < shape_.rank() - 1; ++i) {
            batch_size *= shape_[i];
        }

        LFS_ASSERT(output.numel() == batch_size * out_features && "linear_bias_relu_out: output size mismatch");

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
        assert_float32_same_device(*this, "linear_out", {&weight, &output});
        if (bias.is_valid()) {
            assert_float32_same_device(*this, "linear_out", {&bias});
        }
        LFS_ASSERT(is_valid() && "linear_out: invalid input tensor");
        LFS_ASSERT(weight.is_valid() && "linear_out: invalid weight tensor");
        LFS_ASSERT(output.is_valid() && "linear_out: invalid output tensor");
        LFS_ASSERT(device_ == Device::CUDA && "linear_out: CUDA only");

        const size_t in_features = weight.shape_[1];
        const size_t out_features = weight.shape_[0];

        size_t batch_size = 1;
        for (size_t i = 0; i < shape_.rank() - 1; ++i) {
            batch_size *= shape_[i];
        }

        LFS_ASSERT(output.numel() == batch_size * out_features && "linear_out: output size mismatch");

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
