/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/lpips_cuda.hpp"

#include "core/assert.hpp"
#include "core/nn/nn_kernels.hpp"
#include "core/tensor_cuda_interop.hpp"

#include <format>
#include <limits>

namespace lfs::training {
    namespace {
        using core::DataType;
        using core::Tensor;

        void validate_operand(const Tensor& tensor, const DataType dtype, const std::string_view name) {
            LFS_ASSERT_MSG(tensor.is_valid() && tensor.device() == core::Device::GPU &&
                               core::gpu_backend_of(tensor) == core::GpuBackend::CUDA &&
                               tensor.is_contiguous() && tensor.dtype() == dtype && tensor.numel() > 0 &&
                               tensor.numel() <= static_cast<size_t>(std::numeric_limits<int>::max()),
                           std::format("CUDA LPIPS {} must be a nonempty contiguous CUDA {} tensor with at most INT_MAX elements "
                                       "(valid={}, device={}, backend={}, dtype={}, shape={}, contiguous={})",
                                       name, core::dtype_name(dtype), tensor.is_valid(), static_cast<int>(tensor.device()),
                                       core::gpu_backend_of(tensor).has_value() ? core::gpu_backend_name(*core::gpu_backend_of(tensor)) : "none",
                                       core::dtype_name(tensor.dtype()), tensor.shape().str(), tensor.is_contiguous()));
        }

        void validate_shape(const Tensor& tensor, const core::TensorShape& shape, const std::string_view name) {
            LFS_ASSERT_MSG(tensor.shape() == shape,
                           std::format("CUDA LPIPS {} must have shape {} (shape={})", name, shape.str(), tensor.shape().str()));
        }

        void weight_taps(const Tensor& weight, Tensor& taps) {
            validate_operand(weight, DataType::Float16, "weight");
            validate_operand(taps, DataType::Float16, "taps");
            LFS_ASSERT_MSG(weight.ndim() == 4 && weight.shape()[2] == 3 && weight.shape()[3] == 3,
                           std::format("CUDA LPIPS weight must have shape [C_out, C_in, 3, 3] (shape={})", weight.shape().str()));
            validate_shape(taps, {9, weight.shape()[0], weight.shape()[1]}, "taps");
            core::nn::kernels::conv3x3_weight_taps(weight.data_ptr(), taps.data_ptr(),
                                                   weight.shape()[0], weight.shape()[1], core::getCurrentCUDAStream());
        }
        void rgb_conv(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
                      const gpu_ops::RGBConvParams& p) {
            validate_operand(input, DataType::Float32, "input");
            validate_operand(weight, DataType::Float16, "weight");
            validate_operand(bias, DataType::Float16, "bias");
            validate_operand(output, DataType::Float16, "output");
            LFS_ASSERT_MSG(input.ndim() == 4 && input.shape()[1] == 3,
                           std::format("CUDA LPIPS RGB input must have shape [N, 3, H, W] (shape={})", input.shape().str()));
            validate_shape(weight, {64, 3, 3, 3}, "weight");
            validate_shape(bias, {64}, "bias");
            validate_shape(output, {input.shape()[0], 64, input.shape()[2], input.shape()[3]}, "output");
            core::nn::kernels::lpips_rgb_conv3x3(input.ptr<float>(), weight.data_ptr(), bias.data_ptr(),
                                                 output.data_ptr(), p.shift.data(), p.scale.data(), p.official_scaling,
                                                 input.shape()[0], input.shape()[2], input.shape()[3], core::getCurrentCUDAStream());
        }
        void convolution(const Tensor& input, const Tensor& weight, const Tensor& taps, const Tensor& bias,
                         Tensor& output, Tensor& scratch, const gpu_ops::ConvParams& p) {
            LFS_ASSERT_MSG(input.dtype() == DataType::Float16 || input.dtype() == DataType::Float32,
                           std::format("CUDA LPIPS convolution input must be float16 or float32 (dtype={})", core::dtype_name(input.dtype())));
            validate_operand(input, input.dtype(), "input");
            validate_operand(weight, input.dtype(), "weight");
            validate_operand(output, input.dtype(), "output");
            LFS_ASSERT_MSG(input.ndim() == 4 && weight.ndim() == 4 &&
                               weight.shape()[1] == input.shape()[1] && weight.shape()[2] == 3 && weight.shape()[3] == 3,
                           std::format("CUDA LPIPS convolution needs [N, C_in, H, W] input and [C_out, C_in, 3, 3] weight "
                                       "(input={}, weight={})",
                                       input.shape().str(), weight.shape().str()));
            LFS_ASSERT_MSG(p.stride_h == 1 && p.stride_w == 1 && p.dilation_h == 1 && p.dilation_w == 1 &&
                               p.groups == 1 && p.pad_h >= 0 && p.pad_w >= 0,
                           std::format("CUDA LPIPS convolution requires unit stride, dilation and groups, and nonnegative padding "
                                       "(stride=[{},{}], dilation=[{},{}], groups={}, padding=[{},{}])",
                                       p.stride_h, p.stride_w, p.dilation_h, p.dilation_w, p.groups, p.pad_h, p.pad_w));
            const int64_t out_h = static_cast<int64_t>(input.shape()[2]) + 2LL * p.pad_h - 2;
            const int64_t out_w = static_cast<int64_t>(input.shape()[3]) + 2LL * p.pad_w - 2;
            LFS_ASSERT_MSG(out_h > 0 && out_w > 0 && out_h <= std::numeric_limits<int>::max() && out_w <= std::numeric_limits<int>::max(),
                           std::format("CUDA LPIPS convolution output dimensions must be positive int values (height={}, width={})", out_h, out_w));
            validate_shape(output, {input.shape()[0], weight.shape()[0], static_cast<size_t>(out_h), static_cast<size_t>(out_w)}, "output");
            if (bias.is_valid()) {
                validate_operand(bias, input.dtype(), "bias");
                validate_shape(bias, {weight.shape()[0]}, "bias");
            }
            if (taps.is_valid()) {
                validate_operand(taps, DataType::Float16, "taps");
                validate_shape(taps, {9, weight.shape()[0], weight.shape()[1]}, "taps");
            }
            if (scratch.is_valid()) {
                validate_operand(scratch, DataType::Float16, "scratch");
                LFS_ASSERT_MSG(taps.is_valid() || input.dtype() != DataType::Float16 || scratch.numel() >= weight.numel(),
                               std::format("CUDA LPIPS convolution scratch must hold every weight tap when taps are absent "
                                           "(scratch_elements={}, weight_elements={})",
                                           scratch.numel(), weight.numel()));
            }
            core::nn::kernels::conv2d_implicit(input.data_ptr(), weight.data_ptr(),
                                               taps.is_valid() ? taps.data_ptr() : nullptr, bias.is_valid() ? bias.data_ptr() : nullptr,
                                               output.data_ptr(), scratch.is_valid() ? scratch.data_ptr() : nullptr,
                                               input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3], weight.shape()[0],
                                               weight.shape()[2], weight.shape()[3], output.shape()[2], output.shape()[3],
                                               p.stride_h, p.stride_w, p.pad_h, p.pad_w, p.dilation_h, p.dilation_w,
                                               static_cast<int>(p.pad_mode), static_cast<int>(p.activation), input.dtype(), core::getCurrentCUDAStream());
        }
        void pool_reduce(const Tensor& x, const Tensor& y, const Tensor& weight, Tensor& score,
                         Tensor& pooled_x, Tensor& pooled_y, const gpu_ops::PoolReduceParams& p) {
            validate_operand(x, DataType::Float16, "x");
            validate_operand(y, DataType::Float16, "y");
            validate_operand(weight, DataType::Float16, "weight");
            validate_operand(score, DataType::Float32, "score");
            LFS_ASSERT_MSG(x.ndim() == 4 && x.shape()[1] % 8 == 0,
                           std::format("CUDA LPIPS pooling features must have shape [N, C, H, W] with C divisible by 8 (shape={})", x.shape().str()));
            validate_shape(y, x.shape(), "y");
            LFS_ASSERT_MSG(weight.numel() == x.shape()[1],
                           std::format("CUDA LPIPS pooling needs one weight per channel (weight_elements={}, channels={})", weight.numel(), x.shape()[1]));
            LFS_ASSERT_MSG(pooled_x.is_valid() == pooled_y.is_valid(),
                           std::format("CUDA LPIPS pooling outputs must both be present or absent (pooled_x={}, pooled_y={})", pooled_x.is_valid(), pooled_y.is_valid()));
            if (pooled_x.is_valid()) {
                validate_operand(pooled_x, DataType::Float16, "pooled_x");
                validate_operand(pooled_y, DataType::Float16, "pooled_y");
                const core::TensorShape shape{x.shape()[0], x.shape()[1], x.shape()[2] / 2, x.shape()[3] / 2};
                validate_shape(pooled_x, shape, "pooled_x");
                validate_shape(pooled_y, shape, "pooled_y");
            }
            core::nn::kernels::lpips_pool_reduce(x.data_ptr(), y.data_ptr(), weight.data_ptr(), score.ptr<float>(),
                                                 pooled_x.is_valid() ? pooled_x.data_ptr() : nullptr, pooled_y.is_valid() ? pooled_y.data_ptr() : nullptr,
                                                 x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3], p.y0, p.y1, p.x0, p.x1,
                                                 p.inverse_count, core::getCurrentCUDAStream());
        }
    } // namespace

    const gpu_ops::LpipsOps& cuda_lpips_ops() {
        static const gpu_ops::LpipsOps ops{{
            .weight_taps = weight_taps,
            .rgb_conv = rgb_conv,
            .convolution = convolution,
            .pool_reduce = pool_reduce,
        }};
        return ops;
    }
} // namespace lfs::training
