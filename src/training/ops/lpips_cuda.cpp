/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/lpips_cuda.hpp"

#include "core/nn/nn_kernels.hpp"
#include "core/tensor_cuda_interop.hpp"

namespace lfs::training {
    namespace {
        using core::Tensor;
        void weight_taps(const Tensor& weight, Tensor& taps) {
            core::nn::kernels::conv3x3_weight_taps(weight.data_ptr(), taps.data_ptr(),
                                                   weight.shape()[0], weight.shape()[1], core::getCurrentCUDAStream());
        }
        void rgb_conv(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
                      const gpu_ops::RGBConvParams& p) {
            core::nn::kernels::lpips_rgb_conv3x3(input.ptr<float>(), weight.data_ptr(), bias.data_ptr(),
                                                 output.data_ptr(), p.shift.data(), p.scale.data(), p.official_scaling,
                                                 input.shape()[0], input.shape()[2], input.shape()[3], core::getCurrentCUDAStream());
        }
        void convolution(const Tensor& input, const Tensor& weight, const Tensor& taps, const Tensor& bias,
                         Tensor& output, Tensor& scratch, const gpu_ops::ConvParams& p) {
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
