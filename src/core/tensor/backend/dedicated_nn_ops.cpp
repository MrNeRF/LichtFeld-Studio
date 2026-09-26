/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "backend_kernels.hpp"
#include "gpu_backend_ops.hpp"

#include "core/tensor.hpp"

#include <optional>

namespace lfs::core::nn::dedicated {
    namespace {
        std::optional<internal::StorageRef> optional_storage(const Tensor* tensor) {
            if (tensor == nullptr)
                return std::nullopt;
            return internal::storage_ref(*tensor);
        }
    } // namespace

    bool available(const Tensor& tensor) { return internal::backend_ops_for(tensor).nn_kernels(); }

    Tensor linear(const Tensor& a, const Tensor& w, const Tensor* bias, const Tensor* scale, const Tensor* residual,
                  const TensorShape& output_shape, const std::size_t batch, const std::size_t m, const std::size_t n,
                  const std::size_t k, const bool trans_b, const bool batched_b, const Activation activation) {
        auto out = internal::allocate_like(a, output_shape, a.dtype());
        internal::backend_ops_for(a).nn_linear(
            internal::storage_ref(a), internal::storage_ref(w), optional_storage(bias), optional_storage(scale),
            optional_storage(residual), internal::storage_ref(out),
            {.batch = batch, .m = m, .n = n, .k = k, .trans_b = trans_b, .batched_b = batched_b,
             .activation = static_cast<int>(activation)},
            {});
        return out;
    }

    Tensor norm(const Tensor& input, const Tensor& weight, const Tensor* bias, const float eps) {
        const std::size_t cols = input.shape()[input.ndim() - 1];
        auto out = internal::allocate_like(input, input.shape(), input.dtype());
        internal::backend_ops_for(input).nn_norm(internal::storage_ref(input), internal::storage_ref(weight),
                                                 optional_storage(bias), internal::storage_ref(out),
                                                 {.rows = input.numel() / cols, .cols = cols, .eps = eps}, {});
        return out;
    }

    Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* mask, const float scale,
                     const std::array<long long, 4>& mask_strides) {
        auto out = internal::allocate_like(q, q.shape(), q.dtype());
        internal::backend_ops_for(q).nn_attention(
            internal::storage_ref(q), internal::storage_ref(k), internal::storage_ref(v), optional_storage(mask),
            internal::storage_ref(out),
            {.groups = q.shape()[0] * q.shape()[1],
             .heads = q.shape()[1],
             .queries = q.shape()[2],
             .keys = k.shape()[2],
             .dim = q.shape()[3],
             .scale = scale,
             .mask_strides = {mask_strides[0], mask_strides[1], mask_strides[2], mask_strides[3]}},
            {});
        return out;
    }

    Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor* bias, const Conv2dParams& params,
                  const int out_h, const int out_w, const bool transpose) {
        const std::size_t batch = input.shape()[0];
        const int in_group = static_cast<int>(input.shape()[1]) / params.groups;
        const int kh = static_cast<int>(weight.shape()[2]), kw = static_cast<int>(weight.shape()[3]);
        const std::size_t out_channels = transpose ? weight.shape()[1] * params.groups : weight.shape()[0];
        auto out = internal::allocate_like(
            input, TensorShape{batch, out_channels, static_cast<std::size_t>(out_h), static_cast<std::size_t>(out_w)},
            input.dtype());
        // Each group's [in][out * kh * kw] transposed-convolution weights
        // become [out * kh * kw][in].
        const Tensor weights = transpose ? weight
                                               .reshape({params.groups, in_group,
                                                         static_cast<int>(weight.shape()[1]) * kh * kw})
                                               .transpose(1, 2)
                                               .contiguous()
                                         : weight;
        internal::backend_ops_for(input).nn_conv2d(
            internal::storage_ref(input), internal::storage_ref(weights), optional_storage(bias),
            internal::storage_ref(out),
            {.batch = batch,
             .out_channels = out_channels,
             .groups = static_cast<std::size_t>(params.groups),
             .transpose = transpose,
             .geometry = {.channels = in_group,
                          .height = static_cast<int32_t>(input.shape()[2]),
                          .width = static_cast<int32_t>(input.shape()[3]),
                          .out_height = out_h,
                          .out_width = out_w,
                          .kernel_h = kh,
                          .kernel_w = kw,
                          .stride_h = params.stride_h,
                          .stride_w = params.stride_w,
                          .pad_h = params.pad_h,
                          .pad_w = params.pad_w,
                          .dilation_h = params.dilation_h,
                          .dilation_w = params.dilation_w,
                          .mode = transpose ? 0 : static_cast<int32_t>(params.pad_mode)},
             .activation = static_cast<int>(params.activation)},
            {});
        return out;
    }
} // namespace lfs::core::nn::dedicated
