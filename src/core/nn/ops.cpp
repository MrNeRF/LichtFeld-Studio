/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nn/ops.hpp"

#include "core/cuda_error.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "nn_kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <vector>

namespace lfs::core::nn {
    namespace {

        void require_nn_tensor(const Tensor& tensor, const std::string_view op,
                               const std::string_view role) {
            tensor_contract::require_valid(tensor, op, role, LFS_SOURCE_SITE_CURRENT());
            tensor_contract::require_dtype(tensor, {DataType::Float32, DataType::Float16}, op,
                                           role, LFS_SOURCE_SITE_CURRENT());
            LFS_ASSERT_MSG(tensor.device() == Device::CUDA,
                           std::format("{} requires CUDA {} (device={})", op, role,
                                       device_name(tensor.device())));
        }

        void require_same_dtype_device(const Tensor& a, const Tensor& b, const std::string_view op,
                                       const std::string_view a_role, const std::string_view b_role) {
            tensor_contract::require_same_device(a, b, op, a_role, b_role, LFS_SOURCE_SITE_CURRENT());
            LFS_ASSERT_MSG(a.dtype() == b.dtype(),
                           std::format("{} dtype mismatch ({}={}, {}={})", op, a_role,
                                       dtype_name(a.dtype()), b_role, dtype_name(b.dtype())));
        }

        Tensor empty_like_shape(const Tensor& like, const TensorShape& shape) {
            auto out = Tensor::empty(shape, like.device(), like.dtype());
            out.set_stream(like.stream());
            return out;
        }

        const void* raw(const Tensor& t) {
            return t.dtype() == DataType::Float16
                       ? static_cast<const void*>(t.ptr<__half>())
                       : static_cast<const void*>(t.ptr<float>());
        }

        void* raw_mut(Tensor& t) {
            return t.dtype() == DataType::Float16 ? static_cast<void*>(t.ptr<__half>())
                                                  : static_cast<void*>(t.ptr<float>());
        }

        std::size_t leading_product(const TensorShape& shape, const std::size_t skip_last) {
            std::size_t prod = 1;
            const std::size_t limit = shape.rank() >= skip_last ? shape.rank() - skip_last : 0;
            for (std::size_t i = 0; i < limit; ++i) {
                prod *= shape[i];
            }
            return prod;
        }

        int conv_out_dim(const int in, const int kernel, const int stride, const int pad,
                         const int dil) {
            return (in + 2 * pad - dil * (kernel - 1) - 1) / stride + 1;
        }

        int conv_transpose_out_dim(const int in, const int kernel, const int stride, const int pad,
                                   const int dil, const int output_pad) {
            return (in - 1) * stride - 2 * pad + dil * (kernel - 1) + output_pad + 1;
        }

        Tensor ensure_workspace(Tensor* workspace, const std::size_t bytes, const Tensor& like,
                                const std::string_view op) {
            if (workspace != nullptr) {
                require_nn_tensor(*workspace, op, "workspace");
                LFS_ASSERT_MSG(workspace->bytes() >= bytes,
                               std::format("{} workspace is too small (have {}, need {})", op,
                                           workspace->bytes(), bytes));
                workspace->set_stream(like.stream());
                return *workspace;
            }
            const std::size_t elem = dtype_size(like.dtype());
            const std::size_t count = (bytes + elem - 1) / elem;
            return empty_like_shape(like, TensorShape{std::vector<std::size_t>{count}});
        }

        void pack_nhwc_to_nchw(const Tensor& nhwc, Tensor& nchw) {
            auto perm = nhwc.permute({0, 3, 1, 2}).contiguous();
            nchw = std::move(perm);
        }

    } // namespace

    Tensor gemm(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b, const Tensor* bias,
                Activation activation, const Tensor* residual, const Tensor* scale) {
        require_nn_tensor(a, "gemm", "a");
        require_nn_tensor(b, "gemm", "b");
        require_same_dtype_device(a, b, "gemm", "a", "b");
        LFS_ASSERT_MSG(a.ndim() >= 2 && b.ndim() >= 2, "gemm requires rank >= 2");

        const Tensor a_c0 = a.contiguous();
        const Tensor b_c = b.contiguous();
        Tensor a_t_store;
        const Tensor& a_c = trans_a ? (a_t_store = a_c0.t().contiguous()) : a_c0;

        const int m = static_cast<int>(a_c.size(a_c.ndim() - 2));
        const int ka = static_cast<int>(a_c.size(a_c.ndim() - 1));
        const int n = static_cast<int>(trans_b ? b_c.size(b_c.ndim() - 2) : b_c.size(b_c.ndim() - 1));
        const int kb = static_cast<int>(trans_b ? b_c.size(b_c.ndim() - 1) : b_c.size(b_c.ndim() - 2));
        LFS_ASSERT_MSG(ka == kb, std::format("gemm inner dimension mismatch (Ka={}, Kb={})", ka, kb));

        const std::size_t batch_a = leading_product(a_c.shape(), 2);
        const std::size_t batch_b = leading_product(b_c.shape(), 2);
        LFS_ASSERT_MSG(batch_b == batch_a || batch_b == 1,
                       std::format("gemm batch mismatch (a={}, b={})", batch_a, batch_b));
        const int batch = static_cast<int>(batch_a);

        std::vector<std::size_t> out_dims;
        for (std::size_t i = 0; i + 2 < a_c.ndim(); ++i) {
            out_dims.push_back(a_c.shape()[i]);
        }
        out_dims.push_back(static_cast<std::size_t>(m));
        out_dims.push_back(static_cast<std::size_t>(n));

        Tensor bias_store;
        const Tensor* bias_c = nullptr;
        if (bias != nullptr && bias->is_valid()) {
            require_nn_tensor(*bias, "gemm", "bias");
            require_same_dtype_device(a_c, *bias, "gemm", "a", "bias");
            LFS_ASSERT_MSG(bias->numel() == static_cast<std::size_t>(n),
                           "gemm bias must have N elements");
            bias_store = bias->contiguous();
            bias_c = &bias_store;
        }

        Tensor residual_store;
        const Tensor* residual_c = nullptr;
        if (residual != nullptr && residual->is_valid()) {
            require_nn_tensor(*residual, "gemm", "residual");
            require_same_dtype_device(a_c, *residual, "gemm", "a", "residual");
            LFS_ASSERT_MSG(residual->numel() == static_cast<std::size_t>(m) * static_cast<std::size_t>(n) *
                                                    batch_a,
                           "gemm residual must match the output");
            residual_store = residual->contiguous();
            residual_c = &residual_store;
        }

        Tensor scale_store;
        const Tensor* scale_c = nullptr;
        if (scale != nullptr && scale->is_valid()) {
            require_nn_tensor(*scale, "gemm", "scale");
            require_same_dtype_device(a_c, *scale, "gemm", "a", "scale");
            LFS_ASSERT_MSG(scale->numel() == static_cast<std::size_t>(n),
                           "gemm scale must have N elements");
            scale_store = scale->contiguous();
            scale_c = &scale_store;
        }

        auto out = empty_like_shape(a_c, TensorShape(out_dims));
        pin_operands({&a_c, &b_c});
        const cudaStream_t stream = prepare_inputs_for_stream({&a_c, &b_c}, out.stream());
        out.set_stream(stream);

        const long long stride_a = static_cast<long long>(m) * ka;
        const long long stride_b =
            batch_b == 1 ? 0 : (trans_b ? static_cast<long long>(n) * kb : static_cast<long long>(kb) * n);
        const long long stride_c = static_cast<long long>(m) * n;

        kernels::gemm(raw(a_c), raw(b_c), raw_mut(out), m, n, ka, stride_a, stride_b, stride_c,
                      batch, false, trans_b, bias_c ? raw(*bias_c) : nullptr,
                      static_cast<int>(activation), a_c.dtype(), stream, false,
                      residual_c ? raw(*residual_c) : nullptr, scale_c ? raw(*scale_c) : nullptr);
        return out;
    }

    Tensor linear(const Tensor& input, const Tensor& weight, const Tensor* bias,
                  Activation activation) {
        require_nn_tensor(input, "linear", "input");
        require_nn_tensor(weight, "linear", "weight");
        require_same_dtype_device(input, weight, "linear", "input", "weight");
        LFS_ASSERT_MSG(weight.ndim() == 2, "linear weight must be [N, K]");
        LFS_ASSERT_MSG(input.ndim() >= 1, "linear input must have a feature dim");
        const std::size_t k = input.shape()[input.ndim() - 1];
        LFS_ASSERT_MSG(weight.shape()[1] == k, "linear inner dimension mismatch");
        const std::size_t n = weight.shape()[0];
        const std::size_t m = input.numel() / k;

        const Tensor in_c = input.contiguous();
        const Tensor w_c = weight.contiguous();
        auto in_2d = in_c.reshape(TensorShape{std::vector<std::size_t>{m, k}});

        Tensor bias_store;
        const Tensor* bias_c = nullptr;
        if (bias != nullptr && bias->is_valid()) {
            require_nn_tensor(*bias, "linear", "bias");
            require_same_dtype_device(in_c, *bias, "linear", "input", "bias");
            LFS_ASSERT_MSG(bias->numel() == n, "linear bias must have N elements");
            bias_store = bias->contiguous();
            bias_c = &bias_store;
        }

        std::vector<std::size_t> out_dims;
        for (std::size_t i = 0; i + 1 < input.ndim(); ++i) {
            out_dims.push_back(input.shape()[i]);
        }
        out_dims.push_back(n);
        auto out = empty_like_shape(in_c, TensorShape(out_dims));
        pin_operands({&in_2d, &w_c});
        const cudaStream_t stream = prepare_inputs_for_stream({&in_2d, &w_c}, out.stream());
        out.set_stream(stream);
        kernels::gemm(raw(in_2d), raw(w_c), raw_mut(out), static_cast<int>(m), static_cast<int>(n),
                      static_cast<int>(k), static_cast<long long>(m) * static_cast<long long>(k),
                      static_cast<long long>(n) * static_cast<long long>(k),
                      static_cast<long long>(m) * static_cast<long long>(n), 1, false, true,
                      bias_c ? raw(*bias_c) : nullptr, static_cast<int>(activation), in_c.dtype(),
                      stream);
        return out;
    }

    Tensor bmm(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b, const Tensor* bias,
               Activation activation) {
        LFS_ASSERT_MSG(a.ndim() == 3 && (b.ndim() == 3 || b.ndim() == 2),
                       "bmm requires a [B,M,K] and b [B,K,N] or [K,N]");
        return gemm(a, b, trans_a, trans_b, bias, activation);
    }

    Tensor layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias, float eps) {
        require_nn_tensor(input, "layer_norm", "input");
        require_nn_tensor(weight, "layer_norm", "weight");
        require_nn_tensor(bias, "layer_norm", "bias");
        require_same_dtype_device(input, weight, "layer_norm", "input", "weight");
        require_same_dtype_device(input, bias, "layer_norm", "input", "bias");
        const int cols = static_cast<int>(input.shape()[input.ndim() - 1]);
        LFS_ASSERT_MSG(weight.numel() == static_cast<std::size_t>(cols) &&
                           bias.numel() == static_cast<std::size_t>(cols),
                       "layer_norm affine tensors must match the last dim");
        const Tensor in_c = input.contiguous();
        const Tensor w_c = weight.contiguous();
        const Tensor b_c = bias.contiguous();
        auto out = empty_like_shape(in_c, in_c.shape());
        pin_operands({&in_c, &w_c, &b_c});
        const cudaStream_t stream = prepare_inputs_for_stream({&in_c, &w_c, &b_c}, out.stream());
        out.set_stream(stream);
        const int rows = static_cast<int>(in_c.numel() / static_cast<std::size_t>(cols));
        kernels::layer_norm(raw(in_c), raw(w_c), raw(b_c), raw_mut(out), rows, cols, eps,
                            in_c.dtype(), stream);
        return out;
    }

    Tensor rms_norm(const Tensor& input, const Tensor& weight, float eps) {
        require_nn_tensor(input, "rms_norm", "input");
        require_nn_tensor(weight, "rms_norm", "weight");
        require_same_dtype_device(input, weight, "rms_norm", "input", "weight");
        const int cols = static_cast<int>(input.shape()[input.ndim() - 1]);
        LFS_ASSERT_MSG(weight.numel() == static_cast<std::size_t>(cols),
                       "rms_norm weight must match the last dim");
        const Tensor in_c = input.contiguous();
        const Tensor w_c = weight.contiguous();
        auto out = empty_like_shape(in_c, in_c.shape());
        pin_operands({&in_c, &w_c});
        const cudaStream_t stream = prepare_inputs_for_stream({&in_c, &w_c}, out.stream());
        out.set_stream(stream);
        const int rows = static_cast<int>(in_c.numel() / static_cast<std::size_t>(cols));
        kernels::rms_norm(raw(in_c), raw(w_c), raw_mut(out), rows, cols, eps, in_c.dtype(), stream);
        return out;
    }

    Tensor softmax(const Tensor& input, const Tensor* mask) {
        require_nn_tensor(input, "softmax", "input");
        const Tensor in_c = input.contiguous();
        const int cols = static_cast<int>(in_c.shape()[in_c.ndim() - 1]);
        const int rows = static_cast<int>(in_c.numel() / static_cast<std::size_t>(cols));
        Tensor mask_s;
        const Tensor* mask_c = nullptr;
        long long msr = cols;
        long long msc = 1;
        if (mask != nullptr && mask->is_valid()) {
            require_nn_tensor(*mask, "softmax", "mask");
            require_same_dtype_device(in_c, *mask, "softmax", "input", "mask");
            mask_s = mask->contiguous();
            mask_c = &mask_s;
            if (mask_c->ndim() == in_c.ndim()) {
                msr = static_cast<long long>(mask_c->shape()[mask_c->ndim() - 1] == 1
                                                 ? 0
                                                 : mask_c->shape()[mask_c->ndim() - 1]);
                msc = mask_c->shape()[mask_c->ndim() - 1] == 1 ? 0 : 1;
            } else if (mask_c->ndim() == 1) {
                msr = 0;
                msc = 1;
            }
        }
        auto out = empty_like_shape(in_c, in_c.shape());
        pin_operands({&in_c});
        const cudaStream_t stream = prepare_inputs_for_stream({&in_c}, out.stream());
        out.set_stream(stream);
        kernels::softmax(raw(in_c), mask_c ? raw(*mask_c) : nullptr, raw_mut(out), rows, cols, msr,
                         msc, mask_c != nullptr, in_c.dtype(), stream);
        return out;
    }

    Tensor attention(const Tensor& query, const Tensor& key, const Tensor& value,
                     const Tensor* mask, float scale) {
        require_nn_tensor(query, "attention", "query");
        require_nn_tensor(key, "attention", "key");
        require_nn_tensor(value, "attention", "value");
        require_same_dtype_device(query, key, "attention", "query", "key");
        require_same_dtype_device(query, value, "attention", "query", "value");
        LFS_ASSERT_MSG(query.ndim() == 4 && key.ndim() == 4 && value.ndim() == 4,
                       "attention expects [B, H, N, d]");
        const int b = static_cast<int>(query.shape()[0]);
        const int h = static_cast<int>(query.shape()[1]);
        const int n_q = static_cast<int>(query.shape()[2]);
        const int d = static_cast<int>(query.shape()[3]);
        const int n_k = static_cast<int>(key.shape()[2]);
        LFS_ASSERT_MSG(key.shape()[0] == static_cast<std::size_t>(b) &&
                           key.shape()[1] == static_cast<std::size_t>(h) &&
                           key.shape()[3] == static_cast<std::size_t>(d),
                       "attention key shape mismatch");
        LFS_ASSERT_MSG(value.shape()[0] == static_cast<std::size_t>(b) &&
                           value.shape()[1] == static_cast<std::size_t>(h) &&
                           value.shape()[2] == static_cast<std::size_t>(n_k) &&
                           value.shape()[3] == static_cast<std::size_t>(d),
                       "attention value shape mismatch");
        LFS_ASSERT_MSG(d <= 128, "attention supports head dim d <= 128");
        const float used_scale = scale > 0.0f ? scale : 1.0f / std::sqrt(static_cast<float>(d));

        Tensor m_s;
        const Tensor q_c = query.contiguous();
        const Tensor k_c = key.contiguous();
        const Tensor v_c = value.contiguous();
        const Tensor* m_c = nullptr;
        long long sb = 0, sh = 0, sq = 0, sk = 0;
        if (mask != nullptr && mask->is_valid()) {
            require_nn_tensor(*mask, "attention", "mask");
            require_same_dtype_device(q_c, *mask, "attention", "query", "mask");
            m_s = mask->contiguous();
            m_c = &m_s;
            const auto& ms = m_c->shape();
            if (m_c->ndim() == 2) {
                sq = ms[1] == 1 ? 0 : static_cast<long long>(ms[1]);
                sk = ms[1] == 1 ? 0 : 1;
            } else if (m_c->ndim() == 4) {
                const long long nk = static_cast<long long>(ms[3]);
                const long long nq = static_cast<long long>(ms[2]);
                const long long hh = static_cast<long long>(ms[1]);
                sb = ms[0] == 1 ? 0 : hh * nq * nk;
                sh = ms[1] == 1 ? 0 : nq * nk;
                sq = ms[2] == 1 ? 0 : nk;
                sk = ms[3] == 1 ? 0 : 1;
            } else {
                LFS_ASSERT_MSG(false, "attention mask must be rank 2 or 4");
            }
        }

        auto out = empty_like_shape(q_c, q_c.shape());
        pin_operands({&q_c, &k_c, &v_c});
        const cudaStream_t stream = prepare_inputs_for_stream({&q_c, &k_c, &v_c}, out.stream());
        out.set_stream(stream);
        kernels::attention(raw(q_c), raw(k_c), raw(v_c), m_c ? raw(*m_c) : nullptr, raw_mut(out), b,
                           h, n_q, n_k, d, used_scale, sb, sh, sq, sk, m_c != nullptr, q_c.dtype(),
                           stream);
        return out;
    }

    Tensor window_partition(const Tensor& input, int window_size) {
        require_nn_tensor(input, "window_partition", "input");
        LFS_ASSERT_MSG(input.ndim() == 4, "window_partition expects [B, H, N, d]");
        LFS_ASSERT_MSG(window_size > 0, "window_size must be positive");
        const Tensor in_c = input.contiguous();
        const int b = static_cast<int>(in_c.shape()[0]);
        const int h = static_cast<int>(in_c.shape()[1]);
        const int n = static_cast<int>(in_c.shape()[2]);
        const int d = static_cast<int>(in_c.shape()[3]);
        const int n_win = (n + window_size - 1) / window_size;
        const int pad = n_win * window_size - n;
        Tensor seq = in_c;
        if (pad > 0) {
            auto z = Tensor::zeros(
                TensorShape{std::vector<std::size_t>{static_cast<std::size_t>(b),
                                                     static_cast<std::size_t>(h),
                                                     static_cast<std::size_t>(pad),
                                                     static_cast<std::size_t>(d)}},
                in_c.device(), in_c.dtype());
            z.set_stream(in_c.stream());
            seq = Tensor::cat({in_c, z}, 2);
        }
        auto viewed = seq.reshape(TensorShape{std::vector<std::size_t>{
            static_cast<std::size_t>(b), static_cast<std::size_t>(h),
            static_cast<std::size_t>(n_win), static_cast<std::size_t>(window_size),
            static_cast<std::size_t>(d)}});
        return viewed.permute({0, 2, 1, 3, 4})
            .contiguous()
            .reshape(TensorShape{std::vector<std::size_t>{
                static_cast<std::size_t>(b * n_win), static_cast<std::size_t>(h),
                static_cast<std::size_t>(window_size), static_cast<std::size_t>(d)}});
    }

    Tensor window_unpartition(const Tensor& windows, int window_size, int original_n) {
        require_nn_tensor(windows, "window_unpartition", "windows");
        LFS_ASSERT_MSG(windows.ndim() == 4, "window_unpartition expects [B*n_win, H, W, d]");
        LFS_ASSERT_MSG(window_size > 0 && original_n >= 0, "window_unpartition sizes invalid");
        const Tensor w_c = windows.contiguous();
        const int n_win = (original_n + window_size - 1) / window_size;
        LFS_ASSERT_MSG(n_win > 0 && w_c.shape()[0] % static_cast<std::size_t>(n_win) == 0,
                       "window_unpartition batch is not divisible by n_windows");
        const int b = static_cast<int>(w_c.shape()[0] / static_cast<std::size_t>(n_win));
        const int h = static_cast<int>(w_c.shape()[1]);
        const int d = static_cast<int>(w_c.shape()[3]);
        auto viewed = w_c.reshape(TensorShape{std::vector<std::size_t>{
            static_cast<std::size_t>(b), static_cast<std::size_t>(n_win),
            static_cast<std::size_t>(h), static_cast<std::size_t>(window_size),
            static_cast<std::size_t>(d)}});
        auto folded = viewed.permute({0, 2, 1, 3, 4})
                          .contiguous()
                          .reshape(TensorShape{std::vector<std::size_t>{
                              static_cast<std::size_t>(b), static_cast<std::size_t>(h),
                              static_cast<std::size_t>(n_win * window_size),
                              static_cast<std::size_t>(d)}});
        if (original_n == n_win * window_size) {
            return folded;
        }
        return folded.slice(2, 0, static_cast<std::size_t>(original_n)).contiguous();
    }

    std::pair<int, int> conv2d_output_hw(int height, int width, int kernel_h, int kernel_w,
                                         const Conv2dParams& params) {
        return {conv_out_dim(height, kernel_h, params.stride_h, params.pad_h, params.dilation_h),
                conv_out_dim(width, kernel_w, params.stride_w, params.pad_w, params.dilation_w)};
    }

    std::pair<int, int> conv_transpose2d_output_hw(int height, int width, int kernel_h, int kernel_w,
                                                   const Conv2dParams& params) {
        return {conv_transpose_out_dim(height, kernel_h, params.stride_h, params.pad_h,
                                       params.dilation_h, params.output_pad_h),
                conv_transpose_out_dim(width, kernel_w, params.stride_w, params.pad_w,
                                       params.dilation_w, params.output_pad_w)};
    }

    std::size_t conv2d_workspace_bytes(const TensorShape& input_shape, const TensorShape& weight_shape,
                                       const Conv2dParams& params, DataType dtype) {
        LFS_ASSERT_MSG(input_shape.rank() == 4 && weight_shape.rank() == 4,
                       "conv2d workspace expects 4D input and weight");
        const int kh = static_cast<int>(weight_shape[2]);
        const int kw = static_cast<int>(weight_shape[3]);
        const bool pointwise = kh == 1 && kw == 1 && params.groups == 1 && params.stride_h == 1 &&
                               params.stride_w == 1 && params.pad_h == 0 && params.pad_w == 0 &&
                               params.dilation_h == 1 && params.dilation_w == 1;
        const bool implicit_3x3 = kh == 3 && kw == 3 && params.groups == 1 && params.stride_h == 1 &&
                                  params.stride_w == 1 && params.dilation_h == 1 &&
                                  params.dilation_w == 1 && params.pad_h == 1 && params.pad_w == 1 &&
                                  dtype == DataType::Float16;
        if (pointwise || implicit_3x3) {
            return 0;
        }
        const auto hw = conv2d_output_hw(static_cast<int>(input_shape[2]),
                                         static_cast<int>(input_shape[3]), kh, kw, params);
        const std::size_t cin_g = weight_shape[1];
        const std::size_t k = cin_g * weight_shape[2] * weight_shape[3];
        const std::size_t m = input_shape[0] * static_cast<std::size_t>(hw.first) *
                              static_cast<std::size_t>(hw.second);
        return m * k * dtype_size(dtype);
    }

    std::size_t conv_transpose2d_workspace_bytes(const TensorShape& input_shape,
                                                 const TensorShape& weight_shape,
                                                 const Conv2dParams& params, DataType dtype) {
        LFS_ASSERT_MSG(input_shape.rank() == 4 && weight_shape.rank() == 4,
                       "conv_transpose2d workspace expects 4D input and weight");
        const int kh = static_cast<int>(weight_shape[2]);
        const int kw = static_cast<int>(weight_shape[3]);
        const bool scatter_s2 =
            kh == 2 && kw == 2 && params.stride_h == 2 && params.stride_w == 2 && params.pad_h == 0 &&
            params.pad_w == 0 && params.dilation_h == 1 && params.dilation_w == 1 &&
            params.groups == 1 && params.output_pad_h == 0 && params.output_pad_w == 0 &&
            dtype == DataType::Float16;
        if (scatter_s2) {
            return 0;
        }
        const std::size_t cout_g = weight_shape[1];
        const std::size_t k = cout_g * weight_shape[2] * weight_shape[3];
        const std::size_t m = input_shape[0] * input_shape[2] * input_shape[3];
        return m * k * dtype_size(dtype);
    }

    Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor* bias,
                  const Conv2dParams& params, Tensor* workspace) {
        require_nn_tensor(input, "conv2d", "input");
        require_nn_tensor(weight, "conv2d", "weight");
        require_same_dtype_device(input, weight, "conv2d", "input", "weight");
        LFS_ASSERT_MSG(input.ndim() == 4 && weight.ndim() == 4, "conv2d expects NCHW and OIHW");
        LFS_ASSERT_MSG(params.groups > 0, "conv2d groups must be positive");
        const int n = static_cast<int>(input.shape()[0]);
        const int cin = static_cast<int>(input.shape()[1]);
        const int h = static_cast<int>(input.shape()[2]);
        const int w = static_cast<int>(input.shape()[3]);
        const int cout = static_cast<int>(weight.shape()[0]);
        const int cin_g = static_cast<int>(weight.shape()[1]);
        const int kh = static_cast<int>(weight.shape()[2]);
        const int kw = static_cast<int>(weight.shape()[3]);
        LFS_ASSERT_MSG(cin == cin_g * params.groups && cout % params.groups == 0,
                       "conv2d channel/groups mismatch");
        const auto hw = conv2d_output_hw(h, w, kh, kw, params);
        LFS_ASSERT_MSG(hw.first > 0 && hw.second > 0, "conv2d produced a non-positive output size");
        const int out_h = hw.first;
        const int out_w = hw.second;
        const int cout_g = cout / params.groups;
        const bool pointwise = kh == 1 && kw == 1 && params.groups == 1 && params.stride_h == 1 &&
                               params.stride_w == 1 && params.pad_h == 0 && params.pad_w == 0 &&
                               params.dilation_h == 1 && params.dilation_w == 1;
        const bool implicit_3x3 = kh == 3 && kw == 3 && params.groups == 1 && params.stride_h == 1 &&
                                  params.stride_w == 1 && params.dilation_h == 1 &&
                                  params.dilation_w == 1 && params.pad_h == 1 && params.pad_w == 1 &&
                                  input.dtype() == DataType::Float16;

        Tensor b_s;
        const Tensor in_c = input.contiguous();
        const Tensor w_c = weight.contiguous();
        const Tensor* b_c = nullptr;
        if (bias != nullptr && bias->is_valid()) {
            require_nn_tensor(*bias, "conv2d", "bias");
            require_same_dtype_device(in_c, *bias, "conv2d", "input", "bias");
            LFS_ASSERT_MSG(bias->numel() == static_cast<std::size_t>(cout),
                           "conv2d bias must have C_out elements");
            b_s = bias->contiguous();
            b_c = &b_s;
        }

        pin_operands({&in_c, &w_c});

        if (pointwise) {
            auto nchw = empty_like_shape(
                in_c, TensorShape{std::vector<std::size_t>{static_cast<std::size_t>(n),
                                                           static_cast<std::size_t>(cout),
                                                           static_cast<std::size_t>(out_h),
                                                           static_cast<std::size_t>(out_w)}});
            const cudaStream_t stream = prepare_inputs_for_stream({&in_c, &w_c}, nchw.stream());
            nchw.set_stream(stream);
            const int hw_n = out_h * out_w;
            // Y^T = X^T @ W^T, NT GEMM, bias along Cout, store NCHW (trans_c).
            // X is [N, Cin, HW], W is [Cout, Cin]. Same K-loop as im2col+NT.
            kernels::gemm(raw(in_c), raw(w_c), raw_mut(nchw), hw_n, cout, cin,
                          static_cast<long long>(cin) * hw_n, 0,
                          static_cast<long long>(cout) * hw_n, n, true, true,
                          b_c ? raw(*b_c) : nullptr, 0, in_c.dtype(), stream, true);
            return nchw;
        }

        if (implicit_3x3) {
            auto nchw = empty_like_shape(
                in_c, TensorShape{std::vector<std::size_t>{static_cast<std::size_t>(n),
                                                           static_cast<std::size_t>(cout),
                                                           static_cast<std::size_t>(out_h),
                                                           static_cast<std::size_t>(out_w)}});
            const cudaStream_t stream = prepare_inputs_for_stream({&in_c, &w_c}, nchw.stream());
            nchw.set_stream(stream);
            kernels::conv2d_implicit(raw(in_c), raw(w_c), b_c ? raw(*b_c) : nullptr, raw_mut(nchw),
                                     n, cin, h, w, cout, kh, kw, out_h, out_w, params.stride_h,
                                     params.stride_w, params.pad_h, params.pad_w, params.dilation_h,
                                     params.dilation_w, static_cast<int>(params.pad_mode),
                                     in_c.dtype(), stream);
            return nchw;
        }
        const std::size_t bytes = conv2d_workspace_bytes(in_c.shape(), w_c.shape(), params, in_c.dtype());
        Tensor scratch = ensure_workspace(workspace, bytes, in_c, "conv2d");
        auto nhwc = empty_like_shape(
            in_c, TensorShape{std::vector<std::size_t>{static_cast<std::size_t>(n),
                                                       static_cast<std::size_t>(out_h),
                                                       static_cast<std::size_t>(out_w),
                                                       static_cast<std::size_t>(cout)}});
        const cudaStream_t stream = prepare_inputs_for_stream({&in_c, &w_c}, nhwc.stream());
        nhwc.set_stream(stream);
        scratch.set_stream(stream);

        const int m = n * out_h * out_w;
        const int kdim = cin_g * kh * kw;
        const std::size_t elem = dtype_size(in_c.dtype());
        for (int g = 0; g < params.groups; ++g) {
            kernels::im2col(raw(in_c), raw_mut(scratch), n, cin, h, w, kh, kw, out_h, out_w,
                            params.stride_h, params.stride_w, params.pad_h, params.pad_w,
                            params.dilation_h, params.dilation_w, g * cin_g, cin_g,
                            static_cast<int>(params.pad_mode), in_c.dtype(), stream);
            const char* w_base = static_cast<const char*>(raw(w_c)) +
                                 static_cast<std::size_t>(g * cout_g) * static_cast<std::size_t>(kdim) *
                                     elem;
            if (params.groups == 1) {
                kernels::gemm(raw(scratch), w_base, raw_mut(nhwc), m, cout, kdim,
                              static_cast<long long>(m) * kdim, static_cast<long long>(cout) * kdim,
                              static_cast<long long>(m) * cout, 1, false, true, nullptr, 0,
                              in_c.dtype(), stream);
            } else {
                auto group_out = empty_like_shape(
                    in_c, TensorShape{std::vector<std::size_t>{static_cast<std::size_t>(m),
                                                               static_cast<std::size_t>(cout_g)}});
                group_out.set_stream(stream);
                kernels::gemm(raw(scratch), w_base, raw_mut(group_out), m, cout_g, kdim,
                              static_cast<long long>(m) * kdim,
                              static_cast<long long>(cout_g) * kdim,
                              static_cast<long long>(m) * cout_g, 1, false, true, nullptr, 0,
                              in_c.dtype(), stream);
                LFS_CUDA_CHECK(cudaMemcpy2DAsync(
                    static_cast<char*>(raw_mut(nhwc)) + static_cast<std::size_t>(g * cout_g) * elem,
                    static_cast<std::size_t>(cout) * elem, raw(group_out),
                    static_cast<std::size_t>(cout_g) * elem, static_cast<std::size_t>(cout_g) * elem,
                    static_cast<std::size_t>(m), cudaMemcpyDeviceToDevice, stream));
            }
        }

        Tensor nchw;
        pack_nhwc_to_nchw(nhwc, nchw);
        if (b_c) {
            kernels::channel_bias(raw_mut(nchw), raw(*b_c), n, cout, out_h * out_w, nchw.dtype(),
                                  stream);
        }
        return nchw;
    }

    Tensor conv_transpose2d(const Tensor& input, const Tensor& weight, const Tensor* bias,
                            const Conv2dParams& params, Tensor* workspace) {
        require_nn_tensor(input, "conv_transpose2d", "input");
        require_nn_tensor(weight, "conv_transpose2d", "weight");
        require_same_dtype_device(input, weight, "conv_transpose2d", "input", "weight");
        LFS_ASSERT_MSG(input.ndim() == 4 && weight.ndim() == 4, "conv_transpose2d expects 4D tensors");
        const int n = static_cast<int>(input.shape()[0]);
        const int cin = static_cast<int>(input.shape()[1]);
        const int hin = static_cast<int>(input.shape()[2]);
        const int win = static_cast<int>(input.shape()[3]);
        const int cin_w = static_cast<int>(weight.shape()[0]);
        const int cout_g = static_cast<int>(weight.shape()[1]);
        const int kh = static_cast<int>(weight.shape()[2]);
        const int kw = static_cast<int>(weight.shape()[3]);
        LFS_ASSERT_MSG(cin == cin_w, "conv_transpose2d input channels must match weight");
        LFS_ASSERT_MSG(cin % params.groups == 0, "conv_transpose2d groups must divide C_in");
        const int cin_g = cin / params.groups;
        const int cout = cout_g * params.groups;
        const auto hw = conv_transpose2d_output_hw(hin, win, kh, kw, params);
        const int out_h = hw.first;
        const int out_w = hw.second;

        Tensor b_s;
        const Tensor in_c = input.contiguous();
        const Tensor w_c = weight.contiguous();
        const Tensor* b_c = nullptr;
        if (bias != nullptr && bias->is_valid()) {
            require_nn_tensor(*bias, "conv_transpose2d", "bias");
            b_s = bias->contiguous();
            b_c = &b_s;
        }

        const bool scatter_s2 =
            kh == 2 && kw == 2 && params.stride_h == 2 && params.stride_w == 2 && params.pad_h == 0 &&
            params.pad_w == 0 && params.dilation_h == 1 && params.dilation_w == 1 &&
            params.groups == 1 && params.output_pad_h == 0 && params.output_pad_w == 0 &&
            in_c.dtype() == DataType::Float16;
        if (scatter_s2) {
            auto out = empty_like_shape(
                in_c, TensorShape{std::vector<std::size_t>{static_cast<std::size_t>(n),
                                                           static_cast<std::size_t>(cout),
                                                           static_cast<std::size_t>(out_h),
                                                           static_cast<std::size_t>(out_w)}});
            pin_operands({&in_c, &w_c});
            const cudaStream_t stream = prepare_inputs_for_stream({&in_c, &w_c}, out.stream());
            out.set_stream(stream);
            auto in_g = in_c.permute({0, 2, 3, 1})
                            .contiguous()
                            .reshape(TensorShape{std::vector<std::size_t>{
                                static_cast<std::size_t>(n) * static_cast<std::size_t>(hin) *
                                    static_cast<std::size_t>(win),
                                static_cast<std::size_t>(cin)}});
            auto w_g = w_c.reshape(TensorShape{std::vector<std::size_t>{
                static_cast<std::size_t>(cin), static_cast<std::size_t>(cout) * 4}});
            in_g.set_stream(stream);
            w_g.set_stream(stream);
            const int m = n * hin * win;
            const int kcol = cout * 4;
            kernels::gemm(raw(in_g), raw(w_g), raw_mut(out), m, kcol, cin,
                          static_cast<long long>(m) * cin, static_cast<long long>(cin) * kcol,
                          static_cast<long long>(n) * cout * out_h * out_w, 1, false, false,
                          b_c ? raw(*b_c) : nullptr, 0, in_c.dtype(), stream, false, nullptr,
                          nullptr, hin, win);
            return out;
        }

        const std::size_t bytes =
            conv_transpose2d_workspace_bytes(in_c.shape(), w_c.shape(), params, in_c.dtype());
        Tensor scratch = ensure_workspace(workspace, bytes, in_c, "conv_transpose2d");
        auto out = Tensor::zeros(
            TensorShape{std::vector<std::size_t>{static_cast<std::size_t>(n),
                                                 static_cast<std::size_t>(cout),
                                                 static_cast<std::size_t>(out_h),
                                                 static_cast<std::size_t>(out_w)}},
            in_c.device(), in_c.dtype());
        pin_operands({&in_c, &w_c});
        const cudaStream_t stream = prepare_inputs_for_stream({&in_c, &w_c}, out.stream());
        out.set_stream(stream);
        scratch.set_stream(stream);

        const int m = n * hin * win;
        const int kcol = cout_g * kh * kw;
        for (int g = 0; g < params.groups; ++g) {
            auto in_g = in_c.slice(1, static_cast<std::size_t>(g * cin_g),
                                   static_cast<std::size_t>((g + 1) * cin_g))
                            .contiguous()
                            .permute({0, 2, 3, 1})
                            .contiguous()
                            .reshape(TensorShape{std::vector<std::size_t>{
                                static_cast<std::size_t>(m), static_cast<std::size_t>(cin_g)}});
            auto w_g = w_c.slice(0, static_cast<std::size_t>(g * cin_g),
                                 static_cast<std::size_t>((g + 1) * cin_g))
                           .contiguous()
                           .reshape(TensorShape{std::vector<std::size_t>{
                               static_cast<std::size_t>(cin_g), static_cast<std::size_t>(kcol)}});
            kernels::gemm(raw(in_g), raw(w_g), raw_mut(scratch), m, kcol, cin_g,
                          static_cast<long long>(m) * cin_g, static_cast<long long>(cin_g) * kcol,
                          static_cast<long long>(m) * kcol, 1, false, false, nullptr, 0,
                          in_c.dtype(), stream);
            kernels::col2im(raw(scratch), raw_mut(out), n, cout, out_h, out_w, kh, kw, hin, win,
                            params.stride_h, params.stride_w, params.pad_h, params.pad_w,
                            params.dilation_h, params.dilation_w, g * cout_g, cout_g, in_c.dtype(),
                            stream);
        }
        if (b_c) {
            kernels::channel_bias(raw_mut(out), raw(*b_c), n, cout, out_h * out_w, out.dtype(),
                                  stream);
        }
        return out;
    }

    Tensor resize2d(const Tensor& input, int out_h, int out_w, ResizeMode mode,
                    CoordTransform coord) {
        require_nn_tensor(input, "resize2d", "input");
        LFS_ASSERT_MSG(input.ndim() == 4, "resize2d expects NCHW");
        LFS_ASSERT_MSG(out_h > 0 && out_w > 0, "resize2d output size must be positive");
        const Tensor in_c = input.contiguous();
        auto out = empty_like_shape(
            in_c, TensorShape{std::vector<std::size_t>{
                      in_c.shape()[0], in_c.shape()[1], static_cast<std::size_t>(out_h),
                      static_cast<std::size_t>(out_w)}});
        pin_operands({&in_c});
        const cudaStream_t stream = prepare_inputs_for_stream({&in_c}, out.stream());
        out.set_stream(stream);
        kernels::resize2d(raw(in_c), raw_mut(out), static_cast<int>(in_c.shape()[0]),
                          static_cast<int>(in_c.shape()[1]), static_cast<int>(in_c.shape()[2]),
                          static_cast<int>(in_c.shape()[3]), out_h, out_w, static_cast<int>(mode),
                          static_cast<int>(coord), in_c.dtype(), stream);
        return out;
    }

    Tensor max_pool2d(const Tensor& input, int kernel_h, int kernel_w, int stride_h, int stride_w,
                      int pad_h, int pad_w) {
        require_nn_tensor(input, "max_pool2d", "input");
        LFS_ASSERT_MSG(input.ndim() == 4, "max_pool2d expects NCHW");
        const Tensor in_c = input.contiguous();
        const int n = static_cast<int>(in_c.shape()[0]);
        const int c = static_cast<int>(in_c.shape()[1]);
        const int h = static_cast<int>(in_c.shape()[2]);
        const int w = static_cast<int>(in_c.shape()[3]);
        const int out_h = conv_out_dim(h, kernel_h, stride_h, pad_h, 1);
        const int out_w = conv_out_dim(w, kernel_w, stride_w, pad_w, 1);
        auto out = empty_like_shape(
            in_c, TensorShape{std::vector<std::size_t>{static_cast<std::size_t>(n),
                                                       static_cast<std::size_t>(c),
                                                       static_cast<std::size_t>(out_h),
                                                       static_cast<std::size_t>(out_w)}});
        pin_operands({&in_c});
        const cudaStream_t stream = prepare_inputs_for_stream({&in_c}, out.stream());
        out.set_stream(stream);
        kernels::max_pool2d(raw(in_c), raw_mut(out), n, c, h, w, out_h, out_w, kernel_h, kernel_w,
                            stride_h, stride_w, pad_h, pad_w, in_c.dtype(), stream);
        return out;
    }

    Tensor avg_pool2d(const Tensor& input, int kernel_h, int kernel_w, int stride_h, int stride_w,
                      int pad_h, int pad_w, bool count_include_pad) {
        require_nn_tensor(input, "avg_pool2d", "input");
        LFS_ASSERT_MSG(input.ndim() == 4, "avg_pool2d expects NCHW");
        const Tensor in_c = input.contiguous();
        const int n = static_cast<int>(in_c.shape()[0]);
        const int c = static_cast<int>(in_c.shape()[1]);
        const int h = static_cast<int>(in_c.shape()[2]);
        const int w = static_cast<int>(in_c.shape()[3]);
        const int out_h = conv_out_dim(h, kernel_h, stride_h, pad_h, 1);
        const int out_w = conv_out_dim(w, kernel_w, stride_w, pad_w, 1);
        auto out = empty_like_shape(
            in_c, TensorShape{std::vector<std::size_t>{static_cast<std::size_t>(n),
                                                       static_cast<std::size_t>(c),
                                                       static_cast<std::size_t>(out_h),
                                                       static_cast<std::size_t>(out_w)}});
        pin_operands({&in_c});
        const cudaStream_t stream = prepare_inputs_for_stream({&in_c}, out.stream());
        out.set_stream(stream);
        kernels::avg_pool2d(raw(in_c), raw_mut(out), n, c, h, w, out_h, out_w, kernel_h, kernel_w,
                            stride_h, stride_w, pad_h, pad_w, count_include_pad, in_c.dtype(),
                            stream);
        return out;
    }

    Tensor gelu(const Tensor& input, GELUApprox approx) {
        require_nn_tensor(input, "gelu", "input");
        const Tensor in_c = input.contiguous();
        auto out = empty_like_shape(in_c, in_c.shape());
        pin_operands({&in_c});
        const cudaStream_t stream = prepare_inputs_for_stream({&in_c}, out.stream());
        out.set_stream(stream);
        kernels::gelu(raw(in_c), raw_mut(out), in_c.numel(), static_cast<int>(approx), in_c.dtype(),
                      stream);
        return out;
    }

    Tensor silu(const Tensor& input) {
        require_nn_tensor(input, "silu", "input");
        const Tensor in_c = input.contiguous();
        auto out = empty_like_shape(in_c, in_c.shape());
        pin_operands({&in_c});
        const cudaStream_t stream = prepare_inputs_for_stream({&in_c}, out.stream());
        out.set_stream(stream);
        kernels::silu(raw(in_c), raw_mut(out), in_c.numel(), in_c.dtype(), stream);
        return out;
    }

    Tensor relu(const Tensor& input) {
        require_nn_tensor(input, "relu", "input");
        const Tensor in_c = input.contiguous();
        auto out = empty_like_shape(in_c, in_c.shape());
        pin_operands({&in_c});
        const cudaStream_t stream = prepare_inputs_for_stream({&in_c}, out.stream());
        out.set_stream(stream);
        kernels::relu(raw(in_c), raw_mut(out), in_c.numel(), in_c.dtype(), stream);
        return out;
    }

    Tensor cast(const Tensor& input, DataType dtype) {
        tensor_contract::require_valid(input, "cast", "input", LFS_SOURCE_SITE_CURRENT());
        return input.to(dtype);
    }

    std::array<Tensor, 3> split_qkv(const Tensor& qkv, int heads) {
        require_nn_tensor(qkv, "split_qkv", "qkv");
        LFS_ASSERT_MSG(heads > 0, "split_qkv heads must be positive");
        const Tensor in_c = qkv.contiguous();
        LFS_ASSERT_MSG(in_c.ndim() == 3 || in_c.ndim() == 5, "split_qkv expects [B,S,3HD] or [B,S,3,H,D]");
        const int b = static_cast<int>(in_c.shape()[0]);
        const int seq = static_cast<int>(in_c.shape()[1]);
        int d = 0;
        if (in_c.ndim() == 5) {
            LFS_ASSERT_MSG(static_cast<int>(in_c.shape()[3]) == heads, "split_qkv head count mismatch");
            d = static_cast<int>(in_c.shape()[4]);
        } else {
            const int packed = static_cast<int>(in_c.shape()[2]);
            LFS_ASSERT_MSG(packed % (3 * heads) == 0, "split_qkv packed width is not 3*H*D");
            d = packed / (3 * heads);
        }
        const auto shape = TensorShape{std::vector<std::size_t>{
            static_cast<std::size_t>(b), static_cast<std::size_t>(heads),
            static_cast<std::size_t>(seq), static_cast<std::size_t>(d)}};
        auto q = empty_like_shape(in_c, shape);
        auto k = empty_like_shape(in_c, shape);
        auto v = empty_like_shape(in_c, shape);
        pin_operands({&in_c});
        const cudaStream_t stream = prepare_inputs_for_stream({&in_c}, q.stream());
        q.set_stream(stream);
        k.set_stream(stream);
        v.set_stream(stream);
        kernels::split_qkv(raw(in_c), raw_mut(q), raw_mut(k), raw_mut(v), b, seq, heads, d,
                           in_c.dtype(), stream);
        return {std::move(q), std::move(k), std::move(v)};
    }

    Tensor merge_heads(const Tensor& context) {
        require_nn_tensor(context, "merge_heads", "context");
        LFS_ASSERT_MSG(context.ndim() == 4, "merge_heads expects [B, H, S, D]");
        const Tensor in_c = context.contiguous();
        const int b = static_cast<int>(in_c.shape()[0]);
        const int heads = static_cast<int>(in_c.shape()[1]);
        const int seq = static_cast<int>(in_c.shape()[2]);
        const int d = static_cast<int>(in_c.shape()[3]);
        auto out = empty_like_shape(
            in_c, TensorShape{std::vector<std::size_t>{
                      static_cast<std::size_t>(b), static_cast<std::size_t>(seq),
                      static_cast<std::size_t>(heads) * static_cast<std::size_t>(d)}});
        pin_operands({&in_c});
        const cudaStream_t stream = prepare_inputs_for_stream({&in_c}, out.stream());
        out.set_stream(stream);
        kernels::merge_heads(raw(in_c), raw_mut(out), b, heads, seq, d, in_c.dtype(), stream);
        return out;
    }

    Tensor uv_grid(int height, int width, float aspect, DataType dtype, Device device,
                   cudaStream_t stream) {
        LFS_ASSERT_MSG(height > 0 && width > 0, "uv_grid size must be positive");
        LFS_ASSERT_MSG(device == Device::CUDA, "uv_grid requires CUDA");
        LFS_ASSERT_MSG(dtype == DataType::Float16 || dtype == DataType::Float32,
                       "uv_grid dtype must be float16 or float32");
        const float span_x = aspect / std::sqrt(1.0f + aspect * aspect);
        const float span_y = 1.0f / std::sqrt(1.0f + aspect * aspect);
        const float u0 = -span_x * static_cast<float>(width - 1) / static_cast<float>(width);
        const float u1 = span_x * static_cast<float>(width - 1) / static_cast<float>(width);
        const float v0 = -span_y * static_cast<float>(height - 1) / static_cast<float>(height);
        const float v1 = span_y * static_cast<float>(height - 1) / static_cast<float>(height);
        auto out = Tensor::empty(
            TensorShape{std::vector<std::size_t>{1, 2, static_cast<std::size_t>(height),
                                                 static_cast<std::size_t>(width)}},
            device, dtype);
        out.set_stream(stream);
        kernels::uv_grid(raw_mut(out), height, width, u0, u1, v0, v1, dtype, stream);
        return out;
    }

    Tensor residual_scale(const Tensor& x, const Tensor& hidden, const Tensor& gamma) {
        require_nn_tensor(x, "residual_scale", "x");
        require_nn_tensor(hidden, "residual_scale", "hidden");
        require_nn_tensor(gamma, "residual_scale", "gamma");
        require_same_dtype_device(x, hidden, "residual_scale", "x", "hidden");
        require_same_dtype_device(x, gamma, "residual_scale", "x", "gamma");
        LFS_ASSERT_MSG(x.shape() == hidden.shape(), "residual_scale x/hidden shape mismatch");
        const int cols = static_cast<int>(x.shape()[x.ndim() - 1]);
        LFS_ASSERT_MSG(gamma.numel() == static_cast<std::size_t>(cols),
                       "residual_scale gamma must match the last dim");
        const Tensor x_c = x.contiguous();
        const Tensor h_c = hidden.contiguous();
        const Tensor g_c = gamma.contiguous();
        auto out = empty_like_shape(x_c, x_c.shape());
        pin_operands({&x_c, &h_c, &g_c});
        const cudaStream_t stream = prepare_inputs_for_stream({&x_c, &h_c, &g_c}, out.stream());
        out.set_stream(stream);
        const int rows = static_cast<int>(x_c.numel() / static_cast<std::size_t>(cols));
        kernels::residual_scale(raw(x_c), raw(h_c), raw(g_c), raw_mut(out), rows, cols, x_c.dtype(),
                                stream);
        return out;
    }

} // namespace lfs::core::nn
