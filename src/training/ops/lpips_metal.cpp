/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal LpipsOps; see ops/lpips_cuda.cpp for the reference behaviour.

#include "metal_families.hpp"
#include "metal_kernels.hpp"

#include <format>
#include <stdexcept>
#include <string_view>

namespace lfs::training {
    namespace {
        using core::DataType;
        using core::Tensor;
        namespace mk = metal;

        void require(const bool condition, const std::string_view what, const Tensor& observed) {
            if (!condition)
                throw std::invalid_argument(std::format("LPIPS {}, got {} {}", what, core::dtype_name(observed.dtype()),
                                                        observed.shape().str()));
        }

        bool is_half(const Tensor& tensor) { return tensor.dtype() == DataType::Float16; }

        struct TapsParams {
            uint64_t weight, taps;
            uint32_t cout, cin;
        };

        void weight_taps(const Tensor& weight, Tensor& taps) {
            require(is_half(weight) && weight.ndim() == 4 && weight.shape()[2] == 3 && weight.shape()[3] == 3,
                    "weight taps need a float16 [C_out, C_in, 3, 3] weight", weight);
            require(is_half(taps) && taps.numel() == weight.numel(),
                    "weight taps need a float16 [9, C_out, C_in] destination", taps);
            const TapsParams params{mk::address(weight), mk::address(taps), static_cast<uint32_t>(weight.shape()[0]),
                                    static_cast<uint32_t>(weight.shape()[1])};
            mk::launch_items("lpips_weight_taps", params, {&weight, &taps}, weight.numel());
        }

        struct RgbParams {
            uint64_t input, weight, bias, output;
            float shift[3], scale[3];
            uint32_t official;
            int32_t n, h, w;
        };

        void rgb_conv(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
                      const gpu_ops::RGBConvParams& p) {
            require(input.ndim() == 4 && input.shape()[1] == 3 && input.dtype() == DataType::Float32,
                    "RGB conv needs a float32 [N, 3, H, W] input", input);
            require(is_half(weight) && weight.numel() == 64 * 27, "RGB conv needs float16 [64, 3, 3, 3] weights", weight);
            require(is_half(bias) && bias.numel() == 64, "RGB conv needs 64 float16 biases", bias);
            const auto n = input.shape()[0], h = input.shape()[2], w = input.shape()[3];
            require(is_half(output) && output.numel() == n * 64 * h * w,
                    "RGB conv needs a float16 [N, 64, H, W] output", output);
            const RgbParams params{.input = mk::address(input),
                                   .weight = mk::address(weight),
                                   .bias = mk::address(bias),
                                   .output = mk::address(output),
                                   .shift = {p.shift[0], p.shift[1], p.shift[2]},
                                   .scale = {p.scale[0], p.scale[1], p.scale[2]},
                                   .official = p.official_scaling ? 1u : 0u,
                                   .n = static_cast<int32_t>(n),
                                   .h = static_cast<int32_t>(h),
                                   .w = static_cast<int32_t>(w)};
            mk::launch_items("lpips_rgb_conv", params, {&input, &weight, &bias, &output}, output.numel());
        }

        struct ConvParams {
            uint64_t input, weight, bias, output;
            int32_t n, cin, h, w, cout, out_h, out_w, pad_h, pad_w, pad_mode, activation;
            uint32_t half_type;
        };

        // Stride and dilation are 1 and the kernel 3x3, as conv2d_implicit assumes;
        // taps and scratch only feed CUDA's tensor-core path.
        void convolution(const Tensor& input, const Tensor& weight, const Tensor& /*taps*/, const Tensor& bias,
                         Tensor& output, Tensor& /*scratch*/, const gpu_ops::ConvParams& p) {
            require(input.ndim() == 4 && (is_half(input) || input.dtype() == DataType::Float32),
                    "convolution needs a float16 or float32 [N, C, H, W] input", input);
            require(weight.ndim() == 4 && weight.dtype() == input.dtype() && weight.shape()[1] == input.shape()[1] &&
                        weight.shape()[2] == 3 && weight.shape()[3] == 3,
                    "convolution needs a [C_out, C, 3, 3] weight of the input dtype", weight);
            require(!bias.is_valid() || (bias.dtype() == input.dtype() && bias.numel() == weight.shape()[0]),
                    "convolution needs C_out biases of the input dtype", bias);
            require(output.ndim() == 4 && output.dtype() == input.dtype() && output.shape()[0] == input.shape()[0] &&
                        output.shape()[1] == weight.shape()[0],
                    "convolution needs an [N, C_out, H_out, W_out] output of the input dtype", output);
            if (p.activation != core::nn::Activation::None && p.activation != core::nn::Activation::Relu)
                throw std::invalid_argument(std::format("LPIPS convolution supports no activation or ReLU, got {}",
                                                        static_cast<int>(p.activation)));
            const ConvParams params{mk::address(input), mk::address(weight), mk::address(bias), mk::address(output),
                                    static_cast<int32_t>(input.shape()[0]), static_cast<int32_t>(input.shape()[1]),
                                    static_cast<int32_t>(input.shape()[2]), static_cast<int32_t>(input.shape()[3]),
                                    static_cast<int32_t>(weight.shape()[0]), static_cast<int32_t>(output.shape()[2]),
                                    static_cast<int32_t>(output.shape()[3]), p.pad_h, p.pad_w,
                                    static_cast<int32_t>(p.pad_mode), static_cast<int32_t>(p.activation),
                                    is_half(input) ? 1u : 0u};
            mk::launch_items("lpips_conv", params, {&input, &weight, &bias, &output}, output.numel());
        }

        struct PoolParams {
            uint64_t x, y, lin, score, pooled_x, pooled_y;
            int32_t n, channels, h, w;
            int32_t y0, y1, x0, x1;
            float inverse_count;
            uint64_t weights;
            int32_t weights_width, weights_y0, weights_x0;
        };

        // score accumulates; pooled outputs are written only when both are bound.
        void pool_reduce(const Tensor& x, const Tensor& y, const Tensor& weight, Tensor& score, Tensor& pooled_x,
                         Tensor& pooled_y, const gpu_ops::PoolReduceParams& p) {
            require(is_half(x) && x.ndim() == 4, "pool reduce needs float16 [N, C, H, W] features", x);
            require(is_half(y) && y.shape() == x.shape(), "pool reduce needs matching float16 features", y);
            require(is_half(weight) && weight.numel() == x.shape()[1],
                    "pool reduce needs one float16 lin weight per channel", weight);
            require(score.dtype() == DataType::Float32 && score.numel() >= 1, "pool reduce needs a float32 score", score);
            const auto n = x.shape()[0], c = x.shape()[1], h = x.shape()[2], w = x.shape()[3];
            const bool pool = pooled_x.is_valid() && pooled_y.is_valid();
            if (pool) {
                const auto pooled = n * c * (h / 2) * (w / 2);
                require(is_half(pooled_x) && pooled_x.numel() == pooled,
                        "pool reduce needs a float16 [N, C, H/2, W/2] pool", pooled_x);
                require(is_half(pooled_y) && pooled_y.numel() == pooled,
                        "pool reduce needs a float16 [N, C, H/2, W/2] pool", pooled_y);
            }
            if (auto error = gpu_ops::pool_reduce_region_error(x, p); !error.empty())
                throw std::invalid_argument(std::move(error));
            const PoolParams params{mk::address(x), mk::address(y), mk::address(weight), mk::address(score),
                                    pool ? mk::address(pooled_x) : 0, pool ? mk::address(pooled_y) : 0,
                                    static_cast<int32_t>(n), static_cast<int32_t>(c), static_cast<int32_t>(h),
                                    static_cast<int32_t>(w), p.y0, p.y1, p.x0, p.x1, p.inverse_count,
                                    p.weights ? mk::address(*p.weights) : 0, p.weights_width, p.weights_y0, p.weights_x0};
            const Tensor no_weights;
            mk::launch_items("lpips_pool_reduce", params, {&x, &y, &weight, &score, &pooled_x, &pooled_y, p.weights ? p.weights : &no_weights},
                             n * ((h + 1) / 2) * w);
        }
    } // namespace

    const gpu_ops::LpipsOps& metal_lpips_ops() {
        static const gpu_ops::LpipsOps ops{{
            .weight_taps = weight_taps,
            .rgb_conv = rgb_conv,
            .convolution = convolution,
            .pool_reduce = pool_reduce,
        }};
        return ops;
    }
} // namespace lfs::training
