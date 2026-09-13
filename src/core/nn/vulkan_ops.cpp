/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_ops.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor/backend/vulkan/vk_pipelines.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor/internal/tensor_impl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace lfs::core::nn::vulkan {
    namespace {
        struct Push {
            uint64_t input_address = 0, output_address = 0;
            uint32_t total = 0, step = 0;
            int32_t channels = 0, height = 0, width = 0, out_height = 0, out_width = 0;
            int32_t kernel_h = 0, kernel_w = 0, stride_h = 0, stride_w = 0;
            int32_t pad_h = 0, pad_w = 0, dilation_h = 0, dilation_w = 0;
            int32_t offset = 0, columns = 0, mode = 0, coord = 0, include_pad = 0;
            float u0 = 0, u1 = 0, v0 = 0, v1 = 0;
        };
        static_assert(sizeof(Push) == 112);

        Tensor fp32(const Tensor& t) { return t.to(DataType::Float32).contiguous(); }

        Tensor empty(const Tensor& like, const TensorShape& shape) {
            return internal::allocate_like(like, shape, DataType::Float32);
        }

        void dispatch(uint32_t kind, const Tensor& input, Tensor& output, Push push) {
            if (output.numel() == 0)
                return;
            LFS_ASSERT_MSG(output.numel() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                           "NN shader output exceeds int32 indexing");
            const auto src = internal::storage_ref(input);
            const auto dst = internal::storage_ref(output);
            const auto context = internal::acquire_vulkan_context();
            const uint32_t groups = internal::vk::dispatch_groups(*context, output.numel());
            push.input_address = internal::vk::address(src);
            push.output_address = internal::vk::address(dst);
            push.total = static_cast<uint32_t>(output.numel());
            push.step = groups * internal::vk::kLocalSize;
            const std::array constants{kind};
            const auto& pipeline = context->pipelines().specialized("inference", sizeof(Push), constants);
            const std::array reads{src};
            const std::array writes{dst};
            context->recorders().record(reads, writes, [&](VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
                vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                vkCmdDispatch(command, groups, 1, 1);
            });
        }

        Tensor affine(Tensor out, const Tensor* bias, Activation activation,
                      const Tensor* residual, const Tensor* scale) {
            if (bias)
                out = out.add(fp32(*bias));
            out = activate(out, activation);
            if (scale)
                out = out.mul(fp32(*scale));
            if (residual)
                out = out.add(fp32(*residual).reshape(out.shape()));
            return out;
        }
    } // namespace

    Tensor activate(const Tensor& input, Activation activation) {
        if (activation == Activation::None)
            return input;
        const auto x = fp32(input);
        auto out = empty(x, x.shape());
        dispatch(4, x, out, Push{.mode = static_cast<int32_t>(activation)});
        return out.to(input.dtype());
    }

    Tensor gemm(const Tensor& a, const Tensor& b, bool trans_b, const Tensor* bias,
                Activation activation, const Tensor* residual, const Tensor* scale) {
        const auto x = fp32(a);
        const auto source_weight = fp32(b);
        const auto w = trans_b ? source_weight.transpose(-2, -1).contiguous() : source_weight;
        const auto m = a.shape()[a.ndim() - 2], k = a.shape()[a.ndim() - 1];
        const auto n = w.shape()[w.ndim() - 1];
        const auto batches = a.numel() / (m * k);
        Tensor out;
        if (w.numel() == k * n) {
            out = x.reshape(TensorShape{batches * m, k}).mm(w.reshape(TensorShape{k, n}));
        } else {
            out = x.reshape(TensorShape{batches, m, k}).bmm(w.reshape(TensorShape{batches, k, n}));
        }
        std::vector<size_t> shape;
        for (size_t i = 0; i + 1 < a.ndim(); ++i)
            shape.push_back(a.shape()[i]);
        shape.push_back(n);
        out = out.reshape(TensorShape(shape));
        return affine(std::move(out), bias, activation, residual, scale).to(a.dtype());
    }

    Tensor norm(const Tensor& input, const Tensor& weight, const Tensor* bias, float eps) {
        const auto source = fp32(input);
        const auto x = bias ? source.sub(source.mean(-1, true)) : source;
        auto variance = x.mul(x).mean(-1, true);
        auto out = x.div(variance.add(eps).sqrt()).mul(fp32(weight));
        if (bias)
            out = out.add(fp32(*bias));
        return out.to(input.dtype());
    }

    Tensor softmax(const Tensor& input, const Tensor* mask) {
        const auto source = fp32(input);
        const auto x = mask ? source.add(fp32(*mask)) : source;
        auto shifted = x.sub(x.max(-1, true));
        auto e = shifted.exp();
        return e.div(e.sum(-1, true)).to(input.dtype());
    }

    Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* mask, float scale) {
        const int batches = static_cast<int>(q.shape()[0] * q.shape()[1]);
        const int nq = static_cast<int>(q.shape()[2]), nk = static_cast<int>(k.shape()[2]);
        const int d = static_cast<int>(q.shape()[3]);
        const auto queries = fp32(q).reshape({batches, nq, d});
        const auto keys = fp32(k).reshape({batches, nk, d}).transpose(1, 2).contiguous();
        const auto values = fp32(v).reshape({batches, nk, d});
        auto out = empty(q, TensorShape{static_cast<size_t>(batches), static_cast<size_t>(nq), static_cast<size_t>(d)});
        Tensor additive;
        if (mask) {
            additive = fp32(*mask).expand(TensorShape{q.shape()[0], q.shape()[1], q.shape()[2], k.shape()[2]});
        }
        // Bound score storage independently of the number of queries. Global
        // SAM2 attention must not allocate a full Nq x Nk matrix per head.
        const int tile = std::max(1, std::min(64, (8 * 1024 * 1024) / std::max(1, batches * nk * 4)));
        for (int start = 0; start < nq; start += tile) {
            const int end = std::min(nq, start + tile);
            auto scores = queries.slice(1, start, end).contiguous().bmm(keys).mul(scale);
            Tensor mask_tile;
            if (mask)
                mask_tile = additive.slice(2, start, end).contiguous().reshape({batches, end - start, nk});
            auto probs = softmax(scores, mask ? &mask_tile : nullptr);
            out.slice(1, start, end).copy_from(probs.bmm(values));
        }
        return out.reshape(q.shape()).to(q.dtype());
    }

    Tensor conv(const Tensor& input, const Tensor& weight, const Tensor* bias,
                const Conv2dParams& params, bool transpose) {
        const int n = static_cast<int>(input.shape()[0]), cin = static_cast<int>(input.shape()[1]);
        const int h = static_cast<int>(input.shape()[2]), w = static_cast<int>(input.shape()[3]);
        const int kh = static_cast<int>(weight.shape()[2]), kw = static_cast<int>(weight.shape()[3]);
        const int cout = transpose ? static_cast<int>(weight.shape()[1]) * params.groups : static_cast<int>(weight.shape()[0]);
        const int cig = cin / params.groups, cog = cout / params.groups;
        const auto [oh, ow] = transpose ? conv_transpose2d_output_hw(h, w, kh, kw, params)
                                        : conv2d_output_hw(h, w, kh, kw, params);
        LFS_ASSERT_MSG(oh > 0 && ow > 0, "NN convolution requires positive output dimensions");
        const auto x = fp32(input);
        const auto weights = fp32(weight);
        auto out = empty(x, TensorShape{static_cast<size_t>(n), static_cast<size_t>(cout), static_cast<size_t>(oh), static_cast<size_t>(ow)});
        Push push{.channels = cig, .height = h, .width = w, .out_height = oh, .out_width = ow, .kernel_h = kh, .kernel_w = kw, .stride_h = params.stride_h, .stride_w = params.stride_w, .pad_h = params.pad_h, .pad_w = params.pad_w, .dilation_h = params.dilation_h, .dilation_w = params.dilation_w, .mode = static_cast<int32_t>(params.pad_mode)};
        for (int batch = 0; batch < n; ++batch) {
            for (int group = 0; group < params.groups; ++group) {
                auto image = x.slice(0, batch, batch + 1).slice(1, group * cig, (group + 1) * cig).contiguous();
                // Reshape the full dense output before slicing channels. A
                // grouped NCHW slice can materialize when reshaped, losing the
                // connection to the output storage we must write.
                auto destination = out.reshape({n, cout, oh * ow}).slice(0, batch, batch + 1).squeeze(0).slice(0, group * cog, (group + 1) * cog);
                if (transpose) {
                    auto wt = weights.slice(0, group * cig, (group + 1) * cig).reshape({cig, cog * kh * kw}).transpose(0, 1).contiguous();
                    auto columns = wt.mm(image.reshape({cig, h * w}));
                    auto result = empty(x, destination.shape());
                    dispatch(1, columns, result, push);
                    destination.copy_from(result);
                } else {
                    auto wt = weights.slice(0, group * cog, (group + 1) * cog).reshape({cog, cig * kh * kw});
                    // im2col scratch is capped at 16 MiB even for large images.
                    const int chunk = std::max(1, std::min(1024, (16 * 1024 * 1024) / (4 * cig * kh * kw)));
                    for (int offset = 0; offset < oh * ow; offset += chunk) {
                        const int count = std::min(chunk, oh * ow - offset);
                        auto columns = empty(x, TensorShape{static_cast<size_t>(cig * kh * kw), static_cast<size_t>(count)});
                        push.offset = offset;
                        push.columns = count;
                        dispatch(0, image, columns, push);
                        destination.slice(1, offset, offset + count).copy_from(wt.mm(columns));
                    }
                }
            }
        }
        if (bias)
            out = out.add(fp32(*bias).reshape({1, cout, 1, 1}));
        return activate(out, params.activation).to(input.dtype());
    }

    Tensor resize(const Tensor& input, int height, int width, ResizeMode mode, CoordTransform coord) {
        auto x = fp32(input);
        auto out = empty(x, TensorShape{input.shape()[0], input.shape()[1], static_cast<size_t>(height), static_cast<size_t>(width)});
        dispatch(2, x, out, Push{.height = static_cast<int32_t>(input.shape()[2]), .width = static_cast<int32_t>(input.shape()[3]), .out_height = height, .out_width = width, .mode = static_cast<int32_t>(mode), .coord = static_cast<int32_t>(coord)});
        return out.to(input.dtype());
    }

    Tensor pool(const Tensor& input, int kh, int kw, int sh, int sw, int ph, int pw, bool average, bool count_include_pad) {
        const int h = static_cast<int>(input.shape()[2]), w = static_cast<int>(input.shape()[3]);
        const int oh = (h + 2 * ph - kh) / sh + 1, ow = (w + 2 * pw - kw) / sw + 1;
        LFS_ASSERT_MSG(oh > 0 && ow > 0, "NN pooling requires positive output dimensions");
        auto x = fp32(input);
        auto out = empty(x, TensorShape{input.shape()[0], input.shape()[1], static_cast<size_t>(oh), static_cast<size_t>(ow)});
        dispatch(3, x, out, Push{.height = h, .width = w, .out_height = oh, .out_width = ow, .kernel_h = kh, .kernel_w = kw, .stride_h = sh, .stride_w = sw, .pad_h = ph, .pad_w = pw, .mode = average ? 1 : 0, .include_pad = count_include_pad ? 1 : 0});
        return out.to(input.dtype());
    }

    Tensor window_partition(const Tensor& input, int window) {
        const int b = static_cast<int>(input.shape()[0]), h = static_cast<int>(input.shape()[1]);
        const int w = static_cast<int>(input.shape()[2]), c = static_cast<int>(input.shape()[3]);
        const int nh = (h + window - 1) / window, nw = (w + window - 1) / window;
        Tensor padded = input;
        if (nh * window != h || nw * window != w) {
            padded = internal::allocate_zeros_like(input, TensorShape{static_cast<size_t>(b), static_cast<size_t>(nh * window), static_cast<size_t>(nw * window), static_cast<size_t>(c)}, input.dtype());
            padded.slice(1, 0, h).slice(2, 0, w).copy_from(input);
        }
        return padded.reshape({b, nh, window, nw, window, c}).permute({0, 1, 3, 2, 4, 5}).contiguous().reshape({b * nh * nw, window, window, c});
    }

    Tensor window_unpartition(const Tensor& input, int window, int height, int width) {
        const int nh = (height + window - 1) / window, nw = (width + window - 1) / window;
        const int b = static_cast<int>(input.shape()[0]) / (nh * nw), c = static_cast<int>(input.shape()[3]);
        return input.reshape({b, nh, nw, window, window, c}).permute({0, 1, 3, 2, 4, 5}).contiguous().reshape({b, nh * window, nw * window, c}).slice(1, 0, height).slice(2, 0, width).contiguous();
    }

    std::array<Tensor, 3> split_qkv(const Tensor& input, int heads) {
        const int b = static_cast<int>(input.shape()[0]), seq = static_cast<int>(input.shape()[1]);
        const int d = static_cast<int>(input.shape()[2]) / (3 * heads);
        auto t = input.reshape({b, seq, 3, heads, d}).permute({2, 0, 3, 1, 4});
        return {t.slice(0, 0, 1).squeeze(0).contiguous(), t.slice(0, 1, 2).squeeze(0).contiguous(), t.slice(0, 2, 3).squeeze(0).contiguous()};
    }

    Tensor merge_heads(const Tensor& input) {
        return input.permute({0, 2, 1, 3}).contiguous().reshape({static_cast<int>(input.shape()[0]), static_cast<int>(input.shape()[2]), static_cast<int>(input.shape()[1] * input.shape()[3])});
    }

    Tensor fourier_pe(const Tensor& coords, const Tensor& gaussian) {
        const auto count = coords.numel() / 2;
        auto projection = fp32(coords).reshape(TensorShape{count, 2}).mul(2.0f).sub(1.0f).mm(fp32(gaussian)).mul(6.2831853071795864769f);
        std::vector<size_t> shape;
        for (size_t i = 0; i + 1 < coords.ndim(); ++i)
            shape.push_back(coords.shape()[i]);
        shape.push_back(gaussian.shape()[1] * 2);
        return Tensor::cat({projection.sin(), projection.cos()}, -1).reshape(TensorShape(shape)).to(coords.dtype());
    }

    Tensor grid(const Tensor& like, int height, int width, float u0, float u1, float v0, float v1) {
        auto out = empty(like, TensorShape{1, 2, static_cast<size_t>(height), static_cast<size_t>(width)});
        dispatch(5, like, out, Push{.height = height, .width = width, .u0 = u0, .u1 = u1, .v0 = v0, .v1 = v1});
        return out.to(like.dtype());
    }
} // namespace lfs::core::nn::vulkan
