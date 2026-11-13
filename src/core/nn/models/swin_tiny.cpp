/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/nn/models/swin_tiny.hpp"
#include "core/nn/ops.hpp"
#include <map>

namespace lfs::core::nn::models {
    namespace {
        constexpr int kDepths[]{2, 2, 6, 2};
        constexpr auto kPrefix = "model.backbone.conv_encoder.model.";

        lfs::Error swin_error(std::string detail) {
            return lfs::make_error({.code = lfs::ErrorCode::InvalidArgument,
                                    .domain = lfs::ErrorDomain::Core,
                                    .user_message = "Image encoding failed",
                                    .detail = std::move(detail),
                                    .detection = LFS_SOURCE_SITE_CURRENT()});
        }

        std::map<std::string, TensorShape> expected_weights() {
            std::map<std::string, TensorShape> result;
            const auto add = [&](const std::string& name, std::initializer_list<size_t> shape) {
                result.emplace(name, TensorShape(std::vector<size_t>(shape)));
            };
            const auto norm = [&](const std::string& p, size_t c) {
                add(p + ".weight", {c});
                add(p + ".bias", {c});
            };
            const auto linear = [&](const std::string& p, size_t in, size_t out) {
                add(p + ".weight", {out, in});
                add(p + ".bias", {out});
            };
            add("embeddings.patch_embeddings.projection.weight", {96, 3, 4, 4});
            add("embeddings.patch_embeddings.projection.bias", {96});
            norm("embeddings.norm", 96);
            for (int s = 0; s < 4; ++s) {
                const size_t c = size_t(96) << s;
                const auto stage = "encoder.layers." + std::to_string(s);
                for (int b = 0; b < kDepths[s]; ++b) {
                    const auto p = stage + ".blocks." + std::to_string(b);
                    norm(p + ".layernorm_before", c);
                    norm(p + ".layernorm_after", c);
                    for (const auto* qkv : {"query", "key", "value"})
                        linear(p + ".attention.self." + qkv, c, c);
                    add(p + ".attention.self.relative_position_bias_table", {169, c / 32});
                    linear(p + ".attention.output.dense", c, c);
                    linear(p + ".intermediate.dense", c, c * 4);
                    linear(p + ".output.dense", c * 4, c);
                }
                if (s < 3) {
                    norm(stage + ".downsample.norm", c * 4);
                    add(stage + ".downsample.reduction.weight", {c * 2, c * 4});
                }
                if (s > 0)
                    norm("hidden_states_norms.stage" + std::to_string(s + 1), c);
            }
            return result;
        }

        Tensor pad_bhwc(const Tensor& x, int multiple) {
            const auto h = x.size(1), w = x.size(2);
            const auto hp = (h + multiple - 1) / multiple * multiple;
            const auto wp = (w + multiple - 1) / multiple * multiple;
            if (h == hp && w == wp)
                return x;
            auto padded = Tensor::zeros({x.size(0), hp, wp, x.size(3)}, x.device(), x.dtype());
            padded.slice(1, 0, h).slice(2, 0, w).copy_from(x);
            return padded;
        }

        Tensor roll_bhwc(const Tensor& x, int shift) {
            const auto roll = [&](const Tensor& value, size_t dim) {
                const auto n = value.size(dim);
                const auto offset = size_t((shift % int(n) + int(n)) % int(n));
                if (!offset)
                    return value;
                return Tensor::cat({value.slice(dim, n - offset, n), value.slice(dim, 0, n - offset)}, int(dim));
            };
            return roll(roll(x, 1), 2);
        }

        Tensor upload_indices(const std::vector<int>& values) {
            return Tensor::from_vector(values, {values.size()}, Device::CPU).to(Device::CUDA);
        }

        struct ShiftedWindows {
            Tensor order, restore, mask;
            int interior_count = 0;
        };

        ShiftedWindows shifted_windows(int hp, int wp) {
            std::vector<int> inside, boundary, restore((hp / 7) * (wp / 7));
            std::vector<float> mask;
            const auto region = [](int coordinate, int extent) {
                return coordinate < extent - 7 ? 0 : coordinate < extent - 3 ? 1
                                                                             : 2;
            };
            for (int y = 0; y < hp / 7; ++y)
                for (int x = 0; x < wp / 7; ++x) {
                    const int index = y * (wp / 7) + x;
                    if (y + 1 < hp / 7 && x + 1 < wp / 7) {
                        inside.push_back(index);
                        continue;
                    }
                    boundary.push_back(index);
                    int labels[49];
                    for (int i = 0; i < 49; ++i)
                        labels[i] = region(y * 7 + i / 7, hp) * 3 + region(x * 7 + i % 7, wp);
                    for (int i = 0; i < 49; ++i)
                        for (int j = 0; j < 49; ++j)
                            mask.push_back(labels[i] == labels[j] ? 0.f : -100.f);
                }
            ShiftedWindows result;
            result.interior_count = int(inside.size());
            for (size_t i = 0; i < inside.size(); ++i)
                restore[inside[i]] = int(i);
            for (size_t i = 0; i < boundary.size(); ++i)
                restore[boundary[i]] = int(inside.size() + i);
            if (!inside.empty()) {
                inside.insert(inside.end(), boundary.begin(), boundary.end());
                result.order = upload_indices(inside);
                result.restore = upload_indices(restore);
            }
            result.mask = Tensor::from_vector(mask, {boundary.size(), 1, 49, 49}, Device::CPU).to(Device::CUDA);
            return result;
        }
    } // namespace

    lfs::Result<SwinTiny> SwinTiny::load(const WeightFile& archive, const Device device) {
        if (device != Device::CUDA)
            return swin_error("Swin inference requires CUDA");
        const auto expected = expected_weights();
        for (const auto& [name, shape] : expected) {
            const auto* info = archive.info(std::string(kPrefix) + name);
            if (!info || info->shape != shape || info->dtype != DataType::Float32)
                return swin_error("Invalid Swin-T weight: " + name);
        }
        SwinTiny model;
        model.weights_.reserve(expected.size());
        for (const auto& [name, shape] : expected) {
            auto value = archive.load(std::string(kPrefix) + name, device, DataType::Float32);
            if (!value)
                return std::move(value.error());
            model.weights_.emplace(name, std::move(*value));
        }
        std::vector<int> relative;
        for (int q = 0; q < 49; ++q)
            for (int k = 0; k < 49; ++k)
                relative.push_back((q / 7 - k / 7 + 6) * 13 + q % 7 - k % 7 + 6);
        model.relative_indices_ = upload_indices(relative);
        return model;
    }

    lfs::Result<std::array<Tensor, 3>> SwinTiny::encode(const Tensor& image) const {
        if (!image.is_valid() || image.device() != Device::CUDA || image.dtype() != DataType::Float32 ||
            image.ndim() != 4 || image.size(0) != 1 || image.size(1) != 3 ||
            image.size(2) < 1 || image.size(3) < 1 || image.size(2) > 1333 || image.size(3) > 1333)
            return swin_error("Invalid Swin RGB image shape or type");
        if (weights_.empty())
            return swin_error("Swin weights are not loaded");
        const auto w = [&](const std::string& name) -> const Tensor& { return weights_.at(name); };
        const auto norm = [&](const Tensor& x, const std::string& p) {
            return layer_norm(x, w(p + ".weight"), w(p + ".bias"), 1e-5f);
        };
        const auto dense = [&](const Tensor& x, const std::string& p, Activation act = Activation::None) {
            return linear(x, w(p + ".weight"), &w(p + ".bias"), act);
        };
        auto x = pad_bhwc(image.permute({0, 2, 3, 1}), 4).permute({0, 3, 1, 2}).contiguous();
        x = conv2d(x, w("embeddings.patch_embeddings.projection.weight"),
                   &w("embeddings.patch_embeddings.projection.bias"), {.stride_h = 4, .stride_w = 4});
        x = norm(x.permute({0, 2, 3, 1}).contiguous(), "embeddings.norm");
        std::array<Tensor, 3> output;
        for (int s = 0; s < 4; ++s) {
            const int height = int(x.size(1)), width = int(x.size(2)), c = 96 << s, heads = c / 32;
            const int hp = (height + 6) / 7 * 7, wp = (width + 6) / 7 * 7;
            const auto shifted = shifted_windows(hp, wp);
            const auto stage = "encoder.layers." + std::to_string(s);
            for (int b = 0; b < kDepths[s]; ++b) {
                const auto p = stage + ".blocks." + std::to_string(b);
                const auto shortcut = x;
                auto padded = pad_bhwc(norm(x, p + ".layernorm_before"), 7);
                if (b % 2)
                    padded = roll_bhwc(padded, -3);
                auto windows = window_partition_2d(padded, 7).windows.reshape({-1, 49, c});
                if (b % 2 && shifted.interior_count)
                    windows = windows.index_select(0, shifted.order);
                const int count = int(windows.size(0));
                const auto project = [&](const char* suffix) {
                    return dense(windows, p + ".attention.self." + suffix)
                        .reshape({count, 49, heads, 32})
                        .permute({0, 2, 1, 3})
                        .contiguous();
                };
                const auto q = project("query"), k = project("key"), v = project("value");
                const auto bias = w(p + ".attention.self.relative_position_bias_table")
                                      .index_select(0, relative_indices_)
                                      .reshape({49, 49, heads})
                                      .permute({2, 0, 1})
                                      .contiguous()
                                      .reshape({1, heads, 49, 49});
                Tensor attended;
                if (!(b % 2)) {
                    attended = attention(q, k, v, &bias);
                } else {
                    // Interior windows need only relative bias. Materialize the
                    // per-window separation mask exclusively for border windows.
                    const Tensor mask = shifted.mask + bias;
                    if (!shifted.interior_count)
                        attended = attention(q, k, v, &mask);
                    else {
                        const auto run = [&](int first, int last, const Tensor& m) {
                            return attention(q.slice(0, first, last), k.slice(0, first, last),
                                             v.slice(0, first, last), &m);
                        };
                        attended = Tensor::cat({run(0, shifted.interior_count, bias), run(shifted.interior_count, count, mask)}, 0)
                                       .index_select(0, shifted.restore);
                    }
                }
                auto result = dense(merge_heads(attended), p + ".attention.output.dense")
                                  .reshape({count, 7, 7, c});
                result = window_unpartition_2d(result, 7, 0, 0, hp, wp);
                if (b % 2)
                    result = roll_bhwc(result, 3);
                x = shortcut + result.slice(1, 0, height).slice(2, 0, width);
                x = x + dense(dense(norm(x, p + ".layernorm_after"), p + ".intermediate.dense", Activation::GeluErf), p + ".output.dense");
            }
            if (s > 0)
                output[s - 1] = norm(x, "hidden_states_norms.stage" + std::to_string(s + 1))
                                    .permute({0, 3, 1, 2})
                                    .contiguous();
            if (s < 3) {
                const auto padded = pad_bhwc(x, 2);
                const auto h = padded.size(1), width_padded = padded.size(2);
                const auto sample = [&](size_t y, size_t col) {
                    return padded.slice(1, y, h, 2).slice(2, col, width_padded, 2);
                };
                const auto merged = Tensor::cat({sample(0, 0), sample(1, 0), sample(0, 1), sample(1, 1)}, 3);
                x = linear(norm(merged, stage + ".downsample.norm"), w(stage + ".downsample.reduction.weight"));
            }
        }
        return output;
    }

    size_t SwinTiny::weights_bytes() const {
        size_t bytes = relative_indices_.bytes();
        for (const auto& [name, value] : weights_)
            bytes += value.bytes();
        return bytes;
    }
} // namespace lfs::core::nn::models
