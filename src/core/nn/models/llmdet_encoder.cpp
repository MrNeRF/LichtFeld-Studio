/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/nn/models/llmdet_encoder.hpp"
#include "core/nn/models/bert_text.hpp"
#include "core/nn/ops.hpp"
#include "llmdet_math.hpp"
#include <algorithm>
#include <map>

namespace lfs::core::nn::models {
    namespace {
        lfs::Error encoder_error(std::string detail) {
            return lfs::make_error({.code = lfs::ErrorCode::InvalidArgument,
                                    .domain = lfs::ErrorDomain::Core,
                                    .user_message = "Image encoding failed",
                                    .detail = std::move(detail),
                                    .detection = LFS_SOURCE_SITE_CURRENT()});
        }

        std::map<std::string, TensorShape> expected_weights() {
            std::map<std::string, TensorShape> result;
            const auto add = [&](const std::string& p, std::initializer_list<size_t> shape) {
                result.emplace(p, TensorShape(std::vector<size_t>(shape)));
            };
            const auto norm = [&](const std::string& p) {
                add(p + ".weight", {256});
                add(p + ".bias", {256});
            };
            const auto dense = [&](const std::string& p, size_t in, size_t out) {
                add(p + ".weight", {out, in});
                add(p + ".bias", {out});
            };
            add("level_embed", {4, 256});
            dense("text_projection", 768, 256);
            for (int s = 0; s < 4; ++s) {
                const auto p = "input_proj_vision." + std::to_string(s);
                const size_t channels[]{192, 384, 768, 768};
                const size_t k = s == 3 ? 3 : 1;
                add(p + ".0.weight", {256, channels[s], k, k});
                add(p + ".0.bias", {256});
                norm(p + ".1");
            }
            for (int i = 0; i < 6; ++i) {
                const auto p = "encoder.layers." + std::to_string(i);
                const auto f = p + ".fusion_layer";
                norm(f + ".layer_norm_vision");
                norm(f + ".layer_norm_text");
                add(f + ".vision_param", {256});
                add(f + ".text_param", {256});
                for (const auto* name : {"vision_proj", "text_proj", "values_vision_proj", "values_text_proj"})
                    dense(f + ".attn." + name, 256, 1024);
                dense(f + ".attn.out_vision_proj", 1024, 256);
                dense(f + ".attn.out_text_proj", 1024, 256);
                const auto t = p + ".text_enhancer_layer";
                for (const auto* name : {"query", "key", "value", "out_proj"})
                    dense(t + ".self_attn." + name, 256, 256);
                norm(t + ".layer_norm_before");
                norm(t + ".layer_norm_after");
                dense(t + ".fc1", 256, 1024);
                dense(t + ".fc2", 1024, 256);
                const auto d = p + ".deformable_layer";
                for (const auto* name : {"value_proj", "output_proj", "sampling_offsets"})
                    dense(d + ".self_attn." + name, 256, 256);
                dense(d + ".self_attn.attention_weights", 256, 128);
                norm(d + ".self_attn_layer_norm");
                norm(d + ".final_layer_norm");
                dense(d + ".fc1", 256, 2048);
                dense(d + ".fc2", 2048, 256);
            }
            return result;
        }

        using llmdet_detail::heads;
        using llmdet_detail::sine_positions;

        Tensor bmm_heads(const Tensor& x, int count) {
            return heads(x, count).reshape({count, int(x.size(1)), int(x.size(2)) / count});
        }

        Tensor merge_bmm_heads(const Tensor& x) {
            return merge_heads(x.reshape({1, int(x.size(0)), int(x.size(1)), int(x.size(2))}));
        }
    } // namespace

    lfs::Result<LlmDetEncoder> LlmDetEncoder::load(const WeightFile& archive, Device device) {
        if (device != Device::CUDA)
            return encoder_error("LLMDet encoder requires CUDA");
        const auto expected = expected_weights();
        for (const auto& [name, shape] : expected) {
            const auto* info = archive.info("model." + name);
            if (!info || info->shape != shape || info->dtype != DataType::Float32)
                return encoder_error("Invalid LLMDet encoder weight: " + name);
        }
        LlmDetEncoder model;
        model.weights_.reserve(expected.size());
        for (const auto& [name, shape] : expected) {
            auto value = archive.load("model." + name, device, DataType::Float32);
            if (!value)
                return std::move(value.error());
            model.weights_.emplace(name, std::move(*value));
        }
        return model;
    }

    lfs::Result<LlmDetEncoded> LlmDetEncoder::encode(const std::array<Tensor, 3>& swin_features,
                                                     const Tensor& bert_features, std::span<const int> positions,
                                                     std::span<const float> attention_mask) const {
        if (!bert_features.is_valid() || bert_features.device() != Device::CUDA || bert_features.dtype() != DataType::Float32 ||
            bert_features.ndim() != 3 || bert_features.size(0) != 1 || bert_features.size(2) != 768 ||
            bert_features.size(1) != positions.size() ||
            !BertText::valid_attention(positions, attention_mask))
            return encoder_error("Invalid LLMDet text features or attention shape");
        for (size_t i = 0; i < swin_features.size(); ++i) {
            const auto& f = swin_features[i];
            if (!f.is_valid() || f.device() != Device::CUDA || f.dtype() != DataType::Float32 || f.ndim() != 4 ||
                f.size(0) != 1 || f.size(1) != (size_t(192) << i) || f.size(2) < 1 || f.size(3) < 1 ||
                f.size(2) > 167 || f.size(3) > 167)
                return encoder_error("Invalid LLMDet Swin feature shape or type");
        }
        if (weights_.empty())
            return encoder_error("LLMDet encoder weights are not loaded");
        const auto w = [&](const std::string& name) -> const Tensor& { return weights_.at(name); };
        const auto norm = [&](const Tensor& x, const std::string& p) {
            return layer_norm(x, w(p + ".weight"), w(p + ".bias"), 1e-5f);
        };
        const auto dense = [&](const Tensor& x, const std::string& p, Activation act = Activation::None) {
            return linear(x, w(p + ".weight"), &w(p + ".bias"), act);
        };
        std::vector<Tensor> flattened, position_levels, reference_levels;
        LlmDetEncoded result;
        std::vector<float> normalizers;
        for (int s = 0; s < 4; ++s) {
            const auto p = "input_proj_vision." + std::to_string(s);
            Conv2dParams params;
            if (s == 3) {
                params.stride_h = params.stride_w = 2;
                params.pad_h = params.pad_w = 1;
            }
            auto f = conv2d(swin_features[std::min(s, 2)], w(p + ".0.weight"), &w(p + ".0.bias"), params);
            f = group_norm(f, w(p + ".1.weight"), w(p + ".1.bias"), 32);
            const int h = int(f.size(2)), width = int(f.size(3));
            result.spatial_shapes[s] = {h, width};
            normalizers.insert(normalizers.end(), {float(width), float(h)});
            flattened.push_back(f.permute({0, 2, 3, 1}).contiguous().reshape({1, h * width, 256}));
            // Upload positional coordinates through the stream-aware Tensor transfer.
            const auto ys = Tensor::arange(0.f, float(h), 1.f, Device::CPU).to(Device::CUDA).reshape({h, 1, 1});
            const auto xs = Tensor::arange(0.f, float(width), 1.f, Device::CPU).to(Device::CUDA).reshape({1, width, 1});
            const Tensor y_positions = (ys + 1.f) / (h + 1e-6f), x_positions = (xs + 1.f) / (width + 1e-6f);
            const auto py = sine_positions(y_positions, 128, 20.f).expand({h, width, 128});
            const auto px = sine_positions(x_positions, 128, 20.f).expand({h, width, 128});
            const Tensor pe = Tensor::cat({py, px}, -1).reshape({1, h * width, 256}) + w("level_embed").slice(0, s, s + 1);
            position_levels.push_back(pe);
            const Tensor ref_x = (xs + .5f) / float(width), ref_y = (ys + .5f) / float(h);
            reference_levels.push_back(Tensor::cat({ref_x.expand({h, width, 1}), ref_y.expand({h, width, 1})}, -1).reshape({1, h * width, 1, 1, 1, 2}));
        }
        auto vision = Tensor::cat(flattened, 1);
        auto text = dense(bert_features, "text_projection");
        const int spatial = int(vision.size(1)), length = int(text.size(1));
        const auto vision_positions = Tensor::cat(position_levels, 1);
        const auto reference = Tensor::cat(reference_levels, 1);
        const auto offset_normalizer = Tensor::from_vector(normalizers, {1, 1, 1, 4, 1, 2}, Device::CPU).to(Device::CUDA);
        const auto text_mask = Tensor::from_vector(std::vector<float>(attention_mask.begin(), attention_mask.end()),
                                                   {1, 1, positions.size(), positions.size()}, Device::CPU)
                                   .to(Device::CUDA);
        const auto ids = Tensor::from_vector(std::vector<int>(positions.begin(), positions.end()), {positions.size(), 1}, Device::CPU)
                             .to(Device::CUDA)
                             .to(DataType::Float32);
        const auto text_positions = sine_positions(ids, 256, 10000.f).reshape({1, length, 256});
        for (int layer = 0; layer < 6; ++layer) {
            const auto p = "encoder.layers." + std::to_string(layer);
            const auto f = p + ".fusion_layer";
            vision = norm(vision, f + ".layer_norm_vision");
            text = norm(text, f + ".layer_norm_text");
            const auto qv = bmm_heads(dense(vision, f + ".attn.vision_proj"), 4).mul(1.f / 16.f);
            const auto kt = bmm_heads(dense(text, f + ".attn.text_proj"), 4);
            const auto vv = bmm_heads(dense(vision, f + ".attn.values_vision_proj"), 4);
            const auto vt = bmm_heads(dense(text, f + ".attn.values_text_proj"), 4);
            const auto scores = bmm(qv, kt, false, true);
            const Tensor stabilized = (scores - scores.max()).clamp(-50000.f, 50000.f);
            const auto transposed = stabilized.permute({0, 2, 1}).contiguous();
            const Tensor text_scores = (transposed - transposed.max(-1, true)).clamp(-50000.f, 50000.f);
            const auto dv = dense(merge_bmm_heads(bmm(softmax(stabilized), vt)), f + ".attn.out_vision_proj");
            const auto dt = dense(merge_bmm_heads(bmm(softmax(text_scores), vv)), f + ".attn.out_text_proj");
            vision = vision + dv * w(f + ".vision_param");
            text = text + dt * w(f + ".text_param");
            const auto t = p + ".text_enhancer_layer";
            const Tensor text_queries = text + text_positions;
            const auto q = heads(dense(text_queries, t + ".self_attn.query"), 4);
            const auto k = heads(dense(text_queries, t + ".self_attn.key"), 4);
            const auto v = heads(dense(text, t + ".self_attn.value"), 4);
            text = norm(text + dense(merge_heads(attention(q, k, v, &text_mask)), t + ".self_attn.out_proj"), t + ".layer_norm_before");
            text = norm(text + dense(dense(text, t + ".fc1", Activation::Relu), t + ".fc2"), t + ".layer_norm_after");
            const auto d = p + ".deformable_layer";
            const Tensor queries = vision + vision_positions;
            const auto offsets = dense(queries, d + ".self_attn.sampling_offsets").reshape({1, spatial, 8, 4, 4, 2});
            const Tensor locations = reference + offsets / offset_normalizer;
            const auto weights = softmax(dense(queries, d + ".self_attn.attention_weights").reshape({1, spatial, 8, 16}))
                                     .reshape({1, spatial, 8, 4, 4});
            const auto values = dense(vision, d + ".self_attn.value_proj").reshape({1, spatial, 8, 32});
            const auto attended = deformable_attention(values, locations, weights, result.spatial_shapes);
            vision = norm(vision + dense(attended, d + ".self_attn.output_proj"), d + ".self_attn_layer_norm");
            vision = norm(vision + dense(dense(vision, d + ".fc1", Activation::Relu), d + ".fc2"), d + ".final_layer_norm");
        }
        result.vision = std::move(vision);
        result.text = std::move(text);
        return result;
    }

    size_t LlmDetEncoder::weights_bytes() const {
        size_t bytes = 0;
        for (const auto& [name, value] : weights_)
            bytes += value.bytes();
        return bytes;
    }
} // namespace lfs::core::nn::models
