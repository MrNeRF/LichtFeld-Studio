/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/nn/models/llmdet_decoder.hpp"
#include "llmdet_math.hpp"
#include <limits>
#include <map>

namespace lfs::core::nn::models {
    namespace {
        using llmdet_detail::heads;
        using llmdet_detail::sine_positions;
        lfs::Error decoder_error(std::string detail) {
            return lfs::make_error({.code = lfs::ErrorCode::InvalidArgument,
                                    .domain = lfs::ErrorDomain::Core,
                                    .user_message = "Object detection failed",
                                    .detail = std::move(detail),
                                    .detection = LFS_SOURCE_SITE_CURRENT()});
        }
        std::string bbox_prefix(int layer) {
            return layer == 0 ? "bbox_embed.0" : "model.decoder.bbox_embed." + std::to_string(layer);
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
            const auto bbox = [&](const std::string& p) {
                dense(p + ".layers.0", 256, 256);
                dense(p + ".layers.1", 256, 256);
                dense(p + ".layers.2", 256, 4);
            };
            dense("model.enc_output", 256, 256);
            norm("model.enc_output_norm");
            bbox("model.encoder_output_bbox_embed");
            add("model.encoder_output_class_embed.bias", {});
            add("model.decoder.class_embed.5.bias", {});
            add("model.query_position_embeddings.weight", {900, 256});
            norm("model.decoder.layer_norm");
            dense("model.decoder.reference_points_head.layers.0", 512, 256);
            dense("model.decoder.reference_points_head.layers.1", 256, 256);
            for (int i = 0; i < 6; ++i) {
                bbox(bbox_prefix(i));
                const auto p = "model.decoder.layers." + std::to_string(i);
                for (const auto* attention : {"self_attn", "encoder_attn_text"})
                    for (const auto* name : {"query", "key", "value", "out_proj"})
                        dense(p + "." + attention + "." + name, 256, 256);
                for (const auto* name : {"self_attn_layer_norm", "encoder_attn_text_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"})
                    norm(p + "." + name);
                for (const auto* name : {"value_proj", "output_proj", "sampling_offsets"})
                    dense(p + ".encoder_attn." + name, 256, 256);
                dense(p + ".encoder_attn.attention_weights", 256, 128);
                dense(p + ".fc1", 256, 2048);
                dense(p + ".fc2", 2048, 256);
            }
            return result;
        }
        Tensor logit(const Tensor& x) {
            const auto bounded = x.clamp(1e-5f, 1.f - 1e-5f);
            const Tensor complement = bounded * -1.f + 1.f;
            const Tensor result = (bounded / complement).log();
            return result;
        }
    } // namespace

    lfs::Result<LlmDetDecoder> LlmDetDecoder::load(const WeightFile& archive, Device device) {
        if (device != Device::CUDA)
            return decoder_error("LLMDet decoder requires CUDA");
        const auto expected = expected_weights();
        for (const auto& [name, shape] : expected) {
            const auto* info = archive.info(name);
            if (!info || info->shape != shape || info->dtype != DataType::Float32)
                return decoder_error("Invalid LLMDet decoder weight: " + name);
        }
        LlmDetDecoder model;
        model.weights_.reserve(expected.size());
        for (const auto& [name, shape] : expected) {
            auto tensor = archive.load(name, device, DataType::Float32);
            if (!tensor)
                return std::move(tensor.error());
            model.weights_.emplace(name, std::move(*tensor));
        }
        return model;
    }

    lfs::Result<LlmDetPredictions> LlmDetDecoder::forward(const LlmDetEncoded& encoded) const {
        const auto& vision = encoded.vision;
        const auto& text = encoded.text;
        const auto valid_feature = [](const Tensor& x) {
            return x.is_valid() && x.device() == Device::CUDA && x.dtype() == DataType::Float32 &&
                   x.ndim() == 3 && x.size(0) == 1 && x.size(2) == 256;
        };
        if (!valid_feature(vision) || !valid_feature(text) || text.size(1) < 3 || text.size(1) > 256 || vision.size(1) < 900)
            return decoder_error("Invalid LLMDet decoder feature shape or type");
        size_t spatial = 0;
        for (const auto [h, width] : encoded.spatial_shapes) {
            if (h < 1 || width < 1 || h > 167 || width > 167)
                return decoder_error("Invalid LLMDet decoder spatial extent");
            spatial += size_t(h) * width;
        }
        if (spatial != vision.size(1) || weights_.empty())
            return decoder_error("Invalid LLMDet decoder spatial geometry or missing weights");
        const auto w = [&](const std::string& name) -> const Tensor& { return weights_.at(name); };
        const auto dense = [&](const Tensor& x, const std::string& p, Activation act = Activation::None) {
            return linear(x, w(p + ".weight"), &w(p + ".bias"), act);
        };
        const auto norm = [&](const Tensor& x, const std::string& p) {
            return layer_norm(x, w(p + ".weight"), w(p + ".bias"), 1e-5f);
        };
        const auto bbox = [&](const Tensor& x, const std::string& p) {
            return dense(dense(dense(x, p + ".layers.0", Activation::Relu), p + ".layers.1", Activation::Relu), p + ".layers.2");
        };
        std::vector<float> proposals, valid;
        proposals.reserve(spatial * 4);
        valid.reserve(spatial);
        for (size_t level = 0; level < encoded.spatial_shapes.size(); ++level) {
            const auto [h, width] = encoded.spatial_shapes[level];
            const float size = .05f * float(1 << level);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < width; ++x) {
                    const float cx = (x + .5f) / width, cy = (y + .5f) / h;
                    const bool accepted = cx > .01f && cx < .99f && cy > .01f && cy < .99f;
                    valid.push_back(accepted ? 1.f : 0.f);
                    for (const float p : {cx, cy, size, size})
                        proposals.push_back(accepted ? std::log(p / (1.f - p)) : std::numeric_limits<float>::infinity());
                }
        }
        const auto proposal = Tensor::from_vector(proposals, {1, spatial, 4}, Device::CPU).to(Device::CUDA);
        const auto keep = Tensor::from_vector(valid, {1, spatial, 1}, Device::CPU).to(Device::CUDA);
        const Tensor masked = vision * keep;
        const auto object_query = norm(dense(masked, "model.enc_output"), "model.enc_output_norm");
        const auto classify = [&](const Tensor& x, const char* bias) {
            const Tensor scores = bmm(x, text, false, true) * (1.f / 16.f) + w(bias);
            return scores;
        };
        const auto encoder_logits = classify(object_query, "model.encoder_output_class_embed.bias");
        // The canonical 1D CUDA sort supplies the proposal ranking. No CPU top-k.
        const auto ranked = encoder_logits.max(-1).flatten().sort(-1, true).second.slice(0, 0, 900).to(DataType::Int32);
        const Tensor coordinates = bbox(object_query, "model.encoder_output_bbox_embed") + proposal;
        auto reference = sigmoid(coordinates.index_select(1, ranked));
        auto hidden = w("model.query_position_embeddings.weight").reshape({1, 900, 256});
        Tensor last_reference;
        for (int i = 0; i < 6; ++i) {
            last_reference = reference;
            // Official sine order is y,x,w,h, 128 features per coordinate.
            const auto component = [&](int axis) { return sine_positions(reference.slice(2, axis, axis + 1), 128, 10000.f); };
            const auto position = dense(dense(Tensor::cat({component(1), component(0), component(2), component(3)}, -1),
                                              "model.decoder.reference_points_head.layers.0", Activation::Relu),
                                        "model.decoder.reference_points_head.layers.1");
            const auto p = "model.decoder.layers." + std::to_string(i);
            const auto mha = [&](const Tensor& queries, const Tensor& keys, const Tensor& values, const std::string& prefix) {
                const auto q = heads(dense(queries, prefix + ".query"), 8);
                const auto k = heads(dense(keys, prefix + ".key"), 8);
                const auto v = heads(dense(values, prefix + ".value"), 8);
                return dense(merge_heads(attention(q, k, v)), prefix + ".out_proj");
            };
            const Tensor self_queries = hidden + position;
            hidden = norm(hidden + mha(self_queries, self_queries, hidden, p + ".self_attn"), p + ".self_attn_layer_norm");
            hidden = norm(hidden + mha(hidden + position, text, text, p + ".encoder_attn_text"), p + ".encoder_attn_text_layer_norm");
            const Tensor queries = hidden + position;
            const auto offsets = dense(queries, p + ".encoder_attn.sampling_offsets").reshape({1, 900, 8, 4, 4, 2});
            const auto centers = reference.slice(2, 0, 2).reshape({1, 900, 1, 1, 1, 2});
            const auto sizes = reference.slice(2, 2, 4).reshape({1, 900, 1, 1, 1, 2});
            const Tensor locations = centers + offsets * sizes * .125f;
            const auto weights = softmax(dense(queries, p + ".encoder_attn.attention_weights").reshape({1, 900, 8, 16}))
                                     .reshape({1, 900, 8, 4, 4});
            const auto values = dense(vision, p + ".encoder_attn.value_proj").reshape({1, int(spatial), 8, 32});
            const auto attended = deformable_attention(values, locations, weights, encoded.spatial_shapes);
            hidden = norm(hidden + dense(attended, p + ".encoder_attn.output_proj"), p + ".encoder_attn_layer_norm");
            hidden = norm(hidden + dense(dense(hidden, p + ".fc1", Activation::Relu), p + ".fc2"), p + ".final_layer_norm");
            reference = sigmoid(bbox(hidden, bbox_prefix(i)) + logit(reference));
        }
        const auto normalized = norm(hidden, "model.decoder.layer_norm");
        return LlmDetPredictions{sigmoid(bbox(normalized, bbox_prefix(5)) + logit(last_reference)),
                                 classify(normalized, "model.decoder.class_embed.5.bias")};
    }

    size_t LlmDetDecoder::weights_bytes() const {
        size_t bytes = 0;
        for (const auto& [name, value] : weights_)
            bytes += value.bytes();
        return bytes;
    }
} // namespace lfs::core::nn::models
