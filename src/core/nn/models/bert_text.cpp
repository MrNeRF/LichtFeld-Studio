/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/nn/models/bert_text.hpp"
#include "core/nn/ops.hpp"
#include <algorithm>
#include <cmath>
#include <map>

namespace lfs::core::nn::models {
    namespace {
        lfs::Error bert_error(std::string detail) {
            return lfs::make_error({.code = lfs::ErrorCode::InvalidArgument,
                                    .domain = lfs::ErrorDomain::Core,
                                    .user_message = "Text encoding failed",
                                    .detail = std::move(detail),
                                    .detection = LFS_SOURCE_SITE_CURRENT()});
        }

        std::map<std::string, TensorShape> expected_weights() {
            std::map<std::string, TensorShape> result;
            const auto tensor = [&](std::string name, std::initializer_list<size_t> shape) {
                result.emplace(std::move(name), TensorShape(std::vector<size_t>(shape)));
            };
            const auto norm = [&](const std::string& p) {
                tensor(p + ".weight", {768});
                tensor(p + ".bias", {768});
            };
            const auto linear = [&](const std::string& p, size_t in, size_t out) {
                tensor(p + ".weight", {out, in});
                tensor(p + ".bias", {out});
            };
            tensor("embeddings.word_embeddings.weight", {30522, 768});
            tensor("embeddings.position_embeddings.weight", {512, 768});
            tensor("embeddings.token_type_embeddings.weight", {2, 768});
            norm("embeddings.LayerNorm");
            for (int i = 0; i < 12; ++i) {
                const auto p = "encoder.layer." + std::to_string(i);
                for (const auto* name : {"query", "key", "value"})
                    linear(p + ".attention.self." + name, 768, 768);
                linear(p + ".attention.output.dense", 768, 768);
                norm(p + ".attention.output.LayerNorm");
                linear(p + ".intermediate.dense", 768, 3072);
                linear(p + ".output.dense", 3072, 768);
                norm(p + ".output.LayerNorm");
            }
            return result;
        }
    } // namespace

    lfs::Result<BertText> BertText::load(const WeightFile& archive,
                                         const std::string_view prefix, const Device device) {
        if (device != Device::CUDA)
            return bert_error("BERT text inference requires CUDA");
        const auto expected = expected_weights();
        for (const auto& [name, shape] : expected) {
            const auto* info = archive.info(std::string(prefix) + name);
            if (!info || info->shape != shape ||
                (info->dtype != DataType::Float32 && info->dtype != DataType::Float16))
                return bert_error("Invalid BERT-base text weight: " + name);
        }
        BertText model;
        model.weights_.reserve(expected.size());
        for (const auto& [name, shape] : expected) {
            auto value = archive.load(std::string(prefix) + name, device, DataType::Float32);
            if (!value)
                return std::move(value.error());
            model.weights_.emplace(name, std::move(*value));
        }
        return model;
    }

    bool BertText::valid_attention(const std::span<const int> positions, const std::span<const float> attention_mask) {
        const auto length = positions.size();
        if (length < 3 || length > 256 || attention_mask.size() != length * length ||
            !std::ranges::all_of(positions, [](int id) { return id >= 0 && id < 512; }) ||
            !std::ranges::all_of(attention_mask, [](float v) { return v == 0 || v == -INFINITY; }))
            return false;
        for (size_t row = 0; row < length; ++row)
            if (attention_mask[row * length + row] != 0)
                return false;
        return true;
    }

    lfs::Result<Tensor> BertText::encode(const std::span<const int> tokens, const std::span<const int> positions,
                                         const std::span<const float> attention_mask) const {
        const auto length = tokens.size();
        if (length < 3 || length > 256 || tokens.front() != 101 || tokens.back() != 102 ||
            positions.size() != length || !valid_attention(positions, attention_mask) ||
            !std::ranges::all_of(tokens, [](int id) { return id >= 0 && id < 30522; }))
            return bert_error("Invalid or overlong BERT token, position or attention input");
        if (weights_.empty())
            return bert_error("BERT text weights are not loaded");
        const auto w = [this](const std::string& name) -> const Tensor& { return weights_.at(name); };
        const auto norm = [&](const Tensor& x, const std::string& p) {
            return layer_norm(x, w(p + ".weight"), w(p + ".bias"), 1e-12f);
        };
        const auto linear_at = [&](const Tensor& x, const std::string& p, Activation activation = Activation::None) {
            return linear(x, w(p + ".weight"), &w(p + ".bias"), activation);
        };
        const auto upload_ids = [](std::span<const int> ids) {
            return Tensor::from_vector(std::vector<int>(ids.begin(), ids.end()), {ids.size()}, Device::CPU).to(Device::CUDA);
        };
        Tensor x = w("embeddings.word_embeddings.weight").index_select(0, upload_ids(tokens));
        x = x + w("embeddings.position_embeddings.weight").index_select(0, upload_ids(positions));
        x = x + w("embeddings.token_type_embeddings.weight").slice(0, 0, 1);
        x = norm(x, "embeddings.LayerNorm").reshape({1, static_cast<int>(length), 768});
        const auto mask = Tensor::from_vector(std::vector<float>(attention_mask.begin(), attention_mask.end()),
                                              {1, 1, length, length}, Device::CPU)
                              .to(Device::CUDA);
        for (int i = 0; i < 12; ++i) {
            const auto p = "encoder.layer." + std::to_string(i);
            const auto project = [&](const char* name) {
                return linear_at(x, p + ".attention.self." + name)
                    .reshape({1, static_cast<int>(length), 12, 64})
                    .permute({0, 2, 1, 3})
                    .contiguous();
            };
            const auto q = project("query"), k = project("key"), v = project("value");
            x = norm(x + linear_at(merge_heads(attention(q, k, v, &mask)), p + ".attention.output.dense"),
                     p + ".attention.output.LayerNorm");
            x = norm(x + linear_at(linear_at(x, p + ".intermediate.dense", Activation::GeluErf), p + ".output.dense"),
                     p + ".output.LayerNorm");
        }
        return x;
    }

    size_t BertText::weights_bytes() const {
        size_t bytes = 0;
        for (const auto& [name, value] : weights_)
            bytes += value.bytes();
        return bytes;
    }
} // namespace lfs::core::nn::models
