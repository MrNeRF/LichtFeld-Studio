/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/nn/models/llmdet.hpp"
#include "core/nn/ops.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace lfs::core::nn::models {
    namespace {
        lfs::Error text_error() {
            return lfs::make_error({.code = lfs::ErrorCode::InvalidArgument,
                                    .domain = lfs::ErrorDomain::Core,
                                    .user_message = "Text encoding failed",
                                    .detail = "LLMDet requires 1..16 nonempty BERT descriptions within one 256-token context",
                                    .detection = LFS_SOURCE_SITE_CURRENT()});
        }
        bool delimiter(int id) { return id == 1012 || id == 1029; }
    } // namespace

    lfs::Result<LlmDet> LlmDet::load(const WeightFile& archive, Device device) {
        auto bert = BertText::load(archive, "model.text_backbone.", device);
        if (!bert)
            return std::move(bert.error());
        auto swin = SwinTiny::load(archive, device);
        if (!swin)
            return std::move(swin.error());
        auto encoder = LlmDetEncoder::load(archive, device);
        if (!encoder)
            return std::move(encoder.error());
        auto decoder = LlmDetDecoder::load(archive, device);
        if (!decoder)
            return std::move(decoder.error());
        LlmDet result;
        result.bert_ = std::move(*bert);
        result.swin_ = std::move(*swin);
        result.encoder_ = std::move(*encoder);
        result.decoder_ = std::move(*decoder);
        result.mean_ = Tensor::from_vector({.485f, .456f, .406f}, {1, 3, 1, 1}, Device::CPU).to(device);
        result.std_ = Tensor::from_vector({.229f, .224f, .225f}, {1, 3, 1, 1}, Device::CPU).to(device);
        return result;
    }

    bool LlmDet::valid_phrases(std::span<const std::vector<int>> descriptions) {
        if (descriptions.empty() || descriptions.size() > 16)
            return false;
        size_t count = 2;
        for (const auto& ids : descriptions) {
            if (ids.size() < 4 || ids.size() > 256 || ids.front() != 101 || ids.back() != 102 || ids[ids.size() - 2] != 1012)
                return false;
            const auto content = std::span(ids).subspan(1, ids.size() - 2);
            if (!std::ranges::all_of(content, [](int id) { return id > 0 && id < 30522 && id != 101 && id != 102; }) ||
                !std::ranges::any_of(content, [](int id) { return !delimiter(id); }))
                return false;
            count += content.size();
            if (count > 256)
                return false;
        }
        return true;
    }

    lfs::Result<LlmDetText> LlmDet::encode(std::span<const std::vector<int>> descriptions) const {
        if (!valid_phrases(descriptions))
            return text_error();
        LlmDetText result;
        std::vector<int> tokens{101};
        for (const auto& phrase : descriptions) {
            std::vector<int> indices;
            for (size_t i = 1; i + 1 < phrase.size(); ++i) {
                if (!delimiter(phrase[i]))
                    indices.push_back(int(tokens.size()));
                tokens.push_back(phrase[i]);
            }
            result.category_tokens.push_back(std::move(indices));
        }
        tokens.push_back(102);
        result.positions.resize(tokens.size());
        result.attention_mask.assign(tokens.size() * tokens.size(), -std::numeric_limits<float>::infinity());
        size_t previous = 0;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i == 0 || i + 1 == tokens.size()) {
                result.attention_mask[i * tokens.size() + i] = 0;
            } else if (delimiter(tokens[i])) {
                for (size_t row = previous + 1; row <= i; ++row) {
                    result.positions[row] = int(row - previous - 1);
                    for (size_t col = previous + 1; col <= i; ++col)
                        result.attention_mask[row * tokens.size() + col] = 0;
                }
                previous = i;
            }
        }
        auto encoded = bert_.encode(tokens, result.positions, result.attention_mask);
        if (!encoded)
            return std::move(encoded.error());
        result.features = std::move(*encoded);
        return result;
    }

    std::optional<std::array<int, 2>> LlmDet::input_size(size_t h, size_t width) {
        if (!h || !width)
            return std::nullopt;
        const double shorter = double(std::min(h, width)), longer = double(std::max(h, width));
        const bool capped = longer / shorter * 800. > 1333.;
        const double raw_size = capped ? 1333. * shorter / longer : 800.;
        const int short_side = std::max(1, int(std::nearbyint(raw_size)));
        const int long_side = std::max(1, int(raw_size * longer / shorter));
        const int out_h = h <= width ? short_side : long_side, out_w = h <= width ? long_side : short_side;
        size_t proposals = 0;
        for (const int stride : {8, 16, 32, 64})
            proposals += size_t((out_h + stride - 1) / stride) * size_t((out_w + stride - 1) / stride);
        if (proposals < 900)
            return std::nullopt;
        return std::array{out_h, out_w};
    }

    lfs::Result<LlmDetPredictions> LlmDet::forward(const Tensor& image, const LlmDetText& text) const {
        if (!mean_.is_valid() || !image.is_valid() || image.device() != Device::CUDA || image.dtype() != DataType::Float32 ||
            image.ndim() != 4 || image.size(0) != 1 || image.size(1) != 3 || image.size(2) < 1 || image.size(3) < 1)
            return lfs::make_error({.code = lfs::ErrorCode::InvalidArgument, .domain = lfs::ErrorDomain::Core, .user_message = "Object detection failed", .detail = "Invalid LLMDet RGB input or missing model", .detection = LFS_SOURCE_SITE_CURRENT()});
        const auto size = input_size(image.size(2), image.size(3));
        if (!size)
            return lfs::make_error({.code = lfs::ErrorCode::InvalidArgument, .domain = lfs::ErrorDomain::Core, .user_message = "Object detection failed", .detail = "LLMDet image proportions provide fewer than 900 spatial proposals", .detection = LFS_SOURCE_SITE_CURRENT()});
        const Tensor normalized = (resize2d_pil(image, (*size)[0], (*size)[1], ResizeMode::Bilinear) - mean_) / std_;
        auto features = swin_.encode(normalized);
        if (!features)
            return std::move(features.error());
        auto encoded = encoder_.encode(*features, text.features, text.positions, text.attention_mask);
        if (!encoded)
            return std::move(encoded.error());
        return decoder_.forward(*encoded);
    }

    size_t LlmDet::weights_bytes() const {
        return bert_.weights_bytes() + swin_.weights_bytes() + encoder_.weights_bytes() + decoder_.weights_bytes() + mean_.bytes() + std_.bytes();
    }
} // namespace lfs::core::nn::models
