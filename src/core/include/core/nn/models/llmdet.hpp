/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/nn/models/bert_text.hpp"
#include "core/nn/models/llmdet_decoder.hpp"
#include "core/nn/models/swin_tiny.hpp"
#include <array>
#include <optional>

namespace lfs::core::nn::models {
    struct LlmDetText {
        Tensor features;
        std::vector<int> positions;
        std::vector<float> attention_mask;
        std::vector<std::vector<int>> category_tokens;
    };

    // Native LLMDet Tiny. Uses the caller's serialized inference stream; the
    // caller retains the archive until uploads drain. Owns no file/image cache.
    class LFS_CORE_API LlmDet {
    public:
        static lfs::Result<LlmDet> load(const WeightFile& archive, Device device);
        static bool valid_phrases(std::span<const std::vector<int>> descriptions);
        // HF 4.55 resize geometry; rejects inputs with fewer than 900 spatial proposals.
        static std::optional<std::array<int, 2>> input_size(size_t height, size_t width);
        lfs::Result<LlmDetText> encode(std::span<const std::vector<int>> descriptions) const;
        // Image is original RGB [0,1], NCHW. Owns official 800/1333 bilinear
        // preprocessing and ImageNet normalization; boxes remain normalized.
        lfs::Result<LlmDetPredictions> forward(const Tensor& image, const LlmDetText& text) const;
        [[nodiscard]] size_t weights_bytes() const;

    private:
        BertText bert_;
        SwinTiny swin_;
        LlmDetEncoder encoder_;
        LlmDetDecoder decoder_;
        Tensor mean_, std_;
    };
} // namespace lfs::core::nn::models
