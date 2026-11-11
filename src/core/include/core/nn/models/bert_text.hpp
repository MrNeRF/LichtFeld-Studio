/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/nn/weight_file.hpp"
#include <span>
#include <string_view>
#include <unordered_map>

namespace lfs::core::nn::models {
    // The uncased BERT-base tower used by LLMDet. Like the other native models,
    // the caller serializes load/forward/destruction on its inference stream
    // and retains the archive until asynchronous weight uploads have drained.
    class LFS_CORE_API BertText {
    public:
        static lfs::Result<BertText> load(const WeightFile& archive,
                                          std::string_view prefix, Device device);
        // Shared sub-sentence attention contract for BERT and the multimodal
        // encoder. A token must attend to itself; masks use zero/-infinity.
        static bool valid_attention(std::span<const int> positions, std::span<const float> attention_mask);
        // Unpadded input including CLS/SEP, <=256 tokens. Position IDs and the
        // row-major additive attention mask are supplied by the detector's
        // sub-sentence policy; this tower neither tokenizes nor truncates text.
        lfs::Result<Tensor> encode(std::span<const int> tokens, std::span<const int> positions,
                                   std::span<const float> attention_mask) const;
        [[nodiscard]] std::size_t weights_bytes() const;

    private:
        std::unordered_map<std::string, Tensor> weights_;
    };
} // namespace lfs::core::nn::models
