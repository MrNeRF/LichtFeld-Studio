/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/nn/weight_file.hpp"
#include <array>
#include <span>
#include <unordered_map>

namespace lfs::core::nn::models {
    struct LlmDetEncoded {
        Tensor vision, text;
        std::array<std::pair<int, int>, 4> spatial_shapes;
    };

    // Multi-scale projection and six multimodal encoder layers of LLMDet Tiny.
    // Caller owns the inference stream and drains archive uploads before closing.
    class LFS_CORE_API LlmDetEncoder {
    public:
        static lfs::Result<LlmDetEncoder> load(const WeightFile& archive, Device device);
        lfs::Result<LlmDetEncoded> encode(const std::array<Tensor, 3>& swin_features,
                                          const Tensor& bert_features, std::span<const int> positions,
                                          std::span<const float> attention_mask) const;
        [[nodiscard]] size_t weights_bytes() const;

    private:
        std::unordered_map<std::string, Tensor> weights_;
    };
} // namespace lfs::core::nn::models
