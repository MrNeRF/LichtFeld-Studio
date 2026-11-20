/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/nn/models/llmdet_encoder.hpp"

namespace lfs::core::nn::models {
    struct LlmDetPredictions {
        Tensor boxes;  // [1,900,4] normalized cx,cy,w,h
        Tensor logits; // [1,900,text_length], before sigmoid
    };

    class LFS_CORE_API LlmDetDecoder {
    public:
        static lfs::Result<LlmDetDecoder> load(const WeightFile& archive, Device device);
        lfs::Result<LlmDetPredictions> forward(const LlmDetEncoded& encoded) const;
        [[nodiscard]] size_t weights_bytes() const;

    private:
        std::unordered_map<std::string, Tensor> weights_;
    };
} // namespace lfs::core::nn::models
