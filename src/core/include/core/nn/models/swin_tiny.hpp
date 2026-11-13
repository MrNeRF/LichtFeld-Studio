/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/nn/weight_file.hpp"
#include <array>
#include <unordered_map>

namespace lfs::core::nn::models {
    // LLMDet's Swin-T backbone. The inference owner serializes operations on its
    // stream and retains the archive until asynchronous uploads have drained.
    class LFS_CORE_API SwinTiny {
    public:
        static lfs::Result<SwinTiny> load(const WeightFile& archive, Device device);
        // One normalized RGB NCHW image, with each dimension in [1,1333].
        // Returns normalized NCHW stage2/3/4 maps (192/384/768 channels).
        lfs::Result<std::array<Tensor, 3>> encode(const Tensor& image) const;
        [[nodiscard]] size_t weights_bytes() const;

    private:
        std::unordered_map<std::string, Tensor> weights_;
        Tensor relative_indices_;
    };
} // namespace lfs::core::nn::models
