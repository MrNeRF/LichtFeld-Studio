/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/nn/ops.hpp"
#include <cmath>
#include <numbers>

namespace lfs::core::nn::models::llmdet_detail {
    inline Tensor sine_positions(const Tensor& positions, int features, float temperature) {
        std::vector<float> divisors(features / 2);
        for (int i = 0; i < features / 2; ++i)
            divisors[i] = std::pow(temperature, float(2 * i) / features);
        const auto divisor = Tensor::from_vector(divisors, {divisors.size()}, Device::CPU).to(Device::CUDA);
        const Tensor angles = positions * (2.f * std::numbers::pi_v<float>) / divisor;
        return Tensor::stack({angles.sin(), angles.cos()}, -1).flatten(-2, -1);
    }

    inline Tensor heads(const Tensor& x, int count) {
        return x.reshape({1, int(x.size(1)), count, int(x.size(2)) / count})
            .permute({0, 2, 1, 3})
            .contiguous();
    }
} // namespace lfs::core::nn::models::llmdet_detail
