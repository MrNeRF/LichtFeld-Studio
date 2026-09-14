/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "training/losses/effective_rank_regularization.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;
using namespace lfs::training::losses;
namespace {
    double reference(const std::vector<float>& x, double rw, double tw, double eps) {
        double result = 0;
        for (size_t i = 0; i < x.size(); i += 3) {
            std::array<double, 3> s{std::exp(static_cast<double>(x[i])), std::exp(static_cast<double>(x[i + 1])), std::exp(static_cast<double>(x[i + 2]))};
            const double sum = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
            double entropy = 0;
            for (double scale : s) {
                const double p = scale * scale / std::max(sum, eps);
                entropy -= p * std::log(std::max(p, eps));
            }
            result += rw * std::max(-std::log(std::exp(entropy) - 1.0 + eps), 0.0) + tw * *std::min_element(s.begin(), s.end());
        }
        return result / (x.size() / 3);
    }
} // namespace
TEST(POPSpaRegularization, ManualGradientsMatchFiniteDifferencesAndAccumulate) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
        GTEST_SKIP();
    const std::vector<float> raw{1.3f, -1.7f, -2.6f, 0.4f, 0.2f, -0.1f};
    const EffectiveRankRegularization::Params params{.erank_weight = 0.03f, .thin_scale_weight = 0.7f, .epsilon = 1e-7f};
    auto x = Tensor::from_vector(raw, {2, 3}, Device::CUDA);
    auto grad = Tensor::ones({2, 3}, Device::CUDA);
    auto loss = EffectiveRankRegularization::forward(x, grad, params);
    ASSERT_TRUE(loss);
    EXPECT_NEAR(loss.value().item<float>(), reference(raw, params.erank_weight, params.thin_scale_weight, params.epsilon), 1e-6);
    const auto actual = grad.to_vector();
    for (size_t i = 0; i < raw.size(); ++i) {
        auto lo = raw, hi = raw;
        lo[i] -= 0.001f;
        hi[i] += 0.001f;
        const double finite_difference = (reference(hi, params.erank_weight, params.thin_scale_weight, params.epsilon) -
                                          reference(lo, params.erank_weight, params.thin_scale_weight, params.epsilon)) /
                                         (hi[i] - lo[i]);
        EXPECT_NEAR(actual[i] - 1.0, finite_difference, 2e-5) << "coordinate=" << i;
    }
}
TEST(POPSpaRegularization, FrozenGradientAndInactiveLossAreExcluded) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
        GTEST_SKIP();
    auto x = Tensor::from_vector(std::vector<float>{0, -1, -2, 0, -1, -2, 0, -1, -2}, {3, 3}, Device::CUDA);
    auto grad = Tensor::zeros({3, 3}, Device::CUDA);
    auto active = Tensor::from_vector(std::vector<bool>{true, true, false}, {3}, Device::CUDA);
    auto frozen = Tensor::from_vector(std::vector<bool>{false, true, false}, {3}, Device::CUDA);
    auto loss = EffectiveRankRegularization::forward(x, grad, {}, active, frozen, 2);
    ASSERT_TRUE(loss);
    EXPECT_NEAR(loss.value().item<float>(), reference({0, -1, -2}, 0.01, 1, 1e-7), 1e-6);
    const auto g = grad.to_vector();
    EXPECT_NE(g[2], 0);
    for (size_t i = 3; i < g.size(); ++i)
        EXPECT_EQ(g[i], 0);
}
