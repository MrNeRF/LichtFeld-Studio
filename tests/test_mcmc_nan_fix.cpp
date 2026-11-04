/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Tests for NaN fixes in MCMC relocation

#include "core/cuda/memory_arena.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "training/kernels/mcmc_kernels.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace lfs::core;

class MCMCNaNFixTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceSynchronize();
        lfs::training::mcmc::init_relocation_coefficients(51);
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

namespace {
    // Rows [0, n) are sources, rows [n, 2n) copies of them, as relocation
    // leaves them before launch_relocate_copies.
    struct RelocatedRows {
        std::vector<float> opacity_raw; // [2n]
        std::vector<float> scaling_raw; // [2n, 3]
    };

    RelocatedRows relocate(const std::vector<float>& opacity_raw,
                           const std::vector<float>& scaling_raw,
                           const std::vector<int>& ratios,
                           const float min_opacity = 0.005f) {
        const size_t n = opacity_raw.size();
        std::vector<float> opacity(opacity_raw);
        opacity.insert(opacity.end(), opacity_raw.begin(), opacity_raw.end());
        std::vector<float> scaling(scaling_raw);
        scaling.insert(scaling.end(), scaling_raw.begin(), scaling_raw.end());
        std::vector<int> sources(n);
        for (size_t i = 0; i < n; ++i) {
            sources[i] = static_cast<int>(i);
        }
        auto opacity_t = Tensor::from_vector(opacity, {2 * n, 1}, Device::CUDA);
        auto scaling_t = Tensor::from_vector(scaling, {2 * n, 3}, Device::CUDA);
        const auto ratios_t = Tensor::from_vector(ratios, {n}, Device::CUDA);
        const auto sources_t = Tensor::from_vector(sources, {n}, Device::CUDA).to(DataType::Int64);
        lfs::training::mcmc::launch_relocate_copies(
            sources_t.ptr<int64_t>(), nullptr, static_cast<int64_t>(n), ratios_t.ptr<int32_t>(), min_opacity,
            scaling_t.ptr<float>(), opacity_t.ptr<float>(), n, 2 * n);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return {opacity_t.cpu().to_vector(), scaling_t.cpu().to_vector()};
    }

    float logit(const float p) { return std::log(p / (1.0f - p)); }

    // Equation (9) on the host, as the relocation kernel evaluates it.
    std::pair<float, std::vector<float>> reference_relocation(const float opacity_raw,
                                                              const std::vector<float>& scale,
                                                              const int n_idx, const float min_opacity) {
        const float opacity = std::clamp(1.0f / (1.0f + std::exp(-opacity_raw)), 1e-6f, 1.0f - 1e-6f);
        const float new_opacity = std::clamp(1.0f - std::pow(1.0f - opacity, 1.0f / static_cast<float>(n_idx)),
                                             std::max(1e-6f, min_opacity), 1.0f - 1e-6f);
        double denom = 0.0;
        for (int i = 1; i <= n_idx; ++i) {
            double binom = 1.0;
            for (int k = 0; k <= i - 1; ++k) {
                denom += binom * ((k % 2 == 0) ? 1.0 : -1.0) / std::sqrt(k + 1.0) * std::pow(new_opacity, k + 1.0);
                binom *= static_cast<double>(i - 1 - k) / static_cast<double>(k + 1);
            }
        }
        const double coeff = opacity / denom;
        std::vector<float> new_scale(3);
        for (int c = 0; c < 3; ++c) {
            new_scale[c] = static_cast<float>(std::abs(coeff * scale[c]));
        }
        return {new_opacity, new_scale};
    }
} // namespace

// Raw opacities whose sigmoid rounds to 1.0f (or is exactly 1) must not drive
// pow(1 - opacity, 1/n) and the denominator to NaN.
TEST_F(MCMCNaNFixTest, RelocateCopies_ExtremeOpacity_StaysFinite) {
    const size_t N = 100;
    std::vector<float> opacity_raw(N);
    std::vector<float> scaling_raw(N * 3);
    std::vector<int> ratios(N);
    for (size_t i = 0; i < N; ++i) {
        if (i < 10) {
            opacity_raw[i] = 16.0f; // sigmoid 0.9999999
        } else if (i < 20) {
            opacity_raw[i] = 40.0f; // sigmoid rounds to 1.0f
        } else if (i < 30) {
            opacity_raw[i] = std::numeric_limits<float>::infinity();
        } else {
            opacity_raw[i] = logit(0.5f + 0.3f * static_cast<float>(i % 10) / 10.0f);
        }
        for (int j = 0; j < 3; ++j) {
            scaling_raw[i * 3 + j] = std::log(0.01f + 0.001f * static_cast<float>(j));
        }
        ratios[i] = 2 + static_cast<int>(i % 5);
    }

    const auto rows = relocate(opacity_raw, scaling_raw, ratios);
    for (const float v : rows.opacity_raw) {
        ASSERT_TRUE(std::isfinite(v));
    }
    for (const float v : rows.scaling_raw) {
        ASSERT_TRUE(std::isfinite(v));
    }
}

TEST_F(MCMCNaNFixTest, RelocateCopies_ReproduceGaussian15975) {
    // From a crash dump: raw opacity 2.4869 (0.923216) relocated with ratio 2.
    const auto rows = relocate({2.4869f}, {std::log(0.01f), std::log(0.01f), std::log(0.01f)}, {2});
    for (const float v : rows.opacity_raw) {
        EXPECT_TRUE(std::isfinite(v));
    }
    for (const float v : rows.scaling_raw) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

// Copies and their source end with the same relocated values, matching
// Equation (9); a wrong row, ratio or raw-space conversion breaks this.
TEST_F(MCMCNaNFixTest, RelocateCopies_WritesEquationNineToSourceAndCopy) {
    const std::vector<float> test_opacities = {0.01f, 0.1f, 0.5f, 0.9f, 0.99f, 0.999f, 0.9999f};
    const std::vector<float> scale = {0.1f, 0.02f, 0.3f};
    for (const int ratio : {1, 2, 5, 13}) {
        std::vector<float> opacity_raw;
        std::vector<float> scaling_raw;
        for (const float p : test_opacities) {
            opacity_raw.push_back(logit(p));
            for (const float s : scale) {
                scaling_raw.push_back(std::log(s));
            }
        }
        const size_t n = opacity_raw.size();
        const auto rows = relocate(opacity_raw, scaling_raw, std::vector<int>(n, ratio));
        for (size_t i = 0; i < n; ++i) {
            const auto [new_opacity, new_scale] = reference_relocation(opacity_raw[i], scale, ratio, 0.005f);
            EXPECT_EQ(rows.opacity_raw[i], rows.opacity_raw[n + i]);
            EXPECT_NEAR(1.0f / (1.0f + std::exp(-rows.opacity_raw[n + i])), new_opacity, 2e-4f * new_opacity + 1e-6f)
                << "opacity " << test_opacities[i] << " ratio " << ratio;
            for (int c = 0; c < 3; ++c) {
                EXPECT_EQ(rows.scaling_raw[i * 3 + c], rows.scaling_raw[(n + i) * 3 + c]);
                EXPECT_NEAR(std::exp(rows.scaling_raw[(n + i) * 3 + c]), new_scale[c], 1e-3f * new_scale[c])
                    << "opacity " << test_opacities[i] << " ratio " << ratio;
            }
        }
    }
}
