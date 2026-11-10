/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Tests for NaN fixes in MCMC relocation kernel

#include "core/cuda/memory_arena.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "cuda_backend_test.hpp"
#include "training/kernels/mcmc_kernels.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace lfs::core;

class MCMCNaNFixTest : public lfs::test::CudaBackendTest {
protected:
    void SetUp() override {
        LFS_CUDA_BACKEND_OR_RETURN();
        cudaDeviceSynchronize();
        lfs::training::mcmc::init_relocation_coefficients(51);
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }

    bool hasNaNOrInf(const Tensor& t) {
        auto cpu = t.cpu();
        const float* data = cpu.ptr<float>();
        for (size_t i = 0; i < cpu.numel(); ++i) {
            if (std::isnan(data[i]) || std::isinf(data[i])) {
                return true;
            }
        }
        return false;
    }

    std::pair<int, int> countNaNInf(const Tensor& t) {
        auto cpu = t.cpu();
        const float* data = cpu.ptr<float>();
        int nan_count = 0, inf_count = 0;
        for (size_t i = 0; i < cpu.numel(); ++i) {
            if (std::isnan(data[i]))
                ++nan_count;
            if (std::isinf(data[i]))
                ++inf_count;
        }
        return {nan_count, inf_count};
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
        auto opacity_t = Tensor::from_vector(opacity, {2 * n, 1}, Device::GPU);
        auto scaling_t = Tensor::from_vector(scaling, {2 * n, 3}, Device::GPU);
        const auto ratios_t = Tensor::from_vector(ratios, {n}, Device::GPU);
        const auto sources_t = Tensor::from_vector(sources, {n}, Device::GPU).to(DataType::Int64);
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

// Test that relocation kernel produces NaN with extreme opacity values
TEST_F(MCMCNaNFixTest, RelocationKernel_ExtremeOpacity_ProducesNaN) {
    const size_t N = 100;

    // Create test data with some extreme opacities
    std::vector<float> opacities_data(N);
    std::vector<float> scales_data(N * 3);
    std::vector<int> ratios_data(N);

    for (size_t i = 0; i < N; ++i) {
        // Most are normal, but some are extreme
        if (i < 10) {
            opacities_data[i] = 0.9999999f; // Very high opacity - causes denom_sum issues
        } else if (i < 20) {
            opacities_data[i] = 1.0f; // Exactly 1.0 - pow(0, 1/n) = 0, causes issues
        } else if (i < 30) {
            opacities_data[i] = 1.0f + 1e-7f; // Just above 1.0 - pow(negative, 1/n) = NaN
        } else {
            opacities_data[i] = 0.5f + 0.3f * static_cast<float>(i % 10) / 10.0f; // Normal range
        }

        for (int j = 0; j < 3; ++j) {
            scales_data[i * 3 + j] = 0.01f + 0.001f * static_cast<float>(j);
        }

        ratios_data[i] = 2 + static_cast<int>(i % 5); // Ratios between 2-6
    }

    Tensor opacities = Tensor::from_vector(opacities_data, {N}, Device::CPU).to(Device::GPU);
    Tensor scales = Tensor::from_vector(scales_data, {N, 3}, Device::CPU).to(Device::GPU);
    Tensor ratios = Tensor::from_vector(ratios_data, {N}, Device::CPU).to(Device::GPU);

    // Output tensors
    Tensor new_opacities = Tensor::empty({N}, Device::GPU, DataType::Float32);
    Tensor new_scales = Tensor::empty({N, 3}, Device::GPU, DataType::Float32);

    lfs::training::mcmc::launch_relocation_kernel(
        opacities.ptr<float>(),
        scales.ptr<float>(),
        ratios.ptr<int32_t>(),
        0.005f,
        new_opacities.ptr<float>(),
        new_scales.ptr<float>(),
        N);
    cudaDeviceSynchronize();

    // Check for NaN/Inf in outputs
    auto [nan_opac, inf_opac] = countNaNInf(new_opacities);
    auto [nan_scale, inf_scale] = countNaNInf(new_scales);

    LOG_INFO("Relocation kernel results:");
    LOG_INFO("  new_opacities: NaN={}, Inf={}", nan_opac, inf_opac);
    LOG_INFO("  new_scales: NaN={}, Inf={}", nan_scale, inf_scale);

    // After fix, these should all pass with no NaN/Inf:
    EXPECT_EQ(nan_opac, 0) << "new_opacities contains NaN";
    EXPECT_EQ(inf_opac, 0) << "new_opacities contains Inf";
    EXPECT_EQ(nan_scale, 0) << "new_scales contains NaN";
    EXPECT_EQ(inf_scale, 0) << "new_scales contains Inf";
}

// Test that normal opacity values don't produce NaN
TEST_F(MCMCNaNFixTest, RelocationKernel_NormalOpacity_NoNaN) {
    const size_t N = 1000;

    // Create test data with normal opacity values
    std::vector<float> opacities_data(N);
    std::vector<float> scales_data(N * 3);
    std::vector<int> ratios_data(N);

    for (size_t i = 0; i < N; ++i) {
        opacities_data[i] = 0.1f + 0.8f * static_cast<float>(i % 100) / 100.0f; // 0.1 to 0.9

        for (int j = 0; j < 3; ++j) {
            scales_data[i * 3 + j] = 0.01f + 0.001f * static_cast<float>(j);
        }

        ratios_data[i] = 1 + static_cast<int>(i % 10); // Ratios between 1-10
    }

    Tensor opacities = Tensor::from_vector(opacities_data, {N}, Device::CPU).to(Device::GPU);
    Tensor scales = Tensor::from_vector(scales_data, {N, 3}, Device::CPU).to(Device::GPU);
    Tensor ratios = Tensor::from_vector(ratios_data, {N}, Device::CPU).to(Device::GPU);

    Tensor new_opacities = Tensor::empty({N}, Device::GPU, DataType::Float32);
    Tensor new_scales = Tensor::empty({N, 3}, Device::GPU, DataType::Float32);

    lfs::training::mcmc::launch_relocation_kernel(
        opacities.ptr<float>(),
        scales.ptr<float>(),
        ratios.ptr<int32_t>(),
        0.005f,
        new_opacities.ptr<float>(),
        new_scales.ptr<float>(),
        N);
    cudaDeviceSynchronize();

    auto [nan_opac, inf_opac] = countNaNInf(new_opacities);
    auto [nan_scale, inf_scale] = countNaNInf(new_scales);

    EXPECT_EQ(nan_opac, 0) << "new_opacities contains NaN";
    EXPECT_EQ(inf_opac, 0) << "new_opacities contains Inf";
    EXPECT_EQ(nan_scale, 0) << "new_scales contains NaN";
    EXPECT_EQ(inf_scale, 0) << "new_scales contains Inf";
}

TEST_F(MCMCNaNFixTest, RelocationKernel_ReproduceGaussian15975) {
    // From the crash dump analysis:
    // Gaussian 15975: raw_opacity=2.4869, opacity=0.923216
    // The NaN appeared in position, scale, rotation - all from relocation

    const size_t N = 1;

    // Use the exact opacity that caused the issue
    float opacity_sigmoid = 0.923216f;
    std::vector<float> opacities_data = {opacity_sigmoid};
    std::vector<float> scales_data = {0.01f, 0.01f, 0.01f}; // Some scales
    std::vector<int> ratios_data = {2};                     // Ratio of 2

    Tensor opacities = Tensor::from_vector(opacities_data, {N}, Device::CPU).to(Device::GPU);
    Tensor scales = Tensor::from_vector(scales_data, {N, 3}, Device::CPU).to(Device::GPU);
    Tensor ratios = Tensor::from_vector(ratios_data, {N}, Device::CPU).to(Device::GPU);

    Tensor new_opacities = Tensor::empty({N}, Device::GPU, DataType::Float32);
    Tensor new_scales = Tensor::empty({N, 3}, Device::GPU, DataType::Float32);

    lfs::training::mcmc::launch_relocation_kernel(
        opacities.ptr<float>(),
        scales.ptr<float>(),
        ratios.ptr<int32_t>(),
        0.005f,
        new_opacities.ptr<float>(),
        new_scales.ptr<float>(),
        N);
    cudaDeviceSynchronize();

    // Print the results
    Tensor new_opac_cpu = new_opacities.cpu();
    Tensor new_scales_cpu = new_scales.cpu();

    LOG_INFO("Gaussian 15975 reproduction:");
    LOG_INFO("  Input opacity: {}", opacity_sigmoid);
    LOG_INFO("  Output new_opacity: {}", new_opac_cpu.ptr<float>()[0]);
    LOG_INFO("  Output new_scales: ({}, {}, {})",
             new_scales_cpu.ptr<float>()[0],
             new_scales_cpu.ptr<float>()[1],
             new_scales_cpu.ptr<float>()[2]);

    EXPECT_FALSE(std::isnan(new_opac_cpu.ptr<float>()[0])) << "new_opacity is NaN";
    EXPECT_FALSE(std::isinf(new_opac_cpu.ptr<float>()[0])) << "new_opacity is Inf";
    for (int i = 0; i < 3; ++i) {
        EXPECT_FALSE(std::isnan(new_scales_cpu.ptr<float>()[i])) << "new_scales[" << i << "] is NaN";
        EXPECT_FALSE(std::isinf(new_scales_cpu.ptr<float>()[i])) << "new_scales[" << i << "] is Inf";
    }
}

// Test that the log() of new_scales doesn't produce NaN for valid inputs
TEST_F(MCMCNaNFixTest, NewScalesLog_ProducesValidOutput) {
    const size_t N = 100;

    // Test with various opacity values, including edge cases
    std::vector<float> test_opacities = {0.01f, 0.1f, 0.5f, 0.9f, 0.99f, 0.999f, 0.9999f};

    for (float test_opacity : test_opacities) {
        std::vector<float> opacities_data(N, test_opacity);
        std::vector<float> scales_data(N * 3, 0.1f);
        std::vector<int> ratios_data(N, 2);

        Tensor opacities = Tensor::from_vector(opacities_data, {N}, Device::CPU).to(Device::GPU);
        Tensor scales = Tensor::from_vector(scales_data, {N, 3}, Device::CPU).to(Device::GPU);
        Tensor ratios = Tensor::from_vector(ratios_data, {N}, Device::CPU).to(Device::GPU);

        Tensor new_opacities = Tensor::empty({N}, Device::GPU, DataType::Float32);
        Tensor new_scales = Tensor::empty({N, 3}, Device::GPU, DataType::Float32);

        lfs::training::mcmc::launch_relocation_kernel(
            opacities.ptr<float>(),
            scales.ptr<float>(),
            ratios.ptr<int32_t>(),
            0.005f,
            new_opacities.ptr<float>(),
            new_scales.ptr<float>(),
            N);
        cudaDeviceSynchronize();

        // Check new_scales before log
        auto [nan_scale, inf_scale] = countNaNInf(new_scales);

        // Now apply log and check
        Tensor new_scales_log = new_scales.log();
        auto [nan_scale_log, inf_scale_log] = countNaNInf(new_scales_log);

        if (nan_scale > 0 || inf_scale > 0 || nan_scale_log > 0 || inf_scale_log > 0) {
            LOG_WARN("opacity={}: new_scales NaN={}, Inf={}; log(new_scales) NaN={}, Inf={}",
                     test_opacity, nan_scale, inf_scale, nan_scale_log, inf_scale_log);
        }

        // After fix, these should all pass:
        EXPECT_EQ(nan_scale, 0) << "new_scales NaN for opacity=" << test_opacity;
        EXPECT_EQ(inf_scale, 0) << "new_scales Inf for opacity=" << test_opacity;
        EXPECT_EQ(nan_scale_log, 0) << "log(new_scales) NaN for opacity=" << test_opacity;
    }
}

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
