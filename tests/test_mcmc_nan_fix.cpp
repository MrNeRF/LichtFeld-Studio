/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Tests for NaN fixes in MCMC relocation kernel

#include "core/cuda/memory_arena.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "training/kernels/mcmc_kernels.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;

class MCMCNaNFixTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceSynchronize();
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }

    // Helper to check for NaN/Inf in a tensor
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

    // Count NaN/Inf values
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

    // Create binomial coefficient matrix (n_max x n_max)
    Tensor createBinomials(int n_max) {
        std::vector<float> binoms(n_max * n_max, 0.0f);
        for (int n = 0; n < n_max; ++n) {
            binoms[n * n_max + 0] = 1.0f;
            for (int k = 1; k <= n; ++k) {
                binoms[n * n_max + k] = binoms[(n - 1) * n_max + k - 1] + binoms[(n - 1) * n_max + k];
            }
        }
        auto t = Tensor::from_vector(binoms, {static_cast<size_t>(n_max), static_cast<size_t>(n_max)}, Device::CPU);
        return t.to(Device::CUDA);
    }
};

// Test that relocation kernel produces NaN with extreme opacity values
TEST_F(MCMCNaNFixTest, RelocationKernel_ExtremeOpacity_ProducesNaN) {
    const size_t N = 100;
    const int n_max = 51;

    // Create binomial coefficients
    Tensor binoms = createBinomials(n_max);

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

    Tensor opacities = Tensor::from_vector(opacities_data, {N}, Device::CPU).to(Device::CUDA);
    Tensor scales = Tensor::from_vector(scales_data, {N, 3}, Device::CPU).to(Device::CUDA);
    Tensor ratios = Tensor::from_vector(ratios_data, {N}, Device::CPU).to(Device::CUDA);

    // Output tensors
    Tensor new_opacities = Tensor::empty({N}, Device::CUDA, DataType::Float32);
    Tensor new_scales = Tensor::empty({N, 3}, Device::CUDA, DataType::Float32);

    // Run the relocation kernel
    lfs::training::mcmc::launch_relocation_kernel(
        opacities.ptr<float>(),
        scales.ptr<float>(),
        ratios.ptr<int32_t>(),
        binoms.ptr<float>(),
        n_max,
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
    const int n_max = 51;

    Tensor binoms = createBinomials(n_max);

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

    Tensor opacities = Tensor::from_vector(opacities_data, {N}, Device::CPU).to(Device::CUDA);
    Tensor scales = Tensor::from_vector(scales_data, {N, 3}, Device::CPU).to(Device::CUDA);
    Tensor ratios = Tensor::from_vector(ratios_data, {N}, Device::CPU).to(Device::CUDA);

    Tensor new_opacities = Tensor::empty({N}, Device::CUDA, DataType::Float32);
    Tensor new_scales = Tensor::empty({N, 3}, Device::CUDA, DataType::Float32);

    lfs::training::mcmc::launch_relocation_kernel(
        opacities.ptr<float>(),
        scales.ptr<float>(),
        ratios.ptr<int32_t>(),
        binoms.ptr<float>(),
        n_max,
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

// Test the specific Gaussian that caused NaN in the real training run
TEST_F(MCMCNaNFixTest, RelocationKernel_ReproduceGaussian15975) {
    // From the crash dump analysis:
    // Gaussian 15975: raw_opacity=2.4869, opacity=0.923216
    // The NaN appeared in position, scale, rotation - all from relocation

    const size_t N = 1;
    const int n_max = 51;

    Tensor binoms = createBinomials(n_max);

    // Use the exact opacity that caused the issue
    float opacity_sigmoid = 0.923216f;
    std::vector<float> opacities_data = {opacity_sigmoid};
    std::vector<float> scales_data = {0.01f, 0.01f, 0.01f}; // Some scales
    std::vector<int> ratios_data = {2};                     // Ratio of 2

    Tensor opacities = Tensor::from_vector(opacities_data, {N}, Device::CPU).to(Device::CUDA);
    Tensor scales = Tensor::from_vector(scales_data, {N, 3}, Device::CPU).to(Device::CUDA);
    Tensor ratios = Tensor::from_vector(ratios_data, {N}, Device::CPU).to(Device::CUDA);

    Tensor new_opacities = Tensor::empty({N}, Device::CUDA, DataType::Float32);
    Tensor new_scales = Tensor::empty({N, 3}, Device::CUDA, DataType::Float32);

    lfs::training::mcmc::launch_relocation_kernel(
        opacities.ptr<float>(),
        scales.ptr<float>(),
        ratios.ptr<int32_t>(),
        binoms.ptr<float>(),
        n_max,
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
    const int n_max = 51;

    Tensor binoms = createBinomials(n_max);

    // Test with various opacity values, including edge cases
    std::vector<float> test_opacities = {0.01f, 0.1f, 0.5f, 0.9f, 0.99f, 0.999f, 0.9999f};

    for (float test_opacity : test_opacities) {
        std::vector<float> opacities_data(N, test_opacity);
        std::vector<float> scales_data(N * 3, 0.1f);
        std::vector<int> ratios_data(N, 2);

        Tensor opacities = Tensor::from_vector(opacities_data, {N}, Device::CPU).to(Device::CUDA);
        Tensor scales = Tensor::from_vector(scales_data, {N, 3}, Device::CPU).to(Device::CUDA);
        Tensor ratios = Tensor::from_vector(ratios_data, {N}, Device::CPU).to(Device::CUDA);

        Tensor new_opacities = Tensor::empty({N}, Device::CUDA, DataType::Float32);
        Tensor new_scales = Tensor::empty({N, 3}, Device::CUDA, DataType::Float32);

        lfs::training::mcmc::launch_relocation_kernel(
            opacities.ptr<float>(),
            scales.ptr<float>(),
            ratios.ptr<int32_t>(),
            binoms.ptr<float>(),
            n_max,
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
