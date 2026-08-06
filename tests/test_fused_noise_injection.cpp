/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Phase 1.8 — Fused MCMC noise injection (RNG + covariance transform + add).
 * Distribution must match N(0, σ²) with σ = noise_factor * (cov^{1/2} scale);
 * bit-identical trajectories are NOT required.
 */

#include "core/tensor.hpp"
#include "training/kernels/mcmc_kernels.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    // Expected opacity-based scale for raw_opacity → −∞ (opacity → 0):
    // op_sigmoid = 1 / (1 + exp(−0.5)) ≈ 0.622459
    constexpr float kOpSigNearZero = 0.622459331f;

} // namespace

TEST(FusedNoiseInjectionTest, MeanAndVarMatchIdentityCovariance) {
    constexpr size_t N = 50000;
    constexpr float lr = 1.0f;
    constexpr uint64_t seed = 0xC0FFEEu;

    // Identity rotation (w=1), raw_scale=0 → S²=I → cov=I.
    // raw_opacity = −20 → opacity≈0 → noise_factor ≈ lr * kOpSigNearZero.
    auto means = Tensor::zeros({N, size_t{3}}, Device::CUDA);
    auto opacities = Tensor::full({N}, -20.f, Device::CUDA);
    auto scales = Tensor::zeros({N, size_t{3}}, Device::CUDA);
    std::vector<float> quat_h(N * 4, 0.f);
    for (size_t i = 0; i < N; ++i)
        quat_h[i * 4] = 1.f;
    auto quats = Tensor::from_vector(quat_h, {N, size_t{4}}, Device::CUDA);

    mcmc::launch_inject_noise_kernel(
        opacities.ptr<float>(),
        scales.ptr<float>(),
        quats.ptr<float>(),
        means.ptr<float>(),
        /*frozen_mask=*/nullptr,
        /*frozen_mask_size=*/0,
        lr,
        N,
        seed);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto cpu = means.cpu();
    const float* p = cpu.ptr<float>();
    double sum = 0.0, sum_sq = 0.0;
    const size_t n_vals = N * 3;
    for (size_t i = 0; i < n_vals; ++i) {
        sum += p[i];
        sum_sq += static_cast<double>(p[i]) * p[i];
    }
    const double mean = sum / static_cast<double>(n_vals);
    const double var = sum_sq / static_cast<double>(n_vals) - mean * mean;
    const double expected_std = static_cast<double>(lr * kOpSigNearZero);
    const double expected_var = expected_std * expected_std;

    EXPECT_NEAR(mean, 0.0, 0.02) << "noise mean should be ~0";
    EXPECT_NEAR(var, expected_var, expected_var * 0.08)
        << "var=" << var << " expected≈" << expected_var;
}

TEST(FusedNoiseInjectionTest, FrozenMaskBlocksNoise) {
    constexpr size_t N = 256;
    constexpr uint64_t seed = 42;

    auto means = Tensor::zeros({N, size_t{3}}, Device::CUDA);
    auto opacities = Tensor::full({N}, -20.f, Device::CUDA);
    auto scales = Tensor::zeros({N, size_t{3}}, Device::CUDA);
    std::vector<float> quat_h(N * 4, 0.f);
    for (size_t i = 0; i < N; ++i)
        quat_h[i * 4] = 1.f;
    auto quats = Tensor::from_vector(quat_h, {N, size_t{4}}, Device::CUDA);

    std::vector<bool> frozen(N, false);
    for (size_t i = 0; i < N / 2; ++i)
        frozen[i] = true;
    auto frozen_t = Tensor::from_vector(frozen, TensorShape({N}), Device::CUDA);

    mcmc::launch_inject_noise_kernel(
        opacities.ptr<float>(),
        scales.ptr<float>(),
        quats.ptr<float>(),
        means.ptr<float>(),
        frozen_t.ptr<bool>(),
        N,
        1.0f,
        N,
        seed);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto cpu = means.cpu();
    const float* p = cpu.ptr<float>();
    for (size_t i = 0; i < N / 2; ++i) {
        EXPECT_FLOAT_EQ(p[i * 3 + 0], 0.f);
        EXPECT_FLOAT_EQ(p[i * 3 + 1], 0.f);
        EXPECT_FLOAT_EQ(p[i * 3 + 2], 0.f);
    }
    // Unfrozen half should move (with overwhelming probability).
    double energy = 0.0;
    for (size_t i = N / 2; i < N; ++i) {
        energy += std::abs(p[i * 3]) + std::abs(p[i * 3 + 1]) + std::abs(p[i * 3 + 2]);
    }
    EXPECT_GT(energy, 1.0) << "unfrozen rows should receive noise";
}
