/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Phase 1.8 — Fused MCMC noise injection (RNG + covariance transform + add).
 * Wave 5 WO-NOISE-PHILOX — Philox4_32_10 + curand_normal4 must preserve
 * N(0, σ²) moments and crush per-thread XORWOW curand_init cost.
 * Bit-identical trajectories are NOT required.
 */

#include "core/tensor.hpp"
#include "training/kernels/mcmc_kernels.hpp"
#include "training/kernels/mrnf_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    // Expected opacity-based scale for raw_opacity → −∞ (opacity → 0):
    // op_sigmoid = 1 / (1 + exp(−0.5)) ≈ 0.622459
    constexpr float kOpSigNearZero = 0.622459331f;

    struct Moments {
        double mean = 0.0;
        double var = 0.0;
        double skew = 0.0;
        double excess_kurtosis = 0.0;
    };

    Moments compute_moments(const float* p, size_t n) {
        Moments m;
        if (n == 0)
            return m;
        double sum = 0.0, sum_sq = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum += p[i];
            sum_sq += static_cast<double>(p[i]) * p[i];
        }
        m.mean = sum / static_cast<double>(n);
        m.var = sum_sq / static_cast<double>(n) - m.mean * m.mean;
        const double stddev = std::sqrt(std::max(m.var, 1e-30));
        double m3 = 0.0, m4 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double z = (static_cast<double>(p[i]) - m.mean) / stddev;
            m3 += z * z * z;
            m4 += z * z * z * z;
        }
        m.skew = m3 / static_cast<double>(n);
        m.excess_kurtosis = m4 / static_cast<double>(n) - 3.0;
        return m;
    }

    void fill_identity_quat(std::vector<float>& quat_h, size_t N) {
        quat_h.assign(N * 4, 0.f);
        for (size_t i = 0; i < N; ++i)
            quat_h[i * 4] = 1.f;
    }

    double time_inject_noise_us(
        const float* opacities,
        const float* scales,
        const float* quats,
        float* means,
        size_t N,
        uint64_t seed,
        int warmup,
        int reps) {

        for (int i = 0; i < warmup; ++i) {
            mcmc::launch_inject_noise_kernel(
                opacities, scales, quats, means,
                nullptr, 0, 1.0f, N, seed + static_cast<uint64_t>(i));
        }
        if (cudaDeviceSynchronize() != cudaSuccess)
            return -1.0;

        cudaEvent_t start = nullptr, stop = nullptr;
        if (cudaEventCreate(&start) != cudaSuccess ||
            cudaEventCreate(&stop) != cudaSuccess)
            return -1.0;

        std::vector<float> times_ms;
        times_ms.reserve(static_cast<size_t>(reps));
        for (int i = 0; i < reps; ++i) {
            // Zero means so each rep measures pure kernel cost, not growing means.
            if (cudaMemset(means, 0, N * 3 * sizeof(float)) != cudaSuccess)
                return -1.0;
            cudaEventRecord(start);
            mcmc::launch_inject_noise_kernel(
                opacities, scales, quats, means,
                nullptr, 0, 1.0f, N, seed + 1000ull + static_cast<uint64_t>(i));
            cudaEventRecord(stop);
            if (cudaEventSynchronize(stop) != cudaSuccess)
                return -1.0;
            float ms = 0.f;
            if (cudaEventElapsedTime(&ms, start, stop) != cudaSuccess)
                return -1.0;
            times_ms.push_back(ms);
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        std::sort(times_ms.begin(), times_ms.end());
        const float median_ms = times_ms[times_ms.size() / 2];
        return static_cast<double>(median_ms) * 1000.0; // µs
    }

    double time_mrnf_noise_us(
        float* means,
        const float* opacities,
        const float* vis,
        size_t N,
        uint64_t seed,
        int warmup,
        int reps) {

        for (int i = 0; i < warmup; ++i) {
            mrnf_strategy::launch_mrnf_noise_injection(
                means, opacities, vis, nullptr, 0,
                /*lr_mean=*/1.0f, /*noise_weight=*/1.0f, /*median_scale=*/1e6f,
                N, seed + static_cast<uint64_t>(i));
        }
        if (cudaDeviceSynchronize() != cudaSuccess)
            return -1.0;

        cudaEvent_t start = nullptr, stop = nullptr;
        if (cudaEventCreate(&start) != cudaSuccess ||
            cudaEventCreate(&stop) != cudaSuccess)
            return -1.0;

        std::vector<float> times_ms;
        times_ms.reserve(static_cast<size_t>(reps));
        for (int i = 0; i < reps; ++i) {
            if (cudaMemset(means, 0, N * 3 * sizeof(float)) != cudaSuccess)
                return -1.0;
            cudaEventRecord(start);
            mrnf_strategy::launch_mrnf_noise_injection(
                means, opacities, vis, nullptr, 0,
                1.0f, 1.0f, 1e6f, N, seed + 1000ull + static_cast<uint64_t>(i));
            cudaEventRecord(stop);
            if (cudaEventSynchronize(stop) != cudaSuccess)
                return -1.0;
            float ms = 0.f;
            if (cudaEventElapsedTime(&ms, start, stop) != cudaSuccess)
                return -1.0;
            times_ms.push_back(ms);
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        std::sort(times_ms.begin(), times_ms.end());
        return static_cast<double>(times_ms[times_ms.size() / 2]) * 1000.0;
    }

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
    std::vector<float> quat_h;
    fill_identity_quat(quat_h, N);
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
    const size_t n_vals = N * 3;
    const Moments m = compute_moments(p, n_vals);
    const double expected_std = static_cast<double>(lr * kOpSigNearZero);
    const double expected_var = expected_std * expected_std;

    EXPECT_NEAR(m.mean, 0.0, 0.02) << "noise mean should be ~0";
    EXPECT_NEAR(m.var, expected_var, expected_var * 0.08)
        << "var=" << m.var << " expected≈" << expected_var;
}

// Wave-5 TDD: Philox must keep Gaussian moments (not just mean/var).
TEST(FusedNoiseInjectionTest, NormalityMomentsUnchanged) {
    constexpr size_t N = 100000;
    constexpr float lr = 1.0f;
    constexpr uint64_t seed = 0xA11CEu;

    auto means = Tensor::zeros({N, size_t{3}}, Device::CUDA);
    auto opacities = Tensor::full({N}, -20.f, Device::CUDA);
    auto scales = Tensor::zeros({N, size_t{3}}, Device::CUDA);
    std::vector<float> quat_h;
    fill_identity_quat(quat_h, N);
    auto quats = Tensor::from_vector(quat_h, {N, size_t{4}}, Device::CUDA);

    mcmc::launch_inject_noise_kernel(
        opacities.ptr<float>(), scales.ptr<float>(), quats.ptr<float>(),
        means.ptr<float>(), nullptr, 0, lr, N, seed);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto cpu = means.cpu();
    const Moments m = compute_moments(cpu.ptr<float>(), N * 3);
    const double expected_var =
        static_cast<double>(lr * kOpSigNearZero) * static_cast<double>(lr * kOpSigNearZero);

    EXPECT_NEAR(m.mean, 0.0, 0.01);
    EXPECT_NEAR(m.var, expected_var, expected_var * 0.05);
    // Standard normal: skew≈0, excess kurtosis≈0. Loose tolerances for N=3e5 samples.
    EXPECT_NEAR(m.skew, 0.0, 0.05) << "skew=" << m.skew;
    EXPECT_NEAR(m.excess_kurtosis, 0.0, 0.15) << "ex_kurt=" << m.excess_kurtosis;
}

// MRNF path: weight = (1-σ(raw_op))^150 * lr * noise_weight with vis>0.
// raw_op→−∞ ⇒ σ→0 ⇒ weight→lr*noise_weight; clamp disabled via large median_scale.
TEST(FusedNoiseInjectionTest, MrnfNoiseMeanVarNormal) {
    constexpr size_t N = 80000;
    constexpr float lr = 1.0f;
    constexpr float noise_weight = 0.5f;
    constexpr uint64_t seed = 0xB0A7u;

    auto means = Tensor::zeros({N, size_t{3}}, Device::CUDA);
    auto opacities = Tensor::full({N}, -20.f, Device::CUDA); // σ≈0 → inv_op≈1 → weight^150≈1
    auto vis = Tensor::full({N}, 1.f, Device::CUDA);

    mrnf_strategy::launch_mrnf_noise_injection(
        means.ptr<float>(),
        opacities.ptr<float>(),
        vis.ptr<float>(),
        nullptr, 0,
        lr, noise_weight,
        /*median_scale=*/1e6f,
        N, seed);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto cpu = means.cpu();
    const Moments m = compute_moments(cpu.ptr<float>(), N * 3);
    const double expected_std = static_cast<double>(lr * noise_weight);
    const double expected_var = expected_std * expected_std;

    EXPECT_NEAR(m.mean, 0.0, 0.015) << "mrnf mean=" << m.mean;
    EXPECT_NEAR(m.var, expected_var, expected_var * 0.08)
        << "mrnf var=" << m.var << " expected≈" << expected_var;
    EXPECT_NEAR(m.skew, 0.0, 0.06) << "mrnf skew=" << m.skew;
    EXPECT_NEAR(m.excess_kurtosis, 0.0, 0.2) << "mrnf ex_kurt=" << m.excess_kurtosis;
}

TEST(FusedNoiseInjectionTest, FrozenMaskBlocksNoise) {
    constexpr size_t N = 256;
    constexpr uint64_t seed = 42;

    auto means = Tensor::zeros({N, size_t{3}}, Device::CUDA);
    auto opacities = Tensor::full({N}, -20.f, Device::CUDA);
    auto scales = Tensor::zeros({N, size_t{3}}, Device::CUDA);
    std::vector<float> quat_h;
    fill_identity_quat(quat_h, N);
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

// Directive-2 microbench shape: N=400k. XORWOW curand_init path is ~1ms+;
// Philox + normal4 is ~6.5µs. Gate: >100x → median must be well under 50µs.
TEST(FusedNoiseInjectionTest, PhiloxKernelTimeMcmcInjectUnder50usAt400k) {
    constexpr size_t N = 400000;
    constexpr uint64_t seed = 0x51EEdu;

    auto means = Tensor::zeros({N, size_t{3}}, Device::CUDA);
    auto opacities = Tensor::full({N}, -20.f, Device::CUDA);
    auto scales = Tensor::zeros({N, size_t{3}}, Device::CUDA);
    std::vector<float> quat_h;
    fill_identity_quat(quat_h, N);
    auto quats = Tensor::from_vector(quat_h, {N, size_t{4}}, Device::CUDA);

    const double median_us = time_inject_noise_us(
        opacities.ptr<float>(), scales.ptr<float>(), quats.ptr<float>(),
        means.ptr<float>(), N, seed, /*warmup=*/5, /*reps=*/21);

    // Recorded for PROGRESS.md; threshold encodes the >100x claim vs ~1ms XORWOW.
    std::cout << "[PhiloxKernelTime] mcmc inject_noise N=400k median_us=" << median_us
              << std::endl;
    EXPECT_LT(median_us, 50.0)
        << "inject_noise kernel median " << median_us
        << " µs — expected <50µs with Philox (XORWOW baseline ~1ms+)";
}

TEST(FusedNoiseInjectionTest, PhiloxKernelTimeMrnfUnder50usAt400k) {
    constexpr size_t N = 400000;
    constexpr uint64_t seed = 0x51EEu;

    auto means = Tensor::zeros({N, size_t{3}}, Device::CUDA);
    // raw_op≈−20 → full weight path (no early return before RNG) for every thread.
    auto opacities = Tensor::full({N}, -20.f, Device::CUDA);
    auto vis = Tensor::full({N}, 1.f, Device::CUDA);

    const double median_us = time_mrnf_noise_us(
        means.ptr<float>(), opacities.ptr<float>(), vis.ptr<float>(),
        N, seed, /*warmup=*/5, /*reps=*/21);

    std::cout << "[PhiloxKernelTime] mrnf_noise_injection N=400k median_us=" << median_us
              << std::endl;
    EXPECT_LT(median_us, 50.0)
        << "mrnf_noise kernel median " << median_us
        << " µs — expected <50µs with Philox (XORWOW baseline ~1ms+)";
}
