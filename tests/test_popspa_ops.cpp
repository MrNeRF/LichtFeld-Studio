/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
// The default backend's PopspaOps against host oracles: selection order, the ADMM
// proxy/dual/penalty and the effective-rank terms of the CUDA kernels, in double.
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"
#include "training/components/popspa_controller.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {
    using namespace lfs::core;
    using lfs::training::POPSpaController;

    class POPSpaOps : public ::testing::Test {
    protected:
        void SetUp() override {
            if (!gpu_backend_available(default_gpu_backend()))
                GTEST_SKIP() << "GPU backend required";
            ops_ = lfs::training::training_ops(default_gpu_backend()).popspa;
            ASSERT_NE(ops_, nullptr);
        }
        Tensor scores(const std::vector<double>& values) const {
            auto host = Tensor::empty({values.size(), 8}, Device::CPU, DataType::UInt8);
            for (size_t i = 0; i < values.size(); ++i)
                lfs::gpu_ops::encode_pop_score(values[i], static_cast<std::byte*>(host.data_ptr()) + 8 * i,
                                               ops_->score_encoding);
            return host.gpu();
        }
        static Tensor mask(const std::vector<uint8_t>& values) {
            std::vector<bool> bits(values.begin(), values.end());
            return Tensor::from_vector(bits, {values.size()}, Device::GPU);
        }
        const lfs::gpu_ops::PopspaOps* ops_ = nullptr;
    };

    float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

    // Host transcription of effective_rank_row in effective_rank_regularization.cu.
    double rank_row(const float* raw, double* grad, double inv_count, double rw, double tw, double eps, bool frozen) {
        double scales[3], eigen[3], p[3], dh[3], sum = 0;
        int minimum = 0;
        for (int d = 0; d < 3; ++d) {
            scales[d] = std::exp(static_cast<double>(raw[d]));
            eigen[d] = scales[d] * scales[d];
            sum += eigen[d];
            if (scales[d] < scales[minimum])
                minimum = d;
        }
        const double denom = std::max(sum, eps);
        double entropy = 0, weighted_dh = 0;
        for (int d = 0; d < 3; ++d) {
            p[d] = eigen[d] / denom;
            const double logp = std::log(std::max(p[d], eps));
            entropy -= p[d] * logp;
            dh[d] = -logp - (p[d] >= eps ? 1.0 : 0.0);
            weighted_dh += p[d] * dh[d];
        }
        const double rank = std::exp(entropy);
        const double argument = rank - 1.0 + eps;
        const double needle = std::max(-std::log(argument), 0.0);
        if (!frozen) {
            const double rank_factor = argument <= 1.0 ? -rw * rank / argument : 0.0;
            for (int d = 0; d < 3; ++d) {
                const double dh_dlog = 2.0 * p[d] * (dh[d] - (sum >= eps ? weighted_dh : 0.0));
                grad[d] = (rank_factor * dh_dlog + (d == minimum ? tw * scales[d] : 0.0)) * inv_count;
            }
        }
        return (rw * needle + tw * scales[minimum]) * inv_count;
    }
} // namespace

TEST_F(POPSpaOps, KeepIndicesMatchHostOracleWithTiesFrozenAndInactiveRows) {
    std::mt19937 rng(7);
    for (const size_t n : {size_t{5}, size_t{257}, size_t{3001}}) {
        std::vector<double> values(n);
        std::vector<uint8_t> active(n), frozen(n);
        size_t active_count = 0, frozen_count = 0;
        for (size_t i = 0; i < n; ++i) {
            // Few distinct values force ties; large values share their float hi and
            // differ only in lo, which the second ordering level must resolve.
            const int bucket = static_cast<int>(rng() % 9);
            values[i] = bucket < 6 ? bucket * 0.25 : 1.0e8 + 0.125 * (bucket - 6);
            active[i] = rng() % 7 != 0;
            frozen[i] = active[i] && rng() % 11 == 0;
            active_count += active[i];
            frozen_count += frozen[i];
        }
        const auto device_scores = scores(values);
        const auto active_mask = mask(active), frozen_mask = mask(frozen);
        for (const size_t k : {frozen_count + 1, (frozen_count + active_count) / 2, active_count}) {
            if (k == 0 || k < frozen_count || k > active_count)
                continue;
            const auto expected = POPSpaController::select_keep_indices(values, k, active, frozen);
            ASSERT_TRUE(expected.has_value());
            Tensor keep;
            ops_->keep_indices(device_scores, active_mask, frozen_mask, k, keep);
            EXPECT_EQ(keep.to_vector_int64(), *expected) << "n=" << n << " k=" << k;
        }
    }
}

TEST_F(POPSpaOps, ScoreExportIsExactFloat64) {
    if (!ops_->export_scores)
        GTEST_SKIP() << "scores are FP64 on this backend";
    std::mt19937 rng(3);
    std::vector<double> values{0.0, 1.0, 0.1, 1e-30, 3.5e30, 1.0e8 + 0.125, 7.0 / 3.0};
    for (int i = 0; i < 1000; ++i)
        values.push_back(std::ldexp(static_cast<double>(rng()) / 4294967296.0, static_cast<int>(rng() % 120) - 60));
    const auto pairs = scores(values);
    auto float64 = Tensor::empty(pairs.shape(), Device::GPU, DataType::UInt8);
    ops_->export_scores(pairs, float64);
    const auto host_pairs = pairs.cpu(), host = float64.cpu();
    for (size_t i = 0; i < values.size(); ++i) {
        const double expected = lfs::gpu_ops::decode_pop_score(static_cast<const std::byte*>(host_pairs.data_ptr()) + 8 * i,
                                                               ops_->score_encoding);
        const double actual = lfs::gpu_ops::decode_pop_score(static_cast<const std::byte*>(host.data_ptr()) + 8 * i,
                                                             lfs::gpu_ops::PopScoreEncoding::Float64);
        EXPECT_EQ(actual, expected) << "row " << i << " value " << values[i];
    }
}

TEST_F(POPSpaOps, ProjectAndPenaltyMatchHostReference) {
    constexpr size_t n = 2049, k = 300;
    constexpr float rho = 0.3f;
    std::mt19937 rng(11);
    // The host oracle takes nonnegative values: sigmoid(-4) + u stays above zero.
    std::uniform_real_distribution<float> spread(-4.f, 4.f), dual(-0.015f, 0.2f);
    std::vector<float> raw(n), u(n);
    std::vector<uint8_t> active(n), frozen(n);
    for (size_t i = 0; i < n; ++i) {
        raw[i] = spread(rng);
        u[i] = dual(rng);
        active[i] = i % 13 != 4;
        frozen[i] = active[i] && i % 97 == 5;
    }
    std::vector<double> shifted(n);
    for (size_t i = 0; i < n; ++i)
        shifted[i] = sigmoid(raw[i]) + u[i];
    const auto keep = POPSpaController::select_keep_indices(shifted, k, active, frozen);
    ASSERT_TRUE(keep.has_value());
    std::vector<float> z(n, 0.f), u_next = u;
    for (const auto row : *keep)
        z[row] = static_cast<float>(shifted[row]);
    for (size_t i = 0; i < n; ++i)
        if (active[i] && !frozen[i])
            u_next[i] += sigmoid(raw[i]) - z[i];

    const auto raw_t = Tensor::from_vector(raw, {n, 1}, Device::GPU);
    const auto active_t = mask(active), frozen_t = mask(frozen);
    auto z_t = Tensor::zeros({n, 1}, Device::GPU), u_t = Tensor::from_vector(u, {n, 1}, Device::GPU);
    ops_->project(raw_t, z_t, u_t, active_t, frozen_t, k, true);
    const auto z_gpu = z_t.to_vector(), u_gpu = u_t.to_vector();
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(z_gpu[i], z[i], 2e-6f) << "z row " << i;
        EXPECT_NEAR(u_gpu[i], u_next[i], 4e-6f) << "u row " << i;
    }

    std::vector<float> grad(n, 0.25f);
    double loss = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!active[i] || frozen[i])
            continue;
        const float a = sigmoid(raw[i]);
        const float d = a - z[i] + u_next[i];
        grad[i] += rho * d * a * (1.f - a);
        loss += 0.5 * rho * d * d;
    }
    auto grad_t = Tensor::full({n, 1}, 0.25f, Device::GPU);
    auto loss_t = Tensor::full({1}, 2.f, Device::GPU);
    ops_->penalty(raw_t, z_t, u_t, active_t, frozen_t, grad_t, loss_t, rho);
    const auto grad_gpu = grad_t.to_vector();
    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(grad_gpu[i], grad[i], 2e-6f) << "gradient row " << i;
    EXPECT_NEAR(loss_t.to_vector()[0], 2.0 + loss, 1e-5 * (2.0 + loss));
}

TEST_F(POPSpaOps, EffectiveRankMatchesDoubleReferenceNearRankOne) {
    // Rows from isotropic to needle-like, plus scales whose variance sum is below
    // epsilon (the unnormalized branch) and rows with an exact share cut-off.
    std::vector<std::array<float, 3>> rows{
        {0.f, 0.f, 0.f},
        {0.4f, 0.2f, -0.1f},
        {1.3f, -1.7f, -2.6f},
        {0.f, -3.f, -3.f},
        {0.f, -5.f, -6.f},
        {0.f, -7.f, -7.5f},
        {0.f, -8.f, -8.f},
        {2.f, -6.f, -9.f},
        {-9.f, -9.5f, -10.f},
        {-8.f, -12.f, -12.f},
        {0.f, -9.f, -2.f},
        {-1.f, -1.f, -1.0001f}};
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> spread(-6.f, 1.f);
    for (int i = 0; i < 500; ++i)
        rows.push_back({spread(rng), spread(rng), spread(rng)});
    const size_t n = rows.size();
    std::vector<float> raw(3 * n);
    std::vector<uint8_t> active(n), frozen(n);
    size_t active_count = 0;
    for (size_t i = 0; i < n; ++i) {
        std::copy(rows[i].begin(), rows[i].end(), raw.begin() + 3 * i);
        active[i] = i % 17 != 3;
        frozen[i] = active[i] && i % 23 == 7;
        active_count += active[i];
    }
    constexpr double rw = 0.01, tw = 1.0, eps = 1e-7;
    const double inv_count = 1.0 / static_cast<double>(active_count);
    std::vector<double> grad(3 * n, 0.0);
    double loss = 0;
    for (size_t i = 0; i < n; ++i)
        if (active[i])
            loss += rank_row(&raw[3 * i], &grad[3 * i], inv_count, rw, tw, eps, frozen[i]);

    const auto raw_t = Tensor::from_vector(raw, {n, 3}, Device::GPU);
    auto grad_t = Tensor::zeros({n, 3}, Device::GPU), loss_t = Tensor::zeros({1}, Device::GPU);
    ops_->effective_rank(raw_t, grad_t, mask(active), mask(frozen), loss_t, active_count, static_cast<float>(rw),
                         static_cast<float>(tw), static_cast<float>(eps));
    EXPECT_NEAR(loss_t.to_vector()[0], loss, 2e-5 * loss);
    const auto actual = grad_t.to_vector();
    for (size_t i = 0; i < 3 * n; ++i)
        // MoltenVK compiles with fast math (~0.3% here); a naive float rank-1 is off by ~16%.
        EXPECT_NEAR(actual[i], grad[i], 1e-9 + 1e-2 * std::abs(grad[i]))
            << "row " << i / 3 << " axis " << i % 3 << " raw " << raw[i - i % 3] << ' ' << raw[i - i % 3 + 1] << ' '
            << raw[i - i % 3 + 2];
}
