/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_splat.hpp"
#include <algorithm>
#include <cmath>
#include <glm/gtc/quaternion.hpp>
#include <gtest/gtest.h>
#include <optional>
#include <random>
namespace {
    using namespace lfs::core;
    class TensorSplatAffine : public testing::TestWithParam<int> {
    protected:
        std::optional<GpuBackendScope> scope;
        Device device = Device::CPU;
        void SetUp() override {
            if (GetParam() == 0)
                return;
            auto backend = GetParam() == 1 ? GpuBackend::CUDA : GpuBackend::Vulkan;
            if (!gpu_backend_available(backend))
                GTEST_SKIP();
            scope.emplace(backend);
            device = Device::GPU;
        }
    };
    TEST_P(TensorSplatAffine, MatchesSharedMathForShearReflectionAndThinAxes) {
        for (size_t n : {0u, 1u, 31u, 32u, 33u, 255u, 256u, 257u}) {
            std::mt19937 rng(917);
            std::uniform_real_distribution<float> v(-1, 1);
            std::vector<float> s(n * 3), q(n * 4);
            for (auto& x : s)
                x = v(rng) * 12;
            for (auto& x : q)
                x = v(rng);
            auto hs = Tensor::from_vector(s, {n, 3}, Device::CPU), hq = Tensor::from_vector(q, {n, 4}, Device::CPU);
            for (const auto linear : {splat_transform::LinearTransform{{2, 0.7f, 0, 0, 0.6f, -0.4f, 0, 0, 1.4f}},
                                      splat_transform::LinearTransform{{-1, 0, 0, 0, 2, 0, 0, 0, 0.4f}}}) {
                SCOPED_TRACE(n);
                auto rs = Tensor::empty({n, 3}, Device::CPU), rq = Tensor::empty({n, 4}, Device::CPU);
                affine_splat_geometry(linear, hs, hq, rs, rq);
                auto ds = hs.to(device), dq = hq.to(device);
                affine_splat_geometry(linear, ds, dq, ds, dq);
                auto actual_s = ds.cpu(), actual_q = dq.cpu();
                for (size_t i = 0; i < s.size(); ++i) {
                    if (std::isinf(rs.ptr<float>()[i]))
                        EXPECT_EQ(actual_s.ptr<float>()[i], rs.ptr<float>()[i]);
                    else
                        EXPECT_NEAR(actual_s.ptr<float>()[i], rs.ptr<float>()[i], 2e-6f);
                }
                for (size_t i = 0; i < q.size(); ++i)
                    EXPECT_NEAR(actual_q.ptr<float>()[i], rq.ptr<float>()[i], 2e-6f);
            }
        }
    }
    TEST_P(TensorSplatAffine, ZeroScalesAndQuaternionRemainDefined) {
        auto scales = Tensor::full({33, 3}, -INFINITY, device), rotations = Tensor::zeros({33, 4}, device);
        affine_splat_geometry({{2, 0, 0, 0, 1, 0, 0, 0, 3}}, scales, rotations, scales, rotations);
        for (auto v : scales.to_vector())
            EXPECT_EQ(v, -INFINITY);
        const auto q = rotations.to_vector();
        for (size_t i = 0; i < q.size(); ++i)
            EXPECT_EQ(q[i], i % 4 == 0 ? 1 : 0);
    }
    TEST_P(TensorSplatAffine, AdversarialCovarianceMatchesFp64Reference) {
        constexpr size_t n = 513;
        std::mt19937 rng(1827);
        std::uniform_real_distribution<float> value(-1, 1);
        std::vector<float> scales(3 * n), quaternions(4 * n);
        for (size_t i = 0; i < n; ++i) {
            const float common = float(int(i % 5) - 2) * 40;
            for (int j = 0; j < 3; ++j)
                scales[3 * i + j] = common + value(rng) * 12;
            if (i % 4 == 0)
                scales[3 * i + 1] = scales[3 * i + 2] = common;
            if (i % 7 == 0)
                scales[3 * i + 1] = scales[3 * i + 2] = -INFINITY;
            for (int j = 0; j < 4; ++j)
                quaternions[4 * i + j] = value(rng);
            if (i % 11 == 0) {
                quaternions[4 * i] = quaternions[4 * i + 2] = std::sqrt(0.5f);
                quaternions[4 * i + 1] = quaternions[4 * i + 3] = 0;
            }
            if (i % 13 == 0)
                quaternions[4 * i] = 1e-7f;
        }
        const auto covariance = [](const float* s, const float* q, double shift) {
            const auto r = glm::mat3_cast(glm::normalize(glm::dquat(q[0], q[1], q[2], q[3])));
            glm::dmat3 d(0.0);
            for (int i = 0; i < 3; ++i)
                d[i][i] = std::exp(2 * (double(s[i]) - shift));
            return r * d * glm::transpose(r);
        };
        for (float factor : {1e-20f, 1.0f, 1e20f}) {
            for (auto linear : {splat_transform::LinearTransform{{2, .7f, 0, 0, .6f, -.4f, 0, 0, 1.4f}},
                                splat_transform::LinearTransform{{-1, 0, 0, 0, 2, 0, 0, 0, .4f}},
                                splat_transform::LinearTransform{{1, 1, 0, 0, 1e-7f, 0, 0, 0, 1e-12f}}}) {
                for (auto& x : linear.rows)
                    x *= factor;
                glm::dmat3 a(0.0);
                for (int row = 0; row < 3; ++row)
                    for (int col = 0; col < 3; ++col)
                        a[col][row] = double(linear.rows[3 * row + col]) / factor;
                auto ds = Tensor::from_vector(scales, {n, 3}, device);
                auto dq = Tensor::from_vector(quaternions, {n, 4}, device);
                affine_splat_geometry(linear, ds, dq, ds, dq);
                const auto result_s = ds.to_vector(), result_q = dq.to_vector();
                for (size_t i = 0; i < n; ++i) {
                    const double common = std::max({scales[3 * i], scales[3 * i + 1], scales[3 * i + 2]});
                    const auto input = covariance(scales.data() + 3 * i, quaternions.data() + 4 * i, common);
                    const auto expected = a * input * glm::transpose(a);
                    const auto actual = covariance(result_s.data() + 3 * i, result_q.data() + 4 * i, common + std::log(double(factor)));
                    double error = 0, anorm = 0, cnorm = 0;
                    for (int col = 0; col < 3; ++col)
                        for (int row = 0; row < 3; ++row) {
                            error += std::pow(actual[col][row] - expected[col][row], 2);
                            anorm += a[col][row] * a[col][row];
                            cnorm += input[col][row] * input[col][row];
                        }
                    // Near-null transformed covariances need an input-scaled backward-error bound.
                    EXPECT_LE(std::sqrt(error) / (anorm * std::sqrt(cnorm)), 4e-5)
                        << "splat=" << i << " factor=" << factor;
                }
            }
        }
    }
    INSTANTIATE_TEST_SUITE_P(CpuCudaVulkan, TensorSplatAffine, testing::Values(0, 1, 2),
                             [](const testing::TestParamInfo<int>& p) { return p.param == 0 ? "CPU" : p.param == 1 ? "CUDA"
                                                                                                                   : "Vulkan"; });
} // namespace
