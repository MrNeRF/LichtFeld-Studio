/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// MCMC's relocation draw past a 32-bit int midpoint: mcmc_sample searches the inclusive cumulative sum of the
// weights for each draw, and a midpoint taken as (left + right) / 2 in int overflows once left + right passes
// INT_MAX, which a search over more than 2^30 categories reaches when its target lies near the top. Here 2^30 + 2
// categories have weight 0 except the last two, so every draw must select one of those two, and every search climbs
// with right at the last category: in int, left + right passes INT_MAX from left = 2^30 - 1 on.
//
// Runs with LFS_TEST_LARGE_SH=1: about 25.8 GB of device memory (weights, their cumulative sum, and the opacity and
// scales every category row needs).

#include "large_model_test_support.hpp"

#include "core/tensor_backend.hpp"
#include "lfs/training/ops/mcmc.hpp"
#include "lfs/training/ops/registry.hpp"

#include "cuda_backend_test.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    namespace lm = lfs::test::large_model;
    namespace ops = lfs::gpu_ops;

    constexpr size_t kCategories = (size_t{1} << 30) + 2;
    constexpr size_t kSamples = 4096;
    // The two categories with weight: the second-to-last (weight 1) and the last (weight 3).
    constexpr size_t kFirstTarget = kCategories - 2, kLastTarget = kCategories - 1;
    static_assert(kFirstTarget + kLastTarget > size_t{std::numeric_limits<int>::max()});

    Tensor host_floats(const std::vector<float>& values) {
        Tensor host = Tensor::empty({values.size()}, Device::CPU, DataType::Float32);
        std::memcpy(host.data_ptr(), values.data(), values.size() * sizeof(float));
        return host;
    }

    class LargeModelMcmc : public ::testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (const auto reason = lm::skip_reason(kCategories * 6 * sizeof(float)))
                GTEST_SKIP() << *reason;
            if (!lfs::core::gpu_backend_available(GetParam()))
                GTEST_SKIP() << lfs::core::gpu_backend_name(GetParam()) << " device unavailable";
            if (lfs::training::training_ops(GetParam()).mcmc == nullptr)
                GTEST_SKIP() << "MCMC slot is empty";
            session_.emplace(GetParam());
            ASSERT_TRUE(session_->switched());
        }

        void TearDown() override { session_.reset(); }

        std::optional<lfs::test::DefaultGpuBackendForTesting> session_;
    };

    TEST_P(LargeModelMcmc, SampleSearchesPastTheIntMidpoint) {
        const ops::McmcOps& mcmc = *lfs::training::training_ops(GetParam()).mcmc;
        // Weights 0 except 1 and 3 at the two targets; the targets' opacities and log scales are their own.
        Tensor weights = Tensor::zeros({kCategories}, Device::GPU);
        weights.slice(0, kFirstTarget, kCategories).copy_(host_floats({1.0f, 3.0f}).gpu());
        Tensor opacity = Tensor::zeros({kCategories}, Device::GPU);
        opacity.slice(0, kFirstTarget, kCategories).copy_(host_floats({0.25f, 0.75f}).gpu());
        Tensor scales = Tensor::zeros({kCategories, 3}, Device::GPU);
        scales.slice(0, kFirstTarget, kCategories)
            .copy_(host_floats({std::log(2.0f), 0.0f, std::log(4.0f), std::log(8.0f), 0.0f, std::log(0.5f)})
                       .reshape({2, 3})
                       .gpu());

        Tensor indices = Tensor::empty({kSamples}, Device::GPU, DataType::Int64);
        Tensor sampled_opacity = Tensor::empty({kSamples}, Device::GPU);
        Tensor sampled_scales = Tensor::empty({kSamples, 3}, Device::GPU);
        mcmc.sample(weights, opacity, scales, Tensor{}, indices, sampled_opacity, sampled_scales,
                    ops::SampleDomain::All, 4202);

        const auto drawn = lm::host_copy<int64_t>(indices);
        const auto drawn_opacity = lm::host_copy<float>(sampled_opacity);
        const auto drawn_scales = lm::host_copy<float>(sampled_scales);
        ASSERT_EQ(drawn.size(), kSamples);
        size_t elsewhere = 0, first = 0, wrong_rows = 0;
        std::string first_elsewhere, first_wrong_row;
        const float exp_scales[2][3] = {{2.0f, 1.0f, 4.0f}, {8.0f, 1.0f, 0.5f}};
        for (size_t i = 0; i < kSamples; ++i) {
            const auto index = drawn[i];
            if (index != static_cast<int64_t>(kFirstTarget) && index != static_cast<int64_t>(kLastTarget)) {
                if (elsewhere++ == 0)
                    first_elsewhere = std::format("draw {} selected category {}", i, index);
                continue;
            }
            const size_t k = static_cast<size_t>(index) - kFirstTarget;
            first += k == 0;
            bool right = drawn_opacity[i] == (k == 0 ? 0.25f : 0.75f);
            for (size_t d = 0; d < 3; ++d)
                right = right && std::fabs(drawn_scales[i * 3 + d] - exp_scales[k][d]) <= 1e-5f * exp_scales[k][d];
            if (!right && wrong_rows++ == 0)
                first_wrong_row = std::format("draw {} (category {}): opacity {}, scales {} {} {}", i, index,
                                              drawn_opacity[i], drawn_scales[i * 3], drawn_scales[i * 3 + 1],
                                              drawn_scales[i * 3 + 2]);
        }
        EXPECT_EQ(elsewhere, 0u) << elsewhere << " of " << kSamples
                                 << " draws selected a category of weight 0; first: " << first_elsewhere;
        EXPECT_EQ(wrong_rows, 0u) << wrong_rows << " draws gathered another row's opacity or scales; first: "
                                  << first_wrong_row;
        // A quarter of the weight sits on the first target.
        EXPECT_NEAR(static_cast<double>(first) / kSamples, 0.25, 0.04) << first << " of " << kSamples;
    }

    INSTANTIATE_TEST_SUITE_P(Backends, LargeModelMcmc,
                             testing::Values(GpuBackend::CUDA, GpuBackend::Metal, GpuBackend::Vulkan),
                             [](const auto& info) { return std::string(lfs::core::gpu_backend_name(info.param)); });

} // namespace
