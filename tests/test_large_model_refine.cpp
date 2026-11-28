/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The densification split past a 32-bit int: refine_split addressed its source Gaussian's rotation at src * 4 and
// its means, scales and colour at src * 3 in int, which overflows from source row 536,870,912 (rotations) and
// 715,827,883 (the rest). A model of 536,871,040 Gaussians at SH degree 0 splits three sources: row 5, row
// 536,870,913 and the last row. The same three sources alone, in a model of three rows, are the reference: the
// children and the parents' new values must match bit for bit.
//
// Runs with LFS_TEST_HUGE_SH=1: about 30 GB of device memory.

#include "large_model_test_support.hpp"

#include "core/tensor_backend.hpp"
#include "lfs/training/ops/refine.hpp"
#include "lfs/training/ops/registry.hpp"

#include "cuda_backend_test.hpp"
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
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

    constexpr size_t kRows = 536'871'040;
    constexpr std::array<size_t, 3> kSources{5, 536'870'913, kRows - 1};
    static_assert(kSources[1] * 4 > 0x7fffffffull);

    struct Splats {
        Tensor means, rotations, scales, sh0, opacity;
    };

    Tensor host_tensor(const std::vector<float>& values, const lfs::core::TensorShape& shape) {
        Tensor host = Tensor::empty(shape, Device::CPU, DataType::Float32);
        std::memcpy(host.data_ptr(), values.data(), values.size() * sizeof(float));
        return host;
    }

    // `rows` Gaussians; source k of kSources sits at rows_of[k] with its own values.
    Splats make_splats(const size_t rows, const std::array<size_t, 3>& rows_of) {
        Splats s{Tensor::zeros({rows, 3}, Device::GPU), Tensor::full({rows, 4}, 0.5f, Device::GPU),
                 Tensor::full({rows, 3}, -2.0f, Device::GPU), Tensor::zeros({rows, 3}, Device::GPU),
                 Tensor::zeros({rows}, Device::GPU)};
        for (size_t k = 0; k < rows_of.size(); ++k) {
            const size_t r = rows_of[k];
            const auto f = static_cast<float>(k + 1);
            s.means.slice(0, r, r + 1).copy_(host_tensor({f, -f, 0.5f * f}, {1, 3}).gpu());
            s.rotations.slice(0, r, r + 1).copy_(host_tensor({std::cos(0.2f * f), std::sin(0.2f * f), 0.1f, 0.0f}, {1, 4}).gpu());
            s.scales.slice(0, r, r + 1).copy_(host_tensor({-2.0f + 0.1f * f, -3.0f, -2.5f + 0.05f * f}, {1, 3}).gpu());
            s.sh0.slice(0, r, r + 1).copy_(host_tensor({0.1f * f, 0.2f, -0.3f * f}, {1, 3}).gpu());
            s.opacity.slice(0, r, r + 1).copy_(host_tensor({0.3f * f}, {1}).gpu());
        }
        return s;
    }

    class LargeModelRefine : public ::testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (const auto reason = lm::skip_reason(kRows * 14 * sizeof(float)))
                GTEST_SKIP() << *reason;
            if (!lfs::core::gpu_backend_available(GetParam()))
                GTEST_SKIP() << lfs::core::gpu_backend_name(GetParam()) << " device unavailable";
            if (lfs::training::training_ops(GetParam()).refine == nullptr)
                GTEST_SKIP() << "Refine slot is empty";
            session_.emplace(GetParam());
            ASSERT_TRUE(session_->switched());
        }

        void TearDown() override { session_.reset(); }

        std::optional<lfs::test::DefaultGpuBackendForTesting> session_;
    };

    TEST_P(LargeModelRefine, SplitReadsSourcesPastTheIntWrap) {
        const ops::RefineOps& refine = *lfs::training::training_ops(GetParam()).refine;
        const auto split = [&](Splats& parents, const std::vector<size_t>& sources) {
            Splats children{Tensor::zeros({3, 3}, Device::GPU), Tensor::zeros({3, 4}, Device::GPU),
                            Tensor::zeros({3, 3}, Device::GPU), Tensor::zeros({3, 3}, Device::GPU),
                            Tensor::zeros({3}, Device::GPU)};
            const Tensor indices = lm::index_tensor<int64_t>(sources, Device::GPU);
            refine.split({parents.means, parents.rotations, parents.scales, parents.sh0, parents.opacity},
                         {children.means, children.rotations, children.scales, children.sh0, children.opacity},
                         indices);
            return children;
        };
        Splats small = make_splats(3, {0, 1, 2});
        const Splats small_children = split(small, {0, 1, 2});
        Splats big = make_splats(kRows, kSources);
        const Splats big_children = split(big, {kSources[0], kSources[1], kSources[2]});

        const auto same = [](const Tensor& a, const Tensor& b) {
            const auto x = lm::host_copy<uint32_t>(a), y = lm::host_copy<uint32_t>(b);
            return x == y;
        };
        EXPECT_TRUE(same(big_children.means, small_children.means)) << "children's means";
        EXPECT_TRUE(same(big_children.rotations, small_children.rotations)) << "children's rotations";
        EXPECT_TRUE(same(big_children.scales, small_children.scales)) << "children's scales";
        EXPECT_TRUE(same(big_children.sh0, small_children.sh0)) << "children's colours";
        EXPECT_TRUE(same(big_children.opacity, small_children.opacity)) << "children's opacities";
        for (size_t k = 0; k < kSources.size(); ++k) {
            const size_t r = kSources[k];
            const std::string row = "parent row " + std::to_string(r);
            EXPECT_TRUE(same(big.means.slice(0, r, r + 1), small.means.slice(0, k, k + 1))) << row << ", means";
            EXPECT_TRUE(same(big.scales.slice(0, r, r + 1), small.scales.slice(0, k, k + 1))) << row << ", scales";
            EXPECT_TRUE(same(big.opacity.slice(0, r, r + 1), small.opacity.slice(0, k, k + 1))) << row << ", opacity";
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, LargeModelRefine,
                             testing::Values(GpuBackend::CUDA, GpuBackend::Metal, GpuBackend::Vulkan),
                             [](const auto& info) { return std::string(lfs::core::gpu_backend_name(info.param)); });

} // namespace
