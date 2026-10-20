/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "cuda_backend_test.hpp"
#include "lfs/training/ops/registry.hpp"
#include "training/kernels/regularization.cuh"

#include <gtest/gtest.h>
#include <vector>

namespace {
    using namespace lfs::core;
    namespace ops = lfs::gpu_ops;
    using ExtraLossOpsTest = lfs::test::CudaBackendTest;

    std::vector<uint8_t> bytes(const Tensor& t) {
        const auto cpu = t.cpu().contiguous();
        const auto* p = static_cast<const uint8_t*>(cpu.data_ptr());
        return {p, p + cpu.bytes()};
    }

    TEST_F(ExtraLossOpsTest, RegularizeMatchesDirectLaunchers) {
        constexpr size_t n = 1027;
        const auto& table = *lfs::training::training_ops(GpuBackend::CUDA).extra_loss;
        for (const auto kind : {ops::Regularizer::Scale, ops::Regularizer::Opacity}) {
            const size_t attrs = kind == ops::Regularizer::Scale ? 3 : 1;
            std::vector<float> values(n * attrs);
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = float(int((i * 79 + 31) % 257) - 128) / 64.f;
            }
            const auto raw = Tensor::from_vector(values, {n, attrs}, Device::GPU);
            for (const bool backward : {false, true}) {
                auto actual = backward ? Tensor::full({n, attrs}, 0.17f, Device::GPU) : Tensor{};
                auto expected = backward ? actual.clone() : Tensor{};
                auto loss = Tensor::full({1}, -9.f, Device::GPU);
                auto direct_loss = loss.clone();
                auto temp = Tensor::zeros({1024}, Device::GPU);
                auto direct_temp = temp.clone();
                table.regularize(raw, actual, loss, temp, kind, 0.013f);
                const auto launcher = kind == ops::Regularizer::Scale
                                          ? lfs::training::kernels::launch_fused_scale_regularization
                                          : lfs::training::kernels::launch_fused_opacity_regularization;
                launcher(raw.ptr<float>(), backward ? expected.ptr<float>() : nullptr,
                         direct_loss.ptr<float>(), direct_temp.ptr<float>(), raw.numel(), 0.013f, nullptr);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                if (backward) {
                    EXPECT_EQ(bytes(actual), bytes(expected));
                }
                EXPECT_EQ(bytes(loss), bytes(direct_loss));
                EXPECT_EQ(bytes(temp), bytes(direct_temp));
            }
        }
    }

} // namespace
