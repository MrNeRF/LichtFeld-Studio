/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_execution.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

    using namespace lfs::core;

    TEST(TensorBackendSelftest, CudaSucceeds) {
        if (!gpu_backend_available(GpuBackend::CUDA)) {
            GTEST_SKIP() << "CUDA backend unavailable";
        }
        const lfs::Status status = tensor_backend_selftest(GpuBackend::CUDA);
        ASSERT_TRUE(status.has_value()) << (status ? "" : std::string(status.error().user_message()));
    }

    TEST(TensorBackendSelftest, VulkanSucceedsWhenAvailableAndLeavesBackendUsable) {
        if (!gpu_backend_available(GpuBackend::Vulkan)) {
            GTEST_SKIP() << "Vulkan backend unavailable";
        }

        const lfs::Status status = tensor_backend_selftest(GpuBackend::Vulkan);
        ASSERT_TRUE(status.has_value()) << (status ? "" : std::string(status.error().user_message()));

        GpuBackendScope scope(GpuBackend::Vulkan);
        const Tensor input = Tensor::from_vector(std::vector<float>{1.0f, 4.0f, 9.0f},
                                                 {3}, Device::GPU);
        const Tensor rooted = input.sqrt();
        EXPECT_EQ(gpu_backend_of(rooted), GpuBackend::Vulkan);
        EXPECT_EQ(rooted.to_vector(), (std::vector<float>{1.0f, 2.0f, 3.0f}));
    }

    class HipTensorBackendSelftest : public testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!gpu_backend_available(GetParam())) {
                GTEST_SKIP() << "HIP backend unavailable";
            }
        }
    };

    TEST_P(HipTensorBackendSelftest, SucceedsAndLeavesBackendUsable) {
        const lfs::Status status = tensor_backend_selftest(GetParam());
        ASSERT_TRUE(status.has_value()) << (status ? "" : std::string(status.error().user_message()));

        GpuBackendScope scope(GetParam());
        const Tensor input = Tensor::from_vector(std::vector<float>{1.0f, 4.0f, 9.0f},
                                                 {3}, Device::GPU);
        const Tensor rooted = input.sqrt();
        EXPECT_EQ(gpu_backend_of(rooted), GetParam());
        EXPECT_EQ(rooted.to_vector(), (std::vector<float>{1.0f, 2.0f, 3.0f}));
    }

    TEST_P(HipTensorBackendSelftest, RepeatedSelftestsReleaseTemporaryAllocations) {
        const auto warmup = tensor_backend_selftest(GetParam());
        ASSERT_TRUE(warmup) << (warmup ? "" : std::string(warmup.error().user_message()));
        TensorExecutionTarget::default_queue(GetParam()).wait();
        const auto warm = gpu_backend_memory_info(GetParam());
        ASSERT_GT(warm.total_bytes, 0u);
        for (int i = 0; i < 3; ++i) {
            SCOPED_TRACE(i);
            const auto status = tensor_backend_selftest(GetParam());
            ASSERT_TRUE(status) << (status ? "" : std::string(status.error().user_message()));
            TensorExecutionTarget::default_queue(GetParam()).wait();
            const auto after = gpu_backend_memory_info(GetParam());
            EXPECT_LE(after.allocated_bytes, warm.allocated_bytes);
            RecordProperty("live_bytes_after_repeat_" + std::to_string(i), std::to_string(after.allocated_bytes));
        }
        RecordProperty("live_bytes_after_warmup", std::to_string(warm.allocated_bytes));
    }

    INSTANTIATE_TEST_SUITE_P(Backends, HipTensorBackendSelftest,
                             testing::Values(GpuBackend::HIP),
                             [](const testing::TestParamInfo<GpuBackend>& info) {
                                 return std::string(gpu_backend_name(info.param));
                             });

} // namespace
