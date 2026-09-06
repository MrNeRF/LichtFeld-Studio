/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

    using namespace lfs::core;

    TEST(TensorBackendSelftest, CudaSucceeds) {
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

} // namespace
