/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/gpu_device_info.hpp"
#include "core/tensor_backend.hpp"
#include "gui/gpu_memory_query.hpp"

#include <gtest/gtest.h>

using namespace lfs::core;

TEST(ViewerGpuMemory, VulkanStatusDoesNotRequireCudaOrReportDeviceUsageAsProcessUsage) {
    if (!gpu_backend_available(GpuBackend::Vulkan)) {
        GTEST_SKIP() << "Vulkan backend unavailable";
    }
    const GpuBackendScope scope(GpuBackend::Vulkan);
    const auto resident = Tensor::zeros({1024}, Device::GPU);
    const auto device = gpu_backend_device_info(GpuBackend::Vulkan);
    ASSERT_TRUE(device);
    ASSERT_FALSE(device->name.empty());
    ASSERT_GT(device->total_memory_bytes, 0u);
    const auto status = lfs::vis::gui::queryGpuMemory(GpuBackend::Vulkan);
    EXPECT_FALSE(status.device_name.empty());
    EXPECT_EQ(status.total, device->total_memory_bytes);
    EXPECT_TRUE(status.uses_process_budget);
    EXPECT_EQ(status.total_used, 0u);
    EXPECT_EQ(status.process_used, 0u);
    EXPECT_FALSE(status.gpu_utilization_valid);
    EXPECT_LT(status.gpu_utilization_percent, 0.0f);
    if (device->supports_process_memory_budget) {
        EXPECT_GT(status.process_budget, 0u);
        EXPECT_GT(status.process_budget_used, 0u);
        EXPECT_LE(status.process_budget, status.total);
    } else {
        EXPECT_EQ(status.process_budget, 0u);
        EXPECT_EQ(status.process_budget_used, 0u);
    }
}

TEST(ViewerGpuMemory, CudaStatusKeepsDeviceUsageAndItsOptionalUtilizationReading) {
    if (!gpu_backend_available(GpuBackend::CUDA)) {
        GTEST_SKIP() << "CUDA backend unavailable";
    }
    const auto status = lfs::vis::gui::queryGpuMemory(GpuBackend::CUDA);
    EXPECT_FALSE(status.device_name.empty());
    EXPECT_GT(status.total, 0u);
    EXPECT_LE(status.total_used, status.total);
    EXPECT_FALSE(status.uses_process_budget);
    EXPECT_EQ(status.process_budget, 0u);
    EXPECT_EQ(status.process_budget_used, 0u);
    if (status.gpu_utilization_valid) {
        EXPECT_GE(status.gpu_utilization_percent, 0.0f);
        EXPECT_LE(status.gpu_utilization_percent, 100.0f);
    }
}

TEST(ViewerGpuMemory, UnavailableCudaProducesUnknownStatusWithoutThrowing) {
    if (gpu_backend_available(GpuBackend::CUDA)) {
        GTEST_SKIP() << "Requires CUDA unavailable";
    }
    const auto status = lfs::vis::gui::queryGpuMemory(GpuBackend::CUDA);
    EXPECT_TRUE(status.device_name.empty());
    EXPECT_EQ(status.total, 0u);
    EXPECT_EQ(status.total_used, 0u);
    EXPECT_FALSE(status.gpu_utilization_valid);
}
