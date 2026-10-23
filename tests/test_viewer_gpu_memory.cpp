/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/gpu_device_info.hpp"
#include "core/tensor_backend.hpp"
#include "gui/gpu_memory_query.hpp"
#include "gui/hip_memory_status.hpp"

#include <gtest/gtest.h>

using namespace lfs::core;

#ifdef LFS_TENSOR_HIP
TEST(ViewerGpuMemory, HipStatusUsesOnlyItsDeviceMatchedNativeSample) {
    GpuDeviceInfo device;
    device.name = "Selected HIP adapter";
    device.total_memory_bytes = 8ull << 30;
    device.supports_process_memory_budget = true;
    device.process_memory_budget_bytes = 6ull << 30;
    device.process_memory_used_bytes = 2ull << 30;
    const auto status = lfs::vis::gui::selectHipGpuMemory(&device);
    EXPECT_EQ(status.device_name, device.name);
    EXPECT_EQ(status.total, device.total_memory_bytes);
    EXPECT_TRUE(status.uses_process_budget);
    EXPECT_TRUE(status.native_process_budget);
    EXPECT_FALSE(status.native_metrics_unavailable);
    EXPECT_EQ(status.process_budget, device.process_memory_budget_bytes);
    EXPECT_EQ(status.process_used, device.process_memory_used_bytes);
    EXPECT_TRUE(status.process_valid);
    EXPECT_FALSE(status.process_over_budget);
    EXPECT_EQ(status.total_used, 0u);
    EXPECT_FALSE(status.gpu_utilization_valid);
    EXPECT_STREQ(lfs::vis::gui::gpuDeviceMemoryTooltipKey(status), "ui.vram_hip_budget_tooltip");
    device.process_memory_used_bytes = device.process_memory_budget_bytes + 1;
    EXPECT_TRUE(lfs::vis::gui::selectHipGpuMemory(&device).process_over_budget);
}

TEST(ViewerGpuMemory, HipMissingTelemetryRemainsUnknown) {
    GpuDeviceInfo device;
    device.name = "Selected HIP adapter";
    device.total_memory_bytes = 8ull << 30;
    // Stale values must not become valid without the native validity flag.
    device.process_memory_budget_bytes = 6ull << 30;
    device.process_memory_used_bytes = 2ull << 30;
    for (const auto* sample : {&device, static_cast<GpuDeviceInfo*>(nullptr)}) {
        const auto status = lfs::vis::gui::selectHipGpuMemory(sample);
        EXPECT_TRUE(status.native_metrics_unavailable);
        EXPECT_FALSE(status.native_process_budget);
        EXPECT_FALSE(status.process_valid);
        EXPECT_EQ(status.process_budget, 0u);
        EXPECT_EQ(status.process_used, 0u);
        EXPECT_FALSE(status.gpu_utilization_valid);
        EXPECT_STREQ(lfs::vis::gui::gpuProcessMemoryTooltipKey(status), "ui.vram_native_unavailable_tooltip");
    }
}

#endif

TEST(ViewerGpuMemory, VulkanStatusDoesNotRequireCudaOrReportDeviceUsageAsProcessUsage) {
#ifdef __APPLE__
    GTEST_SKIP() << "Apple GPUs report unified memory";
#endif
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

#ifdef __APPLE__
TEST(ViewerGpuMemory, AppleStatusUsesAvailableTelemetryOrBackendBudget) {
    const auto backend = default_gpu_backend();
    if (!gpu_backend_available(backend)) {
        GTEST_SKIP() << "GPU backend unavailable";
    }
    const GpuBackendScope scope(backend);
    const auto resident = Tensor::zeros({1 << 20}, Device::GPU);
    const auto device = gpu_backend_device_info(backend);
    ASSERT_TRUE(device);
    const auto status = lfs::vis::gui::queryGpuMemory(backend);
    EXPECT_FALSE(status.device_name.empty());
    if (!status.unified_memory) {
        // Virtualized macOS can expose Metal/MoltenVK without IOKit accelerator statistics.
        EXPECT_TRUE(status.uses_process_budget);
        EXPECT_EQ(status.total, device->total_memory_bytes);
        EXPECT_EQ(status.process_used, 0u);
        EXPECT_FALSE(status.process_valid);
        EXPECT_EQ(status.total_used, 0u);
        EXPECT_FALSE(status.gpu_utilization_valid);
        EXPECT_LT(status.gpu_utilization_percent, 0.0f);
        EXPECT_EQ(status.process_budget, device->supports_process_memory_budget ? device->process_memory_budget_bytes : 0u);
        EXPECT_EQ(status.process_budget_used, device->supports_process_memory_budget ? device->process_memory_used_bytes : 0u);
        return;
    }
    EXPECT_FALSE(status.uses_process_budget);
    EXPECT_EQ(status.total, device->process_memory_budget_bytes);
    EXPECT_GT(status.process_used, 0u);
    EXPECT_GE(status.total_used, status.process_used);
    EXPECT_LE(status.total_used, status.total);
    ASSERT_TRUE(status.gpu_utilization_valid);
    EXPECT_GE(status.gpu_utilization_percent, 0.0f);
    EXPECT_LE(status.gpu_utilization_percent, 100.0f);
    EXPECT_GE(lfs::vis::gui::queryGpuUtilization(), 0.0f);
}
#endif
