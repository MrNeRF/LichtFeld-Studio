/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/gpu_preflight.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "rendering/selection_ops.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {

    using namespace lfs::core;
    using lfs::app::decide_gpu_preflight;
    using lfs::app::GpuPreflightDecision;

} // namespace

TEST(ViewerNoCuda, PreflightViewerOnlyCudaUsableUsesCuda) {
    EXPECT_EQ(decide_gpu_preflight(true, true, true), GpuPreflightDecision::UseCuda);
    EXPECT_EQ(decide_gpu_preflight(true, true, false), GpuPreflightDecision::UseCuda);
}

TEST(ViewerNoCuda, PreflightViewerOnlyCudaUnusableUsesVulkanWhenAvailable) {
    EXPECT_EQ(decide_gpu_preflight(true, false, true), GpuPreflightDecision::UseVulkanViewer);
}

TEST(ViewerNoCuda, PreflightTrainingCudaUsableUsesCuda) {
    EXPECT_EQ(decide_gpu_preflight(false, true, true), GpuPreflightDecision::UseCuda);
}

TEST(ViewerNoCuda, PreflightTrainingCudaUnusableIsFatal) {
    EXPECT_EQ(decide_gpu_preflight(false, false, true), GpuPreflightDecision::Fatal);
    EXPECT_EQ(decide_gpu_preflight(true, false, false), GpuPreflightDecision::Fatal);
}

TEST(ViewerNoCuda, HoverGroupCountTensorProgramMatchesCpu) {
    if (!gpu_backend_available(GpuBackend::Vulkan)) {
        GTEST_SKIP() << "Vulkan backend unavailable";
    }

    GpuBackendScope scope(GpuBackend::Vulkan);
    constexpr size_t n = 4096;
    std::mt19937 rng(20260906);
    std::uniform_int_distribution<int> dist(0, 7);
    std::vector<std::uint8_t> host(n);
    std::array<size_t, 256> cpu_counts{};
    for (size_t i = 0; i < n; ++i) {
        host[i] = static_cast<std::uint8_t>(dist(rng));
        if (host[i] != 0) {
            ++cpu_counts[host[i]];
        }
    }

    Tensor cpu_mask = Tensor::empty({n}, Device::CPU, DataType::UInt8);
    std::memcpy(cpu_mask.ptr<std::uint8_t>(), host.data(), n);
    Tensor mask = cpu_mask.to(Device::CUDA);
    ASSERT_TRUE(mask.is_valid());
    EXPECT_EQ(gpu_backend_of(mask), GpuBackend::Vulkan);

    Tensor scratch;
    lfs::rendering::count_selection_groups_async(mask, scratch);
    const auto gpu_counts = lfs::rendering::read_selection_group_counts(scratch);
    for (size_t group = 0; group < 256; ++group) {
        EXPECT_EQ(gpu_counts[group], cpu_counts[group]) << "group " << group;
    }

    std::array<int, 257> host_counts{};
    lfs::rendering::enqueue_selection_group_count_read(scratch, host_counts.data(), nullptr);
    for (size_t group = 0; group < 256; ++group) {
        EXPECT_EQ(static_cast<size_t>(host_counts[group]), cpu_counts[group]) << "group " << group;
    }
}
