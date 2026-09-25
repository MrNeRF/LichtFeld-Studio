/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/gpu_device_runtime.hpp"
#include "core/gpu_elapsed.hpp"
#include "core/pinned_allocator_stats.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#if LFS_TEST_TENSOR_VULKAN
#include "core/tensor_completion.hpp"
#endif

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <span>
#include <thread>
#include <utility>
#if LFS_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace {
    void verify_barrier(const lfs::core::GpuBackend backend) {
        using namespace lfs::core;
        if (!gpu_backend_available(backend))
            GTEST_SKIP() << "Backend unavailable";
        const GpuBackendScope scope(backend);
        const Tensor values = Tensor::full({32}, 4.0f, Device::GPU);
        gpu_device_barrier(backend);
        for (const float value : values.to_vector())
            EXPECT_EQ(value, 4.0f);
    }
} // namespace

TEST(TensorDeviceRuntime, CudaSlicePrefixCopy) {
    using namespace lfs::core;
    if (!gpu_backend_available(GpuBackend::CUDA))
        GTEST_SKIP() << "Backend unavailable";
    const GpuBackendScope scope(GpuBackend::CUDA);
    Tensor source = Tensor::from_vector(std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}, {8}, Device::GPU).to(DataType::UInt8);
    Tensor destination = Tensor::full({12}, 255, Device::GPU, DataType::UInt8);
    destination.slice(0, 4, 8).copy_(source.slice(0, 2, 6));
    EXPECT_EQ(destination.to_vector_uint8(), (std::vector<uint8_t>{255, 255, 255, 255, 2, 3, 4, 5, 255, 255, 255, 255}));
}

TEST(TensorDeviceRuntime, VulkanSlicePrefixCopy) {
    using namespace lfs::core;
    if (!gpu_backend_available(GpuBackend::Vulkan))
        GTEST_SKIP() << "Backend unavailable";
    const GpuBackendScope scope(GpuBackend::Vulkan);
    Tensor source = Tensor::from_vector(std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}, {8}, Device::GPU).to(DataType::UInt8);
    Tensor destination = Tensor::full({12}, 255, Device::GPU, DataType::UInt8);
    destination.slice(0, 4, 8).copy_(source.slice(0, 2, 6));
    EXPECT_EQ(destination.to_vector_uint8(), (std::vector<uint8_t>{255, 255, 255, 255, 2, 3, 4, 5, 255, 255, 255, 255}));
}

TEST(TensorDeviceRuntime, CudaDeviceBarrier) {
    verify_barrier(lfs::core::GpuBackend::CUDA);
}

TEST(TensorDeviceRuntime, CudaDeviceBarrierWaitsForQueuedHostWork) {
#if LFS_HAS_CUDA
    using namespace lfs::core;
    if (!gpu_backend_available(GpuBackend::CUDA))
        GTEST_SKIP() << "Backend unavailable";
    const GpuBackendScope scope(GpuBackend::CUDA);
    std::atomic<bool> release{false};
    std::atomic<bool> completed{false};
    std::pair<std::atomic<bool>*, std::atomic<bool>*> state{&release, &completed};
    ASSERT_EQ(cudaLaunchHostFunc(nullptr, [](void* context) {
        auto* state = static_cast<std::pair<std::atomic<bool>*, std::atomic<bool>*>*>(context);
        while (!state->first->load(std::memory_order_acquire))
            std::this_thread::yield();
        state->second->store(true, std::memory_order_release); }, &state), cudaSuccess);
    std::thread releaser([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        release.store(true, std::memory_order_release);
    });
    gpu_device_barrier(GpuBackend::CUDA);
    const bool completed_at_barrier_return = completed.load(std::memory_order_acquire);
    releaser.join();
    EXPECT_TRUE(completed_at_barrier_return);
#else
    GTEST_SKIP() << "CUDA support is not built";
#endif
}

TEST(TensorDeviceRuntime, VulkanDeviceBarrier) {
    verify_barrier(lfs::core::GpuBackend::Vulkan);
}

TEST(TensorDeviceRuntime, VulkanDeviceBarrierWaitsForRecordedWork) {
#if LFS_TEST_TENSOR_VULKAN
    using namespace lfs::core;
    if (!gpu_backend_available(GpuBackend::Vulkan))
        GTEST_SKIP() << "Backend unavailable";
    const GpuBackendScope scope(GpuBackend::Vulkan);
    const Tensor values = Tensor::full({8 * 1024 * 1024}, 4.0f, Device::GPU);
    const Tensor* operands[] = {&values};
    const TensorCompletion completion{std::span<const Tensor* const>(operands)};
    EXPECT_FALSE(completion.ready());
    gpu_device_barrier(GpuBackend::Vulkan);
    EXPECT_TRUE(completion.ready());
#else
    GTEST_SKIP() << "Vulkan support is not built";
#endif
}

TEST(TensorDeviceRuntime, CudaDeviceCount) {
    using namespace lfs::core;
    if (!gpu_backend_available(GpuBackend::CUDA))
        GTEST_SKIP() << "Backend unavailable";
    EXPECT_GT(gpu_device_count(GpuBackend::CUDA), 0);
}

TEST(TensorDeviceRuntime, VulkanDeviceCount) {
    if (!lfs::core::gpu_backend_available(lfs::core::GpuBackend::Vulkan))
        GTEST_SKIP() << "Backend unavailable";
    EXPECT_GT(lfs::core::gpu_device_count(lfs::core::GpuBackend::Vulkan), 0);
}

TEST(TensorDeviceRuntime, CudaMemoryStats) {
    if (!lfs::core::gpu_backend_available(lfs::core::GpuBackend::CUDA))
        GTEST_SKIP() << "Backend unavailable";
    const auto stats = lfs::core::gpu_backend_memory_info(lfs::core::GpuBackend::CUDA);
    EXPECT_GT(stats.total_bytes, 0u);
    EXPECT_LE(stats.free_bytes, stats.total_bytes);
}

TEST(TensorDeviceRuntime, VulkanMemoryStats) {
    using namespace lfs::core;
    if (!gpu_backend_available(GpuBackend::Vulkan))
        GTEST_SKIP() << "Backend unavailable";
    const auto stats = gpu_backend_memory_info(GpuBackend::Vulkan);
    EXPECT_GT(stats.total_bytes, 0u);
    EXPECT_LE(stats.free_bytes, stats.total_bytes);
    EXPECT_EQ(stats.pool_used_current, 0u);
}

TEST(TensorDeviceRuntime, ReservedAllocationBytesUnsupportedWithoutTensor) {
    using namespace lfs::core;
    EXPECT_FALSE(reserved_allocation_bytes(Tensor{}));
}

TEST(TensorDeviceRuntime, CudaReservedAllocationSizeUnknown) {
    using namespace lfs::core;
    if (!gpu_backend_available(GpuBackend::CUDA))
        GTEST_SKIP() << "Backend unavailable";
    const GpuBackendScope scope(GpuBackend::CUDA);
    const Tensor tensor = Tensor::empty({17}, Device::GPU);
    EXPECT_FALSE(reserved_allocation_bytes(tensor));
}

TEST(TensorDeviceRuntime, VulkanReservedAllocationBytesUnsupported) {
    using namespace lfs::core;
    const Tensor host = Tensor::empty({17}, Device::CPU);
    EXPECT_FALSE(reserved_allocation_bytes(host));
}

TEST(TensorDeviceRuntime, CudaPinnedAllocatorStats) {
#if !LFS_HAS_CUDA
    GTEST_SKIP() << "CUDA support is not built";
#else
    using namespace lfs::core;
    const auto before = pinned_allocator_stats();
    {
        const Tensor pinned = Tensor::empty({1 << 20}, Device::CPU, DataType::UInt8);
        EXPECT_EQ(pinned.bytes(), 1u << 20);
        EXPECT_GT(pinned_allocator_stats().allocated_bytes, before.allocated_bytes);
    }
    EXPECT_GE(pinned_allocator_stats().cached_bytes, before.cached_bytes);
#endif
}

TEST(TensorDeviceRuntime, VulkanPinnedAllocatorStats) {
#if !LFS_HAS_CUDA
    GTEST_SKIP() << "CUDA support is not built";
#else
    using namespace lfs::core;
    const auto before = pinned_allocator_stats();
    {
        const Tensor pinned = Tensor::empty({1 << 20}, Device::CPU, DataType::UInt8);
        EXPECT_GT(pinned_allocator_stats().allocated_bytes, before.allocated_bytes);
    }
    EXPECT_GE(pinned_allocator_stats().cached_bytes, before.cached_bytes);
#endif
}

TEST(TensorDeviceRuntime, CudaElapsedGpuTime) {
    using namespace lfs::core;
    if (!gpu_backend_available(GpuBackend::CUDA))
        GTEST_SKIP() << "Backend unavailable";
    const GpuBackendScope scope(GpuBackend::CUDA);
    GpuElapsed timer(GpuBackend::CUDA, 2);
    ASSERT_TRUE(timer.ready());
    ASSERT_TRUE(timer.mark(0, nullptr));
    const Tensor values = Tensor::full({32}, 4.0f, Device::GPU);
    static_cast<void>(values);
    ASSERT_TRUE(timer.mark(1, nullptr));
    ASSERT_TRUE(timer.wait_event(1));
    const auto elapsed = timer.milliseconds(0, 1);
    ASSERT_TRUE(elapsed);
    EXPECT_GE(*elapsed, 0.0f);
}

TEST(TensorDeviceRuntime, VulkanElapsedGpuTimeUnsupported) {
    using namespace lfs::core;
    GpuElapsed timer(GpuBackend::Vulkan, 2);
    EXPECT_FALSE(timer.ready());
    EXPECT_FALSE(timer.mark(0, nullptr));
    EXPECT_FALSE(timer.wait_queue(nullptr));
    EXPECT_FALSE(timer.wait_queue(reinterpret_cast<void*>(1)));
    EXPECT_FALSE(timer.milliseconds(0, 1));
}
