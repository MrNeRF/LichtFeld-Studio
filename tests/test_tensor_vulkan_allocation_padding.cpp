/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/memory_pressure.hpp"
#include "core/tensor.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_memory.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_upload.hpp"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <vector>
#include <vulkan/vulkan.h>

namespace {
    using namespace lfs::core;

    constexpr size_t kDirectLimitBytes = 16ull * 1024ull * 1024ull;

    class TensorVulkanAllocationPadding : public testing::Test {
    protected:
        void SetUp() override {
            ASSERT_TRUE(gpu_backend_available(GpuBackend::Vulkan));
        }

        void TearDown() override {
            const auto status = shutdown_gpu_backend(GpuBackend::Vulkan);
            EXPECT_TRUE(status.has_value());
            EXPECT_EQ(internal::vulkan_live_vma_objects_for_testing(), 0u);
            for (const std::string& message :
                 internal::vulkan_validation_messages_for_testing()) {
                ADD_FAILURE() << message;
            }
        }
    };

    void expect_logical_bytes_unchanged(const Tensor& tensor, const size_t logical) {
        EXPECT_EQ(tensor.bytes(), logical);
        const internal::StorageRef storage = internal::storage_ref(tensor);
        ASSERT_NE(storage.meta, nullptr);
        EXPECT_EQ(storage.meta->gpu_descriptor.byte_size, logical);
        EXPECT_EQ(storage.byte_offset, 0u);

        const auto queried = tensor_vulkan_buffer(tensor);
        ASSERT_TRUE(queried.has_value());
        EXPECT_EQ(queried->bytes, logical);
        EXPECT_EQ(queried->offset, 0u);
        ASSERT_NE(queried->buffer, nullptr);
    }

    // vkCmdCopyBuffer of the last covering uint32; API validation flags a
    // VkBuffer whose create size is the unpadded logical tail.
    void copy_covering_word(const Tensor& tensor, const size_t logical) {
        std::array<std::byte, 4> word{};
        internal::StorageRef src = internal::storage_ref(tensor);
        src.byte_offset = logical & ~size_t{3};
        internal::backend_ops(GpuBackend::Vulkan)
            .copy_device_to_host(internal::CopyRequest{
                .src = src,
                .dst = internal::raw_storage_ref(word.data()),
                .bytes = sizeof(word),
                .synchronous = true,
            });
    }

    TEST_F(TensorVulkanAllocationPadding, TrimDiscardsPendingFreeBuffersAfterCompletion) {
#ifdef __APPLE__
        GTEST_SKIP() << "MoltenVK must wait for device residency before freeing storage";
#else
        GpuBackendScope scope(GpuBackend::Vulkan);
        auto& ops = internal::backend_ops(GpuBackend::Vulkan);
        ops.synchronize_device();
        ops.trim();
        const auto context = internal::acquire_vulkan_context();
        const auto device = context->device();
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        info.pNext = &type;
        VkSemaphore gate = VK_NULL_HANDLE;
        ASSERT_EQ(vkCreateSemaphore(device, &info, nullptr, &gate), VK_SUCCESS);
        {
            TensorWorkQueue queue(GpuBackend::Vulkan);
            queue.set_consumer_timeline(device, {gate, 1, {}});
            queue.wait_timeline(1);
            {
                TensorWorkQueue::Scope binding(queue);
                Tensor pending = Tensor::zeros({size_t{8} << 20}, Device::GPU, DataType::UInt8);
            }
            EXPECT_FALSE(queue.ready());
            auto trimmed = std::async(std::launch::async, [&] { ops.trim(); });
            const auto status = trimmed.wait_for(std::chrono::seconds(1));
            VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            signal.semaphore = gate;
            signal.value = 1;
            EXPECT_EQ(vkSignalSemaphore(device, &signal), VK_SUCCESS);
            EXPECT_EQ(status, std::future_status::ready);
            trimmed.get();
            queue.wait();
            // An allocation collects completed retirements without trimming again.
            const Tensor probe = Tensor::empty({1}, Device::GPU, DataType::UInt8);
            EXPECT_EQ(context->memory().cached_bytes(), 0u);
        }
        vkDestroySemaphore(device, gate, nullptr);
#endif
    }

    TEST_F(TensorVulkanAllocationPadding,
           LargeStorageDoesNotReserveAnOversizedBlock) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        const auto context = internal::acquire_vulkan_context();
        for (const size_t mib : {17u, 65u}) {
            const size_t bytes = mib << 20;
            const Tensor tensor = Tensor::empty({bytes}, Device::GPU, DataType::UInt8);
            const auto storage = internal::storage_ref(tensor);
            VmaAllocationInfo2 allocation{};
            vmaGetAllocationInfo2(context->allocator(),
                                  reinterpret_cast<VmaAllocation>(static_cast<uintptr_t>(
                                      storage.meta->gpu_descriptor.native_allocation)),
                                  &allocation);
            EXPECT_LE(allocation.blockSize, 2 * bytes)
                << "One large tensor should not reserve several times its storage";
        }
    }

    TEST_F(TensorVulkanAllocationPadding,
           DirectRangeOddByteBufferCoversLastWordAndKeepsLogicalSize) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        auto& ops = internal::backend_ops(GpuBackend::Vulkan);
        ops.synchronize_device();
        ops.trim();
        const uint64_t live_before = internal::vulkan_live_vma_objects_for_testing();

        {
            const Tensor slab = Tensor::empty({255}, Device::GPU, DataType::UInt8);
            expect_logical_bytes_unchanged(slab, 255);
        }
        EXPECT_GT(internal::vulkan_live_vma_objects_for_testing(), live_before);
        ops.synchronize_device();
        ops.trim();
        const uint64_t live_baseline = internal::vulkan_live_vma_objects_for_testing();

        for (const size_t extra : {size_t{0}, size_t{1}, size_t{2}, size_t{3}}) {
            SCOPED_TRACE(extra);
            const size_t logical = kDirectLimitBytes + extra;
            Tensor tensor = Tensor::empty({logical}, Device::GPU, DataType::UInt8);
            expect_logical_bytes_unchanged(tensor, logical);
            if (extra != 0) {
                copy_covering_word(tensor, logical);
            }
            ops.synchronize_device();
            tensor = Tensor();
            EXPECT_LE(internal::vulkan_live_vma_objects_for_testing(),
                      live_baseline + 256ull * 1024ull * 1024ull / kDirectLimitBytes);
            ops.trim();
            EXPECT_EQ(internal::vulkan_live_vma_objects_for_testing(), live_baseline);
        }
    }

    TEST_F(TensorVulkanAllocationPadding, DirectRangeOddByteConvertRoundtrip) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        for (const size_t extra : {size_t{1}, size_t{2}, size_t{3}}) {
            SCOPED_TRACE(extra);
            const size_t logical = kDirectLimitBytes + extra;
            const Tensor source =
                Tensor::full({logical}, 165.0f, Device::GPU, DataType::UInt8);
            expect_logical_bytes_unchanged(source, logical);
            copy_covering_word(source, logical);

            const Tensor as_bool = source.to(DataType::Bool);
            expect_logical_bytes_unchanged(as_bool, logical);
            copy_covering_word(as_bool, logical);

            const Tensor roundtrip = as_bool.to(DataType::UInt8);
            EXPECT_EQ(roundtrip.bytes(), logical);
            const Tensor roundtrip_head = roundtrip.slice(0, 0, 8).contiguous().cpu();
            const Tensor roundtrip_tail =
                roundtrip.slice(0, logical - 8, logical).contiguous().cpu();
            EXPECT_EQ(roundtrip_head.to_vector_uint8(), std::vector<uint8_t>(8, 1));
            EXPECT_EQ(roundtrip_tail.to_vector_uint8(), std::vector<uint8_t>(8, 1));
        }
    }

    // MoltenVK makes all device memory resident for every command buffer it
    // submits, so freeing a buffer also waits for unrelated work submitted
    // while the buffer existed.
    TEST_F(TensorVulkanAllocationPadding, DirectFreeWaitsForWorkSubmittedWhileItLived) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        // Large enough for dedicated memory, which returns to the driver when freed.
        Tensor direct = Tensor::full({size_t{40} << 20}, 1.0f, Device::GPU);
        EXPECT_FLOAT_EQ(direct.slice(0, 0, 4).sum().item<float>(), 4.0f);
        const Tensor lhs = Tensor::ones({2048, 2048}, Device::GPU);
        const Tensor product = lhs.mm(lhs);
        // Submits the product without waiting for it.
        ASSERT_TRUE(tensor_vulkan_buffer(product).has_value());
        direct = Tensor();
        EXPECT_FLOAT_EQ(product.slice(0, 0, 1).slice(1, 0, 1).cpu().item<float>(), 2048.0f);
    }

    TEST_F(TensorVulkanAllocationPadding,
           DirectRangeWordPaddingOverflowThrowsTypedError) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        auto& ops = internal::backend_ops(GpuBackend::Vulkan);
        // Initialize the backend before counting its staging allocations.
        const Tensor warmup = Tensor::empty({1}, Device::GPU, DataType::UInt8);
        const uint64_t live = internal::vulkan_live_vma_objects_for_testing();
        const size_t too_big = std::numeric_limits<size_t>::max() - 1;
        try {
            const internal::StorageRef storage = ops.allocate(too_big, 16, {});
            ops.deallocate(storage, {});
            FAIL() << "expected MemoryAllocationError for word-padding overflow";
        } catch (const MemoryAllocationError& error) {
            EXPECT_EQ(error.domain(), MemoryDomain::VulkanDevice);
            EXPECT_EQ(error.requested_bytes(), too_big);
            EXPECT_EQ(error.failure().alignment, 16u);
            EXPECT_EQ(error.failure().native_error,
                      static_cast<long long>(VK_ERROR_OUT_OF_DEVICE_MEMORY));
        }
        EXPECT_EQ(internal::vulkan_live_vma_objects_for_testing(), live);
    }

} // namespace
