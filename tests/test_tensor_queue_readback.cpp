/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/gpu_device_runtime.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_readback.hpp"
#include "core/tensor_upload.hpp"
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <span>
#include <vector>
#if LFS_HAS_CUDA
#include "core/alloc_counter.hpp"
#include "core/headless_vulkan_device.hpp"
#include "cuda_stream_gate.hpp"
#include <chrono>
#include <future>
#endif

namespace {
    using namespace lfs::core;

    TEST(TensorQueueContract, UnavailableConstructorsRejectExplicitly) {
        EXPECT_THROW((void)TensorWorkQueue(GpuBackend::Vulkan), std::runtime_error);
        EXPECT_THROW(TensorWorkQueue(GpuBackend::Vulkan, nullptr), std::runtime_error);
        EXPECT_THROW((void)TensorFence(GpuBackend::Vulkan), std::runtime_error);
        if (!gpu_backend_available(GpuBackend::CUDA)) {
            EXPECT_THROW((void)TensorWorkQueue(GpuBackend::CUDA), std::runtime_error);
#if !LFS_HAS_CUDA
            EXPECT_THROW(TensorWorkQueue(GpuBackend::CUDA, nullptr), std::runtime_error);
#else
            // A borrowed adapter can exist without touching the runtime. Its
            // first GPU operation must still report an unavailable device.
            TensorWorkQueue borrowed(GpuBackend::CUDA, nullptr);
            EXPECT_THROW(borrowed.wait(), std::runtime_error);
            EXPECT_THROW((void)borrowed.ready(), std::runtime_error);
#endif
            EXPECT_THROW(TensorWorkQueue(GpuBackend::CUDA, nullptr, nullptr), std::runtime_error);
            EXPECT_THROW((void)TensorFence(GpuBackend::CUDA), std::runtime_error);
        }
    }

    TEST(TensorQueueContract, BackendOwnedVulkanReadbackAndUnsupportedOperations) {
        if (!gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP();
        GpuBackendScope scope(GpuBackend::Vulkan);
        TensorWorkQueue queue(GpuBackend::Vulkan, nullptr, nullptr);
        Tensor source = Tensor::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, Device::GPU);
        TensorReadback readback;
        readback.enqueue(source.transpose(0, 1), queue);
        std::array<float, 6> output{};
        readback.wait(std::as_writable_bytes(std::span(output)));
        EXPECT_EQ(output, (std::array<float, 6>{1, 4, 2, 5, 3, 6}));
        EXPECT_THROW(queue.ready(), std::runtime_error);
        EXPECT_THROW(queue.wait(), std::runtime_error);
        EXPECT_THROW(queue.enqueue_host_callback([](void*) {}, nullptr), std::runtime_error);
        EXPECT_THROW(queue.set_consumer_timeline(nullptr, {}), std::runtime_error);
        EXPECT_THROW(queue.wait_timeline(1), std::runtime_error);
        EXPECT_THROW(TensorReadbackRing(GpuBackend::Vulkan, 1, 16, queue), std::runtime_error);
#if !LFS_HAS_CUDA
        EXPECT_THROW(TensorReadbackRing(GpuBackend::CUDA, 1, 16, queue), std::invalid_argument);
#endif
        Tensor destination = Tensor::empty({6}, Device::CPU);
        EXPECT_THROW(readback.prepare(source, destination), std::runtime_error);
        EXPECT_FALSE(reserved_allocation_bytes(source).has_value());
    }

#if LFS_HAS_CUDA
    using namespace std::chrono_literals;
    TEST(TensorQueueReadback, BlockedProducerOrdersExplicitAndPackedDownloads) {
        if (!gpu_backend_available(GpuBackend::CUDA))
            GTEST_SKIP();
        GpuBackendScope backend(GpuBackend::CUDA);
        for (bool default_stream : {false, true})
            for (bool strided : {false, true}) {
                SCOPED_TRACE(default_stream);
                SCOPED_TRACE(strided);
                TensorWorkQueue owned(GpuBackend::CUDA);
                TensorWorkQueue producer(GpuBackend::CUDA, default_stream ? nullptr : owned.native_handle());
                TensorWorkQueue consumer(GpuBackend::CUDA);
                TensorWorkQueue::Scope producer_scope(producer);
                Tensor source = Tensor::full({2, 3}, 1.f, Device::GPU);
                Tensor view = strided ? source.transpose(0, 1) : source;
                TensorReadback readback;
                TensorReadbackRing ring(GpuBackend::CUDA, 1, 24, consumer);
                std::array<float, 6> output{};
                readback.enqueue(view, consumer);
                readback.wait(std::as_writable_bytes(std::span(output)));
                ring.enqueue(view, 0, 24, 0, 0);
                ring.seal(0);
                ring.wait(0);
                ring.release(0);
                producer.wait();
                lfs::test::CudaStreamGate gate;
                ASSERT_EQ(gate.block(static_cast<cudaStream_t>(producer.native_handle())), cudaSuccess);
                ASSERT_TRUE(gate.entered());
                source.fill_(7.f, static_cast<cudaStream_t>(producer.native_handle()));
                auto submitted = std::async(std::launch::async, [&] {
                    GpuBackendScope scope(GpuBackend::CUDA);
                    readback.enqueue(view, consumer);
                    ring.enqueue(view, 0, 24, 0, 0);
                    ring.seal(0);
                });
                const auto status = submitted.wait_for(1s);
                EXPECT_FALSE(gate.released());
                if (status == std::future_status::ready) {
                    EXPECT_FALSE(readback.poll(std::as_writable_bytes(std::span(output))));
                    EXPECT_FALSE(ring.poll(0));
                }
                gate.release();
                EXPECT_EQ(status, std::future_status::ready);
                submitted.get();
                readback.wait(std::as_writable_bytes(std::span(output)));
                EXPECT_EQ(output, (std::array<float, 6>{7, 7, 7, 7, 7, 7}));
                ring.wait(0);
                std::memcpy(output.data(), ring.slot_bytes(0).data(), 24);
                EXPECT_EQ(output, (std::array<float, 6>{7, 7, 7, 7, 7, 7}));
                ring.release(0);
            }
    }

    TEST(TensorQueueReadback, SmallAndEmptyRingReadsPreserveLazySnapshotStorage) {
        if (!gpu_backend_available(GpuBackend::CUDA))
            GTEST_SKIP();
        GpuBackendScope backend(GpuBackend::CUDA);
        TensorWorkQueue queue(GpuBackend::CUDA);
        TensorWorkQueue::Scope scope(queue);
        const Tensor source = Tensor::full({4 * 1024 * 1024}, 9.f, Device::GPU);
        auto deferred = TensorLeaf(source).snapshot();
        TensorReadbackRing ring(GpuBackend::CUDA, 1, 4, queue);
        queue.wait();
        Tensor::trim_device_memory_pool();
        const auto before = alloc_counter::snapshot();
        const auto original = source.data_ptr();
        ring.enqueue(source, 0, 0, 0, 0);
        ring.enqueue(source, 0, 4, 0, 0);
        ring.seal(0);
        ring.wait(0);
        const Tensor retained = deferred.eval();
        EXPECT_EQ(retained.data_ptr(), original);
        EXPECT_EQ(alloc_counter::delta_since(before), 0u);
        float value = 0;
        std::memcpy(&value, ring.slot_bytes(0).data(), 4);
        EXPECT_EQ(value, 9.f);
    }

    TEST(TensorQueueReadback, SuppliedStagingDoesNotAllocateAnotherDeviceBand) {
        if (!gpu_backend_available(GpuBackend::CUDA))
            GTEST_SKIP();
        GpuBackendScope backend(GpuBackend::CUDA);
        TensorWorkQueue queue(GpuBackend::CUDA);
        TensorWorkQueue::Scope scope(queue);
        const Tensor source = Tensor::full({1024 * 1024}, 3.f, Device::GPU);
        const Tensor scratch = Tensor::empty({4 * 1024 * 1024}, Device::GPU, DataType::UInt8);
        TensorReadbackRing ring(GpuBackend::CUDA, 1, scratch.bytes(), queue, &scratch);
        queue.wait();
        Tensor::trim_device_memory_pool();
        const auto before = alloc_counter::snapshot();
        ring.enqueue(source, 0, source.bytes(), 0, 0, true);
        ring.seal(0);
        ring.wait(0);
        EXPECT_EQ(alloc_counter::delta_since(before), 0u);
        float first = 0, last = 0;
        std::memcpy(&first, ring.slot_bytes(0).data(), sizeof(float));
        std::memcpy(&last, ring.slot_bytes(0).data() + source.bytes() - sizeof(float), sizeof(float));
        EXPECT_EQ(first, 3.f);
        EXPECT_EQ(last, 3.f);
        const Tensor wrong = Tensor::empty({4}, Device::CPU, DataType::UInt8);
        EXPECT_THROW(TensorReadbackRing(GpuBackend::CUDA, 1, 4, queue, &wrong), std::invalid_argument);
        EXPECT_THROW(TensorReadbackRing(GpuBackend::CUDA, 1, scratch.bytes() + 1, queue, &scratch), std::invalid_argument);

        // Staging is borrowed: the owner may allocate it after construction and
        // release it between captures.
        Tensor lent;
        TensorReadbackRing borrowing(GpuBackend::CUDA, 1, source.bytes(), queue, &lent);
        for (const bool allocated : {true, false}) {
            lent = allocated ? Tensor::empty({source.bytes()}, Device::GPU, DataType::UInt8) : Tensor();
            borrowing.enqueue(source, 0, source.bytes(), 0, 0, true);
            borrowing.seal(0);
            borrowing.wait(0);
            std::memcpy(&last, borrowing.slot_bytes(0).data() + source.bytes() - sizeof(float), sizeof(float));
            EXPECT_EQ(last, 3.f);
            borrowing.release(0);
        }
    }

    TEST(TensorQueueReadback, PreparedDestinationRetainsStorageAndReusesCompletion) {
        if (!gpu_backend_available(GpuBackend::CUDA))
            GTEST_SKIP();
        GpuBackendScope backend(GpuBackend::CUDA);
        TensorWorkQueue producer(GpuBackend::CUDA), consumer(GpuBackend::CUDA);
        TensorWorkQueue::Scope scope(producer);
        Tensor source = Tensor::full({32}, 2.f, Device::GPU);
        Tensor destination = Tensor::empty({32}, Device::CPU);
        TensorReadback readback;
        const Tensor unpinned = Tensor::empty_unpinned({32});
        EXPECT_THROW(readback.prepare(source, unpinned), std::invalid_argument);
        readback.prepare(source, destination);
        producer.wait();
        for (int i = 0; i < 3; ++i) {
            lfs::test::CudaStreamGate gate;
            ASSERT_EQ(gate.block(static_cast<cudaStream_t>(producer.native_handle())), cudaSuccess);
            ASSERT_TRUE(gate.entered());
            source.fill_(float(i + 3), static_cast<cudaStream_t>(producer.native_handle()));
            auto submitted = std::async(std::launch::async, [&] { readback.enqueue(consumer); });
            const auto status = submitted.wait_for(1s);
            EXPECT_FALSE(gate.released());
            if (status == std::future_status::ready)
                EXPECT_FALSE(readback.poll());
            gate.release();
            EXPECT_EQ(status, std::future_status::ready);
            submitted.get();
            readback.wait();
            EXPECT_FALSE(readback.pending());
            EXPECT_EQ(destination.to_vector(), std::vector<float>(32, float(i + 3)));
        }
        readback.enqueue(consumer);
        source = {};
        readback.wait();
        EXPECT_EQ(destination.to_vector(), std::vector<float>(32, 5.f));
    }

    TEST(TensorQueueReadback, BorrowedTimelineImportsSurviveBlockedWaitAndReplacement) {
        if (!gpu_backend_available(GpuBackend::CUDA))
            GTEST_SKIP();
        auto device_owner = HeadlessAdoptedDevice::try_create(true);
        if (!device_owner)
            GTEST_SKIP() << "Timeline exports unavailable";
        const auto device = static_cast<VkDevice>(device_owner->handles().device);
        for (bool default_stream : {false, true}) {
            SCOPED_TRACE(default_stream);
            TensorWorkQueue owner(GpuBackend::CUDA);
            auto borrowed = std::make_unique<TensorWorkQueue>(GpuBackend::CUDA,
                                                              default_stream ? nullptr : owner.native_handle());
            VkExportSemaphoreCreateInfo export_info{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
#ifdef _WIN32
            export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
            export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
            VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            type.pNext = &export_info;
            VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            info.pNext = &type;
            VkSemaphore semaphore = VK_NULL_HANDLE;
            ASSERT_EQ(vkCreateSemaphore(device, &info, nullptr, &semaphore), VK_SUCCESS);
            auto token = std::make_shared<int>(1);
            std::weak_ptr<int> weak = token;
            borrowed->set_consumer_timeline(device, {semaphore, 1, token});
            token.reset();
            borrowed->wait_timeline(1);
            borrowed->set_consumer_timeline(device, {});
            auto destroyed = std::async(std::launch::async, [&] { borrowed.reset(); });
            const auto status = destroyed.wait_for(20ms);
            EXPECT_EQ(status, std::future_status::timeout);
            EXPECT_FALSE(weak.expired());
            VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            signal.semaphore = semaphore;
            signal.value = 1;
            EXPECT_EQ(vkSignalSemaphore(device, &signal), VK_SUCCESS);
            destroyed.get();
            EXPECT_TRUE(weak.expired());
            owner.wait();
            vkDestroySemaphore(device, semaphore, nullptr);
        }
    }
#endif
} // namespace
