/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/gpu_device_runtime.hpp"
#include "core/gpu_elapsed.hpp"
#include "core/headless_vulkan_device.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_execution.hpp"
#include "core/tensor_readback.hpp"
#include "core/tensor_upload.hpp"
#include "core/tensor_vulkan_interop.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <span>
#include <type_traits>
#include <vector>
#include <vulkan/vulkan.h>
#if LFS_HAS_CUDA
#include "core/alloc_counter.hpp"
#include "cuda_stream_gate.hpp"
#include <chrono>
#include <future>
#endif

namespace {
    using namespace lfs::core;

    static_assert(std::is_trivially_copyable_v<TensorExecutionTarget>);

    TEST(TensorQueueContract, VulkanTargetWaitLeavesUnrelatedQueuePending) {
        if (!gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP();
        const GpuBackendScope backend(GpuBackend::Vulkan);
        for (const bool bound : {false, true}) {
            SCOPED_TRACE(bound);
            TensorWorkQueue producer(GpuBackend::Vulkan);
            TensorWorkQueue consumer(GpuBackend::Vulkan);
            Tensor work;
            {
                const TensorWorkQueue::Scope scope(producer);
                work = Tensor::full({1 << 16}, 1.f, Device::GPU).mul(2.f);
            }
            ASSERT_FALSE(producer.ready());
            if (bound) {
                const TensorWorkQueue::Scope scope(consumer);
                TensorExecutionTarget::current().wait();
            } else {
                TensorExecutionTarget(consumer).wait();
            }
            EXPECT_FALSE(producer.ready());
            producer.wait();
            EXPECT_FLOAT_EQ(work.cpu().to_vector().front(), 2.f);
        }
    }

    TEST(TensorQueueContract, VulkanElapsedTracksBoundQueue) {
        if (!gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP();
        const GpuBackendScope backend(GpuBackend::Vulkan);
        TensorWorkQueue queue(GpuBackend::Vulkan);
        GpuElapsed timer(GpuBackend::Vulkan, 2);
        if (!timer.ready())
            GTEST_SKIP();
        {
            const TensorWorkQueue::Scope scope(queue);
            EXPECT_TRUE(timer.mark(0, TensorExecutionTarget::current()));
            const Tensor work = Tensor::full({1 << 16}, 1.f, Device::GPU);
            EXPECT_TRUE(timer.mark(1, TensorExecutionTarget::current()));
            EXPECT_TRUE(timer.wait_queue(TensorExecutionTarget::current()));
            EXPECT_TRUE(timer.milliseconds(0, 1).has_value());
        }
        // Event ownership survives leaving the bound queue scope.
        EXPECT_TRUE(timer.wait_event(1));
        queue.wait();
    }

    TEST(TensorQueueContract, VulkanExecutionTargetUsesStorageTimeline) {
        if (!gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP();
        GpuBackendScope scope(GpuBackend::Vulkan);
        TensorWorkQueue queue(GpuBackend::Vulkan, nullptr, nullptr);
        const auto target = TensorExecutionTarget::current();
        EXPECT_EQ(target, TensorExecutionTarget(queue));
        const Tensor source = Tensor::from_vector(std::vector<float>{2, 4, 6, 8}, {2, 2}, Device::CPU);
        Tensor destination = Tensor::empty({2, 2}, Device::GPU);
        TensorUpload upload;
        upload.enqueue(destination, source, queue);
        destination.set_stream(queue);
        destination.sync_to_stream(target);
        destination.record_stream(target);
        EXPECT_EQ(destination.execution_target(), target);
        TensorReadback readback;
        readback.enqueue(destination.transpose(0, 1), target);
        std::array<float, 4> output{};
        readback.wait(std::as_writable_bytes(std::span(output)));
        EXPECT_EQ(output, (std::array<float, 4>{2, 6, 4, 8}));
        upload.wait();
        upload.enqueue(destination, std::as_bytes(std::span(output)), target);
        upload.wait();
        readback.enqueue_range(destination, sizeof(float), 2 * sizeof(float), target);
        std::array<float, 2> middle{};
        readback.wait(std::as_writable_bytes(std::span(middle)));
        EXPECT_EQ(middle, (std::array<float, 2>{6, 4}));
        target.wait();
        EXPECT_NO_THROW(target.wait_for(target));
        EXPECT_NO_THROW(target.set_name("test.queue"));
        EXPECT_NO_THROW(push_gpu_range("test.range"));
        EXPECT_NO_THROW(pop_gpu_range());
        GpuElapsed timer(GpuBackend::Vulkan, 2);
        if (timer.ready()) {
            EXPECT_TRUE(timer.mark(0, target));
            EXPECT_TRUE(timer.wait_event(0));
            EXPECT_TRUE(timer.wait_queue(target));
        } else {
            EXPECT_FALSE(timer.mark(0, target));
            EXPECT_FALSE(timer.wait_queue(target));
        }
        const auto other = TensorExecutionTarget::default_queue(GpuBackend::CUDA);
        EXPECT_THROW(destination.set_stream(other), std::invalid_argument);
        EXPECT_THROW(destination.sync_to_stream(other), std::invalid_argument);
        EXPECT_THROW(destination.record_stream(other), std::invalid_argument);
        EXPECT_THROW(upload.enqueue(destination, source, other), std::invalid_argument);
        EXPECT_THROW(readback.enqueue(destination, other), std::invalid_argument);
        EXPECT_THROW(timer.mark(0, other), std::invalid_argument);
    }

    TEST(TensorQueueContract, UnavailableConstructorsRejectExplicitly) {
        if (gpu_backend_available(GpuBackend::Vulkan)) {
            const GpuBackendScope scope(GpuBackend::Vulkan);
            TensorWorkQueue independent(GpuBackend::Vulkan);
            TensorWorkQueue legacy(GpuBackend::Vulkan, TensorWorkQueue::Mode::LegacyOrdered);
            TensorWorkQueue borrowed(GpuBackend::Vulkan, independent.native_handle());
            TensorWorkQueue implicit(GpuBackend::Vulkan, nullptr);
            TensorFence fence(GpuBackend::Vulkan);
            EXPECT_EQ(independent.backend(), GpuBackend::Vulkan);
            EXPECT_TRUE(fence.ready());
            EXPECT_TRUE(independent.ready());
            independent.record(fence);
            legacy.wait_for(fence);
            borrowed.wait_for(fence);
            implicit.record(fence);
            independent.wait();
            legacy.wait();
            borrowed.wait();
            implicit.wait();
            EXPECT_TRUE(fence.ready());
            EXPECT_TRUE(independent.ready());
            EXPECT_THROW(TensorWorkQueue(GpuBackend::Vulkan, reinterpret_cast<void*>(~uintptr_t{0})), std::runtime_error);
            std::atomic<bool> called{false};
            independent.enqueue_host_callback([](void* flag) {
                static_cast<std::atomic<bool>*>(flag)->store(true);
            },
                                              &called);
            independent.wait();
            EXPECT_TRUE(called.load());
        } else {
            EXPECT_THROW((void)TensorWorkQueue(GpuBackend::Vulkan), std::runtime_error);
            EXPECT_THROW(TensorWorkQueue(GpuBackend::Vulkan, nullptr), std::runtime_error);
            EXPECT_THROW((void)TensorFence(GpuBackend::Vulkan), std::runtime_error);
        }
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
        TensorWorkQueue copy(GpuBackend::Vulkan);
        readback.prepare(source, destination);
        readback.enqueue(copy);
        readback.wait();
        EXPECT_EQ(destination.to_vector(), (std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
        EXPECT_FALSE(reserved_allocation_bytes(source).has_value());
    }

    TEST(TensorQueueContract, VulkanQueueUploadReadbackAndFenceReuse) {
        if (!gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP();
        const GpuBackendScope scope(GpuBackend::Vulkan);
        TensorWorkQueue producer(GpuBackend::Vulkan, TensorWorkQueue::Mode::LegacyOrdered);
        TensorWorkQueue consumer(GpuBackend::Vulkan);
        TensorFence fence(GpuBackend::Vulkan);
        Tensor destination = Tensor::empty({4}, Device::GPU);
        TensorUpload upload;
        const std::array<float, 4> first{1.f, 2.f, 3.f, 4.f};
        upload.enqueue(destination, std::as_bytes(std::span(first)), TensorExecutionTarget(producer));
        producer.record(fence);
        consumer.wait_for(fence);
        TensorReadback readback;
        readback.enqueue(destination, consumer);
        std::array<float, 4> output{};
        readback.wait(std::as_writable_bytes(std::span(output)));
        EXPECT_EQ(output, first);
        upload.wait();
        EXPECT_TRUE(fence.ready());
        producer.wait();
        consumer.wait();
        EXPECT_TRUE(producer.ready());
        EXPECT_TRUE(consumer.ready());

        const std::array<float, 4> second{8.f, 7.f, 6.f, 5.f};
        upload.enqueue(destination, std::as_bytes(std::span(second)), TensorExecutionTarget(producer));
        producer.record(fence);
        consumer.wait_for(fence);
        readback.enqueue(destination, TensorExecutionTarget(consumer));
        readback.wait(std::as_writable_bytes(std::span(output)));
        EXPECT_EQ(output, second);
        upload.wait();
        consumer.wait();
        EXPECT_TRUE(fence.ready());

        TensorWorkQueue::Scope producer_scope(producer);
        const Tensor filled = Tensor::full({4}, 9.f, Device::GPU);
        producer.record(fence);
        consumer.wait_for(fence);
        consumer.wait();
        EXPECT_EQ(filled.to_vector(), std::vector<float>(4, 9.f));
        EXPECT_TRUE(fence.ready());
    }

    TEST(TensorQueueContract, VulkanHostPathsOrderCopiesCallbacksAndTimestamps) {
        if (!gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP();
        const GpuBackendScope scope(GpuBackend::Vulkan);
        TensorWorkQueue producer(GpuBackend::Vulkan, TensorWorkQueue::Mode::LegacyOrdered);
        TensorWorkQueue consumer(GpuBackend::Vulkan);
        Tensor device = Tensor::empty({4}, Device::GPU);
        const auto buffer = tensor_vulkan_buffer(device);
        ASSERT_TRUE(buffer.has_value());
        ASSERT_NE(buffer->device, nullptr);
        const auto vk_device = static_cast<VkDevice>(buffer->device);
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        info.pNext = &type;
        VkSemaphore semaphore = VK_NULL_HANDLE;
        ASSERT_EQ(vkCreateSemaphore(vk_device, &info, nullptr, &semaphore), VK_SUCCESS);
        auto token = std::make_shared<int>(1);
        std::weak_ptr<int> weak = token;
        producer.set_consumer_timeline(buffer->device, {semaphore, 1, token});
        token.reset();
        EXPECT_FALSE(weak.expired());
        producer.wait_timeline(1);

        TensorUpload upload;
        const std::array<float, 4> values{4.f, 5.f, 6.f, 7.f};
        upload.enqueue(device, std::as_bytes(std::span(values)), TensorExecutionTarget(producer));
        const auto pending = tensor_vulkan_buffer(device);
        ASSERT_TRUE(pending.has_value());
        ASSERT_GT(pending->pending_timeline_value, 0u);
        TensorFence adopted = TensorFence::adopt(
            GpuBackend::Vulkan, reinterpret_cast<void*>(static_cast<uintptr_t>(pending->pending_timeline_value)));
        EXPECT_FALSE(adopted.ready());
        TensorExecutionTarget(consumer).wait_for(TensorExecutionTarget(producer));

        TensorReadbackRing ring(GpuBackend::Vulkan, 2, 16, consumer);
        ring.enqueue(device, sizeof(float), 2 * sizeof(float), 0, 4, true);
        ring.enqueue(device, 0, sizeof(float), 1, 0);
        ring.seal(0);
        ring.seal(1);
        EXPECT_FALSE(ring.poll(0));
        EXPECT_FALSE(adopted.ready());

        std::atomic<bool> called{false};
        consumer.enqueue_host_callback([](void* flag) {
            static_cast<std::atomic<bool>*>(flag)->store(true);
        },
                                       &called);
        EXPECT_FALSE(called.load());
        EXPECT_FALSE(consumer.ready());

        VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        signal.semaphore = semaphore;
        signal.value = 1;
        ASSERT_EQ(vkSignalSemaphore(vk_device, &signal), VK_SUCCESS);

        adopted.wait();
        EXPECT_TRUE(adopted.ready());
        ring.wait(0);
        ring.wait(1);
        consumer.wait();
        EXPECT_TRUE(called.load());
        std::array<float, 4> slot{};
        std::memcpy(slot.data(), ring.slot_bytes(0).data(), sizeof(slot));
        EXPECT_EQ(slot[1], 5.f);
        EXPECT_EQ(slot[2], 6.f);
        float first = 0.f;
        std::memcpy(&first, ring.slot_bytes(1).data(), sizeof(first));
        EXPECT_EQ(first, 4.f);
        upload.wait();
        producer.wait();
        GpuElapsed timer(GpuBackend::Vulkan, 2);
        if (timer.ready()) {
            EXPECT_TRUE(timer.mark(0, TensorExecutionTarget(consumer)));
            {
                const TensorWorkQueue::Scope mark_scope(consumer);
                const Tensor work = Tensor::full({1 << 16}, 1.f, Device::GPU);
                (void)work;
            }
            EXPECT_TRUE(timer.mark(1, TensorExecutionTarget(consumer)));
            EXPECT_TRUE(timer.wait_event(1));
            const auto elapsed = timer.milliseconds(0, 1);
            ASSERT_TRUE(elapsed.has_value());
            EXPECT_GE(*elapsed, 0.f);
        }
        producer.set_consumer_timeline(buffer->device, {});
        vkDestroySemaphore(vk_device, semaphore, nullptr);
        EXPECT_FALSE(weak.expired());
    }

    TEST(TensorQueueContract, MetalQueuesFencesAndCallbacks) {
        if (!gpu_backend_available(GpuBackend::Metal)) {
            EXPECT_THROW((void)TensorWorkQueue(GpuBackend::Metal), std::runtime_error);
            EXPECT_THROW(TensorWorkQueue(GpuBackend::Metal, nullptr), std::runtime_error);
            EXPECT_THROW((void)TensorFence(GpuBackend::Metal), std::runtime_error);
            GTEST_SKIP();
        }
        const GpuBackendScope scope(GpuBackend::Metal);
        TensorWorkQueue independent(GpuBackend::Metal);
        TensorWorkQueue legacy(GpuBackend::Metal, TensorWorkQueue::Mode::LegacyOrdered);
        TensorWorkQueue borrowed(GpuBackend::Metal, independent.native_handle());
        TensorWorkQueue implicit(GpuBackend::Metal, nullptr);
        TensorFence fence(GpuBackend::Metal);
        EXPECT_EQ(independent.backend(), GpuBackend::Metal);
        EXPECT_NE(independent.native_handle(), nullptr);
        EXPECT_NE(independent.native_handle(), legacy.native_handle());
        EXPECT_EQ(borrowed.native_handle(), independent.native_handle());
        EXPECT_EQ(implicit.native_handle(), nullptr);
        EXPECT_TRUE(fence.ready());
        EXPECT_TRUE(independent.ready());
        independent.record(fence);
        legacy.wait_for(fence);
        borrowed.wait_for(fence);
        implicit.record(fence);
        independent.wait();
        legacy.wait();
        borrowed.wait();
        implicit.wait();
        EXPECT_TRUE(fence.ready());
        EXPECT_TRUE(independent.ready());
        EXPECT_THROW(TensorWorkQueue(GpuBackend::Metal, reinterpret_cast<void*>(~uintptr_t{0})), std::runtime_error);
        EXPECT_THROW(fence.record(reinterpret_cast<void*>(~uintptr_t{0})), std::invalid_argument);
        EXPECT_THROW(fence.record(TensorExecutionTarget::default_queue(GpuBackend::Vulkan)), std::invalid_argument);

        std::atomic<bool> called{false};
        {
            const TensorWorkQueue::Scope work(independent);
            Tensor busy = Tensor::full({1 << 20}, 1.f, Device::GPU);
            for (int i = 0; i < 8; ++i)
                busy = busy * 1.0001f + 0.5f;
            independent.enqueue_host_callback([](void* flag) {
                static_cast<std::atomic<bool>*>(flag)->store(true);
            },
                                              &called);
        }
        independent.wait();
        EXPECT_TRUE(called.load());
        EXPECT_TRUE(independent.ready());

        // A destroyed queue's handle is no longer a valid target.
        void* const stale = [] {
            const TensorWorkQueue temporary(GpuBackend::Metal);
            return temporary.native_handle();
        }();
        EXPECT_THROW(TensorWorkQueue(GpuBackend::Metal, stale), std::runtime_error);
    }

    TEST(TensorQueueContract, MetalExecutionTargetUploadReadbackAndTimestamps) {
        if (!gpu_backend_available(GpuBackend::Metal))
            GTEST_SKIP();
        const GpuBackendScope scope(GpuBackend::Metal);
        TensorWorkQueue producer(GpuBackend::Metal, TensorWorkQueue::Mode::LegacyOrdered);
        TensorWorkQueue consumer(GpuBackend::Metal);
        const TensorExecutionTarget target(consumer);
        const Tensor source = Tensor::from_vector(std::vector<float>{2, 4, 6, 8}, {2, 2}, Device::CPU);
        Tensor destination = Tensor::empty({2, 2}, Device::GPU);
        TensorUpload upload;
        upload.enqueue(destination, source, TensorExecutionTarget(producer));
        destination.set_stream(target);
        destination.sync_to_stream(target);
        destination.record_stream(target);
        TensorFence fence(GpuBackend::Metal);
        producer.record(fence);
        consumer.wait_for(fence);
        TensorReadback readback;
        readback.enqueue(destination.transpose(0, 1), target);
        std::array<float, 4> output{};
        readback.wait(std::as_writable_bytes(std::span(output)));
        EXPECT_EQ(output, (std::array<float, 4>{2, 6, 4, 8}));
        upload.wait();
        upload.enqueue(destination, std::as_bytes(std::span(output)), target);
        upload.wait();
        readback.enqueue_range(destination, sizeof(float), 2 * sizeof(float), target);
        std::array<float, 2> middle{};
        readback.wait(std::as_writable_bytes(std::span(middle)));
        EXPECT_EQ(middle, (std::array<float, 2>{6, 4}));

        const Tensor uploaded = source.to(Device::GPU, target);
        EXPECT_EQ(uploaded.to_vector(), (std::vector<float>{2, 4, 6, 8}));
        Tensor filled = Tensor::empty({3}, Device::GPU);
        filled.fill_(7.f, target);
        EXPECT_EQ(filled.to_vector(), std::vector<float>(3, 7.f));

        target.wait();
        EXPECT_NO_THROW(target.wait_for(TensorExecutionTarget(producer)));
        EXPECT_NO_THROW(target.set_name("test.queue"));
        EXPECT_THROW(target.wait_for(TensorExecutionTarget::default_queue(GpuBackend::Vulkan)), std::invalid_argument);

        Tensor prepared_source = Tensor::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6}, {6}, Device::GPU);
        Tensor prepared_destination = Tensor::empty({6}, Device::CPU);
        TensorReadback prepared;
        prepared.prepare(prepared_source, prepared_destination);
        prepared.enqueue(consumer);
        prepared.wait();
        EXPECT_EQ(prepared_destination.to_vector(), (std::vector<float>{1, 2, 3, 4, 5, 6}));
        {
            const TensorWorkQueue::Scope work(producer);
            prepared_source.fill_(3.f);
        }
        prepared.enqueue(consumer);
        while (!prepared.poll()) {
        }
        EXPECT_EQ(prepared_destination.to_vector(), std::vector<float>(6, 3.f));

        GpuElapsed timer(GpuBackend::Metal, 2);
        ASSERT_TRUE(timer.ready());
        EXPECT_TRUE(timer.mark(0, target));
        {
            const TensorWorkQueue::Scope work(consumer);
            Tensor busy = Tensor::full({1 << 22}, 1.f, Device::GPU);
            for (int i = 0; i < 16; ++i)
                busy = busy * 1.0001f + 0.5f;
        }
        EXPECT_TRUE(timer.mark(1, target));
        EXPECT_FALSE(timer.mark(2, target));
        EXPECT_TRUE(timer.wait_event(1));
        EXPECT_TRUE(timer.wait_queue(target));
        const auto elapsed = timer.milliseconds(0, 1);
        ASSERT_TRUE(elapsed.has_value());
        EXPECT_GT(*elapsed, 0.f);
        EXPECT_LT(*elapsed, 10'000.f);

        const auto other = TensorExecutionTarget::default_queue(GpuBackend::Vulkan);
        EXPECT_THROW(destination.set_stream(other), std::invalid_argument);
        EXPECT_THROW(upload.enqueue(destination, source, other), std::invalid_argument);
        EXPECT_THROW(readback.enqueue(destination, other), std::invalid_argument);
        EXPECT_THROW(timer.mark(0, other), std::invalid_argument);
        EXPECT_THROW((void)timer.mark(0, reinterpret_cast<void*>(~uintptr_t{0})), std::invalid_argument);
    }

    TEST(TensorQueueContract, MetalReadbackRingWaitsForConsumerTimeline) {
        if (!gpu_backend_available(GpuBackend::Metal) || !gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP();
        // A timeline semaphore of a MoltenVK device stands in for a viewer's
        // timeline that the Metal queue waits on.
        auto consumer = HeadlessAdoptedDevice::try_create();
        if (!consumer || !consumer->handles().metal_objects)
            GTEST_SKIP() << "No Vulkan device with VK_EXT_metal_objects";
        const auto vk_device = static_cast<VkDevice>(consumer->handles().device);
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        info.pNext = &type;
        VkSemaphore semaphore = VK_NULL_HANDLE;
        ASSERT_EQ(vkCreateSemaphore(vk_device, &info, nullptr, &semaphore), VK_SUCCESS);

        const GpuBackendScope scope(GpuBackend::Metal);
        Tensor device = Tensor::from_vector(std::vector<float>{4.f, 5.f, 6.f, 7.f}, {4}, Device::GPU);
        Tensor lent = Tensor::empty({16}, Device::GPU, DataType::UInt8);
        {
            TensorWorkQueue queue(GpuBackend::Metal);
            auto token = std::make_shared<int>(1);
            const std::weak_ptr<int> weak = token;
            queue.set_consumer_timeline(vk_device, {semaphore, 1, token});
            token.reset();
            EXPECT_FALSE(weak.expired());
            queue.wait_timeline(1);

            TensorReadbackRing ring(GpuBackend::Metal, 3, 16, queue, &lent);
            ring.enqueue(device, sizeof(float), 2 * sizeof(float), 0, 4, true);
            ring.enqueue(device, 0, sizeof(float), 1, 0);
            ring.enqueue(device, 0, device.bytes(), 2, 0);
            ring.seal(0);
            ring.seal(1);
            const TensorFence& last = ring.seal(2);
            std::atomic<bool> called{false};
            queue.enqueue_host_callback([](void* flag) {
                static_cast<std::atomic<bool>*>(flag)->store(true);
            },
                                        &called);
            EXPECT_FALSE(ring.poll(0));
            EXPECT_FALSE(last.ready());
            EXPECT_FALSE(queue.ready());
            EXPECT_FALSE(called.load());
            EXPECT_THROW(ring.release(0), std::logic_error);

            VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            signal.semaphore = semaphore;
            signal.value = 1;
            ASSERT_EQ(vkSignalSemaphore(vk_device, &signal), VK_SUCCESS);

            ring.wait(0);
            ring.wait(1);
            ring.wait(2);
            queue.wait();
            EXPECT_TRUE(called.load());
            std::array<float, 4> slot{};
            std::memcpy(slot.data(), ring.slot_bytes(0).data(), sizeof(slot));
            EXPECT_EQ(slot[1], 5.f);
            EXPECT_EQ(slot[2], 6.f);
            float first = 0.f;
            std::memcpy(&first, ring.slot_bytes(1).data(), sizeof(first));
            EXPECT_EQ(first, 4.f);
            std::memcpy(slot.data(), ring.slot_bytes(2).data(), sizeof(slot));
            EXPECT_EQ(slot, (std::array<float, 4>{4.f, 5.f, 6.f, 7.f}));
            for (size_t index = 0; index < 3; ++index)
                ring.release(index);

            // A released slot is reusable and reads the newest values.
            device.fill_(9.f);
            ring.enqueue(device, 0, sizeof(float), 1, 12);
            ring.seal(1);
            ring.wait(1);
            std::memcpy(&first, ring.slot_bytes(1).data() + 12, sizeof(first));
            EXPECT_EQ(first, 9.f);
            queue.set_consumer_timeline(vk_device, {});
            EXPECT_FALSE(weak.expired());
        }
        vkDestroySemaphore(vk_device, semaphore, nullptr);
    }

#if LFS_HAS_CUDA
    using namespace std::chrono_literals;
    TEST(TensorQueueContract, CudaExecutionTargetPreservesStreamAndDoesNotWaitOnHost) {
        if (!gpu_backend_available(GpuBackend::CUDA))
            GTEST_SKIP();
        GpuBackendScope backend(GpuBackend::CUDA);
        const auto previous = TensorExecutionTarget::current();
        for (bool default_stream : {false, true}) {
            SCOPED_TRACE(default_stream);
            TensorWorkQueue owned(GpuBackend::CUDA);
            TensorWorkQueue producer(GpuBackend::CUDA, default_stream ? nullptr : owned.native_handle());
            TensorWorkQueue consumer(GpuBackend::CUDA);
            TensorExecutionTarget::Scope producer_scope(producer);
            const auto target = TensorExecutionTarget::current();
            EXPECT_EQ(target.native_handle(), producer.native_handle());
            Tensor value = Tensor::full({16}, 1.f, Device::GPU);
            EXPECT_EQ(value.execution_target(), target);
            TensorFence done(GpuBackend::CUDA);
            GpuElapsed timer(GpuBackend::CUDA, 2);
            TensorUpload upload;
            const Tensor host = Tensor::full({16}, 7.f, Device::CPU);
            producer.wait();
            consumer.wait();
            lfs::test::CudaStreamGate gate;
            ASSERT_EQ(gate.block(static_cast<cudaStream_t>(producer.native_handle())), cudaSuccess);
            ASSERT_TRUE(gate.entered());
            auto submitted = std::async(std::launch::async, [&] {
                TensorExecutionTarget::Scope scope(producer);
                EXPECT_EQ(TensorExecutionTarget::current(), target);
                EXPECT_TRUE(timer.mark(0, producer));
                upload.enqueue(value, host, target);
                EXPECT_TRUE(timer.mark(1, producer));
                value.sync_to_stream(consumer);
                value.record_stream(consumer);
                value.set_stream(consumer);
                done.record(consumer);
                done.wait_on(producer);
            });
            const auto status = submitted.wait_for(1s);
            EXPECT_FALSE(gate.released());
            if (status == std::future_status::ready)
                EXPECT_FALSE(done.ready());
            gate.release();
            EXPECT_EQ(status, std::future_status::ready);
            submitted.get();
            EXPECT_EQ(value.stream(), static_cast<cudaStream_t>(consumer.native_handle()));
            done.wait();
            upload.wait();
            EXPECT_TRUE(timer.wait_event(1));
            EXPECT_TRUE(timer.milliseconds(0, 1).has_value());
            EXPECT_TRUE(timer.wait_queue(producer));
            TensorReadback readback;
            readback.enqueue(value, TensorExecutionTarget(consumer));
            std::array<float, 16> result{};
            readback.wait(std::as_writable_bytes(std::span(result)));
            for (float element : result)
                EXPECT_EQ(element, 7.f);
            target.set_name("test.queue");
            push_gpu_range("test.range");
            pop_gpu_range();
        }
        EXPECT_EQ(TensorExecutionTarget::current(), previous);
    }

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
