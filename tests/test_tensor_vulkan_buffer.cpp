/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/headless_vulkan_device.hpp"
#include "core/tensor.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_rad.hpp"
#include "core/tensor_readback.hpp"
#include "core/tensor_upload.hpp"
#include "core/tensor_vulkan_interop.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <thread>
#include <vector>

namespace {
    using namespace lfs::core;

    class TensorVulkanBufferQuery : public testing::Test {
    protected:
        std::optional<lfs::core::HeadlessAdoptedDevice> adopted_;

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
            adopted_.reset();
        }
    };

    void wait_timeline_value(const uint64_t value) {
        internal::backend_ops(GpuBackend::Vulkan).synchronize_device();
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        uint64_t counter = 0;
        do {
            counter = internal::vulkan_completed_timeline_for_testing();
            if (counter >= value) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (std::chrono::steady_clock::now() < deadline);
        FAIL() << "vulkan backend timeline did not reach " << value
               << " (counter=" << counter << ")";
    }

    TEST_F(TensorVulkanBufferQuery, EmptyQueueBatchStillWaitsForTheConsumerTimeline) {
        ASSERT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan));
        auto candidate = lfs::core::HeadlessAdoptedDevice::try_create();
        if (!candidate)
            GTEST_SKIP() << "No Vulkan tensor device";
        adopted_.emplace(std::move(*candidate));
        const auto handles = adopted_->handles();
        ASSERT_TRUE(adopt_vulkan_device(handles));
        const auto device = static_cast<VkDevice>(handles.device);
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        info.pNext = &type;
        VkSemaphore semaphore = VK_NULL_HANDLE;
        ASSERT_EQ(vkCreateSemaphore(device, &info, nullptr, &semaphore), VK_SUCCESS);
        {
            TensorWorkQueue queue(GpuBackend::Vulkan, device, semaphore);
            const auto completion = queue.execute([] {}, 1);
            EXPECT_GT(completion.timeline().value, 0u);
            EXPECT_FALSE(completion.ready());
            VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            signal.semaphore = semaphore;
            signal.value = 1;
            ASSERT_EQ(vkSignalSemaphore(device, &signal), VK_SUCCESS);
            completion.wait();
            EXPECT_TRUE(completion.ready());
        }
        vkDestroySemaphore(device, semaphore, nullptr);
    }

    TEST_F(TensorVulkanBufferQuery, PendingValueIsSignalledAfterSynchronize) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        const Tensor source = Tensor::full({256}, 1.25f, Device::GPU);
        const Tensor result = source.mul(3.0f);
        const auto buffer = tensor_vulkan_buffer(result);
        ASSERT_TRUE(buffer.has_value());
        EXPECT_NE(buffer->buffer, nullptr);
        EXPECT_GT(buffer->pending_timeline_value, 0u);
        EXPECT_EQ(buffer->bytes, result.bytes());
        EXPECT_EQ(buffer->device_address,
                  static_cast<uint64_t>(reinterpret_cast<uintptr_t>(result.data_ptr())));
        wait_timeline_value(buffer->pending_timeline_value);
    }

    TEST_F(TensorVulkanBufferQuery, ViewOffsetBytesAndDeviceAddressMatchStorage) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        const Tensor base = Tensor::from_vector(
            std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, Device::GPU);
        const Tensor sliced = base.slice(0, 1, 2);
        const auto buffer = tensor_vulkan_buffer(sliced);
        ASSERT_TRUE(buffer.has_value());
        EXPECT_EQ(buffer->offset, 3u * sizeof(float));
        EXPECT_EQ(buffer->bytes, 3u * sizeof(float));
        EXPECT_EQ(buffer->device_address,
                  static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sliced.data_ptr())));
        EXPECT_EQ(buffer->buffer, tensor_vulkan_buffer(base)->buffer);
    }

    TEST_F(TensorVulkanBufferQuery, KeepAlivePinsStorageAfterTensorDrop) {
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor warm = Tensor::ones({8}, Device::GPU);
            ASSERT_FLOAT_EQ(warm.sum_scalar(), 8.0f);
        }
        Tensor::trim_memory_pool();
        const uint64_t baseline = internal::vulkan_live_vma_objects_for_testing();
        std::shared_ptr<void> pin;
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            Tensor tensor = Tensor::ones({4096}, Device::GPU);
            const auto buffer = tensor_vulkan_buffer(tensor);
            ASSERT_TRUE(buffer.has_value());
            ASSERT_TRUE(static_cast<bool>(buffer->keep_alive));
            pin = buffer->keep_alive;
            tensor = Tensor();
            Tensor::trim_memory_pool();
            EXPECT_GT(internal::vulkan_live_vma_objects_for_testing(), baseline);
        }
        Tensor::trim_memory_pool();
        EXPECT_GT(internal::vulkan_live_vma_objects_for_testing(), baseline);
        pin.reset();
        internal::backend_ops(GpuBackend::Vulkan).synchronize_device();
        Tensor::trim_memory_pool();
        EXPECT_EQ(internal::vulkan_live_vma_objects_for_testing(), baseline);
    }

    TEST_F(TensorVulkanBufferQuery, CudaAndCpuTensorsReturnNullopt) {
        if (gpu_backend_available(GpuBackend::CUDA)) {
            GpuBackendScope cuda_scope(GpuBackend::CUDA);
            const Tensor cuda = Tensor::ones({8}, Device::GPU);
            EXPECT_EQ(gpu_backend_of(cuda), GpuBackend::CUDA);
            EXPECT_FALSE(tensor_vulkan_buffer(cuda).has_value());
        }
        const Tensor cpu = Tensor::ones({8}, Device::CPU);
        EXPECT_FALSE(tensor_vulkan_buffer(cpu).has_value());
    }

    class TensorVulkanInteropOrdering : public TensorVulkanBufferQuery,
                                        public testing::WithParamInterface<std::pair<GpuBackend, bool>> {};

    TEST_P(TensorVulkanInteropOrdering, ProducerCopyAndReverseWaitPreserveValues) {
        const auto [source_backend, foreign_device] = GetParam();
        if (!gpu_backend_available(source_backend))
            GTEST_SKIP() << "Source backend unavailable";
        ASSERT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan));
        if (foreign_device) {
            GpuBackendScope backend(GpuBackend::Vulkan);
            const auto initialize = Tensor::ones({1}, Device::GPU);
        }
        auto candidate = lfs::core::HeadlessAdoptedDevice::try_create(true);
        if (!candidate)
            GTEST_SKIP() << "Vulkan external memory and semaphore exports unavailable";
        adopted_.emplace(std::move(*candidate));
        const auto handles = adopted_->handles();
        if (!foreign_device)
            ASSERT_TRUE(adopt_vulkan_device(handles));
        const auto device = static_cast<VkDevice>(handles.device);
        const auto queue = static_cast<VkQueue>(handles.queue);
        TensorVulkanInterop interop(VulkanInteropDevice{
            .physical_device = handles.physical_device,
            .device = handles.device,
            .queue_families = {handles.queue_family},
            .queue_family_count = 1,
            .external_memory = true,
            .external_semaphore = true,
            .metal_objects = handles.metal_objects});
        Tensor source_storage = interop.empty({40}, DataType::Float32, source_backend);
        Tensor source = source_storage.slice(0, 4, 36);
        {
            GpuBackendScope scope(source_backend);
            source.copy_from(Tensor::full({32}, 7.25f, Device::GPU));
        }
        const auto source_buffer = interop.buffer(source);
        ASSERT_TRUE(source_buffer);
        EXPECT_EQ(source_buffer->buffer, interop.buffer(source)->buffer);
        EXPECT_EQ(source_buffer->device, handles.device);
        const std::array<const Tensor*, 1> source_tensors{&source};
        const auto ready = interop.ready(source_tensors).timeline();
        Tensor destination = interop.empty({32}, DataType::Float32, GpuBackend::Vulkan);
        const auto destination_buffer = interop.buffer(destination);
        ASSERT_TRUE(destination_buffer);

        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = handles.queue_family;
        VkCommandPool pool = VK_NULL_HANDLE;
        ASSERT_EQ(vkCreateCommandPool(device, &pool_info, nullptr, &pool), VK_SUCCESS);
        VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_info.commandPool = pool;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        ASSERT_EQ(vkAllocateCommandBuffers(device, &command_info, &command), VK_SUCCESS);

        VkExportSemaphoreCreateInfo export_info{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
#ifdef _WIN32
        export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
        export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
        VkSemaphoreTypeCreateInfo timeline_info{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        timeline_info.pNext = &export_info;
        timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        semaphore_info.pNext = &timeline_info;
        VkSemaphore completion = VK_NULL_HANDLE;
        ASSERT_EQ(vkCreateSemaphore(device, &semaphore_info, nullptr, &completion), VK_SUCCESS);

        const auto submit = [&](VkSemaphore wait_semaphore, uint64_t wait_value, uint64_t signal_value) {
            VkTimelineSemaphoreSubmitInfo values{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            values.waitSemaphoreValueCount = wait_semaphore ? 1u : 0u;
            values.pWaitSemaphoreValues = wait_semaphore ? &wait_value : nullptr;
            values.signalSemaphoreValueCount = 1;
            values.pSignalSemaphoreValues = &signal_value;
            constexpr VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit_info.pNext = &values;
            submit_info.waitSemaphoreCount = wait_semaphore ? 1u : 0u;
            submit_info.pWaitSemaphores = wait_semaphore ? &wait_semaphore : nullptr;
            submit_info.pWaitDstStageMask = wait_semaphore ? &stage : nullptr;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &command;
            submit_info.signalSemaphoreCount = 1;
            submit_info.pSignalSemaphores = &completion;
            return vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
        };
        const auto record = [&](auto&& write) {
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS)
                return false;
            write();
            VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.memoryBarrierCount = 1;
            dependency.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(command, &dependency);
            return vkEndCommandBuffer(command) == VK_SUCCESS;
        };
        ASSERT_TRUE(record([&] {
            const VkBufferCopy copy{source_buffer->offset, destination_buffer->offset, source.bytes()};
            vkCmdCopyBuffer(command, static_cast<VkBuffer>(source_buffer->buffer),
                            static_cast<VkBuffer>(destination_buffer->buffer), 1, &copy);
        }));
        ASSERT_EQ(submit(static_cast<VkSemaphore>(ready.semaphore), ready.value, 1), VK_SUCCESS);
        const std::array<const Tensor*, 1> destinations{&destination};
        interop.wait(destinations, {completion, 1});
        TensorReadback readback;
        std::array<float, 32> values{};
        readback.enqueue(destination);
        readback.wait(std::as_writable_bytes(std::span(values)));
        for (const auto value : values)
            EXPECT_FLOAT_EQ(value, 7.25f);

        Tensor tail = source.slice(0, 28, 32);
        const auto tail_buffer = interop.buffer(tail);
        ASSERT_TRUE(tail_buffer);
        ASSERT_EQ(vkResetCommandBuffer(command, 0), VK_SUCCESS);
        ASSERT_TRUE(record([&] {
            vkCmdFillBuffer(command, static_cast<VkBuffer>(source_buffer->buffer),
                            source_buffer->offset, source.bytes() - tail.bytes(), 0x40200000u);
            vkCmdFillBuffer(command, static_cast<VkBuffer>(tail_buffer->buffer),
                            tail_buffer->offset, tail.bytes(), 0x40600000u);
        }));
        ASSERT_EQ(submit(VK_NULL_HANDLE, 0, 2), VK_SUCCESS);
        const std::array<const Tensor*, 2> written_views{&source, &tail};
        interop.wait(written_views, {completion, 2});
        readback.enqueue(source);
        readback.wait(std::as_writable_bytes(std::span(values)));
        for (size_t i = 0; i < values.size(); ++i)
            EXPECT_FLOAT_EQ(values[i], i < 28 ? 2.5f : 3.5f);
        ASSERT_EQ(vkQueueWaitIdle(queue), VK_SUCCESS);
        interop.release_timeline(completion);
        vkDestroySemaphore(device, completion, nullptr);
        vkDestroyCommandPool(device, pool, nullptr);
    }

    INSTANTIATE_TEST_SUITE_P(Storage, TensorVulkanInteropOrdering,
                             testing::Values(std::pair{GpuBackend::CUDA, false},
                                             std::pair{GpuBackend::Vulkan, false},
                                             std::pair{GpuBackend::Vulkan, true},
                                             std::pair{GpuBackend::Metal, false}));

    // The LOD page queue on Metal: its work waits on the GPU for the consumer
    // timeline, uploads reuse storage that Metal already used, and completion
    // reaches the consumer device through the queue timeline.
    TEST_F(TensorVulkanBufferQuery, MetalWorkQueueOrdersAgainstTheConsumerTimeline) {
        if (!gpu_backend_available(GpuBackend::Metal))
            GTEST_SKIP() << "Metal backend unavailable";
        auto candidate = lfs::core::HeadlessAdoptedDevice::try_create(true);
        if (!candidate || !candidate->handles().metal_objects)
            GTEST_SKIP() << "No Vulkan device shares Metal objects";
        adopted_.emplace(std::move(*candidate));
        const auto device = static_cast<VkDevice>(adopted_->handles().device);
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        info.pNext = &type;
        VkSemaphore consumer = VK_NULL_HANDLE;
        ASSERT_EQ(vkCreateSemaphore(device, &info, nullptr, &consumer), VK_SUCCESS);
        {
            const GpuBackendScope scope(GpuBackend::Metal);
            TensorWorkQueue queue(GpuBackend::Metal, device, consumer);
            ASSERT_NE(queue.timeline(), nullptr);
            Tensor staging = Tensor::zeros({64}, Device::GPU);
            Tensor output = Tensor::empty({64}, Device::GPU);
            TensorUpload upload;
            const auto completion = queue.execute([&] {
                upload.enqueue(staging, Tensor::full({64}, 3.0f, Device::CPU));
                output.copy_from(staging.mul(2.0f));
            },
                                                  1);
            EXPECT_FALSE(upload.pending());
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            EXPECT_FALSE(completion.ready());

            VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            signal.semaphore = consumer;
            signal.value = 1;
            ASSERT_EQ(vkSignalSemaphore(device, &signal), VK_SUCCESS);
            completion.wait();
            const auto point = completion.timeline();
            EXPECT_EQ(point.semaphore, queue.timeline());
            const auto timeline = static_cast<VkSemaphore>(point.semaphore);
            VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            wait.semaphoreCount = 1;
            wait.pSemaphores = &timeline;
            wait.pValues = &point.value;
            EXPECT_EQ(vkWaitSemaphores(device, &wait, 1'000'000'000), VK_SUCCESS);
            for (const float value : output.cpu().to_vector())
                EXPECT_FLOAT_EQ(value, 6.0f);

            upload.enqueue(staging, Tensor::full({64}, 5.0f, Device::CPU));
            EXPECT_FALSE(upload.pending());
            for (const float value : staging.cpu().to_vector())
                EXPECT_FLOAT_EQ(value, 5.0f);
        }
        vkDestroySemaphore(device, consumer, nullptr);
    }

    // Metal waits for consumer work on the GPU instead of the host, yet a host
    // read of a waited-for tensor still sees the consumer's writes.
    TEST_F(TensorVulkanBufferQuery, MetalWaitOrdersHostReadsAfterConsumerWrites) {
        if (!gpu_backend_available(GpuBackend::Metal))
            GTEST_SKIP() << "Metal backend unavailable";
        auto candidate = lfs::core::HeadlessAdoptedDevice::try_create(true);
        if (!candidate || !candidate->handles().metal_objects)
            GTEST_SKIP() << "No Vulkan device shares Metal objects";
        adopted_.emplace(std::move(*candidate));
        const auto handles = adopted_->handles();
        const auto device = static_cast<VkDevice>(handles.device);
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        semaphore_info.pNext = &type;
        VkSemaphore gate = VK_NULL_HANDLE, written = VK_NULL_HANDLE;
        ASSERT_EQ(vkCreateSemaphore(device, &semaphore_info, nullptr, &gate), VK_SUCCESS);
        ASSERT_EQ(vkCreateSemaphore(device, &semaphore_info, nullptr, &written), VK_SUCCESS);
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.queueFamilyIndex = handles.queue_family;
        VkCommandPool pool = VK_NULL_HANDLE;
        ASSERT_EQ(vkCreateCommandPool(device, &pool_info, nullptr, &pool), VK_SUCCESS);
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = pool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        ASSERT_EQ(vkAllocateCommandBuffers(device, &allocate, &command), VK_SUCCESS);
        {
            TensorVulkanInterop interop(VulkanInteropDevice{
                .physical_device = handles.physical_device,
                .device = handles.device,
                .queue_families = {handles.queue_family},
                .queue_family_count = 1,
                .metal_objects = true});
            const GpuBackendScope scope(GpuBackend::Metal);
            Tensor target = interop.empty({64}, DataType::Float32, GpuBackend::Metal);
            target.fill_(1.0f);
            const std::array<const Tensor*, 1> tensors{&target};
            const auto ready = interop.ready(tensors).timeline();
            const auto buffer = interop.buffer(target);
            ASSERT_TRUE(buffer);
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            ASSERT_EQ(vkBeginCommandBuffer(command, &begin), VK_SUCCESS);
            vkCmdFillBuffer(command, static_cast<VkBuffer>(buffer->buffer), buffer->offset, target.bytes(), 0x40E80000u);
            ASSERT_EQ(vkEndCommandBuffer(command), VK_SUCCESS);
            // The fill waits for the Metal writes and for the host to open the gate.
            const std::array<VkSemaphore, 2> waits{static_cast<VkSemaphore>(ready.semaphore), gate};
            const std::array<uint64_t, 2> wait_values{ready.value, 1};
            const std::array<VkPipelineStageFlags, 2> stages{VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
            const uint64_t signal_value = 1;
            VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            timeline.waitSemaphoreValueCount = 2;
            timeline.pWaitSemaphoreValues = wait_values.data();
            timeline.signalSemaphoreValueCount = 1;
            timeline.pSignalSemaphoreValues = &signal_value;
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.pNext = &timeline;
            submit.waitSemaphoreCount = 2;
            submit.pWaitSemaphores = waits.data();
            submit.pWaitDstStageMask = stages.data();
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores = &written;
            ASSERT_EQ(vkQueueSubmit(static_cast<VkQueue>(handles.queue), 1, &submit, VK_NULL_HANDLE), VK_SUCCESS);

            auto waited = std::async(std::launch::async, [&] { interop.wait(tensors, {written, 1}); });
            const bool queued = waited.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
            EXPECT_TRUE(queued) << "the wait blocked the host";
            std::future<std::vector<float>> read;
            if (queued) {
                read = std::async(std::launch::async, [&] { return target.cpu().to_vector(); });
                EXPECT_EQ(read.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
            }
            VkSemaphoreSignalInfo open{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            open.semaphore = gate;
            open.value = 1;
            ASSERT_EQ(vkSignalSemaphore(device, &open), VK_SUCCESS);
            waited.get();
            if (queued) {
                for (const float value : read.get())
                    EXPECT_FLOAT_EQ(value, 7.25f);
            }
            ASSERT_EQ(vkQueueWaitIdle(static_cast<VkQueue>(handles.queue)), VK_SUCCESS);
        }
        vkDestroyCommandPool(device, pool, nullptr);
        vkDestroySemaphore(device, written, nullptr);
        vkDestroySemaphore(device, gate, nullptr);
    }

    // Degree-zero splats carry an empty SH tensor, which has no storage, so
    // quantizing their resident LOD pages must not record it as a read.
    TEST_F(TensorVulkanBufferQuery, RadResidentPagesQuantizeWithoutSh) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        constexpr uint32_t splats = 64;
        RadPagePool pool{.page_splats = splats};
        const std::array<size_t, 7> sizes{splats * 12, splats * 8, 0, splats * 8, splats * 8, splats * 2,
                                          radq::kPageFrameBytes};
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (sizes[i] != 0)
                pool.regions[i] = Tensor::zeros({sizes[i]}, Device::GPU, DataType::UInt8);
        }
        const RadPageSources source{.means = Tensor::full({splats, 3}, 1.5f, Device::GPU),
                                    .sh0 = Tensor::zeros({splats, 3}, Device::GPU),
                                    .shN = Tensor::empty({splats, 0, 3}, Device::GPU),
                                    .rotation = Tensor::zeros({splats, 4}, Device::GPU),
                                    .scaling = Tensor::zeros({splats, 3}, Device::GPU),
                                    .opacity = Tensor::zeros({splats}, Device::GPU),
                                    .count = splats};
        ASSERT_NO_THROW(rad_page_quantize(source, pool, 0));
        const Tensor means = pool.regions[0].cpu();
        EXPECT_EQ(static_cast<const float*>(means.data_ptr())[splats * 3 - 1], 1.5f);
    }

    TEST_F(TensorVulkanBufferQuery, InteropBatchesVulkanReadinessAndPreservesViewOffsets) {
        ASSERT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan));
        auto candidate = lfs::core::HeadlessAdoptedDevice::try_create();
        if (!candidate)
            GTEST_SKIP() << "No Vulkan tensor device";
        adopted_.emplace(std::move(*candidate));
        const auto handles = adopted_->handles();
        ASSERT_TRUE(adopt_vulkan_device(handles));
        TensorVulkanInterop interop(VulkanInteropDevice{
            .physical_device = handles.physical_device,
            .device = handles.device,
            .queue_families = {handles.queue_family},
            .queue_family_count = 1});
        GpuBackendScope scope(GpuBackend::Vulkan);
        Tensor first = Tensor::full({16}, 2.0f, Device::GPU);
        Tensor second = first.mul(3.0f);
        const std::array<const Tensor*, 2> tensors{&first, &second};
        const auto point = interop.ready(tensors).timeline();
        EXPECT_NE(point.semaphore, nullptr);
        EXPECT_GT(point.value, 0u);
        const auto base = interop.buffer(second);
        const auto view = interop.buffer(second.slice(0, 4, 8));
        ASSERT_TRUE(base);
        ASSERT_TRUE(view);
        EXPECT_EQ(view->buffer, base->buffer);
        EXPECT_EQ(view->offset, base->offset + 4 * sizeof(float));
        EXPECT_EQ(view->bytes, 4 * sizeof(float));
        EXPECT_LE(base->pending_timeline_value, point.value);
    }

    class TensorWhereInto : public TensorVulkanBufferQuery,
                            public testing::WithParamInterface<std::pair<GpuBackend, DataType>> {};

    TEST_P(TensorWhereInto, SelectsScalarAndPreservesSourceWithAliasedOutput) {
        const auto [backend, dtype] = GetParam();
        if (!gpu_backend_available(backend))
            GTEST_SKIP() << "Source backend unavailable";
        const GpuBackendScope scope(backend);
        const Tensor source = Tensor::from_vector(std::vector<float>{1, 2, 3, 4}, {4}, Device::GPU).to(dtype);
        const Tensor mask = Tensor::from_vector(std::vector<float>{1, 0, 1, 0}, {4}, Device::GPU).to(DataType::Bool);
        Tensor output = Tensor::empty({4}, Device::GPU, dtype);
        where_into(output, mask, -20.0f, source);
        const auto verify = [](const Tensor& tensor, const std::array<float, 4>& expected) {
            TensorReadback readback;
            readback.enqueue(tensor.to(DataType::Float32));
            std::array<float, 4> values{};
            readback.wait(std::as_writable_bytes(std::span(values)));
            EXPECT_EQ(values, expected);
        };
        verify(output, {-20, 2, -20, 4});
        where_into(output, mask, 13.0f, output);
        verify(output, {13, 2, 13, 4});
        verify(source, {1, 2, 3, 4});
    }

    INSTANTIATE_TEST_SUITE_P(TensorBackends, TensorWhereInto,
                             testing::Values(std::pair{GpuBackend::CUDA, DataType::Float32},
                                             std::pair{GpuBackend::CUDA, DataType::Float16},
                                             std::pair{GpuBackend::Vulkan, DataType::Float32},
                                             std::pair{GpuBackend::Vulkan, DataType::Float16},
                                             std::pair{GpuBackend::Metal, DataType::Float32},
                                             std::pair{GpuBackend::Metal, DataType::Float16}));

} // namespace
