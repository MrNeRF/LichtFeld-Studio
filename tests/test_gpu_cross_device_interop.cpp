/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/gpu_device_info.hpp"
#include "core/gpu_device_selection.hpp"
#include "core/headless_vulkan_device.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_upload.hpp"
#include "core/tensor_vulkan_interop.hpp"
#include <cstdlib>
#include <gtest/gtest.h>
#include <type_traits>

namespace {
    using namespace lfs::core;
    struct DirectInteropRestore {
        bool previous = cuda_vulkan_direct_interop();
        explicit DirectInteropRestore(bool enabled = false) { set_cuda_vulkan_direct_interop(enabled); }
        ~DirectInteropRestore() { set_cuda_vulkan_direct_interop(previous); }
    };
    struct Submission {
        VkDevice device;
        VkQueue queue;
        VkCommandPool pool = VK_NULL_HANDLE;
        VkSemaphore timeline = VK_NULL_HANDLE;
        ~Submission() {
            vkQueueWaitIdle(queue);
            if (timeline)
                vkDestroySemaphore(device, timeline, nullptr);
            if (pool)
                vkDestroyCommandPool(device, pool, nullptr);
        }
    };

    void verifyAdapterRoundTrips(const GpuBackend backend) {
        if (!gpu_backend_available(backend))
            GTEST_SKIP() << "Requested tensor backend is unavailable";
        DirectInteropRestore restore;
        GpuBackendScope scope(backend);
        const auto source_probe = Tensor::zeros({1}, Device::GPU);
        const auto source = gpu_backend_device_info(backend);
        ASSERT_TRUE(source);
        ::testing::Test::RecordProperty("source_adapter", source->name + " " + gpu_identity_string(source->uuid));
        if (backend == GpuBackend::Vulkan) {
            if (const auto* requested = std::getenv("LFS_TEST_VULKAN_DEVICE"))
                EXPECT_EQ(gpu_identity_string(source->uuid), normalize_gpu_identity(requested));
        }
        size_t exercised = 0;
        for (const auto& hardware : enumerate_vulkan_hardware()) {
            if (!hardware.rendering_unavailable_reason.empty())
                continue;
            SCOPED_TRACE(hardware.name + " " + hardware.uuid);
            ::testing::Test::RecordProperty("consumer_adapter_" + std::to_string(exercised), hardware.name + " " + hardware.uuid);
            auto owner = HeadlessAdoptedDevice::try_create(false, true, hardware.uuid);
            ASSERT_TRUE(owner);
            const auto handles = owner->handles();
            {
                // A secondary consumer must use its own compatibility policy,
                // even when the main viewer supports direct CUDA interop.
                DirectInteropRestore main_viewer_direct(true);
                Submission submission{static_cast<VkDevice>(handles.device), static_cast<VkQueue>(handles.queue)};
                VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
                type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
                type.initialValue = 1;
                VkSemaphoreCreateInfo create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
                create.pNext = &type;
                ASSERT_EQ(vkCreateSemaphore(submission.device, &create, nullptr, &submission.timeline), VK_SUCCESS);
                TensorWorkQueue queue(backend, handles.device, nullptr, false);
                queue.set_consumer_timeline(handles.device, {submission.timeline, 1, {}});
                EXPECT_EQ(queue.timeline(), nullptr);
                auto value = Tensor::zeros({1}, Device::GPU);
                queue.execute([&] { value.fill_(3.0f); }, 1).wait();
                EXPECT_FLOAT_EQ(value.cpu().ptr<float>()[0], 3.0f);
            }
            TensorVulkanInterop interop({.physical_device = handles.physical_device,
                                         .device = handles.device,
                                         .queue_families = {handles.queue_family},
                                         .queue_family_count = 1});
            const auto verify = [&]<typename Scalar>(const std::vector<Scalar>& initial,
                                                     const std::vector<Scalar>& replacement_values,
                                                     bool offset_view = false, bool half = false) {
                Submission submit{static_cast<VkDevice>(handles.device), static_cast<VkQueue>(handles.queue)};
                const auto make_tensor = [half](const std::vector<Scalar>& values) {
                    auto result = Tensor::from_vector(std::vector<float>(values.begin(), values.end()), {values.size()}, Device::GPU);
                    if constexpr (std::is_same_v<Scalar, uint8_t>)
                        return result.to(DataType::UInt8);
                    else
                        return half ? result.to(DataType::Float16) : result;
                };
                auto input = make_tensor(initial);
                if (offset_view) {
                    auto padded = initial;
                    padded.insert(padded.begin(), Scalar{});
                    padded.push_back(Scalar{});
                    input = make_tensor(padded).slice(0, 1, initial.size() + 1);
                }
                auto output = make_tensor(std::vector<Scalar>(initial.size(), Scalar{}));
                auto first = interop.buffer(input);
                ASSERT_TRUE(first);
                // Preparing another frame must never mutate a lease still read by the viewer.
                auto replacement = make_tensor(replacement_values);
                input.copy_from(replacement);
                auto second = interop.buffer(input);
                auto destination = interop.buffer(output);
                ASSERT_TRUE(second);
                ASSERT_TRUE(destination);
                EXPECT_NE(first->buffer, second->buffer);
                VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
                pool.queueFamilyIndex = handles.queue_family;
                ASSERT_EQ(vkCreateCommandPool(submit.device, &pool, nullptr, &submit.pool), VK_SUCCESS);
                VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                allocate.commandPool = submit.pool;
                allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                allocate.commandBufferCount = 1;
                VkCommandBuffer command{};
                ASSERT_EQ(vkAllocateCommandBuffers(submit.device, &allocate, &command), VK_SUCCESS);
                VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                ASSERT_EQ(vkBeginCommandBuffer(command, &begin), VK_SUCCESS);
                VkBufferCopy copy{first->offset, destination->offset, (input.bytes() + 3) & ~size_t{3}};
                vkCmdCopyBuffer(command, static_cast<VkBuffer>(first->buffer), static_cast<VkBuffer>(destination->buffer), 1, &copy);
                VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                     0, 1, &host, 0, nullptr, 0, nullptr);
                ASSERT_EQ(vkEndCommandBuffer(command), VK_SUCCESS);
                VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
                type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
                VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
                semaphore.pNext = &type;
                ASSERT_EQ(vkCreateSemaphore(submit.device, &semaphore, nullptr, &submit.timeline), VK_SUCCESS);
                uint64_t value = 1;
                VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
                timeline.signalSemaphoreValueCount = 1;
                timeline.pSignalSemaphoreValues = &value;
                VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                info.pNext = &timeline;
                info.commandBufferCount = 1;
                info.pCommandBuffers = &command;
                info.signalSemaphoreCount = 1;
                info.pSignalSemaphores = &submit.timeline;
                ASSERT_EQ(vkQueueSubmit(submit.queue, 1, &info, VK_NULL_HANDLE), VK_SUCCESS);
                const Tensor* outputs[]{&output};
                interop.wait(outputs, {submit.timeline, value, {}});
                const auto result = half ? output.to(DataType::Float32).cpu() : output.cpu();
                const auto* values = static_cast<const Scalar*>(result.data_ptr());
                for (size_t i = 0; i < initial.size(); ++i)
                    EXPECT_EQ(values[i], initial[i]);
                interop.drain(backend);
            };
            verify(std::vector<float>{1, 2, 3, 4}, std::vector<float>{5, 6, 7, 8});
            verify(std::vector<float>{1, 2, 3, 4}, std::vector<float>{5, 6, 7, 8}, false, true);
            verify(std::vector<uint8_t>{9, 10, 11}, std::vector<uint8_t>{12, 13, 14});
            verify(std::vector<float>{21, 22, 23, 24}, std::vector<float>{25, 26, 27, 28}, true);
            ++exercised;
        }
        ::testing::Test::RecordProperty("exercised_adapters", static_cast<int>(exercised));
        if (!exercised)
            GTEST_SKIP() << "No compatible Vulkan rendering GPU";
    }
    TEST(GpuCrossDeviceInterop, CudaSnapshotsRoundTripThroughEveryCompatibleVulkanAdapter) {
        verifyAdapterRoundTrips(GpuBackend::CUDA);
    }

    TEST(GpuCrossDeviceInterop, VulkanSnapshotsRoundTripThroughEveryCompatibleVulkanAdapter) {
        // Run in a separate filtered process for each source GPU: tensor options
        // intentionally become immutable once a Vulkan context is created.
        if (const auto* device = std::getenv("LFS_TEST_VULKAN_DEVICE")) {
            TensorBackendOptions options;
            options.vulkan_device = device;
            ASSERT_TRUE(set_tensor_backend_options(options));
        }
        verifyAdapterRoundTrips(GpuBackend::Vulkan);
    }
} // namespace
