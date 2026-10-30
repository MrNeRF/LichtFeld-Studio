/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "rendering/passes/collision_preview_upload.hpp"
#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>

namespace {
    using Status = lfs::vis::detail::CollisionUploadStatus;
    void check(VkResult result) {
        if (result != VK_SUCCESS)
            throw std::runtime_error("Vulkan call failed: " + std::to_string(int(result)));
    }
    struct Upload {
        VkDevice device = VK_NULL_HANDLE;
        VkBuffer buffers[2]{};
        VkDeviceMemory memory[2]{};
        VkFence fence = VK_NULL_HANDLE;
        ~Upload() {
            for (int i = 0; i < 2; ++i) {
                if (buffers[i])
                    vkDestroyBuffer(device, buffers[i], nullptr);
                if (memory[i])
                    vkFreeMemory(device, memory[i], nullptr);
            }
            if (fence)
                vkDestroyFence(device, fence, nullptr);
        }
    };
    class CollisionPreviewVulkan : public ::testing::TestWithParam<uint32_t> {
    protected:
        VkInstance instance = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkPhysicalDevice physical = VK_NULL_HANDLE;
        VkQueue queue = VK_NULL_HANDLE;
        VkCommandPool pool = VK_NULL_HANDLE;
        void SetUp() override {
            VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS)
                GTEST_SKIP() << "Vulkan instance unavailable";
            uint32_t count = 0;
            check(vkEnumeratePhysicalDevices(instance, &count, nullptr));
            std::vector<VkPhysicalDevice> devices(count);
            check(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
            if (GetParam() >= count)
                GTEST_SKIP() << "Adapter unavailable";
            physical = devices[GetParam()];
            uint32_t count_families = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physical, &count_families, nullptr);
            std::vector<VkQueueFamilyProperties> families(count_families);
            vkGetPhysicalDeviceQueueFamilyProperties(physical, &count_families, families.data());
            uint32_t family = count_families;
            for (uint32_t i = 0; i < count_families; ++i)
                if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                    family = i;
                    break;
                }
            if (family == count_families)
                GTEST_SKIP() << "Graphics queue unavailable";
            const float priority = 1;
            VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            qi.queueFamilyIndex = family;
            qi.queueCount = 1;
            qi.pQueuePriorities = &priority;
            VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
            di.queueCreateInfoCount = 1;
            di.pQueueCreateInfos = &qi;
            check(vkCreateDevice(physical, &di, nullptr, &device));
            vkGetDeviceQueue(device, family, 0, &queue);
            VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pi.queueFamilyIndex = family;
            check(vkCreateCommandPool(device, &pi, nullptr, &pool));
        }
        void TearDown() override {
            if (device) {
                vkDeviceWaitIdle(device);
                if (pool)
                    vkDestroyCommandPool(device, pool, nullptr);
                vkDestroyDevice(device, nullptr);
            }
            if (instance)
                vkDestroyInstance(instance, nullptr);
        }
        std::shared_ptr<Upload> allocate(VkDeviceSize staging, VkDeviceSize total) {
            auto upload = std::make_shared<Upload>();
            upload->device = device;
            VkPhysicalDeviceMemoryProperties properties{};
            vkGetPhysicalDeviceMemoryProperties(physical, &properties);
            for (int i = 0; i < 2; ++i) {
                VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                bi.size = i == 0 ? staging : total;
                bi.usage = i == 0 ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                check(vkCreateBuffer(device, &bi, nullptr, &upload->buffers[i]));
                VkMemoryRequirements requirements{};
                vkGetBufferMemoryRequirements(device, upload->buffers[i], &requirements);
                uint32_t type = properties.memoryTypeCount;
                for (uint32_t j = 0; j < properties.memoryTypeCount; ++j)
                    if ((requirements.memoryTypeBits & (1u << j)) &&
                        (properties.memoryTypes[j].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                        type = j;
                        break;
                    }
                if (type == properties.memoryTypeCount)
                    throw std::runtime_error("No host-visible transfer memory");
                VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                ai.allocationSize = requirements.size;
                ai.memoryTypeIndex = type;
                check(vkAllocateMemory(device, &ai, nullptr, &upload->memory[i]));
                check(vkBindBufferMemory(device, upload->buffers[i], upload->memory[i], 0));
            }
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            check(vkCreateFence(device, &fi, nullptr, &upload->fence));
            return upload;
        }
    };
    TEST_P(CollisionPreviewVulkan, WorkerStreamsThreeChunksAndPublishesExactGpuReadback) {
        constexpr VkDeviceSize chunk = 8 * 1024 * 1024;
        constexpr VkDeviceSize total = chunk * 2 + 28;
        auto upload = allocate(chunk, total);
        lfs::vis::detail::CollisionPreviewUploadJob<Upload> job;
        std::promise<void> entered, release, woke;
        auto gate = release.get_future().share();
        struct Drain {
            std::function<void()> run;
            ~Drain() { run(); }
        } drain{[&] { job.finish(); vkDeviceWaitIdle(device); }};
        ASSERT_TRUE(job.start(upload, [&](Upload& resource, std::stop_token) {
            entered.set_value();
            gate.wait();
            for (VkDeviceSize offset = 0; offset < total; offset += chunk) {
                const auto bytes = std::min(chunk, total - offset);
                void* mapped = nullptr;
                check(vkMapMemory(device, resource.memory[0], 0, VK_WHOLE_SIZE, 0, &mapped));
                for (VkDeviceSize i = 0; i < bytes / sizeof(uint32_t); ++i)
                    static_cast<uint32_t*>(mapped)[i] = uint32_t(offset / sizeof(uint32_t) + i) ^ 0x56780000u;
                VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
                range.memory = resource.memory[0];
                range.size = VK_WHOLE_SIZE;
                check(vkFlushMappedMemoryRanges(device, 1, &range));
                vkUnmapMemory(device, resource.memory[0]);
                check(vkResetCommandPool(device, pool, 0));
                check(vkResetFences(device, 1, &resource.fence));
                VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                ai.commandPool = pool;
                ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                ai.commandBufferCount = 1;
                VkCommandBuffer commands = VK_NULL_HANDLE;
                check(vkAllocateCommandBuffers(device, &ai, &commands));
                VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                check(vkBeginCommandBuffer(commands, &begin));
                VkBufferCopy copy{0, offset, bytes};
                vkCmdCopyBuffer(commands, resource.buffers[0], resource.buffers[1], 1, &copy);
                VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                     0, 1, &barrier, 0, nullptr, 0, nullptr);
                check(vkEndCommandBuffer(commands));
                VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                submit.commandBufferCount = 1;
                submit.pCommandBuffers = &commands;
                check(vkQueueSubmit(queue, 1, &submit, resource.fence));
                check(vkWaitForFences(device, 1, &resource.fence, VK_TRUE, 5'000'000'000ull));
            }
            return Status::Ready; }, [&] { woke.set_value(); }));
        entered.get_future().wait();
        EXPECT_FALSE(job.poll());
        release.set_value();
        auto wake = woke.get_future();
        ASSERT_EQ(wake.wait_for(std::chrono::seconds(20)), std::future_status::ready);
        auto completed = job.poll();
        ASSERT_TRUE(completed);
        ASSERT_EQ(completed->status, Status::Ready);
        ASSERT_EQ(vkGetFenceStatus(device, upload->fence), VK_SUCCESS);
        void* mapped = nullptr;
        check(vkMapMemory(device, upload->memory[1], 0, VK_WHOLE_SIZE, 0, &mapped));
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = upload->memory[1];
        range.size = VK_WHOLE_SIZE;
        check(vkInvalidateMappedMemoryRanges(device, 1, &range));
        const auto* values = static_cast<const uint32_t*>(mapped);
        for (uint32_t i = 0; i < total / sizeof(uint32_t); ++i)
            if (values[i] != (i ^ 0x56780000u)) {
                ADD_FAILURE() << "Incorrect GPU byte stream at word " << i;
                break;
            }
        vkUnmapMemory(device, upload->memory[1]);
    }
    INSTANTIATE_TEST_SUITE_P(Adapters, CollisionPreviewVulkan, ::testing::Values(0u, 1u));
} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
