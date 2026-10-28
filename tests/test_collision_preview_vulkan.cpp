/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "rendering/passes/collision_preview_cache.hpp"
#include <array>
#include <functional>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vulkan/vulkan.h>

namespace {
    using Status = lfs::vis::detail::CollisionUploadStatus;
    struct Source {};
    struct Upload {
        std::shared_ptr<const Source> owner;
        VkDevice device = VK_NULL_HANDLE;
        std::array<VkBuffer, 2> buffers{};
        std::array<VkDeviceMemory, 2> memory{};
        VkFence fence = VK_NULL_HANDLE;
        ~Upload() {
            for (size_t i = 0; i < buffers.size(); ++i) {
                if (buffers[i])
                    vkDestroyBuffer(device, buffers[i], nullptr);
                if (memory[i])
                    vkFreeMemory(device, memory[i], nullptr);
            }
            if (fence)
                vkDestroyFence(device, fence, nullptr);
        }
    };
    using Cache = lfs::vis::detail::CollisionPreviewCache<Source, Upload>;

    void check(VkResult result) {
        if (result != VK_SUCCESS)
            throw std::runtime_error("Vulkan call failed: " + std::to_string(int(result)));
    }

    class CollisionPreviewVulkan : public ::testing::TestWithParam<uint32_t> {
    protected:
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physical = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkQueue queue = VK_NULL_HANDLE;
        VkCommandPool pool = VK_NULL_HANDLE;
        VkSemaphore gate = VK_NULL_HANDLE;
        uint32_t family = 0;

        void SetUp() override {
            VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            app.pApplicationName = "Collision preview lifetime tests";
            app.apiVersion = VK_API_VERSION_1_2;
            VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            info.pApplicationInfo = &app;
            if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS)
                GTEST_SKIP() << "Vulkan 1.2 instance unavailable";
            uint32_t count = 0;
            check(vkEnumeratePhysicalDevices(instance, &count, nullptr));
            std::vector<VkPhysicalDevice> devices(count);
            check(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
            for (uint32_t index = 0; index < devices.size(); ++index) {
                if (index != GetParam())
                    continue;
                const auto candidate = devices[index];
                VkPhysicalDeviceProperties properties{};
                vkGetPhysicalDeviceProperties(candidate, &properties);
                VkPhysicalDeviceTimelineSemaphoreFeatures timeline{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
                VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                features.pNext = &timeline;
                vkGetPhysicalDeviceFeatures2(candidate, &features);
                if (properties.apiVersion < VK_API_VERSION_1_2 || !timeline.timelineSemaphore)
                    continue;
                uint32_t n = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, nullptr);
                std::vector<VkQueueFamilyProperties> families(n);
                vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, families.data());
                for (uint32_t i = 0; i < n; ++i)
                    if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                        physical = candidate;
                        family = i;
                        RecordProperty("device", properties.deviceName);
                        break;
                    }
                if (physical)
                    break;
            }
            if (!physical)
                GTEST_SKIP() << "No Vulkan 1.2 graphics device with timeline semaphores";
            const float priority = 1;
            VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            qi.queueFamilyIndex = family;
            qi.queueCount = 1;
            qi.pQueuePriorities = &priority;
            VkPhysicalDeviceTimelineSemaphoreFeatures timeline{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
            timeline.timelineSemaphore = VK_TRUE;
            VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
            di.pNext = &timeline;
            di.queueCreateInfoCount = 1;
            di.pQueueCreateInfos = &qi;
            check(vkCreateDevice(physical, &di, nullptr, &device));
            vkGetDeviceQueue(device, family, 0, &queue);
            VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pi.queueFamilyIndex = family;
            check(vkCreateCommandPool(device, &pi, nullptr, &pool));
            VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            si.pNext = &ti;
            check(vkCreateSemaphore(device, &si, nullptr, &gate));
        }

        void releaseGate() {
            uint64_t value = 0;
            if (gate && vkGetSemaphoreCounterValue(device, gate, &value) == VK_SUCCESS && value == 0) {
                VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
                signal.semaphore = gate;
                signal.value = 1;
                check(vkSignalSemaphore(device, &signal));
            }
        }
        void TearDown() override {
            if (device) {
                releaseGate();
                vkDeviceWaitIdle(device);
                if (gate)
                    vkDestroySemaphore(device, gate, nullptr);
                if (pool)
                    vkDestroyCommandPool(device, pool, nullptr);
                vkDestroyDevice(device, nullptr);
            }
            if (instance)
                vkDestroyInstance(instance, nullptr);
        }

        std::shared_ptr<Upload> allocate(const std::shared_ptr<const Source>& source) {
            auto result = std::make_shared<Upload>();
            result->owner = source;
            result->device = device;
            VkPhysicalDeviceMemoryProperties properties{};
            vkGetPhysicalDeviceMemoryProperties(physical, &properties);
            for (size_t i = 0; i < 2; ++i) {
                VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                bi.size = 256;
                bi.usage = i == 0 ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                check(vkCreateBuffer(device, &bi, nullptr, &result->buffers[i]));
                VkMemoryRequirements requirements{};
                vkGetBufferMemoryRequirements(device, result->buffers[i], &requirements);
                uint32_t type = properties.memoryTypeCount;
                for (uint32_t j = 0; j < properties.memoryTypeCount; ++j)
                    if ((requirements.memoryTypeBits & (1u << j)) &&
                        (properties.memoryTypes[j].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                        type = j;
                        break;
                    }
                if (type == properties.memoryTypeCount)
                    throw std::runtime_error("No host-visible transfer buffer memory");
                VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                ai.allocationSize = requirements.size;
                ai.memoryTypeIndex = type;
                check(vkAllocateMemory(device, &ai, nullptr, &result->memory[i]));
                check(vkBindBufferMemory(device, result->buffers[i], result->memory[i], 0));
            }
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            check(vkCreateFence(device, &fi, nullptr, &result->fence));
            return result;
        }

        static bool retired(const Upload& resource) {
            return vkGetFenceStatus(resource.device, resource.fence) == VK_SUCCESS;
        }
    };

    TEST_P(CollisionPreviewVulkan, PendingFailedUploadSurvivesIdleUntilRealFenceRetires) {
        Cache cache{0};
        // On assertion failure, unblock/retire before cache-owned GPU resources
        // destruct; fixture teardown runs only after local variables destruct.
        struct Drain {
            std::function<void()> run;
            ~Drain() { run(); }
        } drain{[&] {
            releaseGate();
            vkDeviceWaitIdle(device);
        }};
        std::shared_ptr<const Source> source = std::make_shared<Source>();
        std::weak_ptr<Upload> weak;
        auto now = Cache::Time{};
        const auto upload = [&](std::shared_ptr<Upload>& resource) {
            resource = allocate(source);
            weak = resource;
            void* mapped = nullptr;
            check(vkMapMemory(device, resource->memory[0], 0, VK_WHOLE_SIZE, 0, &mapped));
            for (uint32_t i = 0; i < 64; ++i)
                static_cast<uint32_t*>(mapped)[i] = 0x12340000u + i;
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = resource->memory[0];
            range.size = VK_WHOLE_SIZE;
            const auto flushed = vkFlushMappedMemoryRanges(device, 1, &range);
            vkUnmapMemory(device, resource->memory[0]);
            check(flushed);
            VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ai.commandPool = pool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            VkCommandBuffer commands = VK_NULL_HANDLE;
            check(vkAllocateCommandBuffers(device, &ai, &commands));
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            check(vkBeginCommandBuffer(commands, &begin));
            VkBufferCopy copy{0, 0, 256};
            vkCmdCopyBuffer(commands, resource->buffers[0], resource->buffers[1], 1, &copy);
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 1, &barrier, 0, nullptr, 0, nullptr);
            check(vkEndCommandBuffer(commands));
            const uint64_t value = 1;
            VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            timeline.waitSemaphoreValueCount = 1;
            timeline.pWaitSemaphoreValues = &value;
            const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.pNext = &timeline;
            submit.waitSemaphoreCount = 1;
            submit.pWaitSemaphores = &gate;
            submit.pWaitDstStageMask = &stage;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &commands;
            check(vkQueueSubmit(queue, 1, &submit, resource->fence));
            return Status::PendingFailure;
        };
        EXPECT_FALSE(cache.request(source, 256, 1, now, upload, retired));
        ASSERT_FALSE(weak.expired());
        EXPECT_EQ(vkGetFenceStatus(device, weak.lock()->fence), VK_NOT_READY);
        now += std::chrono::seconds(100);
        source.reset();
        cache.maintain(now, retired);
        EXPECT_FALSE(weak.expired());
        EXPECT_FALSE(cache.retryDue(1, now));
        EXPECT_EQ(cache.stats().pending_bytes, 256u);
        releaseGate();
        auto completed = weak.lock();
        ASSERT_TRUE(completed);
        ASSERT_EQ(vkWaitForFences(device, 1, &completed->fence, VK_TRUE, 5'000'000'000ull), VK_SUCCESS);
        void* mapped = nullptr;
        check(vkMapMemory(device, completed->memory[1], 0, VK_WHOLE_SIZE, 0, &mapped));
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = completed->memory[1];
        range.size = VK_WHOLE_SIZE;
        check(vkInvalidateMappedMemoryRanges(device, 1, &range));
        for (uint32_t i = 0; i < 64; ++i)
            EXPECT_EQ(static_cast<uint32_t*>(mapped)[i], 0x12340000u + i);
        vkUnmapMemory(device, completed->memory[1]);
        completed.reset();
        cache.maintain(now, retired);
        EXPECT_TRUE(weak.expired());
        EXPECT_EQ(cache.stats().pending_bytes, 0u);
        EXPECT_FALSE(cache.nextDeadline(1, now));
    }

    // Exercise separate integrated/discrete adapters where available. A machine
    // with only one adapter explicitly skips the second case.
    INSTANTIATE_TEST_SUITE_P(Adapters, CollisionPreviewVulkan, ::testing::Values(0u, 1u));
} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
