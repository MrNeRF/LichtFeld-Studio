/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "rendering/passes/collision_preview_cache.hpp"
#include "rendering/passes/collision_preview_upload.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <random>
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

        std::shared_ptr<Upload> allocate(const std::shared_ptr<const Source>& source, size_t bytes = 256) {
            auto result = std::make_shared<Upload>();
            result->owner = source;
            result->device = device;
            VkPhysicalDeviceMemoryProperties properties{};
            vkGetPhysicalDeviceMemoryProperties(physical, &properties);
            for (size_t i = 0; i < 2; ++i) {
                VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                bi.size = bytes;
                bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | (i == 0 ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : VK_BUFFER_USAGE_TRANSFER_DST_BIT);
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

        void submitCopy(Upload& resource, uint32_t pattern) {
            void* mapped = nullptr;
            check(vkMapMemory(device, resource.memory[0], 0, VK_WHOLE_SIZE, 0, &mapped));
            for (uint32_t i = 0; i < 64; ++i)
                static_cast<uint32_t*>(mapped)[i] = pattern + i;
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = resource.memory[0];
            range.size = VK_WHOLE_SIZE;
            const auto flushed = vkFlushMappedMemoryRanges(device, 1, &range);
            vkUnmapMemory(device, resource.memory[0]);
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
            vkCmdCopyBuffer(commands, resource.buffers[0], resource.buffers[1], 1, &copy);
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
            check(vkQueueSubmit(queue, 1, &submit, resource.fence));
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
            submitCopy(*resource, 0x12340000u);
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

    TEST_P(CollisionPreviewVulkan, WorkerUploadPublishesOnlyAfterRealGpuFenceAndReadback) {
        Cache cache{1024};
        lfs::vis::detail::CollisionPreviewUploadJob<Upload> job;
        struct Drain {
            std::function<void()> run;
            ~Drain() { run(); }
        } drain{[&] { releaseGate(); vkDeviceWaitIdle(device); }};
        auto source = std::make_shared<const Source>();
        std::promise<void> submitted;
        auto submitted_future = submitted.get_future();
        auto now = Cache::Time{};
        const auto start = [&](std::shared_ptr<Upload>& resource) {
            resource = allocate(source);
            job.start(resource, [&](Upload& upload, std::stop_token) {
                try {
                    submitCopy(upload, 0x56780000u);
                    submitted.set_value();
                } catch (...) {
                    submitted.set_exception(std::current_exception());
                    return Status::RetryableFailure;
                }
                return vkWaitForFences(device, 1, &upload.fence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS
                           ? Status::Ready
                           : Status::PendingFailure;
            });
            return Status::Uploading;
        };
        EXPECT_FALSE(cache.request(source, 256, 1, now, start, retired));
        submitted_future.get();
        EXPECT_FALSE(job.poll());
        EXPECT_FALSE(cache.ready(source.get()));
        cache.maintain(now, retired, 1);
        EXPECT_EQ(cache.stats().pending_bytes, 256u);
        EXPECT_EQ(cache.stats().failures, 0u);
        releaseGate();
        auto result = job.finish();
        ASSERT_TRUE(result);
        ASSERT_EQ(result->status, Status::Ready);
        cache.completeUpload(source.get(), result->status, now);
        EXPECT_EQ(cache.ready(source.get()), result->resource);
        void* mapped = nullptr;
        check(vkMapMemory(device, result->resource->memory[1], 0, VK_WHOLE_SIZE, 0, &mapped));
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = result->resource->memory[1];
        range.size = VK_WHOLE_SIZE;
        check(vkInvalidateMappedMemoryRanges(device, 1, &range));
        for (uint32_t i = 0; i < 64; ++i)
            EXPECT_EQ(static_cast<uint32_t*>(mapped)[i], 0x56780000u + i);
        vkUnmapMemory(device, result->resource->memory[1]);
    }

#ifdef LFS_COLLISION_QUERY_SPIRV
    TEST_P(CollisionPreviewVulkan, TerminalRegionCacheMatchesIndependentRootTraversal) {
        struct Pipeline {
            VkDevice device;
            VkShaderModule shader{};
            VkDescriptorSetLayout descriptors{};
            VkDescriptorPool pool{};
            VkPipelineLayout layout{};
            VkPipeline pipeline{};
            ~Pipeline() {
                if (pipeline)
                    vkDestroyPipeline(device, pipeline, nullptr);
                if (layout)
                    vkDestroyPipelineLayout(device, layout, nullptr);
                if (pool)
                    vkDestroyDescriptorPool(device, pool, nullptr);
                if (descriptors)
                    vkDestroyDescriptorSetLayout(device, descriptors, nullptr);
                if (shader)
                    vkDestroyShaderModule(device, shader, nullptr);
            }
        } gpu{device};
        std::ifstream file(LFS_COLLISION_QUERY_SPIRV, std::ios::binary | std::ios::ate);
        ASSERT_TRUE(file);
        std::vector<uint32_t> code(size_t(file.tellg()) / 4);
        file.seekg(0);
        file.read(reinterpret_cast<char*>(code.data()), code.size() * 4);
        VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module.codeSize = code.size() * 4;
        module.pCode = code.data();
        check(vkCreateShaderModule(device, &module, nullptr, &gpu.shader));
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        for (uint32_t i = 0; i < 2; ++i)
            bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ds{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ds.bindingCount = 2;
        ds.pBindings = bindings.data();
        check(vkCreateDescriptorSetLayout(device, &ds, nullptr, &gpu.descriptors));
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &gpu.descriptors;
        check(vkCreatePipelineLayout(device, &pl, nullptr, &gpu.layout));
        VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cp.layout = gpu.layout;
        cp.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp.stage.module = gpu.shader;
        cp.stage.pName = "main";
        check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp, nullptr, &gpu.pipeline));
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
        VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dp.maxSets = 1;
        dp.poolSizeCount = 1;
        dp.pPoolSizes = &ps;
        check(vkCreateDescriptorPool(device, &dp, nullptr, &gpu.pool));
        VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        da.descriptorPool = gpu.pool;
        da.descriptorSetCount = 1;
        da.pSetLayouts = &gpu.descriptors;
        VkDescriptorSet set{};
        check(vkAllocateDescriptorSets(device, &da, &set));
        auto buffers = allocate(std::make_shared<Source>(), 1024 * 1024);
        std::array<VkDescriptorBufferInfo, 2> bi{};
        std::array<VkWriteDescriptorSet, 2> writes{};
        for (uint32_t i = 0; i < 2; ++i) {
            bi[i] = {buffers->buffers[i], 0, VK_WHOLE_SIZE};
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(device, 2, writes.data(), 0, nullptr);
        std::mt19937 random(0x5eed);
        constexpr size_t query_count = 16384;
        std::vector<std::array<int32_t, 12>> probes(query_count), expected(query_count);
        for (int trial = 0; trial < 32; ++trial) {
            const int edge = trial >= 28 ? 1 << (18 + trial - 28) : 4 << (trial % 5);
            const std::array<int32_t, 4> dimensions{edge - (trial % 4), edge - ((trial / 2) % 4), edge - ((trial / 3) % 4), edge};
            std::vector<uint32_t> words(1);
            auto build = [&](auto&& self, uint32_t index, std::array<int, 3> lo, int size) -> void {
                auto kind = random() % 5;
                if (trial >= 28) {
                    // A sparse path reaches the maximum supported depth
                    // without allocating an exponentially large dense tree.
                    if (lo != std::array<int, 3>{0, 0, 0}) {
                        words[index] = random() % 2;
                        return;
                    }
                    if (size > 4)
                        kind = 2;
                }
                if (trial < 2 || kind < 2) {
                    words[index] = trial < 2 ? trial : kind;
                    return;
                }
                if (size == 4) {
                    const auto base = uint32_t(words.size());
                    words[index] = 0x80000000u | base;
                    uint64_t mask = uint64_t(random()) | (uint64_t(random()) << 32);
                    for (int bit = 0; bit < 64; ++bit)
                        if (lo[0] + (bit & 3) >= dimensions[0] || lo[1] + ((bit >> 2) & 3) >= dimensions[1] || lo[2] + (bit >> 4) >= dimensions[2])
                            mask &= ~(uint64_t(1) << bit);
                    words.push_back(uint32_t(mask));
                    words.push_back(uint32_t(mask >> 32));
                    return;
                }
                const auto base = uint32_t(words.size());
                words[index] = base + 2;
                words.resize(base + 8);
                for (int c = 0; c < 8; ++c)
                    self(self, base + c, {lo[0] + (c & 1) * (size / 2), lo[1] + ((c >> 1) & 1) * (size / 2), lo[2] + ((c >> 2) & 1) * (size / 2)}, size / 2);
            };
            build(build, 0, {0, 0, 0}, edge);
            for (size_t i = 0; i < query_count; ++i) {
                auto& q = probes[i];
                q.fill(-999);
                for (int a = 0; a < 3; ++a)
                    q[a] = i % 7 == 0 && i % 256 != 0 ? probes[i - 1][a]
                           : i % 13 == 0              ? random() % std::min(dimensions[a], 4)
                                                      : random() % dimensions[a];
                expected[i] = q;
                std::array<int, 3> lo{};
                int size = edge;
                auto word = words[0];
                while (word > 1 && !(word & 0x80000000u)) {
                    size /= 2;
                    int child = 0;
                    for (int a = 0; a < 3; ++a)
                        if (q[a] >= lo[a] + size) {
                            child |= 1 << a;
                            lo[a] += size;
                        }
                    word = words[word - 2 + child];
                }
                bool hit = word == 1;
                if (word & 0x80000000u) {
                    auto base = word & 0x7fffffffu;
                    auto bit = (q[0] & 3) + 4 * (q[1] & 3) + 16 * (q[2] & 3);
                    hit = ((uint64_t(words[base]) | (uint64_t(words[base + 1]) << 32)) >> bit) & 1;
                    for (int a = 0; a < 3; ++a) {
                        expected[i][4 + a] = q[a];
                        expected[i][8 + a] = q[a] + 1;
                    }
                } else
                    for (int a = 0; a < 3; ++a) {
                        expected[i][4 + a] = lo[a];
                        expected[i][8 + a] = std::min(lo[a] + size, dimensions[a]);
                    }
                expected[i][7] = hit;
                expected[i][11] = 0;
            }
            void* mapped{};
            for (int b = 0; b < 2; ++b) {
                check(vkMapMemory(device, buffers->memory[b], 0, VK_WHOLE_SIZE, 0, &mapped));
                if (b == 0) {
                    memcpy(mapped, dimensions.data(), 16);
                    memcpy(static_cast<char*>(mapped) + 16, words.data(), words.size() * 4);
                } else
                    memcpy(mapped, probes.data(), probes.size() * sizeof(probes[0]));
                VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
                range.memory = buffers->memory[b];
                range.size = VK_WHOLE_SIZE;
                check(vkFlushMappedMemoryRanges(device, 1, &range));
                vkUnmapMemory(device, buffers->memory[b]);
            }
            VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ca.commandPool = pool;
            ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ca.commandBufferCount = 1;
            VkCommandBuffer cmd{};
            check(vkAllocateCommandBuffers(device, &ca, &cmd));
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            check(vkBeginCommandBuffer(cmd, &begin));
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gpu.layout, 0, 1, &set, 0, nullptr);
            vkCmdDispatch(cmd, 64, 1, 1);
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            check(vkEndCommandBuffer(cmd));
            check(vkResetFences(device, 1, &buffers->fence));
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &cmd;
            check(vkQueueSubmit(queue, 1, &submit, buffers->fence));
            check(vkWaitForFences(device, 1, &buffers->fence, VK_TRUE, 10'000'000'000ull));
            check(vkMapMemory(device, buffers->memory[1], 0, VK_WHOLE_SIZE, 0, &mapped));
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = buffers->memory[1];
            range.size = VK_WHOLE_SIZE;
            check(vkInvalidateMappedMemoryRanges(device, 1, &range));
            bool identical = memcmp(mapped, expected.data(), expected.size() * sizeof(expected[0])) == 0;
            vkUnmapMemory(device, buffers->memory[1]);
            vkFreeCommandBuffers(device, pool, 1, &cmd);
            EXPECT_TRUE(identical) << "trial=" << trial << " edge=" << edge;
        }
    }
#endif

    // Exercise separate integrated/discrete adapters where available. A machine
    // with only one adapter explicitly skips the second case.
    INSTANTIATE_TEST_SUITE_P(Adapters, CollisionPreviewVulkan, ::testing::Values(0u, 1u));
} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
