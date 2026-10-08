/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "core/logger.hpp"
#include "gui/rmlui/rmlui_vk_backend.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <mutex>
#include <string>
#include <vector>

class RenderInterfaceVKTestAccess {
public:
    using Pool = RenderInterface_VK::MemoryPool;
    using Allocation = decltype(RenderInterface_VK::geometry_handle_t{}.m_p_vertex_allocation);
    static auto decode(const std::filesystem::path& path) {
        return RenderInterface_VK::DecodePreviewTexture(path, 256, true);
    }
};

namespace {

    class WarningCapture {
    public:
        WarningCapture()
            : token_(lfs::core::Logger::get().add_log_handler(
                  [this](const lfs::core::LogLevel level, const lfs::core::SourceSite&, const std::string_view message) {
                      if (level != lfs::core::LogLevel::Warn)
                          return;
                      std::scoped_lock lock(mutex_);
                      messages_.emplace_back(message);
                  })) {}
        ~WarningCapture() { lfs::core::Logger::get().remove_log_handler(token_); }

        [[nodiscard]] std::size_t count() const {
            std::scoped_lock lock(mutex_);
            return messages_.size();
        }

    private:
        lfs::core::LogHandlerToken token_;
        mutable std::mutex mutex_;
        std::vector<std::string> messages_;
    };

    class PreviewDecodeTest : public ::testing::Test {
    protected:
        void SetUp() override {
            directory_ = std::filesystem::temp_directory_path() /
                         ("lfs_preview_decode_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                          ::testing::UnitTest::GetInstance()->current_test_info()->name());
            std::filesystem::create_directories(directory_);
        }
        void TearDown() override { std::filesystem::remove_all(directory_); }

        std::filesystem::path directory_;
    };

    // Catches a catalog entry whose file was deleted flooding the log with a warning and stack on every start.
    TEST_F(PreviewDecodeTest, MissingProjectHasNoPreviewAndNoWarning) {
        WarningCapture warnings;
        const auto result = RenderInterfaceVKTestAccess::decode(directory_ / "deleted.licht");
        EXPECT_TRUE(result.pixels.empty());
        EXPECT_EQ(warnings.count(), 0u);
    }

    // Catches the missing-file case being silenced so broadly that unreadable projects go unreported.
    TEST_F(PreviewDecodeTest, DamagedProjectStillWarns) {
        const auto damaged = directory_ / "damaged.licht";
        std::ofstream(damaged, std::ios::binary) << "not a project";
        WarningCapture warnings;
        const auto result = RenderInterfaceVKTestAccess::decode(damaged);
        EXPECT_TRUE(result.pixels.empty());
        EXPECT_EQ(warnings.count(), 1u);
    }

} // namespace

namespace {
    class RmlUiGeometryPoolTest : public ::testing::Test {
    protected:
        void SetUp() override {
            VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            app.apiVersion = VK_API_VERSION_1_3;
            VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            instance_info.pApplicationInfo = &app;
            if (vkCreateInstance(&instance_info, nullptr, &instance_) != VK_SUCCESS)
                GTEST_SKIP() << "Vulkan instance unavailable";
            uint32_t count = 0;
            if (vkEnumeratePhysicalDevices(instance_, &count, nullptr) != VK_SUCCESS || count == 0)
                GTEST_SKIP() << "No Vulkan device";
            std::vector<VkPhysicalDevice> devices(count);
            ASSERT_EQ(vkEnumeratePhysicalDevices(instance_, &count, devices.data()), VK_SUCCESS);
            VkPhysicalDevice physical = devices.front();
            const float priority = 1.0f;
            VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queue.queueCount = 1;
            queue.pQueuePriorities = &priority;
            VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
            device_info.queueCreateInfoCount = 1;
            device_info.pQueueCreateInfos = &queue;
            ASSERT_EQ(vkCreateDevice(physical, &device_info, nullptr, &device_), VK_SUCCESS);
            VmaAllocatorCreateInfo allocator_info{};
            allocator_info.instance = instance_;
            allocator_info.physicalDevice = physical;
            allocator_info.device = device_;
            allocator_info.vulkanApiVersion = VK_API_VERSION_1_3;
            ASSERT_EQ(vmaCreateAllocator(&allocator_info, &allocator_), VK_SUCCESS);
            ASSERT_TRUE(pool_.Initialize(256, 64, allocator_, device_));
            initialized_ = true;
        }
        void TearDown() override {
            if (initialized_) {
                for (auto allocation : allocations_)
                    pool_.Free_Allocation(allocation);
                pool_.Shutdown();
            }
            if (allocator_)
                vmaDestroyAllocator(allocator_);
            if (device_)
                vkDestroyDevice(device_, nullptr);
            if (instance_)
                vkDestroyInstance(instance_, nullptr);
        }
        RenderInterfaceVKTestAccess::Pool pool_;
        std::vector<RenderInterfaceVKTestAccess::Allocation> allocations_;
        VkInstance instance_{};
        VkDevice device_{};
        VmaAllocator allocator_{};
        bool initialized_ = false;
    };

    TEST_F(RmlUiGeometryPoolTest, ExhaustionPreservesLiveAllocationsAndReusesFreedStorage) {
        std::vector<VkDescriptorBufferInfo> buffers;
        std::vector<unsigned char*> data;
        for (int i = 0; i < 8; ++i) {
            allocations_.emplace_back();
            VkDescriptorBufferInfo buffer{};
            void* mapped = nullptr;
            ASSERT_TRUE(pool_.Alloc_GeneralBuffer(128, &mapped, &buffer, &allocations_.back()));
            ASSERT_NE(mapped, nullptr);
            ASSERT_LT(buffer.offset, 256u) << "An exhausted pool must not return VMA's invalid offset";
            ASSERT_EQ(buffer.offset % 64, 0u);
            buffers.push_back(buffer);
            data.push_back(static_cast<unsigned char*>(mapped));
            std::fill_n(data.back(), 128, static_cast<unsigned char>(i + 1));
        }
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 128; ++j)
                ASSERT_EQ(data[i][j], i + 1) << "Growth overwrote live geometry";
        for (auto allocation : allocations_)
            pool_.Free_Allocation(allocation);
        allocations_.clear();
        allocations_.emplace_back();
        VkDescriptorBufferInfo reused{};
        void* mapped = nullptr;
        ASSERT_TRUE(pool_.Alloc_GeneralBuffer(256, &mapped, &reused, &allocations_.back()));
        EXPECT_EQ(reused.buffer, buffers.front().buffer);
        EXPECT_EQ(reused.offset, 0u);
    }
    TEST_F(RmlUiGeometryPoolTest, OversizedGeometryGrowsOnceWithoutMovingLiveData) {
        allocations_.emplace_back();
        VkDescriptorBufferInfo first{};
        void* original = nullptr;
        ASSERT_TRUE(pool_.Alloc_VertexBuffer(4, 20, &original, &first, &allocations_.back()));
        std::fill_n(static_cast<unsigned char*>(original), 80, 42);
        VmaTotalStatistics stats{};
        vmaCalculateStatistics(allocator_, &stats);
        EXPECT_EQ(stats.total.statistics.allocationCount, 1u);
        VkBuffer overflow_buffer{};
        for (int frame = 0; frame < 100; ++frame) {
            allocations_.emplace_back();
            VkDescriptorBufferInfo overflow{};
            void* mapped = nullptr;
            ASSERT_TRUE(pool_.Alloc_IndexBuffer(256, 4, &mapped, &overflow, &allocations_.back()));
            ASSERT_EQ(overflow.offset, 0u);
            EXPECT_EQ(overflow.range, 1024u);
            EXPECT_NE(overflow.buffer, first.buffer);
            if (frame == 0)
                overflow_buffer = overflow.buffer;
            EXPECT_EQ(overflow.buffer, overflow_buffer);
            std::fill_n(static_cast<unsigned char*>(mapped), 1024, 7);
            for (int i = 0; i < 80; ++i)
                ASSERT_EQ(static_cast<unsigned char*>(original)[i], 42);
            pool_.Free_Allocation(allocations_.back());
            allocations_.pop_back();
        }
        vmaCalculateStatistics(allocator_, &stats);
        EXPECT_EQ(stats.total.statistics.allocationCount, 2u);
    }

} // namespace
