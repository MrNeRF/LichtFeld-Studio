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
    static RenderInterface_VK::frame_state_t frame(const RenderInterface_VK& backend) {
        return backend.CaptureFrameState();
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

namespace {
    // Drives the real RenderInterface_VK layer path the way RmlUi's box-shadow callback does:
    // a context drawn at a negative offset, a texture-space scissor, PushLayer, SaveLayerAsTexture.
    class RmlUiLayerFrameTest : public ::testing::Test {
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
            physical_ = devices.front();

            uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, families.data());
            if (families.empty() || (families.front().queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0)
                GTEST_SKIP() << "First queue family is not graphics";

            VkFormatProperties depth_properties{};
            vkGetPhysicalDeviceFormatProperties(physical_, kDepthFormat, &depth_properties);
            if ((depth_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) == 0)
                GTEST_SKIP() << "Depth/stencil format unsupported";

            const float priority = 1.0f;
            VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queue.queueCount = 1;
            queue.pQueuePriorities = &priority;
            VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
            features13.dynamicRendering = VK_TRUE;
            features13.synchronization2 = VK_TRUE;
            VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
            device_info.pNext = &features13;
            device_info.queueCreateInfoCount = 1;
            device_info.pQueueCreateInfos = &queue;
            ASSERT_EQ(vkCreateDevice(physical_, &device_info, nullptr, &device_), VK_SUCCESS);
            vkGetDeviceQueue(device_, 0, 0, &queue_);

            VmaAllocatorCreateInfo allocator_info{};
            allocator_info.instance = instance_;
            allocator_info.physicalDevice = physical_;
            allocator_info.device = device_;
            allocator_info.vulkanApiVersion = VK_API_VERSION_1_3;
            ASSERT_EQ(vmaCreateAllocator(&allocator_info, &allocator_), VK_SUCCESS);

            VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            ASSERT_EQ(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_), VK_SUCCESS);
            VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocate.commandPool = command_pool_;
            allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocate.commandBufferCount = 1;
            ASSERT_EQ(vkAllocateCommandBuffers(device_, &allocate, &command_buffer_), VK_SUCCESS);
        }

        void TearDown() override {
            if (device_) {
                vkDeviceWaitIdle(device_);
                for (auto& image : images_) {
                    if (image.view)
                        vkDestroyImageView(device_, image.view, nullptr);
                    if (image.image)
                        vmaDestroyImage(allocator_, image.image, image.allocation);
                }
                if (command_pool_)
                    vkDestroyCommandPool(device_, command_pool_, nullptr);
            }
            if (allocator_)
                vmaDestroyAllocator(allocator_);
            if (device_)
                vkDestroyDevice(device_, nullptr);
            if (instance_)
                vkDestroyInstance(instance_, nullptr);
        }

        struct image_t {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
        };

        image_t CreateImage(VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VkExtent2D extent) {
            image_t result;
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = format;
            info.extent = {extent.width, extent.height, 1};
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = usage;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo allocation_info{};
            allocation_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            EXPECT_EQ(vmaCreateImage(allocator_, &info, &allocation_info, &result.image, &result.allocation, nullptr), VK_SUCCESS);
            images_.push_back(result);

            VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            view.image = result.image;
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = format;
            view.subresourceRange = {aspect, 0, 1, 0, 1};
            EXPECT_EQ(vkCreateImageView(device_, &view, nullptr, &result.view), VK_SUCCESS);
            images_.back().view = result.view;
            return result;
        }

        void TransitionForTest(VkImage image, VkImageAspectFlags aspect, VkImageLayout layout) {
            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = layout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange = {aspect, 0, 1, 0, 1};
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);
        }

        // Records the frame the app opens before RmlUi (transitions, a dynamic rendering pass) and starts the backend.
        bool BeginFrame(RenderInterface_VK& backend, VkExtent2D extent, const image_t& color, const image_t& depth) {
            if (color.view == VK_NULL_HANDLE || depth.view == VK_NULL_HANDLE)
                return false;
            const RenderInterface_VK::ExternalContext context{instance_, physical_, device_, VK_NULL_HANDLE, queue_, 0,
                                                              kColorFormat, kDepthFormat, extent, false};
            if (!backend.InitializeExternal(context))
                return false;
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(command_buffer_, &begin) != VK_SUCCESS)
                return false;
            TransitionForTest(color.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            TransitionForTest(depth.image, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            VkRenderingAttachmentInfo color_attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            color_attachment.imageView = color.view;
            color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            VkRenderingAttachmentInfo depth_attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            depth_attachment.imageView = depth.view;
            depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_attachment.clearValue.depthStencil = {1.0f, 0};
            VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
            rendering.renderArea = {{0, 0}, extent};
            rendering.layerCount = 1;
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachments = &color_attachment;
            rendering.pDepthAttachment = &depth_attachment;
            rendering.pStencilAttachment = &depth_attachment;
            vkCmdBeginRendering(command_buffer_, &rendering);
            backend.BeginExternalFrame(command_buffer_, extent, color.image, color.view, depth.view, 0);
            return true;
        }

        // Closes the frame, submits it, releases the saved texture and shuts the backend down.
        void EndFrame(RenderInterface_VK& backend, Rml::TextureHandle saved_texture) {
            backend.EndExternalFrame();
            vkCmdEndRendering(command_buffer_);
            ASSERT_EQ(vkEndCommandBuffer(command_buffer_), VK_SUCCESS);
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command_buffer_;
            ASSERT_EQ(vkQueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE), VK_SUCCESS);
            vkQueueWaitIdle(queue_);
            if (saved_texture != 0)
                backend.ReleaseTexture(saved_texture);
            backend.ShutdownExternal();
        }

        static constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
        static constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;

        VkInstance instance_{};
        VkPhysicalDevice physical_{};
        VkDevice device_{};
        VkQueue queue_{};
        VmaAllocator allocator_{};
        VkCommandPool command_pool_{};
        VkCommandBuffer command_buffer_{};
        std::vector<image_t> images_;
    };

    // Regression: the Video Extractor's focused button drew as a solid white box. RmlUi's box-shadow callback
    // texture is captured in texture space while the context sits at a negative offset (its document origin is
    // above the window). The saved texture region then fell outside the layer, SaveLayerAsTexture returned no
    // texture, and the shadow geometry was drawn untextured.
    TEST_F(RmlUiLayerFrameTest, CallbackTextureInsideOffsetContextIsSaved) {
        const VkExtent2D extent{1280, 720};
        const auto color = CreateImage(kColorFormat,
                                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                       VK_IMAGE_ASPECT_COLOR_BIT, extent);
        const auto depth = CreateImage(kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                       VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, extent);
        RenderInterface_VK backend;
        ASSERT_TRUE(BeginFrame(backend, extent, color, depth));

        // Panel content origin in window space: x = 191, y = -38 (the same offset as the failing case).
        backend.SetContextOffset(191.0f, -38.0f);
        backend.SetContextClipRect(191.0f, 0.0f, 1409.0f, 720.0f);

        // RmlUi box-shadow callback: ResetState, then a texture-space scissor, then PushLayer.
        const Rml::Rectanglei texture_region = Rml::Rectanglei::FromSize({40, 24});
        backend.EnableScissorRegion(false);
        backend.EnableScissorRegion(true);
        backend.SetScissorRegion(texture_region);
        const Rml::LayerHandle layer = backend.PushLayer();
        ASSERT_NE(layer, 0u);
        backend.SetScissorRegion(texture_region);
        const Rml::TextureHandle shadow_texture = backend.SaveLayerAsTexture();
        EXPECT_NE(shadow_texture, 0u) << "box-shadow texture was not saved; the element would draw as a white quad";
        backend.PopLayer();

        EndFrame(backend, shadow_texture);
    }

    // Regression guard for cached panel captures: PushContextLayer must keep the context offset, clip and
    // cache-capture area the manager set, while an RmlUi PushLayer inside it uses render-target coordinates.
    TEST_F(RmlUiLayerFrameTest, ContextLayerKeepsContextFrameAndNeutralLayerDoesNot) {
        const VkExtent2D extent{1280, 720};
        const auto color = CreateImage(kColorFormat,
                                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                       VK_IMAGE_ASPECT_COLOR_BIT, extent);
        const auto depth = CreateImage(kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                       VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, extent);
        RenderInterface_VK backend;
        ASSERT_TRUE(BeginFrame(backend, extent, color, depth));

        // Same sequence as RmlUIManager's cached panel refresh.
        backend.BeginCacheCapture(191, 0, 1218, 720);
        backend.SetContextOffset(191.0f, -38.0f);
        backend.SetContextClipRect(191.0f, 0.0f, 1409.0f, 720.0f);
        const Rml::LayerHandle context_layer = backend.PushContextLayer();
        ASSERT_NE(context_layer, 0u);
        {
            const auto frame = RenderInterfaceVKTestAccess::frame(backend);
            EXPECT_EQ(frame.context_offset.x, 191.0f);
            EXPECT_EQ(frame.context_offset.y, -38.0f);
            EXPECT_TRUE(frame.context_clip_enabled);
            EXPECT_EQ(frame.context_clip_scissor.offset.x, 191);
            EXPECT_EQ(frame.context_clip_scissor.extent.width, 1089u); // clamped to the 1280px target
            EXPECT_TRUE(frame.cache_capture_active);
        }

        // A nested RmlUi layer is in render-target coordinates: no offset, no context clip, no cache clamp.
        const Rml::LayerHandle neutral_layer = backend.PushLayer();
        ASSERT_NE(neutral_layer, 0u);
        {
            const auto frame = RenderInterfaceVKTestAccess::frame(backend);
            EXPECT_EQ(frame.context_offset.x, 0.0f);
            EXPECT_EQ(frame.context_offset.y, 0.0f);
            EXPECT_FALSE(frame.context_clip_enabled);
            EXPECT_FALSE(frame.cache_capture_active);
        }
        backend.PopLayer();

        // Popping the neutral layer restores the context frame exactly.
        {
            const auto frame = RenderInterfaceVKTestAccess::frame(backend);
            EXPECT_EQ(frame.context_offset.x, 191.0f);
            EXPECT_EQ(frame.context_offset.y, -38.0f);
            EXPECT_TRUE(frame.context_clip_enabled);
            EXPECT_TRUE(frame.cache_capture_active);
        }
        const Rml::TextureHandle panel_texture =
            backend.SaveLayerRegionAsTexture(VkRect2D{{191, 0}, {40, 24}}, {});
        EXPECT_NE(panel_texture, 0u);
        backend.PopLayer();
        backend.EndCacheCapture();

        EndFrame(backend, panel_texture);
    }

} // namespace
