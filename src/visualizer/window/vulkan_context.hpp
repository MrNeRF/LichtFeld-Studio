/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "config.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#ifdef LFS_VULKAN_VIEWER_ENABLED
#include <vulkan/vulkan.h>
#endif

struct SDL_Window;

namespace lfs::vis {

    class VulkanContext {
    public:
        VulkanContext() = default;
        ~VulkanContext();

        VulkanContext(const VulkanContext&) = delete;
        VulkanContext& operator=(const VulkanContext&) = delete;

        bool init(SDL_Window* window, int framebuffer_width, int framebuffer_height);
        void shutdown();
        void notifyFramebufferResized(int width, int height);

        [[nodiscard]] bool presentBootstrapFrame(float r, float g, float b, float a);
        [[nodiscard]] const std::string& lastError() const { return last_error_; }

#ifdef LFS_VULKAN_VIEWER_ENABLED
        struct Frame {
            uint32_t image_index = 0;
            VkCommandBuffer command_buffer = VK_NULL_HANDLE;
            VkFramebuffer framebuffer = VK_NULL_HANDLE;
            VkImage swapchain_image = VK_NULL_HANDLE;
            VkExtent2D extent{};
        };

#ifdef _WIN32
        using ExternalNativeHandle = void*;
        static constexpr ExternalNativeHandle kInvalidExternalNativeHandle = nullptr;
#else
        using ExternalNativeHandle = int;
        static constexpr ExternalNativeHandle kInvalidExternalNativeHandle = -1;
#endif

        struct ExternalImage {
            VkImage image = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VkExtent2D extent{};
            VkFormat format = VK_FORMAT_UNDEFINED;
            VkDeviceSize allocation_size = 0;
            ExternalNativeHandle native_handle = kInvalidExternalNativeHandle;
        };

        struct ExternalSemaphore {
            VkSemaphore semaphore = VK_NULL_HANDLE;
            std::uint64_t initial_value = 0;
            ExternalNativeHandle native_handle = kInvalidExternalNativeHandle;
        };

        [[nodiscard]] VkInstance instance() const { return instance_; }
        [[nodiscard]] VkPhysicalDevice physicalDevice() const { return physical_device_; }
        [[nodiscard]] VkDevice device() const { return device_; }
        [[nodiscard]] VkSurfaceKHR surface() const { return surface_; }
        [[nodiscard]] VkQueue graphicsQueue() const { return graphics_queue_; }
        [[nodiscard]] VkQueue presentQueue() const { return present_queue_; }
        [[nodiscard]] uint32_t graphicsQueueFamily() const { return graphics_queue_family_; }
        [[nodiscard]] uint32_t presentQueueFamily() const { return present_queue_family_; }
        [[nodiscard]] VkPipelineCache pipelineCache() const { return pipeline_cache_; }
        [[nodiscard]] VkRenderPass renderPass() const { return render_pass_; }
        [[nodiscard]] VkFormat swapchainFormat() const { return swapchain_format_; }
        [[nodiscard]] VkFormat depthStencilFormat() const { return depth_stencil_format_; }
        [[nodiscard]] VkExtent2D swapchainExtent() const { return swapchain_extent_; }
        [[nodiscard]] uint32_t minImageCount() const { return min_image_count_; }
        [[nodiscard]] uint32_t imageCount() const { return static_cast<uint32_t>(swapchain_images_.size()); }
        [[nodiscard]] bool externalMemoryInteropEnabled() const { return external_memory_interop_enabled_; }
        [[nodiscard]] bool externalSemaphoreInteropEnabled() const { return external_semaphore_interop_enabled_; }
        [[nodiscard]] bool externalMemoryDedicatedAllocationEnabled() const {
            return external_memory_dedicated_allocation_enabled_;
        }

        template <typename VkHandle>
        void setDebugObjectName(VkObjectType object_type, VkHandle object, std::string_view name) const {
            setDebugObjectName(object_type, vulkanObjectHandle(object), name);
        }
        void setDebugObjectName(VkObjectType object_type, std::uint64_t object_handle, std::string_view name) const;

        [[nodiscard]] bool beginFrame(const VkClearValue& clear_value, Frame& frame);
        [[nodiscard]] bool endFrame();
        void addFrameTimelineWait(VkSemaphore semaphore,
                                  std::uint64_t value,
                                  VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

        [[nodiscard]] bool createExternalImage(VkExtent2D extent, VkFormat format, ExternalImage& out);
        void destroyExternalImage(ExternalImage& image);
        [[nodiscard]] ExternalNativeHandle releaseExternalImageNativeHandle(ExternalImage& image) const;
        [[nodiscard]] bool createExternalTimelineSemaphore(std::uint64_t initial_value, ExternalSemaphore& out);
        void destroyExternalSemaphore(ExternalSemaphore& semaphore);
        [[nodiscard]] ExternalNativeHandle releaseExternalSemaphoreNativeHandle(ExternalSemaphore& semaphore) const;
        [[nodiscard]] static bool externalNativeHandleValid(ExternalNativeHandle handle);
        void closeExternalNativeHandle(ExternalNativeHandle& handle) const;
        [[nodiscard]] bool transitionImageLayoutImmediate(VkImage image,
                                                          VkImageLayout old_layout,
                                                          VkImageLayout new_layout,
                                                          VkImageAspectFlags aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT);
#endif

    private:
        bool fail(std::string message);

#ifdef LFS_VULKAN_VIEWER_ENABLED
        struct QueueFamilies {
            std::optional<uint32_t> graphics;
            std::optional<uint32_t> present;
            [[nodiscard]] bool complete() const { return graphics.has_value() && present.has_value(); }
        };

        struct SwapchainSupport {
            VkSurfaceCapabilitiesKHR capabilities{};
            std::vector<VkSurfaceFormatKHR> formats;
            std::vector<VkPresentModeKHR> present_modes;
        };

        struct FrameTimelineWait {
            VkSemaphore semaphore = VK_NULL_HANDLE;
            std::uint64_t value = 0;
            VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        };

        bool createInstance();
        bool createSurface(SDL_Window* window);
        bool pickPhysicalDevice();
        bool createDevice();
        bool createSwapchain(int framebuffer_width, int framebuffer_height);
        bool createImageViews();
        bool createRenderPass();
        bool createDepthStencilResources();
        bool createFramebuffers();
        bool createCommandPool();
        bool createCommandBuffers();
        bool createSyncObjects();
        bool createDebugMessenger();
        bool createPipelineCache();
        bool recreateSwapchain();

        void destroyDebugMessenger();
        void saveAndDestroyPipelineCache();
        void destroySwapchain();

        template <typename VkHandle>
        [[nodiscard]] static std::uint64_t vulkanObjectHandle(VkHandle object) {
            if constexpr (std::is_pointer_v<VkHandle>) {
                return reinterpret_cast<std::uint64_t>(object);
            } else {
                return static_cast<std::uint64_t>(object);
            }
        }

        [[nodiscard]] QueueFamilies findQueueFamilies(VkPhysicalDevice device) const;
        [[nodiscard]] bool deviceSupportsSwapchain(VkPhysicalDevice device) const;
        [[nodiscard]] SwapchainSupport querySwapchainSupport(VkPhysicalDevice device) const;
        [[nodiscard]] VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const;
        [[nodiscard]] VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes) const;
        [[nodiscard]] VkExtent2D chooseSwapchainExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                                                       int framebuffer_width,
                                                       int framebuffer_height) const;
        [[nodiscard]] VkFormat chooseDepthStencilFormat() const;
        [[nodiscard]] uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties) const;

        VkInstance instance_ = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
        VkSurfaceKHR surface_ = VK_NULL_HANDLE;
        VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
        VkDevice device_ = VK_NULL_HANDLE;
        VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
        VkQueue graphics_queue_ = VK_NULL_HANDLE;
        VkQueue present_queue_ = VK_NULL_HANDLE;
        uint32_t graphics_queue_family_ = 0;
        uint32_t present_queue_family_ = 0;

        VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
        VkFormat swapchain_format_ = VK_FORMAT_UNDEFINED;
        VkExtent2D swapchain_extent_{};
        VkImageUsageFlags swapchain_image_usage_ = 0;
        uint32_t min_image_count_ = 2;
        std::vector<VkImage> swapchain_images_;
        std::vector<VkImageView> swapchain_image_views_;
        VkRenderPass render_pass_ = VK_NULL_HANDLE;
        VkFormat depth_stencil_format_ = VK_FORMAT_UNDEFINED;
        VkImage depth_stencil_image_ = VK_NULL_HANDLE;
        VkDeviceMemory depth_stencil_memory_ = VK_NULL_HANDLE;
        VkImageView depth_stencil_image_view_ = VK_NULL_HANDLE;
        std::vector<VkFramebuffer> swapchain_framebuffers_;

        VkCommandPool command_pool_ = VK_NULL_HANDLE;
        std::vector<VkCommandBuffer> command_buffers_;
        std::vector<FrameTimelineWait> frame_timeline_waits_;
        VkSemaphore image_available_ = VK_NULL_HANDLE;
        VkSemaphore render_finished_ = VK_NULL_HANDLE;
        VkFence in_flight_ = VK_NULL_HANDLE;

        bool framebuffer_resized_ = false;
        bool frame_active_ = false;
        bool frame_suboptimal_ = false;
        bool debug_utils_enabled_ = false;
        bool validation_enabled_ = false;
        bool instance_external_memory_capabilities_enabled_ = false;
        bool instance_external_semaphore_capabilities_enabled_ = false;
        bool external_memory_interop_enabled_ = false;
        bool external_semaphore_interop_enabled_ = false;
        bool external_memory_dedicated_allocation_enabled_ = false;
        PFN_vkSetDebugUtilsObjectNameEXT vk_set_debug_utils_object_name_ = nullptr;
        uint32_t active_image_index_ = 0;
        int framebuffer_width_ = 0;
        int framebuffer_height_ = 0;
#endif

        std::string last_error_;
    };

} // namespace lfs::vis
