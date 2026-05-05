/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#ifdef LFS_VULKAN_VIEWER_ENABLED
#include "vulkan_image_barrier_tracker.hpp"
#include <vulkan/vulkan.h>
#ifndef VMA_STATIC_VULKAN_FUNCTIONS
#define VMA_STATIC_VULKAN_FUNCTIONS 1
#endif
#ifndef VMA_DYNAMIC_VULKAN_FUNCTIONS
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#endif
#include <vk_mem_alloc.h>
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
            std::size_t frame_slot = 0;
            VkCommandBuffer command_buffer = VK_NULL_HANDLE;
            VkImage swapchain_image = VK_NULL_HANDLE;
            VkImageView swapchain_image_view = VK_NULL_HANDLE;
            VkImageView depth_stencil_image_view = VK_NULL_HANDLE;
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

        struct ExternalBuffer {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VkDeviceSize size = 0;
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
        [[nodiscard]] VmaAllocator allocator() const { return allocator_; }
        [[nodiscard]] VkPipelineCache pipelineCache() const { return pipeline_cache_; }
        [[nodiscard]] VkFormat swapchainFormat() const { return swapchain_format_; }
        [[nodiscard]] VkColorSpaceKHR swapchainColorSpace() const { return swapchain_color_space_; }
        [[nodiscard]] bool hasHdr() const noexcept { return has_hdr_; }
        [[nodiscard]] VkFormat depthStencilFormat() const { return depth_stencil_format_; }
        [[nodiscard]] VkImageView depthStencilImageView() const {
            return active_image_index_ < depth_stencil_resources_.size()
                       ? depth_stencil_resources_[active_image_index_].view
                       : VK_NULL_HANDLE;
        }
        [[nodiscard]] VkExtent2D swapchainExtent() const { return swapchain_extent_; }
        [[nodiscard]] uint32_t minImageCount() const { return min_image_count_; }
        [[nodiscard]] uint32_t imageCount() const { return static_cast<uint32_t>(swapchain_images_.size()); }
        [[nodiscard]] std::size_t framesInFlight() const { return kFramesInFlight; }
        [[nodiscard]] std::size_t currentFrameSlot() const { return frame_index_; }
        [[nodiscard]] bool externalMemoryInteropEnabled() const { return external_memory_interop_enabled_; }
        [[nodiscard]] bool externalSemaphoreInteropEnabled() const { return external_semaphore_interop_enabled_; }
        [[nodiscard]] VulkanImageBarrierTracker& imageBarriers() { return image_barriers_; }
        [[nodiscard]] bool hasPushDescriptor() const { return has_push_descriptor_; }
        [[nodiscard]] PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSet() const { return vk_cmd_push_descriptor_set_; }
        [[nodiscard]] bool hasShaderObject() const { return has_shader_object_; }
        [[nodiscard]] bool hasExtendedDynamicState3() const { return has_extended_dynamic_state3_; }
        [[nodiscard]] bool hasCooperativeMatrix() const { return has_cooperative_matrix_; }
        [[nodiscard]] bool hasHostImageCopy() const { return has_host_image_copy_; }
        [[nodiscard]] bool hasDescriptorIndexing() const { return has_descriptor_indexing_; }
        // Optional dedicated async-compute queue. When hasDedicatedComputeQueue() is
        // true, computeQueue() / computeQueueFamily() are distinct from graphicsQueue();
        // otherwise they alias the graphics queue and submitting on either is equivalent.
        [[nodiscard]] VkQueue computeQueue() const { return compute_queue_; }
        [[nodiscard]] uint32_t computeQueueFamily() const { return compute_queue_family_; }
        [[nodiscard]] bool hasDedicatedComputeQueue() const { return has_dedicated_compute_queue_; }
        [[nodiscard]] const std::array<std::uint8_t, VK_UUID_SIZE>& deviceUUID() const { return device_uuid_; }
#ifdef _WIN32
        [[nodiscard]] const std::array<std::uint8_t, VK_LUID_SIZE>& deviceLUID() const { return device_luid_; }
        [[nodiscard]] bool deviceLUIDValid() const { return device_luid_valid_; }
        [[nodiscard]] std::uint32_t deviceNodeMask() const { return device_node_mask_; }
#endif
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
        [[nodiscard]] bool waitForCurrentFrameSlot();
        [[nodiscard]] bool waitForSubmittedFrames();
        [[nodiscard]] bool deviceWaitIdle();
        void addFrameTimelineWait(VkSemaphore semaphore,
                                  std::uint64_t value,
                                  VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

        [[nodiscard]] bool createExternalImage(VkExtent2D extent, VkFormat format, ExternalImage& out);
        void destroyExternalImage(ExternalImage& image);
        [[nodiscard]] ExternalNativeHandle releaseExternalImageNativeHandle(ExternalImage& image) const;
        [[nodiscard]] bool createExternalBuffer(VkDeviceSize size,
                                                VkBufferUsageFlags usage,
                                                ExternalBuffer& out);
        void destroyExternalBuffer(ExternalBuffer& buffer);
        [[nodiscard]] ExternalNativeHandle releaseExternalBufferNativeHandle(ExternalBuffer& buffer) const;
        [[nodiscard]] bool createExternalTimelineSemaphore(std::uint64_t initial_value, ExternalSemaphore& out);
        void destroyExternalSemaphore(ExternalSemaphore& semaphore);
        [[nodiscard]] ExternalNativeHandle releaseExternalSemaphoreNativeHandle(ExternalSemaphore& semaphore) const;
        [[nodiscard]] static bool externalNativeHandleValid(ExternalNativeHandle handle);
        void closeExternalNativeHandle(ExternalNativeHandle& handle) const;
        [[nodiscard]] bool transitionImageLayoutImmediate(VkImage image,
                                                          VkImageLayout old_layout,
                                                          VkImageLayout new_layout,
                                                          VkImageAspectFlags aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                          VkSemaphore wait_semaphore = VK_NULL_HANDLE,
                                                          std::uint64_t wait_value = 0,
                                                          VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
#endif

    private:
        bool fail(std::string message);

#ifdef LFS_VULKAN_VIEWER_ENABLED
        struct QueueFamilies {
            std::optional<uint32_t> graphics;
            std::optional<uint32_t> present;
            std::optional<uint32_t> async_compute; // optional dedicated compute family
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

        struct DepthStencilResource {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
        };

        bool createInstance();
        bool createSurface(SDL_Window* window);
        bool pickPhysicalDevice();
        bool createDevice();
        bool createAllocator();
        bool createSwapchain(int framebuffer_width, int framebuffer_height);
        bool createImageViews();
        bool createDepthStencilResources();
        bool createCommandPool();
        bool createCommandBuffers();
        bool createSyncObjects();
        bool createDebugMessenger();
        bool createPipelineCache();
        bool recreateSwapchain();

        void destroyDebugMessenger();
        void destroyAllocator();
        void saveAndDestroyPipelineCache();
        void destroySwapchain();
        [[nodiscard]] bool waitForFrameFences();

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
        [[nodiscard]] VkImageAspectFlags depthStencilAspectMask() const;
        VkInstance instance_ = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
        VkSurfaceKHR surface_ = VK_NULL_HANDLE;
        VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
        std::array<std::uint8_t, VK_UUID_SIZE> device_uuid_{};
#ifdef _WIN32
        std::array<std::uint8_t, VK_LUID_SIZE> device_luid_{};
        bool device_luid_valid_ = false;
        std::uint32_t device_node_mask_ = 0;
#endif
        VkDevice device_ = VK_NULL_HANDLE;
        VmaAllocator allocator_ = VK_NULL_HANDLE;
        VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
        VkQueue graphics_queue_ = VK_NULL_HANDLE;
        VkQueue present_queue_ = VK_NULL_HANDLE;
        uint32_t graphics_queue_family_ = 0;
        uint32_t present_queue_family_ = 0;
        VkQueue compute_queue_ = VK_NULL_HANDLE;
        uint32_t compute_queue_family_ = 0;
        bool has_dedicated_compute_queue_ = false;

        VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
        VkFormat swapchain_format_ = VK_FORMAT_UNDEFINED;
        VkColorSpaceKHR swapchain_color_space_ = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        bool has_hdr_ = false;
        VkExtent2D swapchain_extent_{};
        VkImageUsageFlags swapchain_image_usage_ = 0;
        uint32_t min_image_count_ = 2;
        std::vector<VkImage> swapchain_images_;
        std::vector<VkImageView> swapchain_image_views_;
        VulkanImageBarrierTracker image_barriers_;
        VkFormat depth_stencil_format_ = VK_FORMAT_UNDEFINED;
        std::vector<DepthStencilResource> depth_stencil_resources_;

        // Single frame in flight — CPU waits for GPU each frame instead of pre-recording
        // the next one. Eliminates the frame of cursor latency that comes from "CPU is
        // 1 frame ahead of GPU".
        static constexpr std::size_t kFramesInFlight = 1;
        std::array<VkCommandPool, kFramesInFlight> command_pools_{};
        std::array<VkCommandBuffer, kFramesInFlight> command_buffers_{};
        VkCommandPool immediate_command_pool_ = VK_NULL_HANDLE;
        // Async cleanup queue for transitionImageLayoutImmediate. The function
        // used to vkWaitForFences synchronously after submit (3-9ms/frame on
        // the CUDA→Vulkan handoff path because it also blocked CPU on the
        // CUDA-signaled semaphore via vkWaitSemaphores). Now we fire-and-
        // forget: submit, push (cmd, fence) here, drain on next call.
        struct PendingImmediateSubmit {
            VkCommandBuffer cmd = VK_NULL_HANDLE;
            VkFence fence = VK_NULL_HANDLE;
        };
        std::vector<PendingImmediateSubmit> pending_immediate_submits_;
        void drainCompletedImmediateSubmits();
        std::vector<FrameTimelineWait> frame_timeline_waits_;
        // image_available_ is sized to swapchain image count (not framesInFlight). We must
        // pass a fresh semaphore to each vkAcquireNextImageKHR — reusing one before its
        // signal has been consumed by submit is undefined per spec. Rotation is independent
        // of frame slot (next_acquire_index_) and the index used for an acquire is stashed
        // in active_acquire_index_ so endFrame's submit waits on the same semaphore.
        std::vector<VkSemaphore> image_available_;
        std::size_t next_acquire_index_ = 0;
        std::size_t active_acquire_index_ = 0;
        std::array<VkSemaphore, kFramesInFlight> render_finished_{};
        std::array<VkFence, kFramesInFlight> in_flight_{};
        std::vector<VkFence> swapchain_images_in_flight_;

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
        bool has_push_descriptor_ = false;
        bool has_shader_object_ = false;
        bool has_extended_dynamic_state3_ = false;
        bool has_cooperative_matrix_ = false;
        bool has_host_image_copy_ = false;
        bool has_descriptor_indexing_ = false;
        PFN_vkSetDebugUtilsObjectNameEXT vk_set_debug_utils_object_name_ = nullptr;
        PFN_vkCmdPushDescriptorSetKHR vk_cmd_push_descriptor_set_ = nullptr;
        uint32_t active_image_index_ = 0;
        std::size_t frame_index_ = 0;
        std::size_t active_frame_index_ = 0;
        int framebuffer_width_ = 0;
        int framebuffer_height_ = 0;
#endif

        std::string last_error_;
    };

} // namespace lfs::vis
