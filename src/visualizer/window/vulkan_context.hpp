/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "renderer_terminal_state.hpp"
#include "rendering/vulkan_result.hpp"
#include "rendering/vulkan_wait.hpp"
#include "vulkan_image_barrier_tracker.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>

#ifndef VMA_STATIC_VULKAN_FUNCTIONS
#define VMA_STATIC_VULKAN_FUNCTIONS 1
#endif
#ifndef VMA_DYNAMIC_VULKAN_FUNCTIONS
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#endif
#include <vk_mem_alloc.h>

struct SDL_Window;

namespace lfs::vis {

    class VulkanContext {
    public:
        enum class ResizeIntent {
            Interactive,
            Exact,
        };

        VulkanContext() = default;
        ~VulkanContext();

        VulkanContext(const VulkanContext&) = delete;
        VulkanContext& operator=(const VulkanContext&) = delete;

        bool init(SDL_Window* window, int framebuffer_width, int framebuffer_height);
        void shutdown();
        void notifyFramebufferResized(int width, int height, ResizeIntent intent = ResizeIntent::Exact);
        [[nodiscard]] bool hasPendingSwapchainResize() const {
            return framebuffer_resize_deferred_ || framebuffer_resize_exact_after_interactive_;
        }
        [[nodiscard]] bool pendingSwapchainResizeReady() const;
        [[nodiscard]] double secondsUntilPendingSwapchainResizeReady() const;

        [[nodiscard]] bool presentBootstrapFrame(float r, float g, float b, float a);
        [[nodiscard]] const std::string& lastError() const { return last_error_; }

        // Typed terminal-renderer state polled by the frame state machine. Acquire
        // loads of the two 7B cause latches; DeviceLost dominates a bare quarantine.
        // Additive: never parse lastError() for state.
        [[nodiscard]] RendererTerminalState rendererTerminalState() const noexcept {
            return renderer_terminal_state(gpu_device_lost_.load(std::memory_order_acquire),
                                           gpu_wait_quarantined_.load(std::memory_order_acquire));
        }

        struct Frame {
            uint32_t image_index = 0;
            std::size_t frame_slot = 0;
            VkCommandBuffer command_buffer = VK_NULL_HANDLE;
            VkImage swapchain_image = VK_NULL_HANDLE;
            VkImageView swapchain_image_view = VK_NULL_HANDLE;
            VkImageView depth_stencil_image_view = VK_NULL_HANDLE;
            VkExtent2D extent{};
        };

        struct WindowCapture {
            int width = 0;
            int height = 0;
            std::vector<std::uint8_t> rgba;
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
            std::string diagnostic_scope;
            std::string diagnostic_label;
            ExternalNativeHandle native_handle = kInvalidExternalNativeHandle;
        };

        struct ExternalBuffer {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VkDeviceSize size = 0;
            VkDeviceSize allocation_size = 0;
            std::string diagnostic_scope;
            std::string diagnostic_label;
            ExternalNativeHandle native_handle = kInvalidExternalNativeHandle;
        };

        struct ExternalSemaphore {
            VkSemaphore semaphore = VK_NULL_HANDLE;
            std::uint64_t initial_value = 0;
            ExternalNativeHandle native_handle = kInvalidExternalNativeHandle;
        };

        struct TimelinePoint {
            TimelinePoint(VkSemaphore semaphore, std::uint64_t value) noexcept
                : semaphore(semaphore),
                  value(value) {}

            VkSemaphore semaphore;
            std::uint64_t value;
        };

        struct ImmediateTransitionOptions {
            VkImageAspectFlags aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT;
            std::optional<TimelinePoint> wait;
            VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            std::optional<TimelinePoint> signal;

            [[nodiscard]] static ImmediateTransitionOptions waitOn(
                TimelinePoint point,
                VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) noexcept {
                return {
                    .aspect_mask = aspect,
                    .wait = point,
                    .wait_stage = stage,
                    .signal = std::nullopt,
                };
            }

            [[nodiscard]] static ImmediateTransitionOptions signalAt(
                TimelinePoint point,
                VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) noexcept {
                return {
                    .aspect_mask = aspect,
                    .wait = std::nullopt,
                    .wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    .signal = point,
                };
            }
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
        [[nodiscard]] std::size_t queryVmaUsedBytes() const;
        [[nodiscard]] VkPipelineCache pipelineCache() const { return pipeline_cache_; }
        [[nodiscard]] VkFormat swapchainFormat() const { return swapchain_format_; }
        [[nodiscard]] VkColorSpaceKHR swapchainColorSpace() const { return swapchain_color_space_; }
        [[nodiscard]] bool hasHdr() const noexcept { return has_hdr_; }
        [[nodiscard]] VkFormat depthStencilFormat() const { return depth_stencil_format_; }
        [[nodiscard]] VkImageAspectFlags depthStencilAspectMask() const;
        [[nodiscard]] VkImageView depthStencilImageView() const {
            return active_frame_index_ < depth_stencil_resources_.size()
                       ? depth_stencil_resources_[active_frame_index_].view
                       : VK_NULL_HANDLE;
        }
        [[nodiscard]] VkExtent2D swapchainExtent() const { return swapchain_extent_; }
        [[nodiscard]] VkExtent2D framebufferExtent() const;
        [[nodiscard]] uint32_t minImageCount() const { return min_image_count_; }
        [[nodiscard]] uint32_t imageCount() const { return static_cast<uint32_t>(swapchain_images_.size()); }
        [[nodiscard]] std::size_t framesInFlight() const { return kFramesInFlight; }
        [[nodiscard]] std::size_t currentFrameSlot() const { return frame_index_; }
        [[nodiscard]] bool externalMemoryInteropEnabled() const { return external_memory_interop_enabled_; }
        [[nodiscard]] bool externalSemaphoreInteropEnabled() const { return external_semaphore_interop_enabled_; }
        [[nodiscard]] VulkanImageBarrierTracker& imageBarriers() { return image_barriers_; }
        [[nodiscard]] bool hasPushDescriptor() const { return has_push_descriptor_; }
        [[nodiscard]] PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSet() const { return vk_cmd_push_descriptor_set_; }
        [[nodiscard]] bool hasHostImageCopy() const { return has_host_image_copy_; }
        [[nodiscard]] bool hasFloat16Storage() const { return has_float16_storage_; }
        [[nodiscard]] bool hasFillModeNonSolid() const { return has_fill_mode_non_solid_; }
        [[nodiscard]] bool hasWideLines() const { return has_wide_lines_; }
        [[nodiscard]] float minLineWidth() const { return line_width_range_[0]; }
        [[nodiscard]] float maxLineWidth() const { return line_width_range_[1]; }
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
        template <typename VkHandle, typename... Args>
        void setDebugObjectNamef(VkObjectType object_type,
                                 VkHandle object,
                                 std::format_string<Args...> format,
                                 Args&&... args) const {
            if (!debugObjectNamingEnabled() || object == VK_NULL_HANDLE) {
                return;
            }
            setDebugObjectName(object_type,
                               vulkanObjectHandle(object),
                               std::format(format, std::forward<Args>(args)...));
        }
        void setDebugObjectName(VkObjectType object_type, std::uint64_t object_handle, std::string_view name) const;
        [[nodiscard]] bool debugObjectNamingEnabled() const noexcept {
            return debug_utils_enabled_ && debug_name_writer_.enabled();
        }

        [[nodiscard]] bool beginFrame(const VkClearValue& clear_value, Frame& frame);
        [[nodiscard]] bool endFrame();
        [[nodiscard]] bool hasActiveFrame() const noexcept { return frame_active_; }
        [[nodiscard]] LFS_VIS_API std::expected<WindowCapture, std::string> captureAndEndActiveFrameRgba();
        [[nodiscard]] bool waitForCurrentFrameSlot();
        [[nodiscard]] bool waitForSubmittedFrames();
        [[nodiscard]] bool waitForImmediateSubmits();
        [[nodiscard]] bool deviceWaitIdle();
        void addFrameTimelineWait(VkSemaphore semaphore,
                                  std::uint64_t value,
                                  VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

        [[nodiscard]] bool createExternalImage(VkExtent2D extent,
                                               VkFormat format,
                                               ExternalImage& out,
                                               std::string_view diagnostic_scope = "vulkan.external.image",
                                               std::string_view diagnostic_label = {});
        void destroyExternalImage(ExternalImage& image);
        [[nodiscard]] ExternalNativeHandle releaseExternalImageNativeHandle(ExternalImage& image) const;
        [[nodiscard]] bool createExternalBuffer(VkDeviceSize size,
                                                VkBufferUsageFlags usage,
                                                ExternalBuffer& out,
                                                std::string_view diagnostic_scope = "vulkan.external.buffer",
                                                std::string_view diagnostic_label = {});
        void destroyExternalBuffer(ExternalBuffer& buffer);
        [[nodiscard]] ExternalNativeHandle releaseExternalBufferNativeHandle(ExternalBuffer& buffer) const;
        // Import a foreign-allocated external memory handle (e.g. from CUDA's
        // cuMemExportToShareableHandle) into Vulkan. The exporter retains ownership
        // of the handle; this method dup()'s on Linux and the imported VkDeviceMemory
        // is released by destroyExternalBuffer. The returned ExternalBuffer's
        // native_handle stays kInvalidExternalNativeHandle (we are not the owner).
        [[nodiscard]] bool importExternalBuffer(ExternalNativeHandle handle,
                                                VkDeviceSize size,
                                                VkBufferUsageFlags usage,
                                                ExternalBuffer& out,
                                                std::string_view diagnostic_scope = "vulkan.external.imported_buffer",
                                                std::string_view diagnostic_label = {});
        [[nodiscard]] bool createExternalTimelineSemaphore(std::uint64_t initial_value, ExternalSemaphore& out);
        void destroyExternalSemaphore(ExternalSemaphore& semaphore);
        [[nodiscard]] ExternalNativeHandle releaseExternalSemaphoreNativeHandle(ExternalSemaphore& semaphore) const;
        [[nodiscard]] static bool externalNativeHandleValid(ExternalNativeHandle handle);
        void closeExternalNativeHandle(ExternalNativeHandle& handle) const;
        [[nodiscard]] bool transitionImageLayoutImmediate(VkImage image,
                                                          VkImageLayout old_layout,
                                                          VkImageLayout new_layout,
                                                          const ImmediateTransitionOptions& options);

    private:
        bool fail(std::string message,
                  std::source_location location = std::source_location::current());
        bool setVkFailure(std::string message);

        // Phase 7B: shared WaitContext for the six UI-frame bounded waits.
        [[nodiscard]] lfs::rendering::WaitContext makeWaitContext(std::string_view fingerprint);
        // Map fence/semaphore WaitOutcome → bool + last_error_. Soft Cancelled/Shutdown
        // clear last_error_ and return false; Quarantined/DeviceLost/other fail().
        // When set_framebuffer_resized_on_hard_error is true (image-fence site), only
        // hard Error results set framebuffer_resized_ (never Quarantined/soft).
        [[nodiscard]] bool mapWaitOutcome(lfs::Result<lfs::rendering::WaitOutcome> result,
                                          std::string_view op,
                                          bool set_framebuffer_resized_on_hard_error = false);
        // Validation timeline waits (sites 5–6): every non-Ready path fails with detail.
        [[nodiscard]] bool mapValidationWaitOutcome(lfs::Result<lfs::rendering::WaitOutcome> result,
                                                    std::string_view detail_on_fail);

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
        bool createSwapchain(int framebuffer_width, int framebuffer_height,
                             VkSwapchainKHR old_swapchain = VK_NULL_HANDLE);
        bool createImageViews();
        bool createDepthStencilResources();
        bool createCommandPool();
        bool createCommandBuffers();
        bool createSyncObjects();
        bool replaceFrameFenceSignaled(std::size_t frame_slot);
        bool createDebugMessenger();
        bool createPipelineCache();
        bool recreateSwapchain();
        bool finishActiveRendering(VkCommandBuffer command_buffer);
        void deferSwapchainResizeRecreate(bool requires_recreate = true,
                                          std::optional<bool> allow_headroom = std::nullopt);
        [[nodiscard]] bool promoteDeferredSwapchainResizeIfSettled();
        [[nodiscard]] bool framebufferFitsSwapchainExtent() const;
        [[nodiscard]] bool framebufferResizeRequiresSwapchainRecreate() const;

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
        [[nodiscard]] std::optional<VkSurfacePresentScalingCapabilitiesEXT>
        queryPresentScalingCapabilities(VkPresentModeKHR present_mode) const;
        [[nodiscard]] VkExtent2D chooseSwapchainExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                                                       int framebuffer_width,
                                                       int framebuffer_height,
                                                       bool add_resize_headroom,
                                                       const VkSurfacePresentScalingCapabilitiesEXT* scaling_capabilities) const;
        [[nodiscard]] VkFormat chooseDepthStencilFormat() const;
        [[nodiscard]] uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties) const;
        [[nodiscard]] std::string makeAllocationDiagnosticLabel(std::string_view label);
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
        std::size_t swapchain_estimated_bytes_ = 0;
        VulkanImageBarrierTracker image_barriers_;
        VkFormat depth_stencil_format_ = VK_FORMAT_UNDEFINED;
        std::vector<DepthStencilResource> depth_stencil_resources_;

        // Two frame slots keep resize/input processing from blocking on the
        // previous present's image-availability fence while remaining shallow
        // enough to avoid the latency of a deep render queue.
        static constexpr std::size_t kFramesInFlight = 2;
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
        [[nodiscard]] bool drainCompletedImmediateSubmits();
        std::vector<FrameTimelineWait> frame_timeline_waits_;
        bool frame_timeline_waits_valid_ = true;
        std::unordered_map<VkSemaphore, std::uint64_t> last_frame_timeline_wait_values_;
        std::unordered_map<VkSemaphore, std::uint64_t> last_immediate_timeline_wait_values_;
        std::unordered_map<VkSemaphore, std::uint64_t> last_immediate_timeline_signal_values_;
        // image_available_ is sized to swapchain image count (not framesInFlight). We must
        // pass a fresh semaphore to each vkAcquireNextImageKHR — reusing one before its
        // signal has been consumed by submit is undefined per spec. Rotation is independent
        // of frame slot (next_acquire_index_) and the index used for an acquire is stashed
        // in active_acquire_index_ so endFrame's submit waits on the same semaphore.
        std::vector<VkSemaphore> image_available_;
        std::size_t next_acquire_index_ = 0;
        std::size_t active_acquire_index_ = 0;
        // Presentation waits are owned per swapchain image. Frame-slot indexing can re-signal a
        // binary semaphore while an earlier vkQueuePresentKHR wait is still pending whenever the
        // swapchain has more images than frames in flight.
        std::vector<VkSemaphore> render_finished_;
        std::array<VkFence, kFramesInFlight> in_flight_{};
        std::array<std::uint64_t, kFramesInFlight> frame_submit_serials_{};
        std::uint64_t frame_submit_serial_ = 0;
        std::vector<VkFence> swapchain_images_in_flight_;

        bool framebuffer_resized_ = false;
        bool framebuffer_resize_deferred_ = false;
        bool framebuffer_resize_requires_recreate_ = false;
        bool framebuffer_resize_allow_headroom_ = false;
        bool framebuffer_resize_exact_after_interactive_ = false;
        bool swapchain_extent_fixed_to_surface_ = false;
        std::chrono::steady_clock::time_point framebuffer_resize_last_change_{};
        std::chrono::steady_clock::time_point framebuffer_resize_last_recreate_{};
        bool frame_active_ = false;
        bool frame_rendering_active_ = false;
        bool frame_suboptimal_ = false;
        // Phase 7B: owner quarantine latch (10 s policy) shared by the six UI waits.
        std::atomic<bool> gpu_wait_quarantined_{false};
        // Phase 8 P3: second cause latch — set only at the three DeviceLost sites so
        // rendererTerminalState() distinguishes a lost device from a bare stall.
        std::atomic<bool> gpu_device_lost_{false};
        // Phase 7B AMB-B3: set at shutdown() entry so mid-teardown waits yield Shutdown.
        std::atomic<bool> context_shutdown_started_{false};
        bool debug_utils_enabled_ = false;
        bool validation_enabled_ = false;
        bool validation_errors_fatal_ = false;
        bool instance_external_memory_capabilities_enabled_ = false;
        bool instance_external_semaphore_capabilities_enabled_ = false;
        bool instance_surface_maintenance_enabled_ = false;
        bool external_memory_interop_enabled_ = false;
        bool external_semaphore_interop_enabled_ = false;
        bool external_memory_dedicated_allocation_enabled_ = false;
        bool swapchain_maintenance1_enabled_ = false;
        bool swapchain_present_scaling_enabled_ = false;
        bool has_push_descriptor_ = false;
        bool has_float16_storage_ = false;
        bool has_host_image_copy_ = false;
        bool has_fill_mode_non_solid_ = false;
        bool has_wide_lines_ = false;
        std::array<float, 2> line_width_range_{1.0f, 1.0f};
        lfs::rendering::VulkanDebugNameWriter debug_name_writer_;
        PFN_vkCmdPushDescriptorSetKHR vk_cmd_push_descriptor_set_ = nullptr;
        uint32_t active_image_index_ = 0;
        std::size_t frame_index_ = 0;
        std::size_t active_frame_index_ = 0;
        int framebuffer_width_ = 0;
        int framebuffer_height_ = 0;
        std::uint64_t allocation_diagnostic_serial_ = 0;

        std::string last_error_;
    };

} // namespace lfs::vis
