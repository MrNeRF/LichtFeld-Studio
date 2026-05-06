/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_context.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "vulkan_result.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <set>
#include <utility>

#ifdef LFS_VULKAN_VIEWER_ENABLED
#include <SDL3/SDL_vulkan.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#endif

namespace lfs::vis {
    namespace {
#ifdef LFS_VULKAN_VIEWER_ENABLED
#ifdef _WIN32
        constexpr VkExternalMemoryHandleTypeFlagBits kExternalMemoryHandleType =
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        constexpr VkExternalSemaphoreHandleTypeFlagBits kExternalSemaphoreHandleType =
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
        constexpr VkExternalMemoryHandleTypeFlagBits kExternalMemoryHandleType =
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        constexpr VkExternalSemaphoreHandleTypeFlagBits kExternalSemaphoreHandleType =
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
        constexpr std::uint64_t kExternalTimelineWaitTimeoutNs = 2'000'000'000ull;

        [[nodiscard]] bool extensionAvailable(const std::vector<VkExtensionProperties>& extensions,
                                              const char* const extension_name) {
            return std::ranges::any_of(extensions, [extension_name](const VkExtensionProperties& extension) {
                return std::strcmp(extension.extensionName, extension_name) == 0;
            });
        }

        [[nodiscard]] bool layerAvailable(const std::vector<VkLayerProperties>& layers,
                                          const char* const layer_name) {
            return std::ranges::any_of(layers, [layer_name](const VkLayerProperties& layer) {
                return std::strcmp(layer.layerName, layer_name) == 0;
            });
        }

        void appendUniqueExtension(std::vector<const char*>& extensions, const char* const extension_name) {
            const auto existing = std::ranges::find_if(extensions, [extension_name](const char* const enabled) {
                return std::strcmp(enabled, extension_name) == 0;
            });
            if (existing == extensions.end()) {
                extensions.push_back(extension_name);
            }
        }

        [[nodiscard]] std::string vulkanApiVersionString(const uint32_t api_version) {
            return std::format("{}.{}.{}",
                               VK_API_VERSION_MAJOR(api_version),
                               VK_API_VERSION_MINOR(api_version),
                               VK_API_VERSION_PATCH(api_version));
        }

        struct RequiredFeatureSupport {
            bool synchronization2 = false;
            bool dynamic_rendering = false;
            bool timeline_semaphore = false;
            bool buffer_device_address = false;
        };

        [[nodiscard]] RequiredFeatureSupport queryRequiredFeatureSupport(const VkPhysicalDevice device) {
            VkPhysicalDeviceVulkan13Features features13{};
            features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

            VkPhysicalDeviceVulkan12Features features12{};
            features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            features12.pNext = &features13;

            VkPhysicalDeviceFeatures2 features2{};
            features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features2.pNext = &features12;
            vkGetPhysicalDeviceFeatures2(device, &features2);

            RequiredFeatureSupport support{};
            support.synchronization2 = features13.synchronization2 == VK_TRUE;
            support.dynamic_rendering = features13.dynamicRendering == VK_TRUE;
            support.timeline_semaphore = features12.timelineSemaphore == VK_TRUE;
            support.buffer_device_address = features12.bufferDeviceAddress == VK_TRUE;
            return support;
        }

        [[nodiscard]] bool hasRequiredFeatures(const RequiredFeatureSupport& support) {
            return support.synchronization2 &&
                   support.dynamic_rendering &&
                   support.timeline_semaphore &&
                   support.buffer_device_address;
        }

        void appendMissingFeature(std::string& missing, const bool present, std::string_view feature_name) {
            if (present) {
                return;
            }
            if (!missing.empty()) {
                missing += ", ";
            }
            missing += feature_name;
        }

        [[nodiscard]] std::string missingRequiredFeatures(const RequiredFeatureSupport& support) {
            std::string missing;
            appendMissingFeature(missing, support.synchronization2, "synchronization2");
            appendMissingFeature(missing, support.dynamic_rendering, "dynamicRendering");
            appendMissingFeature(missing, support.timeline_semaphore, "timelineSemaphore");
            appendMissingFeature(missing, support.buffer_device_address, "bufferDeviceAddress");
            return missing;
        }

        [[nodiscard]] bool validationRequestedByBuild() {
#if defined(DEBUG_BUILD) || !defined(NDEBUG)
            return true;
#else
            return false;
#endif
        }

        VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
            VkDebugUtilsMessageTypeFlagsEXT,
            const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
            void*) {
            const char* const message = callback_data != nullptr && callback_data->pMessage != nullptr
                                            ? callback_data->pMessage
                                            : "<missing validation message>";

            if ((message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
                LOG_ERROR("Vulkan validation: {}", message);
            } else if ((message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
                LOG_WARN("Vulkan validation: {}", message);
            }
            return VK_FALSE;
        }

        void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& create_info) {
            create_info = {};
            create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            create_info.pfnUserCallback = vulkanDebugCallback;
        }

        [[nodiscard]] std::filesystem::path defaultPipelineCachePath() {
#ifdef _WIN32
            if (const char* local_app_data = std::getenv("LOCALAPPDATA"); local_app_data && local_app_data[0] != '\0') {
                return std::filesystem::path(local_app_data) / "LichtFeld" / "pipeline_cache.bin";
            }
            return std::filesystem::current_path() / "LichtFeld" / "pipeline_cache.bin";
#else
            if (const char* xdg_cache_home = std::getenv("XDG_CACHE_HOME"); xdg_cache_home && xdg_cache_home[0] != '\0') {
                return std::filesystem::path(xdg_cache_home) / "lichtfeld" / "pipeline_cache.bin";
            }
            if (const char* home = std::getenv("HOME"); home && home[0] != '\0') {
                return std::filesystem::path(home) / ".cache" / "lichtfeld" / "pipeline_cache.bin";
            }
            return std::filesystem::current_path() / ".cache" / "lichtfeld" / "pipeline_cache.bin";
#endif
        }

        [[nodiscard]] bool readFile(const std::filesystem::path& path, std::vector<char>& data) {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary | std::ios::ate, file)) {
                return false;
            }

            const std::streamoff size = file.tellg();
            if (size <= 0) {
                return false;
            }

            data.resize(static_cast<std::size_t>(size));
            file.seekg(0, std::ios::beg);
            return static_cast<bool>(file.read(data.data(), size));
        }

        [[nodiscard]] bool validPipelineCacheHeader(const std::vector<char>& data,
                                                    const VkPhysicalDeviceProperties& device_props) {
            if (data.size() < sizeof(VkPipelineCacheHeaderVersionOne)) {
                return false;
            }

            VkPipelineCacheHeaderVersionOne header{};
            std::memcpy(&header, data.data(), sizeof(header));
            if (header.headerSize < sizeof(VkPipelineCacheHeaderVersionOne) ||
                header.headerSize > data.size() ||
                header.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE ||
                header.vendorID != device_props.vendorID ||
                header.deviceID != device_props.deviceID) {
                return false;
            }

            return std::memcmp(header.pipelineCacheUUID, device_props.pipelineCacheUUID, VK_UUID_SIZE) == 0;
        }

        constexpr std::uint64_t kWaitForeverNs = std::numeric_limits<std::uint64_t>::max();

        [[nodiscard]] const char* vkFormatToString(const VkFormat format) noexcept {
            switch (format) {
            case VK_FORMAT_B8G8R8A8_UNORM: return "VK_FORMAT_B8G8R8A8_UNORM";
            case VK_FORMAT_B8G8R8A8_SRGB: return "VK_FORMAT_B8G8R8A8_SRGB";
            case VK_FORMAT_R8G8B8A8_UNORM: return "VK_FORMAT_R8G8B8A8_UNORM";
            case VK_FORMAT_R8G8B8A8_SRGB: return "VK_FORMAT_R8G8B8A8_SRGB";
            case VK_FORMAT_A8B8G8R8_UNORM_PACK32: return "VK_FORMAT_A8B8G8R8_UNORM_PACK32";
            case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return "VK_FORMAT_A8B8G8R8_SRGB_PACK32";
            case VK_FORMAT_A2R10G10B10_UNORM_PACK32: return "VK_FORMAT_A2R10G10B10_UNORM_PACK32";
            case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return "VK_FORMAT_A2B10G10R10_UNORM_PACK32";
            case VK_FORMAT_R16G16B16A16_SFLOAT: return "VK_FORMAT_R16G16B16A16_SFLOAT";
            default: return "VK_FORMAT_UNKNOWN";
            }
        }

        [[nodiscard]] const char* vkColorSpaceToString(const VkColorSpaceKHR cs) noexcept {
            switch (cs) {
            case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR: return "VK_COLOR_SPACE_SRGB_NONLINEAR_KHR";
            case VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT: return "VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT";
            case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT: return "VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT";
            case VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT: return "VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT";
            case VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT: return "VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT";
            case VK_COLOR_SPACE_BT709_LINEAR_EXT: return "VK_COLOR_SPACE_BT709_LINEAR_EXT";
            case VK_COLOR_SPACE_BT709_NONLINEAR_EXT: return "VK_COLOR_SPACE_BT709_NONLINEAR_EXT";
            case VK_COLOR_SPACE_BT2020_LINEAR_EXT: return "VK_COLOR_SPACE_BT2020_LINEAR_EXT";
            case VK_COLOR_SPACE_HDR10_ST2084_EXT: return "VK_COLOR_SPACE_HDR10_ST2084_EXT";
            case VK_COLOR_SPACE_HDR10_HLG_EXT: return "VK_COLOR_SPACE_HDR10_HLG_EXT";
            case VK_COLOR_SPACE_DOLBYVISION_EXT: return "VK_COLOR_SPACE_DOLBYVISION_EXT";
            case VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT: return "VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT";
            case VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT: return "VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT";
            case VK_COLOR_SPACE_PASS_THROUGH_EXT: return "VK_COLOR_SPACE_PASS_THROUGH_EXT";
            case VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT: return "VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT";
            case VK_COLOR_SPACE_DISPLAY_NATIVE_AMD: return "VK_COLOR_SPACE_DISPLAY_NATIVE_AMD";
            default: return "VK_COLOR_SPACE_UNKNOWN";
            }
        }
#endif
    } // namespace

    VulkanContext::~VulkanContext() {
        shutdown();
    }

    bool VulkanContext::fail(std::string message) {
        last_error_ = std::move(message);
        LOG_ERROR("Vulkan: {}", last_error_);
        return false;
    }

    bool VulkanContext::init(SDL_Window* window, const int framebuffer_width, const int framebuffer_height) {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        framebuffer_width_ = framebuffer_width;
        framebuffer_height_ = framebuffer_height;

        return createInstance() &&
               createSurface(window) &&
               pickPhysicalDevice() &&
               createDevice() &&
               createAllocator() &&
               createPipelineCache() &&
               createSwapchain(framebuffer_width, framebuffer_height) &&
               createImageViews() &&
               createDepthStencilResources() &&
               createCommandPool() &&
               createCommandBuffers() &&
               createSyncObjects();
#else
        (void)window;
        (void)framebuffer_width;
        (void)framebuffer_height;
        return fail("Vulkan viewer dependencies are disabled at compile time");
#endif
    }

    void VulkanContext::shutdown() {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        if (device_ != VK_NULL_HANDLE) {
            // Shutdown is the one place where a whole-device wait is intentional:
            // all swapchain, UI, and external interop resources are about to be destroyed.
            vkDeviceWaitIdle(device_);
        }

        for (VkSemaphore& semaphore : render_finished_) {
            if (semaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(device_, semaphore, nullptr);
                semaphore = VK_NULL_HANDLE;
            }
        }
        for (VkFence& fence : in_flight_) {
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(device_, fence, nullptr);
                fence = VK_NULL_HANDLE;
            }
        }

        destroySwapchain();

        if (immediate_command_pool_ != VK_NULL_HANDLE) {
            // Drain any in-flight async submits before destroying their pool.
            // Bound the per-fence wait so a wedged GPU cannot deadlock shutdown — leak
            // the fence/cmd buffer in that case and let device destruction reap them.
            constexpr std::uint64_t kImmediateDrainTimeoutNs = 2'000'000'000ull; // 2 s
            for (auto& pending : pending_immediate_submits_) {
                if (pending.fence != VK_NULL_HANDLE) {
                    const VkResult drain = vkWaitForFences(device_, 1, &pending.fence, VK_TRUE,
                                                           kImmediateDrainTimeoutNs);
                    if (drain != VK_SUCCESS) {
                        LOG_ERROR("Immediate submit fence stuck during shutdown: {}; leaking command buffer",
                                  vkResultToString(drain));
                        continue;
                    }
                    vkDestroyFence(device_, pending.fence, nullptr);
                }
                if (pending.cmd != VK_NULL_HANDLE) {
                    vkFreeCommandBuffers(device_, immediate_command_pool_, 1, &pending.cmd);
                }
            }
            pending_immediate_submits_.clear();
            vkDestroyCommandPool(device_, immediate_command_pool_, nullptr);
            immediate_command_pool_ = VK_NULL_HANDLE;
        }
        for (VkCommandPool& command_pool : command_pools_) {
            if (command_pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device_, command_pool, nullptr);
                command_pool = VK_NULL_HANDLE;
            }
        }
        saveAndDestroyPipelineCache();
        destroyAllocator();
        if (device_ != VK_NULL_HANDLE) {
            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
            vk_set_debug_utils_object_name_ = nullptr;
        }
        if (surface_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
            SDL_Vulkan_DestroySurface(instance_, surface_, nullptr);
            surface_ = VK_NULL_HANDLE;
        }
        if (instance_ != VK_NULL_HANDLE) {
            destroyDebugMessenger();
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
            debug_utils_enabled_ = false;
            validation_enabled_ = false;
        }
#endif
    }

    void VulkanContext::notifyFramebufferResized(const int width, const int height) {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        if (width == framebuffer_width_ && height == framebuffer_height_) {
            return;
        }
        framebuffer_width_ = width;
        framebuffer_height_ = height;
        framebuffer_resized_ = true;
#else
        (void)width;
        (void)height;
#endif
    }

    bool VulkanContext::presentBootstrapFrame(const float r, const float g, const float b, const float a) {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        VkClearValue clear_value{};
        clear_value.color = VkClearColorValue{{r, g, b, a}};

        Frame frame{};
        if (!beginFrame(clear_value, frame)) {
            return false;
        }
        return endFrame();
#else
        (void)r;
        (void)g;
        (void)b;
        (void)a;
        return false;
#endif
    }

#ifdef LFS_VULKAN_VIEWER_ENABLED
    bool VulkanContext::beginFrame(const VkClearValue& clear_value, Frame& frame) {
        if (frame_active_) {
            return fail("beginFrame called while another Vulkan frame is active");
        }
        frame_timeline_waits_.clear();
        frame = {};
        if (device_ == VK_NULL_HANDLE || framebuffer_width_ <= 0 || framebuffer_height_ <= 0) {
            last_error_.clear();
            return false;
        }

        if (swapchain_ == VK_NULL_HANDLE || framebuffer_resized_) {
            if (!recreateSwapchain()) {
                return false;
            }
        }

        const bool depth_stencil_ready =
            depth_stencil_resources_.size() == swapchain_images_.size() &&
            std::all_of(depth_stencil_resources_.begin(),
                        depth_stencil_resources_.end(),
                        [](const DepthStencilResource& resource) {
                            return resource.image != VK_NULL_HANDLE &&
                                   resource.view != VK_NULL_HANDLE;
                        });
        if (!depth_stencil_ready) {
            return fail("Vulkan swapchain depth/stencil resources are incomplete");
        }

        const std::size_t current_frame = frame_index_;
        VkFence frame_fence = in_flight_[current_frame];
        VkResult result = vkWaitForFences(device_, 1, &frame_fence, VK_TRUE, kWaitForeverNs);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkWaitForFences failed: {}", vkResultToString(result)));
        }

        uint32_t image_index = 0;
        if (image_available_.empty()) {
            return fail("Vulkan acquire semaphores have not been created");
        }
        const std::size_t acquire_index = next_acquire_index_;
        result = vkAcquireNextImageKHR(device_, swapchain_, kWaitForeverNs,
                                       image_available_[acquire_index],
                                       VK_NULL_HANDLE, &image_index);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            if (recreateSwapchain()) {
                last_error_.clear();
            }
            return false;
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            return fail(std::format("vkAcquireNextImageKHR failed: {}", vkResultToString(result)));
        }
        if (image_index >= swapchain_images_in_flight_.size()) {
            return fail(std::format("vkAcquireNextImageKHR returned invalid image index {}", image_index));
        }
        if (swapchain_images_in_flight_[image_index] != VK_NULL_HANDLE) {
            VkFence image_fence = swapchain_images_in_flight_[image_index];
            result = vkWaitForFences(device_, 1, &image_fence, VK_TRUE, kWaitForeverNs);
            if (result != VK_SUCCESS) {
                return fail(std::format("vkWaitForFences(swapchain image {}) failed: {}",
                                        image_index,
                                        vkResultToString(result)));
            }
        }

        frame_suboptimal_ = (result == VK_SUBOPTIMAL_KHR);
        active_image_index_ = image_index;
        active_frame_index_ = current_frame;
        active_acquire_index_ = acquire_index;
        next_acquire_index_ = (acquire_index + 1) % image_available_.size();

        result = vkResetCommandPool(device_, command_pools_[current_frame], 0);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkResetCommandPool failed: {}", vkResultToString(result)));
        }

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = vkBeginCommandBuffer(command_buffers_[current_frame], &begin_info);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkBeginCommandBuffer failed: {}", vkResultToString(result)));
        }

        VkCommandBuffer command_buffer = command_buffers_[current_frame];
        image_barriers_.transitionImage(command_buffer,
                                        swapchain_images_[image_index],
                                        VK_IMAGE_ASPECT_COLOR_BIT,
                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        if (image_index >= depth_stencil_resources_.size() ||
            depth_stencil_resources_[image_index].image == VK_NULL_HANDLE ||
            depth_stencil_resources_[image_index].view == VK_NULL_HANDLE) {
            return fail(std::format("Missing depth/stencil resource for swapchain image {}", image_index));
        }
        const DepthStencilResource& depth_stencil = depth_stencil_resources_[image_index];
        image_barriers_.transitionImage(command_buffer,
                                        depth_stencil.image,
                                        depthStencilAspectMask(),
                                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        VkRenderingAttachmentInfo color_attachment{};
        color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color_attachment.imageView = swapchain_image_views_[image_index];
        color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.clearValue = clear_value;

        VkClearValue depth_clear{};
        depth_clear.depthStencil = {1.0f, 0};

        VkRenderingAttachmentInfo depth_attachment{};
        depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth_attachment.imageView = depth_stencil.view;
        depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth_attachment.clearValue = depth_clear;

        VkRenderingInfo rendering_info{};
        rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering_info.renderArea.offset = {0, 0};
        rendering_info.renderArea.extent = swapchain_extent_;
        rendering_info.layerCount = 1;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachments = &color_attachment;
        rendering_info.pDepthAttachment = &depth_attachment;
        rendering_info.pStencilAttachment = &depth_attachment;
        vkCmdBeginRendering(command_buffer, &rendering_info);

        frame.image_index = image_index;
        frame.frame_slot = current_frame;
        frame.command_buffer = command_buffer;
        frame.swapchain_image = (swapchain_image_usage_ & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0
                                    ? swapchain_images_[image_index]
                                    : VK_NULL_HANDLE;
        frame.swapchain_image_view = swapchain_image_views_[image_index];
        frame.depth_stencil_image_view = depth_stencil.view;
        frame.extent = swapchain_extent_;
        frame_active_ = true;
        last_error_.clear();
        return true;
    }

    bool VulkanContext::endFrame() {
        if (!frame_active_) {
            return true;
        }

        const std::size_t current_frame = active_frame_index_;
        VkCommandBuffer command_buffer = command_buffers_[current_frame];
        vkCmdEndRendering(command_buffer);
        image_barriers_.transitionImage(command_buffer,
                                        swapchain_images_[active_image_index_],
                                        VK_IMAGE_ASPECT_COLOR_BIT,
                                        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

        VkResult result = vkEndCommandBuffer(command_buffer);
        if (result != VK_SUCCESS) {
            frame_active_ = false;
            return fail(std::format("vkEndCommandBuffer failed: {}", vkResultToString(result)));
        }

        std::vector<VkSemaphore> wait_semaphores;
        wait_semaphores.reserve(1 + frame_timeline_waits_.size());
        // Wait on the same semaphore that beginFrame passed to vkAcquireNextImageKHR — not
        // image_available_[current_frame]. The acquire-rotation index is independent of
        // the frame slot; a per-frame-slot wait would race with reuse on >2-image swapchains.
        wait_semaphores.push_back(image_available_[active_acquire_index_]);

        std::vector<VkPipelineStageFlags> wait_stages;
        wait_stages.reserve(1 + frame_timeline_waits_.size());
        wait_stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        std::vector<std::uint64_t> wait_values;
        wait_values.reserve(1 + frame_timeline_waits_.size());
        wait_values.push_back(0);
        for (const auto& wait : frame_timeline_waits_) {
            if (wait.semaphore == VK_NULL_HANDLE) {
                continue;
            }
            wait_semaphores.push_back(wait.semaphore);
            wait_stages.push_back(wait.wait_stage);
            wait_values.push_back(wait.value);
        }

        const std::uint64_t signal_value = 0;
        VkTimelineSemaphoreSubmitInfo timeline_submit_info{};
        timeline_submit_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_submit_info.waitSemaphoreValueCount = static_cast<std::uint32_t>(wait_values.size());
        timeline_submit_info.pWaitSemaphoreValues = wait_values.data();
        timeline_submit_info.signalSemaphoreValueCount = 1;
        timeline_submit_info.pSignalSemaphoreValues = &signal_value;

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.pNext = wait_values.size() > 1 ? &timeline_submit_info : nullptr;
        submit_info.waitSemaphoreCount = static_cast<std::uint32_t>(wait_semaphores.size());
        submit_info.pWaitSemaphores = wait_semaphores.data();
        submit_info.pWaitDstStageMask = wait_stages.data();
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &render_finished_[current_frame];

        VkFence frame_fence = in_flight_[current_frame];
        result = vkResetFences(device_, 1, &frame_fence);
        if (result != VK_SUCCESS) {
            frame_active_ = false;
            return fail(std::format("vkResetFences failed: {}", vkResultToString(result)));
        }
        result = vkQueueSubmit(graphics_queue_, 1, &submit_info, frame_fence);
        frame_timeline_waits_.clear();
        if (result != VK_SUCCESS) {
            frame_active_ = false;
            return fail(std::format("vkQueueSubmit failed: {}", vkResultToString(result)));
        }
        if (active_image_index_ < swapchain_images_in_flight_.size()) {
            swapchain_images_in_flight_[active_image_index_] = frame_fence;
        }

        VkPresentInfoKHR present_info{};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &render_finished_[current_frame];
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapchain_;
        present_info.pImageIndices = &active_image_index_;
        result = vkQueuePresentKHR(present_queue_, &present_info);

        frame_active_ = false;
        frame_index_ = (frame_index_ + 1) % kFramesInFlight;
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || frame_suboptimal_) {
            frame_suboptimal_ = false;
            return recreateSwapchain();
        }
        frame_suboptimal_ = false;
        if (result != VK_SUCCESS) {
            return fail(std::format("vkQueuePresentKHR failed: {}", vkResultToString(result)));
        }

        return true;
    }

    bool VulkanContext::waitForCurrentFrameSlot() {
        if (device_ == VK_NULL_HANDLE) {
            return fail("Cannot wait for Vulkan frame slot before device initialization");
        }
        const std::size_t current_frame = frame_index_;
        VkFence frame_fence = in_flight_[current_frame];
        if (frame_fence == VK_NULL_HANDLE) {
            return fail("Cannot wait for Vulkan frame slot before sync objects are initialized");
        }
        const VkResult result = vkWaitForFences(device_, 1, &frame_fence, VK_TRUE, kWaitForeverNs);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkWaitForFences(frame slot {}) failed: {}",
                                    current_frame,
                                    vkResultToString(result)));
        }
        last_error_.clear();
        return true;
    }

    bool VulkanContext::waitForSubmittedFrames() {
        return waitForFrameFences();
    }

    bool VulkanContext::deviceWaitIdle() {
        if (device_ == VK_NULL_HANDLE) {
            return true;
        }
        const VkResult result = vkDeviceWaitIdle(device_);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkDeviceWaitIdle failed: {}", vkResultToString(result)));
        }
        last_error_.clear();
        return true;
    }

    void VulkanContext::addFrameTimelineWait(const VkSemaphore semaphore,
                                             const std::uint64_t value,
                                             const VkPipelineStageFlags wait_stage) {
        if (semaphore == VK_NULL_HANDLE) {
            return;
        }
        const VkPipelineStageFlags resolved_wait_stage =
            wait_stage == 0 ? static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) : wait_stage;
        frame_timeline_waits_.push_back(FrameTimelineWait{
            .semaphore = semaphore,
            .value = value,
            .wait_stage = resolved_wait_stage,
        });
    }

    bool VulkanContext::createInstance() {
        uint32_t extension_count = 0;
        const char* const* sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&extension_count);
        if (!sdl_extensions || extension_count == 0) {
            return fail(std::format("SDL_Vulkan_GetInstanceExtensions failed: {}", SDL_GetError()));
        }

        std::vector<const char*> extensions(sdl_extensions, sdl_extensions + extension_count);

        uint32_t available_extension_count = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &available_extension_count, nullptr);
        std::vector<VkExtensionProperties> available_extensions(available_extension_count);
        if (available_extension_count > 0) {
            vkEnumerateInstanceExtensionProperties(nullptr, &available_extension_count, available_extensions.data());
        }
        instance_external_memory_capabilities_enabled_ =
            extensionAvailable(available_extensions, VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
        if (instance_external_memory_capabilities_enabled_) {
            appendUniqueExtension(extensions, VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
        }
        instance_external_semaphore_capabilities_enabled_ =
            extensionAvailable(available_extensions, VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
        if (instance_external_semaphore_capabilities_enabled_) {
            appendUniqueExtension(extensions, VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
        }

        debug_utils_enabled_ = extensionAvailable(available_extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        if (debug_utils_enabled_) {
            appendUniqueExtension(extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        uint32_t available_layer_count = 0;
        vkEnumerateInstanceLayerProperties(&available_layer_count, nullptr);
        std::vector<VkLayerProperties> available_layers(available_layer_count);
        if (available_layer_count > 0) {
            vkEnumerateInstanceLayerProperties(&available_layer_count, available_layers.data());
        }

        std::vector<const char*> layers;
        const bool validation_requested = validationRequestedByBuild();
        const bool validation_layer_available = layerAvailable(available_layers, "VK_LAYER_KHRONOS_validation");
        validation_enabled_ = validation_requested && validation_layer_available && debug_utils_enabled_;
        if (validation_enabled_) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            LOG_INFO("Vulkan validation enabled");
        } else if (validation_requested) {
            if (!validation_layer_available) {
                LOG_WARN("Vulkan validation requested by build type, but VK_LAYER_KHRONOS_validation is unavailable");
            }
            if (!debug_utils_enabled_) {
                LOG_WARN("Vulkan validation requested by build type, but VK_EXT_debug_utils is unavailable");
            }
        }

        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "LichtFeld Studio";
        app_info.applicationVersion = VK_MAKE_VERSION(0, 0, 0);
        app_info.pEngineName = "LichtFeld Studio";
        app_info.engineVersion = VK_MAKE_VERSION(0, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_3;

        VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};
        if (validation_enabled_) {
            populateDebugMessengerCreateInfo(debug_create_info);
        }

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pNext = validation_enabled_ ? &debug_create_info : nullptr;
        create_info.pApplicationInfo = &app_info;
        create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
        create_info.ppEnabledLayerNames = layers.data();
        create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.data();

        const VkResult result = vkCreateInstance(&create_info, nullptr, &instance_);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkCreateInstance failed: {}", vkResultToString(result)));
        }
        if (validation_enabled_ && !createDebugMessenger()) {
            return false;
        }
        return true;
    }

    bool VulkanContext::createSurface(SDL_Window* window) {
        if (!SDL_Vulkan_CreateSurface(window, instance_, nullptr, &surface_)) {
            return fail(std::format("SDL_Vulkan_CreateSurface failed: {}", SDL_GetError()));
        }
        return true;
    }

    VulkanContext::QueueFamilies VulkanContext::findQueueFamilies(VkPhysicalDevice device) const {
        QueueFamilies indices;

        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

        for (uint32_t i = 0; i < count; ++i) {
            if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0 &&
                (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                if (!indices.graphics.has_value())
                    indices.graphics = i;
            }

            VkBool32 present_supported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &present_supported);
            if (present_supported == VK_TRUE) {
                if (!indices.present.has_value())
                    indices.present = i;
            }

            // Async-compute family: compute-capable, NOT graphics-capable. Typical NVIDIA
            // layouts have a dedicated compute family at index 2; AMD has one at 1. If the
            // device exposes only a single graphics+compute family, async_compute stays
            // unset and the rasterizer submits on the graphics queue (correct, just no
            // overlap with UI/swapchain work).
            if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 &&
                (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0 &&
                !indices.async_compute.has_value()) {
                indices.async_compute = i;
            }
        }
        return indices;
    }

    bool VulkanContext::deviceSupportsSwapchain(VkPhysicalDevice device) const {
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> extensions(count);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data());

        std::set<std::string> required{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        for (const auto& extension : extensions) {
            required.erase(extension.extensionName);
        }
        return required.empty();
    }

    VulkanContext::SwapchainSupport VulkanContext::querySwapchainSupport(VkPhysicalDevice device) const {
        SwapchainSupport details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &details.capabilities);

        uint32_t count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &count, nullptr);
        details.formats.resize(count);
        if (count > 0) {
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &count, details.formats.data());
        }

        count = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &count, nullptr);
        details.present_modes.resize(count);
        if (count > 0) {
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &count, details.present_modes.data());
        }

        return details;
    }

    bool VulkanContext::pickPhysicalDevice() {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);
        if (count == 0) {
            return fail("No Vulkan physical devices found");
        }

        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance_, &count, devices.data());

        VkPhysicalDevice fallback = VK_NULL_HANDLE;
        for (const auto device : devices) {
            const QueueFamilies families = findQueueFamilies(device);
            if (!families.complete() || !deviceSupportsSwapchain(device)) {
                continue;
            }

            const SwapchainSupport swapchain = querySwapchainSupport(device);
            if (swapchain.formats.empty() || swapchain.present_modes.empty()) {
                continue;
            }

            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(device, &props);
            if (props.apiVersion < VK_API_VERSION_1_3) {
                LOG_WARN("Skipping Vulkan device '{}' because it exposes Vulkan {}, but 1.3 is required",
                         props.deviceName,
                         vulkanApiVersionString(props.apiVersion));
                continue;
            }

            const RequiredFeatureSupport feature_support = queryRequiredFeatureSupport(device);
            if (!hasRequiredFeatures(feature_support)) {
                LOG_WARN("Skipping Vulkan device '{}' because required Vulkan 1.2/1.3 features are missing: {}",
                         props.deviceName,
                         missingRequiredFeatures(feature_support));
                continue;
            }

            if (fallback == VK_NULL_HANDLE) {
                fallback = device;
            }
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physical_device_ = device;
                break;
            }
        }

        if (physical_device_ == VK_NULL_HANDLE) {
            physical_device_ = fallback;
        }
        if (physical_device_ == VK_NULL_HANDLE) {
            return fail("No Vulkan device supports graphics presentation, swapchain creation, Vulkan 1.3, and required features");
        }

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical_device_, &props);
        LOG_INFO("Vulkan device: {} (API {})", props.deviceName, vulkanApiVersionString(props.apiVersion));

        VkPhysicalDeviceIDProperties id_props{};
        id_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &id_props;
        vkGetPhysicalDeviceProperties2(physical_device_, &props2);
        std::memcpy(device_uuid_.data(), id_props.deviceUUID, VK_UUID_SIZE);
#ifdef _WIN32
        std::memcpy(device_luid_.data(), id_props.deviceLUID, VK_LUID_SIZE);
        device_luid_valid_ = id_props.deviceLUIDValid != VK_FALSE;
        device_node_mask_ = id_props.deviceNodeMask;
#endif
        return true;
    }

    bool VulkanContext::createDevice() {
        const QueueFamilies families = findQueueFamilies(physical_device_);
        if (!families.complete()) {
            return fail("Selected Vulkan device is missing graphics or present queues");
        }

        graphics_queue_family_ = *families.graphics;
        present_queue_family_ = *families.present;

        std::set<uint32_t> unique_families{graphics_queue_family_, present_queue_family_};
        if (families.async_compute.has_value() &&
            *families.async_compute != graphics_queue_family_) {
            unique_families.insert(*families.async_compute);
            compute_queue_family_ = *families.async_compute;
            has_dedicated_compute_queue_ = true;
        } else {
            compute_queue_family_ = graphics_queue_family_;
            has_dedicated_compute_queue_ = false;
        }
        std::vector<VkDeviceQueueCreateInfo> queue_infos;
        constexpr float queue_priority = 1.0f;
        for (const uint32_t family : unique_families) {
            VkDeviceQueueCreateInfo queue_info{};
            queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queue_info.queueFamilyIndex = family;
            queue_info.queueCount = 1;
            queue_info.pQueuePriorities = &queue_priority;
            queue_infos.push_back(queue_info);
        }

        uint32_t available_extension_count = 0;
        vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &available_extension_count, nullptr);
        std::vector<VkExtensionProperties> available_extensions(available_extension_count);
        if (available_extension_count > 0) {
            vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &available_extension_count,
                                                 available_extensions.data());
        }

        std::vector<const char*> extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        const bool has_external_memory =
            instance_external_memory_capabilities_enabled_ &&
            extensionAvailable(available_extensions, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
#ifdef _WIN32
        const bool has_platform_external_memory =
            extensionAvailable(available_extensions, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#else
        const bool has_platform_external_memory =
            extensionAvailable(available_extensions, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
        const bool enable_external_memory = has_external_memory && has_platform_external_memory;
        if (enable_external_memory) {
            appendUniqueExtension(extensions, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
#ifdef _WIN32
            appendUniqueExtension(extensions, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#else
            appendUniqueExtension(extensions, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
        }

        const bool has_external_semaphore =
            instance_external_semaphore_capabilities_enabled_ &&
            extensionAvailable(available_extensions, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#ifdef _WIN32
        const bool has_platform_external_semaphore =
            extensionAvailable(available_extensions, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
        const bool has_platform_external_semaphore =
            extensionAvailable(available_extensions, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif
        const bool enable_external_semaphore = has_external_semaphore && has_platform_external_semaphore;
        if (enable_external_semaphore) {
            appendUniqueExtension(extensions, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#ifdef _WIN32
            appendUniqueExtension(extensions, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
            appendUniqueExtension(extensions, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif
        }

        const bool enable_dedicated_allocation =
            enable_external_memory &&
            extensionAvailable(available_extensions, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME) &&
            extensionAvailable(available_extensions, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
        if (enable_dedicated_allocation) {
            appendUniqueExtension(extensions, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
            appendUniqueExtension(extensions, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
        }

        const bool enable_subgroup_size_control =
            extensionAvailable(available_extensions, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        if (enable_subgroup_size_control) {
            appendUniqueExtension(extensions, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        }
        const bool enable_shader_atomic_float =
            extensionAvailable(available_extensions, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
        if (enable_shader_atomic_float) {
            appendUniqueExtension(extensions, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
        }

        // Phase 2/3 modernization extensions. Each is opportunistic — enabled
        // when present, and code paths that need them gate on the runtime flag
        // exposed via VulkanContext::has*() accessors.
        const bool enable_push_descriptor =
            extensionAvailable(available_extensions, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        if (enable_push_descriptor) {
            appendUniqueExtension(extensions, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        }
        const bool enable_shader_object =
            extensionAvailable(available_extensions, VK_EXT_SHADER_OBJECT_EXTENSION_NAME);
        if (enable_shader_object) {
            appendUniqueExtension(extensions, VK_EXT_SHADER_OBJECT_EXTENSION_NAME);
        }
        const bool enable_extended_dynamic_state3 =
            extensionAvailable(available_extensions, VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);
        if (enable_extended_dynamic_state3) {
            appendUniqueExtension(extensions, VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);
        }
        const bool enable_cooperative_matrix =
            extensionAvailable(available_extensions, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        if (enable_cooperative_matrix) {
            appendUniqueExtension(extensions, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        }
        const bool enable_host_image_copy =
            extensionAvailable(available_extensions, VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME);
        if (enable_host_image_copy) {
            appendUniqueExtension(extensions, VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME);
        }

        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT supported_atomic_float_features{};
        supported_atomic_float_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        VkPhysicalDeviceSubgroupSizeControlFeaturesEXT supported_subgroup_size_control_features{};
        supported_subgroup_size_control_features.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
        supported_subgroup_size_control_features.pNext =
            enable_shader_atomic_float ? static_cast<void*>(&supported_atomic_float_features) : nullptr;
        VkPhysicalDeviceVulkan12Features supported_features12{};
        supported_features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        supported_features12.pNext = enable_subgroup_size_control
                                         ? static_cast<void*>(&supported_subgroup_size_control_features)
                                     : enable_shader_atomic_float
                                         ? static_cast<void*>(&supported_atomic_float_features)
                                         : nullptr;

        // Optional Phase 3/4 modernization features. Each is queried in a
        // throwaway chain so the main supported-features12 chain stays clean.
        VkPhysicalDeviceShaderObjectFeaturesEXT supported_shader_object{};
        supported_shader_object.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
        VkPhysicalDeviceExtendedDynamicState3FeaturesEXT supported_eds3{};
        supported_eds3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR supported_coop_matrix{};
        supported_coop_matrix.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        VkPhysicalDeviceHostImageCopyFeaturesEXT supported_host_image_copy{};
        supported_host_image_copy.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT;

        void* opt_supported_head = nullptr;
        if (enable_shader_object) {
            supported_shader_object.pNext = opt_supported_head;
            opt_supported_head = &supported_shader_object;
        }
        if (enable_extended_dynamic_state3) {
            supported_eds3.pNext = opt_supported_head;
            opt_supported_head = &supported_eds3;
        }
        if (enable_cooperative_matrix) {
            supported_coop_matrix.pNext = opt_supported_head;
            opt_supported_head = &supported_coop_matrix;
        }
        if (enable_host_image_copy) {
            supported_host_image_copy.pNext = opt_supported_head;
            opt_supported_head = &supported_host_image_copy;
        }

        VkPhysicalDeviceFeatures2 supported_features2{};
        supported_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        supported_features2.pNext = &supported_features12;
        vkGetPhysicalDeviceFeatures2(physical_device_, &supported_features2);

        if (opt_supported_head != nullptr) {
            VkPhysicalDeviceFeatures2 opt_query{};
            opt_query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            opt_query.pNext = opt_supported_head;
            vkGetPhysicalDeviceFeatures2(physical_device_, &opt_query);
        }

        VkPhysicalDeviceVulkan13Features features13{};
        features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        features13.synchronization2 = VK_TRUE;
        features13.dynamicRendering = VK_TRUE;

        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features{};
        atomic_float_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        atomic_float_features.pNext = &features13;
        atomic_float_features.shaderBufferFloat32AtomicAdd =
            enable_shader_atomic_float && supported_atomic_float_features.shaderBufferFloat32AtomicAdd
                ? VK_TRUE
                : VK_FALSE;

        VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroup_size_control_features{};
        subgroup_size_control_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
        subgroup_size_control_features.pNext = enable_shader_atomic_float
                                                   ? static_cast<void*>(&atomic_float_features)
                                                   : static_cast<void*>(&features13);
        subgroup_size_control_features.subgroupSizeControl =
            enable_subgroup_size_control && supported_subgroup_size_control_features.subgroupSizeControl
                ? VK_TRUE
                : VK_FALSE;
        subgroup_size_control_features.computeFullSubgroups =
            enable_subgroup_size_control && supported_subgroup_size_control_features.computeFullSubgroups
                ? VK_TRUE
                : VK_FALSE;

        VkPhysicalDeviceVulkan12Features features12{};
        features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        features12.pNext = enable_subgroup_size_control
                               ? static_cast<void*>(&subgroup_size_control_features)
                           : enable_shader_atomic_float
                               ? static_cast<void*>(&atomic_float_features)
                               : static_cast<void*>(&features13);
        features12.timelineSemaphore = VK_TRUE;
        features12.bufferDeviceAddress = VK_TRUE;
        features12.shaderFloat16 = supported_features12.shaderFloat16;
        // Descriptor indexing (bindless). Required for the descriptor-indexing
        // path used by RmlUi + viewport scene/grid bindings (Phase 3 P8).
        // All four are widely supported on NVIDIA and AMD desktop drivers; we
        // mirror device-reported support and let pickPhysicalDevice gate the
        // mandatory subset via hasRequiredFeatures.
        features12.descriptorIndexing = supported_features12.descriptorIndexing;
        features12.shaderSampledImageArrayNonUniformIndexing =
            supported_features12.shaderSampledImageArrayNonUniformIndexing;
        features12.shaderStorageBufferArrayNonUniformIndexing =
            supported_features12.shaderStorageBufferArrayNonUniformIndexing;
        features12.descriptorBindingPartiallyBound =
            supported_features12.descriptorBindingPartiallyBound;
        features12.descriptorBindingSampledImageUpdateAfterBind =
            supported_features12.descriptorBindingSampledImageUpdateAfterBind;
        features12.descriptorBindingUpdateUnusedWhilePending =
            supported_features12.descriptorBindingUpdateUnusedWhilePending;
        features12.descriptorBindingVariableDescriptorCount =
            supported_features12.descriptorBindingVariableDescriptorCount;
        features12.runtimeDescriptorArray = supported_features12.runtimeDescriptorArray;

        // Optional modernization features. Each is enabled only when both the
        // extension was loaded AND the device reported the feature supported.
        // Each struct is prepended to the features12 pNext chain so the existing
        // chain order (subgroup_size_control / atomic_float / features13) stays
        // unchanged.
        void* enabled_chain_head = features12.pNext;

        VkPhysicalDeviceShaderObjectFeaturesEXT shader_object_features{};
        shader_object_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
        const bool enable_shader_object_feature =
            enable_shader_object && supported_shader_object.shaderObject == VK_TRUE;
        if (enable_shader_object_feature) {
            shader_object_features.shaderObject = VK_TRUE;
            shader_object_features.pNext = enabled_chain_head;
            enabled_chain_head = &shader_object_features;
        }

        VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3_features{};
        eds3_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
        const bool enable_eds3_feature =
            enable_extended_dynamic_state3 &&
            (supported_eds3.extendedDynamicState3ColorBlendEnable == VK_TRUE ||
             supported_eds3.extendedDynamicState3ColorBlendEquation == VK_TRUE ||
             supported_eds3.extendedDynamicState3ColorWriteMask == VK_TRUE);
        if (enable_eds3_feature) {
            eds3_features.extendedDynamicState3ColorBlendEnable =
                supported_eds3.extendedDynamicState3ColorBlendEnable;
            eds3_features.extendedDynamicState3ColorBlendEquation =
                supported_eds3.extendedDynamicState3ColorBlendEquation;
            eds3_features.extendedDynamicState3ColorWriteMask =
                supported_eds3.extendedDynamicState3ColorWriteMask;
            eds3_features.pNext = enabled_chain_head;
            enabled_chain_head = &eds3_features;
        }

        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_matrix_features{};
        coop_matrix_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        const bool enable_coop_matrix_feature =
            enable_cooperative_matrix && supported_coop_matrix.cooperativeMatrix == VK_TRUE;
        if (enable_coop_matrix_feature) {
            coop_matrix_features.cooperativeMatrix = VK_TRUE;
            coop_matrix_features.cooperativeMatrixRobustBufferAccess =
                supported_coop_matrix.cooperativeMatrixRobustBufferAccess;
            coop_matrix_features.pNext = enabled_chain_head;
            enabled_chain_head = &coop_matrix_features;
        }

        VkPhysicalDeviceHostImageCopyFeaturesEXT host_image_copy_features{};
        host_image_copy_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT;
        const bool enable_host_image_copy_feature =
            enable_host_image_copy && supported_host_image_copy.hostImageCopy == VK_TRUE;
        if (enable_host_image_copy_feature) {
            host_image_copy_features.hostImageCopy = VK_TRUE;
            host_image_copy_features.pNext = enabled_chain_head;
            enabled_chain_head = &host_image_copy_features;
        }

        features12.pNext = enabled_chain_head;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &features12;
        features2.features.shaderInt16 = supported_features2.features.shaderInt16;
        features2.features.shaderInt64 = supported_features2.features.shaderInt64;

        VkDeviceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.pNext = &features2;
        create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_infos.size());
        create_info.pQueueCreateInfos = queue_infos.data();
        create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.data();

        const VkResult result = vkCreateDevice(physical_device_, &create_info, nullptr, &device_);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkCreateDevice failed: {}", vkResultToString(result)));
        }

        if (debug_utils_enabled_) {
            vk_set_debug_utils_object_name_ = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
                vkGetDeviceProcAddr(device_, "vkSetDebugUtilsObjectNameEXT"));
            if (vk_set_debug_utils_object_name_ == nullptr) {
                LOG_WARN("VK_EXT_debug_utils is enabled, but vkSetDebugUtilsObjectNameEXT could not be loaded");
            }
        }
        if (enable_push_descriptor) {
            vk_cmd_push_descriptor_set_ = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
                vkGetDeviceProcAddr(device_, "vkCmdPushDescriptorSetKHR"));
            if (vk_cmd_push_descriptor_set_ == nullptr) {
                return fail("VK_KHR_push_descriptor is enabled but vkCmdPushDescriptorSetKHR could not be loaded");
            }
        }

        vkGetDeviceQueue(device_, graphics_queue_family_, 0, &graphics_queue_);
        vkGetDeviceQueue(device_, present_queue_family_, 0, &present_queue_);
        if (has_dedicated_compute_queue_) {
            vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);
            LOG_INFO("Vulkan: dedicated async-compute queue family {} (graphics family {})",
                     compute_queue_family_, graphics_queue_family_);
        } else {
            // Alias graphics so callers can submit unconditionally on computeQueue().
            compute_queue_ = graphics_queue_;
            LOG_INFO("Vulkan: no dedicated async-compute family; sharing graphics queue family {}",
                     graphics_queue_family_);
        }
        setDebugObjectName(VK_OBJECT_TYPE_DEVICE, device_, "LichtFeld Vulkan device");
        external_memory_interop_enabled_ = enable_external_memory;
        external_semaphore_interop_enabled_ = enable_external_semaphore;
        external_memory_dedicated_allocation_enabled_ = enable_dedicated_allocation;
        has_push_descriptor_ = enable_push_descriptor;
        has_shader_object_ = enable_shader_object_feature;
        has_extended_dynamic_state3_ = enable_eds3_feature;
        has_cooperative_matrix_ = enable_coop_matrix_feature;
        has_host_image_copy_ = enable_host_image_copy_feature;
        has_descriptor_indexing_ = supported_features12.descriptorIndexing == VK_TRUE;
        if (!external_memory_interop_enabled_) {
            return fail("Vulkan external memory interop is required (KHR_external_memory + platform variant); device is missing the extension(s)");
        }
        if (!external_semaphore_interop_enabled_) {
            return fail("Vulkan external timeline-semaphore interop is required (KHR_external_semaphore + platform variant); device is missing the extension(s)");
        }
        LOG_INFO("Vulkan external memory interop enabled{}",
                 external_memory_dedicated_allocation_enabled_ ? " with dedicated allocations" : "");
        LOG_INFO("Vulkan external timeline semaphore interop enabled");
        LOG_INFO("Vulkan optional features: descriptor_indexing={} push_descriptor={} shader_object={} extended_dynamic_state3={} cooperative_matrix={} host_image_copy={}",
                 has_descriptor_indexing_,
                 has_push_descriptor_,
                 has_shader_object_,
                 has_extended_dynamic_state3_,
                 has_cooperative_matrix_,
                 has_host_image_copy_);
        return true;
    }

    bool VulkanContext::createAllocator() {
        if (instance_ == VK_NULL_HANDLE || physical_device_ == VK_NULL_HANDLE || device_ == VK_NULL_HANDLE) {
            return fail("VMA allocator requires an initialized Vulkan instance, physical device, and device");
        }

        VmaAllocatorCreateInfo create_info{};
        create_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        create_info.physicalDevice = physical_device_;
        create_info.device = device_;
        create_info.instance = instance_;
        create_info.vulkanApiVersion = VK_API_VERSION_1_3;

        const VkResult result = vmaCreateAllocator(&create_info, &allocator_);
        if (result != VK_SUCCESS) {
            allocator_ = VK_NULL_HANDLE;
            return fail(std::format("vmaCreateAllocator failed: {}", vkResultToString(result)));
        }
        return true;
    }

    VkSurfaceFormatKHR VulkanContext::chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const {
        constexpr std::array preferred_formats{
            VK_FORMAT_B8G8R8A8_UNORM,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_FORMAT_A8B8G8R8_UNORM_PACK32,
        };

        for (const VkFormat preferred_format : preferred_formats) {
            for (const auto& format : formats) {
                if (format.format == preferred_format &&
                    format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    return format;
                }
            }
        }

        for (const auto& format : formats) {
            if (format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return format;
            }
        }
        return formats.front();
    }

    VkPresentModeKHR VulkanContext::choosePresentMode(const std::vector<VkPresentModeKHR>& modes) const {
        for (const auto mode : modes) {
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return mode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D VulkanContext::chooseSwapchainExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                                                    const int framebuffer_width,
                                                    const int framebuffer_height) const {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }

        VkExtent2D extent{};
        extent.width = static_cast<uint32_t>(std::max(1, framebuffer_width));
        extent.height = static_cast<uint32_t>(std::max(1, framebuffer_height));
        extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        return extent;
    }

    VkFormat VulkanContext::chooseDepthStencilFormat() const {
        constexpr std::array<VkFormat, 3> formats{
            VK_FORMAT_D32_SFLOAT_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT,
            VK_FORMAT_D16_UNORM_S8_UINT,
        };

        for (const VkFormat format : formats) {
            VkFormatProperties properties{};
            vkGetPhysicalDeviceFormatProperties(physical_device_, format, &properties);
            if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
                return format;
            }
        }
        return VK_FORMAT_UNDEFINED;
    }

    uint32_t VulkanContext::findMemoryType(const uint32_t type_filter, const VkMemoryPropertyFlags properties) const {
        VkPhysicalDeviceMemoryProperties memory_properties{};
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties);

        for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            const bool supported = (type_filter & (1u << i)) != 0;
            const bool matches = (memory_properties.memoryTypes[i].propertyFlags & properties) == properties;
            if (supported && matches) {
                return i;
            }
        }
        return std::numeric_limits<uint32_t>::max();
    }

    VkImageAspectFlags VulkanContext::depthStencilAspectMask() const {
        switch (depth_stencil_format_) {
        case VK_FORMAT_D16_UNORM_S8_UINT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        default:
            return VK_IMAGE_ASPECT_DEPTH_BIT;
        }
    }

    bool VulkanContext::externalNativeHandleValid(const ExternalNativeHandle handle) {
#ifdef _WIN32
        return handle != nullptr;
#else
        return handle >= 0;
#endif
    }

    void VulkanContext::closeExternalNativeHandle(ExternalNativeHandle& handle) const {
        if (!externalNativeHandleValid(handle)) {
            return;
        }
#ifdef _WIN32
        if (handle != nullptr) {
            CloseHandle(static_cast<HANDLE>(handle));
            handle = nullptr;
        }
#else
        if (handle >= 0) {
            ::close(handle);
            handle = -1;
        }
#endif
    }

    bool VulkanContext::createExternalImage(const VkExtent2D extent, const VkFormat format, ExternalImage& out) {
        out = {};

        if (!device_ || !physical_device_) {
            return fail("Cannot create external Vulkan image before device initialization");
        }
        if (extent.width == 0 || extent.height == 0 || format == VK_FORMAT_UNDEFINED) {
            return fail("External Vulkan image requires a non-zero extent and defined format");
        }

        constexpr VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                            VK_IMAGE_USAGE_STORAGE_BIT |
                                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                            VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        VkPhysicalDeviceExternalImageFormatInfo external_format_info{};
        external_format_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
        external_format_info.handleType = kExternalMemoryHandleType;

        VkPhysicalDeviceImageFormatInfo2 format_info{};
        format_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
        format_info.pNext = &external_format_info;
        format_info.format = format;
        format_info.type = VK_IMAGE_TYPE_2D;
        format_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        format_info.usage = usage;

        VkExternalImageFormatProperties external_format_properties{};
        external_format_properties.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;

        VkImageFormatProperties2 format_properties{};
        format_properties.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
        format_properties.pNext = &external_format_properties;

        VkResult result = vkGetPhysicalDeviceImageFormatProperties2(physical_device_, &format_info, &format_properties);
        if (result != VK_SUCCESS) {
            return fail(std::format("External Vulkan image format is unsupported: {}", vkResultToString(result)));
        }
        if ((external_format_properties.externalMemoryProperties.externalMemoryFeatures &
             VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) == 0) {
            return fail("External Vulkan image format is not exportable");
        }
        if ((external_format_properties.externalMemoryProperties.externalMemoryFeatures &
             VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) != 0 &&
            !external_memory_dedicated_allocation_enabled_) {
            return fail("External Vulkan image format requires dedicated allocation support");
        }

        out.extent = extent;
        out.format = format;

        VkExternalMemoryImageCreateInfo external_image_info{};
        external_image_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        external_image_info.handleTypes = kExternalMemoryHandleType;

        VkImageCreateInfo image_info{};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.pNext = &external_image_info;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.extent.width = extent.width;
        image_info.extent.height = extent.height;
        image_info.extent.depth = 1;
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.format = format;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_info.usage = usage;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        // External images written on the async-compute queue and sampled on the
        // graphics queue need either SHARING_MODE_CONCURRENT or paired ownership-
        // transfer barriers. CONCURRENT trades a tiny driver-side overhead for the
        // ability to drop the transfer barriers entirely; the spec-mandated
        // alternative is fragile when the producer/consumer queue choice can vary.
        std::array<uint32_t, 2> external_image_families{
            graphics_queue_family_,
            has_dedicated_compute_queue_ ? compute_queue_family_ : graphics_queue_family_};
        if (has_dedicated_compute_queue_) {
            image_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
            image_info.queueFamilyIndexCount = static_cast<uint32_t>(external_image_families.size());
            image_info.pQueueFamilyIndices = external_image_families.data();
        } else {
            image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        result = vkCreateImage(device_, &image_info, nullptr, &out.image);
        if (result != VK_SUCCESS) {
            out = {};
            return fail(std::format("vkCreateImage(external) failed: {}", vkResultToString(result)));
        }
        setDebugObjectName(VK_OBJECT_TYPE_IMAGE, out.image,
                           std::format("External image {}x{}", extent.width, extent.height));

        VkMemoryRequirements memory_requirements{};
        vkGetImageMemoryRequirements(device_, out.image, &memory_requirements);
        out.allocation_size = memory_requirements.size;

        VkMemoryDedicatedAllocateInfo dedicated_info{};
        dedicated_info.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        dedicated_info.image = out.image;

        VkExportMemoryAllocateInfo export_info{};
        export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        export_info.handleTypes = kExternalMemoryHandleType;
        if (external_memory_dedicated_allocation_enabled_) {
            export_info.pNext = &dedicated_info;
        }

        VkMemoryAllocateInfo allocate_info{};
        allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate_info.pNext = &export_info;
        allocate_info.allocationSize = memory_requirements.size;
        allocate_info.memoryTypeIndex = findMemoryType(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (allocate_info.memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
            destroyExternalImage(out);
            return fail("Could not find Vulkan device-local memory for external image");
        }

        // Keep this allocation manual: CUDA interop needs exportable VkDeviceMemory with
        // the external-memory pNext chain intact so we can export an OS handle below.
        result = vkAllocateMemory(device_, &allocate_info, nullptr, &out.memory);
        if (result != VK_SUCCESS) {
            destroyExternalImage(out);
            return fail(std::format("vkAllocateMemory(external image) failed: {}", vkResultToString(result)));
        }
        setDebugObjectName(VK_OBJECT_TYPE_DEVICE_MEMORY, out.memory, "External image memory");

        result = vkBindImageMemory(device_, out.image, out.memory, 0);
        if (result != VK_SUCCESS) {
            destroyExternalImage(out);
            return fail(std::format("vkBindImageMemory(external image) failed: {}", vkResultToString(result)));
        }

#ifdef _WIN32
        auto get_memory_handle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
            vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandleKHR"));
        if (get_memory_handle == nullptr) {
            destroyExternalImage(out);
            return fail("vkGetMemoryWin32HandleKHR is unavailable");
        }
        VkMemoryGetWin32HandleInfoKHR handle_info{};
        handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        handle_info.memory = out.memory;
        handle_info.handleType = kExternalMemoryHandleType;
        HANDLE native_handle = nullptr;
        result = get_memory_handle(device_, &handle_info, &native_handle);
        out.native_handle = native_handle;
#else
        auto get_memory_fd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
            vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR"));
        if (get_memory_fd == nullptr) {
            destroyExternalImage(out);
            return fail("vkGetMemoryFdKHR is unavailable");
        }
        VkMemoryGetFdInfoKHR fd_info{};
        fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        fd_info.memory = out.memory;
        fd_info.handleType = kExternalMemoryHandleType;
        int native_handle = -1;
        result = get_memory_fd(device_, &fd_info, &native_handle);
        out.native_handle = native_handle;
#endif
        if (result != VK_SUCCESS) {
            destroyExternalImage(out);
            return fail(std::format("Exporting external image memory handle failed: {}", vkResultToString(result)));
        }

        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = out.image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = format;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.layerCount = 1;

        result = vkCreateImageView(device_, &view_info, nullptr, &out.view);
        if (result != VK_SUCCESS) {
            destroyExternalImage(out);
            return fail(std::format("vkCreateImageView(external image) failed: {}", vkResultToString(result)));
        }
        setDebugObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, out.view, "External image view");
        return true;
    }

    void VulkanContext::destroyExternalImage(ExternalImage& image) {
        if (device_) {
            if (image.view != VK_NULL_HANDLE) {
                vkDestroyImageView(device_, image.view, nullptr);
            }
            if (image.image != VK_NULL_HANDLE) {
                vkDestroyImage(device_, image.image, nullptr);
            }
            if (image.memory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, image.memory, nullptr);
            }
        }
        closeExternalNativeHandle(image.native_handle);
        image = {};
    }

    VulkanContext::ExternalNativeHandle VulkanContext::releaseExternalImageNativeHandle(ExternalImage& image) const {
        const ExternalNativeHandle handle = image.native_handle;
        image.native_handle = kInvalidExternalNativeHandle;
        return handle;
    }

    bool VulkanContext::createExternalBuffer(const VkDeviceSize size,
                                             const VkBufferUsageFlags usage,
                                             ExternalBuffer& out) {
        out = {};

        if (!device_ || !physical_device_) {
            return fail("Cannot create external Vulkan buffer before device initialization");
        }
        if (size == 0) {
            return fail("External Vulkan buffer requires a non-zero size");
        }

        VkExternalMemoryBufferCreateInfo external_buffer_info{};
        external_buffer_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        external_buffer_info.handleTypes = kExternalMemoryHandleType;

        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.pNext = &external_buffer_info;
        buffer_info.size = size;
        buffer_info.usage = usage |
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        // External buffers are CUDA-written and Vulkan-read; with a dedicated async-
        // compute queue, the read may happen on a different family than the implicit
        // graphics submit lane. CONCURRENT avoids the need for ownership-transfer
        // barriers on every cross-API handoff. See createExternalImage for the same
        // reasoning.
        std::array<uint32_t, 2> external_buffer_families{
            graphics_queue_family_,
            has_dedicated_compute_queue_ ? compute_queue_family_ : graphics_queue_family_};
        if (has_dedicated_compute_queue_) {
            buffer_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
            buffer_info.queueFamilyIndexCount = static_cast<uint32_t>(external_buffer_families.size());
            buffer_info.pQueueFamilyIndices = external_buffer_families.data();
        } else {
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        VkResult result = vkCreateBuffer(device_, &buffer_info, nullptr, &out.buffer);
        if (result != VK_SUCCESS) {
            out = {};
            return fail(std::format("vkCreateBuffer(external) failed: {}", vkResultToString(result)));
        }
        setDebugObjectName(VK_OBJECT_TYPE_BUFFER, out.buffer, std::format("External buffer {} bytes", size));

        VkMemoryRequirements memory_requirements{};
        vkGetBufferMemoryRequirements(device_, out.buffer, &memory_requirements);
        out.size = size;
        out.allocation_size = memory_requirements.size;

        VkExportMemoryAllocateInfo export_info{};
        export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        export_info.handleTypes = kExternalMemoryHandleType;

        VkMemoryAllocateInfo allocate_info{};
        allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate_info.pNext = &export_info;
        allocate_info.allocationSize = memory_requirements.size;
        allocate_info.memoryTypeIndex =
            findMemoryType(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (allocate_info.memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
            destroyExternalBuffer(out);
            return fail("Could not find Vulkan device-local memory for external buffer");
        }

        result = vkAllocateMemory(device_, &allocate_info, nullptr, &out.memory);
        if (result != VK_SUCCESS) {
            destroyExternalBuffer(out);
            return fail(std::format("vkAllocateMemory(external buffer) failed: {}", vkResultToString(result)));
        }
        setDebugObjectName(VK_OBJECT_TYPE_DEVICE_MEMORY, out.memory, "External buffer memory");

        result = vkBindBufferMemory(device_, out.buffer, out.memory, 0);
        if (result != VK_SUCCESS) {
            destroyExternalBuffer(out);
            return fail(std::format("vkBindBufferMemory(external buffer) failed: {}", vkResultToString(result)));
        }

#ifdef _WIN32
        auto get_memory_handle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
            vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandleKHR"));
        if (get_memory_handle == nullptr) {
            destroyExternalBuffer(out);
            return fail("vkGetMemoryWin32HandleKHR is unavailable");
        }
        VkMemoryGetWin32HandleInfoKHR handle_info{};
        handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        handle_info.memory = out.memory;
        handle_info.handleType = kExternalMemoryHandleType;
        HANDLE native_handle = nullptr;
        result = get_memory_handle(device_, &handle_info, &native_handle);
        out.native_handle = native_handle;
#else
        auto get_memory_fd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
            vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR"));
        if (get_memory_fd == nullptr) {
            destroyExternalBuffer(out);
            return fail("vkGetMemoryFdKHR is unavailable");
        }
        VkMemoryGetFdInfoKHR fd_info{};
        fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        fd_info.memory = out.memory;
        fd_info.handleType = kExternalMemoryHandleType;
        int native_handle = -1;
        result = get_memory_fd(device_, &fd_info, &native_handle);
        out.native_handle = native_handle;
#endif
        if (result != VK_SUCCESS) {
            destroyExternalBuffer(out);
            return fail(std::format("Exporting external buffer memory handle failed: {}", vkResultToString(result)));
        }
        return true;
    }

    void VulkanContext::destroyExternalBuffer(ExternalBuffer& buffer) {
        if (device_) {
            if (buffer.buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device_, buffer.buffer, nullptr);
            }
            if (buffer.memory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, buffer.memory, nullptr);
            }
        }
        closeExternalNativeHandle(buffer.native_handle);
        buffer = {};
    }

    VulkanContext::ExternalNativeHandle VulkanContext::releaseExternalBufferNativeHandle(ExternalBuffer& buffer) const {
        const ExternalNativeHandle handle = buffer.native_handle;
        buffer.native_handle = kInvalidExternalNativeHandle;
        return handle;
    }

    bool VulkanContext::createExternalTimelineSemaphore(const std::uint64_t initial_value, ExternalSemaphore& out) {
        out = {};

        if (!device_ || !physical_device_) {
            return fail("Cannot create external Vulkan semaphore before device initialization");
        }

        VkPhysicalDeviceExternalSemaphoreInfo semaphore_info{};
        semaphore_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
        semaphore_info.handleType = kExternalSemaphoreHandleType;

        VkExternalSemaphoreProperties semaphore_properties{};
        semaphore_properties.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;
        vkGetPhysicalDeviceExternalSemaphoreProperties(physical_device_, &semaphore_info, &semaphore_properties);
        if ((semaphore_properties.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) == 0) {
            return fail("External Vulkan timeline semaphore handle is not exportable");
        }

        out.initial_value = initial_value;

        VkExportSemaphoreCreateInfo export_info{};
        export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
        export_info.handleTypes = kExternalSemaphoreHandleType;

        VkSemaphoreTypeCreateInfo type_info{};
        type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        type_info.pNext = &export_info;
        type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        type_info.initialValue = initial_value;

        VkSemaphoreCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        create_info.pNext = &type_info;

        VkResult result = vkCreateSemaphore(device_, &create_info, nullptr, &out.semaphore);
        if (result != VK_SUCCESS) {
            out = {};
            return fail(std::format("vkCreateSemaphore(external timeline) failed: {}", vkResultToString(result)));
        }
        setDebugObjectName(VK_OBJECT_TYPE_SEMAPHORE, out.semaphore, "External timeline semaphore");

#ifdef _WIN32
        auto get_semaphore_handle = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
            vkGetDeviceProcAddr(device_, "vkGetSemaphoreWin32HandleKHR"));
        if (get_semaphore_handle == nullptr) {
            destroyExternalSemaphore(out);
            return fail("vkGetSemaphoreWin32HandleKHR is unavailable");
        }
        VkSemaphoreGetWin32HandleInfoKHR handle_info{};
        handle_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
        handle_info.semaphore = out.semaphore;
        handle_info.handleType = kExternalSemaphoreHandleType;
        HANDLE native_handle = nullptr;
        result = get_semaphore_handle(device_, &handle_info, &native_handle);
        out.native_handle = native_handle;
#else
        auto get_semaphore_fd = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
            vkGetDeviceProcAddr(device_, "vkGetSemaphoreFdKHR"));
        if (get_semaphore_fd == nullptr) {
            destroyExternalSemaphore(out);
            return fail("vkGetSemaphoreFdKHR is unavailable");
        }
        VkSemaphoreGetFdInfoKHR fd_info{};
        fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
        fd_info.semaphore = out.semaphore;
        fd_info.handleType = kExternalSemaphoreHandleType;
        int native_handle = -1;
        result = get_semaphore_fd(device_, &fd_info, &native_handle);
        out.native_handle = native_handle;
#endif
        if (result != VK_SUCCESS) {
            destroyExternalSemaphore(out);
            return fail(std::format("Exporting external timeline semaphore handle failed: {}", vkResultToString(result)));
        }
        return true;
    }

    void VulkanContext::destroyExternalSemaphore(ExternalSemaphore& semaphore) {
        if (device_ && semaphore.semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, semaphore.semaphore, nullptr);
        }
        closeExternalNativeHandle(semaphore.native_handle);
        semaphore = {};
    }

    VulkanContext::ExternalNativeHandle VulkanContext::releaseExternalSemaphoreNativeHandle(
        ExternalSemaphore& semaphore) const {
        const ExternalNativeHandle handle = semaphore.native_handle;
        semaphore.native_handle = kInvalidExternalNativeHandle;
        return handle;
    }

    void VulkanContext::drainCompletedImmediateSubmits() {
        if (device_ == VK_NULL_HANDLE || pending_immediate_submits_.empty()) {
            return;
        }
        auto write = pending_immediate_submits_.begin();
        for (auto read = pending_immediate_submits_.begin(); read != pending_immediate_submits_.end(); ++read) {
            const VkResult status = vkGetFenceStatus(device_, read->fence);
            if (status == VK_SUCCESS) {
                vkDestroyFence(device_, read->fence, nullptr);
                vkFreeCommandBuffers(device_, immediate_command_pool_, 1, &read->cmd);
            } else {
                if (write != read) {
                    *write = *read;
                }
                ++write;
            }
        }
        pending_immediate_submits_.erase(write, pending_immediate_submits_.end());
    }

    bool VulkanContext::transitionImageLayoutImmediate(const VkImage image,
                                                       const VkImageLayout old_layout,
                                                       const VkImageLayout new_layout,
                                                       const VkImageAspectFlags aspect_mask,
                                                       const VkSemaphore wait_semaphore,
                                                       const std::uint64_t wait_value,
                                                       const VkPipelineStageFlags wait_stage) {
        if (device_ == VK_NULL_HANDLE || immediate_command_pool_ == VK_NULL_HANDLE ||
            graphics_queue_ == VK_NULL_HANDLE || image == VK_NULL_HANDLE) {
            return fail("Cannot transition Vulkan image layout before graphics resources are initialized");
        }
        if (frame_active_) {
            return fail("Immediate Vulkan image layout transitions cannot run during an active frame");
        }
        if (old_layout == new_layout) {
            last_error_.clear();
            return true;
        }
        // Reap any prior fire-and-forget submits that have completed.
        drainCompletedImmediateSubmits();

        VkCommandBufferAllocateInfo allocate_info{};
        allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate_info.commandPool = immediate_command_pool_;
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = 1;

        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkResult result = vkAllocateCommandBuffers(device_, &allocate_info, &command_buffer);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkAllocateCommandBuffers(layout transition) failed: {}", vkResultToString(result)));
        }

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = vkBeginCommandBuffer(command_buffer, &begin_info);
        if (result != VK_SUCCESS) {
            vkFreeCommandBuffers(device_, immediate_command_pool_, 1, &command_buffer);
            return fail(std::format("vkBeginCommandBuffer(layout transition) failed: {}", vkResultToString(result)));
        }

        VkPipelineStageFlags2 src_stage = VK_PIPELINE_STAGE_2_NONE;
        VkPipelineStageFlags2 dst_stage = VK_PIPELINE_STAGE_2_NONE;
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.oldLayout = old_layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = aspect_mask;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        switch (old_layout) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            barrier.srcAccessMask = VK_ACCESS_2_NONE;
            src_stage = VK_PIPELINE_STAGE_2_NONE;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            src_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            src_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            break;
        case VK_IMAGE_LAYOUT_GENERAL:
            barrier.srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            src_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            break;
        default:
            barrier.srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            src_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            break;
        }

        switch (new_layout) {
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            dst_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            dst_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            break;
        case VK_IMAGE_LAYOUT_GENERAL:
            barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            dst_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            break;
        default:
            barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            dst_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            break;
        }
        barrier.srcStageMask = src_stage;
        barrier.dstStageMask = dst_stage;

        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(command_buffer, &dependency);

        result = vkEndCommandBuffer(command_buffer);
        if (result != VK_SUCCESS) {
            vkFreeCommandBuffers(device_, immediate_command_pool_, 1, &command_buffer);
            return fail(std::format("vkEndCommandBuffer(layout transition) failed: {}", vkResultToString(result)));
        }

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        VkTimelineSemaphoreSubmitInfo timeline_submit_info{};
        VkPipelineStageFlags resolved_wait_stage = wait_stage == 0
                                                       ? static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
                                                       : wait_stage;
        // CPU-side vkWaitSemaphores removed — the submit-time wait below
        // already gates the GPU on the external (CUDA) timeline. Blocking the
        // CPU here doubled the cost of every CUDA→Vulkan handoff (3-9ms/frame
        // observed). The submit's pWaitSemaphores entry is sufficient.
        if (wait_semaphore != VK_NULL_HANDLE) {
            timeline_submit_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timeline_submit_info.waitSemaphoreValueCount = 1;
            timeline_submit_info.pWaitSemaphoreValues = &wait_value;
            submit_info.pNext = &timeline_submit_info;
            submit_info.waitSemaphoreCount = 1;
            submit_info.pWaitSemaphores = &wait_semaphore;
            submit_info.pWaitDstStageMask = &resolved_wait_stage;
        }
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence submit_fence = VK_NULL_HANDLE;
        result = vkCreateFence(device_, &fence_info, nullptr, &submit_fence);
        if (result != VK_SUCCESS) {
            vkFreeCommandBuffers(device_, immediate_command_pool_, 1, &command_buffer);
            return fail(std::format("vkCreateFence(layout transition) failed: {}", vkResultToString(result)));
        }

        result = vkQueueSubmit(graphics_queue_, 1, &submit_info, submit_fence);
        if (result != VK_SUCCESS) {
            vkDestroyFence(device_, submit_fence, nullptr);
            vkFreeCommandBuffers(device_, immediate_command_pool_, 1, &command_buffer);
            return fail(std::format("Immediate Vulkan image layout transition submit failed: {}", vkResultToString(result)));
        }
        // Fire-and-forget: queue cmd+fence for lazy reaping. Vulkan queues are
        // FIFO per VkQueue, so subsequent submits on graphics_queue_ correctly
        // observe the layout transition without any CPU-side wait.
        pending_immediate_submits_.push_back({command_buffer, submit_fence});
        last_error_.clear();
        return true;
    }

    bool VulkanContext::createSwapchain(const int framebuffer_width, const int framebuffer_height) {
        const SwapchainSupport support = querySwapchainSupport(physical_device_);
        if (support.formats.empty() || support.present_modes.empty()) {
            return fail("Vulkan swapchain support is incomplete");
        }

        const VkSurfaceFormatKHR surface_format = chooseSurfaceFormat(support.formats);
        const VkPresentModeKHR present_mode = choosePresentMode(support.present_modes);
        const VkExtent2D extent = chooseSwapchainExtent(support.capabilities, framebuffer_width, framebuffer_height);

        if ((support.capabilities.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0) {
            return fail("Vulkan swapchain does not support color attachment usage");
        }

        uint32_t image_count = support.capabilities.minImageCount + 1;
        if (support.capabilities.maxImageCount > 0 && image_count > support.capabilities.maxImageCount) {
            image_count = support.capabilities.maxImageCount;
        }
        min_image_count_ = std::max(2u, support.capabilities.minImageCount);

        const std::array<uint32_t, 2> queue_indices{graphics_queue_family_, present_queue_family_};
        const bool shared_queues = graphics_queue_family_ != present_queue_family_;
        VkSwapchainCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        create_info.surface = surface_;
        create_info.minImageCount = image_count;
        create_info.imageFormat = surface_format.format;
        create_info.imageColorSpace = surface_format.colorSpace;
        create_info.imageExtent = extent;
        create_info.imageArrayLayers = 1;
        create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if ((support.capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0) {
            create_info.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        }
        create_info.imageSharingMode = shared_queues ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
        create_info.queueFamilyIndexCount = shared_queues ? static_cast<uint32_t>(queue_indices.size()) : 0u;
        create_info.pQueueFamilyIndices = shared_queues ? queue_indices.data() : nullptr;
        create_info.preTransform = support.capabilities.currentTransform;
        create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        create_info.presentMode = present_mode;
        create_info.clipped = VK_TRUE;
        create_info.oldSwapchain = VK_NULL_HANDLE;

        const VkResult result = vkCreateSwapchainKHR(device_, &create_info, nullptr, &swapchain_);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkCreateSwapchainKHR failed: {}", vkResultToString(result)));
        }
        setDebugObjectName(VK_OBJECT_TYPE_SWAPCHAIN_KHR, swapchain_, "Main swapchain");

        vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr);
        swapchain_images_.resize(image_count);
        vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, swapchain_images_.data());
        swapchain_images_in_flight_.assign(image_count, VK_NULL_HANDLE);
        swapchain_format_ = surface_format.format;
        swapchain_color_space_ = surface_format.colorSpace;
        has_hdr_ = surface_format.colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        LOG_INFO("Vulkan swapchain: {} images, format {}, color space {}{}",
                 image_count,
                 vkFormatToString(surface_format.format),
                 vkColorSpaceToString(surface_format.colorSpace),
                 has_hdr_ ? " (HDR-capable)" : "");

        // One image-available semaphore per swapchain image (NOT per frame slot). The
        // active index is captured in beginFrame and held until endFrame's submit waits
        // on it, so a semaphore is never reused before its signal has been consumed.
        VkSemaphoreCreateInfo image_avail_info{};
        image_avail_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        image_available_.assign(image_count, VK_NULL_HANDLE);
        for (std::uint32_t i = 0; i < image_count; ++i) {
            const VkResult sem_result =
                vkCreateSemaphore(device_, &image_avail_info, nullptr, &image_available_[i]);
            if (sem_result != VK_SUCCESS) {
                return fail(std::format("vkCreateSemaphore(image_available {}) failed: {}",
                                        i, vkResultToString(sem_result)));
            }
            setDebugObjectName(VK_OBJECT_TYPE_SEMAPHORE,
                               image_available_[i],
                               std::format("Image-available semaphore {}", i));
        }
        next_acquire_index_ = 0;
        active_acquire_index_ = 0;
        swapchain_extent_ = extent;
        swapchain_image_usage_ = create_info.imageUsage;
        for (size_t i = 0; i < swapchain_images_.size(); ++i) {
            image_barriers_.registerImage(swapchain_images_[i],
                                          VK_IMAGE_ASPECT_COLOR_BIT,
                                          VK_IMAGE_LAYOUT_UNDEFINED);
            setDebugObjectName(VK_OBJECT_TYPE_IMAGE,
                               swapchain_images_[i],
                               std::format("Swapchain image {}", i));
        }
        return true;
    }

    bool VulkanContext::createImageViews() {
        swapchain_image_views_.resize(swapchain_images_.size());
        for (size_t i = 0; i < swapchain_images_.size(); ++i) {
            VkImageViewCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            create_info.image = swapchain_images_[i];
            create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            create_info.format = swapchain_format_;
            create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            create_info.subresourceRange.baseMipLevel = 0;
            create_info.subresourceRange.levelCount = 1;
            create_info.subresourceRange.baseArrayLayer = 0;
            create_info.subresourceRange.layerCount = 1;

            const VkResult result = vkCreateImageView(device_, &create_info, nullptr, &swapchain_image_views_[i]);
            if (result != VK_SUCCESS) {
                return fail(std::format("vkCreateImageView failed: {}", vkResultToString(result)));
            }
            setDebugObjectName(VK_OBJECT_TYPE_IMAGE_VIEW,
                               swapchain_image_views_[i],
                               std::format("Swapchain image view {}", i));
        }
        return true;
    }

    bool VulkanContext::createDepthStencilResources() {
        if (allocator_ == VK_NULL_HANDLE) {
            return fail("Cannot create depth/stencil resources before VMA allocator initialization");
        }
        if (depth_stencil_format_ == VK_FORMAT_UNDEFINED) {
            depth_stencil_format_ = chooseDepthStencilFormat();
            if (depth_stencil_format_ == VK_FORMAT_UNDEFINED) {
                return fail("No supported Vulkan depth/stencil format found");
            }
        }
        if (swapchain_images_.empty()) {
            return fail("Cannot create depth/stencil resources before swapchain images");
        }

        VkImageCreateInfo image_info{};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.extent.width = swapchain_extent_.width;
        image_info.extent.height = swapchain_extent_.height;
        image_info.extent.depth = 1;
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.format = depth_stencil_format_;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = depth_stencil_format_;
        view_info.subresourceRange.aspectMask = depthStencilAspectMask();
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;

        depth_stencil_resources_.assign(swapchain_images_.size(), {});
        const auto destroy_created = [&]() {
            for (DepthStencilResource& resource : depth_stencil_resources_) {
                if (resource.view != VK_NULL_HANDLE) {
                    vkDestroyImageView(device_, resource.view, nullptr);
                    resource.view = VK_NULL_HANDLE;
                }
                if (resource.image != VK_NULL_HANDLE) {
                    vmaDestroyImage(allocator_, resource.image, resource.allocation);
                    resource.image = VK_NULL_HANDLE;
                }
                resource.allocation = VK_NULL_HANDLE;
            }
            depth_stencil_resources_.clear();
        };

        VmaAllocationCreateInfo allocation_info{};
        allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        for (std::size_t i = 0; i < depth_stencil_resources_.size(); ++i) {
            DepthStencilResource& resource = depth_stencil_resources_[i];
            VkResult result = vmaCreateImage(allocator_,
                                             &image_info,
                                             &allocation_info,
                                             &resource.image,
                                             &resource.allocation,
                                             nullptr);
            if (result != VK_SUCCESS) {
                destroy_created();
                return fail(std::format("vmaCreateImage(depth/stencil {}) failed: {}", i, vkResultToString(result)));
            }
            setDebugObjectName(VK_OBJECT_TYPE_IMAGE, resource.image, std::format("Depth/stencil image {}", i));
            const std::string allocation_name = std::format("Depth/stencil allocation {}", i);
            vmaSetAllocationName(allocator_, resource.allocation, allocation_name.c_str());

            view_info.image = resource.image;
            result = vkCreateImageView(device_, &view_info, nullptr, &resource.view);
            if (result != VK_SUCCESS) {
                destroy_created();
                return fail(std::format("vkCreateImageView(depth/stencil {}) failed: {}", i, vkResultToString(result)));
            }
            setDebugObjectName(VK_OBJECT_TYPE_IMAGE_VIEW,
                               resource.view,
                               std::format("Depth/stencil image view {}", i));

            image_barriers_.registerImage(resource.image,
                                          depthStencilAspectMask(),
                                          VK_IMAGE_LAYOUT_UNDEFINED);
        }
        return true;
    }

    bool VulkanContext::createCommandPool() {
        for (std::size_t i = 0; i < kFramesInFlight; ++i) {
            VkCommandPoolCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            create_info.queueFamilyIndex = graphics_queue_family_;
            const VkResult result = vkCreateCommandPool(device_, &create_info, nullptr, &command_pools_[i]);
            if (result != VK_SUCCESS) {
                return fail(std::format("vkCreateCommandPool(frame {}) failed: {}", i, vkResultToString(result)));
            }
            setDebugObjectName(VK_OBJECT_TYPE_COMMAND_POOL,
                               command_pools_[i],
                               std::format("Frame {} graphics command pool", i));
        }

        VkCommandPoolCreateInfo immediate_info{};
        immediate_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        immediate_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        immediate_info.queueFamilyIndex = graphics_queue_family_;
        const VkResult result = vkCreateCommandPool(device_, &immediate_info, nullptr, &immediate_command_pool_);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkCreateCommandPool(immediate) failed: {}", vkResultToString(result)));
        }
        setDebugObjectName(VK_OBJECT_TYPE_COMMAND_POOL, immediate_command_pool_, "Immediate graphics command pool");
        return true;
    }

    bool VulkanContext::createCommandBuffers() {
        for (std::size_t i = 0; i < kFramesInFlight; ++i) {
            VkCommandBufferAllocateInfo allocate_info{};
            allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocate_info.commandPool = command_pools_[i];
            allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocate_info.commandBufferCount = 1;
            const VkResult result = vkAllocateCommandBuffers(device_, &allocate_info, &command_buffers_[i]);
            if (result != VK_SUCCESS) {
                return fail(std::format("vkAllocateCommandBuffers(frame {}) failed: {}", i, vkResultToString(result)));
            }
            setDebugObjectName(VK_OBJECT_TYPE_COMMAND_BUFFER,
                               command_buffers_[i],
                               std::format("Frame {} command buffer", i));
        }
        return true;
    }

    bool VulkanContext::createSyncObjects() {
        VkSemaphoreCreateInfo semaphore_info{};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (std::size_t i = 0; i < kFramesInFlight; ++i) {
            VkResult result = vkCreateSemaphore(device_, &semaphore_info, nullptr, &render_finished_[i]);
            if (result != VK_SUCCESS) {
                return fail(std::format("vkCreateSemaphore(render_finished {}) failed: {}", i, vkResultToString(result)));
            }
            setDebugObjectName(VK_OBJECT_TYPE_SEMAPHORE,
                               render_finished_[i],
                               std::format("Frame {} render finished semaphore", i));

            result = vkCreateFence(device_, &fence_info, nullptr, &in_flight_[i]);
            if (result != VK_SUCCESS) {
                return fail(std::format("vkCreateFence(frame {}) failed: {}", i, vkResultToString(result)));
            }
            setDebugObjectName(VK_OBJECT_TYPE_FENCE,
                               in_flight_[i],
                               std::format("Frame {} in-flight fence", i));
        }
        return true;
    }

    bool VulkanContext::createDebugMessenger() {
        if (!validation_enabled_) {
            return true;
        }

        auto* const create_debug_utils_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (create_debug_utils_messenger == nullptr) {
            return fail("VK_EXT_debug_utils is enabled, but vkCreateDebugUtilsMessengerEXT could not be loaded");
        }

        VkDebugUtilsMessengerCreateInfoEXT create_info{};
        populateDebugMessengerCreateInfo(create_info);
        const VkResult result = create_debug_utils_messenger(instance_, &create_info, nullptr, &debug_messenger_);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkCreateDebugUtilsMessengerEXT failed: {}", vkResultToString(result)));
        }
        return true;
    }

    void VulkanContext::destroyDebugMessenger() {
        if (debug_messenger_ == VK_NULL_HANDLE || instance_ == VK_NULL_HANDLE) {
            return;
        }

        auto* const destroy_debug_utils_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy_debug_utils_messenger != nullptr) {
            destroy_debug_utils_messenger(instance_, debug_messenger_, nullptr);
        }
        debug_messenger_ = VK_NULL_HANDLE;
    }

    void VulkanContext::setDebugObjectName(const VkObjectType object_type,
                                           const std::uint64_t object_handle,
                                           const std::string_view name) const {
        if (device_ == VK_NULL_HANDLE ||
            vk_set_debug_utils_object_name_ == nullptr ||
            object_handle == 0 ||
            name.empty()) {
            return;
        }

        const std::string owned_name{name};
        VkDebugUtilsObjectNameInfoEXT name_info{};
        name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        name_info.objectType = object_type;
        name_info.objectHandle = object_handle;
        name_info.pObjectName = owned_name.c_str();

        const VkResult result = vk_set_debug_utils_object_name_(device_, &name_info);
        if (result != VK_SUCCESS) {
            LOG_WARN("vkSetDebugUtilsObjectNameEXT failed for '{}': {}", owned_name, vkResultToString(result));
        }
    }

    bool VulkanContext::createPipelineCache() {
        if (device_ == VK_NULL_HANDLE || physical_device_ == VK_NULL_HANDLE) {
            return fail("Pipeline cache requires an initialized Vulkan device");
        }

        const std::filesystem::path path = defaultPipelineCachePath();
        std::vector<char> cache_data;
        if (readFile(path, cache_data)) {
            VkPhysicalDeviceProperties device_props{};
            vkGetPhysicalDeviceProperties(physical_device_, &device_props);
            if (!validPipelineCacheHeader(cache_data, device_props)) {
                cache_data.clear();
            }
        }

        VkPipelineCacheCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        create_info.initialDataSize = cache_data.size();
        create_info.pInitialData = cache_data.empty() ? nullptr : cache_data.data();

        VkResult result = vkCreatePipelineCache(device_, &create_info, nullptr, &pipeline_cache_);
        if (result != VK_SUCCESS && !cache_data.empty()) {
            cache_data.clear();
            create_info.initialDataSize = 0;
            create_info.pInitialData = nullptr;
            result = vkCreatePipelineCache(device_, &create_info, nullptr, &pipeline_cache_);
        }
        if (result != VK_SUCCESS) {
            return fail(std::format("vkCreatePipelineCache failed: {}", vkResultToString(result)));
        }

        setDebugObjectName(VK_OBJECT_TYPE_PIPELINE_CACHE, pipeline_cache_, "On-disk pipeline cache");
        if (!cache_data.empty()) {
            LOG_INFO("Loaded Vulkan pipeline cache: {} ({} bytes)",
                     lfs::core::path_to_utf8(path),
                     cache_data.size());
        }
        return true;
    }

    void VulkanContext::saveAndDestroyPipelineCache() {
        if (device_ == VK_NULL_HANDLE || pipeline_cache_ == VK_NULL_HANDLE) {
            return;
        }

        const std::filesystem::path path = defaultPipelineCachePath();
        std::size_t cache_size = 0;
        VkResult result = vkGetPipelineCacheData(device_, pipeline_cache_, &cache_size, nullptr);
        if (result == VK_SUCCESS && cache_size > 0) {
            std::vector<char> cache_data(cache_size);
            result = vkGetPipelineCacheData(device_, pipeline_cache_, &cache_size, cache_data.data());
            if (result == VK_SUCCESS && cache_size > 0) {
                cache_data.resize(cache_size);
                std::error_code ec;
                std::filesystem::create_directories(path.parent_path(), ec);
                if (!ec) {
                    std::ofstream file;
                    if (lfs::core::open_file_for_write(path,
                                                       std::ios::binary | std::ios::trunc,
                                                       file)) {
                        file.write(cache_data.data(), static_cast<std::streamsize>(cache_data.size()));
                        if (file) {
                            LOG_INFO("Saved Vulkan pipeline cache: {} ({} bytes)",
                                     lfs::core::path_to_utf8(path),
                                     cache_data.size());
                        }
                    }
                }
            }
        }

        vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
        pipeline_cache_ = VK_NULL_HANDLE;
    }

    void VulkanContext::destroyAllocator() {
        if (allocator_ != VK_NULL_HANDLE) {
            vmaDestroyAllocator(allocator_);
            allocator_ = VK_NULL_HANDLE;
        }
    }

    void VulkanContext::destroySwapchain() {
        if (device_ == VK_NULL_HANDLE) {
            return;
        }

        for (DepthStencilResource& resource : depth_stencil_resources_) {
            if (resource.view != VK_NULL_HANDLE) {
                vkDestroyImageView(device_, resource.view, nullptr);
                resource.view = VK_NULL_HANDLE;
            }
            if (resource.image != VK_NULL_HANDLE) {
                vmaDestroyImage(allocator_, resource.image, resource.allocation);
                resource.image = VK_NULL_HANDLE;
            }
            resource.allocation = VK_NULL_HANDLE;
        }
        depth_stencil_resources_.clear();
        depth_stencil_format_ = VK_FORMAT_UNDEFINED;
        for (const VkImageView view : swapchain_image_views_) {
            vkDestroyImageView(device_, view, nullptr);
        }
        swapchain_image_views_.clear();
        swapchain_images_.clear();
        swapchain_images_in_flight_.clear();
        for (VkSemaphore& semaphore : image_available_) {
            if (semaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(device_, semaphore, nullptr);
                semaphore = VK_NULL_HANDLE;
            }
        }
        image_available_.clear();
        next_acquire_index_ = 0;
        active_acquire_index_ = 0;
        image_barriers_.clearSwapchainOnly();

        if (swapchain_ != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device_, swapchain_, nullptr);
            swapchain_ = VK_NULL_HANDLE;
        }
        swapchain_image_usage_ = 0;
    }

    bool VulkanContext::waitForFrameFences() {
        // Bound the wait so a wedged GPU surfaces as a swapchain-recreate failure rather
        // than a hang. 2 s is generous; healthy frames complete in <16 ms.
        constexpr std::uint64_t kSwapchainWaitTimeoutNs = 2'000'000'000ull;
        std::vector<VkFence> fences;
        fences.reserve(kFramesInFlight + swapchain_images_in_flight_.size());
        for (const VkFence fence : in_flight_) {
            if (fence != VK_NULL_HANDLE) {
                fences.push_back(fence);
            }
        }
        // The per-swapchain-image fences are aliases of in_flight_ entries set in endFrame;
        // include any not already covered (dedup keeps the wait cheap).
        for (const VkFence fence : swapchain_images_in_flight_) {
            if (fence == VK_NULL_HANDLE) {
                continue;
            }
            if (std::find(fences.begin(), fences.end(), fence) == fences.end()) {
                fences.push_back(fence);
            }
        }
        if (fences.empty()) {
            return true;
        }
        const VkResult result = vkWaitForFences(device_,
                                                static_cast<std::uint32_t>(fences.size()),
                                                fences.data(),
                                                VK_TRUE,
                                                kSwapchainWaitTimeoutNs);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkWaitForFences(swapchain recreate) failed: {}", vkResultToString(result)));
        }
        return true;
    }

    bool VulkanContext::recreateSwapchain() {
        if (framebuffer_width_ <= 0 || framebuffer_height_ <= 0) {
            return true;
        }

        // Wait on all in-flight render fences (per-frame + per-image aliases). Once those
        // are signaled, prior vkQueueSubmit work is complete and the presentation engine
        // can no longer be reading the swapchain images we are about to destroy. We avoid
        // vkQueueWaitIdle(present_queue_) because it stalls the entire CPU thread on the
        // compositor, which can deadlock if the display server is itself blocked.
        if (!waitForFrameFences()) {
            framebuffer_resized_ = true;
            return false;
        }
        destroySwapchain();
        const bool created = createSwapchain(framebuffer_width_, framebuffer_height_) &&
                             createImageViews() &&
                             createDepthStencilResources();
        if (!created) {
            const std::string error = last_error_;
            destroySwapchain();
            framebuffer_resized_ = true;
            last_error_ = error;
            return false;
        }
        framebuffer_resized_ = false;
        return true;
    }
#endif

} // namespace lfs::vis
