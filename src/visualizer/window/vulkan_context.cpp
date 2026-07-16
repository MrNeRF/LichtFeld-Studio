/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_context.hpp"

#include "core/environment.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "vulkan_result.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <set>
#include <utility>

#include <SDL3/SDL_vulkan.h>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifndef LFS_VULKAN_VALIDATION_DEFAULT
#define LFS_VULKAN_VALIDATION_DEFAULT 0
#endif

namespace lfs::vis {
    namespace {
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
#if defined(__linux__)
        constexpr bool kPromoteExposedResizeGrowthImmediately = true;
        constexpr bool kUseResizeHeadroom = false;
        constexpr bool kRecreateExactAfterInteractiveResize = true;
        constexpr bool kRecreateExactAfterResizeHeadroom = false;
#else
        constexpr bool kPromoteExposedResizeGrowthImmediately = false;
        constexpr bool kUseResizeHeadroom = true;
        constexpr bool kRecreateExactAfterInteractiveResize = false;
        constexpr bool kRecreateExactAfterResizeHeadroom = true;
#endif
        // Shrinks can settle because no new pixels are exposed. Windows manual
        // borderless resize tolerates padded, throttled grow recreates. X11
        // exposes grown client pixels immediately and must return to exact
        // swapchain extents after shrink/restore to avoid compositor scaling blur.
        constexpr auto kSwapchainResizeQuietDelay = std::chrono::milliseconds(33);
        constexpr auto kSwapchainResizeGrowInterval = std::chrono::milliseconds(33);
        constexpr auto kSwapchainResizeGrowQuietDelay = std::chrono::milliseconds(48);
        constexpr auto kSwapchainResizeGrowMaxLatency = std::chrono::milliseconds(96);
        constexpr int kSwapchainResizeGrowStepPx = 16;
        constexpr std::uint32_t kSwapchainResizeHeadroomPx = 1024;

        [[nodiscard]] int positiveResizeDelta(const int requested_width,
                                              const int requested_height,
                                              const VkExtent2D current_extent) {
            const int grow_x = requested_width - static_cast<int>(current_extent.width);
            const int grow_y = requested_height - static_cast<int>(current_extent.height);
            return std::max({0, grow_x, grow_y});
        }

        [[nodiscard]] std::uint32_t addResizeHeadroom(const std::uint32_t requested,
                                                      const std::uint32_t max_extent) {
            // Pad both axes during interactive resize so direction changes do not
            // force another WSI rebuild a few mouse deltas later. Do not carry
            // forward the previous swapchain extent: during shrink/restore that
            // can repeatedly compound the headroom into a huge offscreen surface.
            if (requested >= max_extent) {
                return max_extent;
            }
            return requested + std::min(kSwapchainResizeHeadroomPx, max_extent - requested);
        }

        [[nodiscard]] double elapsedMs(const std::chrono::steady_clock::time_point start) {
            return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - start)
                .count();
        }

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

        [[nodiscard]] bool isPreVoltaCudaDevice(const std::array<std::uint8_t, VK_UUID_SIZE>& vk_device_uuid) {
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
                return false;
            }
            for (int device = 0; device < device_count; ++device) {
                cudaDeviceProp props{};
                if (cudaGetDeviceProperties(&props, device) != cudaSuccess) {
                    continue;
                }
                static_assert(sizeof(props.uuid.bytes) == VK_UUID_SIZE);
                if (std::memcmp(props.uuid.bytes, vk_device_uuid.data(), VK_UUID_SIZE) == 0) {
                    return props.major < 7;
                }
            }
            return false;
        }

        // True when the given Vulkan physical device is the same physical GPU as
        // the CUDA device the trainer will use (cuda_device, the current CUDA
        // ordering's device 0 by default). On multi-GPU machines with identical
        // cards, Vulkan's enumeration order can differ from CUDA's; matching by
        // UUID lets pickPhysicalDevice keep the viewer on the same card as the
        // trainer so CUDA<->Vulkan external-memory interop can import the block.
        [[nodiscard]] bool vulkanDeviceMatchesCudaDevice(const VkPhysicalDevice device, const int cuda_device) {
            cudaDeviceProp cuda_props{};
            if (cudaGetDeviceProperties(&cuda_props, cuda_device) != cudaSuccess) {
                return false;
            }
            VkPhysicalDeviceIDProperties vk_id{};
            vk_id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
            VkPhysicalDeviceProperties2 vk_props2{};
            vk_props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            vk_props2.pNext = &vk_id;
            vkGetPhysicalDeviceProperties2(device, &vk_props2);
            static_assert(sizeof(cuda_props.uuid.bytes) == VK_UUID_SIZE);
            return std::memcmp(cuda_props.uuid.bytes, vk_id.deviceUUID, VK_UUID_SIZE) == 0;
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
            return support;
        }

        [[nodiscard]] bool hasRequiredFeatures(const RequiredFeatureSupport& support) {
            return support.synchronization2 &&
                   support.dynamic_rendering &&
                   support.timeline_semaphore;
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
            return missing;
        }

        [[nodiscard]] bool validationRequestedByBuild() {
            return lfs::core::environment::flag(
                "LFS_VK_VALIDATION", LFS_VULKAN_VALIDATION_DEFAULT != 0);
        }

        [[nodiscard]] bool isExpectedPinnedValidationLayerDedup(
            [[maybe_unused]] const VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
            [[maybe_unused]] const VkDebugUtilsMessageTypeFlagsEXT message_type,
            [[maybe_unused]] const VkDebugUtilsMessengerCallbackDataEXT* const callback_data) {
#ifdef LFS_VULKAN_VALIDATION_LAYER_DIR
            constexpr std::string_view loader_message_id = "Loader Message";
            constexpr std::string_view duplicate_prefix =
                "Removing layer VK_LAYER_KHRONOS_validation (";
            constexpr std::string_view duplicate_separator =
                ") because it is a duplicate of VK_LAYER_KHRONOS_validation (";

            if (message_severity != VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ||
                message_type != VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT ||
                callback_data == nullptr || callback_data->pMessageIdName == nullptr ||
                callback_data->pMessage == nullptr || callback_data->messageIdNumber != 0 ||
                std::string_view(callback_data->pMessageIdName) != loader_message_id) {
                return false;
            }

            const std::string_view message = callback_data->pMessage;
            if (!message.starts_with(duplicate_prefix) || !message.ends_with(')')) {
                return false;
            }

            const auto separator = message.find(duplicate_separator, duplicate_prefix.size());
            if (separator == std::string_view::npos || separator == duplicate_prefix.size()) {
                return false;
            }

            const auto retained_manifest_begin = separator + duplicate_separator.size();
            if (retained_manifest_begin >= message.size() - 1) {
                return false;
            }

            const std::filesystem::path retained_manifest(
                std::string(message.substr(retained_manifest_begin, message.size() - retained_manifest_begin - 1)));
            const std::filesystem::path pinned_manifest =
                std::filesystem::path(LFS_VULKAN_VALIDATION_LAYER_DIR) / "VkLayer_khronos_validation.json";
            std::error_code error;
            const auto retained_absolute = std::filesystem::absolute(retained_manifest, error).lexically_normal();
            if (error) {
                return false;
            }
            const auto pinned_absolute = std::filesystem::absolute(pinned_manifest, error).lexically_normal();
            return !error && retained_absolute == pinned_absolute;
#else
            return false;
#endif
        }

        VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
            VkDebugUtilsMessageTypeFlagsEXT message_type,
            const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
            void* user_data) {
            const char* const message = callback_data != nullptr && callback_data->pMessage != nullptr
                                            ? callback_data->pMessage
                                            : "<missing validation message>";

            if ((message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
                LOG_ERROR("Vulkan validation: {}", message);
                const bool fatal = user_data != nullptr && *static_cast<const bool*>(user_data);
                if (fatal) {
                    LOG_CRITICAL("Vulkan validation error is fatal because LFS_VK_VALIDATION_FATAL=1: {}",
                                 message);
                    std::abort();
                }
            } else if (isExpectedPinnedValidationLayerDedup(message_severity, message_type, callback_data)) {
                LOG_INFO("Vulkan validation: {}", message);
            } else if ((message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
                LOG_WARN("Vulkan validation: {}", message);
            }
            return VK_FALSE;
        }

        void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& create_info,
                                              const bool* const validation_errors_fatal) {
            create_info = {};
            create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            create_info.pfnUserCallback = vulkanDebugCallback;
            create_info.pUserData = const_cast<bool*>(validation_errors_fatal);
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

        [[nodiscard]] const char* pipelineCacheRejectReason(const std::vector<char>& data,
                                                            const VkPhysicalDeviceProperties& device_props) {
            if (data.size() < sizeof(VkPipelineCacheHeaderVersionOne)) {
                return "file smaller than cache header";
            }

            VkPipelineCacheHeaderVersionOne header{};
            std::memcpy(&header, data.data(), sizeof(header));
            if (header.headerSize < sizeof(VkPipelineCacheHeaderVersionOne) || header.headerSize > data.size()) {
                return "header size out of bounds";
            }
            if (header.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE) {
                return "unsupported header version";
            }
            if (header.vendorID != device_props.vendorID) {
                return "vendorID mismatch";
            }
            if (header.deviceID != device_props.deviceID) {
                return "deviceID mismatch";
            }
            if (std::memcmp(header.pipelineCacheUUID, device_props.pipelineCacheUUID, VK_UUID_SIZE) != 0) {
                return "pipelineCacheUUID mismatch (driver update?)";
            }
            return nullptr;
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

        [[nodiscard]] std::size_t estimateFormatBytesPerPixel(const VkFormat format) {
            switch (format) {
            case VK_FORMAT_R8_UNORM:
            case VK_FORMAT_R8_SNORM:
            case VK_FORMAT_R8_UINT:
            case VK_FORMAT_R8_SINT:
                return 1;
            case VK_FORMAT_R8G8_UNORM:
            case VK_FORMAT_R8G8_SNORM:
            case VK_FORMAT_R8G8_UINT:
            case VK_FORMAT_R8G8_SINT:
            case VK_FORMAT_R16_UNORM:
            case VK_FORMAT_R16_SNORM:
            case VK_FORMAT_R16_UINT:
            case VK_FORMAT_R16_SINT:
            case VK_FORMAT_R16_SFLOAT:
                return 2;
            case VK_FORMAT_B8G8R8A8_UNORM:
            case VK_FORMAT_B8G8R8A8_SRGB:
            case VK_FORMAT_R8G8B8A8_UNORM:
            case VK_FORMAT_R8G8B8A8_SRGB:
            case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
            case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
            case VK_FORMAT_R16G16_UNORM:
            case VK_FORMAT_R16G16_SNORM:
            case VK_FORMAT_R16G16_UINT:
            case VK_FORMAT_R16G16_SINT:
            case VK_FORMAT_R16G16_SFLOAT:
            case VK_FORMAT_R32_UINT:
            case VK_FORMAT_R32_SINT:
            case VK_FORMAT_R32_SFLOAT:
            case VK_FORMAT_D32_SFLOAT:
            case VK_FORMAT_D24_UNORM_S8_UINT:
                return 4;
            case VK_FORMAT_R16G16B16A16_UNORM:
            case VK_FORMAT_R16G16B16A16_SNORM:
            case VK_FORMAT_R16G16B16A16_UINT:
            case VK_FORMAT_R16G16B16A16_SINT:
            case VK_FORMAT_R16G16B16A16_SFLOAT:
            case VK_FORMAT_R32G32_UINT:
            case VK_FORMAT_R32G32_SINT:
            case VK_FORMAT_R32G32_SFLOAT:
            case VK_FORMAT_D32_SFLOAT_S8_UINT:
                return 8;
            case VK_FORMAT_R32G32B32A32_UINT:
            case VK_FORMAT_R32G32B32A32_SINT:
            case VK_FORMAT_R32G32B32A32_SFLOAT:
                return 16;
            default:
                return 0;
            }
        }

        void recordCurrentVulkanBytes(std::string_view scope,
                                      std::string_view label,
                                      const std::size_t bytes) {
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(scope, label, bytes);
        }
    } // namespace

    VulkanContext::~VulkanContext() {
        shutdown();
    }

    bool VulkanContext::fail(std::string message, const std::source_location location) {
        last_error_ = std::format("{} ({}:{})",
                                  std::move(message),
                                  location.file_name(),
                                  location.line());
        LOG_ERROR("Vulkan: {}", last_error_);
        return false;
    }

    bool VulkanContext::setVkFailure(std::string message) {
        last_error_ = std::move(message);
        LOG_ERROR("Vulkan: {}", last_error_);
        return false;
    }

    bool VulkanContext::init(SDL_Window* window, const int framebuffer_width, const int framebuffer_height) {
        framebuffer_width_ = framebuffer_width;
        framebuffer_height_ = framebuffer_height;

        // Per-step timing so the one-time Vulkan bring-up cost is attributable in the perf log.
        const auto timed = [](const char* name, auto&& fn) {
            LOG_TIMER(name);
            return fn();
        };
        const bool initialized =
            timed("vulkan_init.createInstance", [&] { return createInstance(); }) &&
            timed("vulkan_init.createSurface", [&] { return createSurface(window); }) &&
            timed("vulkan_init.pickPhysicalDevice", [&] { return pickPhysicalDevice(); }) &&
            timed("vulkan_init.createDevice", [&] { return createDevice(); }) &&
            timed("vulkan_init.createAllocator", [&] { return createAllocator(); }) &&
            timed("vulkan_init.createPipelineCache", [&] { return createPipelineCache(); }) &&
            timed("vulkan_init.createSwapchain", [&] { return createSwapchain(framebuffer_width, framebuffer_height); }) &&
            timed("vulkan_init.createImageViews", [&] { return createImageViews(); }) &&
            timed("vulkan_init.createDepthStencilResources", [&] { return createDepthStencilResources(); }) &&
            timed("vulkan_init.createCommandPool", [&] { return createCommandPool(); }) &&
            timed("vulkan_init.createCommandBuffers", [&] { return createCommandBuffers(); }) &&
            timed("vulkan_init.createSyncObjects", [&] { return createSyncObjects(); });
        LOG_INFO("Vulkan diagnostics: validation_layers={}, debug_utils={}, validation_errors_fatal={}",
                 validation_enabled_ ? "active" : "inactive",
                 debugObjectNamingEnabled() ? "active" : "inactive",
                 validation_errors_fatal_ ? "active" : "inactive");
        return initialized;
    }

    void VulkanContext::shutdown() {
        if (device_ != VK_NULL_HANDLE) {
            // Shutdown is the one place where a whole-device wait is intentional:
            // all swapchain, UI, and external interop resources are about to be destroyed.
            const VkResult idle_result = vkDeviceWaitIdle(device_);
            if (idle_result != VK_SUCCESS) {
                LOG_ERROR("Vulkan shutdown could not retire device work before destruction (device={:#x}, pending_immediate_submits={}, frame_active={}, frame_slot={}, result={}({}))",
                          vkHandleValue(device_),
                          pending_immediate_submits_.size(),
                          frame_active_,
                          frame_index_,
                          vkResultToString(idle_result),
                          static_cast<int>(idle_result));
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
            debug_name_writer_.reset();
            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
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
            validation_errors_fatal_ = false;
        }
        frame_active_ = false;
        frame_rendering_active_ = false;
        frame_timeline_waits_.clear();
        frame_timeline_waits_valid_ = true;
        last_frame_timeline_wait_values_.clear();
        last_immediate_timeline_wait_values_.clear();
        last_immediate_timeline_signal_values_.clear();
    }

    VkExtent2D VulkanContext::framebufferExtent() const {
        VkExtent2D extent{};
        extent.width = static_cast<std::uint32_t>(std::max(0, framebuffer_width_));
        extent.height = static_cast<std::uint32_t>(std::max(0, framebuffer_height_));
        if (swapchain_extent_.width > 0) {
            extent.width = std::min(extent.width, swapchain_extent_.width);
        }
        if (swapchain_extent_.height > 0) {
            extent.height = std::min(extent.height, swapchain_extent_.height);
        }
        return extent;
    }

    bool VulkanContext::framebufferFitsSwapchainExtent() const {
        return swapchain_ != VK_NULL_HANDLE &&
               framebuffer_width_ > 0 &&
               framebuffer_height_ > 0 &&
               static_cast<std::uint32_t>(framebuffer_width_) <= swapchain_extent_.width &&
               static_cast<std::uint32_t>(framebuffer_height_) <= swapchain_extent_.height;
    }

    bool VulkanContext::framebufferResizeRequiresSwapchainRecreate() const {
        if (!framebufferFitsSwapchainExtent()) {
            return true;
        }
        return swapchain_extent_fixed_to_surface_ && !swapchain_present_scaling_enabled_;
    }

    void VulkanContext::deferSwapchainResizeRecreate(
        const bool requires_recreate,
        const std::optional<bool> allow_headroom) {
        const auto now = std::chrono::steady_clock::now();
        framebuffer_resize_deferred_ = true;
        framebuffer_resize_requires_recreate_ = framebuffer_resize_requires_recreate_ || requires_recreate;
        if (allow_headroom.has_value()) {
            framebuffer_resize_allow_headroom_ = *allow_headroom;
        }
        framebuffer_resize_last_change_ = now;
        framebuffer_resized_ = false;
        last_error_.clear();
    }

    bool VulkanContext::pendingSwapchainResizeReady() const {
        const auto now = std::chrono::steady_clock::now();
        if (!framebuffer_resize_deferred_) {
            if (!framebuffer_resize_exact_after_interactive_) {
                return true;
            }
            return now - framebuffer_resize_last_change_ >= kSwapchainResizeQuietDelay;
        }

        if (!framebuffer_resize_requires_recreate_) {
            return true;
        }

        const int grow_delta = positiveResizeDelta(framebuffer_width_,
                                                   framebuffer_height_,
                                                   swapchain_extent_);
        if (grow_delta <= 0) {
            return now - framebuffer_resize_last_change_ >= kSwapchainResizeQuietDelay;
        }
        if (kPromoteExposedResizeGrowthImmediately) {
            return true;
        }

        if (framebuffer_resize_last_recreate_ == std::chrono::steady_clock::time_point{}) {
            return true;
        }

        const auto since_last_recreate = now - framebuffer_resize_last_recreate_;
        return (grow_delta >= kSwapchainResizeGrowStepPx &&
                since_last_recreate >= kSwapchainResizeGrowInterval) ||
               (now - framebuffer_resize_last_change_ >= kSwapchainResizeGrowQuietDelay &&
                since_last_recreate >= kSwapchainResizeGrowInterval) ||
               since_last_recreate >= kSwapchainResizeGrowMaxLatency;
    }

    double VulkanContext::secondsUntilPendingSwapchainResizeReady() const {
        const auto now = std::chrono::steady_clock::now();
        if (!framebuffer_resize_deferred_) {
            if (!framebuffer_resize_exact_after_interactive_) {
                return 0.0;
            }
            const auto since_last_change = now - framebuffer_resize_last_change_;
            if (since_last_change >= kSwapchainResizeQuietDelay) {
                return 0.0;
            }
            return std::chrono::duration<double>(
                       kSwapchainResizeQuietDelay - since_last_change)
                .count();
        }

        if (!framebuffer_resize_requires_recreate_) {
            return 0.0;
        }

        const auto since_last_change = now - framebuffer_resize_last_change_;
        const int grow_delta = positiveResizeDelta(framebuffer_width_,
                                                   framebuffer_height_,
                                                   swapchain_extent_);
        if (grow_delta <= 0) {
            if (since_last_change >= kSwapchainResizeQuietDelay) {
                return 0.0;
            }
            const auto until_quiet = kSwapchainResizeQuietDelay - since_last_change;
            return std::chrono::duration<double>(until_quiet).count();
        }
        if (kPromoteExposedResizeGrowthImmediately) {
            return 0.0;
        }

        if (framebuffer_resize_last_recreate_ == std::chrono::steady_clock::time_point{}) {
            return 0.0;
        }

        const auto since_last_recreate = now - framebuffer_resize_last_recreate_;
        if (since_last_recreate >= kSwapchainResizeGrowMaxLatency ||
            (grow_delta >= kSwapchainResizeGrowStepPx &&
             since_last_recreate >= kSwapchainResizeGrowInterval) ||
            (since_last_change >= kSwapchainResizeGrowQuietDelay &&
             since_last_recreate >= kSwapchainResizeGrowInterval)) {
            return 0.0;
        }

        auto until_grow = kSwapchainResizeGrowMaxLatency - since_last_recreate;
        if (grow_delta >= kSwapchainResizeGrowStepPx) {
            until_grow = std::min(until_grow,
                                  kSwapchainResizeGrowInterval - since_last_recreate);
        }
        const auto until_quiet = since_last_change >= kSwapchainResizeGrowQuietDelay
                                     ? std::chrono::steady_clock::duration::zero()
                                     : kSwapchainResizeGrowQuietDelay - since_last_change;
        const auto until_interval = since_last_recreate >= kSwapchainResizeGrowInterval
                                        ? std::chrono::steady_clock::duration::zero()
                                        : kSwapchainResizeGrowInterval - since_last_recreate;
        until_grow = std::min(until_grow, std::max(until_quiet, until_interval));
        return std::chrono::duration<double>(until_grow).count();
    }

    bool VulkanContext::promoteDeferredSwapchainResizeIfSettled() {
        if (!framebuffer_resize_deferred_) {
            return true;
        }

        if (!pendingSwapchainResizeReady()) {
            last_error_.clear();
#if defined(__linux__)
            return framebufferFitsSwapchainExtent();
#else
            return false;
#endif
        }

        const bool requires_recreate = framebuffer_resize_requires_recreate_;
        framebuffer_resize_deferred_ = false;
        framebuffer_resize_requires_recreate_ = false;
        framebuffer_resized_ = requires_recreate;
        return true;
    }

    void VulkanContext::notifyFramebufferResized(const int width,
                                                 const int height,
                                                 const ResizeIntent intent) {
        const bool exact_resize = intent == ResizeIntent::Exact;
        const bool exact_extent_mismatch =
            exact_resize &&
            swapchain_ != VK_NULL_HANDLE &&
            width > 0 &&
            height > 0 &&
            (static_cast<std::uint32_t>(width) != swapchain_extent_.width ||
             static_cast<std::uint32_t>(height) != swapchain_extent_.height);
        if (width == framebuffer_width_ && height == framebuffer_height_ &&
            !exact_extent_mismatch) {
            return;
        }
        framebuffer_width_ = width;
        framebuffer_height_ = height;
        framebuffer_resize_allow_headroom_ = !exact_resize && kUseResizeHeadroom;
        if (width <= 0 || height <= 0) {
            framebuffer_resize_deferred_ = false;
            framebuffer_resize_requires_recreate_ = false;
            framebuffer_resize_allow_headroom_ = false;
            framebuffer_resize_exact_after_interactive_ = false;
            framebuffer_resized_ = true;
            return;
        }

        framebuffer_resize_exact_after_interactive_ =
            !exact_resize &&
            (kRecreateExactAfterInteractiveResize ||
             (framebuffer_resize_allow_headroom_ && kRecreateExactAfterResizeHeadroom));
        const bool requires_recreate =
            framebufferResizeRequiresSwapchainRecreate() || exact_extent_mismatch;
        if (requires_recreate) {
            deferSwapchainResizeRecreate(requires_recreate, framebuffer_resize_allow_headroom_);
        } else {
            framebuffer_resize_deferred_ = false;
            framebuffer_resize_requires_recreate_ = false;
            framebuffer_resize_last_change_ = std::chrono::steady_clock::now();
            last_error_.clear();
        }
    }

    bool VulkanContext::presentBootstrapFrame(const float r, const float g, const float b, const float a) {
        VkClearValue clear_value{};
        clear_value.color = VkClearColorValue{{r, g, b, a}};

        Frame frame{};
        if (!beginFrame(clear_value, frame)) {
            return false;
        }
        return endFrame();
    }

    bool VulkanContext::beginFrame(const VkClearValue& clear_value, Frame& frame) {
        if (frame_active_) {
            return fail(std::format(
                "beginFrame called while another frame is active (frame_active={}, rendering_active={}, frame_index={}, active_frame_index={}, active_image_index={}, active_acquire_index={})",
                frame_active_,
                frame_rendering_active_,
                frame_index_,
                active_frame_index_,
                active_image_index_,
                active_acquire_index_));
        }
        frame_timeline_waits_.clear();
        frame_timeline_waits_valid_ = true;
        frame = {};
        if (device_ == VK_NULL_HANDLE) {
            return fail(std::format(
                "beginFrame requires an initialized Vulkan device (device={:#x}, framebuffer={}x{}, swapchain={:#x}, frame_index={})",
                vkHandleValue(device_),
                framebuffer_width_,
                framebuffer_height_,
                vkHandleValue(swapchain_),
                frame_index_));
        }
        if (framebuffer_width_ <= 0 || framebuffer_height_ <= 0) {
            last_error_.clear();
            return false;
        }

        if (framebuffer_resize_exact_after_interactive_ && !framebuffer_resize_deferred_) {
            const bool extent_matches_framebuffer =
                swapchain_ != VK_NULL_HANDLE &&
                static_cast<std::uint32_t>(framebuffer_width_) == swapchain_extent_.width &&
                static_cast<std::uint32_t>(framebuffer_height_) == swapchain_extent_.height;
            if (extent_matches_framebuffer) {
                framebuffer_resize_exact_after_interactive_ = false;
            } else if (pendingSwapchainResizeReady()) {
                framebuffer_resize_exact_after_interactive_ = false;
                framebuffer_resize_allow_headroom_ = false;
                framebuffer_resized_ = true;
            }
        }

        if (!promoteDeferredSwapchainResizeIfSettled()) {
            return false;
        }

        if (swapchain_ == VK_NULL_HANDLE || framebuffer_resized_) {
            LOG_DEBUG("Vulkan beginFrame requesting swapchain recreate: swapchain_present={}, framebuffer_resized={}, framebuffer={}x{}, old_extent={}x{}, images={}, frame_index={}",
                      swapchain_ != VK_NULL_HANDLE,
                      framebuffer_resized_,
                      framebuffer_width_,
                      framebuffer_height_,
                      swapchain_extent_.width,
                      swapchain_extent_.height,
                      swapchain_images_.size(),
                      frame_index_);
            if (!recreateSwapchain()) {
                return false;
            }
        }

        const std::size_t current_frame = frame_index_;
        if (current_frame >= kFramesInFlight || current_frame >= command_pools_.size() ||
            current_frame >= command_buffers_.size() || current_frame >= in_flight_.size() ||
            current_frame >= frame_submit_serials_.size()) {
            return fail(std::format(
                "beginFrame frame-slot index is outside a per-frame array (frame_index={}, frames_in_flight={}, command_pools={}, command_buffers={}, fences={}, submit_serials={})",
                current_frame,
                kFramesInFlight,
                command_pools_.size(),
                command_buffers_.size(),
                in_flight_.size(),
                frame_submit_serials_.size()));
        }
        if (swapchain_images_.empty() ||
            swapchain_image_views_.size() != swapchain_images_.size() ||
            swapchain_images_in_flight_.size() != swapchain_images_.size() ||
            render_finished_.size() != swapchain_images_.size()) {
            return fail(std::format(
                "beginFrame swapchain arrays must have one entry per image (images={}, image_views={}, image_fences={}, render_finished={}, swapchain={:#x})",
                swapchain_images_.size(),
                swapchain_image_views_.size(),
                swapchain_images_in_flight_.size(),
                render_finished_.size(),
                vkHandleValue(swapchain_)));
        }
        const bool depth_stencil_ready =
            depth_stencil_resources_.size() == kFramesInFlight &&
            std::all_of(depth_stencil_resources_.begin(),
                        depth_stencil_resources_.end(),
                        [](const DepthStencilResource& resource) {
                            return resource.image != VK_NULL_HANDLE &&
                                   resource.view != VK_NULL_HANDLE;
                        });
        if (!depth_stencil_ready) {
            return fail(std::format(
                "beginFrame requires one valid depth/stencil resource per frame slot (resource_count={}, frames_in_flight={}, frame_index={}, format={})",
                depth_stencil_resources_.size(),
                kFramesInFlight,
                current_frame,
                vkFormatToString(depth_stencil_format_)));
        }

        VkFence frame_fence = in_flight_[current_frame];
        if (frame_fence == VK_NULL_HANDLE) {
            return fail(std::format(
                "beginFrame cannot wait a null frame fence (frame_index={}, fence={:#x}, last_submit_id={})",
                current_frame,
                vkHandleValue(frame_fence),
                frame_submit_serials_[current_frame]));
        }
        VkResult result = VK_SUCCESS;
        {
            LOG_TIMER_THRESHOLD("frame_pacing.vulkan_beginFrame.wait_frame_fence", 0.25);
            const auto wait_start = std::chrono::steady_clock::now();
            result = vkWaitForFences(device_, 1, &frame_fence, VK_TRUE, kWaitForeverNs);
            if (result != VK_SUCCESS) {
                return fail(std::format("vkWaitForFences(frame slot {}) failed after {:.1f} ms: {} (frame_index={}, last_submit_id={}, framebuffer={}x{}, swapchain_extent={}x{}, active_image={}, active_frame={}, framebuffer_resized={})",
                                        current_frame,
                                        elapsedMs(wait_start),
                                        vkResultToString(result),
                                        frame_index_,
                                        frame_submit_serials_[current_frame],
                                        framebuffer_width_,
                                        framebuffer_height_,
                                        swapchain_extent_.width,
                                        swapchain_extent_.height,
                                        active_image_index_,
                                        active_frame_index_,
                                        framebuffer_resized_));
            }
        }

        uint32_t image_index = 0;
        if (image_available_.empty()) {
            return fail(std::format(
                "beginFrame requires at least one acquire semaphore (acquire_semaphore_count={}, swapchain_image_count={}, next_acquire_index={})",
                image_available_.size(),
                swapchain_images_.size(),
                next_acquire_index_));
        }
        const std::size_t acquire_index = next_acquire_index_;
        if (acquire_index >= image_available_.size() ||
            image_available_[acquire_index] == VK_NULL_HANDLE) {
            return fail(std::format(
                "beginFrame acquire index must reference a valid semaphore (acquire_index={}, acquire_semaphore_count={}, semaphore={:#x}, frame_index={})",
                acquire_index,
                image_available_.size(),
                acquire_index < image_available_.size()
                    ? vkHandleValue(image_available_[acquire_index])
                    : 0,
                current_frame));
        }
        {
            LOG_TIMER_THRESHOLD("frame_pacing.vulkan_beginFrame.acquire_next_image", 0.25);
            result = vkAcquireNextImageKHR(device_, swapchain_, kWaitForeverNs,
                                           image_available_[acquire_index],
                                           VK_NULL_HANDLE, &image_index);
        }
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            LOG_DEBUG("vkAcquireNextImageKHR returned OUT_OF_DATE: acquire_index={}, framebuffer={}x{}, swapchain_extent={}x{}",
                      acquire_index,
                      framebuffer_width_,
                      framebuffer_height_,
                      swapchain_extent_.width,
                      swapchain_extent_.height);
            framebuffer_resize_exact_after_interactive_ = false;
            deferSwapchainResizeRecreate(true, false);
            return false;
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            return fail(std::format("vkAcquireNextImageKHR failed: {}", vkResultToString(result)));
        }
        if (image_index >= swapchain_images_.size() ||
            image_index >= swapchain_image_views_.size() ||
            image_index >= swapchain_images_in_flight_.size() ||
            image_index >= render_finished_.size()) {
            framebuffer_resized_ = true;
            return fail(std::format(
                "vkAcquireNextImageKHR returned an index outside a swapchain array (image_index={}, images={}, image_views={}, image_fences={}, render_finished={}, acquire_index={})",
                image_index,
                swapchain_images_.size(),
                swapchain_image_views_.size(),
                swapchain_images_in_flight_.size(),
                render_finished_.size(),
                acquire_index));
        }
        if (swapchain_images_in_flight_[image_index] != VK_NULL_HANDLE) {
            VkFence image_fence = swapchain_images_in_flight_[image_index];
            {
                LOG_TIMER_THRESHOLD("frame_pacing.vulkan_beginFrame.wait_image_fence", 0.25);
                const auto wait_start = std::chrono::steady_clock::now();
                result = vkWaitForFences(device_, 1, &image_fence, VK_TRUE, kWaitForeverNs);
                if (result != VK_SUCCESS) {
                    framebuffer_resized_ = true;
                    return fail(std::format("vkWaitForFences(swapchain image {}) failed after {:.1f} ms: {} (frame_slot={}, acquire_index={}, framebuffer={}x{}, swapchain_extent={}x{})",
                                            image_index,
                                            elapsedMs(wait_start),
                                            vkResultToString(result),
                                            current_frame,
                                            acquire_index,
                                            framebuffer_width_,
                                            framebuffer_height_,
                                            swapchain_extent_.width,
                                            swapchain_extent_.height));
                }
            }
        }

        frame_suboptimal_ = (result == VK_SUBOPTIMAL_KHR);
        active_image_index_ = image_index;
        active_frame_index_ = current_frame;
        active_acquire_index_ = acquire_index;
        next_acquire_index_ = (acquire_index + 1) % image_available_.size();

        if (command_pools_[current_frame] == VK_NULL_HANDLE ||
            command_buffers_[current_frame] == VK_NULL_HANDLE) {
            framebuffer_resized_ = true;
            return fail(std::format(
                "beginFrame requires a valid command pool and buffer for the frame slot (frame_index={}, command_pool={:#x}, command_buffer={:#x})",
                current_frame,
                vkHandleValue(command_pools_[current_frame]),
                vkHandleValue(command_buffers_[current_frame])));
        }
        result = vkResetCommandPool(device_, command_pools_[current_frame], 0);
        if (result != VK_SUCCESS) {
            framebuffer_resized_ = true;
            return fail(std::format(
                "vkResetCommandPool failed before frame recording (frame_index={}, command_pool={:#x}, result={}({}))",
                current_frame,
                vkHandleValue(command_pools_[current_frame]),
                vkResultToString(result),
                static_cast<int>(result)));
        }

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = vkBeginCommandBuffer(command_buffers_[current_frame], &begin_info);
        if (result != VK_SUCCESS) {
            framebuffer_resized_ = true;
            return fail(std::format(
                "vkBeginCommandBuffer failed before frame recording (frame_index={}, command_buffer={:#x}, flags={:#x}, result={}({}))",
                current_frame,
                vkHandleValue(command_buffers_[current_frame]),
                static_cast<std::uint32_t>(begin_info.flags),
                vkResultToString(result),
                static_cast<int>(result)));
        }

        const VkExtent2D render_extent = framebufferExtent();
        VkCommandBuffer command_buffer = command_buffers_[current_frame];
        // The acquire wait below is scoped to COLOR_ATTACHMENT_OUTPUT. Match
        // both sides of this first-use transition to that stage even when a
        // newly-created swapchain image still has UNDEFINED layout: discarding
        // its contents removes the source access, not the need to wait until
        // presentation has released the image.
        image_barriers_.transitionImage(
            command_buffer,
            swapchain_images_[image_index],
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VulkanImageBarrierTracker::AccessScope{
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_NONE},
            VulkanImageBarrierTracker::AccessScope{
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT});

        if (current_frame >= depth_stencil_resources_.size() ||
            depth_stencil_resources_[current_frame].image == VK_NULL_HANDLE ||
            depth_stencil_resources_[current_frame].view == VK_NULL_HANDLE) {
            framebuffer_resized_ = true;
            return fail(std::format("Missing depth/stencil resource for frame slot {}", current_frame));
        }
        const DepthStencilResource& depth_stencil = depth_stencil_resources_[current_frame];
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
        rendering_info.renderArea.extent = render_extent;
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
        frame.extent = render_extent;
        frame_active_ = true;
        frame_rendering_active_ = true;
        last_error_.clear();
        return true;
    }

    bool VulkanContext::finishActiveRendering(const VkCommandBuffer command_buffer) {
        if (!frame_rendering_active_)
            return true;
        if (command_buffer == VK_NULL_HANDLE)
            return fail("Cannot finish Vulkan rendering without an active command buffer");

        vkCmdEndRendering(command_buffer);
        frame_rendering_active_ = false;
        return true;
    }

    bool VulkanContext::endFrame() {
        if (!frame_active_) {
            return fail(std::format(
                "endFrame called with no active frame (frame_active={}, rendering_active={}, frame_index={}, active_frame_index={}, active_image_index={}): beginFrame must succeed first",
                frame_active_,
                frame_rendering_active_,
                frame_index_,
                active_frame_index_,
                active_image_index_));
        }

        if (!drainCompletedImmediateSubmits()) {
            frame_active_ = false;
            frame_rendering_active_ = false;
            framebuffer_resized_ = true;
            return false;
        }

        const std::size_t current_frame = active_frame_index_;
        if (current_frame >= kFramesInFlight || current_frame >= command_buffers_.size() ||
            current_frame >= in_flight_.size() || current_frame >= frame_submit_serials_.size()) {
            frame_active_ = false;
            frame_rendering_active_ = false;
            framebuffer_resized_ = true;
            return fail(std::format(
                "endFrame active frame slot is outside a per-frame array (active_frame_index={}, frames_in_flight={}, command_buffers={}, fences={}, submit_serials={})",
                current_frame,
                kFramesInFlight,
                command_buffers_.size(),
                in_flight_.size(),
                frame_submit_serials_.size()));
        }
        if (active_image_index_ >= swapchain_images_.size() ||
            active_image_index_ >= swapchain_images_in_flight_.size() ||
            active_image_index_ >= render_finished_.size() ||
            active_acquire_index_ >= image_available_.size()) {
            frame_active_ = false;
            frame_rendering_active_ = false;
            framebuffer_resized_ = true;
            return fail(std::format(
                "endFrame active swapchain indices are outside their arrays (active_image_index={}, images={}, image_fences={}, render_finished={}, active_acquire_index={}, acquire_semaphores={})",
                active_image_index_,
                swapchain_images_.size(),
                swapchain_images_in_flight_.size(),
                render_finished_.size(),
                active_acquire_index_,
                image_available_.size()));
        }
        if (!frame_timeline_waits_valid_) {
            frame_active_ = false;
            frame_rendering_active_ = false;
            framebuffer_resized_ = true;
            return false;
        }
        VkCommandBuffer command_buffer = command_buffers_[current_frame];
        if (command_buffer == VK_NULL_HANDLE) {
            frame_active_ = false;
            frame_rendering_active_ = false;
            framebuffer_resized_ = true;
            return fail(std::format(
                "endFrame requires a recorded command buffer (frame_slot={}, command_buffer={:#x}, image_index={}, rendering_active={})",
                current_frame,
                vkHandleValue(command_buffer),
                active_image_index_,
                frame_rendering_active_));
        }
        if (!finishActiveRendering(command_buffer)) {
            frame_active_ = false;
            framebuffer_resized_ = true;
            return false;
        }
        image_barriers_.transitionImage(command_buffer,
                                        swapchain_images_[active_image_index_],
                                        VK_IMAGE_ASPECT_COLOR_BIT,
                                        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

        VkResult result = vkEndCommandBuffer(command_buffer);
        if (result != VK_SUCCESS) {
            frame_active_ = false;
            framebuffer_resized_ = true;
            return fail(std::format(
                "vkEndCommandBuffer failed for the active frame (frame_slot={}, command_buffer={:#x}, image_index={}, result={}({}))",
                current_frame,
                vkHandleValue(command_buffer),
                active_image_index_,
                vkResultToString(result),
                static_cast<int>(result)));
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
        if (active_image_index_ >= render_finished_.size()) {
            frame_active_ = false;
            framebuffer_resized_ = true;
            return fail(std::format("Missing render-finished semaphore for swapchain image {}",
                                    active_image_index_));
        }
        const VkSemaphore render_finished = render_finished_[active_image_index_];
        if (render_finished == VK_NULL_HANDLE ||
            image_available_[active_acquire_index_] == VK_NULL_HANDLE) {
            frame_active_ = false;
            framebuffer_resized_ = true;
            return fail(std::format(
                "endFrame submission requires valid acquire and render-finished semaphores (image_index={}, render_finished={:#x}, acquire_index={}, image_available={:#x})",
                active_image_index_,
                vkHandleValue(render_finished),
                active_acquire_index_,
                vkHandleValue(image_available_[active_acquire_index_])));
        }
        submit_info.pSignalSemaphores = &render_finished;

        VkFence frame_fence = in_flight_[current_frame];
        if (frame_fence == VK_NULL_HANDLE) {
            frame_active_ = false;
            framebuffer_resized_ = true;
            return fail(std::format(
                "endFrame submission requires a non-null frame fence (frame_slot={}, fence={:#x}, image_index={}, prior_submit_id={})",
                current_frame,
                vkHandleValue(frame_fence),
                active_image_index_,
                frame_submit_serials_[current_frame]));
        }
        if (wait_semaphores.size() != wait_stages.size() ||
            wait_semaphores.size() != wait_values.size() ||
            timeline_submit_info.waitSemaphoreValueCount != submit_info.waitSemaphoreCount ||
            timeline_submit_info.signalSemaphoreValueCount != submit_info.signalSemaphoreCount ||
            submit_info.commandBufferCount != 1 || submit_info.pCommandBuffers == nullptr) {
            frame_active_ = false;
            framebuffer_resized_ = true;
            return fail(std::format(
                "endFrame submit arrays disagree with VkSubmitInfo counts (wait_handles={}, wait_stages={}, wait_values={}, submit_wait_count={}, timeline_wait_count={}, submit_signal_count={}, timeline_signal_count={}, command_buffer_count={}, command_buffer_pointer={:#x})",
                wait_semaphores.size(),
                wait_stages.size(),
                wait_values.size(),
                submit_info.waitSemaphoreCount,
                timeline_submit_info.waitSemaphoreValueCount,
                submit_info.signalSemaphoreCount,
                timeline_submit_info.signalSemaphoreValueCount,
                submit_info.commandBufferCount,
                vkHandleValue(submit_info.pCommandBuffers)));
        }
        const std::uint64_t submit_id = ++frame_submit_serial_;
        result = vkResetFences(device_, 1, &frame_fence);
        if (result != VK_SUCCESS) {
            frame_active_ = false;
            framebuffer_resized_ = true;
            const bool fence_recovered = replaceFrameFenceSignaled(current_frame);
            return fail(std::format("vkResetFences(frame slot {}, submit_id {}) failed: {}",
                                    current_frame,
                                    submit_id,
                                    vkResultToString(result)) +
                        (fence_recovered ? "; frame fence replaced and swapchain retirement scheduled"
                                         : "; frame fence recovery failed"));
        }
        LOG_DEBUG("Vulkan endFrame submit: submit_id={}, frame_slot={}, image={}, acquire_index={}, waits={}, timeline_waits={}, framebuffer={}x{}, extent={}x{}",
                  submit_id,
                  current_frame,
                  active_image_index_,
                  active_acquire_index_,
                  wait_semaphores.size(),
                  frame_timeline_waits_.size(),
                  framebuffer_width_,
                  framebuffer_height_,
                  swapchain_extent_.width,
                  swapchain_extent_.height);
        result = vkQueueSubmit(graphics_queue_, 1, &submit_info, frame_fence);
        frame_timeline_waits_.clear();
        if (result != VK_SUCCESS) {
            frame_active_ = false;
            // The acquire semaphore was signaled but never consumed, and the acquired image cannot
            // be returned directly. Retire the swapchain before another acquire; replace the reset
            // frame fence now so recreation cannot block forever waiting on an unsignaled fence.
            framebuffer_resized_ = true;
            const bool fence_recovered = replaceFrameFenceSignaled(current_frame);
            return fail(std::format("vkQueueSubmit(frame slot {}, submit_id {}, image {}) failed: {}",
                                    current_frame,
                                    submit_id,
                                    active_image_index_,
                                    vkResultToString(result)) +
                        (fence_recovered ? "; frame fence replaced and swapchain retirement scheduled"
                                         : "; frame fence recovery failed"));
        }
        frame_submit_serials_[current_frame] = submit_id;
        if (active_image_index_ < swapchain_images_in_flight_.size()) {
            swapchain_images_in_flight_[active_image_index_] = frame_fence;
        }

        VkPresentInfoKHR present_info{};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &render_finished;
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapchain_;
        present_info.pImageIndices = &active_image_index_;
        if (present_queue_ == VK_NULL_HANDLE || swapchain_ == VK_NULL_HANDLE) {
            frame_active_ = false;
            framebuffer_resized_ = true;
            return fail(std::format(
                "endFrame presentation requires a valid queue and swapchain (present_queue={:#x}, swapchain={:#x}, image_index={}, wait_semaphore={:#x})",
                vkHandleValue(present_queue_),
                vkHandleValue(swapchain_),
                active_image_index_,
                vkHandleValue(render_finished)));
        }
        result = vkQueuePresentKHR(present_queue_, &present_info);

        frame_active_ = false;
        frame_index_ = (frame_index_ + 1) % kFramesInFlight;
        const bool swapchain_covers_framebuffer = framebufferFitsSwapchainExtent();
        const bool should_recreate_for_present =
            result == VK_ERROR_OUT_OF_DATE_KHR ||
            ((result == VK_SUBOPTIMAL_KHR || frame_suboptimal_) && !swapchain_covers_framebuffer);
        if (should_recreate_for_present) {
            LOG_DEBUG("Vulkan present requested swapchain recreate: result={}, frame_suboptimal={}, image={}, framebuffer={}x{}, extent={}x{}",
                      vkResultToString(result),
                      frame_suboptimal_,
                      active_image_index_,
                      framebuffer_width_,
                      framebuffer_height_,
                      swapchain_extent_.width,
                      swapchain_extent_.height);
            frame_suboptimal_ = false;
            framebuffer_resize_exact_after_interactive_ = false;
            deferSwapchainResizeRecreate(true, false);
            return true;
        }
        frame_suboptimal_ = false;
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            return fail(std::format("vkQueuePresentKHR(image {}, frame slot {}) failed: {}",
                                    active_image_index_,
                                    current_frame,
                                    vkResultToString(result)));
        }

        return true;
    }

    std::expected<VulkanContext::WindowCapture, std::string> VulkanContext::captureAndEndActiveFrameRgba() {
        auto fail_capture = [this](std::string message) -> std::expected<WindowCapture, std::string> {
            fail(std::move(message));
            return std::unexpected(last_error_);
        };

        if (!frame_active_)
            return fail_capture("Full-window capture requires an active Vulkan GUI frame");
        if (device_ == VK_NULL_HANDLE || allocator_ == VK_NULL_HANDLE)
            return fail_capture("Full-window capture requires initialized Vulkan resources");
        if ((swapchain_image_usage_ & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0)
            return fail_capture("The Vulkan swapchain does not support transfer-source readback");
        if (active_image_index_ >= swapchain_images_.size())
            return fail_capture(std::format(
                "Full-window capture active image index is outside the swapchain (active_image_index={}, swapchain_image_count={})",
                active_image_index_,
                swapchain_images_.size()));
        if (active_frame_index_ >= command_buffers_.size() ||
            command_buffers_[active_frame_index_] == VK_NULL_HANDLE) {
            return fail_capture(std::format(
                "Full-window capture active frame slot must reference a command buffer (active_frame_index={}, command_buffer_count={}, command_buffer={:#x})",
                active_frame_index_,
                command_buffers_.size(),
                active_frame_index_ < command_buffers_.size()
                    ? vkHandleValue(command_buffers_[active_frame_index_])
                    : 0));
        }

        const bool bgra =
            swapchain_format_ == VK_FORMAT_B8G8R8A8_UNORM ||
            swapchain_format_ == VK_FORMAT_B8G8R8A8_SRGB;
        const bool rgba =
            swapchain_format_ == VK_FORMAT_R8G8B8A8_UNORM ||
            swapchain_format_ == VK_FORMAT_R8G8B8A8_SRGB;
        if (!bgra && !rgba) {
            return fail_capture(std::format("Full-window capture does not support swapchain format {}",
                                            vkFormatToString(swapchain_format_)));
        }

        const auto extent = framebufferExtent();
        if (extent.width == 0 || extent.height == 0)
            return fail_capture("Cannot capture a zero-sized Vulkan swapchain");

        const VkDeviceSize byte_size =
            static_cast<VkDeviceSize>(extent.width) *
            static_cast<VkDeviceSize>(extent.height) * 4u;

        VkBuffer staging_buffer = VK_NULL_HANDLE;
        VmaAllocation staging_allocation = VK_NULL_HANDLE;
        auto destroy_staging = [&]() {
            if (staging_buffer != VK_NULL_HANDLE && staging_allocation != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator_, staging_buffer, staging_allocation);
            }
            staging_buffer = VK_NULL_HANDLE;
            staging_allocation = VK_NULL_HANDLE;
        };

        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = byte_size;
        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocation_info{};
        allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

        VmaAllocationInfo staging_info{};
        VkResult result = static_cast<VkResult>(vmaCreateBuffer(allocator_,
                                                                &buffer_info,
                                                                &allocation_info,
                                                                &staging_buffer,
                                                                &staging_allocation,
                                                                &staging_info));
        if (result != VK_SUCCESS)
            return fail_capture(std::format(
                "vmaCreateBuffer(window capture) failed: {} ({}) (allocator={:#x}, requested_size={}, usage={:#x})",
                vkResultToString(result),
                static_cast<int>(result),
                reinterpret_cast<std::uintptr_t>(allocator_),
                byte_size,
                static_cast<std::uint32_t>(buffer_info.usage)));
        if (staging_buffer == VK_NULL_HANDLE || staging_allocation == VK_NULL_HANDLE ||
            staging_info.size < byte_size) {
            const std::string error = std::format(
                "Full-window capture staging allocation must cover the image copy (buffer={:#x}, allocation={:#x}, allocation_size={}, copy_size={}, extent={}x{})",
                vkHandleValue(staging_buffer),
                reinterpret_cast<std::uintptr_t>(staging_allocation),
                staging_info.size,
                byte_size,
                extent.width,
                extent.height);
            destroy_staging();
            return fail_capture(error);
        }
        vmaSetAllocationName(allocator_, staging_allocation, "Window capture readback");
        setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                            staging_buffer,
                            "window.capture.readback[{}]",
                            byte_size);

        VkCommandBuffer command_buffer = command_buffers_[active_frame_index_];
        if (!finishActiveRendering(command_buffer)) {
            destroy_staging();
            return std::unexpected(last_error_);
        }

        const VkImage image = swapchain_images_[active_image_index_];
        image_barriers_.transitionImage(command_buffer,
                                        image,
                                        VK_IMAGE_ASPECT_COLOR_BIT,
                                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        VkBufferImageCopy copy_region{};
        copy_region.bufferOffset = 0;
        copy_region.bufferRowLength = 0;
        copy_region.bufferImageHeight = 0;
        copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.imageSubresource.mipLevel = 0;
        copy_region.imageSubresource.baseArrayLayer = 0;
        copy_region.imageSubresource.layerCount = 1;
        copy_region.imageOffset = {0, 0, 0};
        copy_region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(command_buffer,
                               image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               staging_buffer,
                               1,
                               &copy_region);

        if (!endFrame()) {
            const std::string error = last_error_.empty() ? "Vulkan frame submission failed during window capture"
                                                          : last_error_;
            (void)deviceWaitIdle();
            destroy_staging();
            return std::unexpected(error);
        }
        if (!waitForSubmittedFrames()) {
            destroy_staging();
            return std::unexpected(last_error_.empty() ? "Timed out waiting for window capture readback"
                                                       : last_error_);
        }

        result = static_cast<VkResult>(vmaInvalidateAllocation(allocator_, staging_allocation, 0, byte_size));
        if (result != VK_SUCCESS) {
            destroy_staging();
            return fail_capture(std::format("vmaInvalidateAllocation(window capture) failed: {}",
                                            vkResultToString(result)));
        }

        void* mapped = nullptr;
        result = static_cast<VkResult>(vmaMapMemory(allocator_, staging_allocation, &mapped));
        if (result != VK_SUCCESS || !mapped) {
            destroy_staging();
            return fail_capture(std::format("vmaMapMemory(window capture) failed: {}",
                                            vkResultToString(result)));
        }

        WindowCapture capture;
        capture.width = static_cast<int>(extent.width);
        capture.height = static_cast<int>(extent.height);
        capture.rgba.resize(static_cast<std::size_t>(byte_size));

        const auto* src = static_cast<const std::uint8_t*>(mapped);
        if (rgba) {
            std::memcpy(capture.rgba.data(), src, capture.rgba.size());
        } else {
            for (std::size_t i = 0; i < capture.rgba.size(); i += 4) {
                capture.rgba[i + 0] = src[i + 2];
                capture.rgba[i + 1] = src[i + 1];
                capture.rgba[i + 2] = src[i + 0];
                capture.rgba[i + 3] = src[i + 3];
            }
        }

        vmaUnmapMemory(allocator_, staging_allocation);
        destroy_staging();
        last_error_.clear();
        return capture;
    }

    bool VulkanContext::waitForCurrentFrameSlot() {
        if (device_ == VK_NULL_HANDLE) {
            return fail("Cannot wait for Vulkan frame slot before device initialization");
        }
        const std::size_t current_frame = frame_index_;
        if (current_frame >= in_flight_.size() ||
            current_frame >= frame_submit_serials_.size()) {
            return fail(std::format(
                "Current Vulkan frame slot is outside synchronization arrays (frame_index={}, in_flight_count={}, submit_serial_count={})",
                current_frame,
                in_flight_.size(),
                frame_submit_serials_.size()));
        }
        VkFence frame_fence = in_flight_[current_frame];
        if (frame_fence == VK_NULL_HANDLE) {
            return fail(std::format(
                "Current Vulkan frame slot has no synchronization fence (frame_index={}, fence={:#x}, last_submit_id={})",
                current_frame,
                vkHandleValue(frame_fence),
                frame_submit_serials_[current_frame]));
        }
        const VkResult result = vkWaitForFences(device_, 1, &frame_fence, VK_TRUE, kWaitForeverNs);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkWaitForFences(frame slot {}) failed: {} (last_submit_id={})",
                                    current_frame,
                                    vkResultToString(result),
                                    frame_submit_serials_[current_frame]));
        }
        last_error_.clear();
        return true;
    }

    bool VulkanContext::waitForSubmittedFrames() {
        return waitForFrameFences();
    }

    bool VulkanContext::waitForImmediateSubmits() {
        if (device_ == VK_NULL_HANDLE || pending_immediate_submits_.empty()) {
            return true;
        }

        constexpr std::uint64_t kImmediateWaitTimeoutNs = 2'000'000'000ull;
        std::vector<VkFence> fences;
        fences.reserve(pending_immediate_submits_.size());
        for (const auto& pending : pending_immediate_submits_) {
            if (pending.fence != VK_NULL_HANDLE) {
                fences.push_back(pending.fence);
            }
        }
        if (!fences.empty()) {
            const VkResult result = vkWaitForFences(device_,
                                                    static_cast<std::uint32_t>(fences.size()),
                                                    fences.data(),
                                                    VK_TRUE,
                                                    kImmediateWaitTimeoutNs);
            if (result != VK_SUCCESS) {
                return fail(std::format("vkWaitForFences(immediate submits) failed: {}",
                                        vkResultToString(result)));
            }
        }

        for (auto& pending : pending_immediate_submits_) {
            if (pending.fence != VK_NULL_HANDLE) {
                vkDestroyFence(device_, pending.fence, nullptr);
            }
            if (pending.cmd != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(device_, immediate_command_pool_, 1, &pending.cmd);
            }
        }
        pending_immediate_submits_.clear();
        last_error_.clear();
        return true;
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
        if (!frame_active_ || semaphore == VK_NULL_HANDLE || value == 0 || wait_stage == 0) {
            frame_timeline_waits_valid_ = false;
            fail(std::format(
                "addFrameTimelineWait requires an active frame and a valid non-zero timeline edge (frame_active={}, semaphore={:#x}, value={}, wait_stage={:#x}, pending_waits={})",
                frame_active_,
                vkHandleValue(semaphore),
                value,
                static_cast<std::uint64_t>(wait_stage),
                frame_timeline_waits_.size()));
            return;
        }
        const std::uint64_t previous = last_frame_timeline_wait_values_[semaphore];
        if (value <= previous) {
            frame_timeline_waits_valid_ = false;
            fail(std::format(
                "Frame timeline waits must increase strictly (semaphore={:#x}, requested_value={}, previous_value={}, wait_stage={:#x}, frame_slot={}, image_index={})",
                vkHandleValue(semaphore),
                value,
                previous,
                static_cast<std::uint64_t>(wait_stage),
                active_frame_index_,
                active_image_index_));
            return;
        }
        last_frame_timeline_wait_values_[semaphore] = value;
        frame_timeline_waits_.push_back(FrameTimelineWait{
            .semaphore = semaphore,
            .value = value,
            .wait_stage = wait_stage,
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
        LFS_VK_CONTEXT_CHECK_MSG(
            vkEnumerateInstanceExtensionProperties(nullptr, &available_extension_count, nullptr),
            "Failed to enumerate Vulkan instance-extension count (observed_count={})",
            available_extension_count);
        std::vector<VkExtensionProperties> available_extensions(available_extension_count);
        if (available_extension_count > 0) {
            LFS_VK_CONTEXT_CHECK_MSG(
                vkEnumerateInstanceExtensionProperties(
                    nullptr, &available_extension_count, available_extensions.data()),
                "Failed to enumerate Vulkan instance extensions (destination_capacity={}, observed_count={})",
                available_extensions.size(),
                available_extension_count);
            available_extensions.resize(available_extension_count);
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

        instance_surface_maintenance_enabled_ =
            extensionAvailable(available_extensions, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME) &&
            extensionAvailable(available_extensions, VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
        if (instance_surface_maintenance_enabled_) {
            appendUniqueExtension(extensions, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
            appendUniqueExtension(extensions, VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
        }

        debug_utils_enabled_ = extensionAvailable(available_extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        if (debug_utils_enabled_) {
            appendUniqueExtension(extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        uint32_t available_layer_count = 0;
        LFS_VK_CONTEXT_CHECK_MSG(
            vkEnumerateInstanceLayerProperties(&available_layer_count, nullptr),
            "Failed to enumerate Vulkan instance-layer count (observed_count={})",
            available_layer_count);
        std::vector<VkLayerProperties> available_layers(available_layer_count);
        if (available_layer_count > 0) {
            LFS_VK_CONTEXT_CHECK_MSG(
                vkEnumerateInstanceLayerProperties(&available_layer_count, available_layers.data()),
                "Failed to enumerate Vulkan instance layers (destination_capacity={}, observed_count={})",
                available_layers.size(),
                available_layer_count);
            available_layers.resize(available_layer_count);
        }

        std::vector<const char*> layers;
        const bool validation_requested = validationRequestedByBuild();
        validation_errors_fatal_ = lfs::core::environment::flag("LFS_VK_VALIDATION_FATAL");
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
            populateDebugMessengerCreateInfo(debug_create_info, &validation_errors_fatal_);
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
            const VkResult present_result =
                vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &present_supported);
            if (present_result != VK_SUCCESS) {
                LOG_ERROR("vkGetPhysicalDeviceSurfaceSupportKHR failed while discovering queue families (physical_device={:#x}, surface={:#x}, queue_family={}, queue_family_count={}, result={}({}))",
                          vkHandleValue(device),
                          vkHandleValue(surface_),
                          i,
                          count,
                          vkResultToString(present_result),
                          static_cast<int>(present_result));
                return {};
            }
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
        VkResult result = vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
        if (result != VK_SUCCESS) {
            LOG_ERROR("Vulkan: {}",
                      formatVkCheckFailure(
                          "vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr)",
                          result,
                          std::format("Failed to enumerate device-extension count (physical_device={:#x}, observed_count={})",
                                      vkHandleValue(device),
                                      count),
                          __FILE__,
                          __LINE__));
            return false;
        }
        std::vector<VkExtensionProperties> extensions(count);
        if (count > 0) {
            result = vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data());
            if (result != VK_SUCCESS) {
                LOG_ERROR("Vulkan: {}",
                          formatVkCheckFailure(
                              "vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data())",
                              result,
                              std::format("Failed to enumerate device extensions (physical_device={:#x}, destination_capacity={}, observed_count={})",
                                          vkHandleValue(device),
                                          extensions.size(),
                                          count),
                              __FILE__,
                              __LINE__));
                return false;
            }
            extensions.resize(count);
        }

        std::set<std::string> required{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        for (const auto& extension : extensions) {
            required.erase(extension.extensionName);
        }
        return required.empty();
    }

    VulkanContext::SwapchainSupport VulkanContext::querySwapchainSupport(VkPhysicalDevice device) const {
        SwapchainSupport details;
        VkResult result =
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &details.capabilities);
        if (result != VK_SUCCESS) {
            LOG_ERROR("vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed (physical_device={:#x}, surface={:#x}, result={}({}))",
                      vkHandleValue(device),
                      vkHandleValue(surface_),
                      vkResultToString(result),
                      static_cast<int>(result));
            return {};
        }

        uint32_t count = 0;
        result = vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &count, nullptr);
        if (result != VK_SUCCESS) {
            LOG_ERROR("vkGetPhysicalDeviceSurfaceFormatsKHR count query failed (physical_device={:#x}, surface={:#x}, observed_count={}, result={}({}))",
                      vkHandleValue(device),
                      vkHandleValue(surface_),
                      count,
                      vkResultToString(result),
                      static_cast<int>(result));
            return {};
        }
        details.formats.resize(count);
        if (count > 0) {
            result = vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &count, details.formats.data());
            if (result != VK_SUCCESS) {
                LOG_ERROR("vkGetPhysicalDeviceSurfaceFormatsKHR data query failed (physical_device={:#x}, surface={:#x}, destination_capacity={}, observed_count={}, result={}({}))",
                          vkHandleValue(device),
                          vkHandleValue(surface_),
                          details.formats.size(),
                          count,
                          vkResultToString(result),
                          static_cast<int>(result));
                return {};
            }
            details.formats.resize(count);
        }

        count = 0;
        result = vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &count, nullptr);
        if (result != VK_SUCCESS) {
            LOG_ERROR("vkGetPhysicalDeviceSurfacePresentModesKHR count query failed (physical_device={:#x}, surface={:#x}, observed_count={}, result={}({}))",
                      vkHandleValue(device),
                      vkHandleValue(surface_),
                      count,
                      vkResultToString(result),
                      static_cast<int>(result));
            return {};
        }
        details.present_modes.resize(count);
        if (count > 0) {
            result = vkGetPhysicalDeviceSurfacePresentModesKHR(
                device, surface_, &count, details.present_modes.data());
            if (result != VK_SUCCESS) {
                LOG_ERROR("vkGetPhysicalDeviceSurfacePresentModesKHR data query failed (physical_device={:#x}, surface={:#x}, destination_capacity={}, observed_count={}, result={}({}))",
                          vkHandleValue(device),
                          vkHandleValue(surface_),
                          details.present_modes.size(),
                          count,
                          vkResultToString(result),
                          static_cast<int>(result));
                return {};
            }
            details.present_modes.resize(count);
        }

        return details;
    }

    bool VulkanContext::pickPhysicalDevice() {
        uint32_t count = 0;
        LFS_VK_CONTEXT_CHECK_MSG(vkEnumeratePhysicalDevices(instance_, &count, nullptr),
                                 "Failed to enumerate physical-device count (instance={:#x}, observed_count={})",
                                 vkHandleValue(instance_),
                                 count);
        if (count == 0) {
            return fail(std::format("No Vulkan physical devices found (instance={:#x}, observed_count=0)",
                                    vkHandleValue(instance_)));
        }

        std::vector<VkPhysicalDevice> devices(count);
        LFS_VK_CONTEXT_CHECK_MSG(vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
                                 "Failed to enumerate physical devices (instance={:#x}, destination_capacity={}, observed_count={})",
                                 vkHandleValue(instance_),
                                 devices.size(),
                                 count);
        devices.resize(count);

        VkPhysicalDevice fallback = VK_NULL_HANDLE;
        VkPhysicalDevice first_discrete = VK_NULL_HANDLE;
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
                // Prefer the discrete GPU that matches CUDA's device 0. Picking
                // the first discrete GPU in Vulkan's enumeration order can land
                // the viewer on a different physical card than the CUDA trainer
                // on multi-GPU systems; CUDA<->Vulkan external-memory import then
                // fails (VK_ERROR_OUT_OF_DEVICE_MEMORY) and corrupts the CUDA
                // context, surfacing as a spurious "out of GPU memory".
                if (first_discrete == VK_NULL_HANDLE) {
                    first_discrete = device;
                }
                if (vulkanDeviceMatchesCudaDevice(device, 0)) {
                    physical_device_ = device;
                    break;
                }
            }
        }

        // No CUDA-matched discrete GPU: keep the legacy "first discrete" choice,
        // then any presentable device.
        if (physical_device_ == VK_NULL_HANDLE) {
            physical_device_ = first_discrete;
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
        LFS_VK_CONTEXT_CHECK_MSG(
            vkEnumerateDeviceExtensionProperties(
                physical_device_, nullptr, &available_extension_count, nullptr),
            "Failed to enumerate selected-device extension count (physical_device={:#x}, observed_count={})",
            vkHandleValue(physical_device_),
            available_extension_count);
        std::vector<VkExtensionProperties> available_extensions(available_extension_count);
        if (available_extension_count > 0) {
            LFS_VK_CONTEXT_CHECK_MSG(
                vkEnumerateDeviceExtensionProperties(physical_device_,
                                                     nullptr,
                                                     &available_extension_count,
                                                     available_extensions.data()),
                "Failed to enumerate selected-device extensions (physical_device={:#x}, destination_capacity={}, observed_count={})",
                vkHandleValue(physical_device_),
                available_extensions.size(),
                available_extension_count);
            available_extensions.resize(available_extension_count);
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

        const bool enable_shader_atomic_float =
            extensionAvailable(available_extensions, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
        if (enable_shader_atomic_float) {
            appendUniqueExtension(extensions, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
        }

        // Optional extensions with active consumers.
        const bool enable_push_descriptor =
            extensionAvailable(available_extensions, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        if (enable_push_descriptor) {
            appendUniqueExtension(extensions, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        }
        const bool enable_host_image_copy =
            extensionAvailable(available_extensions, VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME);
        if (enable_host_image_copy) {
            appendUniqueExtension(extensions, VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME);
        }
        const bool swapchain_maintenance1_available =
            instance_surface_maintenance_enabled_ &&
            extensionAvailable(available_extensions, VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);

        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT supported_atomic_float_features{};
        supported_atomic_float_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        VkPhysicalDeviceVulkan13Features supported_features13{};
        supported_features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        supported_features13.pNext =
            enable_shader_atomic_float ? static_cast<void*>(&supported_atomic_float_features) : nullptr;
        VkPhysicalDeviceVulkan12Features supported_features12{};
        supported_features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        supported_features12.pNext = &supported_features13;

        // Query extension feature structs separately so the main Vulkan 1.2
        // supported-feature chain remains stable.
        VkPhysicalDeviceHostImageCopyFeaturesEXT supported_host_image_copy{};
        supported_host_image_copy.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT;
        VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT supported_swapchain_maintenance1{};
        supported_swapchain_maintenance1.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT;

        void* opt_supported_head = nullptr;
        if (swapchain_maintenance1_available) {
            supported_swapchain_maintenance1.pNext = opt_supported_head;
            opt_supported_head = &supported_swapchain_maintenance1;
        }
        if (enable_host_image_copy) {
            supported_host_image_copy.pNext = opt_supported_head;
            opt_supported_head = &supported_host_image_copy;
        }

        VkPhysicalDeviceVulkan11Features supported_features11{};
        supported_features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        supported_features11.pNext = &supported_features12;

        VkPhysicalDeviceFeatures2 supported_features2{};
        supported_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        supported_features2.pNext = &supported_features11;
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
        features13.subgroupSizeControl = supported_features13.subgroupSizeControl;
        features13.computeFullSubgroups = supported_features13.computeFullSubgroups;

        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features{};
        atomic_float_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        atomic_float_features.shaderBufferFloat32AtomicAdd =
            enable_shader_atomic_float && supported_atomic_float_features.shaderBufferFloat32AtomicAdd
                ? VK_TRUE
                : VK_FALSE;
        features13.pNext = enable_shader_atomic_float
                               ? static_cast<void*>(&atomic_float_features)
                               : nullptr;

        VkPhysicalDeviceVulkan12Features features12{};
        features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        features12.pNext = &features13;
        features12.timelineSemaphore = VK_TRUE;
        features12.shaderFloat16 = supported_features12.shaderFloat16;

        // Optional feature structs are prepended to the Vulkan 1.2 chain.
        void* enabled_chain_head = features12.pNext;

        VkPhysicalDeviceHostImageCopyFeaturesEXT host_image_copy_features{};
        host_image_copy_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT;
        bool enable_host_image_copy_feature =
            enable_host_image_copy && supported_host_image_copy.hostImageCopy == VK_TRUE;
        // Pascal drivers advertise hostImageCopy but crash inside
        // vkTransitionImageLayoutEXT (#1298).
        if (enable_host_image_copy_feature && isPreVoltaCudaDevice(device_uuid_)) {
            LOG_INFO("Vulkan: disabling VK_EXT_host_image_copy on pre-Volta GPU (driver bug)");
            enable_host_image_copy_feature = false;
        }
        if (enable_host_image_copy_feature) {
            host_image_copy_features.hostImageCopy = VK_TRUE;
            host_image_copy_features.pNext = enabled_chain_head;
            enabled_chain_head = &host_image_copy_features;
        }

        const bool enable_swapchain_maintenance1 =
            swapchain_maintenance1_available &&
            supported_swapchain_maintenance1.swapchainMaintenance1 == VK_TRUE;
        VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT swapchain_maintenance1_features{};
        swapchain_maintenance1_features.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT;
        if (enable_swapchain_maintenance1) {
            appendUniqueExtension(extensions, VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
            swapchain_maintenance1_features.swapchainMaintenance1 = VK_TRUE;
            swapchain_maintenance1_features.pNext = enabled_chain_head;
            enabled_chain_head = &swapchain_maintenance1_features;
        }

        // 16-bit storage for the fp16 splat raster path (half4 partials,
        // half-packed staging). Mirrors device support; consumers must check
        // hasFloat16Storage() and fall back to fp32 shader variants.
        VkPhysicalDeviceVulkan11Features features11{};
        features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        features11.storageBuffer16BitAccess = supported_features11.storageBuffer16BitAccess;
        features11.uniformAndStorageBuffer16BitAccess =
            supported_features11.uniformAndStorageBuffer16BitAccess;
        features11.pNext = enabled_chain_head;
        features12.pNext = &features11;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &features12;
        features2.features.shaderInt16 = supported_features2.features.shaderInt16;
        features2.features.shaderInt64 = supported_features2.features.shaderInt64;
        features2.features.fillModeNonSolid = supported_features2.features.fillModeNonSolid;
        features2.features.wideLines = supported_features2.features.wideLines;

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
            debug_name_writer_.initialize(device_);
            if (!debug_name_writer_.enabled()) {
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
        setDebugObjectName(VK_OBJECT_TYPE_DEVICE, device_, "lichtfeld.device");
        setDebugObjectName(VK_OBJECT_TYPE_QUEUE, graphics_queue_, "lichtfeld.queue.graphics");
        setDebugObjectName(VK_OBJECT_TYPE_QUEUE, present_queue_, "lichtfeld.queue.present");
        setDebugObjectName(VK_OBJECT_TYPE_QUEUE,
                           compute_queue_,
                           has_dedicated_compute_queue_ ? "lichtfeld.queue.compute"
                                                        : "lichtfeld.queue.graphics_compute");
        external_memory_interop_enabled_ = enable_external_memory;
        external_semaphore_interop_enabled_ = enable_external_semaphore;
        external_memory_dedicated_allocation_enabled_ = enable_dedicated_allocation;
        swapchain_maintenance1_enabled_ = enable_swapchain_maintenance1;
        has_push_descriptor_ = enable_push_descriptor;
        has_float16_storage_ = features12.shaderFloat16 == VK_TRUE &&
                               features11.storageBuffer16BitAccess == VK_TRUE;
        has_host_image_copy_ = enable_host_image_copy_feature;
        has_fill_mode_non_solid_ = features2.features.fillModeNonSolid == VK_TRUE;
        has_wide_lines_ = features2.features.wideLines == VK_TRUE;
        VkPhysicalDeviceProperties device_properties{};
        vkGetPhysicalDeviceProperties(physical_device_, &device_properties);
        line_width_range_[0] = device_properties.limits.lineWidthRange[0];
        line_width_range_[1] = device_properties.limits.lineWidthRange[1];
        if (!external_memory_interop_enabled_) {
            return fail("Vulkan external memory interop is required (KHR_external_memory + platform variant); device is missing the extension(s)");
        }
        if (!external_semaphore_interop_enabled_) {
            return fail("Vulkan external timeline-semaphore interop is required (KHR_external_semaphore + platform variant); device is missing the extension(s)");
        }
        LOG_INFO("Vulkan external memory interop enabled{}",
                 external_memory_dedicated_allocation_enabled_ ? " with dedicated allocations" : "");
        LOG_INFO("Vulkan external timeline semaphore interop enabled");
        LOG_INFO("Vulkan optional features: push_descriptor={} host_image_copy={} swapchain_maintenance1={} fill_mode_non_solid={} wide_lines={}",
                 has_push_descriptor_,
                 has_host_image_copy_,
                 swapchain_maintenance1_enabled_,
                 has_fill_mode_non_solid_,
                 has_wide_lines_);
        return true;
    }

    bool VulkanContext::createAllocator() {
        if (instance_ == VK_NULL_HANDLE || physical_device_ == VK_NULL_HANDLE || device_ == VK_NULL_HANDLE) {
            return fail("VMA allocator requires an initialized Vulkan instance, physical device, and device");
        }

        VmaAllocatorCreateInfo create_info{};
        create_info.flags = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
        create_info.physicalDevice = physical_device_;
        create_info.device = device_;
        create_info.instance = instance_;
        create_info.vulkanApiVersion = VK_API_VERSION_1_3;
        // Default large-heap block size is 256 MiB, which left ~130-200 MiB of
        // device memory parked in partially-filled blocks (block_bytes vs
        // allocation_bytes). 64 MiB caps trailing-block waste; large buffers still
        // get dedicated allocations and bypass blocks entirely.
        create_info.preferredLargeHeapBlockSize = VkDeviceSize{64} << 20;

        const VkResult result = vmaCreateAllocator(&create_info, &allocator_);
        if (result != VK_SUCCESS) {
            allocator_ = VK_NULL_HANDLE;
            return fail(std::format("vmaCreateAllocator failed: {}", vkResultToString(result)));
        }
        return true;
    }

    std::size_t VulkanContext::queryVmaUsedBytes() const {
        if (allocator_ == VK_NULL_HANDLE)
            return 0;
        // VK_EXT_memory_budget reports the *full* per-heap memory the driver attributes
        // to this process — swap-chain, framebuffer attachments, descriptor pools,
        // internal command-buffer state — not just VMA-routed allocations. Sum the
        // device-local heaps; host-visible heaps are reported but don't affect VRAM.
        VkPhysicalDeviceMemoryProperties mem_props{};
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);

        std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
        vmaGetHeapBudgets(allocator_, budgets.data());

        std::uint64_t total_usage = 0;
        std::uint64_t total_block_bytes = 0;
        std::uint64_t total_allocation_bytes = 0;
        for (std::uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
            if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                total_usage += budgets[i].usage;
                total_block_bytes += budgets[i].statistics.blockBytes;
                total_allocation_bytes += budgets[i].statistics.allocationBytes;
            }
        }
        auto& profiler = lfs::diagnostics::VramProfiler::instance();
        profiler.setGauge("vulkan.vma.budget_usage", static_cast<double>(total_usage));
        profiler.setGauge("vulkan.vma.block_bytes", static_cast<double>(total_block_bytes));
        profiler.setGauge("vulkan.vma.allocation_bytes", static_cast<double>(total_allocation_bytes));
        profiler.setVulkanVmaBlockBytes(static_cast<std::size_t>(total_block_bytes));
        const std::uint64_t block_free =
            total_block_bytes > total_allocation_bytes ? total_block_bytes - total_allocation_bytes : 0;
        recordCurrentVulkanBytes("vulkan.vma", "allocator_free_in_blocks", static_cast<std::size_t>(block_free));
        return static_cast<std::size_t>(total_usage);
    }

    std::string VulkanContext::makeAllocationDiagnosticLabel(const std::string_view label) {
        const std::uint64_t serial = ++allocation_diagnostic_serial_;
        if (label.empty()) {
            return std::format("allocation#{}", serial);
        }
        return std::format("{}#{}", label, serial);
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

    std::optional<VkSurfacePresentScalingCapabilitiesEXT>
    VulkanContext::queryPresentScalingCapabilities(const VkPresentModeKHR present_mode) const {
        if (!instance_surface_maintenance_enabled_ || !swapchain_maintenance1_enabled_) {
            return std::nullopt;
        }

        VkSurfacePresentModeEXT surface_present_mode{};
        surface_present_mode.sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_EXT;
        surface_present_mode.presentMode = present_mode;

        VkPhysicalDeviceSurfaceInfo2KHR surface_info{};
        surface_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR;
        surface_info.pNext = &surface_present_mode;
        surface_info.surface = surface_;

        VkSurfacePresentScalingCapabilitiesEXT scaling_capabilities{};
        scaling_capabilities.sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_SCALING_CAPABILITIES_EXT;

        VkSurfaceCapabilities2KHR capabilities{};
        capabilities.sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR;
        capabilities.pNext = &scaling_capabilities;

        const VkResult result =
            vkGetPhysicalDeviceSurfaceCapabilities2KHR(physical_device_, &surface_info, &capabilities);
        if (result != VK_SUCCESS) {
            LOG_DEBUG("Vulkan present scaling capability query failed: {}", vkResultToString(result));
            return std::nullopt;
        }

        const bool supports_one_to_one =
            (scaling_capabilities.supportedPresentScaling & VK_PRESENT_SCALING_ONE_TO_ONE_BIT_EXT) != 0;
        const bool supports_min_gravity =
            (scaling_capabilities.supportedPresentGravityX & VK_PRESENT_GRAVITY_MIN_BIT_EXT) != 0 &&
            (scaling_capabilities.supportedPresentGravityY & VK_PRESENT_GRAVITY_MIN_BIT_EXT) != 0;
        if (!supports_one_to_one || !supports_min_gravity) {
            return std::nullopt;
        }
        return scaling_capabilities;
    }

    VkExtent2D VulkanContext::chooseSwapchainExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                                                    const int framebuffer_width,
                                                    const int framebuffer_height,
                                                    const bool add_resize_headroom,
                                                    const VkSurfacePresentScalingCapabilitiesEXT* scaling_capabilities) const {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max() &&
            scaling_capabilities == nullptr) {
            return capabilities.currentExtent;
        }

        VkExtent2D extent{};
        extent.width = static_cast<uint32_t>(std::max(1, framebuffer_width));
        extent.height = static_cast<uint32_t>(std::max(1, framebuffer_height));
        const VkExtent2D min_extent = scaling_capabilities != nullptr
                                          ? scaling_capabilities->minScaledImageExtent
                                          : capabilities.minImageExtent;
        const VkExtent2D max_extent = scaling_capabilities != nullptr
                                          ? scaling_capabilities->maxScaledImageExtent
                                          : capabilities.maxImageExtent;
        if (add_resize_headroom) {
            extent.width = addResizeHeadroom(extent.width, max_extent.width);
            extent.height = addResizeHeadroom(extent.height, max_extent.height);
        }
        extent.width = std::clamp(extent.width, min_extent.width, max_extent.width);
        extent.height = std::clamp(extent.height, min_extent.height, max_extent.height);
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

    bool VulkanContext::createExternalImage(const VkExtent2D extent,
                                            const VkFormat format,
                                            ExternalImage& out,
                                            const std::string_view diagnostic_scope,
                                            const std::string_view diagnostic_label) {
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
        const VkExtent3D max_extent = format_properties.imageFormatProperties.maxExtent;
        if (extent.width > max_extent.width || extent.height > max_extent.height) {
            return fail(std::format(
                "External Vulkan image {}x{} exceeds device-supported limit {}x{} for format {}",
                extent.width,
                extent.height,
                max_extent.width,
                max_extent.height,
                static_cast<int>(format)));
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
        out.diagnostic_scope = diagnostic_scope.empty() ? "vulkan.external.image" : std::string(diagnostic_scope);
        out.diagnostic_label = makeAllocationDiagnosticLabel(diagnostic_label);

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
        setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE,
                            out.image,
                            "interop.external.image[{}x{}]",
                            extent.width,
                            extent.height);

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
        setDebugObjectNamef(VK_OBJECT_TYPE_DEVICE_MEMORY,
                            out.memory,
                            "interop.external.image[{}x{}].memory",
                            extent.width,
                            extent.height);
        recordCurrentVulkanBytes(out.diagnostic_scope, out.diagnostic_label, static_cast<std::size_t>(out.allocation_size));

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
        setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE_VIEW,
                            out.view,
                            "interop.external.image[{}x{}].view",
                            extent.width,
                            extent.height);
        return true;
    }

    void VulkanContext::destroyExternalImage(ExternalImage& image) {
        if (!image.diagnostic_scope.empty() && !image.diagnostic_label.empty()) {
            recordCurrentVulkanBytes(image.diagnostic_scope, image.diagnostic_label, 0);
        }
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
                                             ExternalBuffer& out,
                                             const std::string_view diagnostic_scope,
                                             const std::string_view diagnostic_label) {
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
        setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                            out.buffer,
                            "interop.external.buffer[{}]",
                            size);

        VkMemoryRequirements memory_requirements{};
        vkGetBufferMemoryRequirements(device_, out.buffer, &memory_requirements);
        out.size = size;
        out.allocation_size = memory_requirements.size;
        out.diagnostic_scope = diagnostic_scope.empty() ? "vulkan.external.buffer" : std::string(diagnostic_scope);
        out.diagnostic_label = makeAllocationDiagnosticLabel(diagnostic_label);

        // Mirror createExternalImage: when dedicated allocation is enabled the CUDA side
        // is told the import is dedicated (cudaExternalMemoryDedicated), so the Vulkan
        // allocation must actually be dedicated to this buffer or the two disagree.
        VkMemoryDedicatedAllocateInfo dedicated_info{};
        dedicated_info.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        dedicated_info.buffer = out.buffer;

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
        setDebugObjectNamef(VK_OBJECT_TYPE_DEVICE_MEMORY,
                            out.memory,
                            "interop.external.buffer[{}].memory",
                            out.allocation_size);
        recordCurrentVulkanBytes(out.diagnostic_scope, out.diagnostic_label, static_cast<std::size_t>(out.allocation_size));

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

    bool VulkanContext::importExternalBuffer(ExternalNativeHandle handle,
                                             const VkDeviceSize size,
                                             const VkBufferUsageFlags usage,
                                             ExternalBuffer& out,
                                             const std::string_view diagnostic_scope,
                                             const std::string_view diagnostic_label) {
        out = {};

        if (!device_ || !physical_device_) {
            return fail("Cannot import external Vulkan buffer before device initialization");
        }
        if (size == 0) {
            return fail("Imported external Vulkan buffer requires a non-zero size");
        }
        if (!externalNativeHandleValid(handle)) {
            return fail("Imported external Vulkan buffer requires a valid native handle");
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
        // Same concurrent-queue rationale as createExternalBuffer.
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
            return fail(std::format("vkCreateBuffer(imported) failed: {}", vkResultToString(result)));
        }
        setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                            out.buffer,
                            "interop.imported.buffer[{}]",
                            size);

        VkMemoryRequirements memory_requirements{};
        vkGetBufferMemoryRequirements(device_, out.buffer, &memory_requirements);
        out.size = size;
        out.allocation_size = memory_requirements.size;
        out.diagnostic_scope = diagnostic_scope.empty() ? "vulkan.external.imported_buffer" : std::string(diagnostic_scope);
        out.diagnostic_label = makeAllocationDiagnosticLabel(diagnostic_label);

#ifdef _WIN32
        VkImportMemoryWin32HandleInfoKHR import_info{};
        import_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        import_info.handleType = kExternalMemoryHandleType;
        import_info.handle = handle;
#else
        // NVIDIA's driver takes ownership of the fd we pass to vkAllocateMemory and
        // will close it on vkFreeMemory. Dup so the original exporter (CUDA) can
        // still own its copy; both close their fd independently on teardown.
        const int dup_fd = ::dup(handle);
        if (dup_fd < 0) {
            destroyExternalBuffer(out);
            return fail("dup() of external memory fd failed for Vulkan import");
        }
        VkImportMemoryFdInfoKHR import_info{};
        import_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        import_info.handleType = kExternalMemoryHandleType;
        import_info.fd = dup_fd;
#endif

        VkMemoryAllocateInfo allocate_info{};
        allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate_info.pNext = &import_info;
        // The exporter created exactly `size` bytes; the importer must allocate the
        // SAME size, not memory_requirements.size which can be larger (e.g. when
        // Vulkan would round up for its own alignment). vkAllocateMemory with import
        // requires the original allocation size.
        allocate_info.allocationSize = size;
        allocate_info.memoryTypeIndex =
            findMemoryType(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (allocate_info.memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
#ifndef _WIN32
            ::close(dup_fd);
#endif
            destroyExternalBuffer(out);
            return fail("Could not find Vulkan device-local memory type for imported buffer");
        }

        result = vkAllocateMemory(device_, &allocate_info, nullptr, &out.memory);
        if (result != VK_SUCCESS) {
#ifndef _WIN32
            // On failure the driver did NOT take the fd; close it ourselves.
            ::close(dup_fd);
#endif
            destroyExternalBuffer(out);
            return fail(std::format("vkAllocateMemory(import) failed: {}", vkResultToString(result)));
        }
        setDebugObjectNamef(VK_OBJECT_TYPE_DEVICE_MEMORY,
                            out.memory,
                            "interop.imported.buffer[{}].memory",
                            size);

        result = vkBindBufferMemory(device_, out.buffer, out.memory, 0);
        if (result != VK_SUCCESS) {
            destroyExternalBuffer(out);
            return fail(std::format("vkBindBufferMemory(import) failed: {}", vkResultToString(result)));
        }

        // We do NOT own the handle; the exporter retains it. Leave native_handle invalid.
        out.native_handle = kInvalidExternalNativeHandle;
        return true;
    }

    void VulkanContext::destroyExternalBuffer(ExternalBuffer& buffer) {
        if (!buffer.diagnostic_scope.empty() && !buffer.diagnostic_label.empty()) {
            recordCurrentVulkanBytes(buffer.diagnostic_scope, buffer.diagnostic_label, 0);
        }
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
        setDebugObjectNamef(VK_OBJECT_TYPE_SEMAPHORE,
                            out.semaphore,
                            "interop.timeline.external[{}]",
                            initial_value);

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

    bool VulkanContext::drainCompletedImmediateSubmits() {
        if (device_ == VK_NULL_HANDLE || pending_immediate_submits_.empty()) {
            return true;
        }
        auto write = pending_immediate_submits_.begin();
        for (auto read = pending_immediate_submits_.begin(); read != pending_immediate_submits_.end(); ++read) {
            if (read->fence == VK_NULL_HANDLE || read->cmd == VK_NULL_HANDLE) {
                return fail(std::format(
                    "Immediate-submit retirement encountered an incomplete entry (entry={}, pending_count={}, command_buffer={:#x}, fence={:#x})",
                    std::distance(pending_immediate_submits_.begin(), read),
                    pending_immediate_submits_.size(),
                    vkHandleValue(read->cmd),
                    vkHandleValue(read->fence)));
            }
            const VkResult status = vkGetFenceStatus(device_, read->fence);
            if (status == VK_SUCCESS) {
                vkDestroyFence(device_, read->fence, nullptr);
                vkFreeCommandBuffers(device_, immediate_command_pool_, 1, &read->cmd);
            } else if (status == VK_NOT_READY) {
                if (write != read) {
                    *write = *read;
                }
                ++write;
            } else {
                return fail(std::format(
                    "vkGetFenceStatus failed while retiring an immediate submit (entry={}, pending_count={}, command_buffer={:#x}, fence={:#x}, result={}({}))",
                    std::distance(pending_immediate_submits_.begin(), read),
                    pending_immediate_submits_.size(),
                    vkHandleValue(read->cmd),
                    vkHandleValue(read->fence),
                    vkResultToString(status),
                    static_cast<int>(status)));
            }
        }
        pending_immediate_submits_.erase(write, pending_immediate_submits_.end());
        return true;
    }

    bool VulkanContext::transitionImageLayoutImmediate(const VkImage image,
                                                       const VkImageLayout old_layout,
                                                       const VkImageLayout new_layout,
                                                       const ImmediateTransitionOptions& options) {
        if (device_ == VK_NULL_HANDLE || immediate_command_pool_ == VK_NULL_HANDLE ||
            graphics_queue_ == VK_NULL_HANDLE || image == VK_NULL_HANDLE) {
            return fail(std::format(
                "Immediate image transition requires initialized graphics resources (device={:#x}, command_pool={:#x}, queue={:#x}, image={:#x}, old_layout={}({}), new_layout={}({}))",
                vkHandleValue(device_),
                vkHandleValue(immediate_command_pool_),
                vkHandleValue(graphics_queue_),
                vkHandleValue(image),
                vkImageLayoutToString(old_layout),
                static_cast<int>(old_layout),
                vkImageLayoutToString(new_layout),
                static_cast<int>(new_layout)));
        }
        if (frame_active_) {
            return fail(std::format(
                "Immediate image transition cannot run during an active frame (frame_active={}, frame_slot={}, image_index={}, image={:#x}, old_layout={}({}), new_layout={}({}))",
                frame_active_,
                active_frame_index_,
                active_image_index_,
                vkHandleValue(image),
                vkImageLayoutToString(old_layout),
                static_cast<int>(old_layout),
                vkImageLayoutToString(new_layout),
                static_cast<int>(new_layout)));
        }
        if (options.aspect_mask == 0 ||
            (options.wait &&
             (options.wait->semaphore == VK_NULL_HANDLE || options.wait->value == 0)) ||
            (options.signal &&
             (options.signal->semaphore == VK_NULL_HANDLE || options.signal->value == 0))) {
            return fail(std::format(
                "Immediate image transition requires a non-zero aspect and valid timeline points (image={:#x}, aspect_mask={:#x}, wait_semaphore={:#x}, wait_value={}, wait_stage={:#x}, signal_semaphore={:#x}, signal_value={})",
                vkHandleValue(image),
                static_cast<std::uint32_t>(options.aspect_mask),
                vkHandleValue(options.wait ? options.wait->semaphore : VK_NULL_HANDLE),
                options.wait ? options.wait->value : 0,
                static_cast<std::uint64_t>(options.wait_stage),
                vkHandleValue(options.signal ? options.signal->semaphore : VK_NULL_HANDLE),
                options.signal ? options.signal->value : 0));
        }
        if (old_layout == new_layout) {
            if (options.wait || options.signal) {
                return fail(std::format(
                    "A no-op image transition cannot carry timeline synchronization (image={:#x}, layout={}({}), wait_semaphore={:#x}, wait_value={}, signal_semaphore={:#x}, signal_value={})",
                    vkHandleValue(image),
                    vkImageLayoutToString(old_layout),
                    static_cast<int>(old_layout),
                    vkHandleValue(options.wait ? options.wait->semaphore : VK_NULL_HANDLE),
                    options.wait ? options.wait->value : 0,
                    vkHandleValue(options.signal ? options.signal->semaphore : VK_NULL_HANDLE),
                    options.signal ? options.signal->value : 0));
            }
            last_error_.clear();
            return true;
        }
        const std::uint64_t previous_wait_value =
            options.wait ? last_immediate_timeline_wait_values_[options.wait->semaphore] : 0;
        const std::uint64_t previous_signal_value =
            options.signal ? last_immediate_timeline_signal_values_[options.signal->semaphore] : 0;
        if (options.wait && options.wait->value <= previous_wait_value) {
            return fail(std::format(
                "Immediate-submit timeline waits must increase strictly (semaphore={:#x}, requested_value={}, previous_value={}, image={:#x}, old_layout={}({}), new_layout={}({}))",
                vkHandleValue(options.wait->semaphore),
                options.wait->value,
                previous_wait_value,
                vkHandleValue(image),
                vkImageLayoutToString(old_layout),
                static_cast<int>(old_layout),
                vkImageLayoutToString(new_layout),
                static_cast<int>(new_layout)));
        }
        if (options.signal && options.signal->value <= previous_signal_value) {
            return fail(std::format(
                "Immediate-submit timeline signals must increase strictly (semaphore={:#x}, requested_value={}, previous_value={}, image={:#x}, old_layout={}({}), new_layout={}({}))",
                vkHandleValue(options.signal->semaphore),
                options.signal->value,
                previous_signal_value,
                vkHandleValue(image),
                vkImageLayoutToString(old_layout),
                static_cast<int>(old_layout),
                vkImageLayoutToString(new_layout),
                static_cast<int>(new_layout)));
        }

        // Imported CUDA timeline operations are opaque to Vulkan validation.
        // Observe CUDA's signal before submitting its corresponding Vulkan wait
        // so the layer sees the external half of the ownership handoff. Keep the
        // queue wait below: unlike a host wait, it supplies the external-memory
        // acquire needed before Vulkan accesses CUDA-written image contents.
        if (validation_enabled_ && options.wait) {
            VkSemaphoreWaitInfo wait_info{};
            wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
            wait_info.semaphoreCount = 1;
            wait_info.pSemaphores = &options.wait->semaphore;
            wait_info.pValues = &options.wait->value;
            const VkResult wait_result = vkWaitSemaphores(device_, &wait_info, UINT64_MAX);
            if (wait_result != VK_SUCCESS) {
                return fail(std::format(
                    "vkWaitSemaphores failed while observing an external timeline for validation (semaphore={:#x}, value={}, image={:#x}, result={}({}))",
                    vkHandleValue(options.wait->semaphore),
                    options.wait->value,
                    vkHandleValue(image),
                    vkResultToString(wait_result),
                    static_cast<int>(wait_result)));
            }
        }
        // Reap any prior fire-and-forget submits that have completed. Bound
        // the backlog under a stalled GPU so command buffers and fences cannot
        // grow without limit; this path only blocks once 64 submits are live.
        if (!drainCompletedImmediateSubmits()) {
            return false;
        }
        constexpr std::size_t kMaxPendingImmediateSubmits = 64;
        if (pending_immediate_submits_.size() >= kMaxPendingImmediateSubmits &&
            !waitForImmediateSubmits()) {
            return false;
        }

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
        setDebugObjectNamef(VK_OBJECT_TYPE_COMMAND_BUFFER,
                            command_buffer,
                            "immediate.transition[{}].command",
                            pending_immediate_submits_.size());

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = vkBeginCommandBuffer(command_buffer, &begin_info);
        if (result != VK_SUCCESS) {
            vkFreeCommandBuffers(device_, immediate_command_pool_, 1, &command_buffer);
            return fail(std::format("vkBeginCommandBuffer(layout transition) failed: {}", vkResultToString(result)));
        }

        const auto source = VulkanImageBarrierTracker::layoutAccess(
            old_layout, VulkanImageBarrierTracker::AccessDirection::Source);
        const auto destination = VulkanImageBarrierTracker::layoutAccess(
            new_layout, VulkanImageBarrierTracker::AccessDirection::Destination);
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = source.stage;
        barrier.srcAccessMask = source.access;
        barrier.dstStageMask = destination.stage;
        barrier.dstAccessMask = destination.access;
        barrier.oldLayout = old_layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = options.aspect_mask;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

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
        VkPipelineStageFlags resolved_wait_stage = options.wait_stage == 0
                                                       ? static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
                                                       : options.wait_stage;
        // The submit-time wait gates the GPU on the external (CUDA) timeline.
        // The validation-only host observation above is solely for a layer that
        // cannot otherwise see CUDA's signal; normal rendering remains async.
        if (options.wait) {
            timeline_submit_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timeline_submit_info.waitSemaphoreValueCount = 1;
            timeline_submit_info.pWaitSemaphoreValues = &options.wait->value;
            submit_info.pNext = &timeline_submit_info;
            submit_info.waitSemaphoreCount = 1;
            submit_info.pWaitSemaphores = &options.wait->semaphore;
            submit_info.pWaitDstStageMask = &resolved_wait_stage;
        }
        if (options.signal) {
            timeline_submit_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timeline_submit_info.signalSemaphoreValueCount = 1;
            timeline_submit_info.pSignalSemaphoreValues = &options.signal->value;
            submit_info.pNext = &timeline_submit_info;
            submit_info.signalSemaphoreCount = 1;
            submit_info.pSignalSemaphores = &options.signal->semaphore;
        }
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence submit_fence = VK_NULL_HANDLE;
        result = vkCreateFence(device_, &fence_info, nullptr, &submit_fence);
        if (result != VK_SUCCESS) {
            vkFreeCommandBuffers(device_, immediate_command_pool_, 1, &command_buffer);
            return fail(std::format("vkCreateFence(layout transition) failed: {}", vkResultToString(result)));
        }
        setDebugObjectNamef(VK_OBJECT_TYPE_FENCE,
                            submit_fence,
                            "immediate.transition[{}].fence",
                            pending_immediate_submits_.size());

        result = vkQueueSubmit(graphics_queue_, 1, &submit_info, submit_fence);
        if (result != VK_SUCCESS) {
            vkDestroyFence(device_, submit_fence, nullptr);
            vkFreeCommandBuffers(device_, immediate_command_pool_, 1, &command_buffer);
            return fail(std::format("Immediate Vulkan image layout transition submit failed: {}", vkResultToString(result)));
        }
        if (options.wait) {
            last_immediate_timeline_wait_values_[options.wait->semaphore] = options.wait->value;
        }
        if (options.signal) {
            last_immediate_timeline_signal_values_[options.signal->semaphore] = options.signal->value;
        }
        // Fire-and-forget: queue cmd+fence for lazy reaping. Vulkan queues are
        // FIFO per VkQueue, so subsequent submits on graphics_queue_ correctly
        // observe the layout transition without any CPU-side wait.
        pending_immediate_submits_.push_back({command_buffer, submit_fence});
        if (validation_enabled_ && options.signal) {
            // Finish Vulkan's half before CUDA is allowed to advance this same
            // imported timeline. Without this observation, validation may see
            // CUDA's later value while the preceding Vulkan signal is still
            // pending and diagnose a false non-monotonic signal. The production
            // path remains fire-and-forget.
            VkSemaphoreWaitInfo wait_info{};
            wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
            wait_info.semaphoreCount = 1;
            wait_info.pSemaphores = &options.signal->semaphore;
            wait_info.pValues = &options.signal->value;
            const VkResult wait_result = vkWaitSemaphores(device_, &wait_info, UINT64_MAX);
            if (wait_result != VK_SUCCESS) {
                return fail(std::format(
                    "vkWaitSemaphores failed while publishing a Vulkan timeline signal for validation (semaphore={:#x}, value={}, image={:#x}, result={}({}))",
                    vkHandleValue(options.signal->semaphore),
                    options.signal->value,
                    vkHandleValue(image),
                    vkResultToString(wait_result),
                    static_cast<int>(wait_result)));
            }
        }
        last_error_.clear();
        return true;
    }

    bool VulkanContext::createSwapchain(const int framebuffer_width, const int framebuffer_height,
                                        VkSwapchainKHR old_swapchain) {
        const SwapchainSupport support = querySwapchainSupport(physical_device_);
        if (support.formats.empty() || support.present_modes.empty()) {
            return fail("Vulkan swapchain support is incomplete");
        }

        const VkSurfaceFormatKHR surface_format = chooseSurfaceFormat(support.formats);
        const VkPresentModeKHR present_mode = choosePresentMode(support.present_modes);
        const bool extent_fixed_to_surface =
            support.capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max();
        const auto present_scaling_capabilities = queryPresentScalingCapabilities(present_mode);
        const bool use_present_scaling =
            old_swapchain != VK_NULL_HANDLE && present_scaling_capabilities.has_value();
        const bool add_resize_headroom =
            old_swapchain != VK_NULL_HANDLE && framebuffer_resize_allow_headroom_;
        const VkExtent2D extent = chooseSwapchainExtent(support.capabilities,
                                                        framebuffer_width,
                                                        framebuffer_height,
                                                        add_resize_headroom,
                                                        use_present_scaling ? &*present_scaling_capabilities : nullptr);
        if (extent.width == 0 || extent.height == 0) {
            // Surface reports zero extent (window minimized); skip creation and retry next frame.
            last_error_.clear();
            return false;
        }

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
        create_info.oldSwapchain = old_swapchain;
        VkSwapchainPresentScalingCreateInfoEXT present_scaling_info{};
        if (use_present_scaling) {
            present_scaling_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_EXT;
            present_scaling_info.scalingBehavior = VK_PRESENT_SCALING_ONE_TO_ONE_BIT_EXT;
            present_scaling_info.presentGravityX = VK_PRESENT_GRAVITY_MIN_BIT_EXT;
            present_scaling_info.presentGravityY = VK_PRESENT_GRAVITY_MIN_BIT_EXT;
            create_info.pNext = &present_scaling_info;
        }

        VkResult result;
        {
            // First real GPU use on NVIDIA triggers lazy driver init here — measure it.
            LOG_TIMER("vulkan_init.swapchain.vkCreateSwapchainKHR");
            result = vkCreateSwapchainKHR(device_, &create_info, nullptr, &swapchain_);
        }
        if (result != VK_SUCCESS) {
            return fail(std::format("vkCreateSwapchainKHR failed: {}", vkResultToString(result)));
        }
        setDebugObjectName(VK_OBJECT_TYPE_SWAPCHAIN_KHR, swapchain_, "swapchain.main");

        LFS_VK_CONTEXT_CHECK_MSG(
            vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr),
            "swapchain image-count query returned no usable array (device={:#x}, swapchain={:#x}, requested_min_images={}, observed_image_count={})",
            vkHandleValue(device_),
            vkHandleValue(swapchain_),
            create_info.minImageCount,
            image_count);
        if (image_count == 0) {
            return fail(std::format(
                "vkGetSwapchainImagesKHR returned zero images (device={:#x}, swapchain={:#x}, requested_min_images={}, extent={}x{})",
                vkHandleValue(device_),
                vkHandleValue(swapchain_),
                create_info.minImageCount,
                extent.width,
                extent.height));
        }
        swapchain_images_.resize(image_count);
        LFS_VK_CONTEXT_CHECK_MSG(
            vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, swapchain_images_.data()),
            "swapchain image query failed (device={:#x}, swapchain={:#x}, destination_capacity={}, observed_image_count={})",
            vkHandleValue(device_),
            vkHandleValue(swapchain_),
            swapchain_images_.size(),
            image_count);
        swapchain_images_.resize(image_count);
        if (std::ranges::any_of(swapchain_images_, [](const VkImage image) {
                return image == VK_NULL_HANDLE;
            })) {
            return fail(std::format(
                "vkGetSwapchainImagesKHR returned a null image handle (swapchain={:#x}, image_count={}, extent={}x{}, format={})",
                vkHandleValue(swapchain_),
                swapchain_images_.size(),
                extent.width,
                extent.height,
                vkFormatToString(surface_format.format)));
        }
        const std::size_t bytes_per_pixel = estimateFormatBytesPerPixel(surface_format.format);
        const std::size_t pixel_count = static_cast<std::size_t>(extent.width) * extent.height;
        if (bytes_per_pixel > 0 &&
            (pixel_count > std::numeric_limits<std::size_t>::max() / image_count ||
             pixel_count * image_count > std::numeric_limits<std::size_t>::max() / bytes_per_pixel)) {
            return fail(std::format(
                "Swapchain byte estimate overflowed size_t (extent={}x{}, image_count={}, bytes_per_pixel={}, size_t_max={})",
                extent.width,
                extent.height,
                image_count,
                bytes_per_pixel,
                std::numeric_limits<std::size_t>::max()));
        }
        swapchain_estimated_bytes_ = bytes_per_pixel > 0
                                         ? pixel_count * image_count * bytes_per_pixel
                                         : 0;
        recordCurrentVulkanBytes("vulkan.swapchain", "driver_owned_images_estimate", swapchain_estimated_bytes_);
        swapchain_images_in_flight_.assign(image_count, VK_NULL_HANDLE);
        swapchain_format_ = surface_format.format;
        swapchain_color_space_ = surface_format.colorSpace;
        swapchain_extent_fixed_to_surface_ = extent_fixed_to_surface;
        swapchain_present_scaling_enabled_ = use_present_scaling;
        has_hdr_ = surface_format.colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        if (old_swapchain == VK_NULL_HANDLE) {
            LOG_INFO("Vulkan swapchain: {} images, extent {}x{}, format {}, color space {}{}{}",
                     image_count,
                     extent.width,
                     extent.height,
                     vkFormatToString(surface_format.format),
                     vkColorSpaceToString(surface_format.colorSpace),
                     has_hdr_ ? " (HDR-capable)" : "",
                     use_present_scaling ? " (one-to-one present scaling)" : "");
        } else {
            LOG_DEBUG("Vulkan swapchain: {} images, extent {}x{}, format {}, color space {}{}{}",
                      image_count,
                      extent.width,
                      extent.height,
                      vkFormatToString(surface_format.format),
                      vkColorSpaceToString(surface_format.colorSpace),
                      has_hdr_ ? " (HDR-capable)" : "",
                      use_present_scaling ? " (one-to-one present scaling)" : "");
        }

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
            setDebugObjectNamef(VK_OBJECT_TYPE_SEMAPHORE,
                                image_available_[i],
                                "swapchain.acquire[{}]",
                                i);
        }
        // vkQueuePresentKHR consumes the render-finished wait per swapchain image, not per frame
        // slot. Pairing this semaphore with the acquired image prevents binary re-signal while a
        // compositor still owns the prior wait on a >2-image swapchain.
        render_finished_.assign(image_count, VK_NULL_HANDLE);
        for (std::uint32_t i = 0; i < image_count; ++i) {
            const VkResult sem_result =
                vkCreateSemaphore(device_, &image_avail_info, nullptr, &render_finished_[i]);
            if (sem_result != VK_SUCCESS) {
                return fail(std::format("vkCreateSemaphore(render_finished image {}) failed: {}",
                                        i,
                                        vkResultToString(sem_result)));
            }
            setDebugObjectNamef(VK_OBJECT_TYPE_SEMAPHORE,
                                render_finished_[i],
                                "swapchain.present[{}]",
                                i);
        }
        next_acquire_index_ = 0;
        active_acquire_index_ = 0;
        swapchain_extent_ = extent;
        swapchain_image_usage_ = create_info.imageUsage;
        for (size_t i = 0; i < swapchain_images_.size(); ++i) {
            image_barriers_.registerImage(swapchain_images_[i],
                                          VK_IMAGE_ASPECT_COLOR_BIT,
                                          VK_IMAGE_LAYOUT_UNDEFINED);
            setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE,
                                swapchain_images_[i],
                                "swapchain.image[{}]",
                                i);
        }
        framebuffer_resize_last_recreate_ = std::chrono::steady_clock::now();
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
            setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE_VIEW,
                                swapchain_image_views_[i],
                                "swapchain.image[{}].view",
                                i);
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

        depth_stencil_resources_.assign(kFramesInFlight, {});
        const auto destroy_created = [&]() {
            for (std::size_t i = 0; i < depth_stencil_resources_.size(); ++i) {
                DepthStencilResource& resource = depth_stencil_resources_[i];
                recordCurrentVulkanBytes("vulkan.swapchain.depth_stencil",
                                         std::format("frame#{}", i),
                                         0);
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
            VmaAllocationInfo created_allocation_info{};
            VkResult result = vmaCreateImage(allocator_,
                                             &image_info,
                                             &allocation_info,
                                             &resource.image,
                                             &resource.allocation,
                                             &created_allocation_info);
            if (result != VK_SUCCESS) {
                destroy_created();
                return fail(std::format("vmaCreateImage(depth/stencil frame {}) failed: {}", i, vkResultToString(result)));
            }
            setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE,
                                resource.image,
                                "swapchain.depth[{}]",
                                i);
            const std::string allocation_name = std::format("Depth/stencil allocation frame {}", i);
            vmaSetAllocationName(allocator_, resource.allocation, allocation_name.c_str());
            recordCurrentVulkanBytes("vulkan.swapchain.depth_stencil",
                                     std::format("frame#{}", i),
                                     static_cast<std::size_t>(created_allocation_info.size));

            view_info.image = resource.image;
            result = vkCreateImageView(device_, &view_info, nullptr, &resource.view);
            if (result != VK_SUCCESS) {
                destroy_created();
                return fail(std::format("vkCreateImageView(depth/stencil frame {}) failed: {}", i, vkResultToString(result)));
            }
            setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE_VIEW,
                                resource.view,
                                "swapchain.depth[{}].view",
                                i);

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
            setDebugObjectNamef(VK_OBJECT_TYPE_COMMAND_POOL,
                                command_pools_[i],
                                "frame[{}].graphics.pool",
                                i);
        }

        VkCommandPoolCreateInfo immediate_info{};
        immediate_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        immediate_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        immediate_info.queueFamilyIndex = graphics_queue_family_;
        const VkResult result = vkCreateCommandPool(device_, &immediate_info, nullptr, &immediate_command_pool_);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkCreateCommandPool(immediate) failed: {}", vkResultToString(result)));
        }
        setDebugObjectName(VK_OBJECT_TYPE_COMMAND_POOL,
                           immediate_command_pool_,
                           "immediate.graphics.pool");
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
            setDebugObjectNamef(VK_OBJECT_TYPE_COMMAND_BUFFER,
                                command_buffers_[i],
                                "frame[{}].graphics.command",
                                i);
        }
        return true;
    }

    bool VulkanContext::createSyncObjects() {
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (std::size_t i = 0; i < kFramesInFlight; ++i) {
            const VkResult result = vkCreateFence(device_, &fence_info, nullptr, &in_flight_[i]);
            if (result != VK_SUCCESS) {
                return fail(std::format("vkCreateFence(frame {}) failed: {}", i, vkResultToString(result)));
            }
            setDebugObjectNamef(VK_OBJECT_TYPE_FENCE,
                                in_flight_[i],
                                "frame[{}].in_flight.fence",
                                i);
        }
        return true;
    }

    bool VulkanContext::replaceFrameFenceSignaled(const std::size_t frame_slot) {
        if (frame_slot >= in_flight_.size() || device_ == VK_NULL_HANDLE) {
            LOG_ERROR("Cannot replace frame fence without a live device and in-range slot (device={:#x}, frame_slot={}, fence_count={}) ({}:{})",
                      vkHandleValue(device_),
                      frame_slot,
                      in_flight_.size(),
                      __FILE__,
                      __LINE__);
            return false;
        }

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VkFence replacement = VK_NULL_HANDLE;
        const VkResult result = vkCreateFence(device_, &fence_info, nullptr, &replacement);
        if (result != VK_SUCCESS) {
            LOG_ERROR("Vulkan: {}",
                      formatVkCheckFailure(
                          "vkCreateFence(device_, &fence_info, nullptr, &replacement)",
                          result,
                          std::format("Frame-fence recovery failed after submission failure (device={:#x}, frame_slot={}, old_fence={:#x}, flags={:#x})",
                                      vkHandleValue(device_),
                                      frame_slot,
                                      vkHandleValue(in_flight_[frame_slot]),
                                      static_cast<std::uint32_t>(fence_info.flags)),
                          __FILE__,
                          __LINE__));
            return false;
        }

        const VkFence retired = in_flight_[frame_slot];
        in_flight_[frame_slot] = replacement;
        frame_submit_serials_[frame_slot] = 0;
        for (VkFence& image_fence : swapchain_images_in_flight_) {
            if (image_fence == retired) {
                image_fence = VK_NULL_HANDLE;
            }
        }
        if (retired != VK_NULL_HANDLE) {
            vkDestroyFence(device_, retired, nullptr);
        }
        setDebugObjectNamef(VK_OBJECT_TYPE_FENCE,
                            replacement,
                            "frame[{}].in_flight.fence",
                            frame_slot);
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
        populateDebugMessengerCreateInfo(create_info, &validation_errors_fatal_);
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
        const std::string owned_name{name};
        const VkResult result = debug_name_writer_.set(object_type, object_handle, owned_name);
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
            if (const char* reason = pipelineCacheRejectReason(cache_data, device_props)) {
                LOG_WARN("Discarding on-disk Vulkan pipeline cache ({}): {} — pipelines will be recompiled",
                         lfs::core::path_to_utf8(path), reason);
                cache_data.clear();
            }
        }

        VkPipelineCacheCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        create_info.initialDataSize = cache_data.size();
        create_info.pInitialData = cache_data.empty() ? nullptr : cache_data.data();

        VkResult result = vkCreatePipelineCache(device_, &create_info, nullptr, &pipeline_cache_);
        if (result != VK_SUCCESS && !cache_data.empty()) {
            LOG_WARN("vkCreatePipelineCache rejected on-disk data ({}); retrying empty — pipelines will be recompiled",
                     vkResultToString(result));
            cache_data.clear();
            create_info.initialDataSize = 0;
            create_info.pInitialData = nullptr;
            result = vkCreatePipelineCache(device_, &create_info, nullptr, &pipeline_cache_);
        }
        if (result != VK_SUCCESS) {
            return fail(std::format("vkCreatePipelineCache failed: {}", vkResultToString(result)));
        }

        setDebugObjectName(VK_OBJECT_TYPE_PIPELINE_CACHE,
                           pipeline_cache_,
                           "lichtfeld.pipeline_cache");
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
            const auto resource_index = static_cast<std::size_t>(&resource - depth_stencil_resources_.data());
            recordCurrentVulkanBytes("vulkan.swapchain.depth_stencil",
                                     std::format("frame#{}", resource_index),
                                     0);
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
        swapchain_present_scaling_enabled_ = false;
        for (VkSemaphore& semaphore : image_available_) {
            if (semaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(device_, semaphore, nullptr);
                semaphore = VK_NULL_HANDLE;
            }
        }
        image_available_.clear();
        for (VkSemaphore& semaphore : render_finished_) {
            if (semaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(device_, semaphore, nullptr);
                semaphore = VK_NULL_HANDLE;
            }
        }
        render_finished_.clear();
        next_acquire_index_ = 0;
        active_acquire_index_ = 0;
        image_barriers_.clearSwapchainOnly();

        if (swapchain_ != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device_, swapchain_, nullptr);
            swapchain_ = VK_NULL_HANDLE;
        }
        swapchain_image_usage_ = 0;
        if (swapchain_estimated_bytes_ > 0) {
            recordCurrentVulkanBytes("vulkan.swapchain", "driver_owned_images_estimate", 0);
            swapchain_estimated_bytes_ = 0;
        }
    }

    bool VulkanContext::waitForFrameFences() {
        // Bound the wait so a wedged GPU surfaces as a swapchain-recreate failure rather
        // than a hang. 2 s is generous; healthy frames complete in <16 ms.
        constexpr std::uint64_t kSwapchainWaitTimeoutNs = 2'000'000'000ull;
        std::vector<VkFence> fences;
        fences.reserve(kFramesInFlight + swapchain_images_in_flight_.size());
        std::size_t frame_fence_count = 0;
        std::size_t image_alias_count = 0;
        for (const VkFence fence : in_flight_) {
            if (fence != VK_NULL_HANDLE) {
                fences.push_back(fence);
                ++frame_fence_count;
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
                ++image_alias_count;
            }
        }
        if (fences.empty()) {
            LOG_DEBUG("Vulkan waitForSubmittedFrames: no fences to wait (frame_index={}, last_submit_id={}, framebuffer={}x{}, extent={}x{})",
                      frame_index_,
                      frame_submit_serials_[frame_index_],
                      framebuffer_width_,
                      framebuffer_height_,
                      swapchain_extent_.width,
                      swapchain_extent_.height);
            return true;
        }
        const std::uint64_t last_frame_submit_id =
            frame_index_ < frame_submit_serials_.size() ? frame_submit_serials_[frame_index_] : 0;
        LOG_DEBUG("Vulkan waitForSubmittedFrames begin: fences={}, frame_fences={}, image_aliases={}, frame_index={}, last_submit_id={}, framebuffer={}x{}, extent={}x{}, resized={}",
                  fences.size(),
                  frame_fence_count,
                  image_alias_count,
                  frame_index_,
                  last_frame_submit_id,
                  framebuffer_width_,
                  framebuffer_height_,
                  swapchain_extent_.width,
                  swapchain_extent_.height,
                  framebuffer_resized_);
        const auto wait_start = std::chrono::steady_clock::now();
        const VkResult result = vkWaitForFences(device_,
                                                static_cast<std::uint32_t>(fences.size()),
                                                fences.data(),
                                                VK_TRUE,
                                                kSwapchainWaitTimeoutNs);
        if (result != VK_SUCCESS) {
            return fail(std::format("vkWaitForFences(submitted frames) failed after {:.1f} ms: {} (fences={}, frame_fences={}, image_aliases={}, frame_index={}, last_submit_id={}, framebuffer={}x{}, extent={}x{}, resized={})",
                                    elapsedMs(wait_start),
                                    vkResultToString(result),
                                    fences.size(),
                                    frame_fence_count,
                                    image_alias_count,
                                    frame_index_,
                                    last_frame_submit_id,
                                    framebuffer_width_,
                                    framebuffer_height_,
                                    swapchain_extent_.width,
                                    swapchain_extent_.height,
                                    framebuffer_resized_));
        }
        LOG_DEBUG("Vulkan waitForSubmittedFrames complete: fences={}, elapsed_ms={:.1f}",
                  fences.size(),
                  elapsedMs(wait_start));
        return true;
    }

    bool VulkanContext::recreateSwapchain() {
        if (framebuffer_width_ <= 0 || framebuffer_height_ <= 0) {
            return true;
        }

        const auto timed = [](const char* name, auto&& fn) {
            LOG_TIMER_THRESHOLD(name, 0.25);
            return fn();
        };

        LOG_DEBUG("Vulkan recreateSwapchain begin: framebuffer={}x{}, old_extent={}x{}, old_images={}, framebuffer_resized={}, frame_index={}",
                  framebuffer_width_,
                  framebuffer_height_,
                  swapchain_extent_.width,
                  swapchain_extent_.height,
                  swapchain_images_.size(),
                  framebuffer_resized_,
                  frame_index_);

        // Wait on all in-flight render fences (per-frame + per-image aliases). Once those
        // are signaled, prior vkQueueSubmit work is complete and the presentation engine
        // can no longer be reading the swapchain images we are about to destroy. We avoid
        // vkQueueWaitIdle(present_queue_) because it stalls the entire CPU thread on the
        // compositor, which can deadlock if the display server is itself blocked.
        if (!waitForFrameFences()) {
            framebuffer_resized_ = true;
            return false;
        }
        // waitForFrameFences only drains the graphics queue's per-frame fences. The
        // viewport's shared-scratch rasterizer submits on the async-compute queue,
        // whose work is not covered by those fences; tearing down swapchain-tied
        // resources while that work is still in flight loses the device (Xid 109).
        // vkDeviceWaitIdle drains all queue work (it does not touch presentation, so
        // it cannot deadlock on the compositor the way vkQueueWaitIdle(present) can).
        // A bounded wait is not possible here: vkDeviceWaitIdle takes no timeout, and the
        // async-compute submits are timeline-semaphore gated with no VkFence this context
        // tracks. The timer below surfaces a pathological stall instead of hiding it.
        {
            LOG_TIMER_THRESHOLD("frame_pacing.vulkan_recreateSwapchain.device_wait_idle", 1.0);
            const auto wait_start = std::chrono::steady_clock::now();
            const VkResult idle_result = vkDeviceWaitIdle(device_);
            LOG_DEBUG("Vulkan recreateSwapchain deviceWaitIdle result={} elapsed_ms={:.1f}",
                      vkResultToString(idle_result),
                      elapsedMs(wait_start));
            if (idle_result != VK_SUCCESS) {
                framebuffer_resized_ = true;
                return fail(std::format("vkDeviceWaitIdle(swapchain recreate) failed: {}",
                                        vkResultToString(idle_result)));
            }
        }
        // Keep the old swapchain alive across the teardown of its dependent resources so it
        // can be handed to vkCreateSwapchainKHR as oldSwapchain — letting the driver reuse
        // images and present-engine state instead of allocating a fresh swapchain cold.
        // destroySwapchain() skips the handle while swapchain_ is null; we destroy it after
        // the new one is created (the spec requires the old handle stay valid until then).
        const VkSwapchainKHR old_swapchain = swapchain_;
        swapchain_ = VK_NULL_HANDLE;
        {
            LOG_TIMER_THRESHOLD("frame_pacing.vulkan_recreateSwapchain.destroy_dependent_resources", 0.25);
            destroySwapchain();
        }
        const bool swapchain_created =
            timed("frame_pacing.vulkan_recreateSwapchain.createSwapchain",
                  [&] { return createSwapchain(framebuffer_width_, framebuffer_height_, old_swapchain); });
        const bool created =
            swapchain_created &&
            timed("frame_pacing.vulkan_recreateSwapchain.createImageViews",
                  [&] { return createImageViews(); }) &&
            timed("frame_pacing.vulkan_recreateSwapchain.createDepthStencilResources",
                  [&] { return createDepthStencilResources(); });
        if (swapchain_created && old_swapchain != VK_NULL_HANDLE) {
            LOG_TIMER_THRESHOLD("frame_pacing.vulkan_recreateSwapchain.destroy_old_swapchain", 0.25);
            vkDestroySwapchainKHR(device_, old_swapchain, nullptr);
        }
        if (!created) {
            const std::string error = last_error_;
            if (swapchain_created) {
                destroySwapchain();
            } else {
                swapchain_ = old_swapchain;
            }
            framebuffer_resized_ = true;
            last_error_ = error;
            return false;
        }
        framebuffer_resized_ = false;
        LOG_DEBUG("Vulkan recreateSwapchain complete: extent={}x{}, images={}, framebuffer={}x{}",
                  swapchain_extent_.width,
                  swapchain_extent_.height,
                  swapchain_images_.size(),
                  framebuffer_width_,
                  framebuffer_height_);
        return true;
    }

} // namespace lfs::vis
