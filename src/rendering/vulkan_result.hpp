/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cstdio>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vulkan/vulkan.h>

namespace lfs::rendering {

    template <typename... Args>
    [[nodiscard]] inline std::string formatVulkanDiagnostic(
        std::format_string<Args...> format,
        Args&&... args) {
        return std::format(format, std::forward<Args>(args)...);
    }

    [[noreturn]] inline void throwVulkanError(std::string message,
                                              const char* file,
                                              const int line) {
        message += ". From file `";
        message += file;
        message += "`, line ";
        message += std::to_string(line);
        std::printf("\033[91m%s\033[m\n", message.c_str());
        std::fflush(stdout);
        throw std::runtime_error(message);
    }

    [[nodiscard]] inline const char* vkResultToString(const VkResult result) noexcept {
        switch (result) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_FRAGMENTATION: return "VK_ERROR_FRAGMENTATION";
        case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
        case VK_PIPELINE_COMPILE_REQUIRED: return "VK_PIPELINE_COMPILE_REQUIRED";
        case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
        case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR: return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
        case VK_ERROR_VALIDATION_FAILED_EXT: return "VK_ERROR_VALIDATION_FAILED_EXT";
        case VK_ERROR_INVALID_SHADER_NV: return "VK_ERROR_INVALID_SHADER_NV";
        case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT: return "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT";
        case VK_ERROR_NOT_PERMITTED_KHR: return "VK_ERROR_NOT_PERMITTED_KHR";
        case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT: return "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
        case VK_THREAD_IDLE_KHR: return "VK_THREAD_IDLE_KHR";
        case VK_THREAD_DONE_KHR: return "VK_THREAD_DONE_KHR";
        case VK_OPERATION_DEFERRED_KHR: return "VK_OPERATION_DEFERRED_KHR";
        case VK_OPERATION_NOT_DEFERRED_KHR: return "VK_OPERATION_NOT_DEFERRED_KHR";
        default: return "VK_RESULT_UNKNOWN";
        }
    }

    template <typename VkHandle>
    [[nodiscard]] inline std::uint64_t vkHandleValue(const VkHandle handle) noexcept {
        if constexpr (std::is_pointer_v<VkHandle>) {
            return reinterpret_cast<std::uint64_t>(handle);
        } else {
            return static_cast<std::uint64_t>(handle);
        }
    }

    class VulkanDebugNameWriter {
    public:
        void initialize(const VkDevice device) noexcept {
            device_ = device;
            set_debug_name_ = device == VK_NULL_HANDLE
                                  ? nullptr
                                  : reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
                                        vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
        }

        void reset() noexcept { initialize(VK_NULL_HANDLE); }

        [[nodiscard]] bool enabled() const noexcept { return set_debug_name_ != nullptr; }

        VkResult set(const VkObjectType object_type,
                     const std::uint64_t object_handle,
                     const std::string_view object_name) const noexcept {
            if (!enabled() || object_handle == 0 || object_name.empty()) {
                return VK_SUCCESS;
            }
            VkDebugUtilsObjectNameInfoEXT name_info{};
            name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
            name_info.objectType = object_type;
            name_info.objectHandle = object_handle;
            name_info.pObjectName = object_name.data();
            return set_debug_name_(device_, &name_info);
        }

    private:
        VkDevice device_ = VK_NULL_HANDLE;
        PFN_vkSetDebugUtilsObjectNameEXT set_debug_name_ = nullptr;
    };

} // namespace lfs::rendering
