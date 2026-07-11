/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"

#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    inline const char* vkResultToString(const VkResult result) noexcept {
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

    [[nodiscard]] inline const char* vkImageLayoutToString(const VkImageLayout layout) noexcept {
        switch (layout) {
        case VK_IMAGE_LAYOUT_UNDEFINED: return "VK_IMAGE_LAYOUT_UNDEFINED";
        case VK_IMAGE_LAYOUT_GENERAL: return "VK_IMAGE_LAYOUT_GENERAL";
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: return "VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL";
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: return "VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL";
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL: return "VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL";
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: return "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL";
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: return "VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL";
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: return "VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL";
        case VK_IMAGE_LAYOUT_PREINITIALIZED: return "VK_IMAGE_LAYOUT_PREINITIALIZED";
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: return "VK_IMAGE_LAYOUT_PRESENT_SRC_KHR";
        case VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL: return "VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL";
        case VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL: return "VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL";
        default: return "VK_IMAGE_LAYOUT_UNKNOWN";
        }
    }

    [[nodiscard]] inline std::string formatVkCheckFailure(
        const std::string_view expression,
        const VkResult result,
        const std::string_view context,
        const std::string_view file,
        const int line) {
        if (context.empty()) {
            return std::format("{} failed: {} ({}) ({}:{})",
                               expression,
                               vkResultToString(result),
                               static_cast<int>(result),
                               file,
                               line);
        }
        return std::format("{} failed: {} ({}) — {} ({}:{})",
                           expression,
                           vkResultToString(result),
                           static_cast<int>(result),
                           context,
                           file,
                           line);
    }

    // Namespace-level fallback for visualizer helpers that use bool-return
    // failure paths but do not own VulkanContext::lastError(). VulkanContext
    // supplies a member with the same name so unqualified macro lookup routes
    // failures into its persistent error state instead.
    [[nodiscard]] inline bool vkCheckFailed(std::string message) {
        LOG_ERROR("Vulkan: {}", message);
        return false;
    }

    [[noreturn]] inline void vkDebugAssertFailed(std::string message) {
        LOG_ERROR("Vulkan debug invariant: {}", message);
        throw std::logic_error(std::move(message));
    }

} // namespace lfs::vis

#define LFS_VK_CHECK(expr)                                             \
    do {                                                               \
        const VkResult lfs_vk_check_result_ = (expr);                  \
        if (lfs_vk_check_result_ != VK_SUCCESS) {                      \
            return vkCheckFailed(::lfs::vis::formatVkCheckFailure(     \
                #expr, lfs_vk_check_result_, {}, __FILE__, __LINE__)); \
        }                                                              \
    } while (false)

#define LFS_VK_CHECK_MSG(expr, ...)                                                          \
    do {                                                                                     \
        const VkResult lfs_vk_check_result_ = (expr);                                        \
        if (lfs_vk_check_result_ != VK_SUCCESS) {                                            \
            return vkCheckFailed(::lfs::vis::formatVkCheckFailure(                           \
                #expr, lfs_vk_check_result_, std::format(__VA_ARGS__), __FILE__, __LINE__)); \
        }                                                                                    \
    } while (false)

#ifndef LFS_VK_DEBUG_ASSERT
#ifndef NDEBUG
#define LFS_VK_DEBUG_ASSERT(condition, ...)                                       \
    do {                                                                          \
        if (!(condition)) {                                                       \
            ::lfs::vis::vkDebugAssertFailed(std::format("{} ({}:{})",             \
                                                        std::format(__VA_ARGS__), \
                                                        __FILE__,                 \
                                                        __LINE__));               \
        }                                                                         \
    } while (false)
#else
#define LFS_VK_DEBUG_ASSERT(condition, ...) ((void)0)
#endif
#endif
