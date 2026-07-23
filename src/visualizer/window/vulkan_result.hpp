/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/assert.hpp"
#include "core/logger.hpp"
#include "rendering/vulkan_result.hpp"

#include <format>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    using rendering::vkHandleValue;
    using rendering::vkResultToString;

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

    [[nodiscard]] inline std::string formatVkCheckFailure(
        const std::string_view expression,
        const VkResult result,
        const std::string_view context,
        const std::source_location location = std::source_location::current()) {
        return formatVkCheckFailure(
            expression, result, context, location.file_name(), static_cast<int>(location.line()));
    }

    [[nodiscard]] inline bool logVkFailure(std::string message) {
        LOG_ERROR("Vulkan: {}", message);
        return false;
    }

    [[nodiscard]] inline bool reportVkFailure(
        const std::string_view expression,
        const VkResult result,
        const std::string_view context,
        const std::source_location location = std::source_location::current()) {
        return logVkFailure(formatVkCheckFailure(expression, result, context, location));
    }

    [[nodiscard]] inline VkShaderModule createShaderModule(
        const VkDevice device,
        const std::span<const std::uint32_t> spirv,
        const std::string_view label,
        const std::source_location location = std::source_location::current()) {
        VkShaderModuleCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = spirv.size_bytes();
        create_info.pCode = spirv.data();

        VkShaderModule module = VK_NULL_HANDLE;
        const VkResult result = vkCreateShaderModule(device, &create_info, nullptr, &module);
        if (result != VK_SUCCESS) {
            LOG_ERROR("Vulkan: {}",
                      formatVkCheckFailure(
                          "vkCreateShaderModule(device, &create_info, nullptr, &module)",
                          result,
                          std::format("{} shader-module creation failed (device={:#x}, code_ptr={:#x}, code_size={})",
                                      label,
                                      vkHandleValue(device),
                                      reinterpret_cast<std::uintptr_t>(spirv.data()),
                                      spirv.size_bytes()),
                          location.file_name(),
                          static_cast<int>(location.line())));
            return VK_NULL_HANDLE;
        }
        return module;
    }

    // Phase 7C-P5 / AMB-C4: typed bool-dialect VkResult check.
    // Success → true. Failure → reportVkFailure (logs) → false.
    // Call-site source_location is captured at the call site.
    [[nodiscard]] inline bool vk_try_bool(
        const VkResult result,
        const std::string_view expression,
        std::string context,
        const std::source_location location = std::source_location::current()) {
        if (result == VK_SUCCESS) {
            return true;
        }
        return reportVkFailure(expression, result, std::move(context), location);
    }

} // namespace lfs::vis

#define LFS_VK_CONTEXT_CHECK_MSG(expr, ...)                              \
    do {                                                                 \
        const VkResult lfs_vk_check_result_ = (expr);                    \
        if (lfs_vk_check_result_ != VK_SUCCESS) {                        \
            return this->setVkFailure(::lfs::vis::formatVkCheckFailure(  \
                #expr, lfs_vk_check_result_,                             \
                ::lfs::rendering::formatVulkanDiagnostic(__VA_ARGS__))); \
        }                                                                \
    } while (false)

#ifndef LFS_VK_DEBUG_ASSERT
#if defined(NDEBUG)
#define LFS_VK_DEBUG_ASSERT(condition, ...) ((void)0)
#elif defined(__CUDA_ARCH__)
#define LFS_VK_DEBUG_ASSERT(condition, ...) assert(condition)
#else
#define LFS_VK_DEBUG_ASSERT(condition, ...)                            \
    do {                                                               \
        if (!(condition)) [[unlikely]] {                               \
            ::lfs::core::detail::assertion_failed(                     \
                "LFS debug invariant", #condition,                     \
                ::lfs::rendering::formatVulkanDiagnostic(__VA_ARGS__), \
                LFS_SOURCE_SITE_CURRENT());                            \
        }                                                              \
    } while (false)
#endif
#endif
