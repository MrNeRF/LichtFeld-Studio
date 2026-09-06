/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstdint>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vulkan/vulkan.h>

namespace lfs::core::internal {

    class VulkanContext;

    struct VulkanPipeline {
        VkShaderModule shader = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        uint32_t local_size_x = 0;
        uint32_t push_constant_size = 0;
    };

    class VulkanPipelines final {
    public:
        explicit VulkanPipelines(VulkanContext& context);
        ~VulkanPipelines();

        VulkanPipelines(const VulkanPipelines&) = delete;
        VulkanPipelines& operator=(const VulkanPipelines&) = delete;

        [[nodiscard]] const VulkanPipeline& specialized(
            const std::string& module, uint32_t expected_push_constant_size,
            std::span<const uint32_t> constants);
        void shutdown();

    private:
        [[nodiscard]] VulkanPipeline load(const std::string& module,
                                          uint32_t expected_local_size_x,
                                          uint32_t expected_push_constant_size,
                                          std::span<const uint32_t> constants = {});

        VulkanContext& context_;
        std::mutex mutex_;
        std::unordered_map<std::string, VulkanPipeline> pipelines_;
        bool shutting_down_ = false;
    };

} // namespace lfs::core::internal
