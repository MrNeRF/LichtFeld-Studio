/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <filesystem>
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    class VulkanContext;

    struct VulkanEnvironmentParams {
        bool enabled = false;
        std::filesystem::path map_path;
        glm::mat3 camera_to_world{1.0f};
        glm::vec4 intrinsics{0.0f}; // focal_x, focal_y, cx, cy
        glm::vec2 viewport_size{0.0f};
        float exposure = 0.0f;
        float rotation_radians = 0.0f;
        bool equirectangular_view = false;
    };

    class LFS_VIS_API VulkanEnvironmentPass {
    public:
        VulkanEnvironmentPass();
        ~VulkanEnvironmentPass();
        VulkanEnvironmentPass(const VulkanEnvironmentPass&) = delete;
        VulkanEnvironmentPass& operator=(const VulkanEnvironmentPass&) = delete;
        VulkanEnvironmentPass(VulkanEnvironmentPass&&) noexcept;
        VulkanEnvironmentPass& operator=(VulkanEnvironmentPass&&) noexcept;

        [[nodiscard]] bool init(VulkanContext& context,
                                VkFormat color_format,
                                VkFormat depth_stencil_format,
                                VkBuffer screen_quad_buffer);

        // Loads / re-uploads the equirect texture if the path changed. Cheap when path
        // is unchanged.
        void prepare(const VulkanEnvironmentParams& params);

        // Records a full-screen quad draw that fills the color attachment with the
        // ACES-tonemapped equirect background. Caller is mid-render.
        void record(VkCommandBuffer command_buffer,
                    VkExtent2D framebuffer_extent,
                    const VulkanEnvironmentParams& params);

        void shutdown();

        [[nodiscard]] bool hasTexture() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis
