/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::core {
    struct MeshData;
}

namespace lfs::vis {

    class VulkanContext;

    // GPU-rendered mesh draw item. Full PBR (Cook-Torrance: GGX + Smith + Schlick) with
    // albedo / normal / metallic-roughness textures, vertex colors, Reinhard tonemap +
    // sRGB gamma — feature-matched to master's `mesh_pbr.frag`. Shadow maps are not yet
    // wired (master defaults shadow_enabled=false; will be a follow-up depth-only pass).
    struct VulkanMeshDrawItem {
        const lfs::core::MeshData* mesh = nullptr;
        glm::mat4 model{1.0f};
        glm::vec3 light_dir{0.3f, 1.0f, 0.5f};
        float light_intensity = 0.7f;
        float ambient = 0.4f;
        bool backface_culling = true;
        // Selection emphasis (matches master's mesh_pbr.frag).
        bool is_emphasized = false;
        bool dim_non_emphasized = false;
        float flash_intensity = 0.0f;
        // Wireframe overlay drawn after the main mesh pass.
        bool wireframe_overlay = false;
        glm::vec3 wireframe_color{0.2f, 0.2f, 0.2f};
        float wireframe_width = 1.0f;
        // Shadow map: depth-only pre-pass per mesh, then sampler2DShadow read in main shader.
        bool shadow_enabled = false;
        int shadow_map_resolution = 2048;
    };

    struct VulkanMeshPassParams {
        glm::mat4 view_projection{1.0f};
        glm::vec3 camera_position{0.0f};
        std::vector<VulkanMeshDrawItem> items;
    };

    struct VulkanMeshViewportPanel {
        float start_position = 0.0f;
        float end_position = 1.0f;
        glm::mat4 view_projection{1.0f};
        glm::vec3 camera_position{0.0f};
    };

    class LFS_VIS_API VulkanMeshPass {
    public:
        VulkanMeshPass();
        ~VulkanMeshPass();

        VulkanMeshPass(const VulkanMeshPass&) = delete;
        VulkanMeshPass& operator=(const VulkanMeshPass&) = delete;
        VulkanMeshPass(VulkanMeshPass&&) noexcept;
        VulkanMeshPass& operator=(VulkanMeshPass&&) noexcept;

        // Initialize pipelines / descriptor layouts. color_format and depth_format
        // must match the render-pass attachments the caller will record into.
        [[nodiscard]] bool init(VulkanContext& context,
                                VkFormat color_format,
                                VkFormat depth_stencil_format);

        // Upload any new / changed mesh GPU buffers. Idempotent — caches by
        // (MeshData*, generation).
        void prepare(VulkanContext& context, const VulkanMeshPassParams& params);

        // Record draw commands into an already-active dynamic-rendering pass.
        // Caller must have transitioned the color attachment to
        // VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL and bound a depth attachment.
        void record(VkCommandBuffer command_buffer,
                    VkRect2D viewport_rect,
                    const VulkanMeshPassParams& params);

        void shutdown();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis
