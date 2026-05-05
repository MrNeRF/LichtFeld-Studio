/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "vulkan_depth_blit_pass.hpp"
#include "vulkan_environment_pass.hpp"
#include "vulkan_mesh_pass.hpp"
#include "vulkan_split_view_pass.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::core {
    class Tensor;
}

namespace lfs::vis {
    class VulkanContext;

    struct VulkanViewportOverlayVertex {
        glm::vec2 position{0.0f};
        glm::vec4 color{1.0f};
    };

    struct VulkanViewportShapeOverlayVertex {
        glm::vec2 position{0.0f};
        glm::vec2 screen_position{0.0f};
        glm::vec2 p0{0.0f};
        glm::vec2 p1{0.0f};
        glm::vec4 color{1.0f};
        glm::vec4 params{0.0f};
    };

    struct VulkanViewportPivotOverlay {
        glm::vec2 center_ndc{0.0f};
        glm::vec2 size_ndc{0.0f};
        glm::vec3 color{0.26f, 0.59f, 0.98f};
        float opacity = 1.0f;
    };

    struct VulkanViewportTexturedOverlayVertex {
        glm::vec2 position{0.0f};
        glm::vec2 uv{0.0f};
    };

    struct VulkanViewportTexturedOverlay {
        std::uintptr_t texture_id = 0;
        glm::vec4 tint_opacity{1.0f, 1.0f, 1.0f, 0.8f};
        glm::vec4 effects{0.0f};
        std::array<VulkanViewportTexturedOverlayVertex, 6> vertices{};
    };

    struct VulkanViewportGridOverlay {
        glm::vec2 viewport_pos{0.0f, 0.0f};
        glm::vec2 viewport_size{0.0f, 0.0f};
        glm::ivec2 render_size{0, 0};
        glm::mat4 view{1.0f};
        glm::mat4 projection{1.0f};
        glm::mat4 view_projection{1.0f};
        glm::vec3 view_position{0.0f, 0.0f, 0.0f};
        int plane = 2;
        float opacity = 1.0f;
        bool orthographic = false;
    };

    struct VulkanViewportPassParams {
        std::size_t frame_slot = 0;
        glm::vec2 viewport_pos{0.0f, 0.0f};
        glm::vec2 viewport_size{0.0f, 0.0f};
        glm::vec2 framebuffer_scale{1.0f, 1.0f};
        glm::vec3 background_color{0.0f, 0.0f, 0.0f};

        std::shared_ptr<const lfs::core::Tensor> scene_image;
        glm::ivec2 scene_image_size{0, 0};
        bool scene_image_flip_y = false;
        VkImage external_scene_image = VK_NULL_HANDLE;
        VkImageView external_scene_image_view = VK_NULL_HANDLE;
        VkImageLayout external_scene_image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        std::uint64_t external_scene_image_generation = 0;

        bool grid_enabled = false;
        glm::mat4 grid_view{1.0f};
        glm::mat4 grid_projection{1.0f};
        glm::mat4 grid_view_projection{1.0f};
        glm::vec3 grid_view_position{0.0f, 0.0f, 0.0f};
        int grid_plane = 2;
        float grid_opacity = 1.0f;
        bool grid_orthographic = false;
        std::vector<VulkanViewportGridOverlay> grid_overlays;

        bool vignette_enabled = false;
        float vignette_intensity = 0.0f;
        float vignette_radius = 0.75f;
        float vignette_softness = 0.5f;

        std::vector<VulkanViewportOverlayVertex> overlay_triangles;
        std::vector<VulkanViewportShapeOverlayVertex> shape_overlay_triangles;
        std::vector<VulkanViewportShapeOverlayVertex> ui_shape_overlay_triangles;
        std::vector<VulkanViewportPivotOverlay> pivot_overlays;
        std::vector<VulkanViewportTexturedOverlay> textured_overlays;

        // GPU-rendered meshes drawn into the same color/depth attachments as the
        // viewport pass. Replaces the old CPU `rasterizeMeshTriangle` fallback path.
        glm::mat4 mesh_view_projection{1.0f};
        glm::vec3 mesh_camera_position{0.0f};
        std::vector<VulkanMeshDrawItem> mesh_items;
        std::vector<VulkanMeshViewportPanel> mesh_panels;

        // GPU-rendered equirect environment background. Replaces the old CPU
        // `renderEnvironmentBackground` per-pixel sampling loop.
        VulkanEnvironmentParams environment;

        // Splat depth blit. When set, mesh draws will depth-test against the splat
        // depth surface, so meshes occluded by splats render correctly.
        VulkanDepthBlitParams depth_blit;

        // Split-view composite. When enabled, replaces the single-tensor scene quad
        // blit with a two-panel composite (master-style divider/handle/grip in-shader).
        VulkanSplitViewParams split_view;
    };

    class LFS_VIS_API VulkanViewportPass {
    public:
        VulkanViewportPass();
        ~VulkanViewportPass();

        VulkanViewportPass(const VulkanViewportPass&) = delete;
        VulkanViewportPass& operator=(const VulkanViewportPass&) = delete;

        [[nodiscard]] bool init(VulkanContext& context);
        void prepare(VulkanContext& context, const VulkanViewportPassParams& params);
        void record(VkCommandBuffer command_buffer,
                    VkExtent2D framebuffer_extent,
                    const VulkanViewportPassParams& params);
        void shutdown();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis
