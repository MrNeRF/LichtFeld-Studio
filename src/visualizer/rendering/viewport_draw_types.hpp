/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
namespace lfs::core {
    struct MeshData;
}
namespace lfs::vis {
    struct ViewportOverlayVertex {
        glm::vec2 position{0.0f};
        glm::vec4 color{1.0f};
    };

    struct ViewportShapeOverlayVertex {
        glm::vec2 position{0.0f};
        glm::vec2 screen_position{0.0f};
        glm::vec2 p0{0.0f};
        glm::vec2 p1{0.0f};
        glm::vec4 color{1.0f};
        glm::vec4 params{0.0f};
        // Linear view-space depth (positive forward). 0 means "always in front"
        // — used by UI overlays (gizmos, pivot) so they don't fade behind splats.
        float view_depth = 0.0f;
    };

    struct ViewportPivotOverlay {
        glm::vec2 center_ndc{0.0f};
        glm::vec2 size_ndc{0.0f};
        glm::vec3 color{0.26f, 0.59f, 0.98f};
        float opacity = 1.0f;
    };

    struct ViewportTexturedOverlayVertex {
        glm::vec2 position{0.0f};
        glm::vec2 uv{0.0f};
        // Linear view-space depth (positive forward). 0 = "always in front", skips
        // the splat-depth occlusion test.
        float view_depth = 0.0f;
    };

    struct ViewportTexturedOverlay {
        std::shared_ptr<const lfs::core::Tensor> image;
        glm::vec4 tint_opacity{1.0f, 1.0f, 1.0f, 0.8f};
        glm::vec4 effects{0.0f};
        std::array<ViewportTexturedOverlayVertex, 6> vertices{};
    };

    struct ViewportGridOverlay {
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

    // Layout must match frustum.vert's std430 FrustumInstance.
    struct ViewportFrustumInstance {
        glm::mat4 model{1.0f};
        glm::vec4 color{1.0f};
    };

    // One instanced draw range for a viewport panel.
    struct ViewportFrustumBatch {
        glm::mat4 view{1.0f};
        glm::vec2 viewport_pos{0.0f, 0.0f};
        glm::vec2 viewport_size{0.0f, 0.0f};
        glm::vec2 render_size{0.0f, 0.0f};
        float focal_x = 0.0f;
        float focal_y = 0.0f;
        bool orthographic = false;
        bool equirectangular = false;
        std::uint32_t first_instance = 0;
        std::uint32_t instance_count = 0;
    };

    struct ViewportFrustumOverlayData {
        std::uint64_t generation = 0;
        std::vector<ViewportOverlayVertex> overlay_triangles;
        std::vector<ViewportTexturedOverlay> textured_overlays;
        std::vector<ViewportFrustumInstance> frustum_instances;
        std::vector<ViewportFrustumBatch> frustum_batches;
    };

    struct ViewportMeshDrawItem {
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

    struct ViewportMeshPassDesc {
        glm::mat4 view_projection{1.0f};
        glm::vec3 camera_position{0.0f};
        std::vector<ViewportMeshDrawItem> items;
        std::size_t frame_slot = 0;
        std::size_t draw_group = 0;
        std::size_t draw_group_count = 1;
    };
    struct ViewportMeshPanel {
        float start_position = 0.0f;
        float end_position = 1.0f;
        glm::mat4 view_projection{1.0f};
        glm::vec3 camera_position{0.0f};
    };

    struct ViewportEnvironment {
        bool enabled = false;
        std::filesystem::path map_path;
        glm::mat3 camera_to_world{1.0f};
        glm::vec4 intrinsics{0.0f}; // focal_x, focal_y, cx, cy
        glm::vec2 viewport_size{0.0f};
        float exposure = 0.0f;
        float rotation_radians = 0.0f;
        bool equirectangular_view = false;
    };

} // namespace lfs::vis
