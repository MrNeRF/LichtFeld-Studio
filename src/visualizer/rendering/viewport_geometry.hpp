/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>

namespace lfs::vis {
    struct FramebufferRect {
        std::int32_t x = 0;
        std::int32_t y = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
    };

    // A logical-pixel rect scaled to framebuffer pixels and clipped to `extent`.
    inline FramebufferRect scaledFramebufferRect(const glm::vec2 position, const glm::vec2 size,
                                                 const glm::vec2 scale, const glm::ivec2 extent) {
        const float sx = scale.x > 0.0f ? scale.x : 1.0f;
        const float sy = scale.y > 0.0f ? scale.y : 1.0f;
        const int x0 = std::clamp(static_cast<int>(std::lround(position.x * sx)), 0, extent.x);
        const int y0 = std::clamp(static_cast<int>(std::lround(position.y * sy)), 0, extent.y);
        const int x1 = std::clamp(static_cast<int>(std::lround((position.x + size.x) * sx)), 0, extent.x);
        const int y1 = std::clamp(static_cast<int>(std::lround((position.y + size.y) * sy)), 0, extent.y);
        return {.x = x0,
                .y = y0,
                .width = static_cast<std::uint32_t>(std::max(x1 - x0, 0)),
                .height = static_cast<std::uint32_t>(std::max(y1 - y0, 0))};
    }

    // Pixel rays across the screen, from the bottom-left corner, for the grid shaders. Directions
    // are kept apart from the origin: two far-away points subtracted in float lose the direction
    // when zoomed out. The grid fades out on a floor around the pivot of `floor_radius`.
    struct GridRays {
        glm::vec3 origin{0.0f}, origin_x{0.0f}, origin_y{0.0f};
        glm::vec3 dir{0.0f}, dir_x{0.0f}, dir_y{0.0f};
        float floor_radius = 0.0f;
    };

    inline GridRays gridRays(const glm::mat4& view, const glm::mat4& projection, const glm::vec3& pivot,
                             const bool orthographic) {
        const glm::mat4 view_inv = glm::inverse(view);
        const glm::vec3 cam_pos = glm::vec3(view_inv[3]);
        const glm::vec3 cam_right = glm::vec3(view_inv[0]);
        const glm::vec3 cam_up = glm::vec3(view_inv[1]);
        const glm::vec3 cam_forward = -glm::vec3(view_inv[2]);
        // The floor is sized in visible half-heights at the pivot.
        constexpr float kGridFloorRadius = 3.5f;
        GridRays rays{.origin = cam_pos, .dir = cam_forward};
        float pivot_half_height = 0.0f;
        if (orthographic) {
            const float half_width = 1.0f / projection[0][0];
            const float half_height = 1.0f / std::abs(projection[1][1]);
            const glm::vec3 right_offset = cam_right * half_width;
            const glm::vec3 up_offset = cam_up * half_height;
            constexpr float kRayNear = -1000.0f;
            rays.origin = cam_pos + cam_forward * kRayNear - right_offset - up_offset;
            rays.origin_x = right_offset * 2.0f;
            rays.origin_y = up_offset * 2.0f;
            pivot_half_height = half_height;
        } else {
            const float fov_y = 2.0f * std::atan(1.0f / std::abs(projection[1][1]));
            const float aspect = std::abs(projection[1][1] / projection[0][0]);
            const float half_height = std::tan(fov_y * 0.5f);
            const float half_width = half_height * aspect;
            const glm::vec3 right_offset = cam_right * half_width;
            const glm::vec3 up_offset = cam_up * half_height;
            rays.dir = cam_forward - right_offset - up_offset;
            rays.dir_x = right_offset * 2.0f;
            rays.dir_y = up_offset * 2.0f;
            pivot_half_height = glm::length(pivot - cam_pos) * half_height;
        }
        rays.floor_radius = kGridFloorRadius * pivot_half_height;
        return rays;
    }

    // Orthographic light frustum enclosing a mesh's world-space bounds, for its shadow map.
    inline glm::mat4 meshShadowViewProjection(const glm::vec3& aabb_min, const glm::vec3& aabb_max,
                                              const glm::mat4& model, const glm::vec3& light_dir) {
        glm::vec3 ws_min(std::numeric_limits<float>::max());
        glm::vec3 ws_max(std::numeric_limits<float>::lowest());
        for (int corner = 0; corner < 8; ++corner) {
            const glm::vec3 local{(corner & 1) ? aabb_max.x : aabb_min.x,
                                  (corner & 2) ? aabb_max.y : aabb_min.y,
                                  (corner & 4) ? aabb_max.z : aabb_min.z};
            const glm::vec3 world = glm::vec3(model * glm::vec4(local, 1.0f));
            ws_min = glm::min(ws_min, world);
            ws_max = glm::max(ws_max, world);
        }
        const glm::vec3 center = (ws_min + ws_max) * 0.5f;
        const float radius = glm::length(ws_max - ws_min) * 0.5f;
        const glm::vec3 dir = glm::length(light_dir) > 1e-6f ? glm::normalize(light_dir)
                                                             : glm::vec3(0.0f, 1.0f, 0.0f);
        glm::vec3 up(0.0f, 1.0f, 0.0f);
        if (std::abs(glm::dot(dir, up)) > 0.99f)
            up = glm::vec3(0.0f, 0.0f, 1.0f);
        const glm::mat4 view = glm::lookAt(center + dir * radius * 2.0f, center, up);
        return glm::ortho(-radius, radius, -radius, radius, 0.01f, radius * 4.0f) * view;
    }
} // namespace lfs::vis
