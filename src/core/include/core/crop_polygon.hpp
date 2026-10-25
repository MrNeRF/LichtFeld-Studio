/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <glm/glm.hpp>
#include <limits>
#include <optional>
#include <span>
#include <vector>

// Geometry of the extruded polygon crop volume. The outline lies in a plane
// fixed by the camera; the node frame has local z along the camera forward.
namespace lfs::core::crop_polygon {

    inline constexpr std::size_t kMinVertices = 3;
    inline constexpr std::size_t kMaxVertices = 64;
    inline constexpr float kMinHalfDepth = 1e-4f;

    // 3..64 finite vertices, at least three of them distinct, with non-zero area.
    [[nodiscard]] inline bool isValidOutline(const std::span<const glm::vec2> vertices) {
        if (vertices.size() < kMinVertices || vertices.size() > kMaxVertices)
            return false;
        glm::vec2 lo(std::numeric_limits<float>::max());
        glm::vec2 hi(std::numeric_limits<float>::lowest());
        for (const glm::vec2& v : vertices) {
            if (!std::isfinite(v.x) || !std::isfinite(v.y))
                return false;
            lo = glm::min(lo, v);
            hi = glm::max(hi, v);
        }
        std::size_t distinct = 0;
        for (std::size_t i = 0; i < vertices.size() && distinct < kMinVertices; ++i) {
            if (std::none_of(vertices.begin(), vertices.begin() + static_cast<std::ptrdiff_t>(i),
                             [&](const glm::vec2& earlier) { return earlier == vertices[i]; }))
                ++distinct;
        }
        if (distinct < kMinVertices)
            return false;
        double twice_area = 0.0;
        for (std::size_t i = 0; i < vertices.size(); ++i) {
            const glm::vec2 a = vertices[i];
            const glm::vec2 b = vertices[(i + 1) % vertices.size()];
            twice_area += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
        }
        const double box_area = static_cast<double>(hi.x - lo.x) * static_cast<double>(hi.y - lo.y);
        return std::isfinite(twice_area) && box_area > 0.0 && std::abs(twice_area) * 0.5 > box_area * 1e-6;
    }

    [[nodiscard]] inline bool isValidHalfDepth(const float half_depth) {
        return std::isfinite(half_depth) && half_depth > 0.0f;
    }

    // (max|x|, max|y|): with a centred outline this is the prism's local half box.
    [[nodiscard]] inline glm::vec2 outlineHalfExtents(const std::span<const glm::vec2> vertices) {
        glm::vec2 extents(0.0f);
        for (const glm::vec2& v : vertices)
            extents = glm::max(extents, glm::abs(v));
        return extents;
    }

    // Even-odd rule; the CPU reference for the shader and the Tensor crop.
    [[nodiscard]] inline bool containsPoint(const std::span<const glm::vec2> vertices, const glm::vec2 point) {
        bool inside = false;
        for (std::size_t i = 0, j = vertices.size() - 1; i < vertices.size(); j = i++) {
            const glm::vec2 a = vertices[i];
            const glm::vec2 b = vertices[j];
            if ((a.y > point.y) != (b.y > point.y) &&
                point.x < (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x)
                inside = !inside;
        }
        return inside;
    }

    struct Plane {
        glm::vec3 point{0.0f};
        glm::vec3 normal{0.0f, 0.0f, 1.0f};
    };

    struct Ray {
        glm::vec3 origin{0.0f};
        glm::vec3 direction{0.0f, 0.0f, 1.0f};
    };

    struct Frame {
        // Columns: right, camera down, forward (right-handed), then the prism centre.
        glm::mat4 world_from_local{1.0f};
        std::vector<glm::vec2> vertices;
        float half_depth = kMinHalfDepth;
    };

    // Plane through `point`, facing the camera forward direction.
    [[nodiscard]] LFS_CORE_API std::optional<Plane> makePlane(const glm::vec3& camera_forward,
                                                              const glm::vec3& point);

    // True when the camera's orbit pivot is finite and lies in front of the camera.
    [[nodiscard]] LFS_CORE_API bool pivotIsInFront(const glm::vec3& camera_position,
                                                   const glm::vec3& camera_forward,
                                                   const glm::vec3& pivot);

    // Drawing plane fixed at the first click. Outlines usually start at an
    // object's silhouette, where the click hits the background, so the plane
    // goes through the orbit pivot when it is in front of the camera and only
    // falls back to the world point under the first click.
    [[nodiscard]] LFS_CORE_API std::optional<Plane> makeDrawingPlane(const glm::vec3& camera_position,
                                                                     const glm::vec3& camera_forward,
                                                                     const glm::vec3& pivot,
                                                                     const std::optional<glm::vec3>& first_click_point);

    // Ray from two unprojections of the same pixel at different depths. This is
    // exact for perspective and orthographic cameras alike.
    [[nodiscard]] LFS_CORE_API std::optional<Ray> rayThroughPoints(const glm::vec3& near_point,
                                                                   const glm::vec3& far_point);

    // Forward hits only; rays parallel to the plane have no intersection.
    [[nodiscard]] LFS_CORE_API std::optional<glm::vec3> intersect(const Ray& ray, const Plane& plane);

    // Node frame for a closed outline drawn on `plane`. The origin is the centre
    // of the outline's in-plane AABB, on the plane. The default half depth is the
    // larger in-plane half extent, so the prism starts roughly cube-like around
    // the plane and the user extrudes it with the cap handles. Covering the
    // parent's bounds instead spans the whole scene on real captures.
    [[nodiscard]] LFS_CORE_API std::optional<Frame> buildFrame(std::span<const glm::vec3> world_points,
                                                               const Plane& plane,
                                                               const glm::vec3& camera_right);

} // namespace lfs::core::crop_polygon
