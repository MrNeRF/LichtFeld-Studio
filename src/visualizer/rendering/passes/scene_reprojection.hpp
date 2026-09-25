/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cmath>
#include <glm/glm.hpp>

namespace lfs::vis {

    // Maps (source_ndc.xy * depth, depth, 1) of a perspective image rendered with
    // projection * source_view to clip space of the same projection seen from
    // current_view. depth is the planar view depth the splat renderer stores, which
    // keeps the map linear in the shader's per-pixel unknowns.
    [[nodiscard]] inline glm::mat4 sceneReprojectionMatrix(const glm::mat4& source_view,
                                                           const glm::mat4& projection,
                                                           const glm::mat4& current_view) {
        const glm::dmat4 p(projection);
        const double w_per_depth = std::abs(p[2][3]);
        const double view_z_per_depth = p[2][3] < 0.0 ? -1.0 : 1.0;
        glm::dmat4 source_clip(0.0);
        source_clip[0][0] = w_per_depth;
        source_clip[1][1] = w_per_depth;
        source_clip[2][2] = p[2][2] * view_z_per_depth;
        source_clip[2][3] = w_per_depth;
        source_clip[3][2] = p[3][2];
        return glm::mat4(p * glm::dmat4(current_view) * glm::inverse(p * glm::dmat4(source_view)) *
                         source_clip);
    }

    // Maps homogeneous current-view NDC (x, y, 1) to source-image NDC for points on
    // the plane through pivot, facing the source camera. One plane keeps the warp
    // continuous: no per-pixel depth, so no speckle at depth edges. A pivot that is
    // not in front of the source camera falls back to a rotation-only warp.
    [[nodiscard]] inline glm::mat3 sceneReprojectionHomography(const glm::mat4& source_view,
                                                               const glm::mat4& projection,
                                                               const glm::mat4& current_view,
                                                               const glm::vec3& pivot) {
        constexpr double kFarPlaneDepth = 1.0e6;
        const glm::dmat4 p(projection);
        const double w_per_depth = std::abs(p[2][3]);
        const double pivot_depth = (p * glm::dmat4(source_view) * glm::dvec4(glm::dvec3(pivot), 1.0)).w / w_per_depth;
        const double depth = std::isfinite(pivot_depth) && pivot_depth > 0.0 ? pivot_depth : kFarPlaneDepth;
        const glm::dmat4 m(sceneReprojectionMatrix(source_view, projection, current_view));
        const glm::dmat3 source_to_current(
            glm::dvec3(m[0].x, m[0].y, m[0].w) * depth,
            glm::dvec3(m[1].x, m[1].y, m[1].w) * depth,
            glm::dvec3(m[2].x, m[2].y, m[2].w) * depth + glm::dvec3(m[3].x, m[3].y, m[3].w));
        return glm::mat3(glm::inverse(source_to_current));
    }

} // namespace lfs::vis
