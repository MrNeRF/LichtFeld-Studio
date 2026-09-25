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

} // namespace lfs::vis
