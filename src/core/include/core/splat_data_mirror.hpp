/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <glm/glm.hpp>

namespace lfs::core {

    class SplatData;
    class Tensor;

    enum class MirrorAxis : uint8_t {
        X, // Mirror across YZ plane
        Y, // Mirror across XZ plane
        Z  // Mirror across XY plane
    };

    /// Compute the centroid of selected splats
    glm::vec3 compute_selection_center(const SplatData& splat_data, const Tensor& selection_mask);

    /// Mirror selected gaussians in-place (positions, rotations, SH coefficients)
    void mirror_gaussians(SplatData& splat_data,
                          const Tensor& selection_mask,
                          MirrorAxis axis,
                          const glm::vec3& center);

} // namespace lfs::core
