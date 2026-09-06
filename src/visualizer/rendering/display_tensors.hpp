/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"
#include "rendering/render_constants.hpp"

#include <glm/glm.hpp>
#include <memory>

namespace lfs::vis {

    // Palette for the depth display: near_t in [0, 1], 0 far and 1 near.
    LFS_VIS_API glm::vec3 depthPaletteForDisplay(float near_t);

    // [H, W] depth in any device to a CPU [3, H, W] display image: a robust
    // 2 to 98 percent range over the valid pixels, grayscale or the palette,
    // background where the depth is missing.
    LFS_VIS_API std::shared_ptr<lfs::core::Tensor> makeDepthDisplayTensor(
        const lfs::core::Tensor& depth,
        lfs::rendering::DepthVisualizationMode depth_visualization_mode,
        const glm::vec3& background_color);

    // [3, H, W] or [H, W, 3] normals to a CPU [3, H, W] display image: unit
    // normals mapped to [0, 1], mid-gray where the normal is degenerate.
    LFS_VIS_API std::shared_ptr<lfs::core::Tensor> makeNormalDisplayTensor(
        const lfs::core::Tensor& normal);

} // namespace lfs::vis
