/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lfs::preprocessing::detail {

    // Frames are compared on square grids of 8 x 8 pixel cells over the frame resized to the grid: a fine grid locates
    // camera-fixed content, a coarse one measures how far the scene moved.
    inline constexpr int kCellGrid = 48, kSceneGrid = 12, kCellSide = 8;

    // A pair shows camera motion when the median correlation of textured scene cells is at most this.
    inline constexpr float kMaximumSceneCorrelation = 0.5f;

    enum class CellMotion : std::uint8_t { Unknown,
                                           Fixed,
                                           Moving };

    // Zero-mean, unit-norm grayscale patches of every cell: the dot product of two frames' patches is each cell's
    // normalized cross-correlation, which ignores exposure gain and offset. Contrast is the patch standard deviation,
    // zero for cells without gradients in two directions, since stripes, edges and ramps stay correlated when they move.
    struct CellPatches {
        core::Tensor unit;
        std::vector<float> contrast;
    };

    // `gray` is a 1 x 1 x H x W Float32 CUDA image in [0, 1].
    [[nodiscard]] CellPatches cell_patches(const core::Tensor& gray, int grid);

    struct CellComparison {
        std::array<CellMotion, kCellGrid * kCellGrid> cells{};
        float scene_correlation = 1;
        size_t textured = 0;
    };

    [[nodiscard]] CellComparison compare_cells(const CellPatches& first, const CellPatches& second,
                                               const CellPatches& first_scene, const CellPatches& second_scene);

} // namespace lfs::preprocessing::detail
