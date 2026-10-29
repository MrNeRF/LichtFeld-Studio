/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "video_mask_cells.hpp"

#include "core/assert.hpp"
#include "core/nn/ops.hpp"

#include <algorithm>

namespace lfs::preprocessing::detail {
    namespace {
        using core::DataType;
        using core::Device;
        using core::Tensor;

        constexpr float kMinimumCellContrast = 0.02f, kMinimumCellCorner = 0.008f;
        constexpr float kFixedCorrelation = 0.85f, kMovingCorrelation = 0.4f;

        std::vector<float> correlation(const CellPatches& first, const CellPatches& second) {
            LFS_ASSERT_MSG(first.unit.shape() == second.unit.shape(), "Cell patches of different grids");
            return first.unit.mul(second.unit).sum(1).to(Device::CPU).contiguous().to_vector();
        }
    } // namespace

    CellPatches cell_patches(const Tensor& gray, const int grid) {
        LFS_ASSERT_MSG(gray.ndim() == 4 && gray.size(0) == 1 && gray.size(1) == 1 && gray.dtype() == DataType::Float32,
                       "Cell patches expect a 1 x 1 x H x W Float32 image");
        LFS_ASSERT_MSG(grid > 0, "Cell grid must be positive");
        const int per_side = grid, side = kCellSide, cells = per_side * per_side;
        const auto blocks = core::nn::resize2d_bilinear_antialiased(gray, grid * kCellSide, grid * kCellSide)
                                .reshape({per_side, side, per_side, side})
                                .permute({0, 2, 1, 3})
                                .contiguous()
                                .reshape({cells, side, side});
        const auto flat = blocks.reshape({cells, side * side});
        const auto centered = flat.sub(flat.mean(1, true));
        const auto norm = centered.norm(2.0f, 1, true);
        const auto inner = [&](const size_t row, const size_t column) {
            return blocks.slice(1, row, row + side - 2).slice(2, column, column + side - 2).contiguous();
        };
        const auto gx = inner(1, 2).sub(inner(1, 0)).mul(0.5f);
        const auto gy = inner(2, 1).sub(inner(0, 1)).mul(0.5f);
        const auto xx = gx.mul(gx).mean({1, 2}), yy = gy.mul(gy).mean({1, 2}), xy = gx.mul(gy).mean({1, 2});
        const auto half_difference = xx.sub(yy).mul(0.5f);
        const auto smaller_eigenvalue =
            xx.add(yy).mul(0.5f).sub(half_difference.mul(half_difference).add(xy.mul(xy)).sqrt()).clamp_min(0.0f);
        const auto contrast = norm.reshape({cells})
                                  .div(float(kCellSide))
                                  .mul(smaller_eigenvalue.sqrt().ge(kMinimumCellCorner).to(DataType::Float32));
        return {centered.div(norm.clamp_min(1e-6f)), contrast.to(Device::CPU).contiguous().to_vector()};
    }

    CellComparison compare_cells(const CellPatches& first, const CellPatches& second,
                                 const CellPatches& first_scene, const CellPatches& second_scene) {
        LFS_ASSERT_MSG(first.contrast.size() == size_t(kCellGrid * kCellGrid), "Fine cell patches expected");
        CellComparison result;
        const auto fine = correlation(first, second);
        for (size_t cell = 0; cell < fine.size(); ++cell) {
            if (std::min(first.contrast[cell], second.contrast[cell]) < kMinimumCellContrast)
                continue;
            ++result.textured;
            result.cells[cell] = fine[cell] >= kFixedCorrelation    ? CellMotion::Fixed
                                 : fine[cell] <= kMovingCorrelation ? CellMotion::Moving
                                                                    : CellMotion::Unknown;
        }
        const auto scene = correlation(first_scene, second_scene);
        std::vector<float> textured;
        for (size_t cell = 0; cell < scene.size(); ++cell)
            if (std::min(first_scene.contrast[cell], second_scene.contrast[cell]) >= kMinimumCellContrast)
                textured.push_back(scene[cell]);
        if (!textured.empty()) {
            const auto middle = textured.begin() + std::ptrdiff_t(textured.size() / 2);
            std::nth_element(textured.begin(), middle, textured.end());
            result.scene_correlation = *middle;
        }
        return result;
    }

} // namespace lfs::preprocessing::detail
