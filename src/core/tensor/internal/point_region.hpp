/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include "core/tensor_spatial.hpp"
#include "point_math.hpp"

namespace lfs::core::internal {
    template <PointRegion2DKind Kind>
    LFS_POINT_HD inline bool pointInRegion(float x, float y, const PointRegion2D& region,
                                           const float* geometry, size_t geometry_count) {
        if (!(x >= region.minimum_coordinate && y >= region.minimum_coordinate))
            return false;
        const float radius_sq = roundedMul(region.radius, region.radius);
        if constexpr (Kind == PointRegion2DKind::Disk) {
            const float dx = roundedSub(x, region.x0), dy = roundedSub(y, region.y0);
            return roundedAdd(roundedMul(dx, dx), roundedMul(dy, dy)) <= radius_sq;
        } else if constexpr (Kind == PointRegion2DKind::Rectangle) {
            return x >= region.x0 && x <= region.x1 && y >= region.y0 && y <= region.y1;
        } else if constexpr (Kind == PointRegion2DKind::Disks) {
            for (size_t k = 0; k < geometry_count; ++k) {
                const float dx = roundedSub(x, geometry[k * 2]), dy = roundedSub(y, geometry[k * 2 + 1]);
                if (roundedAdd(roundedMul(dx, dx), roundedMul(dy, dy)) <= radius_sq)
                    return true;
            }
            return false;
        } else {
            bool inside = false;
            for (size_t k = 0, j = geometry_count - 1; k < geometry_count; j = k++) {
                const float xi = geometry[2 * k], yi = geometry[2 * k + 1];
                const float xj = geometry[2 * j], yj = geometry[2 * j + 1];
                if ((yi > y) != (yj > y) &&
                    x < roundedAdd(roundedDiv(roundedMul(roundedSub(xj, xi), roundedSub(y, yi)), roundedSub(yj, yi)), xi))
                    inside = !inside;
            }
            return inside;
        }
    }
} // namespace lfs::core::internal
