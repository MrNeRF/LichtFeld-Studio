/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/crop_polygon.hpp"
#include "core/export.hpp"
#include "rendering/rendering.hpp"

#include <cstddef>
#include <vector>

namespace lfs::vis::vksplat_overlay {

    // Overlay parameter table in float4 slots. utils.slang mirrors every index;
    // change both together.
    enum ParamIndex : std::size_t {
        CropFlags = 0,
        CropMin = 1,
        CropMax = 2,
        CropTransform = 3,
        EllipsoidFlags = 7,
        EllipsoidRadii = 8,
        EllipsoidTransform = 9,
        ViewFlags = 13,
        ViewMin = 14,
        ViewMax = 15,
        ViewTransform = 16,
        EmphasisFlags = 20,
        CursorFlags = 21,
        MarkerFlags = 22,
        SelectionCursor = 23,
        SelectionFlags = 24,
        VisibilityFlags = 25,
        CropExtraBase = 26,
        CropParamStride = 7,
        CropExtraCount = 15,
        EllipsoidExtraBase = CropExtraBase + CropParamStride * CropExtraCount,
        EllipsoidParamStride = 5,
        EllipsoidExtraCount = 15,
        // Polygon slots: flags, (half_depth, vertex_count), world->local rows,
        // then two vertices per float4. Packed densely; the first disabled slot ends the list.
        CropPolygonBase = EllipsoidExtraBase + EllipsoidParamStride * EllipsoidExtraCount,
        CropPolygonParams = 1,
        CropPolygonTransform = 2,
        CropPolygonVertices = 6,
        CropPolygonParamStride = CropPolygonVertices + lfs::core::crop_polygon::kMaxVertices / 2,
        CropPolygonCount = 16,
        RevealParams = CropPolygonBase + CropPolygonParamStride * CropPolygonCount,
        RevealEffectBounds = RevealParams + 1,
        ParamCount = RevealEffectBounds + 1,
    };
    static_assert(EllipsoidFlags + EllipsoidParamStride <= ViewFlags);
    static_assert(CropPolygonTransform + 4 == CropPolygonVertices);
    static_assert(CropPolygonBase + CropPolygonParamStride * CropPolygonCount == RevealParams);

    // The table the renderer uploads for `request`; empty if staging fails.
    [[nodiscard]] LFS_VIS_API std::vector<float> buildParams(
        const lfs::rendering::ViewportRenderRequest& request,
        bool selection_enabled,
        bool preview_enabled,
        bool transform_indices_enabled,
        std::size_t node_mask_count,
        bool node_visibility_cull);

} // namespace lfs::vis::vksplat_overlay
