/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/crop_polygon.hpp"
#include "core/error.hpp"
#include "core/export.hpp"
#include "rendering/rendering.hpp"

#include <cstddef>
#include <span>
#include <vector>

struct VulkanGSRendererUniforms;

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
        RevealParams = 26,
        RevealEffectBounds = 27,
        CropDirectory = 28, // (base, bucket count, 0, 0), sorted by parent index.
        ParamCount = 29,    // Fixed prefix; optional custom tail, then dynamic crop directory/payload.
        CropDirectoryStride = 2,
        CropParamStride = 7,
        EllipsoidParamStride = 5,
        CropPolygonParams = 1,
        CropPolygonTransform = 2,
        CropPolygonVertices = 6,
        CropPolygonParamStride = CropPolygonVertices + lfs::core::crop_polygon::kMaxVertices / 2,
    };
    static_assert(EllipsoidFlags + EllipsoidParamStride <= ViewFlags);
    static_assert(CropPolygonTransform + 4 == CropPolygonVertices);

    // The table the renderer uploads for `request`; empty if staging fails.
    [[nodiscard]] LFS_VIS_API std::vector<float> buildParams(
        const lfs::rendering::ViewportRenderRequest& request,
        bool selection_enabled,
        bool preview_enabled,
        bool transform_indices_enabled,
        std::size_t node_mask_count,
        bool node_visibility_cull,
        std::span<const float> custom_tail = {});

    // The custom effect tail (custom_effect_layout.inc) placed after the fixed prefix
    // in a custom effect draw, with `camera` the draw's projection camera.
    [[nodiscard]] LFS_VIS_API lfs::Result<std::vector<float>> buildCustomEffectTail(
        const lfs::rendering::ViewportRenderRequest& request,
        const VulkanGSRendererUniforms& camera);

} // namespace lfs::vis::vksplat_overlay
