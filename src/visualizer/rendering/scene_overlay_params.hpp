/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/crop_polygon.hpp"
#include "core/export.hpp"
#include "rendering/rendering.hpp"
#include <expected>
#include <string>
#include <vector>
namespace lfs::vis {
    namespace detail {
        // Slot indices for the overlay parameter table. Defined here (not in the .cpp anonymous
        // namespace) so the slot-12 packing test can name detail::ViewIntrinsics.
        enum OverlayParamIndex : std::size_t {
            CropFlags = 0,
            CropMin = 1,
            CropMax = 2,
            CropTransform = 3,
            EllipsoidFlags = 7,
            EllipsoidRadii = 8,
            EllipsoidTransform = 9,
            ViewIntrinsics = 12,
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
            ViewWindow = EllipsoidExtraBase + EllipsoidParamStride * EllipsoidExtraCount,
            // Legacy overlay prefix; polygon-free tensor projection uploads only these slots.
            ParamCount = ViewWindow + 1,
            // Crop polygon slots, read by VkSplat and tensor projection: flags, (half_depth,
            // vertex_count), world->local rows, then two vertices per float4. Packed densely;
            // the first disabled slot ends the list. utils.slang mirrors every index.
            CropPolygonBase = ParamCount,
            CropPolygonParams = 1,
            CropPolygonTransform = 2,
            CropPolygonVertices = 6,
            CropPolygonParamStride = CropPolygonVertices + lfs::core::crop_polygon::kMaxVertices / 2,
            CropPolygonCount = 16,
            // The whole table buildOverlayParamsCpuFloats returns.
            TableParamCount = CropPolygonBase + CropPolygonParamStride * CropPolygonCount,
            // Sequencer reveal slots, read by VkSplat only.
            RevealParams = TableParamCount,
            RevealEffectBounds = RevealParams + 1,
            // The table buildVksplatOverlayParamsCpuFloats returns; a custom effect tail follows it.
            VksplatTableParamCount = RevealEffectBounds + 1,
        };
        static_assert(EllipsoidFlags + EllipsoidParamStride <= ViewIntrinsics);
        static_assert(EllipsoidExtraBase + EllipsoidParamStride * EllipsoidExtraCount == ViewWindow);
        static_assert(CropPolygonTransform + 4 == CropPolygonVertices);

        // Exposed for tests (O4): pure function over the request, no device state.
        [[nodiscard]] LFS_VIS_API std::expected<std::vector<float>, std::string>
        buildOverlayParamsCpuFloats(
            const lfs::rendering::ViewportRenderRequest& request,
            bool selection_enabled,
            bool preview_enabled,
            bool transform_indices_enabled,
            std::size_t node_mask_count,
            bool node_visibility_cull);

        // Extends that table by the reveal slots VkSplat reads.
        LFS_VIS_API void appendVksplatRevealParams(std::vector<float>& table,
                                                   const lfs::rendering::ViewportRenderRequest& request);
    } // namespace detail
} // namespace lfs::vis
