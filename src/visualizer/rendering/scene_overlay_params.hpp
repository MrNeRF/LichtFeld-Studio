/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
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
            ParamCount = ViewWindow + 1,
        };
        static_assert(EllipsoidFlags + EllipsoidParamStride <= ViewIntrinsics);
        static_assert(EllipsoidExtraBase + EllipsoidParamStride * EllipsoidExtraCount == ViewWindow);

        // Exposed for tests (O4): pure function over the request, no device state.
        [[nodiscard]] LFS_VIS_API std::expected<std::vector<float>, std::string>
        buildOverlayParamsCpuFloats(
            const lfs::rendering::ViewportRenderRequest& request,
            bool selection_enabled,
            bool preview_enabled,
            bool transform_indices_enabled,
            std::size_t node_mask_count,
            bool node_visibility_cull);
    } // namespace detail
} // namespace lfs::vis
