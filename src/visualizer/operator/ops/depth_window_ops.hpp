/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "selection/depth_window_geometry.hpp"
#include <glm/glm.hpp>

namespace lfs::vis::op {

    void registerDepthWindowOperators();
    void unregisterDepthWindowOperators();

    // Held-modifier hover: updates the overlay handle-highlight state and
    // returns the cursor for the handle under the pointer.
    // Exported (LFS_VIS_API) so tests can drive the hover/overlay state.
    [[nodiscard]] LFS_VIS_API DepthWindowCursor updateDepthWindowHover(
        const glm::vec2& screen,
        const glm::vec4& viewport_bounds,
        bool modifiers_held);
    LFS_VIS_API void clearDepthWindowHover();
    [[nodiscard]] LFS_VIS_API const DepthWindowOverlayState& depthWindowOverlayState();

} // namespace lfs::vis::op
