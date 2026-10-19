/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>

namespace lfs::vis {
    struct VulkanViewportPassParams;
}

namespace lfs::vis::gui::detail {
    // Converts the queued NativeOverlayDrawList commands into viewport pass overlay geometry,
    // clipped to each command's clip rect.
    LFS_VIS_API void appendLineRendererOverlays(VulkanViewportPassParams& params);
} // namespace lfs::vis::gui::detail
