/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::rendering::detail {

    [[nodiscard]] constexpr bool meshFragmentPassesDepthTest(
        const float alpha,
        const float mesh_view_depth,
        const float scene_view_depth) noexcept {
        return alpha > 0.0f && mesh_view_depth > 0.0f && mesh_view_depth < scene_view_depth;
    }

} // namespace lfs::rendering::detail
