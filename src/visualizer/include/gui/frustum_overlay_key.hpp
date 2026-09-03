/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

namespace lfs::vis::gui {

    struct FrustumOverlayInputKey {
        std::uintptr_t scene_identity = 0;
        std::uint64_t camera_list_generation = 0;
        std::uint64_t scene_render_generation = 0;
        std::uint64_t view_projection_hash = 0;
        std::int32_t hovered_camera_id = -1;
        std::uint64_t selected_set_generation = 0;
        std::uint64_t training_loss_color_generation = 0;
        std::uint64_t thumbnail_atlas_generation = 0;
        std::uint64_t overlay_settings_hash = 0;

        bool operator==(const FrustumOverlayInputKey&) const = default;
    };

    [[nodiscard]] inline bool frustumOverlayNeedsRebuild(
        const FrustumOverlayInputKey& previous,
        const FrustumOverlayInputKey& current) noexcept {
        return previous != current;
    }

} // namespace lfs::vis::gui
