/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <string>

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

    [[nodiscard]] inline std::string frustumOverlayKeyDifferenceNames(
        const FrustumOverlayInputKey& previous,
        const FrustumOverlayInputKey& current) {
        std::string result;
        const auto append = [&result](const char* const name) {
            if (!result.empty())
                result += ',';
            result += name;
        };
        if (previous.scene_identity != current.scene_identity)
            append("scene_identity");
        if (previous.camera_list_generation != current.camera_list_generation)
            append("camera_list_generation");
        if (previous.scene_render_generation != current.scene_render_generation)
            append("scene_render_generation");
        if (previous.view_projection_hash != current.view_projection_hash)
            append("view_projection_hash");
        if (previous.hovered_camera_id != current.hovered_camera_id)
            append("hovered_camera_id");
        if (previous.selected_set_generation != current.selected_set_generation)
            append("selected_set_generation");
        if (previous.training_loss_color_generation != current.training_loss_color_generation)
            append("training_loss_color_generation");
        if (previous.thumbnail_atlas_generation != current.thumbnail_atlas_generation)
            append("thumbnail_atlas_generation");
        if (previous.overlay_settings_hash != current.overlay_settings_hash)
            append("overlay_settings_hash");
        return result;
    }

} // namespace lfs::vis::gui
