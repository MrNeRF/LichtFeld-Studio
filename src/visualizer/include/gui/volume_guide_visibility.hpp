/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/scene.hpp"

namespace lfs::vis::gui {

    template <typename Volume>
    [[nodiscard]] bool activeVolumeGuideVisible(const bool affects_render,
                                                const core::NodeId selected_id,
                                                const std::vector<Volume>& volumes) {
        // Selection volumes are independent of any crop helper on the selected model.
        if (!affects_render)
            return true;
        if (selected_id == core::NULL_NODE)
            return false;
        for (const auto& volume : volumes) {
            if (volume.node_id == selected_id)
                return volume.effectively_visible;
        }
        return false;
    }

} // namespace lfs::vis::gui
