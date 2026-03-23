/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/rendering.hpp"
#include "rendering_types.hpp"

namespace lfs::vis {

    class SceneManager;

    struct ViewportInteractionContext {
        SceneManager* scene_manager = nullptr;
        lfs::rendering::ViewportData viewport_data{};
        ViewportRegion viewport_region{};
        bool pick_context_valid = false;

        void updatePickContext(const ViewportRegion* region,
                               const lfs::rendering::ViewportData& data) {
            if (region) {
                viewport_region = *region;
                viewport_data = data;
                pick_context_valid = true;
            } else {
                pick_context_valid = false;
            }
        }
    };

} // namespace lfs::vis
