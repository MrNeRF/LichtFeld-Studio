/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/rendering.hpp"
#include "rendering_types.hpp"
#include <array>
#include <optional>
#include <span>

namespace lfs::vis {

    class SceneManager;

    struct ViewportInteractionPanel {
        SplitViewPanelId panel = SplitViewPanelId::Left;
        lfs::rendering::ViewportData viewport_data{};
        glm::vec2 viewport_pos{0.0f};
        glm::vec2 viewport_size{0.0f};

        [[nodiscard]] bool valid() const {
            return viewport_size.x > 0.0f && viewport_size.y > 0.0f;
        }

        [[nodiscard]] bool contains(const glm::vec2& screen_point) const {
            return screen_point.x >= viewport_pos.x &&
                   screen_point.y >= viewport_pos.y &&
                   screen_point.x < viewport_pos.x + viewport_size.x &&
                   screen_point.y < viewport_pos.y + viewport_size.y;
        }
    };

    struct ViewportInteractionContext {
        SceneManager* scene_manager = nullptr;
        std::array<ViewportInteractionPanel, 2> panels{};
        size_t panel_count = 0;
        bool pick_context_valid = false;

        [[nodiscard]] const ViewportInteractionPanel* findPanel(const SplitViewPanelId panel_id) const {
            for (size_t i = 0; i < panel_count; ++i) {
                if (panels[i].panel == panel_id && panels[i].valid()) {
                    return &panels[i];
                }
            }
            return nullptr;
        }

        [[nodiscard]] const ViewportInteractionPanel* resolvePanel(
            const glm::vec2& screen_point,
            const std::optional<SplitViewPanelId> panel_override = std::nullopt) const {
            if (!pick_context_valid || panel_count == 0) {
                return nullptr;
            }
            if (panel_override) {
                return findPanel(*panel_override);
            }
            for (size_t i = 0; i < panel_count; ++i) {
                if (panels[i].valid() && panels[i].contains(screen_point)) {
                    return &panels[i];
                }
            }
            return nullptr;
        }

        void updatePickContext(const std::span<const ViewportInteractionPanel> active_panels) {
            panel_count = 0;
            pick_context_valid = false;

            for (size_t i = 0; i < active_panels.size() && i < panels.size(); ++i) {
                if (!active_panels[i].valid()) {
                    continue;
                }
                panels[panel_count++] = active_panels[i];
            }

            for (size_t i = panel_count; i < panels.size(); ++i) {
                panels[i] = {};
            }

            pick_context_valid = panel_count > 0;
        }
    };

} // namespace lfs::vis
