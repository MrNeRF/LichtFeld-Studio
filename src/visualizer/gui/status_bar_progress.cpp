/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/status_bar_progress.hpp"
#include <algorithm>
#include <format>

namespace lfs::vis::gui {
    namespace {
        void appendProgressMarkerRml(std::string& markers,
                                     const size_t step,
                                     const int total_iterations,
                                     const bool past,
                                     const bool hovered,
                                     const bool preview,
                                     const ProgressMarkerStyle style) {
            const float left_pct = std::clamp(
                100.0f * static_cast<float>(step) / static_cast<float>(total_iterations),
                0.5f,
                99.5f);

            std::string classes = "progress-marker";
            if (past)
                classes += " is-past";
            if (hovered)
                classes += " is-hovered";
            if (preview)
                classes += " is-preview";

            if (style == ProgressMarkerStyle::Miner) {
                markers += std::format(
                    "<img class=\"{}\" src=\"../icon/mining/{}\" style=\"left:{:.3f}%;\"/>",
                    classes,
                    past ? "gem-collected.png" : "gem.png",
                    left_pct);
            } else if (style == ProgressMarkerStyle::Tessle) {
                markers += std::format(
                    "<img class=\"{}\" src=\"../icon/tessle/cherries.png\" style=\"left:{:.3f}%;\"/>",
                    classes, left_pct);
            } else {
                markers += std::format(
                    "<div class=\"{}\" style=\"left:{:.3f}%;\"></div>",
                    classes,
                    left_pct);
            }
        }

    } // namespace
    std::string buildProgressMarkersRml(std::vector<size_t> save_steps,
                                        const int total_iterations,
                                        const int current_iteration,
                                        const ProgressMarkerRenderState& state,
                                        const ProgressMarkerStyle style) {
        if (total_iterations <= 0)
            return {};

        std::ranges::sort(save_steps);
        save_steps.erase(std::ranges::unique(save_steps).begin(), save_steps.end());

        std::string markers;
        markers.reserve((save_steps.size() + 1) * 136);
        for (const size_t save_step : save_steps) {
            if (save_step == 0 || save_step > static_cast<size_t>(total_iterations))
                continue;
            if (state.dragging && !state.adding && save_step == state.original_step)
                continue;

            appendProgressMarkerRml(markers,
                                    save_step,
                                    total_iterations,
                                    save_step <= static_cast<size_t>(std::max(0, current_iteration)),
                                    !state.dragging && save_step == state.hover_step,
                                    false,
                                    style);
        }

        if (state.dragging && state.preview_step > 0 &&
            state.preview_step <= static_cast<size_t>(total_iterations)) {
            appendProgressMarkerRml(markers,
                                    state.preview_step,
                                    total_iterations,
                                    false,
                                    true,
                                    true,
                                    style);
        }
        return markers;
    }

} // namespace lfs::vis::gui
