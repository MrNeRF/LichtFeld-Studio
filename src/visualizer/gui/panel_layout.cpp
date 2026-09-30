/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panel_layout.hpp"
#include "visualizer/app_store.hpp"

namespace lfs::vis::gui {

    PanelLayoutManager::PanelLayoutManager() = default;

    void PanelLayoutManager::setShowSequencer(const bool visible) {
        if (show_sequencer_ == visible)
            return;
        show_sequencer_ = visible;
        lfs::vis::publish_viewport_toolbar_generation();
    }

    void PanelLayoutManager::loadState() {
        setShowSequencer(false);
    }

    PanelLayoutProjectState PanelLayoutManager::captureProjectState() const {
        return PanelLayoutProjectState{.show_sequencer = show_sequencer_, .active_tab_id = {}};
    }

    void PanelLayoutManager::applyProjectState(const PanelLayoutProjectState& state) {
        setShowSequencer(state.show_sequencer);
    }

    ViewportLayout PanelLayoutManager::computeViewportLayout(bool, bool, bool,
                                                             const ScreenState& screen) const {
        ViewportLayout layout;
        layout.pos = screen.work_pos;
        layout.size = screen.work_size;
        layout.has_focus = !screen.any_item_active;
        return layout;
    }

} // namespace lfs::vis::gui
