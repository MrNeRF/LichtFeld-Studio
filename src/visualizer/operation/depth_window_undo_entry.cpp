/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "operation/undo_entry.hpp"
#include "rendering/rendering_manager.hpp"
#include "visualizer/app_store.hpp"

namespace lfs::vis::op {

    DepthWindowSettingsUndoEntry::DepthWindowSettingsUndoEntry(
        RenderingManager& rendering_manager,
        DepthWindowSettingsState before,
        DepthWindowSettingsState after,
        const bool rebase_readout)
        : rendering_manager_(rendering_manager),
          before_(before),
          after_(after),
          rebase_readout_(rebase_readout) {}

    void DepthWindowSettingsUndoEntry::apply(const DepthWindowSettingsState& state) {
        auto settings = rendering_manager_.getSettings();
        settings.depth_filter_scale_x = state.scale_x;
        settings.depth_filter_scale_y = state.scale_y;
        settings.depth_filter_offset_x = state.offset_x;
        settings.depth_filter_offset_y = state.offset_y;
        rendering_manager_.updateSettings(settings, DirtyFlag::SELECTION);
    }

    void DepthWindowSettingsUndoEntry::undo() {
        apply(before_);
        if (rebase_readout_) {
            publish_depth_window_draw_commit();
        }
    }

    void DepthWindowSettingsUndoEntry::redo() {
        apply(after_);
        if (rebase_readout_) {
            publish_depth_window_draw_commit();
        }
    }

    UndoMetadata DepthWindowSettingsUndoEntry::metadata() const {
        return {
            .id = "selection.depth_window_drag",
            .label = "Drag Depth Window",
            .source = "core",
            .scope = "selection",
        };
    }

} // namespace lfs::vis::op
