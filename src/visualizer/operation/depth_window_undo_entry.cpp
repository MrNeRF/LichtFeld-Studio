/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/localization_manager.hpp"
#include "gui/string_keys.hpp"
#include "operation/undo_entry.hpp"
#include "rendering/rendering_manager.hpp"
#include "visualizer/app_store.hpp"

namespace lfs::vis::op {

    DepthWindowSettingsUndoEntry::DepthWindowSettingsUndoEntry(
        RenderingManager& rendering_manager, DepthWindowModeSnapshot before,
        DepthWindowModeSnapshot after,
        const bool rebase_readout)
        : rendering_manager_(rendering_manager),
          before_(before),
          after_(after),
          rebase_readout_(rebase_readout) {}

    bool DepthWindowSettingsUndoEntry::isExpired() const {
        return !rendering_manager_.depthWindowSnapshotCurrent(before_);
    }

    bool DepthWindowSettingsUndoEntry::apply(const DepthWindowModeSnapshot& state) {
        return rendering_manager_.restoreDepthWindowSnapshotIfEpoch(state,
                                                                    state.mode_epoch);
    }

    void DepthWindowSettingsUndoEntry::undo() {
        if (!apply(before_)) {
            return;
        }
        if (rebase_readout_) {
            publish_depth_window_draw_commit(before_.view, before_.window.scale_x, before_.window.scale_y);
        }
    }

    void DepthWindowSettingsUndoEntry::redo() {
        if (!apply(after_)) {
            return;
        }
        if (rebase_readout_) {
            publish_depth_window_draw_commit(after_.view, after_.window.scale_x, after_.window.scale_y);
        }
    }

    UndoMetadata DepthWindowSettingsUndoEntry::metadata() const {
        return {
            .id = "selection.depth_window_drag",
            .label = isExpired()
                         ? LOC(lichtfeld::Strings::Selection::HISTORY_DEPTH_WINDOW_EXPIRED)
                         : LOC(lichtfeld::Strings::Selection::HISTORY_DEPTH_WINDOW_DRAG),
            .source = "core",
            .scope = "selection",
        };
    }

} // namespace lfs::vis::op
