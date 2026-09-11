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
        RenderingManager& rendering_manager,
        DepthWindowSettingsState before,
        DepthWindowSettingsState after,
        const bool rebase_readout)
        : rendering_manager_(rendering_manager),
          before_(before),
          after_(after),
          rebase_readout_(rebase_readout) {}

    bool DepthWindowSettingsUndoEntry::isExpired() const {
        if (before_.mode_epoch != rendering_manager_.depthWindowModeEpoch()) {
            return true;
        }
        return !rendering_manager_.isIndependentSplitViewActive() && before_.independent_dual_snapshot;
    }

    bool DepthWindowSettingsUndoEntry::apply(const DepthWindowSettingsState& state) {
        // Compare epoch and restore absolute state under one lock to exclude transitions.
        // restore_sync=false preserves the sync flag, which drag entries do not own.
        // The manager derives projection from the panel focused at apply time.
        return rendering_manager_.restoreDepthWindowSnapshotIfEpoch(
            DepthWindowModeSnapshot{
                .panels = state.panels,
                .sync = state.sync,
                .projection = state.projection,
                .mode_epoch = state.mode_epoch,
                .independent_dual = state.independent_dual_snapshot,
            },
            state.mode_epoch,
            /*restore_sync=*/false);
    }

    void DepthWindowSettingsUndoEntry::undo() {
        if (!apply(before_)) {
            return;
        }
        if (rebase_readout_) {
            publish_depth_window_draw_commit(before_.panel);
        }
    }

    void DepthWindowSettingsUndoEntry::redo() {
        if (!apply(after_)) {
            return;
        }
        if (rebase_readout_) {
            publish_depth_window_draw_commit(after_.panel);
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

    DepthWindowSyncUndoEntry::DepthWindowSyncUndoEntry(
        RenderingManager& rendering_manager,
        DepthWindowModeSnapshot before,
        DepthWindowModeSnapshot after)
        : rendering_manager_(rendering_manager),
          before_(before),
          after_(after) {}

    bool DepthWindowSyncUndoEntry::isExpired() const {
        return before_.mode_epoch != rendering_manager_.depthWindowModeEpoch();
    }

    bool DepthWindowSyncUndoEntry::apply(const DepthWindowModeSnapshot& snapshot) {
        // Sync undo also owns the flag; check epoch and restore under one lock.
        if (!rendering_manager_.restoreDepthWindowSnapshotIfEpoch(
                snapshot, snapshot.mode_epoch, /*restore_sync=*/true)) {
            // An expired restore changes nothing and stamps no lineage.
            return false;
        }
        // Stamp here because the shared restore also serves drag undo, which preserves
        // slot-derived references and must not stamp lineage. ProjectRestore means
        // "fresh baseline required": restoring both possibly unequal absolute windows
        // invalidates all cached panel references, even without a project load.
        rendering_manager_.stampDepthWindowSyncRestoreLineage();
        return true;
    }

    void DepthWindowSyncUndoEntry::undo() {
        (void)apply(before_);
    }

    void DepthWindowSyncUndoEntry::redo() {
        (void)apply(after_);
    }

    UndoMetadata DepthWindowSyncUndoEntry::metadata() const {
        return {
            .id = "selection.depth_window_sync",
            .label = isExpired()
                         ? LOC(lichtfeld::Strings::Selection::HISTORY_DEPTH_WINDOW_EXPIRED)
                         : LOC(lichtfeld::Strings::Selection::HISTORY_DEPTH_WINDOW_SYNC),
            .source = "core",
            .scope = "selection",
        };
    }

} // namespace lfs::vis::op
