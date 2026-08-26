/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/scene_temporal_plan.hpp"

namespace lfs::vis {

    SceneHistoryContract SceneHistoryTracker::prepare(
        const TemporalViewId view,
        const SceneTemporalPlan& plan,
        const TemporalFrameState& frame) const {
        if (!plan.valid() || !plan.requirements.history_color || !frame.history_valid) {
            return {};
        }
        const auto& history = entries_.at(index(view));
        if (!history || !history->matches(plan) || history->sequence != frame.sequence) {
            return {};
        }
        return *history;
    }

    bool SceneHistoryTracker::commit(const TemporalViewId view,
                                     const SceneTemporalPlan& plan,
                                     const TemporalFrameState& frame,
                                     const SceneHistoryStorage color_storage,
                                     const SceneHistoryStorage depth_storage) {
        auto& history = entries_.at(index(view));
        if (!plan.valid() || !plan.requirements.history_color ||
            color_storage == SceneHistoryStorage::None ||
            (plan.requirements.history_depth && depth_storage == SceneHistoryStorage::None)) {
            history.reset();
            return false;
        }

        history = SceneHistoryContract{
            .color_storage = color_storage,
            .depth_storage = plan.requirements.history_depth ? depth_storage
                                                             : SceneHistoryStorage::None,
            .color_extent = plan.output_extent,
            .depth_extent = plan.requirements.history_depth ? plan.render_extent : glm::ivec2(0),
            .sequence = frame.sequence + 1,
        };
        return history->valid();
    }

    void SceneHistoryTracker::reset(const TemporalViewId view) {
        entries_.at(index(view)).reset();
    }

    void SceneHistoryTracker::resetAll() {
        for (auto& history : entries_) {
            history.reset();
        }
    }

} // namespace lfs::vis
