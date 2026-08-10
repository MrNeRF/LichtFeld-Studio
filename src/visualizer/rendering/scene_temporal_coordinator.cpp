/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/scene_temporal_coordinator.hpp"

namespace lfs::vis {

    PreparedSceneTemporalFrame SceneTemporalCoordinator::prepare(
        const SceneTemporalRequest& request) {
        PreparedSceneTemporalFrame prepared{
            .view = request.view,
            .plan = makeSceneTemporalPlan(
                request.requirements, request.render_extent, request.output_extent),
        };
        if (!prepared.plan.valid() || !prepared.plan.temporal()) {
            reset(request.view);
            return prepared;
        }

        prepared.frame = frames_.prepare(request.view, request.frame);
        prepared.history = histories_.prepare(request.view, prepared.plan, prepared.frame);
        return prepared;
    }

    bool SceneTemporalCoordinator::commit(const SceneTemporalRequest& request,
                                          const PreparedSceneTemporalFrame& prepared,
                                          const SceneHistoryStorage color_storage,
                                          const SceneHistoryStorage depth_storage) {
        if (prepared.view != request.view || !prepared.temporal() ||
            prepared.plan.requirements != request.requirements ||
            prepared.plan.render_extent != request.render_extent ||
            prepared.plan.output_extent != request.output_extent) {
            reset(request.view, TemporalResetReason::InvalidInput);
            return false;
        }

        bool history_ready = true;
        if (prepared.plan.needsHistoryColor()) {
            history_ready = histories_.commit(
                request.view, prepared.plan, prepared.frame, color_storage, depth_storage);
        }
        frames_.commit(request.view, request.frame);
        return history_ready;
    }

    void SceneTemporalCoordinator::reset(const TemporalViewId view,
                                         const TemporalResetReason reason) {
        frames_.reset(view, reason);
        histories_.reset(view);
    }

    void SceneTemporalCoordinator::resetAll(const TemporalResetReason reason) {
        frames_.resetAll(reason);
        histories_.resetAll();
    }

} // namespace lfs::vis
