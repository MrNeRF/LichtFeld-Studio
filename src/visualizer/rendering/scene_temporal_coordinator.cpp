/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/scene_temporal_coordinator.hpp"

namespace lfs::vis {

    std::uint64_t SceneTemporalCoordinator::nextTicket() noexcept {
        const std::uint64_t ticket = next_ticket_++;
        if (next_ticket_ == 0) {
            next_ticket_ = 1;
        }
        return ticket;
    }

    PreparedSceneTemporalFrame SceneTemporalCoordinator::prepare(
        const SceneTemporalRequest& request) {
        auto& pending = pending_.at(index(request.view));
        pending.reset();

        PreparedSceneTemporalFrame prepared{
            .view = request.view,
            .plan = makeSceneTemporalPlan(
                request.requirements, request.render_extent, request.output_extent),
        };
        if (!prepared.plan.valid() || !prepared.plan.active()) {
            reset(request.view,
                  prepared.plan.valid() ? TemporalResetReason::HistoryDisabled
                                        : TemporalResetReason::InvalidInput);
            return prepared;
        }

        if (!validTemporalFrameInput(request.frame)) {
            reset(request.view, TemporalResetReason::InvalidInput);
            return prepared;
        }
        prepared.frame = frames_.prepare(request.view, request.frame);
        prepared.history = histories_.prepare(request.view, prepared.plan, prepared.frame);
        prepared.ticket = nextTicket();
        pending = PendingFrame{
            .ticket = prepared.ticket,
            .input = request.frame,
            .plan = prepared.plan,
            .frame = prepared.frame,
        };
        return prepared;
    }

    bool SceneTemporalCoordinator::commit(const PreparedSceneTemporalFrame& prepared,
                                          const SceneHistoryStorage color_storage,
                                          const SceneHistoryStorage depth_storage) {
        auto& pending = pending_.at(index(prepared.view));
        if (!prepared.active() || !pending || pending->ticket != prepared.ticket) {
            return false;
        }
        if (pending->plan != prepared.plan) {
            reset(prepared.view, TemporalResetReason::InvalidInput);
            return false;
        }

        const PendingFrame transaction = *pending;
        pending.reset();
        if (prepared.plan.requirements.history_color &&
            !histories_.commit(prepared.view,
                               transaction.plan,
                               transaction.frame,
                               color_storage,
                               depth_storage)) {
            reset(prepared.view, TemporalResetReason::InvalidInput);
            return false;
        }
        if (transaction.frame.reset_reasons != TemporalResetReason::None)
            frames_.reset(prepared.view, transaction.frame.reset_reasons);
        frames_.commit(prepared.view, transaction.input);
        return true;
    }

    void SceneTemporalCoordinator::discard(const PreparedSceneTemporalFrame& prepared,
                                           const TemporalResetReason reason) {
        auto& pending = pending_.at(index(prepared.view));
        if (pending && pending->ticket == prepared.ticket) {
            reset(prepared.view, reason);
        }
    }

    void SceneTemporalCoordinator::reset(const TemporalViewId view,
                                         const TemporalResetReason reason) {
        pending_.at(index(view)).reset();
        frames_.reset(view, reason);
        histories_.reset(view);
    }

    void SceneTemporalCoordinator::resetAll(const TemporalResetReason reason) {
        for (auto& pending : pending_) {
            pending.reset();
        }
        frames_.resetAll(reason);
        histories_.resetAll();
    }

} // namespace lfs::vis
