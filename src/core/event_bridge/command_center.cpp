/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/command_api.hpp"
#include "core/events.hpp"

namespace lfs::training {

    CommandCenter& CommandCenter::instance() {
        // lfs_training is linked into multiple modules, so the singleton storage
        // must stay in this shared library.
        static CommandCenter inst;
        return inst;
    }

    CommandCenter::CommandCenter() {
        bind_state_events();

        ops_.push_back({.name = "set_attribute",
                        .target = CommandTarget::Model,
                        .selectors = {SelectionKind::All, SelectionKind::Range, SelectionKind::Indices},
                        .args = {{"attribute", ArgType::String, true, "Attribute name (means, scaling, rotation, opacity, sh0, shN)"},
                                 {"value", ArgType::Float, false, "Scalar value"},
                                 {"values", ArgType::FloatList, false, "Vector value (broadcast)"}},
                        .description = "Set attribute values for selected splats (scalar or per-dim vector)."});

        ops_.push_back({.name = "scale_attribute",
                        .target = CommandTarget::Model,
                        .selectors = {SelectionKind::All, SelectionKind::Range, SelectionKind::Indices},
                        .args = {{"attribute", ArgType::String, true, "Attribute name"},
                                 {"factor", ArgType::Float, true, "Multiplicative scale"}},
                        .description = "Scale attribute by factor for selected splats."});

        ops_.push_back({.name = "clamp_attribute",
                        .target = CommandTarget::Model,
                        .selectors = {SelectionKind::All, SelectionKind::Range, SelectionKind::Indices},
                        .args = {{"attribute", ArgType::String, true, "Attribute name"},
                                 {"min", ArgType::Float, false, "Optional min"},
                                 {"max", ArgType::Float, false, "Optional max"}},
                        .description = "Clamp attribute values for selected splats."});

        ops_.push_back({.name = "set_lr",
                        .target = CommandTarget::Optimizer,
                        .selectors = {SelectionKind::All},
                        .args = {{"value", ArgType::Float, true, "Learning rate"}},
                        .description = "Set global learning rate."});

        ops_.push_back({.name = "scale_lr",
                        .target = CommandTarget::Optimizer,
                        .selectors = {SelectionKind::All},
                        .args = {{"factor", ArgType::Float, true, "Scale factor"}},
                        .description = "Scale global learning rate."});

        ops_.push_back({.name = "pause",
                        .target = CommandTarget::Session,
                        .selectors = {SelectionKind::All},
                        .args = {},
                        .description = "Request pause."});

        ops_.push_back({.name = "resume",
                        .target = CommandTarget::Session,
                        .selectors = {SelectionKind::All},
                        .args = {},
                        .description = "Request resume."});

        ops_.push_back({.name = "request_stop",
                        .target = CommandTarget::Session,
                        .selectors = {SelectionKind::All},
                        .args = {},
                        .description = "Request graceful stop."});

        // Mutable fields
        mutable_fields_.push_back({"means", CommandTarget::Model, "[N,3]", "Gaussian means", true});
        mutable_fields_.push_back({"scaling", CommandTarget::Model, "[N,3]", "Log scaling", true});
        mutable_fields_.push_back({"rotation", CommandTarget::Model, "[N,4]", "Quaternion rotation", true});
        mutable_fields_.push_back({"opacity", CommandTarget::Model, "[N]", "Opacity logits", true});
        mutable_fields_.push_back({"sh0", CommandTarget::Model, "[N,3]", "SH0 coefficients", true});
        mutable_fields_.push_back({"shN", CommandTarget::Model, "[N,?]", "Higher-order SH coefficients", true});
    }

    void CommandCenter::bind_state_events() {
        using lfs::core::events::state::TrainingPaused;
        auto& bridge = lfs::event::EventBridge::instance();
        if (training_paused_handler_id_) {
            bridge.unsubscribe(typeid(TrainingPaused), *training_paused_handler_id_);
            training_paused_handler_id_.reset();
        }
        training_paused_handler_id_ = TrainingPaused::when(
            [this](const TrainingPaused& event) {
                std::lock_guard<std::mutex> lock(mutex_);
                snapshot_.iteration = event.iteration;
                snapshot_.is_paused = true;
                snapshot_.is_running = false;
            });
    }

    void CommandCenter::overlay_stored_session(std::string strategy, bool hydrated) {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_.strategy = std::move(strategy);
        snapshot_.session_hydrated = hydrated;
    }

    void CommandCenter::reset_snapshot_locked() {
        snapshot_ = {};
        pending_commands_.clear();
        phase_.store(TrainingPhase::Idle, std::memory_order_relaxed);
    }

    void CommandCenter::clear_snapshot(const Trainer* trainer) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (snapshot_.trainer != trainer) {
            return;
        }
        reset_snapshot_locked();
    }

    void CommandCenter::reset_snapshot() {
        std::lock_guard<std::mutex> lock(mutex_);
        reset_snapshot_locked();
    }

    TrainingSnapshot CommandCenter::snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        TrainingSnapshot snap = snapshot_;
        snap.phase = phase_.load(std::memory_order_relaxed);
        return snap;
    }

    std::vector<LossHistoryPoint> CommandCenter::loss_history() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return loss_history_;
    }

    void CommandCenter::clear_loss_history() {
        std::lock_guard<std::mutex> lock(mutex_);
        loss_history_.clear();
        last_recorded_iteration_ = -1;
    }

    void CommandCenter::replace_loss_history(
        std::vector<LossHistoryPoint> history) {
        std::lock_guard<std::mutex> lock(mutex_);
        loss_history_ = std::move(history);
        last_recorded_iteration_ =
            loss_history_.empty()
                ? -1
                : loss_history_.back().iteration;
    }

    std::vector<OperationInfo> CommandCenter::operations(std::optional<CommandTarget> target) const {
        if (!target) {
            return ops_;
        }
        std::vector<OperationInfo> filtered;
        for (const auto& op : ops_) {
            if (op.target == *target) {
                filtered.push_back(op);
            }
        }
        return filtered;
    }

    std::vector<MutableFieldInfo> CommandCenter::mutables(std::optional<CommandTarget> target) const {
        if (!target) {
            return mutable_fields_;
        }
        std::vector<MutableFieldInfo> filtered;
        for (const auto& f : mutable_fields_) {
            if (f.target == *target) {
                filtered.push_back(f);
            }
        }
        return filtered;
    }

#if !LFS_BUILD_TRAINER
    void CommandCenter::set_phase(TrainingPhase phase) { phase_.store(phase, std::memory_order_relaxed); }
    std::expected<void, std::string> CommandCenter::execute(const Command&) {
        return std::unexpected("Training is not included in this build");
    }
#endif

} // namespace lfs::training

#if !LFS_BUILD_TRAINER

namespace lfs::training {
    void CommandCenter::update_snapshot(const HookContext& ctx, int max_iterations, bool is_paused, bool is_running, bool stop_requested, TrainingPhase phase) {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_.iteration = ctx.iteration;
        snapshot_.max_iterations = max_iterations;
        snapshot_.loss = ctx.loss;
        snapshot_.num_gaussians = ctx.num_gaussians;
        snapshot_.is_refining = ctx.is_refining;
        snapshot_.trainer = ctx.trainer;
        snapshot_.is_paused = is_paused;
        snapshot_.is_running = is_running;
        snapshot_.stop_requested = stop_requested;
        snapshot_.phase = phase;
        if (ctx.iteration > last_recorded_iteration_ && ctx.loss > 0.0f) {
            loss_history_.push_back({ctx.iteration, ctx.loss});
            last_recorded_iteration_ = ctx.iteration;
        }
    }

} // namespace lfs::training
#endif
