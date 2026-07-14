/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training_state.hpp"
#include "core/logger.hpp"

namespace lfs::vis {

    TrainingStateMachine::TrainingStateMachine() = default;

    bool TrainingStateMachine::isActive() const {
        const auto s = getState();
        return s == TrainingState::Running || s == TrainingState::Paused;
    }

    FinishReason TrainingStateMachine::getFinishReason() const {
        std::lock_guard lock(mutex_);
        return finish_reason_;
    }

    bool TrainingStateMachine::canPerform(TrainingAction action) const {
        const auto state_idx = static_cast<size_t>(getState());
        const auto action_idx = static_cast<size_t>(action);

        if (action_idx >= ACTION_COUNT)
            return false;
        return PERMISSIONS[state_idx][action_idx];
    }

    std::string_view TrainingStateMachine::getActionBlockedReason(TrainingAction action) const {
        if (canPerform(action))
            return "";

        const auto state = getState();

        switch (action) {
        case TrainingAction::LoadDataset:
        case TrainingAction::LoadCheckpoint:
            if (state == TrainingState::Running)
                return "Cannot load while training is running. Pause or stop first.";
            if (state == TrainingState::Stopping)
                return "Cannot load while training is stopping. Wait for completion.";
            break;

        case TrainingAction::Start:
            if (state == TrainingState::Idle)
                return "No dataset loaded. Load a dataset first.";
            if (state == TrainingState::Running)
                return "Training is already running.";
            if (state == TrainingState::Paused)
                return "Training is paused. Use resume instead.";
            if (state == TrainingState::Finished)
                return "Training finished. Reset to train again.";
            break;

        case TrainingAction::Pause:
            if (state != TrainingState::Running)
                return "Can only pause while training is running.";
            break;

        case TrainingAction::Resume:
            if (state != TrainingState::Paused)
                return "Can only resume from paused state.";
            break;

        case TrainingAction::Stop:
            if (!isActive())
                return "Training is not active.";
            break;

        case TrainingAction::Reset:
            if (state == TrainingState::Running)
                return "Cannot reset while training is running. Stop first.";
            if (state == TrainingState::Idle)
                return "Nothing to reset.";
            break;

        case TrainingAction::ClearScene:
        case TrainingAction::DeleteTrainingNode:
            if (state == TrainingState::Running)
                return "Cannot modify scene while training is running.";
            if (state == TrainingState::Stopping)
                return "Cannot modify scene while training is stopping.";
            break;

        case TrainingAction::SaveCheckpoint:
            if (!isActive())
                return "Can only save checkpoint during active training.";
            break;

        default:
            break;
        }

        return "Action not allowed in current state.";
    }

    bool TrainingStateMachine::transitionTo(TrainingState new_state) {
        return transitionToImpl(new_state, FinishReason::None);
    }

    bool TrainingStateMachine::transitionToFinished(FinishReason reason) {
        return transitionToImpl(TrainingState::Finished, reason);
    }

    bool TrainingStateMachine::transitionToImpl(TrainingState new_state, FinishReason finish_reason) {
        StateChangeCallback callback;
        TrainingState old_state;
        bool owns_callback_dispatch = false;
        const auto current_thread = std::this_thread::get_id();
        {
            std::unique_lock lock(mutex_);
            callback_dispatch_idle_.wait(lock, [this, current_thread] {
                return !callback_dispatch_active_ || callback_dispatch_owner_ == current_thread;
            });
            old_state = getState();

            if (!isValidTransition(old_state, new_state)) {
                LOG_WARN("Invalid state transition: {} -> {}",
                         stateName(old_state), stateName(new_state));
                return false;
            }

            LOG_DEBUG("Training state: {} -> {}", stateName(old_state), stateName(new_state));

            finish_reason_ = new_state == TrainingState::Finished ? finish_reason : FinishReason::None;
            state_.store(new_state, std::memory_order_release);
            callback = on_state_change_;
            if (callback && !callback_dispatch_active_) {
                callback_dispatch_active_ = true;
                callback_dispatch_owner_ = current_thread;
                owns_callback_dispatch = true;
            }
        }

        if (callback) {
            try {
                callback(old_state, new_state);
            } catch (...) {
                if (owns_callback_dispatch) {
                    finishCallbackDispatch();
                }
                throw;
            }
        }
        if (owns_callback_dispatch) {
            finishCallbackDispatch();
        }

        return true;
    }

    void TrainingStateMachine::setStateChangeCallback(StateChangeCallback callback) {
        const auto current_thread = std::this_thread::get_id();
        std::unique_lock lock(mutex_);
        callback_dispatch_idle_.wait(lock, [this, current_thread] {
            return !callback_dispatch_active_ || callback_dispatch_owner_ == current_thread;
        });
        on_state_change_ = std::move(callback);
    }

    void TrainingStateMachine::finishCallbackDispatch() noexcept {
        {
            std::lock_guard lock(mutex_);
            callback_dispatch_active_ = false;
            callback_dispatch_owner_ = {};
        }
        callback_dispatch_idle_.notify_all();
    }

    bool TrainingStateMachine::isValidTransition(TrainingState from, TrainingState to) const {
        const auto from_idx = static_cast<size_t>(from);
        const auto to_idx = static_cast<size_t>(to);

        if (from_idx >= STATE_COUNT || to_idx >= STATE_COUNT)
            return false;
        return TRANSITIONS[from_idx][to_idx];
    }

    std::string_view TrainingStateMachine::stateName(TrainingState state) {
        switch (state) {
        case TrainingState::Idle: return "Idle";
        case TrainingState::Ready: return "Ready";
        case TrainingState::Running: return "Running";
        case TrainingState::Paused: return "Paused";
        case TrainingState::Stopping: return "Stopping";
        case TrainingState::Finished: return "Finished";
        }
        return "Unknown";
    }

    std::string_view TrainingStateMachine::actionName(TrainingAction action) {
        switch (action) {
        case TrainingAction::LoadDataset: return "LoadDataset";
        case TrainingAction::LoadCheckpoint: return "LoadCheckpoint";
        case TrainingAction::Start: return "Start";
        case TrainingAction::Pause: return "Pause";
        case TrainingAction::Resume: return "Resume";
        case TrainingAction::Stop: return "Stop";
        case TrainingAction::Reset: return "Reset";
        case TrainingAction::ClearScene: return "ClearScene";
        case TrainingAction::DeleteTrainingNode: return "DeleteTrainingNode";
        case TrainingAction::SaveCheckpoint: return "SaveCheckpoint";
        case TrainingAction::COUNT: return "Invalid";
        }
        return "Unknown";
    }

} // namespace lfs::vis
