/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string_view>
#include <thread>

namespace lfs::vis {

    // Training lifecycle states
    enum class TrainingState : uint8_t {
        Idle,     // No dataset loaded, no trainer
        Ready,    // Dataset loaded, trainer initialized, can start
        Running,  // Training thread active
        Paused,   // Training paused, can resume or stop
        Stopping, // Stop requested, waiting for thread
        Finished  // Training completed or errored, model preserved
    };

    // Actions that can be performed on the training system
    enum class TrainingAction : uint8_t {
        LoadDataset,
        LoadCheckpoint,
        Start,
        Pause,
        Resume,
        Stop,
        Reset,
        ClearScene,
        DeleteTrainingNode,
        SaveCheckpoint,
        COUNT
    };

    // Result of a finished training session
    enum class FinishReason : uint8_t {
        None,        // Not finished
        Completed,   // Reached max iterations
        UserStopped, // User requested stop
        Error        // Error occurred
    };

    // State machine configuration
    class LFS_VIS_API TrainingStateMachine {
    public:
        // State transition callback signatures
        using StateChangeCallback = std::function<void(TrainingState old_state, TrainingState new_state)>;

        TrainingStateMachine();

        // State queries
        [[nodiscard]] TrainingState getState() const { return state_.load(std::memory_order_acquire); }
        [[nodiscard]] bool isInState(TrainingState state) const { return getState() == state; }
        [[nodiscard]] bool isActive() const; // Running or Paused
        [[nodiscard]] FinishReason getFinishReason() const;

        // Action permission checks - call BEFORE attempting action
        [[nodiscard]] bool canPerform(TrainingAction action) const;
        [[nodiscard]] std::string_view getActionBlockedReason(TrainingAction action) const;

        // State transitions - return false if transition not allowed
        [[nodiscard]] bool transitionTo(TrainingState new_state);
        [[nodiscard]] bool transitionToFinished(FinishReason reason);

        // Callbacks
        void setStateChangeCallback(StateChangeCallback callback);

        // Utility
        [[nodiscard]] static std::string_view stateName(TrainingState state);
        [[nodiscard]] static std::string_view actionName(TrainingAction action);

    private:
        [[nodiscard]] bool isValidTransition(TrainingState from, TrainingState to) const;
        [[nodiscard]] bool transitionToImpl(TrainingState new_state, FinishReason finish_reason);
        void finishCallbackDispatch() noexcept;

        std::atomic<TrainingState> state_{TrainingState::Idle};
        mutable std::mutex mutex_;
        std::condition_variable callback_dispatch_idle_;
        std::thread::id callback_dispatch_owner_;
        bool callback_dispatch_active_ = false;
        FinishReason finish_reason_{FinishReason::None};

        StateChangeCallback on_state_change_;

        // Transition table: [from][to] = allowed
        static constexpr size_t STATE_COUNT = 6;
        static constexpr std::array<std::array<bool, STATE_COUNT>, STATE_COUNT> TRANSITIONS = {{
            // To:    Idle   Ready  Running Paused Stopping Finished
            /* Idle */ {false, true, false, true, false, false}, // Paused: checkpoint load
            /* Ready */ {true, false, true, false, false, false},
            /* Running */ {false, false, false, true, true, false},
            /* Paused */ {false, true, true, false, true, false},
            /* Stopping */ {true, false, false, false, false, true},
            /* Finished */ {true, true, false, false, false, false},
        }};

        // Action permission table: [state][action] = allowed
        static constexpr size_t ACTION_COUNT = static_cast<size_t>(TrainingAction::COUNT);
        static constexpr std::array<std::array<bool, ACTION_COUNT>, STATE_COUNT> PERMISSIONS = {{
            //              Load   LoadCk Start  Pause  Resume Stop   Reset  Clear  DelNode SaveCk
            /* Idle */ {true, true, false, false, false, false, false, true, true, false},
            /* Ready */ {true, true, true, false, false, false, true, true, true, false},
            /* Running */ {false, false, false, true, false, true, false, false, false, true},
            /* Paused */ {true, true, false, false, true, true, true, true, false, true},
            /* Stopping */ {false, false, false, false, false, false, false, false, false, false},
            /* Finished */ {true, true, false, false, false, false, true, true, false, false},
        }};
    };

} // namespace lfs::vis
