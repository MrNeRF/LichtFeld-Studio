/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/frame_state_machine.hpp"

namespace lfs::vis {

    FrameStateMachine::FrameStateMachine(const Limits limits) noexcept
        : limits_(limits) {}

    FrameStateMachine::Effects
    FrameStateMachine::enter_renderer_dead(const RendererTerminalState cause) noexcept {
        state_ = State::RendererDead;
        dead_cause_ = cause;
        Effects fx;
        fx.dead_cause = dead_cause_;
        if (dead_modal_armed_) {
            fx.publish_renderer_dead_modal = true;
            dead_modal_armed_ = false;
        }
        return fx;
    }

    FrameStateMachine::Effects FrameStateMachine::on_fault(const FrameFault fault) noexcept {
        if (state_ == State::RendererDead) {
            return {}; // T12
        }
        if (fault == FrameFault::DeviceLost) {
            return enter_renderer_dead(RendererTerminalState::DeviceLost); // T6
        }
        if (state_ == State::SceneSuspended) {
            return {}; // T11: no reclaim, no publish while suspended
        }

        Effects fx;
        if (fault == FrameFault::OomPressure) {
            ++consecutive_oom_;
            if (consecutive_oom_ <= limits_.oom_retry_budget) {
                fx.run_reclaim_episode = true; // T1/T2
                if (consecutive_oom_ == 1) {
                    fx.publish_pressure_toast = true; // first fault of the episode
                }
                state_ = State::PressureRetry;
            } else {
                if (suspend_modal_armed_) {
                    fx.publish_oom_modal = true; // T3, one-shot
                    suspend_modal_armed_ = false;
                }
                state_ = State::SceneSuspended;
            }
        } else {
            ++consecutive_internal_;
            if (consecutive_internal_ >= limits_.internal_retry_budget) {
                if (suspend_modal_armed_) {
                    fx.publish_internal_modal = true; // T5, one-shot
                    suspend_modal_armed_ = false;
                }
                state_ = State::SceneSuspended;
            }
            // T4: below budget — no effects, state unchanged (caller keeps its log).
        }
        return fx;
    }

    FrameStateMachine::Effects
    FrameStateMachine::on_renderer_terminal(const RendererTerminalState state) noexcept {
        if (state == RendererTerminalState::Running) {
            return {}; // T13
        }
        if (state_ == State::RendererDead) {
            return {}; // T12
        }
        return enter_renderer_dead(state); // T7
    }

    void FrameStateMachine::on_frame_success() noexcept {
        if (state_ == State::Healthy || state_ == State::PressureRetry) {
            consecutive_oom_ = 0;
            consecutive_internal_ = 0;
            state_ = State::Healthy; // T8
        }
    }

    void FrameStateMachine::on_retry_action() noexcept {
        if (state_ == State::SceneSuspended && !renderer_stopped_) {
            state_ = State::Healthy; // T9
            consecutive_oom_ = 0;
            consecutive_internal_ = 0;
            suspend_modal_armed_ = true; // re-arm exactly one further modal
        }
    }

    void FrameStateMachine::on_stop_renderer_action() noexcept {
        if (state_ == State::SceneSuspended) {
            renderer_stopped_ = true;     // T10, session-sticky
            suspend_modal_armed_ = false; // permanently disarmed
        }
    }

} // namespace lfs::vis
