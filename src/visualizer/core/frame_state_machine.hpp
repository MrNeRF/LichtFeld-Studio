/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "window/renderer_terminal_state.hpp"
#include <core/export.hpp>

#include <cstdint>

namespace lfs::vis {

    // Classified frame-loop fault (input; classification happens in
    // VisualizerImpl::handleFrameException).
    enum class FrameFault : std::uint8_t {
        OomPressure,      // lfs::core::MemoryAllocationError with CUDA still available
        RendererInternal, // any other std::exception / lfs::Exception / unknown
        DeviceLost,       // lfs::Exception carrying ErrorCode::DeviceLost
    };

    // Pure, single-threaded policy core for the pinned frame outcomes 3-5.
    // Owned by VisualizerImpl and touched ONLY on the main/UI thread (frame
    // callbacks and modal-button drains both run there), so it carries no
    // synchronization. Headless-testable: no GPU, no bus, no clock.
    class LFS_VIS_API FrameStateMachine {
    public:
        struct Limits {
            int oom_retry_budget = 8;      // consecutive OOM frames with reclaim+retry
            int internal_retry_budget = 3; // consecutive non-OOM faulting frames tolerated
        };

        enum class State : std::uint8_t {
            Healthy,
            PressureRetry,  // consecutive OOM faults, budget not yet exhausted
            SceneSuspended, // scene render halted (OOM budget exhausted or Internal
                            // terminal); last-known-good frame held; Retry can resume
            RendererDead,   // DeviceLost / wait-quarantine; terminal for the session
        };

        struct Effects {
            bool run_reclaim_episode = false;         // run one MemoryPressureCoordinator episode
            bool publish_pressure_toast = false;      // first OOM fault of an episode only
            bool publish_oom_modal = false;           // budget exhausted (one-shot)
            bool publish_internal_modal = false;      // internal budget exhausted (one-shot)
            bool publish_renderer_dead_modal = false; // one-shot
            RendererTerminalState dead_cause = RendererTerminalState::Running;
        };

        FrameStateMachine() noexcept = default;
        explicit FrameStateMachine(Limits limits) noexcept;

        [[nodiscard]] Effects on_fault(FrameFault fault) noexcept;
        [[nodiscard]] Effects on_renderer_terminal(RendererTerminalState state) noexcept;
        void on_frame_success() noexcept;
        void on_retry_action() noexcept;
        void on_stop_renderer_action() noexcept;

        [[nodiscard]] State state() const noexcept { return state_; }
        [[nodiscard]] bool scene_render_suspended() const noexcept {
            return state_ == State::SceneSuspended || state_ == State::RendererDead;
        }
        // Current consecutive OOM-fault count; the reclaim/retry attempt number the
        // frame handler logs (§1.5.2 "sourced from the machine").
        [[nodiscard]] int consecutive_oom_faults() const noexcept { return consecutive_oom_; }

    private:
        [[nodiscard]] Effects enter_renderer_dead(RendererTerminalState cause) noexcept;

        Limits limits_{};
        State state_ = State::Healthy;
        int consecutive_oom_ = 0;
        int consecutive_internal_ = 0;
        bool suspend_modal_armed_ = true; // one-shot for OOM/Internal suspend modals
        bool dead_modal_armed_ = true;    // independent one-shot for the renderer-dead modal
        bool renderer_stopped_ = false;   // StopRenderer sticky: Retry can no longer resume
        RendererTerminalState dead_cause_ = RendererTerminalState::Running;
    };

} // namespace lfs::vis
