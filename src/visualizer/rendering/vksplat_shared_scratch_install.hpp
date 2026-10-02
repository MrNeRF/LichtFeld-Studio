/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <thread>

namespace lfs::vis {

    // Both tenants share one scratch. Budget the viewer's GPU turn plus the
    // CPU recording that holds the arena, so training retains its chosen share.
    // Navigation retains its one-step minimum for inexpensive frames; idle
    // preview cadence adapts to the full measured cost of every viewer turn.
    inline constexpr double kNavigationTrainingShare = 0.30;
    inline constexpr double kIdlePreviewTrainingShare = 0.89;
    // Above this a full-quality frame cannot be interactive; while the camera
    // moves the viewport keeps the last complete frame and renders once the
    // camera rests, so training stalls once per rest instead of once per frame.
    inline constexpr double kInteractiveViewerBudgetMs = 500.0;
    inline constexpr std::chrono::milliseconds kCameraSettle{150};

    // Training steps the viewer owes after a navigation frame so that training
    // keeps at least kNavigationTrainingShare of the GPU: k*T / (k*T + V) >= s.
    [[nodiscard]] inline std::uint32_t trainingTurnsPerViewerFrame(const double viewer_turn_ms,
                                                                   const double training_step_ms,
                                                                   const std::uint32_t minimum) {
        if (!(viewer_turn_ms > 0.0) || !(training_step_ms > 0.0)) {
            return minimum;
        }
        const double share = kNavigationTrainingShare / (1.0 - kNavigationTrainingShare);
        const double turns = std::ceil(viewer_turn_ms * share / training_step_ms);
        return std::clamp<std::uint32_t>(
            static_cast<std::uint32_t>(std::min(turns, 64.0)), minimum, 64u);
    }

    // Idle rest interval keeping training near kIdlePreviewTrainingShare while
    // allowing full-quality previews at up to roughly one every 9 viewer turns.
    // A submit-to-submit clock must also include the viewer turn itself.
    [[nodiscard]] inline double idlePreviewIntervalSec(const double setting_sec, const double viewer_turn_ms) {
        const double share = (1.0 - kIdlePreviewTrainingShare) / kIdlePreviewTrainingShare;
        return std::max(setting_sec, viewer_turn_ms * 1e-3 / share);
    }

    [[nodiscard]] inline bool navigationRendersOnlyAtRest(const double viewer_turn_ms) {
        return viewer_turn_ms > kInteractiveViewerBudgetMs;
    }

    // Called for a retained viewer block BEFORE capacity reuse or growth. B3
    // can detach the arena backing without updating the renderer's cached flag.
    // Keep the ownership decision independent of Vulkan import/allocation work.
    template <typename IsInstalled, typename TryInstall>
    [[nodiscard]] bool ensureRetainedSharedScratchInstalled(
        bool& installed, IsInstalled&& is_installed, TryInstall&& try_install) {
        installed = is_installed();
        if (!installed) {
            if (!try_install()) {
                return false;
            }
            installed = true;
        }
        return true;
    }

    // While the camera moves during training, training runs one step after every
    // viewer frame: frames keep a steady cadence and training keeps progressing.
    inline constexpr std::uint32_t kTrainingFramesPerNavigationRender = 1;

    // A navigation frame waits this long for training's step to leave the shared
    // scratch; presenting the previous splat image would put it under overlays
    // that already follow the new camera.
    inline constexpr std::chrono::milliseconds kNavigationArenaWait{250};
    // How long a navigation frame lets training start the step it is owed. A
    // trainer waiting for its next step starts within microseconds of the
    // release; one still busy elsewhere must not hold the viewer back.
    inline constexpr std::chrono::milliseconds kNavigationTrainingGrace{2};

    // Reserves the arena for the viewer and waits until a render could claim it
    // without waiting, or the timeout passes. Training first gets the frames the
    // reservation owes it, unless it does not start them within the grace period.
    // Takes no lock of its own.
    template <typename Arena, typename Token>
    [[nodiscard]] bool waitForViewerArenaWindow(Arena& arena, Token& token, const std::chrono::milliseconds timeout,
                                                const std::chrono::milliseconds training_grace) {
        const auto start = std::chrono::steady_clock::now();
        const auto deadline = start + timeout;
        const auto grace_end = start + training_grace;
        while (true) {
            token = arena.request_render_handoff(token);
            if (token == 0) {
                return false;
            }
            const auto now = std::chrono::steady_clock::now();
            if (now >= grace_end) {
                arena.withdraw_render_handoff_training_frames(token);
            }
            if (!arena.render_handoff_owes_training(token) && arena.render_frame_ready(token)) {
                return true;
            }
            if (now >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    // The measured next request replaces the standing reservation once a
    // navigation cadence is known. The short lease covers its first frame.
    template <typename Arena, typename Token>
    void releaseViewerArenaFrame(Arena& arena, const std::uint64_t frame_id, Token* const handoff_token,
                                 const std::optional<std::uint32_t> training_frames_before_next_render) {
        if (training_frames_before_next_render) {
            arena.owe_training_frames(*training_frames_before_next_render);
            if (handoff_token) {
                *handoff_token = arena.reserve_next_viewer_turn(*handoff_token);
            }
        }
        arena.end_frame(frame_id, true);
    }

} // namespace lfs::vis
