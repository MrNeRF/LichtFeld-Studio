/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <thread>

namespace lfs::vis {

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
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Releases a viewer arena frame. With a turn given, the next window is
    // reserved before the release and training gets that many frames first;
    // the reservation lapses on its own once the viewer stops asking.
    template <typename Arena, typename Token>
    void releaseViewerArenaFrame(Arena& arena, const std::uint64_t frame_id, Token* const handoff_token,
                                 const std::optional<std::uint32_t> training_frames_before_next_render) {
        if (training_frames_before_next_render && handoff_token) {
            *handoff_token = arena.request_render_handoff(*handoff_token, *training_frames_before_next_render);
        }
        arena.end_frame(frame_id, true);
    }

} // namespace lfs::vis
