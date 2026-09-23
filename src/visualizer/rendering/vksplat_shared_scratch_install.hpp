/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>

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

    // Turn-taking on the shared training scratch while the camera moves during
    // training. Training runs one step after every viewer frame. When a step made
    // the viewer wait, the viewer keeps the scratch for as long as it waited
    // (capped), so both get about half of the time and training never stops.
    class NavigationArenaShare {
    public:
        using Clock = std::chrono::steady_clock;
        static constexpr std::chrono::milliseconds kMaxViewerKeep{66};
        // A navigation frame may wait this long for a training step that is about
        // to finish instead of dropping the frame.
        static constexpr std::uint32_t kRenderWaitMs = 5;

        void noteDeclined(const Clock::time_point now) {
            if (!declined_since_) {
                declined_since_ = now;
            }
        }

        void noteBegan(const Clock::time_point now) {
            if (!declined_since_) {
                return;
            }
            const Clock::duration waited = std::min<Clock::duration>(now - *declined_since_, kMaxViewerKeep);
            keep_until_ = std::max(keep_until_, now + waited);
            declined_since_.reset();
        }

        [[nodiscard]] std::uint32_t trainingFramesBeforeNextRender(const Clock::time_point now) const {
            return now < keep_until_ ? 0u : 1u;
        }

        void reset() { *this = {}; }

    private:
        std::optional<Clock::time_point> declined_since_;
        Clock::time_point keep_until_{};
    };

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
