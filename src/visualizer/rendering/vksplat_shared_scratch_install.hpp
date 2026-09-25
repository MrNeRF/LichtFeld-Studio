/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

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

    // While the camera moves during training, training runs one step after every
    // viewer frame: frames keep a steady cadence and training keeps progressing.
    inline constexpr std::uint32_t kTrainingFramesPerNavigationRender = 1;

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
