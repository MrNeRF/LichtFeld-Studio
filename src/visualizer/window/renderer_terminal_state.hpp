/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstdint>

namespace lfs::vis {

    // Terminal renderer condition derived from the Phase 7B quarantine latch.
    // DeviceLost dominates: a lost device also latches quarantine, but callers
    // must see the cause, not just the symptom.
    enum class RendererTerminalState : std::uint8_t {
        Running,
        Quarantined, // bounded-wait stall policy latched (10 s); terminal by 7B design
        DeviceLost,  // a wait/acquire surfaced VK_ERROR_DEVICE_LOST
    };

    [[nodiscard]] constexpr RendererTerminalState
    renderer_terminal_state(const bool device_lost, const bool quarantined) noexcept {
        if (device_lost)
            return RendererTerminalState::DeviceLost;
        return quarantined ? RendererTerminalState::Quarantined
                           : RendererTerminalState::Running;
    }

} // namespace lfs::vis
