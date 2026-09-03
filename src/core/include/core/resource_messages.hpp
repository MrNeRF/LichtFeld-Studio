/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

namespace lfs::core {

    // Host-RAM exhaustion shares ResourceExhausted/Training with GPU OOM. The
    // snapshot service starts its user message with this prefix so error
    // surfaces can tell the two apart without new event plumbing.
    inline constexpr const char* HOST_MEMORY_SAVE_ERROR_PREFIX =
        "Not enough free memory to save the project";

} // namespace lfs::core
