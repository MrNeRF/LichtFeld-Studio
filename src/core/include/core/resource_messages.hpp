/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <string_view>

namespace lfs::core {

    // Host-RAM exhaustion shares ResourceExhausted/Training with GPU OOM. The
    // snapshot service starts its user message with this prefix so error
    // surfaces can tell the two apart without new event plumbing.
    inline constexpr const char* HOST_MEMORY_SAVE_ERROR_PREFIX =
        "Not enough free memory to save the project";

    // A project save whose volume is full, from the save preflight or a failed write. Other ResourceExhausted
    // IO errors are memory or format limits, and some write paths carry only the text, so surfaces match the
    // message to offer the disk-space recovery.
    inline constexpr const char* DISK_SPACE_SAVE_ERROR_MESSAGE = "There is not enough disk space to save the project.";

    [[nodiscard]] inline bool is_disk_space_save_error(const std::string_view message) noexcept {
        return message.find(DISK_SPACE_SAVE_ERROR_MESSAGE) != std::string_view::npos;
    }

} // namespace lfs::core
