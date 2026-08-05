/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace lfs::core {

    struct LFS_CORE_API SessionBreadcrumb {
        std::uint64_t pid = 0;
        std::string started_at;
        std::string log_path;
        bool clean_exit = false;
    };

    // Multiple concurrent instances intentionally use the same breadcrumb and
    // previous-log snapshot. Last writer wins; cross-instance attribution is not
    // guaranteed. clean_exit == false is also an accepted false positive for
    // SIGKILL, Ctrl+C in headless mode, crashes, and exits that did not reach
    // mark_clean_exit(); the breadcrumb cannot distinguish those cases.
    LFS_CORE_API void record_session_start() noexcept;
    LFS_CORE_API void mark_clean_exit() noexcept;
    [[nodiscard]] LFS_CORE_API std::optional<SessionBreadcrumb> previous_session() noexcept;

} // namespace lfs::core
