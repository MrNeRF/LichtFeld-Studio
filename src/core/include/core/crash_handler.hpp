/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"

#include <functional>

namespace lfs::core {

    // Installs process-wide last-resort diagnostics. Call only after the ABI
    // tripwire: a stale core must never execute current-core startup hooks.
    LFS_CORE_API void install_crash_handlers();

    // Flushes the logger and any other flushable diagnostic sink. Swallows
    // all exceptions; safe to call before the logger has been initialized.
    LFS_CORE_API void flush_diagnostics_noexcept() noexcept;

    // Flushes diagnostics, then terminates the process without running
    // destructors. The single sanctioned replacement for std::_Exit/_exit.
    [[noreturn]] LFS_CORE_API void flush_and_exit(int code) noexcept;

    // Invokes fn, converting any exception that escapes it into a failure
    // report instead of letting it reach std::terminate. Returns fn()'s
    // result on success, or the frozen firewall exit code (70 / EX_SOFTWARE)
    // if fn threw. Use at a single outermost dispatch site.
    LFS_CORE_API int run_with_exception_firewall(const std::function<int()>& fn) noexcept;

} // namespace lfs::core
