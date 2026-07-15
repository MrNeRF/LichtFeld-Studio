/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"

namespace lfs::core {

    // Installs process-wide last-resort diagnostics. Call only after the ABI
    // tripwire: a stale core must never execute current-core startup hooks.
    LFS_CORE_API void install_crash_handlers();

} // namespace lfs::core
