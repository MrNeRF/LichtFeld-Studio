/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstdint>
#include <string>

namespace lfs::core::system_info {

    struct LFS_CORE_API SystemInfo {
        std::string os;
        std::string os_build;
        std::string cpu;
        std::uint64_t ram_mb = 0;
        std::string cuda_runtime;
    };

    // Best-effort system inventory. Missing platform data is represented by an
    // empty string or zero and never makes collection fail.
    [[nodiscard]] LFS_CORE_API SystemInfo collect() noexcept;

} // namespace lfs::core::system_info
