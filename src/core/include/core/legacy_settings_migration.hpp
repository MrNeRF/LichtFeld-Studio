/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"

namespace lfs::core {
    class UserPaths;

    /** Import pre-unified settings without deleting or overwriting sources. */
    [[nodiscard]] LFS_CORE_API lfs::Status migrateLegacySettings(const UserPaths& paths);
    [[nodiscard]] LFS_CORE_API bool takeLegacySettingsMigrationNotice() noexcept;
} // namespace lfs::core
