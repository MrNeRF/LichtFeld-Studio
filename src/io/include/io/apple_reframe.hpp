/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "io/loader.hpp"

namespace lfs::io {
    // False on unsupported hardware/OS, missing assets, or builds without the helper.
    [[nodiscard]] LFS_IO_API bool appleReframeAvailable();
    [[nodiscard]] LFS_IO_API Result<LoadResult> createAppleReframeSplat(
        const std::filesystem::path& photo, const LoadOptions& options = {});
} // namespace lfs::io
