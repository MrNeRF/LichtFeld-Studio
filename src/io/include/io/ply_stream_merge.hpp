/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "io/error.hpp"
#include <filesystem>
#include <functional>
#include <vector>

namespace lfs::io {
    // Concatenate vertex-only binary PLYs whose vertices already share world
    // coordinates. The complete property schema must match, including SH bands.
    // Uses bounded I/O and publishes the result only after every input is copied.
    [[nodiscard]] LFS_IO_API Result<void> merge_binary_ply_streaming(
        const std::vector<std::filesystem::path>& inputs,
        const std::filesystem::path& output,
        const std::function<bool()>& cancelled = {});
} // namespace lfs::io
