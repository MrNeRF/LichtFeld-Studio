/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

namespace lfs::core {

    // The raster arena is a CUDA training resource. These calls are no-ops
    // when that arena is not part of the build.
    [[nodiscard]] LFS_CORE_API bool raster_arena_under_memory_pressure();
    LFS_CORE_API void clear_raster_arena_external_backing(const void* device_ptr = nullptr);

} // namespace lfs::core
