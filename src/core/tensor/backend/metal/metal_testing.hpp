/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"

#include <cstddef>

namespace lfs::core::internal {
    // Test hook: the most elements one generic fill or copy dispatch covers (2^31 by default; 0 restores it). A kernel's
    // thread index is 32 bits wide, so longer calls go in pieces; a test lowers the piece to cover the split without
    // gigabytes.
    LFS_CORE_API void set_metal_copy_piece_elements_for_testing(std::size_t elements) noexcept;
} // namespace lfs::core::internal
