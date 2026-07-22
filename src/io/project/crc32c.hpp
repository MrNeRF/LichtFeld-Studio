/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>

namespace lfs::io::project {

    // CRC32c (Castagnoli), slice-by-8 software implementation. Streaming:
    // seed with 0, feed consecutive spans with the previous return value.
    [[nodiscard]] std::uint32_t crc32c(std::uint32_t crc, const void* data, std::size_t size);

} // namespace lfs::io::project
