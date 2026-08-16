/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>

namespace lfs::io::project {

    // CRC32c (Castagnoli), with runtime SSE4.2 dispatch and a slice-by-8
    // software fallback. Streaming: seed with 0, feed consecutive spans with
    // the previous return value.
    [[nodiscard]] std::uint32_t crc32c(std::uint32_t crc, const void* data, std::size_t size);

    [[nodiscard]] std::uint32_t crc32c_software(std::uint32_t crc,
                                                const void* data,
                                                std::size_t size);
#if defined(__x86_64__) || defined(_M_X64)
    [[nodiscard]] std::uint32_t crc32c_sse42(std::uint32_t crc,
                                             const void* data,
                                             std::size_t size);
#endif

} // namespace lfs::io::project
