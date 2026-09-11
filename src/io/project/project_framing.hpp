/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace lfs::io::project::detail {

    inline constexpr std::array<std::byte, 8> FRAMED_MAGIC{
        std::byte{'L'}, std::byte{'F'}, std::byte{'S'}, std::byte{'Z'},
        std::byte{'F'}, std::byte{'R'}, std::byte{'M'}, std::byte{0}};
    inline constexpr std::uint16_t FRAMED_VERSION = 1;
    inline constexpr std::size_t FRAMED_HEADER_BYTES = 16;
    inline constexpr std::size_t FRAMED_RECORD_BYTES = 16;
    inline constexpr std::size_t FRAMED_RECORD_TARGET_BYTES = 64ull * 1024 * 1024;

    [[nodiscard]] constexpr std::size_t framed_record_count(const std::size_t bytes) noexcept {
        return bytes == 0 ? 0 : (bytes + FRAMED_RECORD_TARGET_BYTES - 1) / FRAMED_RECORD_TARGET_BYTES;
    }

} // namespace lfs::io::project::detail
