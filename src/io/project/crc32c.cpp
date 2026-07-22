/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "crc32c.hpp"

#include <array>
#include <cstring>

namespace lfs::io::project {

    namespace {

        constexpr std::uint32_t CRC32C_POLY_REFLECTED = 0x82F63B78u;

        using Crc32cTables = std::array<std::array<std::uint32_t, 256>, 8>;

        constexpr Crc32cTables make_tables() {
            Crc32cTables tables{};
            for (std::uint32_t i = 0; i < 256; ++i) {
                std::uint32_t crc = i;
                for (int bit = 0; bit < 8; ++bit) {
                    crc = (crc >> 1) ^ ((crc & 1u) ? CRC32C_POLY_REFLECTED : 0u);
                }
                tables[0][i] = crc;
            }
            for (std::uint32_t i = 0; i < 256; ++i) {
                for (std::size_t slice = 1; slice < 8; ++slice) {
                    const std::uint32_t prev = tables[slice - 1][i];
                    tables[slice][i] = (prev >> 8) ^ tables[0][prev & 0xFFu];
                }
            }
            return tables;
        }

        constexpr Crc32cTables TABLES = make_tables();

    } // namespace

    std::uint32_t crc32c(const std::uint32_t crc, const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        std::uint32_t state = ~crc;

        while (size >= 8) {
            std::uint32_t lo = 0;
            std::uint32_t hi = 0;
            std::memcpy(&lo, bytes, 4);
            std::memcpy(&hi, bytes + 4, 4);
            lo ^= state;
            state = TABLES[7][lo & 0xFFu] ^ TABLES[6][(lo >> 8) & 0xFFu] ^
                    TABLES[5][(lo >> 16) & 0xFFu] ^ TABLES[4][lo >> 24] ^
                    TABLES[3][hi & 0xFFu] ^ TABLES[2][(hi >> 8) & 0xFFu] ^
                    TABLES[1][(hi >> 16) & 0xFFu] ^ TABLES[0][hi >> 24];
            bytes += 8;
            size -= 8;
        }
        while (size-- > 0) {
            state = (state >> 8) ^ TABLES[0][(state ^ *bytes++) & 0xFFu];
        }
        return ~state;
    }

} // namespace lfs::io::project
