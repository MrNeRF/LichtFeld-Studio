/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "crc32c.hpp"

#include <array>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <nmmintrin.h>
#endif

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

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
        __attribute__((target("sse4.2")))
#endif
        std::uint32_t
        crc32c_sse42_impl(const std::uint32_t crc, const void* data,
                          std::size_t size) {
            const auto* bytes = static_cast<const std::uint8_t*>(data);
            std::uint64_t state = ~static_cast<std::uint64_t>(crc);
            while (size >= sizeof(std::uint64_t)) {
                std::uint64_t value = 0;
                std::memcpy(&value, bytes, sizeof(value));
                state = _mm_crc32_u64(state, value);
                bytes += sizeof(value);
                size -= sizeof(value);
            }
            auto state32 = static_cast<std::uint32_t>(state);
            while (size-- > 0)
                state32 = _mm_crc32_u8(state32, *bytes++);
            return ~state32;
        }
#endif

        bool has_sse42() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
            return __builtin_cpu_supports("sse4.2");
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
            int cpu_info[4]{};
            __cpuid(cpu_info, 1);
            return (cpu_info[2] & (1 << 20)) != 0;
#else
            return false;
#endif
#else
            return false;
#endif
        }

    } // namespace

    std::uint32_t crc32c_software(const std::uint32_t crc, const void* data,
                                  std::size_t size) {
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
        while (size-- > 0)
            state = (state >> 8) ^ TABLES[0][(state ^ *bytes++) & 0xFFu];
        return ~state;
    }

    std::uint32_t crc32c(const std::uint32_t crc, const void* data, const std::size_t size) {
#if defined(__x86_64__) || defined(_M_X64)
        static const bool use_hardware = has_sse42();
        if (use_hardware)
            return crc32c_sse42(crc, data, size);
#endif
        return crc32c_software(crc, data, size);
    }

    namespace {

        constexpr int kCrc32cGf2Dim = 32;

        std::uint32_t gf2_matrix_times(const std::uint32_t* mat,
                                       std::uint32_t vec) noexcept {
            std::uint32_t sum = 0;
            while (vec != 0) {
                if ((vec & 1u) != 0) {
                    sum ^= *mat;
                }
                vec >>= 1;
                ++mat;
            }
            return sum;
        }

        void gf2_matrix_square(std::uint32_t* square,
                               const std::uint32_t* mat) noexcept {
            for (int n = 0; n < kCrc32cGf2Dim; ++n) {
                square[n] = gf2_matrix_times(mat, mat[n]);
            }
        }

    } // namespace

    std::uint32_t crc32c_combine(const std::uint32_t crc_a,
                                 const std::uint32_t crc_b,
                                 std::uint64_t len_b) {
        if (len_b == 0) {
            return crc_a;
        }

        std::uint32_t odd[kCrc32cGf2Dim];
        std::uint32_t even[kCrc32cGf2Dim];
        odd[0] = CRC32C_POLY_REFLECTED;
        std::uint32_t row = 1;
        for (int n = 1; n < kCrc32cGf2Dim; ++n) {
            odd[n] = row;
            row <<= 1;
        }

        gf2_matrix_square(even, odd);
        gf2_matrix_square(odd, even);

        std::uint32_t crc = crc_a;
        do {
            gf2_matrix_square(even, odd);
            if ((len_b & 1u) != 0) {
                crc = gf2_matrix_times(even, crc);
            }
            len_b >>= 1;
            if (len_b == 0) {
                break;
            }
            gf2_matrix_square(odd, even);
            if ((len_b & 1u) != 0) {
                crc = gf2_matrix_times(odd, crc);
            }
            len_b >>= 1;
        } while (len_b != 0);

        return crc ^ crc_b;
    }

#if defined(__x86_64__) || defined(_M_X64)
    std::uint32_t crc32c_sse42(const std::uint32_t crc, const void* data,
                               const std::size_t size) {
        return crc32c_sse42_impl(crc, data, size);
    }
#endif

} // namespace lfs::io::project
