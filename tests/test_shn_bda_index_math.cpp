/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/sh_value_quant.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <utility>

namespace {

    // Mirror of spherical_harmonics.slang shAtU16. Keep the formula text in sync
    // with that function (same block/lane/n_cells expression; R = SH_REORDER_SIZE).
    [[nodiscard]] constexpr std::uint64_t shAtU16(const std::uint64_t primitive_idx,
                                                  const std::uint64_t cell,
                                                  const std::uint64_t n_cells) {
        constexpr std::uint64_t R = lfs::core::kShReorderSize;
        const std::uint64_t block = primitive_idx / R;
        const std::uint64_t lane = primitive_idx % R;
        return block * (n_cells * R) + cell * R + lane;
    }

    // 128-bit unsigned (hi, lo). Schoolbook 32-bit limbs so the reference is
    // independent of the 64-bit shader-mirror product.
    struct U128 {
        std::uint64_t hi = 0;
        std::uint64_t lo = 0;
    };

    [[nodiscard]] U128 mul_u64(const std::uint64_t a, const std::uint64_t b) {
        const std::uint64_t a_lo = a & 0xffffffffull;
        const std::uint64_t a_hi = a >> 32;
        const std::uint64_t b_lo = b & 0xffffffffull;
        const std::uint64_t b_hi = b >> 32;
        const std::uint64_t p0 = a_lo * b_lo;
        const std::uint64_t p1 = a_lo * b_hi;
        const std::uint64_t p2 = a_hi * b_lo;
        const std::uint64_t p3 = a_hi * b_hi;
        const std::uint64_t mid = (p0 >> 32) + (p1 & 0xffffffffull) + (p2 & 0xffffffffull);
        U128 r;
        r.lo = (p0 & 0xffffffffull) | (mid << 32);
        r.hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
        return r;
    }

    [[nodiscard]] U128 add_u64(const U128 a, const std::uint64_t b) {
        U128 r;
        r.lo = a.lo + b;
        r.hi = a.hi + (r.lo < a.lo ? 1ull : 0ull);
        return r;
    }

    [[nodiscard]] U128 shAtU16_i128(const std::uint64_t primitive_idx,
                                    const std::uint64_t cell,
                                    const std::uint64_t n_cells) {
        constexpr std::uint64_t R = lfs::core::kShReorderSize;
        const std::uint64_t block = primitive_idx / R;
        const std::uint64_t lane = primitive_idx % R;
        const U128 nR = mul_u64(n_cells, R);
        const U128 block_term = mul_u64(block, nR.lo);
        EXPECT_EQ(nR.hi, 0u);
        EXPECT_EQ(block_term.hi, 0u);
        return add_u64(add_u64(block_term, cell * R), lane);
    }

} // namespace

TEST(ShNBdaIndexMath, LastCellMatchesInt128AndOverflowsWhereExpected) {
    using lfs::core::sh_value_quant::n_value_cells_per_prim;

    constexpr std::uint64_t kNs[] = {
        1ull, 32ull, 255ull, 256ull, 47700000ull, 95000000ull, 100000000ull, 268000000ull};
    constexpr std::uint32_t kRests[] = {3u, 8u, 15u};
    constexpr std::uint64_t kU32 = std::numeric_limits<std::uint32_t>::max();

    for (const std::uint64_t n : kNs) {
        for (const std::uint32_t rest : kRests) {
            const std::uint64_t n_cells = n_value_cells_per_prim(rest);
            ASSERT_GT(n_cells, 0u);
            const std::uint64_t last_prim = n - 1ull;
            const std::uint64_t last_cell = n_cells - 1ull;

            const std::uint64_t index = shAtU16(last_prim, last_cell, n_cells);
            const U128 index_i128 = shAtU16_i128(last_prim, last_cell, n_cells);
            EXPECT_EQ(index_i128.hi, 0u) << "N=" << n << " rest=" << rest;
            EXPECT_EQ(index_i128.lo, index) << "N=" << n << " rest=" << rest;

            const U128 byte_i128 = mul_u64(index_i128.lo, 2ull);
            EXPECT_EQ(byte_i128.hi, 0u) << "N=" << n << " rest=" << rest;
            EXPECT_EQ(byte_i128.lo, index * 2ull) << "N=" << n << " rest=" << rest;

            const bool expect_u32_overflow = index_i128.lo > kU32;
            EXPECT_EQ(index > kU32, expect_u32_overflow) << "N=" << n << " rest=" << rest;
            if (n >= 100000000ull && rest == 15u) {
                EXPECT_TRUE(expect_u32_overflow) << "SH3 last-cell index must exceed 2^32 at N=" << n;
            }
            if (n <= 47700000ull) {
                EXPECT_FALSE(expect_u32_overflow) << "index should fit in 32 bits at N=" << n
                                                  << " rest=" << rest;
            }
        }
    }
}
