/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../../internal/expression_c_like.hpp"
namespace lfs::core::internal {
    namespace {
        struct MslDialect {
            static constexpr auto prelude = R"(#include <metal_stdlib>
#pragma METAL fp contract(off)
using namespace metal;
#define LFS_FUNCTION static
#define LFS_THREAD thread
#define lfs_ulong ulong
#define lfs_max max
#define lfs_min min
)";
            static constexpr auto entry = R"(kernel void lfs_expression(constant uint* lfs_args [[buffer(0)]],
                           uint lfs_thread [[thread_position_in_grid]],
                           uint lfs_threads [[threads_per_grid]]) {
)";
            static constexpr auto const_uint = "device const uint";
            static constexpr auto const_uchar = "device const uchar";
            static constexpr auto const_ushort = "device const ushort";
            static constexpr auto uint = "device uint";
            static constexpr auto absolute = "abs";
            static constexpr auto rsqrt = "rsqrt";
        };
    } // namespace
    std::string expression_msl(const ExpressionProgram& program, const ExpressionSignature& signature) {
        return expression_c_like<MslDialect>(program, signature);
    }
} // namespace lfs::core::internal
