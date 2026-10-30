/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "../../internal/expression_c_like.hpp"
namespace lfs::core::internal {
    struct HipExpressionDialect {
        static constexpr auto prelude = R"(#pragma clang fp contract(off)
using uint=unsigned int;using lfs_ulong=unsigned long long;using uchar=unsigned char;using ushort=unsigned short;
template<class T,class U> __device__ T as_type(U value) { static_assert(sizeof(T)==sizeof(U));return __builtin_bit_cast(T,value); }
template<class T> __device__ T lfs_max(T a,T b){return a>b?a:b;}
template<class T> __device__ T lfs_min(T a,T b){return a<b?a:b;}
__device__ uint clz(uint v){return __clz(v);}
__device__ uint mulhi(uint a,uint b){return __umulhi(a,b);}
#define LFS_FUNCTION static __device__
#define LFS_THREAD
)";
        static constexpr auto entry = R"(struct LfsExpressionArgs { uint words[256]; };
extern "C" __global__ void lfs_expression(LfsExpressionArgs arguments) {
const uint* lfs_args=arguments.words;
uint lfs_thread=blockIdx.x*blockDim.x+threadIdx.x;
uint lfs_threads=gridDim.x*blockDim.x;
)";
        static constexpr auto const_uint = "const uint";
        static constexpr auto const_uchar = "const uchar";
        static constexpr auto const_ushort = "const ushort";
        static constexpr auto uint = "uint";
        static constexpr auto absolute = "fabsf";
        static constexpr auto rsqrt = "rsqrtf";
    };
    inline std::string expression_hip(const ExpressionProgram& program, const ExpressionSignature& signature) {
        return expression_c_like<HipExpressionDialect>(program, signature);
    }
} // namespace lfs::core::internal
