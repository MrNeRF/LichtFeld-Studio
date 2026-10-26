/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
namespace {
    template <class T>
    __device__ bool predicate_value(uint32_t op, T a, T b) {

        switch (op) {
        case LFS_HIP_OP_EqualTensor:
        case LFS_HIP_OP_EqualScalar: return a == b;
        case LFS_HIP_OP_NotEqualTensor:
        case LFS_HIP_OP_NotEqualScalar: return a != b;
        case LFS_HIP_OP_LessTensor:
        case LFS_HIP_OP_LessScalar: return a < b;
        case LFS_HIP_OP_LessEqualTensor:
        case LFS_HIP_OP_LessEqualScalar: return a <= b;
        case LFS_HIP_OP_GreaterTensor:
        case LFS_HIP_OP_GreaterScalar: return a > b;
        case LFS_HIP_OP_GreaterEqualTensor:
        case LFS_HIP_OP_GreaterEqualScalar: return a >= b;
        case LFS_HIP_OP_LogicalAndTensor: return (a != T{}) && (b != T{});
        case LFS_HIP_OP_LogicalOrTensor: return (a != T{}) || (b != T{});
        case LFS_HIP_OP_LogicalXorTensor: return (a != T{}) != (b != T{});
        case LFS_HIP_OP_LogicalNot: return a == T{};
        case LFS_HIP_OP_IsNan: return isnan(float(a));
        case LFS_HIP_OP_IsInf: return isinf(float(a));
        case LFS_HIP_OP_IsFinite: return isfinite(float(a));
        default: return false;
        }
    }
} // namespace
