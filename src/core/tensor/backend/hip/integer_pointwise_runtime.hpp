/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "predicate_api.h"
namespace {
    template <class T>
    __device__ T integer_bits_native(std::make_unsigned_t<T> value) { return __builtin_bit_cast(T, value); }
    template <class T>
    __device__ T integer_math_native(uint32_t op, T a, T b) {
        using U = std::make_unsigned_t<T>;
        switch (op) {
        case LFS_HIP_OP_AddScalar:
        case LFS_HIP_OP_AddTensor: return integer_bits_native<T>(lfs::core::ops::add_op{}(U(a), U(b)));
        case LFS_HIP_OP_SubScalar:
        case LFS_HIP_OP_SubTensor: return integer_bits_native<T>(lfs::core::ops::sub_op{}(U(a), U(b)));
        case LFS_HIP_OP_MulScalar:
        case LFS_HIP_OP_MulTensor: return integer_bits_native<T>(lfs::core::ops::mul_op{}(U(a), U(b)));
        case LFS_HIP_OP_DivScalar:
        case LFS_HIP_OP_DivTensor:
            if (!b)
                return 0;
            if constexpr (std::is_signed_v<T>)
                if (a == std::numeric_limits<T>::min() && b == T(-1))
                    return a;
            return lfs::core::ops::div_op{}(a, b);
        case LFS_HIP_OP_ModScalar:
        case LFS_HIP_OP_ModTensor:
            if (!b)
                return 0;
            if constexpr (std::is_signed_v<T>)
                if (a == std::numeric_limits<T>::min() && b == T(-1))
                    return 0;
            return lfs::core::ops::mod_op{}(a, b);
        case LFS_HIP_OP_PowScalar:
        case LFS_HIP_OP_PowTensor: return lfs::core::ops::pow_op{}(a, b);
        case LFS_HIP_OP_MinimumScalar:
        case LFS_HIP_OP_MinimumTensor: return lfs::core::ops::minimum_op{}(a, b);
        case LFS_HIP_OP_MaximumScalar:
        case LFS_HIP_OP_MaximumTensor: return lfs::core::ops::maximum_op{}(a, b);
        case LFS_HIP_OP_Abs: return lfs::core::ops::abs_op{}(a);
        // Operate on unsigned storage so signed overflow remains defined.
        case LFS_HIP_OP_Neg: return integer_bits_native<T>(static_cast<U>(lfs::core::ops::neg_op{}(uint64_t(U(a)))));
        case LFS_HIP_OP_Square: return integer_bits_native<T>(lfs::core::ops::square_op{}(U(a)));
        case LFS_HIP_OP_Relu: return lfs::core::ops::relu_op{}(a);
        case LFS_HIP_OP_Sign: return lfs::core::ops::sign_op{}(a);
        default: return a;
        }
    }
    template <class T>
    __global__ void integer_pointwise_native(const T* lhs, const T* rhs, T* output, size_t count, T scalar, uint32_t op, bool reverse) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            const T a = lhs[i], b = rhs ? rhs[i] : scalar;
            output[i] = reverse ? integer_math_native(op, b, a) : integer_math_native(op, a, b);
        }
    }
    template <class T>
    void launch_integer_pointwise_native(LfsHipContext* context, LfsHipView lhs, LfsHipView rhs, LfsHipView output,
                                         const LfsHipIntegerPointwise& p, bool binary) {
        require(p.count <= SIZE_MAX / sizeof(T), "HIP integer pointwise range overflow");
        const size_t bytes = size_t(p.count) * sizeof(T);
        validate(context, lhs.buffer, lhs.offset, bytes);
        validate(context, output.buffer, output.offset, bytes);
        require(lhs.offset % alignof(T) == 0 && output.offset % alignof(T) == 0, "Misaligned HIP integer pointwise view");
        require_compatible_views(lhs, bytes, output, bytes, true, "HIP integer pointwise partially aliases input");
        if (binary) {
            validate(context, rhs.buffer, rhs.offset, bytes);
            require(rhs.offset % alignof(T) == 0, "Misaligned HIP integer rhs");
            require_compatible_views(rhs, bytes, output, bytes, true, "HIP integer pointwise partially aliases input");
        }
        const bool scalar_op = p.op <= LFS_HIP_OP_DivScalar || p.op == LFS_HIP_OP_PowScalar || p.op == LFS_HIP_OP_ModScalar ||
                               p.op == LFS_HIP_OP_MinimumScalar || p.op == LFS_HIP_OP_MaximumScalar;
        T scalar{};
        if (scalar_op && p.scalar_kind == 0) {
            const double value = std::trunc(double(p.float_scalar));
            const double upper = std::ldexp(1.0, int(sizeof(T) * 8 - (std::is_signed_v<T> ? 1 : 0)));
            require(std::isfinite(value) && value >= (std::is_signed_v<T> ? -upper : 0.0) && value < upper, "HIP integer scalar is out of range");
            scalar = static_cast<T>(p.float_scalar);
        } else if (scalar_op)
            scalar = static_cast<T>(p.integer_scalar);
        if (!p.count)
            return;
        const auto* a = reinterpret_cast<const T*>(static_cast<const char*>(lhs.buffer->data) + lhs.offset);
        const auto* b = binary ? reinterpret_cast<const T*>(static_cast<const char*>(rhs.buffer->data) + rhs.offset) : nullptr;
        auto* out = reinterpret_cast<T*>(static_cast<char*>(output.buffer->data) + output.offset);
        hipLaunchKernelGGL((integer_pointwise_native<T>), dim3(blocks(p.count)), dim3(256), 0, context->stream, a, b, out, size_t(p.count), scalar, p.op, scalar_op && bool(p.scalar_left));
        check(hipGetLastError());
    }
} // namespace
extern "C" int lfs_hip_integer_pointwise(LfsHipContext* context, LfsHipView lhs, LfsHipView rhs, LfsHipView output,
                                         const LfsHipIntegerPointwise* p, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && p && p->struct_bytes == sizeof(*p) && p->scalar_left <= 1 && p->scalar_kind <= 3, "Invalid HIP integer pointwise descriptor");
        bool binary = false;
        switch (p->op) {
        case LFS_HIP_OP_AddTensor:
        case LFS_HIP_OP_SubTensor:
        case LFS_HIP_OP_MulTensor:
        case LFS_HIP_OP_DivTensor:
        case LFS_HIP_OP_ModTensor:
        case LFS_HIP_OP_PowTensor:
        case LFS_HIP_OP_MinimumTensor:
        case LFS_HIP_OP_MaximumTensor: binary = true; break;
        case LFS_HIP_OP_AddScalar:
        case LFS_HIP_OP_SubScalar:
        case LFS_HIP_OP_MulScalar:
        case LFS_HIP_OP_DivScalar:
        case LFS_HIP_OP_ModScalar:
        case LFS_HIP_OP_PowScalar:
        case LFS_HIP_OP_MinimumScalar:
        case LFS_HIP_OP_MaximumScalar:
        case LFS_HIP_OP_Abs:
        case LFS_HIP_OP_Neg:
        case LFS_HIP_OP_Square:
        case LFS_HIP_OP_Relu:
        case LFS_HIP_OP_Sign:
        case LFS_HIP_OP_Floor:
        case LFS_HIP_OP_Ceil:
        case LFS_HIP_OP_Round:
        case LFS_HIP_OP_Trunc: break;
        default: throw std::invalid_argument("Unsupported HIP integer pointwise operation");
        }
        DeviceScope device(context->device);
        switch (p->dtype) {
        case LFS_HIP_I32: launch_integer_pointwise_native<int32_t>(context, lhs, rhs, output, *p, binary); break;
        case LFS_HIP_I64: launch_integer_pointwise_native<int64_t>(context, lhs, rhs, output, *p, binary); break;
        case LFS_HIP_U32: launch_integer_pointwise_native<uint32_t>(context, lhs, rhs, output, *p, binary); break;
        case LFS_HIP_U8: launch_integer_pointwise_native<uint8_t>(context, lhs, rhs, output, *p, binary); break;
        default: throw std::invalid_argument("Unsupported HIP integer pointwise type");
        }
    });
}
