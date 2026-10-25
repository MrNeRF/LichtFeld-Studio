/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "predicate_api.h"
#include "predicate_value_runtime.hpp"
namespace {
    template <class T>
    __global__ void predicate_typed(uint32_t op, const T* lhs, const T* rhs, uint8_t* output, size_t count, T scalar, bool scalar_left) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            const T a = scalar_left ? scalar : lhs[i], b = rhs ? rhs[i] : (scalar_left ? lhs[i] : scalar);
            output[i] = predicate_value(op, a, b) ? 1 : 0;
        }
    }
} // namespace
extern "C" int lfs_hip_predicate_typed(LfsHipContext* context, LfsHipView lhs, LfsHipView rhs, LfsHipView output,
                                       const LfsHipPredicate* p, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && p && p->struct_bytes == sizeof(*p) && p->scalar_left <= 1 && p->count <= SIZE_MAX && p->scalar_kind <= 3,
                "Invalid HIP typed predicate descriptor");
        const bool binary = p->op >= LFS_HIP_OP_EqualTensor && p->op <= LFS_HIP_OP_LogicalXorTensor;
        const bool scalar = p->op >= LFS_HIP_OP_EqualScalar && p->op <= LFS_HIP_OP_GreaterEqualScalar;
        require(binary || scalar || (p->op >= LFS_HIP_OP_IsNan && p->op <= LFS_HIP_OP_LogicalNot), "Invalid HIP typed predicate operation");
        require(p->dtype == LFS_HIP_I32 || p->dtype == LFS_HIP_I64 || p->dtype == LFS_HIP_U32 || p->dtype == LFS_HIP_F16 || p->dtype == LFS_HIP_U8 || p->dtype == LFS_HIP_BOOL,
                "Unsupported HIP typed predicate dtype");
        const size_t width = typed_bytes(p->dtype, 1), bytes = typed_bytes(p->dtype, p->count);
        validate(context, lhs.buffer, lhs.offset, bytes);
        validate(context, output.buffer, output.offset, p->count);
        require(lhs.offset % width == 0 && (!binary || rhs.offset % width == 0), "Misaligned HIP typed predicate input");
        require_compatible_views(lhs, bytes, output, p->count, width == 1, "HIP typed predicate output aliases an input");
        if (binary) {
            validate(context, rhs.buffer, rhs.offset, bytes);
            require_compatible_views(rhs, bytes, output, p->count, width == 1, "HIP typed predicate output aliases an input");
        }
        if (!p->count)
            return;
        DeviceScope device(context->device);
        const auto launch = [&]<class T>() {
            T value{};
            if (scalar) {
                if (p->scalar_kind == 0) {
                    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
                        const double v = std::trunc(double(p->float_scalar));
                        if constexpr (std::is_same_v<T, int64_t>)
                            require(std::isfinite(v) && v >= -9223372036854775808. && v < 9223372036854775808., "HIP predicate scalar is outside Int64");
                        else
                            require(std::isfinite(v) && v >= double(std::numeric_limits<T>::lowest()) && v <= double(std::numeric_limits<T>::max()),
                                    "HIP predicate scalar is outside its integer dtype");
                    }
                    value = static_cast<T>(p->float_scalar);
                } else {
                    if constexpr (std::is_same_v<T, __half>)
                        value = __float2half_rn(float(p->integer_scalar));
                    else
                        value = static_cast<T>(p->integer_scalar);
                }
            }
            hipLaunchKernelGGL((predicate_typed<T>), dim3(blocks(p->count)), dim3(256), 0, context->stream, p->op,
                               reinterpret_cast<const T*>(static_cast<const char*>(lhs.buffer->data) + lhs.offset),
                               binary ? reinterpret_cast<const T*>(static_cast<const char*>(rhs.buffer->data) + rhs.offset) : nullptr,
                               static_cast<uint8_t*>(output.buffer->data) + output.offset, size_t(p->count), value, scalar && p->scalar_left == 1);
        };
        switch (p->dtype) {
        case LFS_HIP_I32: launch.template operator()<int32_t>(); break;
        case LFS_HIP_I64: launch.template operator()<int64_t>(); break;
        case LFS_HIP_U32: launch.template operator()<uint32_t>(); break;
        case LFS_HIP_F16: launch.template operator()<__half>(); break;
        case LFS_HIP_U8: launch.template operator()<uint8_t>(); break;
        case LFS_HIP_BOOL: launch.template operator()<bool>(); break;
        }
        check(hipGetLastError());
    });
}
