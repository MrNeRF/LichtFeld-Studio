/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "predicate_value_runtime.hpp"

namespace {
    // Keep integer indices and mask operands in their native type. The floating
    // broadcast kernel deliberately converts half operands to float and cannot be
    // reused for values outside Float32's exact integer range.
    template <class T, bool Predicate>
    __global__ void integral_broadcast_kernel(const T* lhs, const T* rhs, void* output,
                                              LfsHipLayout a, LfsHipLayout b, LfsHipLayout out, uint32_t op) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < out.elements; i += size_t(gridDim.x) * blockDim.x) {
            const T x = lhs[broadcast_index(i, a, out)], y = rhs[broadcast_index(i, b, out)];
            const auto target = layout_index(i, out);
            if constexpr (Predicate) {
                static_cast<uint8_t*>(output)[target] = uint8_t(predicate_value(op, x, y));
            } else {
                static_cast<T*>(output)[target] = integer_math_native(op, x, y);
            }
        }
    }

    void launch_integral_broadcast(LfsHipContext* context, uint32_t op, LfsHipView lhs, LfsHipView rhs, LfsHipView output,
                                   LfsHipLayout a, LfsHipLayout b, LfsHipLayout out, uint32_t dtype) {
        const bool predicate = (pointwise_flags[op] & 4) != 0;
        require(dtype != LFS_HIP_BOOL || predicate, "HIP Bool broadcast requires a predicate operation");
        if (!predicate) {
            switch (op) {
            case LFS_HIP_OP_AddTensor:
            case LFS_HIP_OP_SubTensor:
            case LFS_HIP_OP_MulTensor:
            case LFS_HIP_OP_DivTensor:
            case LFS_HIP_OP_ModTensor:
            case LFS_HIP_OP_PowTensor:
            case LFS_HIP_OP_MinimumTensor:
            case LFS_HIP_OP_MaximumTensor: break;
            default: throw std::invalid_argument("Unsupported HIP integer broadcast arithmetic");
            }
        }
        if (!out.elements)
            return;
        const auto launch = [&]<class T>() {
            const auto* x = reinterpret_cast<const T*>(static_cast<const char*>(lhs.buffer->data) + lhs.offset);
            const auto* y = reinterpret_cast<const T*>(static_cast<const char*>(rhs.buffer->data) + rhs.offset);
            auto* destination = static_cast<char*>(output.buffer->data) + output.offset;
            if (predicate) {
                hipLaunchKernelGGL((integral_broadcast_kernel<T, true>), dim3(blocks(out.elements)), dim3(256), 0, context->stream, x, y, destination, a, b, out, op);
            } else {
                hipLaunchKernelGGL((integral_broadcast_kernel<T, false>), dim3(blocks(out.elements)), dim3(256), 0, context->stream, x, y, destination, a, b, out, op);
            }
        };
        switch (dtype) {
        case LFS_HIP_I32: launch.template operator()<int32_t>(); break;
        case LFS_HIP_I64: launch.template operator()<int64_t>(); break;
        case LFS_HIP_U32: launch.template operator()<uint32_t>(); break;
        case LFS_HIP_U8:
        case LFS_HIP_BOOL: launch.template operator()<uint8_t>(); break;
        default: throw std::invalid_argument("Unsupported HIP integer broadcast dtype");
        }
        check(hipGetLastError());
    }
} // namespace
