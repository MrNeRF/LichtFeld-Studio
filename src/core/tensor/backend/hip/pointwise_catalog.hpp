/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/detail/pointwise_functors.hpp"
#include "core/device/portability.cuh"
#include "runtime_api.h"
namespace hip_pointwise {
    namespace ops = lfs::core::ops;
    template <class F>
    struct Lowering {
        static constexpr unsigned mode = std::is_invocable_v<F, float, float> ? 2 : 0;
        static constexpr bool predicate = ops::returns_bool<F>::value;
        LFS_HOST_DEVICE static constexpr F make(float) { return {}; }
    };
    template <class F, class Scalar>
    struct Lowering<ops::scalar_right_op<F, Scalar>> {
        static constexpr unsigned mode = 1;
        static constexpr bool predicate = ops::returns_bool<F>::value;
        LFS_HOST_DEVICE static constexpr F make(float) { return {}; }
        LFS_HOST_DEVICE static constexpr auto unary(float scalar) { return ops::scalar_right_op<F, float>{scalar}; }
    };
    template <class F, class G>
    struct Lowering<ops::composed_unary_op<F, G>> {
        static constexpr unsigned mode = 0;
        static constexpr bool predicate = false;
        template <class Op>
        LFS_HOST_DEVICE static constexpr auto part(float scalar) {
            if constexpr (Lowering<Op>::mode == 1)
                return Lowering<Op>::unary(scalar);
            else
                return Lowering<Op>::make(scalar);
        }
        LFS_HOST_DEVICE static constexpr auto make(float scalar) {
            const auto f = part<F>(scalar);
            const auto g = part<G>(scalar);
            return ops::composed_unary_op<decltype(f), decltype(g)>{f, g};
        }
    };
    constexpr bool half_supported(unsigned op) {
        switch (op) {
        case LFS_HIP_OP_AddTensor:
        case LFS_HIP_OP_SubTensor:
        case LFS_HIP_OP_MulTensor:
        case LFS_HIP_OP_DivTensor:
        case LFS_HIP_OP_EqualTensor:
        case LFS_HIP_OP_NotEqualTensor:
        case LFS_HIP_OP_LessTensor:
        case LFS_HIP_OP_LessEqualTensor:
        case LFS_HIP_OP_GreaterTensor:
        case LFS_HIP_OP_GreaterEqualTensor:
        case LFS_HIP_OP_LogicalAndTensor:
        case LFS_HIP_OP_LogicalOrTensor:
        case LFS_HIP_OP_LogicalXorTensor:
        case LFS_HIP_OP_IsNan:
        case LFS_HIP_OP_IsInf:
        case LFS_HIP_OP_IsFinite:
        case LFS_HIP_OP_LogicalNot:
            return true;
        default: return false;
        }
    }
} // namespace hip_pointwise
