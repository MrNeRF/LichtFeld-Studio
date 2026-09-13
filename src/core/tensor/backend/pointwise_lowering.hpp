/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "../internal/tensor_functors.hpp"
#include "descriptors.hpp"

#include <type_traits>

namespace lfs::core::internal {

    using pointwise_add_scalar_functor = ops::scalar_right_op<ops::add_op, float>;
    using pointwise_sub_scalar_functor = ops::scalar_right_op<ops::sub_op, float>;
    using pointwise_mul_scalar_functor = ops::scalar_right_op<ops::mul_op, float>;
    using pointwise_div_scalar_functor = ops::scalar_right_op<ops::div_op, float>;
    using pointwise_pow_scalar_functor = ops::scalar_right_op<ops::pow_op, float>;
    using pointwise_mod_scalar_functor = ops::scalar_right_op<ops::mod_op, float>;
    using pointwise_maximum_scalar_functor = ops::scalar_right_op<ops::maximum_op, int32_t>;
    using pointwise_minimum_scalar_functor = ops::scalar_right_op<ops::minimum_op, int32_t>;
    using pointwise_equal_scalar_functor = ops::scalar_right_op<ops::equal_op, float>;
    using pointwise_not_equal_scalar_functor = ops::scalar_right_op<ops::not_equal_op, float>;
    using pointwise_less_scalar_functor = ops::scalar_right_op<ops::less_op, float>;
    using pointwise_less_equal_scalar_functor = ops::scalar_right_op<ops::less_equal_op, float>;
    using pointwise_greater_scalar_functor = ops::scalar_right_op<ops::greater_op, float>;
    using pointwise_greater_equal_scalar_functor = ops::scalar_right_op<ops::greater_equal_op, float>;
    using pointwise_exp_mul_functor =
        ops::composed_unary_op<ops::exp_op, pointwise_mul_scalar_functor>;
    using pointwise_mul_abs_functor =
        ops::composed_unary_op<pointwise_mul_scalar_functor, ops::abs_op>;
    using pointwise_mul_relu_functor =
        ops::composed_unary_op<pointwise_mul_scalar_functor, ops::relu_op>;

    template <class>
    inline constexpr bool pointwise_dependent_false = false;

    template <class Functor>
    struct pointwise_op_of {
        static_assert(pointwise_dependent_false<Functor>,
                      "pointwise functor is missing from pointwise_ops.def");
    };

#define LFS_POINTWISE_OP(Id, FunctorType, Name)               \
    template <>                                               \
    struct pointwise_op_of<FunctorType> {                     \
        static constexpr PointwiseOp value = PointwiseOp::Id; \
    };
#include "pointwise_ops.def"
#undef LFS_POINTWISE_OP

    template <class BinaryOp>
    struct scalar_pointwise_op_of {
        static_assert(pointwise_dependent_false<BinaryOp>,
                      "scalar pointwise functor is missing from pointwise_ops.def");
    };

#define LFS_SCALAR_POINTWISE_OP(FunctorType, Id)              \
    template <>                                               \
    struct scalar_pointwise_op_of<FunctorType> {              \
        static constexpr PointwiseOp value = PointwiseOp::Id; \
    }

    LFS_SCALAR_POINTWISE_OP(ops::add_op, AddScalar);
    LFS_SCALAR_POINTWISE_OP(ops::sub_op, SubScalar);
    LFS_SCALAR_POINTWISE_OP(ops::mul_op, MulScalar);
    LFS_SCALAR_POINTWISE_OP(ops::div_op, DivScalar);
    LFS_SCALAR_POINTWISE_OP(ops::pow_op, PowScalar);
    LFS_SCALAR_POINTWISE_OP(ops::mod_op, ModScalar);
    LFS_SCALAR_POINTWISE_OP(ops::maximum_op, MaximumScalar);
    LFS_SCALAR_POINTWISE_OP(ops::minimum_op, MinimumScalar);
    LFS_SCALAR_POINTWISE_OP(ops::equal_op, EqualScalar);
    LFS_SCALAR_POINTWISE_OP(ops::not_equal_op, NotEqualScalar);
    LFS_SCALAR_POINTWISE_OP(ops::less_op, LessScalar);
    LFS_SCALAR_POINTWISE_OP(ops::less_equal_op, LessEqualScalar);
    LFS_SCALAR_POINTWISE_OP(ops::greater_op, GreaterScalar);
    LFS_SCALAR_POINTWISE_OP(ops::greater_equal_op, GreaterEqualScalar);

#undef LFS_SCALAR_POINTWISE_OP

    template <class BinaryOp, class Scalar>
    struct pointwise_op_of<ops::scalar_right_op<BinaryOp, Scalar>> {
        static constexpr PointwiseOp value = scalar_pointwise_op_of<BinaryOp>::value;
    };

    template <class BinaryOp, class Scalar>
    struct pointwise_op_of<ops::scalar_left_op<BinaryOp, Scalar>> {
        static constexpr PointwiseOp value = scalar_pointwise_op_of<BinaryOp>::value;
    };

    template <class F, class G>
    struct composed_pointwise_op_of {
        static_assert(pointwise_dependent_false<F>,
                      "this composed unary functor is not fused by the CUDA backend");
    };

    template <>
    struct composed_pointwise_op_of<ops::exp_op, pointwise_mul_scalar_functor> {
        static constexpr PointwiseOp value = PointwiseOp::ExpThenMulScalar;
    };

    template <>
    struct composed_pointwise_op_of<pointwise_mul_scalar_functor, ops::abs_op> {
        static constexpr PointwiseOp value = PointwiseOp::MulScalarThenAbs;
    };

    template <>
    struct composed_pointwise_op_of<pointwise_mul_scalar_functor, ops::relu_op> {
        static constexpr PointwiseOp value = PointwiseOp::MulScalarThenRelu;
    };

    template <class F, class G>
    struct pointwise_op_of<ops::composed_unary_op<F, G>> {
        static constexpr PointwiseOp value = composed_pointwise_op_of<F, G>::value;
    };

    template <class F, class G>
    inline constexpr bool pointwise_composition_is_fused_v =
        std::is_same_v<ops::composed_unary_op<F, G>, pointwise_exp_mul_functor> ||
        std::is_same_v<ops::composed_unary_op<F, G>, pointwise_mul_abs_functor> ||
        std::is_same_v<ops::composed_unary_op<F, G>, pointwise_mul_relu_functor>;

    template <class Scalar>
    constexpr ScalarOperand scalar_operand(const Scalar value,
                                           const bool scalar_on_right = true) {
        ScalarOperand operand{};
        operand.scalar_on_right = scalar_on_right;
        if constexpr (std::is_same_v<std::remove_cv_t<Scalar>, bool>) {
            operand.kind = ScalarKind::Bool;
            operand.value.bool_value = value;
        } else if constexpr (std::is_floating_point_v<Scalar>) {
            operand.kind = ScalarKind::Float;
            operand.value.float_value = static_cast<float>(value);
        } else if constexpr (sizeof(Scalar) <= sizeof(int32_t)) {
            operand.kind = ScalarKind::Int32;
            operand.value.int32_value = static_cast<int32_t>(value);
        } else {
            operand.kind = ScalarKind::Int64;
            operand.value.int64_value = static_cast<int64_t>(value);
        }
        return operand;
    }

    template <class Functor>
    constexpr ScalarOperand pointwise_scalar_operand(const Functor&) {
        return {};
    }

    template <class BinaryOp, class Scalar>
    constexpr ScalarOperand pointwise_scalar_operand(
        const ops::scalar_right_op<BinaryOp, Scalar>& op) {
        return scalar_operand(op.scalar, true);
    }

    template <class BinaryOp, class Scalar>
    constexpr ScalarOperand pointwise_scalar_operand(
        const ops::scalar_left_op<BinaryOp, Scalar>& op) {
        return scalar_operand(op.scalar, false);
    }

    template <class F, class G>
    constexpr ScalarOperand pointwise_scalar_operand(
        const ops::composed_unary_op<F, G>& op) {
        if constexpr (std::is_same_v<F, pointwise_mul_scalar_functor>) {
            return pointwise_scalar_operand(op.f);
        } else {
            return pointwise_scalar_operand(op.g);
        }
    }

    template <class Functor>
    constexpr PointwiseProgram pointwise_program(const DataType in_dtype,
                                                 const DataType out_dtype,
                                                 const Functor& functor) {
        return PointwiseProgram{
            .op = pointwise_op_of<std::remove_cvref_t<Functor>>::value,
            .in_dtype = in_dtype,
            .out_dtype = out_dtype,
            .scalar = pointwise_scalar_operand(functor),
        };
    }

} // namespace lfs::core::internal
