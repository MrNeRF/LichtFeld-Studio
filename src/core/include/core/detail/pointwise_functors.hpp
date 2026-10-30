/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "tensor_functors.hpp"
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

} // namespace lfs::core::internal
