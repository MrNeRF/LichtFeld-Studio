/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cmath>
#include <type_traits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#include <thrust/tuple.h>
#else
#define HOST_DEVICE
#include <tuple>
#endif

namespace lfs::core {
    namespace ops {

        // Helper for clamping (std::clamp not always available in CUDA)
        template <typename T>
        HOST_DEVICE constexpr T clamp_value(const T& v, const T& lo, const T& hi) {
#ifdef __CUDA_ARCH__
            return fminf(fmaxf(v, lo), hi);
#else
            return (v < lo) ? lo : (hi < v) ? hi
                                            : v;
#endif
        }

        // ============= UNARY OPERATIONS =============

        struct neg_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
                return -x;
            }
        };

        struct abs_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return fabsf(x);
#else
                return std::abs(x);
#endif
            }
        };

        struct sign_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
                return T((x > T(0)) - (x < T(0)));
            }
        };

        struct reciprocal_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
                return T(1) / (x + T(1e-8));
            }
        };

        struct inverse_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return T(1) / fmaxf(fabsf(x), T(1e-10));
#else
                return T(1) / std::max(std::abs(x), T(1e-10));
#endif
            }
        };

        struct exp_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return expf(x);
#else
                return std::exp(x);
#endif
            }
        };

        struct exp2_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return exp2f(x);
#else
                return std::exp2(x);
#endif
            }
        };

        struct log_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return logf(fmaxf(x, T(1e-10)));
#else
                return std::log(std::max(x, T(1e-10)));
#endif
            }
        };

        struct log2_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return log2f(fmaxf(x, T(1e-10)));
#else
                return std::log2(std::max(x, T(1e-10)));
#endif
            }
        };

        struct log10_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return log10f(fmaxf(x, T(1e-10)));
#else
                return std::log10(std::max(x, T(1e-10)));
#endif
            }
        };

        struct log1p_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return log1pf(x);
#else
                return std::log1p(x);
#endif
            }
        };

        struct sqrt_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return sqrtf(fmaxf(x, T(0)));
#else
                return std::sqrt(std::max(x, T(0)));
#endif
            }
        };

        struct rsqrt_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return rsqrtf(fmaxf(x, T(1e-10)));
#else
                return T(1) / std::sqrt(std::max(x, T(1e-10)));
#endif
            }
        };

        struct cbrt_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return cbrtf(x);
#else
                return std::cbrt(x);
#endif
            }
        };

        struct square_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
                return x * x;
            }
        };

        struct sin_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return sinf(x);
#else
                return std::sin(x);
#endif
            }
        };

        struct cos_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return cosf(x);
#else
                return std::cos(x);
#endif
            }
        };

        struct tan_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return tanf(x);
#else
                return std::tan(x);
#endif
            }
        };

        struct asin_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return asinf(fminf(fmaxf(x, T(-1)), T(1)));
#else
                return std::asin(clamp_value(x, T(-1), T(1)));
#endif
            }
        };

        struct acos_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return acosf(fminf(fmaxf(x, T(-1)), T(1)));
#else
                return std::acos(clamp_value(x, T(-1), T(1)));
#endif
            }
        };

        struct atan_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return atanf(x);
#else
                return std::atan(x);
#endif
            }
        };

        struct sinh_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return sinhf(x);
#else
                return std::sinh(x);
#endif
            }
        };

        struct cosh_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return coshf(x);
#else
                return std::cosh(x);
#endif
            }
        };

        struct tanh_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return tanhf(x);
#else
                return std::tanh(x);
#endif
            }
        };

        struct sigmoid_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return T(1) / (T(1) + expf(-x));
#else
                return T(1) / (T(1) + std::exp(-x));
#endif
            }
        };

        struct relu_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return fmaxf(x, T(0));
#else
                return std::max(x, T(0));
#endif
            }
        };

        struct gelu_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                T inner = sqrtf(T(2) / T(M_PI)) * (x + T(0.044715) * x * x * x);
                return T(0.5) * x * (T(1) + tanhf(inner));
#else
                T inner = std::sqrt(T(2) / T(M_PI)) * (x + T(0.044715) * x * x * x);
                return T(0.5) * x * (T(1) + std::tanh(inner));
#endif
            }
        };

        struct swish_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return x / (T(1) + expf(-x));
#else
                return x / (T(1) + std::exp(-x));
#endif
            }
        };

        struct floor_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return floorf(x);
#else
                return std::floor(x);
#endif
            }
        };

        struct ceil_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return ceilf(x);
#else
                return std::ceil(x);
#endif
            }
        };

        struct round_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return roundf(x);
#else
                return std::round(x);
#endif
            }
        };

        struct trunc_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return truncf(x);
#else
                return std::trunc(x);
#endif
            }
        };

        struct frac_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return x - floorf(x);
#else
                return x - std::floor(x);
#endif
            }
        };

        struct logical_not_op {
            template <typename T>
            HOST_DEVICE constexpr unsigned char operator()(const T& x) const {
                return !x;
            }
        };

        struct isnan_op {
            template <typename T>
            HOST_DEVICE constexpr unsigned char operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                // Use unqualified name in device code - resolves to CUDA math function
                return isnan(static_cast<float>(x)) ? 1 : 0;
#else
                return std::isnan(x) ? 1 : 0;
#endif
            }
        };

        struct isinf_op {
            template <typename T>
            HOST_DEVICE constexpr unsigned char operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                // Use unqualified name in device code - resolves to CUDA math function
                return isinf(static_cast<float>(x)) ? 1 : 0;
#else
                return std::isinf(x) ? 1 : 0;
#endif
            }
        };

        struct isfinite_op {
            template <typename T>
            HOST_DEVICE constexpr unsigned char operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                // Use unqualified name in device code - resolves to CUDA math function
                return isfinite(static_cast<float>(x)) ? 1 : 0;
#else
                return std::isfinite(x) ? 1 : 0;
#endif
            }
        };

        // ============= BINARY OPERATIONS =============

        struct add_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
                return a + b;
            }
        };

        struct sub_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
                return a - b;
            }
        };

        struct mul_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
                return a * b;
            }
        };

        struct div_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
                return a / b;
            }
        };

        struct pow_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
                // Special case: For squaring (pow 2.0), use multiplication to handle negative bases correctly
                // powf(negative, 2.0) returns NaN because it's implemented as exp(b*log(a))
                if (b == static_cast<T>(2.0)) {
                    return a * a;
                }
#ifdef __CUDA_ARCH__
                return powf(a, b);
#else
                return static_cast<T>(std::pow(a, b));
#endif
            }
        };

        struct mod_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
#ifdef __CUDA_ARCH__
                return fmodf(a, b);
#else
                return std::fmod(a, b);
#endif
            }
        };

        struct fmod_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
#ifdef __CUDA_ARCH__
                return fmodf(a, b);
#else
                return std::fmod(a, b);
#endif
            }
        };

        struct remainder_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
#ifdef __CUDA_ARCH__
                return remainderf(a, b);
#else
                return std::remainder(a, b);
#endif
            }
        };

        struct maximum_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
#ifdef __CUDA_ARCH__
                return fmaxf(a, b);
#else
                return std::max(a, b);
#endif
            }
        };

        struct minimum_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
#ifdef __CUDA_ARCH__
                return fminf(a, b);
#else
                return std::min(a, b);
#endif
            }
        };

        struct atan2_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& y, const T& x) const {
#ifdef __CUDA_ARCH__
                return atan2f(y, x);
#else
                return std::atan2(y, x);
#endif
            }
        };

        struct hypot_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
#ifdef __CUDA_ARCH__
                return hypotf(a, b);
#else
                return std::hypot(a, b);
#endif
            }
        };

        // ============= COMPARISON OPERATIONS =============

        struct equal_op {
            template <typename T>
            HOST_DEVICE constexpr unsigned char operator()(const T& a, const T& b) const {
                return (a == b) ? 1 : 0;
            }
        };

        struct not_equal_op {
            template <typename T>
            HOST_DEVICE constexpr unsigned char operator()(const T& a, const T& b) const {
                return (a != b) ? 1 : 0;
            }
        };

        struct less_op {
            template <typename T>
            HOST_DEVICE constexpr unsigned char operator()(const T& a, const T& b) const {
                return (a < b) ? 1 : 0;
            }
        };

        struct less_equal_op {
            template <typename T>
            HOST_DEVICE constexpr unsigned char operator()(const T& a, const T& b) const {
                return (a <= b) ? 1 : 0;
            }
        };

        struct greater_op {
            template <typename T>
            HOST_DEVICE constexpr unsigned char operator()(const T& a, const T& b) const {
                return (a > b) ? 1 : 0;
            }
        };

        struct greater_equal_op {
            template <typename T>
            HOST_DEVICE constexpr unsigned char operator()(const T& a, const T& b) const {
                return (a >= b) ? 1 : 0;
            }
        };

        // ============= LOGICAL OPERATIONS =============

        struct logical_and_op {
            template <typename T>
            HOST_DEVICE constexpr unsigned char operator()(const T& a, const T& b) const {
                return (a && b) ? 1 : 0;
            }
        };

        struct logical_or_op {
            template <typename T>
            HOST_DEVICE constexpr unsigned char operator()(const T& a, const T& b) const {
                return (a || b) ? 1 : 0;
            }
        };

        struct logical_xor_op {
            template <typename T>
            HOST_DEVICE constexpr unsigned char operator()(const T& a, const T& b) const {
                return (static_cast<bool>(a) != static_cast<bool>(b)) ? 1 : 0;
            }
        };

        struct bitwise_or_op {
            template <typename T>
            HOST_DEVICE constexpr unsigned char operator()(const T& a, const T& b) const {
                return (a || b) ? 1 : 0;
            }
        };

        struct bitwise_and_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
                return static_cast<T>(static_cast<int>(a) & static_cast<int>(b));
            }
        };

        struct bitwise_xor_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
                return static_cast<T>(static_cast<int>(a) ^ static_cast<int>(b));
            }
        };

        struct bitwise_not_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x) const {
                return static_cast<T>(~static_cast<int>(x));
            }
        };

        struct left_shift_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
                return static_cast<T>(static_cast<int>(a) << static_cast<int>(b));
            }
        };

        struct right_shift_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
                return static_cast<T>(static_cast<int>(a) >> static_cast<int>(b));
            }
        };

        // ============= SCALAR OPERATIONS (RIGHT) =============

        template <typename BinOp, typename T>
        struct scalar_right_op {
            T scalar;
            BinOp op;

            HOST_DEVICE constexpr scalar_right_op(T s) : scalar(s),
                                                         op() {}

            template <typename U>
            HOST_DEVICE constexpr auto operator()(const U& x) const {
                return op(x, static_cast<U>(scalar));
            }
        };

        // ============= SCALAR OPERATIONS (LEFT) =============

        template <typename BinOp, typename T>
        struct scalar_left_op {
            T scalar;
            BinOp op;

            HOST_DEVICE constexpr scalar_left_op(T s) : scalar(s),
                                                        op() {}

            template <typename U>
            HOST_DEVICE constexpr auto operator()(const U& x) const {
                return op(static_cast<U>(scalar), x);
            }
        };

        // ============= SCALAR COMPARISON OPERATIONS =============

        template <typename T>
        struct equal_scalar_op {
            T val;
            HOST_DEVICE constexpr equal_scalar_op(T v) : val(v) {}
            HOST_DEVICE constexpr unsigned char operator()(const T& x) const {
                return (x == val) ? 1 : 0;
            }
        };

        template <typename T>
        struct not_equal_scalar_op {
            T val;
            HOST_DEVICE constexpr not_equal_scalar_op(T v) : val(v) {}
            HOST_DEVICE constexpr unsigned char operator()(const T& x) const {
                return (x != val) ? 1 : 0;
            }
        };

        template <typename T>
        struct less_scalar_op {
            T val;
            HOST_DEVICE constexpr less_scalar_op(T v) : val(v) {}
            HOST_DEVICE constexpr unsigned char operator()(const T& x) const {
                return (x < val) ? 1 : 0;
            }
        };

        template <typename T>
        struct less_equal_scalar_op {
            T val;
            HOST_DEVICE constexpr less_equal_scalar_op(T v) : val(v) {}
            HOST_DEVICE constexpr unsigned char operator()(const T& x) const {
                return (x <= val) ? 1 : 0;
            }
        };

        template <typename T>
        struct greater_scalar_op {
            T val;
            HOST_DEVICE constexpr greater_scalar_op(T v) : val(v) {}
            HOST_DEVICE constexpr unsigned char operator()(const T& x) const {
                return (x > val) ? 1 : 0;
            }
        };

        template <typename T>
        struct greater_equal_scalar_op {
            T val;
            HOST_DEVICE constexpr greater_equal_scalar_op(T v) : val(v) {}
            HOST_DEVICE constexpr unsigned char operator()(const T& x) const {
                return (x >= val) ? 1 : 0;
            }
        };

        // ============= REDUCTION OPERATIONS =============

        struct sum_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
                return a + b;
            }
        };

        struct prod_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
                return a * b;
            }
        };

        struct max_reduce_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
#ifdef __CUDA_ARCH__
                return fmaxf(a, b);
#else
                return std::max(a, b);
#endif
            }
        };

        struct min_reduce_op {
            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& a, const T& b) const {
#ifdef __CUDA_ARCH__
                return fminf(a, b);
#else
                return std::min(a, b);
#endif
            }
        };

        struct any_op {
            HOST_DEVICE constexpr bool operator()(bool a, bool b) const {
                return a || b;
            }
            template <typename T>
            HOST_DEVICE constexpr bool operator()(const T& a, const T& b) const {
                return (a != T(0)) || (b != T(0));
            }
        };

        struct all_op {
            HOST_DEVICE constexpr bool operator()(bool a, bool b) const {
                return a && b;
            }
            template <typename T>
            HOST_DEVICE constexpr bool operator()(const T& a, const T& b) const {
                return (a != T(0)) && (b != T(0));
            }
        };

        // ============= TYPE CONVERSION OPERATIONS =============

        template <typename SrcT, typename DstT>
        struct convert_op {
            HOST_DEVICE constexpr DstT operator()(const SrcT& x) const {
                return static_cast<DstT>(x);
            }
        };

        template <typename DstT>
        struct cast_op {
            template <typename SrcT>
            HOST_DEVICE constexpr DstT operator()(const SrcT& x) const {
                return static_cast<DstT>(x);
            }
        };

        struct to_bool_op {
            template <typename T>
            HOST_DEVICE constexpr unsigned char operator()(const T& x) const {
                return (x != T(0)) ? 1 : 0;
            }
        };

        template <typename T>
        struct from_bool_op {
            HOST_DEVICE constexpr T operator()(unsigned char x) const {
                return x ? T(1) : T(0);
            }
        };

        // ============= CLAMPING OPERATIONS =============

        template <typename T>
        struct clamp_min_op {
            T min_val;
            HOST_DEVICE constexpr clamp_min_op(T m) : min_val(m) {}
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return fmaxf(x, min_val);
#else
                return std::max(x, min_val);
#endif
            }
        };

        template <typename T>
        struct clamp_max_op {
            T max_val;
            HOST_DEVICE constexpr clamp_max_op(T m) : max_val(m) {}
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return fminf(x, max_val);
#else
                return std::min(x, max_val);
#endif
            }
        };

        template <typename T>
        struct clamp_range_op {
            T min_val;
            T max_val;
            HOST_DEVICE constexpr clamp_range_op(T min_v, T max_v) : min_val(min_v),
                                                                     max_val(max_v) {}
            HOST_DEVICE constexpr T operator()(const T& x) const {
#ifdef __CUDA_ARCH__
                return fminf(fmaxf(x, min_val), max_val);
#else
                return clamp_value(x, min_val, max_val);
#endif
            }
        };

        // ============= TERNARY/CONDITIONAL OPERATIONS =============

        template <typename T>
        struct where_op {
            HOST_DEVICE constexpr T operator()(unsigned char cond, const T& x, const T& y) const {
                return cond ? x : y;
            }
        };

        template <typename T>
        struct lerp_op {
            HOST_DEVICE constexpr T operator()(const T& a, const T& b, const T& t) const {
                return a + t * (b - a);
            }
        };

        template <typename T>
        struct clamp_ternary_op {
            HOST_DEVICE constexpr T operator()(const T& x, const T& min_val, const T& max_val) const {
#ifdef __CUDA_ARCH__
                return fminf(fmaxf(x, min_val), max_val);
#else
                return clamp_value(x, min_val, max_val);
#endif
            }
        };

        // ============= TUPLE OPERATIONS =============
        // Note: These work with thrust::tuple in CUDA and std::tuple on host

        template <size_t N>
        struct tuple_get_op {
            template <typename Tuple>
            HOST_DEVICE constexpr auto operator()(const Tuple& t) const {
#ifdef __CUDACC__
                return thrust::get<N>(t);
#else
                return std::get<N>(t);
#endif
            }
        };

        struct tuple_first_op {
            template <typename Tuple>
            HOST_DEVICE constexpr auto operator()(const Tuple& t) const {
#ifdef __CUDACC__
                return thrust::get<0>(t);
#else
                return std::get<0>(t);
#endif
            }
        };

        struct tuple_second_op {
            template <typename Tuple>
            HOST_DEVICE constexpr auto operator()(const Tuple& t) const {
#ifdef __CUDACC__
                return thrust::get<1>(t);
#else
                return std::get<1>(t);
#endif
            }
        };

        template <typename BinOp>
        struct tuple_apply_binary_op {
            BinOp op;
            HOST_DEVICE constexpr tuple_apply_binary_op() : op() {}

            template <typename Tuple>
            HOST_DEVICE constexpr auto operator()(const Tuple& t) const {
#ifdef __CUDACC__
                return op(thrust::get<0>(t), thrust::get<1>(t));
#else
                return op(std::get<0>(t), std::get<1>(t));
#endif
            }
        };

        // ============= INDEX/MASKING OPERATIONS =============

        struct index_clamp_op {
            size_t size;
            HOST_DEVICE constexpr index_clamp_op(size_t s) : size(s) {}
            HOST_DEVICE constexpr size_t operator()(int idx) const {
                if (idx < 0)
                    idx += size;
                if (idx < 0)
                    return 0;
                if (idx >= static_cast<int>(size))
                    return size - 1;
                return static_cast<size_t>(idx);
            }
        };

        struct index_wrap_op {
            size_t size;
            HOST_DEVICE constexpr index_wrap_op(size_t s) : size(s) {}
            HOST_DEVICE constexpr size_t operator()(int idx) const {
                int s = static_cast<int>(size);
                return static_cast<size_t>(((idx % s) + s) % s);
            }
        };

        template <typename T>
        struct masked_value_op {
            T default_val;
            HOST_DEVICE constexpr masked_value_op(T d) : default_val(d) {}
            HOST_DEVICE constexpr T operator()(const T& x, unsigned char mask) const {
                return mask ? x : default_val;
            }
        };

        template <typename Predicate>
        struct select_if_op {
            Predicate pred;
            HOST_DEVICE constexpr select_if_op(Predicate p) : pred(p) {}

            template <typename T>
            HOST_DEVICE constexpr T operator()(const T& x, const T& y) const {
                return pred(x) ? x : y;
            }
        };

        template <typename T>
        struct masked_fill_op {
            T val;
            HOST_DEVICE constexpr masked_fill_op(T v) : val(v) {}

            // For use with zip iterators - tuple input
            template <typename Tuple>
            HOST_DEVICE constexpr T operator()(const Tuple& t) const {
#ifdef __CUDACC__
                return thrust::get<1>(t) ? val : thrust::get<0>(t);
#else
                return std::get<1>(t) ? val : std::get<0>(t);
#endif
            }

            // For direct use - separate arguments
            HOST_DEVICE constexpr T operator()(const T& x, unsigned char mask) const {
                return mask ? val : x;
            }
        };

        struct extract_value_op {
            template <typename ValueMaskPair>
            HOST_DEVICE constexpr auto operator()(const ValueMaskPair& p) const {
#ifdef __CUDACC__
                return thrust::get<0>(p);
#else
                return std::get<0>(p);
#endif
            }
        };

        struct extract_mask_op {
            template <typename ValueMaskPair>
            HOST_DEVICE constexpr bool operator()(const ValueMaskPair& p) const {
#ifdef __CUDACC__
                return thrust::get<1>(p) != 0;
#else
                return std::get<1>(p) != 0;
#endif
            }
        };

        // ============= PREDICATE OPERATIONS =============

        template <typename T>
        struct is_nonzero_op {
            HOST_DEVICE constexpr bool operator()(const T& x) const {
                return x != T(0);
            }
        };

        struct is_nonzero_bool_op {
            HOST_DEVICE constexpr bool operator()(unsigned char x) const {
                return x != 0;
            }
        };

        template <typename T>
        struct nonzero_predicate {
            HOST_DEVICE constexpr bool operator()(const T& x) const {
                return x != T(0);
            }
        };

        struct nonzero_bool_predicate {
            HOST_DEVICE constexpr bool operator()(unsigned char x) const {
                return x != 0;
            }
        };

        // ============= LOAD/FILL OPERATIONS =============

        template <typename T>
        struct constant_op {
            T value;
            HOST_DEVICE constexpr constant_op(T v) : value(v) {}
            HOST_DEVICE constexpr T operator()(size_t) const {
                return value;
            }
            template <typename U>
            HOST_DEVICE constexpr T operator()(const U&) const {
                return value;
            }
        };

        template <typename T>
        struct iota_op {
            T start;
            T step;
            HOST_DEVICE constexpr iota_op(T s = T(0), T st = T(1)) : start(s),
                                                                     step(st) {}
            HOST_DEVICE constexpr T operator()(size_t idx) const {
                return start + static_cast<T>(idx) * step;
            }
        };

        template <typename T>
        struct linspace_op {
            T start;
            T end;
            size_t steps;
            HOST_DEVICE constexpr linspace_op(T s, T e, size_t n) : start(s),
                                                                    end(e),
                                                                    steps(n) {}
            HOST_DEVICE constexpr T operator()(size_t idx) const {
                if (steps <= 1)
                    return start;
                T t = static_cast<T>(idx) / static_cast<T>(steps - 1);
                return start + t * (end - start);
            }
        };

        // ============= FUNCTOR COMPOSITION (for kernel fusion) =============

        // Compose two unary functors: g(f(x))
        template <typename F, typename G>
        struct composed_unary_op {
            F f; // First operation
            G g; // Second operation

            HOST_DEVICE constexpr composed_unary_op() : f(),
                                                        g() {}
            HOST_DEVICE constexpr composed_unary_op(F f_, G g_) : f(f_),
                                                                  g(g_) {}

            template <typename T>
            HOST_DEVICE constexpr auto operator()(const T& x) const {
                return g(f(x)); // Apply f, then g
            }
        };

        // Helper to create composed operations
        template <typename F, typename G>
        HOST_DEVICE constexpr auto compose(F f, G g) {
            return composed_unary_op<F, G>{f, g};
        }

        // Compose three unary functors: h(g(f(x)))
        template <typename F, typename G, typename H>
        struct composed_unary_op_3 {
            F f;
            G g;
            H h;

            HOST_DEVICE constexpr composed_unary_op_3() : f(),
                                                          g(),
                                                          h() {}
            HOST_DEVICE constexpr composed_unary_op_3(F f_, G g_, H h_) : f(f_),
                                                                          g(g_),
                                                                          h(h_) {}

            template <typename T>
            HOST_DEVICE constexpr auto operator()(const T& x) const {
                return h(g(f(x)));
            }
        };

        // Helper to create 3-way compositions
        template <typename F, typename G, typename H>
        HOST_DEVICE constexpr auto compose(F f, G g, H h) {
            return composed_unary_op_3<F, G, H>{f, g, h};
        }

        // Compose four unary functors: k(h(g(f(x))))
        template <typename F, typename G, typename H, typename K>
        struct composed_unary_op_4 {
            F f;
            G g;
            H h;
            K k;

            HOST_DEVICE constexpr composed_unary_op_4() : f(),
                                                          g(),
                                                          h(),
                                                          k() {}
            HOST_DEVICE constexpr composed_unary_op_4(F f_, G g_, H h_, K k_) : f(f_),
                                                                                g(g_),
                                                                                h(h_),
                                                                                k(k_) {}

            template <typename T>
            HOST_DEVICE constexpr auto operator()(const T& x) const {
                return k(h(g(f(x))));
            }
        };

        // Helper to create 4-way compositions
        template <typename F, typename G, typename H, typename K>
        HOST_DEVICE constexpr auto compose(F f, G g, H h, K k) {
            return composed_unary_op_4<F, G, H, K>{f, g, h, k};
        }

        // ============= TYPE TRAITS FOR BOOL-RETURNING OPERATIONS =============

        // Default: operations return the same type as input
        template <typename Op>
        struct returns_bool : std::false_type {};

        // Specialize for Bool-returning unary operations
        template <>
        struct returns_bool<isnan_op> : std::true_type {};
        template <>
        struct returns_bool<isinf_op> : std::true_type {};
        template <>
        struct returns_bool<isfinite_op> : std::true_type {};
        template <>
        struct returns_bool<logical_not_op> : std::true_type {};

        // Specialize for Bool-returning comparison operations (binary)
        template <>
        struct returns_bool<equal_op> : std::true_type {};
        template <>
        struct returns_bool<not_equal_op> : std::true_type {};
        template <>
        struct returns_bool<less_op> : std::true_type {};
        template <>
        struct returns_bool<less_equal_op> : std::true_type {};
        template <>
        struct returns_bool<greater_op> : std::true_type {};
        template <>
        struct returns_bool<greater_equal_op> : std::true_type {};

        // Specialize for Bool-returning logical operations (binary)
        template <>
        struct returns_bool<logical_and_op> : std::true_type {};
        template <>
        struct returns_bool<logical_or_op> : std::true_type {};
        template <>
        struct returns_bool<logical_xor_op> : std::true_type {};
        template <>
        struct returns_bool<bitwise_or_op> : std::true_type {};

        // Specialize for scalar comparison operations
        template <typename T>
        struct returns_bool<equal_scalar_op<T>> : std::true_type {};
        template <typename T>
        struct returns_bool<not_equal_scalar_op<T>> : std::true_type {};
        template <typename T>
        struct returns_bool<less_scalar_op<T>> : std::true_type {};
        template <typename T>
        struct returns_bool<less_equal_scalar_op<T>> : std::true_type {};
        template <typename T>
        struct returns_bool<greater_scalar_op<T>> : std::true_type {};
        template <typename T>
        struct returns_bool<greater_equal_scalar_op<T>> : std::true_type {};

        // Specialize for scalar_right_op wrapping Bool-returning operations
        template <typename BinOp, typename T>
        struct returns_bool<scalar_right_op<BinOp, T>> : returns_bool<BinOp> {};

        template <typename BinOp, typename T>
        struct returns_bool<scalar_left_op<BinOp, T>> : returns_bool<BinOp> {};

        // Helper variable template (C++17)
        template <typename Op>
        inline constexpr bool returns_bool_v = returns_bool<Op>::value;

    } // namespace ops
} // namespace lfs::core

#undef HOST_DEVICE