/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "type_utils.h"

/** @file
 *
 * This file defines two compile-time "switches" that are suitable for
 * specializing over different values or types.
 *
 * The #TYPE_SWITCH macro switches over types and provides a typedef
 * with given name for that type in each case block.
 *
 * The #VALUE_SWITCH macro does the same with values and defines a costant
 * with given name within each case block.
 *
 * The macros can be safely (?) nested within each other.
 *
 * Types and values containing commas should be enclosed in parenthesis.
 *
 * Code blocks (case, default) must be enclosed in parenthesis if they contain commas.
 *
 * Proposed usage:
 * ```
 * #define NVIMGCODEC_TYPE_SWITCH(id, type, types, ...) \
 *    TYPE_SWITCH(id, type, types, TypeTag, (__VA_ARGS__), \
 *                (NVIMGCODEC_FAIL("Type id does not match any of " #types);)
 * ```
 *
 * TypeTag is a proposed name for mapping types to TypeId
 */

// HC SVNT DRACONES

#define NVIMGCODEC_REMOVE_PAREN_IMPL(...) __VA_ARGS__
#define NVIMGCODEC_REMOVE_PAREN(args)     NVIMGCODEC_REMOVE_PAREN_IMPL args

#define NVIMGCODEC_PP_IS_PAREN_PROBE(...) ~, 1
#define NVIMGCODEC_PP_IS_PAREN_CHECK(...) NVIMGCODEC_PP_IS_PAREN_CHECK_IMPL(__VA_ARGS__, 0)
#define NVIMGCODEC_PP_IS_PAREN_CHECK_IMPL(_0, n, ...) n
#define NVIMGCODEC_PP_IS_PAREN(args) NVIMGCODEC_PP_IS_PAREN_CHECK(NVIMGCODEC_PP_IS_PAREN_PROBE args)
#define NVIMGCODEC_REMOVE_OPTIONAL_PARENS(args) \
    NVIMGCODEC_PP_CAT(NVIMGCODEC_REMOVE_OPTIONAL_PARENS_, NVIMGCODEC_PP_IS_PAREN(args))(args)
#define NVIMGCODEC_REMOVE_OPTIONAL_PARENS_0(args) args
#define NVIMGCODEC_REMOVE_OPTIONAL_PARENS_1(args) NVIMGCODEC_REMOVE_PAREN(args)

#define NVIMGCODEC_PP_CAT_IMPL(a, b) a##b
#define NVIMGCODEC_PP_CAT(a, b)      NVIMGCODEC_PP_CAT_IMPL(a, b)

#define NVIMGCODEC_PP_NARG_IMPL(                                                                                  \
    _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, n, ...)                           \
    n
#define NVIMGCODEC_PP_NARG(...)                                                                               \
    NVIMGCODEC_PP_NARG_IMPL(__VA_ARGS__, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

#define NVIMGCODEC_PP_FOR_EACH_1(m, args, a1) m(args, a1)
#define NVIMGCODEC_PP_FOR_EACH_2(m, args, a1, a2) m(args, a1) m(args, a2)
#define NVIMGCODEC_PP_FOR_EACH_3(m, args, a1, a2, a3) m(args, a1) m(args, a2) m(args, a3)
#define NVIMGCODEC_PP_FOR_EACH_4(m, args, a1, a2, a3, a4) m(args, a1) m(args, a2) m(args, a3) m(args, a4)
#define NVIMGCODEC_PP_FOR_EACH_5(m, args, a1, a2, a3, a4, a5) m(args, a1) m(args, a2) m(args, a3) m(args, a4) m(args, a5)
#define NVIMGCODEC_PP_FOR_EACH_6(m, args, a1, a2, a3, a4, a5, a6) \
    m(args, a1) m(args, a2) m(args, a3) m(args, a4) m(args, a5) m(args, a6)
#define NVIMGCODEC_PP_FOR_EACH_7(m, args, a1, a2, a3, a4, a5, a6, a7) \
    m(args, a1) m(args, a2) m(args, a3) m(args, a4) m(args, a5) m(args, a6) m(args, a7)
#define NVIMGCODEC_PP_FOR_EACH_8(m, args, a1, a2, a3, a4, a5, a6, a7, a8) \
    m(args, a1) m(args, a2) m(args, a3) m(args, a4) m(args, a5) m(args, a6) m(args, a7) m(args, a8)
#define NVIMGCODEC_PP_FOR_EACH_9(m, args, a1, a2, a3, a4, a5, a6, a7, a8, a9) \
    m(args, a1) m(args, a2) m(args, a3) m(args, a4) m(args, a5) m(args, a6) m(args, a7) m(args, a8) m(args, a9)
#define NVIMGCODEC_PP_FOR_EACH_10(m, args, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10) \
    m(args, a1) m(args, a2) m(args, a3) m(args, a4) m(args, a5) m(args, a6) m(args, a7) m(args, a8) m(args, a9) m(args, a10)
#define NVIMGCODEC_PP_FOR_EACH_11(m, args, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) \
    m(args, a1) m(args, a2) m(args, a3) m(args, a4) m(args, a5) m(args, a6) m(args, a7) m(args, a8) m(args, a9) m(args, a10) m(args, a11)
#define NVIMGCODEC_PP_FOR_EACH_12(m, args, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12) \
    m(args, a1) m(args, a2) m(args, a3) m(args, a4) m(args, a5) m(args, a6) m(args, a7) m(args, a8) m(args, a9) m(args, a10) m(args, a11) m(args, a12)
#define NVIMGCODEC_PP_FOR_EACH_13(m, args, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13) \
    m(args, a1) m(args, a2) m(args, a3) m(args, a4) m(args, a5) m(args, a6) m(args, a7) m(args, a8) m(args, a9) m(args, a10) m(args, a11) m(args, a12) m(args, a13)
#define NVIMGCODEC_PP_FOR_EACH_14(m, args, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14) \
    m(args, a1) m(args, a2) m(args, a3) m(args, a4) m(args, a5) m(args, a6) m(args, a7) m(args, a8) m(args, a9) m(args, a10) m(args, a11) m(args, a12) m(args, a13) m(args, a14)
#define NVIMGCODEC_PP_FOR_EACH_15(m, args, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15) \
    m(args, a1) m(args, a2) m(args, a3) m(args, a4) m(args, a5) m(args, a6) m(args, a7) m(args, a8) m(args, a9) m(args, a10) m(args, a11) m(args, a12) m(args, a13) m(args, a14) m(args, a15)
#define NVIMGCODEC_PP_FOR_EACH_16(m, args, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16) \
    m(args, a1) m(args, a2) m(args, a3) m(args, a4) m(args, a5) m(args, a6) m(args, a7) m(args, a8) m(args, a9) m(args, a10) m(args, a11) m(args, a12) m(args, a13) m(args, a14) m(args, a15) m(args, a16)
#define NVIMGCODEC_PP_FOR_EACH(m, args, ...) \
    NVIMGCODEC_PP_CAT(NVIMGCODEC_PP_FOR_EACH_, NVIMGCODEC_PP_NARG(__VA_ARGS__))(m, args, __VA_ARGS__)

#define NVIMGCODEC_TYPE_SWITCH_IMPL3(type_, type_tag_, type_name_, code_, ...) \
    case type_tag_<NVIMGCODEC_REMOVE_OPTIONAL_PARENS(type_)>::value: {          \
        using type_name_ = NVIMGCODEC_REMOVE_OPTIONAL_PARENS(type_);            \
        NVIMGCODEC_REMOVE_OPTIONAL_PARENS(code_);                               \
        __VA_ARGS__                                                            \
    } break;

#define NVIMGCODEC_TYPE_SWITCH_IMPL2(...) NVIMGCODEC_TYPE_SWITCH_IMPL3(__VA_ARGS__)

#define NVIMGCODEC_TYPE_SWITCH_IMPL(args, type) NVIMGCODEC_TYPE_SWITCH_IMPL2(type, NVIMGCODEC_REMOVE_PAREN(args))

/// Pastes the case_ code specialized for each type in types_.
/// The specialization is performed by aliasing a particular type with a typedef named type_name_.
/// @param id_         - numerical id of the type
/// @param type_tag_   - a class template usable as type_tag<type>::value
///                      the value should be a type id for type
/// @param type_name_  - a name given for selected type in the switch
/// @param types_      - parenthesised, comma-separated list of types
///                      types containing commas should be enclosed with parenthesis
///                      e.g. (int, float, (std::conditional<val, bool, char>::type))
/// @param case_       - code to execute for matching cases
/// @param default_    - code to execute when id doesn't match any type in types
///
/// Usage:
/// ```
/// TYPE_SWITCH(input_type, TypeTag, IType, (int, float, (some_type<args>::type), int64_t), (
///    TYPE_SWITCH(output_type, TypeTag, OType, (int, double, int64_t), (
///        VALUE_SWITCH(channels, num_channels, (1, 2, 3, 4), (
///            SomeFunctor<IType, OType, num_channels>(
///            inputs[0].data<IType>(), outputs[0].mutable_data<OType>());
///          ), assert(!"Unsupported number of channels");
///        )
///      ), assert(!"Unsupported output type");
///    )
///  ), assert(!"Unsupported input type");
/// )
/// ```
#define TYPE_SWITCH(id_, type_tag_, type_name_, types, case_, default_)                    \
    switch (id_) {                                                                         \
        NVIMGCODEC_PP_FOR_EACH(NVIMGCODEC_TYPE_SWITCH_IMPL, (type_tag_, type_name_, case_), \
                               NVIMGCODEC_REMOVE_PAREN(types))                            \
    default: {                                                                             \
        NVIMGCODEC_REMOVE_OPTIONAL_PARENS(default_);                                      \
    }                                                                                      \
    }

#define NVIMGCODEC_VALUE_SWITCH_IMPL3(value_, value_name_, code_) \
    case value_: {                                                \
        const auto value_name_ = value_;                          \
        NVIMGCODEC_REMOVE_OPTIONAL_PARENS(code_);                  \
    } break;

#define NVIMGCODEC_VALUE_SWITCH_IMPL2(...) NVIMGCODEC_VALUE_SWITCH_IMPL3(__VA_ARGS__)

#define NVIMGCODEC_VALUE_SWITCH_IMPL(args, value) \
    NVIMGCODEC_VALUE_SWITCH_IMPL2(value, NVIMGCODEC_REMOVE_PAREN(args))

/// Pastes the case_ code specialized for each value in values.
/// The specialization is performed by aliasing a value with a constant named constant_name_
/// @param value_         - a value to switch by
/// @param constant_name_ - a name given for selected type in the switch
/// @param values_        - parenthesised, comma-separated list of case labels;
///                         expressions containing commas should be enclosed with parenthesis
/// @param case_          - code to execute for matching cases
/// @param default_       - code to execute when value doesn't match any in values
///
/// Usage:
/// ```
/// VALUE_SWITCH(channels, num_channels, (1, 2, 3, 4), (
///     SomeFunctor<IType, OType, num_channels>(
///     inputs[0].data<IType>(), outputs[0].mutable_data<OType>());
///   ), assert(!"Unsupported number of channels");
/// )
/// ```
#define VALUE_SWITCH(value_, value_name_, values, case_, default_)                       \
    switch (value_) {                                                                    \
        NVIMGCODEC_PP_FOR_EACH(NVIMGCODEC_VALUE_SWITCH_IMPL, (value_name_, case_),       \
                               NVIMGCODEC_REMOVE_PAREN(values))                         \
    default: {                                                                           \
        NVIMGCODEC_REMOVE_OPTIONAL_PARENS(default_);                                    \
    }                                                                                    \
    }

/// Pastes the case_ code specialized for true and false values.
/// The specialization is performed by aliasing a value with a constant named constant_name_
/// @param expr_       - a boolean expression to switch by
/// @param const_name_ - a name given for the constexpr bool variable.
/// @param code_       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, (
///     std::integral_constant<bool, BoolConst> constant;
///     some_function<BoolConst>(...);
///   )
/// )
/// ```
#define BOOL_SWITCH(expr_, const_name_, code_) \
    if (expr_) {                               \
        constexpr bool const_name_ = true;     \
        NVIMGCODEC_REMOVE_OPTIONAL_PARENS(code_); \
    } else {                                   \
        constexpr bool const_name_ = false;    \
        NVIMGCODEC_REMOVE_OPTIONAL_PARENS(code_); \
    }
