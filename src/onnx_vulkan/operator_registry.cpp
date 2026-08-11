/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "operator_registry.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace lfs::onnx_vulkan::detail {
    namespace {
        using AT = AttributeType;
        constexpr std::size_t many = std::numeric_limits<std::size_t>::max();

        constexpr std::array<AttributeSchema, 1> cast_attrs{{{"to", AT::Int, true}}};
        constexpr std::array<AttributeSchema, 1> concat_attrs{{{"axis", AT::Int, true}}};
        constexpr std::array<AttributeSchema, 8> constant_attrs{{
            {"value", AT::Tensor}, {"value_float", AT::Float}, {"value_floats", AT::Floats},
            {"value_int", AT::Int}, {"value_ints", AT::Ints}, {"value_string", AT::String},
            {"value_strings", AT::Strings}, {"sparse_value", AT::Undefined},
        }};
        constexpr std::array<AttributeSchema, 1> constant_of_shape_attrs{{{"value", AT::Tensor}}};
        constexpr std::array<AttributeSchema, 6> conv_attrs{{
            {"auto_pad", AT::String}, {"dilations", AT::Ints}, {"group", AT::Int},
            {"kernel_shape", AT::Ints}, {"pads", AT::Ints}, {"strides", AT::Ints},
        }};
        constexpr std::array<AttributeSchema, 8> conv_transpose_attrs{{
            {"auto_pad", AT::String}, {"dilations", AT::Ints}, {"group", AT::Int},
            {"kernel_shape", AT::Ints}, {"output_padding", AT::Ints}, {"output_shape", AT::Ints},
            {"pads", AT::Ints}, {"strides", AT::Ints},
        }};
        constexpr std::array<AttributeSchema, 1> gather_attrs{{{"axis", AT::Int}}};
        constexpr std::array<AttributeSchema, 4> gemm_attrs{{
            {"alpha", AT::Float}, {"beta", AT::Float}, {"transA", AT::Int}, {"transB", AT::Int},
        }};
        constexpr std::array<AttributeSchema, 2> if_attrs{{
            {"else_branch", AT::Graph, true}, {"then_branch", AT::Graph, true},
        }};
        constexpr std::array<AttributeSchema, 1> mod_attrs{{{"fmod", AT::Int}}};
        constexpr std::array<AttributeSchema, 3> pad_attrs{{
            {"mode", AT::String}, {"pads", AT::Ints}, {"value", AT::Float},
        }};
        constexpr std::array<AttributeSchema, 2> reduction_attrs{{
            {"axes", AT::Ints}, {"keepdims", AT::Int},
        }};
        constexpr std::array<AttributeSchema, 3> reduce_sum_attrs{{
            {"axes", AT::Ints}, {"keepdims", AT::Int}, {"noop_with_empty_axes", AT::Int},
        }};
        constexpr std::array<AttributeSchema, 1> reshape_attrs{{{"allowzero", AT::Int}}};
        constexpr std::array<AttributeSchema, 6> resize_attrs{{
            {"coordinate_transformation_mode", AT::String}, {"cubic_coeff_a", AT::Float},
            {"exclude_outside", AT::Int}, {"extrapolation_value", AT::Float},
            {"mode", AT::String}, {"nearest_mode", AT::String},
        }};
        constexpr std::array<AttributeSchema, 2> shape_attrs{{{"end", AT::Int}, {"start", AT::Int}}};
        constexpr std::array<AttributeSchema, 1> softmax_attrs{{{"axis", AT::Int}}};
        constexpr std::array<AttributeSchema, 2> split_attrs{{{"axis", AT::Int}, {"split", AT::Ints}}};
        constexpr std::array<AttributeSchema, 1> transpose_attrs{{{"perm", AT::Ints}}};

        constexpr std::array<OperatorSchema, 45> registry{{
            {"Add", 14, 14, 2, 2, 1, 1, {}},
            {"Cast", 13, 14, 1, 1, 1, 1, cast_attrs},
            {"Clip", 11, 14, 1, 3, 1, 1, {}},
            {"Concat", 13, 14, 1, many, 1, 1, concat_attrs},
            {"Constant", 13, 14, 0, 0, 1, 1, constant_attrs},
            {"ConstantOfShape", 9, 14, 1, 1, 1, 1, constant_of_shape_attrs},
            {"Conv", 11, 14, 2, 3, 1, 1, conv_attrs},
            {"ConvTranspose", 11, 14, 2, 3, 1, 1, conv_transpose_attrs},
            {"Div", 14, 14, 2, 2, 1, 1, {}},
            {"Erf", 13, 14, 1, 1, 1, 1, {}},
            {"Equal", 13, 14, 2, 2, 1, 1, {}},
            {"Exp", 13, 14, 1, 1, 1, 1, {}},
            {"Expand", 13, 14, 2, 2, 1, 1, {}},
            {"Gather", 13, 14, 2, 2, 1, 1, gather_attrs},
            {"Gemm", 13, 14, 2, 3, 1, 1, gemm_attrs},
            {"Identity", 13, 14, 1, 1, 1, 1, {}},
            {"If", 13, 14, 1, 1, 1, many, if_attrs},
            {"MatMul", 13, 14, 2, 2, 1, 1, {}},
            {"Mod", 13, 14, 2, 2, 1, 1, mod_attrs},
            {"Mul", 14, 14, 2, 2, 1, 1, {}},
            {"Neg", 13, 14, 1, 1, 1, 1, {}},
            {"Pad", 13, 14, 2, 3, 1, 1, pad_attrs},
            {"Pow", 12, 14, 2, 2, 1, 1, {}},
            {"Range", 11, 14, 3, 3, 1, 1, {}},
            {"Reciprocal", 13, 14, 1, 1, 1, 1, {}},
            {"ReduceL2", 13, 14, 1, 1, 1, 1, reduction_attrs},
            {"ReduceMean", 13, 14, 1, 1, 1, 1, reduction_attrs},
            {"ReduceSum", 13, 14, 1, 2, 1, 1, reduce_sum_attrs},
            {"Relu", 14, 14, 1, 1, 1, 1, {}},
            {"Reshape", 14, 14, 2, 2, 1, 1, reshape_attrs},
            {"Resize", 13, 14, 1, 4, 1, 1, resize_attrs},
            {"Round", 11, 14, 1, 1, 1, 1, {}},
            {"Shape", 13, 14, 1, 1, 1, 1, shape_attrs},
            {"Sigmoid", 13, 14, 1, 1, 1, 1, {}},
            {"Slice", 13, 14, 3, 5, 1, 1, {}},
            {"Softmax", 13, 14, 1, 1, 1, 1, softmax_attrs},
            {"Split", 13, 14, 1, 2, 1, many, split_attrs},
            {"Sqrt", 13, 14, 1, 1, 1, 1, {}},
            {"Squeeze", 13, 14, 1, 2, 1, 1, {}},
            {"Sub", 14, 14, 2, 2, 1, 1, {}},
            {"Transpose", 13, 14, 1, 1, 1, 1, transpose_attrs},
            {"Unsqueeze", 13, 14, 2, 2, 1, 1, {}},
            {"Where", 9, 14, 3, 3, 1, 1, {}},
            // These aliases keep the registry source-compatible with opset-14
            // models emitted by exporters that retain older schema versions.
            {"ReduceL2", 1, 12, 1, 1, 1, 1, reduction_attrs},
            {"Softmax", 11, 12, 1, 1, 1, 1, softmax_attrs},
        }};
    } // namespace

    const OperatorSchema* find_operator_schema(const std::string_view name,
                                                const std::int64_t opset) noexcept {
        const auto it = std::ranges::find_if(registry, [&](const OperatorSchema& schema) {
            return schema.name == name && opset >= schema.since_version && opset <= schema.through_version;
        });
        return it == registry.end() ? nullptr : &*it;
    }

    std::span<const OperatorSchema> operator_registry() noexcept {
        return registry;
    }

} // namespace lfs::onnx_vulkan::detail
