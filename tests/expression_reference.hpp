/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/expression_program.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <functional>
#include <limits>
#include <vector>

namespace expression_test {
    using namespace lfs::core;
    using namespace lfs::core::internal;
    inline uint32_t bits(float x) { return std::bit_cast<uint32_t>(x); }
    inline float value(uint32_t x) { return std::bit_cast<float>(x); }
    inline uint32_t reference_op(ExprOp op, uint32_t a, uint32_t b, uint32_t c) {
        float x = value(a), y = value(b), z = value(c);
        switch (op) {
        case ExprOp::Add: return bits(x + y);
        case ExprOp::Sub: return bits(x - y);
        case ExprOp::Mul: return bits(x * y);
        case ExprOp::Div: return bits(x / y);
        case ExprOp::Mod: return bits(std::fmod(x, y));
        case ExprOp::Pow: return bits(std::pow(x, y));
        case ExprOp::Min:
            if (std::isnan(x))
                return a;
            if (std::isnan(y))
                return b;
            if (x == 0 && y == 0)
                return a | b;
            return bits(std::fmin(x, y));
        case ExprOp::Max:
            if (std::isnan(x))
                return a;
            if (std::isnan(y))
                return b;
            if (x == 0 && y == 0)
                return a & b;
            return bits(std::fmax(x, y));
        case ExprOp::Atan2: return bits(std::atan2(x, y));
        case ExprOp::Fma: return bits(std::fma(x, y, z));
        case ExprOp::Clamp: return std::isnan(x) ? a : bits(std::fmin(std::fmax(x, y), z));
        case ExprOp::Select: return a ? b : c;
        case ExprOp::Abs: return bits(std::fabs(x));
        case ExprOp::Neg: return bits(-x);
        case ExprOp::Exp: return bits(std::exp(x));
        case ExprOp::Log: return bits(std::log(x));
        case ExprOp::Sqrt: return bits(std::sqrt(x));
        case ExprOp::Sigmoid: return bits(1.0f / (1.0f + std::exp(-x)));
        case ExprOp::Relu: return bits(std::isnan(x) ? x : std::fmax(x, 0.0f));
        case ExprOp::Square: return bits(x * x);
        case ExprOp::Tanh: return bits(std::tanh(x));
        case ExprOp::Rsqrt: return bits((1.0f / std::sqrt(x)));
        case ExprOp::Sign: return bits(float((x > 0) - (x < 0)));
        case ExprOp::Reciprocal: return bits(1.0f / x);
        case ExprOp::Floor: return bits(std::floor(x));
        case ExprOp::Ceil: return bits(std::ceil(x));
        case ExprOp::Round: return bits(std::nearbyint(x));
        case ExprOp::Exp2: return bits(std::exp2(x));
        case ExprOp::Log2: return bits(std::log2(x));
        case ExprOp::Log10: return bits(std::log10(x));
        case ExprOp::Log1p: return bits(std::log1p(x));
        case ExprOp::Sin: return bits(std::sin(x));
        case ExprOp::Cos: return bits(std::cos(x));
        case ExprOp::Tan: return bits(std::tan(x));
        case ExprOp::Asin: return bits(std::asin(std::clamp(x, -1.0f, 1.0f)));
        case ExprOp::Acos: return bits(std::acos(std::clamp(x, -1.0f, 1.0f)));
        case ExprOp::Atan: return bits(std::atan(x));
        case ExprOp::Sinh: return bits(std::sinh(x));
        case ExprOp::Cosh: return bits(std::cosh(x));
        case ExprOp::Gelu: return bits(0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))));
        case ExprOp::Swish: return bits(x / (1.0f + std::exp(-x)));
        case ExprOp::Trunc: return bits(std::trunc(x));
        case ExprOp::IsNan: return std::isnan(x);
        case ExprOp::IsInf: return std::isinf(x);
        case ExprOp::IsFinite: return std::isfinite(x);
        case ExprOp::Equal: return x == y;
        case ExprOp::NotEqual: return x != y;
        case ExprOp::Less: return x < y;
        case ExprOp::LessEqual: return x <= y;
        case ExprOp::Greater: return x > y;
        case ExprOp::GreaterEqual: return x >= y;
        case ExprOp::LogicalAnd: return a != 0 && b != 0;
        case ExprOp::LogicalOr: return a != 0 || b != 0;
        case ExprOp::LogicalXor: return (a != 0) != (b != 0);
        case ExprOp::LogicalNot: return a == 0;
        case ExprOp::BitAnd: return a & b;
        case ExprOp::BitOr: return a | b;
        case ExprOp::BitXor: return a ^ b;
        case ExprOp::BitNot: return ~a;
        case ExprOp::ShiftLeft: return a << (b & 31u);
        case ExprOp::ShiftRight: return a >> (b & 31u);
        case ExprOp::AddInt: return a + b;
        case ExprOp::SubInt: return a - b;
        case ExprOp::MulInt: return a * b;
        case ExprOp::IntDiv: return b ? uint32_t(int64_t(int32_t(a)) / int64_t(int32_t(b))) : 0;
        case ExprOp::IntMod: return b ? uint32_t(int64_t(int32_t(a)) % int64_t(int32_t(b))) : 0;
        case ExprOp::EqualInt: return a == b;
        case ExprOp::NotEqualInt: return a != b;
        case ExprOp::LessInt: return int32_t(a) < int32_t(b);
        case ExprOp::LessEqualInt: return int32_t(a) <= int32_t(b);
        case ExprOp::GreaterInt: return int32_t(a) > int32_t(b);
        case ExprOp::GreaterEqualInt: return int32_t(a) >= int32_t(b);
        case ExprOp::CastFloat: return bits(float(int32_t(a)));
        case ExprOp::CastInt: return uint32_t(int32_t(x));
        case ExprOp::CastUInt: return uint32_t(x);
        case ExprOp::UIntDiv: return b ? a / b : 0;
        case ExprOp::UIntMod: return b ? a % b : 0;
        case ExprOp::LessUInt: return a < b;
        case ExprOp::LessEqualUInt: return a <= b;
        case ExprOp::GreaterUInt: return a > b;
        case ExprOp::GreaterEqualUInt: return a >= b;
        case ExprOp::CastUnsignedFloat: return bits(float(a));
        default: return 0;
        }
    }
    inline uint32_t load(const ExprInput& input, size_t i) {
        const auto* p = static_cast<const std::byte*>(input.storage.data) + input.storage.byte_offset;
        switch (input.storage.dtype) {
        case DataType::Float32: return bits(reinterpret_cast<const float*>(p)[i]);
        case DataType::Float16: return bits(__half2float(reinterpret_cast<const __half*>(p)[i]));
        case DataType::UInt8:
        case DataType::Bool: return reinterpret_cast<const uint8_t*>(p)[i];
        default: return reinterpret_cast<const uint32_t*>(p)[i];
        }
    }
    // Evaluates the program as a DAG at every output element; a fold evaluates its operand
    // at every coordinate of the fold dimension.
    inline std::vector<std::vector<uint32_t>> reference(
        const ExpressionProgram& program, std::span<const ExprInput> inputs,
        const ExprShape& shape, std::span<const DataType> types) {
        const auto instructions = program.instructions();
        std::vector<std::array<uint32_t, 3>> sources(instructions.size());
        uint32_t definitions[32]{};
        int fold = -1;
        for (uint32_t pc = 0; pc < instructions.size(); ++pc) {
            const auto ins = instructions[pc];
            const auto op = ExprOp(ins.op_dst_a_b & 255u);
            const uint32_t args[]{(ins.op_dst_a_b >> 16) & 255u, ins.op_dst_a_b >> 24, ins.c_aux_policy & 255u};
            for (uint32_t a = 0; a < expr_arity(op); ++a)
                sources[pc][a] = definitions[args[a]];
            if (op != ExprOp::Store)
                definitions[(ins.op_dst_a_b >> 8) & 255u] = pc;
            if (op == ExprOp::Fold)
                fold = int((ins.c_aux_policy >> 16) & 255u);
        }
        size_t count = 1;
        for (uint32_t d = 0; d < shape.rank; ++d)
            if (int(d) != fold)
                count *= shape.dims[d];
        std::vector<std::vector<uint32_t>> result(types.size(), std::vector<uint32_t>(count));
        for (size_t element = 0; element < count; ++element) {
            uint32_t coords[4]{};
            size_t rest = element;
            for (int d = int(shape.rank) - 1; d >= 0; --d)
                if (d != fold) {
                    coords[d] = uint32_t(rest % shape.dims[d]);
                    rest /= shape.dims[d];
                }
            std::function<uint32_t(uint32_t)> eval = [&](uint32_t pc) -> uint32_t {
                const auto ins = instructions[pc];
                const auto op = ExprOp(ins.op_dst_a_b & 255u);
                const uint32_t aux = (ins.c_aux_policy >> 8) & 255u, policy = (ins.c_aux_policy >> 16) & 255u;
                const auto arg = [&](uint32_t n) { return eval(sources[pc][n]); };
                switch (op) {
                case ExprOp::Immediate: return ins.immediate;
                case ExprOp::Iota: return coords[aux];
                case ExprOp::Extent: return shape.dims[aux];
                case ExprOp::Load: {
                    size_t index = 0;
                    for (uint32_t d = 0; d < shape.rank; ++d)
                        index += size_t(coords[d]) * inputs[aux].strides[d];
                    return load(inputs[aux], index);
                }
                case ExprOp::Gather:
                case ExprOp::Gather2:
                case ExprOp::Gather3: {
                    size_t index = 0;
                    for (uint32_t d = 0; d < expr_arity(op); ++d) {
                        int64_t position = int32_t(arg(d)), n = inputs[aux].dims[d];
                        if (ExprOob(policy) == ExprOob::Zero && (position < 0 || position >= n))
                            return 0;
                        if (ExprOob(policy) == ExprOob::Clamp)
                            position = std::clamp(position, int64_t(0), n - 1);
                        if (ExprOob(policy) == ExprOob::Wrap)
                            position = (position % n + n) % n;
                        index += size_t(position) * inputs[aux].strides[d];
                    }
                    return load(inputs[aux], index);
                }
                case ExprOp::Fold: {
                    const auto kind = ExprReduce(aux);
                    const auto type = DataType(ins.immediate);
                    const bool floating = type == DataType::Float32;
                    uint32_t acc = 0, compensation = 0;
                    if (kind == ExprReduce::Min)
                        acc = floating ? bits(INFINITY) : type == DataType::Int32 ? uint32_t(INT32_MAX)
                                                                                  : ~0u;
                    if (kind == ExprReduce::Max)
                        acc = floating ? bits(-INFINITY) : type == DataType::Int32 ? uint32_t(INT32_MIN)
                                                                                   : 0u;
                    if (kind == ExprReduce::And)
                        acc = type == DataType::Bool ? 1u : ~0u;
                    const uint32_t saved = coords[policy];
                    for (uint32_t k = 0; k < shape.dims[policy]; ++k) {
                        coords[policy] = k;
                        const uint32_t v = arg(0);
                        switch (kind) {
                        case ExprReduce::Sum:
                            if (floating) {
                                const float y = value(v) - value(compensation);
                                const float t = value(acc) + y;
                                compensation = std::isfinite(t) ? bits((t - value(acc)) - y) : 0;
                                acc = bits(t);
                            } else
                                acc += v;
                            break;
                        case ExprReduce::Min:
                        case ExprReduce::Max:
                            if (floating)
                                acc = reference_op(kind == ExprReduce::Min ? ExprOp::Min : ExprOp::Max, acc, v, 0);
                            else if (type == DataType::Int32)
                                acc = uint32_t(kind == ExprReduce::Min ? std::min(int32_t(acc), int32_t(v)) : std::max(int32_t(acc), int32_t(v)));
                            else
                                acc = kind == ExprReduce::Min ? std::min(acc, v) : std::max(acc, v);
                            break;
                        case ExprReduce::And: acc &= v; break;
                        case ExprReduce::Or: acc |= v; break;
                        case ExprReduce::Xor: acc ^= v; break;
                        case ExprReduce::Count: acc += v != 0; break;
                        }
                    }
                    coords[policy] = saved;
                    return acc;
                }
                default:
                    return reference_op(op, arg(0), expr_arity(op) > 1 ? arg(1) : 0, expr_arity(op) > 2 ? arg(2) : 0);
                }
            };
            for (uint32_t pc = 0; pc < instructions.size(); ++pc)
                if (ExprOp(instructions[pc].op_dst_a_b & 255u) == ExprOp::Store)
                    result[(instructions[pc].c_aux_policy >> 8) & 255u][element] = eval(sources[pc][0]);
        }
        return result;
    }
} // namespace expression_test
