/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Fused expressions as Metal Shading Language source, compiled by the Metal
// backend at run time. Lowerings follow expression_spirv.cpp, so both GPU
// backends of the Mac compute the same values.

#include "../../internal/expression_emitter.hpp"

#include <cmath>
#include <format>
#include <sstream>
#include <stdexcept>

namespace lfs::core::internal {
    namespace {
        // The helpers reproduce the SPIR-V lowerings bit for bit.
        constexpr std::string_view kPrelude = R"(#include <metal_stdlib>
#pragma METAL fp contract(off)
using namespace metal;

static float lfs_minmax(float x, float y, bool maximum) {
    const uint xb = as_type<uint>(x), yb = as_type<uint>(y);
    const uint xm = xb & 0x7fffffffu, ym = yb & 0x7fffffffu;
    if (xm > 0x7f800000u)
        return x;
    if (ym > 0x7f800000u)
        return y;
    if (xm == 0u && ym == 0u)
        return as_type<float>(maximum ? xb & yb : xb | yb);
    return (maximum ? x < y : y < x) ? y : x;
}

static float lfs_round(float x) {
    if (!(as_type<uint>(x) << 1 < 0xff000000u))
        return x;
    const float lower = floor(x), fraction = x - lower, next = lower + 1.0f;
    const float result = fraction < 0.5f ? lower : fraction > 0.5f ? next : fmod(lower, 2.0f) == 0.0f ? lower : next;
    return result == 0.0f && as_type<uint>(x) >= 0x80000000u ? -0.0f : result;
}

static float lfs_log1p(float x) {
    if (x == 0.0f)
        return x;
    if (x <= -1.0f || !(as_type<uint>(x) << 1 < 0xff000000u))
        return log(1.0f + x);
    const uint exponent_bits = (as_type<uint>(1.0f + x) - 1061158912u) & 0xff800000u;
    const float reduced = as_type<float>(as_type<uint>(x) - exponent_bits);
    const float scale = as_type<float>(1082130432u - exponent_bits);
    const float v = reduced + fma(0.25f, scale, -1.0f);
    const float exponent = float(as_type<int>(exponent_bits)) * 1.1920928955078125e-7f;
    float polynomial = -0.04534861445426941f;
    polynomial = fma(polynomial, v, 0.10546888411045074f);
    polynomial = fma(polynomial, v, -0.13229703903198242f);
    polynomial = fma(polynomial, v, 0.14491446316242218f);
    polynomial = fma(polynomial, v, -0.16641564667224884f);
    polynomial = fma(polynomial, v, 0.199888676404953f);
    polynomial = fma(polynomial, v, -0.2500019669532776f);
    polynomial = fma(polynomial, v, 0.33333510160446167f);
    polynomial = fma(polynomial, v, -0.5f);
    return fma(exponent, 0.6931471824645996f, fma(polynomial * v, v, v));
}

static uint lfs_unpack_half(uint value) {
    const uint sign = (value & 0x8000u) << 16, exponent = (value >> 10) & 31u, mantissa = value & 1023u;
    if (exponent == 0u)
        return sign | as_type<uint>(float(mantissa) * 5.9604644775390625e-8f);
    if (exponent == 31u)
        return sign | 0x7f800000u | (mantissa << 13);
    return sign | ((exponent + 112u) << 23) | (mantissa << 13);
}

static uint lfs_pack_half(uint value) {
    const uint sign = (value >> 16) & 0x8000u, magnitude = value & 0x7fffffffu;
    if (magnitude > 0x7f800000u)
        return sign | 0x7c00u | max((magnitude & 0x7fffffu) >> 13, 1u);
    if (magnitude >= 0x477ff000u)
        return sign | 0x7c00u;
    if (magnitude < 0x33000000u)
        return sign;
    if (magnitude < 0x38800000u) {
        const uint shift = 126u - (magnitude >> 23);
        const uint significand = (magnitude & 0x7fffffu) | 0x800000u;
        const uint truncated = significand >> shift;
        const uint remainder = significand & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
        const bool up = remainder > halfway || (remainder == halfway && (truncated & 1u) != 0u);
        return sign | (truncated + (up ? 1u : 0u));
    }
    return sign | ((magnitude + 0xfffu + ((magnitude >> 13) & 1u) - 0x38000000u) >> 13);
}

static uint lfs_int_divide(uint a, uint b, bool modulo) {
    const bool a_negative = as_type<int>(a) < 0, b_negative = as_type<int>(b) < 0;
    const uint magnitude_a = a_negative ? 0u - a : a, magnitude_b = b_negative ? 0u - b : b;
    const uint result = magnitude_b == 0u ? 0u : modulo ? magnitude_a % magnitude_b : magnitude_a / magnitude_b;
    return (modulo ? a_negative : a_negative != b_negative) ? 0u - result : result;
}

)";

        // Every value is a uint of raw bits declared at function scope, so
        // structured control flow never hides a definition from later uses.
        class MslEmitter final : public ExpressionEmitter {
        public:
            Value literal(uint32_t value) override {
                values_.push_back(std::format("{}u", value));
                return static_cast<Value>(values_.size() - 1);
            }

            Value math(const ExprOp op, const Value a, const Value b, const Value c) override {
                switch (op) {
                case ExprOp::AddInt: return define(v(a) + " + " + v(b));
                case ExprOp::SubInt: return define(v(a) + " - " + v(b));
                case ExprOp::MulInt: return define(v(a) + " * " + v(b));
                case ExprOp::BitAnd: return define(v(a) + " & " + v(b));
                case ExprOp::BitOr: return define(v(a) + " | " + v(b));
                case ExprOp::BitXor: return define(v(a) + " ^ " + v(b));
                case ExprOp::BitNot: return define("~" + v(a));
                case ExprOp::ShiftLeft: return define(v(a) + " << (" + v(b) + " & 31u)");
                case ExprOp::ShiftRight: return define(v(a) + " >> (" + v(b) + " & 31u)");
                case ExprOp::EqualInt: return flag(v(a) + " == " + v(b));
                case ExprOp::NotEqualInt: return flag(v(a) + " != " + v(b));
                case ExprOp::LessInt: return flag(i(a) + " < " + i(b));
                case ExprOp::LessEqualInt: return flag(i(a) + " <= " + i(b));
                case ExprOp::GreaterInt: return flag(i(a) + " > " + i(b));
                case ExprOp::GreaterEqualInt: return flag(i(a) + " >= " + i(b));
                case ExprOp::LessUInt: return flag(v(a) + " < " + v(b));
                case ExprOp::LessEqualUInt: return flag(v(a) + " <= " + v(b));
                case ExprOp::GreaterUInt: return flag(v(a) + " > " + v(b));
                case ExprOp::GreaterEqualUInt: return flag(v(a) + " >= " + v(b));
                case ExprOp::LogicalAnd: return flag(v(a) + " != 0u && " + v(b) + " != 0u");
                case ExprOp::LogicalOr: return flag(v(a) + " != 0u || " + v(b) + " != 0u");
                case ExprOp::LogicalXor: return flag("(" + v(a) + " != 0u) != (" + v(b) + " != 0u)");
                case ExprOp::LogicalNot: return flag(v(a) + " == 0u");
                case ExprOp::Select: return define(v(a) + " != 0u ? " + v(b) + " : " + v(c));
                case ExprOp::UIntDiv: return define(v(b) + " != 0u ? " + v(a) + " / " + v(b) + " : 0u");
                case ExprOp::UIntMod: return define(v(b) + " != 0u ? " + v(a) + " % " + v(b) + " : 0u");
                case ExprOp::IntDiv: return define("lfs_int_divide(" + v(a) + ", " + v(b) + ", false)");
                case ExprOp::IntMod: return define("lfs_int_divide(" + v(a) + ", " + v(b) + ", true)");
                case ExprOp::CastFloat: return bits("float(" + i(a) + ")");
                case ExprOp::CastUnsignedFloat: return bits("float(" + v(a) + ")");
                case ExprOp::CastInt: return define("as_type<uint>(int(" + f(a) + "))");
                case ExprOp::CastUInt: return define("uint(" + f(a) + ")");
                default: break;
                }
                const std::string x = f(a), y = expr_arity(op) > 1 ? f(b) : "", z = expr_arity(op) > 2 ? f(c) : "";
                switch (op) {
                case ExprOp::Add: return bits(x + " + " + y);
                case ExprOp::Sub: return bits(x + " - " + y);
                case ExprOp::Mul: return bits(x + " * " + y);
                case ExprOp::Div: return bits(x + " / " + y);
                case ExprOp::Mod: return bits("fmod(" + x + ", " + y + ")");
                case ExprOp::Pow: return bits(y + " == 2.0f ? " + x + " * " + x + " : pow(" + x + ", " + y + ")");
                case ExprOp::Min: return bits("lfs_minmax(" + x + ", " + y + ", false)");
                case ExprOp::Max: return bits("lfs_minmax(" + x + ", " + y + ", true)");
                case ExprOp::Atan2: return bits("atan2(" + x + ", " + y + ")");
                case ExprOp::Fma: return bits("fma(" + x + ", " + y + ", " + z + ")");
                case ExprOp::Clamp: return bits("isnan(" + x + ") ? " + x + " : fmin(fmax(" + x + ", " + y + "), " + z + ")");
                case ExprOp::Equal: return flag(x + " == " + y);
                case ExprOp::NotEqual: return flag(x + " != " + y);
                case ExprOp::Less: return flag(x + " < " + y);
                case ExprOp::LessEqual: return flag(x + " <= " + y);
                case ExprOp::Greater: return flag(x + " > " + y);
                case ExprOp::GreaterEqual: return flag(x + " >= " + y);
                case ExprOp::IsNan: return flag("isnan(" + x + ")");
                case ExprOp::IsInf: return flag("isinf(" + x + ")");
                case ExprOp::IsFinite: return flag("(" + v(a) + " & 0x7fffffffu) < 0x7f800000u");
                case ExprOp::Abs: return bits("abs(" + x + ")");
                case ExprOp::Neg: return bits("-" + x);
                case ExprOp::Exp: return bits("exp(" + x + ")");
                case ExprOp::Log: return bits("log(" + x + ")");
                case ExprOp::Sqrt: return bits("sqrt(" + x + ")");
                case ExprOp::Sigmoid: return bits("1.0f / (1.0f + exp(-" + x + "))");
                case ExprOp::Relu: return bits("lfs_minmax(" + x + ", 0.0f, true)");
                case ExprOp::Square: return bits(x + " * " + x);
                case ExprOp::Tanh: return bits("tanh(" + x + ")");
                case ExprOp::Rsqrt: return bits("rsqrt(" + x + ")");
                case ExprOp::Sign: return bits("float(int(" + x + " > 0.0f) - int(" + x + " < 0.0f))");
                case ExprOp::Reciprocal: return bits("1.0f / " + x);
                case ExprOp::Floor: return bits("floor(" + x + ")");
                case ExprOp::Ceil: return bits("ceil(" + x + ")");
                case ExprOp::Round: return bits("lfs_round(" + x + ")");
                case ExprOp::Exp2: return bits("exp2(" + x + ")");
                case ExprOp::Log2: return bits("log2(" + x + ")");
                case ExprOp::Log10: return bits("log(" + x + ") * 0.4342944819032518f");
                case ExprOp::Log1p: return bits("lfs_log1p(" + x + ")");
                case ExprOp::Sin: return bits("sin(" + x + ")");
                case ExprOp::Cos: return bits("cos(" + x + ")");
                case ExprOp::Tan: return bits("tan(" + x + ")");
                case ExprOp::Asin:
                case ExprOp::Acos: {
                    const auto bounded = bits("clamp(" + x + ", -1.0f, 1.0f)");
                    const auto root = bits("sqrt(fmax(0.0f, 1.0f - " + f(bounded) + " * " + f(bounded) + "))");
                    return op == ExprOp::Asin ? bits("atan2(" + f(bounded) + ", " + f(root) + ")")
                                              : bits("atan2(" + f(root) + ", " + f(bounded) + ")");
                }
                case ExprOp::Atan: return bits("atan(" + x + ")");
                case ExprOp::Sinh: return bits("sinh(" + x + ")");
                case ExprOp::Cosh: return bits("cosh(" + x + ")");
                case ExprOp::Gelu:
                    return bits(std::format("0.5f * {0} * (1.0f + tanh({1}f * ({0} + 0.044715f * {0} * {0} * {0})))", x,
                                            std::sqrt(2.0f / 3.14159265358979323846f)));
                case ExprOp::Swish: return bits(x + " / (1.0f + exp(-" + x + "))");
                case ExprOp::Trunc: return bits("trunc(" + x + ")");
                default: throw std::invalid_argument("No MSL lowering for expression opcode");
                }
            }

            Value mul_hi(const Value a, const Value b) override { return define("mulhi(" + v(a) + ", " + v(b) + ")"); }
            Value variable(const Value initial) override { return define(v(initial)); }
            Value read(const Value slot) override { return define(v(slot)); }
            void assign(const Value slot, const Value value) override { code_ << v(slot) << " = " << v(value) << ";\n"; }

            void condition(const Value predicate, const std::function<void()>& yes,
                           const std::function<void()>& no) override {
                code_ << "if (" << v(predicate) << " != 0u) {\n";
                yes();
                if (no) {
                    code_ << "} else {\n";
                    no();
                }
                code_ << "}\n";
            }

            void loop(const Value first, const Value end, const Value step,
                      const std::function<void(Value)>& body) override {
                const auto slot = variable(first);
                code_ << "for (;;) {\n";
                const auto index = read(slot);
                code_ << "if (" << v(index) << " >= " << v(end) << ") break;\n";
                body(index);
                code_ << v(slot) << " = " << v(index) << " + " << v(step) << ";\n}\n";
            }

            Value argument(const Value word) override { return define("lfs_args[" + v(word) + "]"); }

            Value load(const uint32_t word, const Value index, const DataType dtype) override {
                if (dtype_size(dtype) == 4)
                    return define(pointer(word, "device const uint") + "[" + v(index) + "]");
                if (dtype_size(dtype) == 1)
                    return define("uint(" + pointer(word, "device const uchar") + "[" + v(index) + "])");
                return half(define("uint(" + pointer(word, "device const ushort") + "[" + v(index) + "])"), false);
            }

            std::array<Value, 2> load_pair(uint32_t, Value) override {
                throw std::logic_error("MSL expressions load elements singly");
            }

            Value output(const uint32_t word, const Value index) override {
                return define(pointer(word, "device const uint") + "[" + v(index) + "]");
            }

            void store(const uint32_t word, const Value index, const Value value, const DataType dtype) override {
                if (dtype_size(dtype) != 4)
                    throw std::logic_error("MSL expression outputs own complete words");
                code_ << pointer(word, "device uint") << "[" << v(index) << "] = " << v(value) << ";\n";
            }

            Value half(const Value value, const bool pack) override {
                return define(std::string(pack ? "lfs_pack_half(" : "lfs_unpack_half(") + v(value) + ")");
            }

            Value thread() override { return define("lfs_thread"); }
            Value grid_stride() override { return define("lfs_threads"); }
            bool cuda() const override { return false; }

            std::string finish() const {
                std::string source(kPrelude);
                source += "kernel void lfs_expression(constant uint* lfs_args [[buffer(0)]],\n"
                          "                           uint lfs_thread [[thread_position_in_grid]],\n"
                          "                           uint lfs_threads [[threads_per_grid]]) {\n";
                std::string declarations;
                for (const std::string& value : values_) {
                    if (value.front() == 'v')
                        declarations += (declarations.empty() ? "uint " : ", ") + value;
                }
                if (!declarations.empty())
                    source += declarations + ";\n";
                return source + code_.str() + "}\n";
            }

        private:
            const std::string& v(const Value id) const { return values_.at(id); }
            std::string f(const Value id) const { return "as_type<float>(" + v(id) + ")"; }
            std::string i(const Value id) const { return "as_type<int>(" + v(id) + ")"; }

            // By value: expressions often name other values, and the push may move them.
            Value define(std::string expression) {
                values_.push_back(std::format("v{}", values_.size()));
                code_ << values_.back() << " = " << expression << ";\n";
                return static_cast<Value>(values_.size() - 1);
            }

            Value bits(const std::string& float_expression) { return define("as_type<uint>(" + float_expression + ")"); }
            Value flag(const std::string& condition) { return define("(" + condition + ") ? 1u : 0u"); }

            // A device pointer from the 64-bit address at argument words [word, word + 1].
            static std::string pointer(const uint32_t word, const std::string_view type) {
                return std::format("reinterpret_cast<{}*>(ulong(lfs_args[{}]) | (ulong(lfs_args[{}]) << 32))", type, word,
                                   word + 1);
            }

            std::vector<std::string> values_{"0u"};
            std::ostringstream code_;
        };
    } // namespace

    std::string expression_msl(const ExpressionProgram& program, const ExpressionSignature& signature) {
        MslEmitter emitter;
        emit_expression(emitter, program, signature);
        return emitter.finish();
    }

} // namespace lfs::core::internal
