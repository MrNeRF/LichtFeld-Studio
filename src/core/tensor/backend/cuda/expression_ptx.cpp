/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../internal/expression_emitter.hpp"
#include <algorithm>
#include <bit>
#include <optional>
#include <sstream>

namespace lfs::core::internal {
    namespace {
        class PtxEmitter final : public ExpressionEmitter {
            uint32_t words_;
            std::vector<std::string> values_{"0"};
            std::ostringstream declarations_, code_;
            uint32_t label_ = 0;
            Value reg(const char* type = "b32") {
                auto id = uint32_t(values_.size());
                values_.push_back("%v" + std::to_string(id));
                declarations_ << ".reg ." << type << ' ' << v(id) << ";\n";
                return id;
            }
            const std::string& v(Value id) const { return values_.at(id); }
            Value unary(const char* instruction, std::string operand, const char* type = "b32") {
                const auto r = reg(type);
                code_ << instruction << ' ' << v(r) << ", " << operand << ";\n";
                return r;
            }
            Value binary(const char* instruction, Value a, Value b, const char* type = "b32") {
                return unary(instruction, v(a) + ", " + v(b), type);
            }
            Value sequence(std::string text, Value a, Value b, Value c) {
                auto result = reg();
                const auto prefix = "s" + std::to_string(label_++) + "_";
                for (auto [key, replacement] : {std::pair{std::string("%"), "%" + prefix}, {"$L__", prefix + "L"}, {"@A@", v(a)}, {"@B@", v(b)}, {"@C@", v(c)}, {"@OUT@", v(result)}, {"@DONE@", prefix + "done"}}) {
                    size_t position = 0;
                    while ((position = text.find(key, position)) != std::string::npos) {
                        text.replace(position, key.size(), replacement);
                        position += replacement.size();
                    }
                }
                code_ << "{\n"
                      << text << "}\n";
                return result;
            }
            std::optional<uint32_t> number(Value value) const {
                if (v(value).front() == '%')
                    return {};
                return uint32_t(std::stoul(v(value)));
            }
            Value address(uint32_t word, Value index, uint32_t width) {
                auto base = unary("ld.param.u64", "[bindings+" + std::to_string(word * 4) + "]", "b64");
                auto offset = unary("mul.wide.u32", v(index) + ", " + std::to_string(width), "b64");
                return binary("add.u64", base, offset, "b64");
            }

        public:
            explicit PtxEmitter(uint32_t words) : words_(words) {}
            Value literal(uint32_t value) override {
                values_.push_back(std::to_string(value));
                return uint32_t(values_.size() - 1);
            }
            Value math(ExprOp op, Value a, Value b = 0, Value c = 0) override {
                const auto x = number(a), y = number(b);
                if (x && op == ExprOp::Select)
                    return *x ? b : c;
                if (x && op == ExprOp::BitNot)
                    return literal(~*x);
                if (x && op == ExprOp::LogicalNot)
                    return literal(*x == 0);
                if (x && y) {
                    switch (op) {
                    case ExprOp::AddInt: return literal(*x + *y);
                    case ExprOp::SubInt: return literal(*x - *y);
                    case ExprOp::MulInt: return literal(*x * *y);
                    case ExprOp::BitAnd: return literal(*x & *y);
                    case ExprOp::BitOr: return literal(*x | *y);
                    case ExprOp::BitXor: return literal(*x ^ *y);
                    case ExprOp::ShiftLeft: return literal(*x << (*y & 31));
                    case ExprOp::ShiftRight: return literal(*x >> (*y & 31));
                    case ExprOp::UIntDiv: return literal(*y ? *x / *y : 0);
                    case ExprOp::UIntMod: return literal(*y ? *x % *y : 0);
                    case ExprOp::IntDiv: return literal(*y ? uint32_t(int64_t(int32_t(*x)) / int64_t(int32_t(*y))) : 0);
                    case ExprOp::IntMod: return literal(*y ? uint32_t(int64_t(int32_t(*x)) % int64_t(int32_t(*y))) : 0);
                    case ExprOp::EqualInt: return literal(*x == *y);
                    case ExprOp::NotEqualInt: return literal(*x != *y);
                    case ExprOp::LessInt: return literal(int32_t(*x) < int32_t(*y));
                    case ExprOp::LessEqualInt: return literal(int32_t(*x) <= int32_t(*y));
                    case ExprOp::GreaterInt: return literal(int32_t(*x) > int32_t(*y));
                    case ExprOp::GreaterEqualInt: return literal(int32_t(*x) >= int32_t(*y));
                    case ExprOp::LessUInt: return literal(*x < *y);
                    case ExprOp::LessEqualUInt: return literal(*x <= *y);
                    case ExprOp::GreaterUInt: return literal(*x > *y);
                    case ExprOp::GreaterEqualUInt: return literal(*x >= *y);
                    case ExprOp::LogicalAnd: return literal(*x && *y);
                    case ExprOp::LogicalOr: return literal(*x || *y);
                    case ExprOp::LogicalXor: return literal(bool(*x) != bool(*y));
                    default: break;
                    }
                }
                if (y && std::has_single_bit(*y)) {
                    if (op == ExprOp::UIntDiv)
                        return binary("shr.u32", a, literal(std::countr_zero(*y)));
                    if (op == ExprOp::UIntMod)
                        return binary("and.b32", a, literal(*y - 1));
                }
                if (y && *y == 1 && op == ExprOp::MulInt)
                    return a;
                if (x && *x == 0 && op == ExprOp::AddInt)
                    return b;
                if (y && *y == 0 && op == ExprOp::AddInt)
                    return a;
                switch (op) {
                case ExprOp::AddInt: return binary("add.u32", a, b);
                case ExprOp::SubInt: return binary("sub.u32", a, b);
                case ExprOp::MulInt: return binary("mul.lo.u32", a, b);
                case ExprOp::BitAnd: return binary("and.b32", a, b);
                case ExprOp::BitOr: return binary("or.b32", a, b);
                case ExprOp::BitXor: return binary("xor.b32", a, b);
                case ExprOp::BitNot: return unary("not.b32", v(a));
                default: return sequence(std::string(expression_ptx_opcode(op)), a, b, c);
                }
            }
            Value mul_hi(Value a, Value b) override {
                const auto x = number(a), y = number(b);
                if (x && y)
                    return literal(uint32_t((uint64_t(*x) * *y) >> 32));
                return binary("mul.hi.u32", a, b);
            }
            Value variable(Value initial) override { return unary("mov.b32", v(initial)); }
            Value read(Value slot) override { return unary("mov.b32", v(slot)); }
            void assign(Value slot, Value value) override { code_ << "mov.b32 " << v(slot) << ", " << v(value) << ";\n"; }
            void condition(Value predicate, const std::function<void()>& yes, const std::function<void()>& no) override {
                if (v(predicate).front() != '%') {
                    if (std::stoul(v(predicate)))
                        yes();
                    else if (no)
                        no();
                    return;
                }
                auto end = "if_end_" + std::to_string(label_++), otherwise = "if_else_" + std::to_string(label_++);
                const auto p = binary("setp.ne.u32", predicate, literal(0), "pred");
                code_ << "@!" << v(p) << " bra " << otherwise << ";\n";
                yes();
                code_ << "bra " << end << ";\n"
                      << otherwise << ":\n";
                if (no)
                    no();
                code_ << end << ":\n";
            }
            void loop(Value first, Value end, Value step, const std::function<void(Value)>& body) override {
                auto head = "loop_" + std::to_string(label_++), done = "loop_end_" + std::to_string(label_++);
                const auto index = variable(first);
                code_ << head << ":\n";
                const auto stop = binary("setp.ge.u32", index, end, "pred");
                code_ << '@' << v(stop) << " bra " << done << ";\n";
                body(index);
                assign(index, binary("add.u32", index, step));
                code_ << "bra " << head << ";\n"
                      << done << ":\n";
            }
            Value argument(Value word) override {
                if (const auto offset = number(word))
                    return unary("ld.param.u32", "[bindings+" + std::to_string(*offset * 4) + "]");
                const auto base = unary("mov.u64", "bindings", "b64");
                const auto offset = unary("mul.wide.u32", v(word) + ", 4", "b64");
                return unary("ld.param.u32", "[" + v(binary("add.u64", base, offset, "b64")) + "]");
            }
            Value load(uint32_t word, Value index, DataType dtype) override {
                const auto p = address(word, index, dtype_size(dtype));
                if (dtype == DataType::Float16)
                    return half(unary("ld.global.u16", "[" + v(p) + "]"), false);
                return unary(dtype_size(dtype) == 1 ? "ld.global.u8" : "ld.global.nc.u32", "[" + v(p) + "]");
            }
            std::array<Value, 2> load_pair(uint32_t word, Value index) override {
                const auto p = address(word, index, 4), x = reg(), y = reg();
                code_ << "ld.global.nc.v2.u32 {" << v(x) << ", " << v(y) << "}, [" << v(p) << "];\n";
                return {x, y};
            }
            Value output(uint32_t word, Value index) override {
                const auto p = address(word, index, 4);
                return unary("ld.global.u32", "[" + v(p) + "]");
            }
            void store(uint32_t word, Value index, Value value, DataType dtype) override {
                const auto p = address(word, index, dtype_size(dtype));
                if (dtype == DataType::Float16)
                    value = half(value, true);
                code_ << "st.global.u" << dtype_size(dtype) * 8 << " [" << v(p) << "], " << v(value) << ";\n";
            }
            Value half(Value value, bool pack) override {
                const auto converted = sequence(std::string(expression_ptx_half(pack)), value, 0, 0);
                const auto magnitude = math(ExprOp::BitAnd, value, literal(pack ? 0x7fffffffu : 0x7fffu));
                const auto nan = math(ExprOp::GreaterUInt, magnitude, literal(pack ? 0x7f800000u : 0x7c00u));
                Value preserved;
                if (pack) {
                    const auto sign = math(ExprOp::BitAnd, math(ExprOp::ShiftRight, value, literal(16)), literal(0x8000));
                    auto payload = math(ExprOp::BitAnd, math(ExprOp::ShiftRight, value, literal(13)), literal(1023));
                    payload = math(ExprOp::Select, payload, payload, literal(1));
                    preserved = math(ExprOp::BitOr, sign, math(ExprOp::BitOr, literal(0x7c00), payload));
                } else {
                    const auto sign = math(ExprOp::ShiftLeft, math(ExprOp::BitAnd, value, literal(0x8000)), literal(16));
                    const auto payload = math(ExprOp::ShiftLeft, math(ExprOp::BitAnd, value, literal(1023)), literal(13));
                    preserved = math(ExprOp::BitOr, sign, math(ExprOp::BitOr, literal(0x7f800000), payload));
                }
                return math(ExprOp::Select, nan, preserved, converted);
            }
            Value thread() override {
                const auto block = unary("mov.u32", "%ctaid.x"), lane = unary("mov.u32", "%tid.x");
                return binary("add.u32", binary("mul.lo.u32", block, literal(256)), lane);
            }
            Value grid_stride() override {
                return binary("mul.lo.u32", unary("mov.u32", "%nctaid.x"), literal(256));
            }
            bool cuda() const override { return true; }
            std::string finish() const {
                return std::string(expression_ptx_header()) +
                       ".visible .entry expression_main(.param .align 8 .b8 bindings[" + std::to_string(words_ * 4) + "])\n.maxntid 256,1,1\n{\n" +
                       declarations_.str() + code_.str() + "ret;\n}\n";
            }
        };
    } // namespace
    std::string expression_ptx(const ExpressionProgram& program, const ExpressionSignature& signature) {
        PtxEmitter emitter(expression_layout(program, signature).words);
        emit_expression(emitter, program, signature);
        return emitter.finish();
    }
} // namespace lfs::core::internal
