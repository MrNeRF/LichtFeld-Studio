/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../internal/expression_emitter.hpp"
#include "spirv_module.hpp"
#include <bit>
#include <cmath>
#include <cstring>
#include <map>
#include <spirv/unified1/GLSL.std.450.h>
#include <stdexcept>

namespace lfs::core::internal {
    namespace {
        class SpirvEmitter final : public ExpressionEmitter {
            using Words = std::vector<uint32_t>;
            Words header_, annotations_, types_, globals_, variables_, body_;
            uint32_t next_ = 1;
            bool rounded_ = true;
            uint32_t void_, bool_, uint_, int_, float_, ulong_, uint3_, wide_, function_type_;
            uint32_t local_uint_, physical_uint_, push_uint_, push_ulong_, glsl_, entry_, push_, thread_, groups_;
            std::map<std::pair<uint32_t, uint64_t>, uint32_t> constants_;
            std::map<uint32_t, uint32_t> words_;
            // Words up to 128 bytes are push constants; larger blocks are read through the
            // device address pushed in their place.
            bool direct_;
            std::vector<Value> arguments_;

            uint32_t id() { return next_++; }
            static void emit(Words& words, spv::Op code, std::initializer_list<uint32_t> args) {
                words.push_back((uint32_t(args.size() + 1) << 16) | uint32_t(code));
                words.insert(words.end(), args);
            }
            static void string(Words& words, spv::Op code, std::initializer_list<uint32_t> prefix,
                               std::string_view text, std::initializer_list<uint32_t> suffix = {}) {
                Words args(prefix);
                const auto offset = args.size();
                args.insert(args.end(), text.size() / 4 + 1, 0);
                std::memcpy(args.data() + offset, text.data(), text.size());
                args.insert(args.end(), suffix);
                words.push_back((uint32_t(args.size() + 1) << 16) | uint32_t(code));
                words.insert(words.end(), args.begin(), args.end());
            }
            uint32_t type(spv::Op code, std::initializer_list<uint32_t> operands = {}) {
                const auto result = id();
                types_.push_back((uint32_t(operands.size() + 2) << 16) | uint32_t(code));
                types_.push_back(result);
                types_.insert(types_.end(), operands);
                return result;
            }
            Value instruction(spv::Op code, uint32_t type, std::initializer_list<uint32_t> operands) {
                const auto result = id();
                body_.push_back((uint32_t(operands.size() + 3) << 16) | uint32_t(code));
                body_.push_back(type);
                body_.push_back(result);
                body_.insert(body_.end(), operands);
                if (rounded_ && (code == spv::OpFAdd || code == spv::OpFSub || code == spv::OpFMul || code == spv::OpFDiv))
                    emit(annotations_, spv::OpDecorate, {result, spv::DecorationNoContraction});
                return result;
            }
            Value ext(uint32_t code, std::initializer_list<uint32_t> operands) {
                const auto result = id();
                body_.push_back((uint32_t(operands.size() + 5) << 16) | spv::OpExtInst);
                body_.insert(body_.end(), {float_, result, glsl_, code});
                body_.insert(body_.end(), operands);
                return result;
            }
            Value constant(uint32_t type, uint64_t value) {
                const auto key = std::pair{type, value};
                if (auto found = constants_.find(key); found != constants_.end())
                    return found->second;
                const auto result = id();
                if (type == ulong_)
                    emit(types_, spv::OpConstant, {type, result, uint32_t(value), uint32_t(value >> 32)});
                else
                    emit(types_, spv::OpConstant, {type, result, uint32_t(value)});
                constants_[key] = result;
                if (type == uint_)
                    words_[result] = uint32_t(value);
                return result;
            }
            Value f(float value) { return constant(float_, std::bit_cast<uint32_t>(value)); }
            Value as_float(Value v) { return instruction(spv::OpBitcast, float_, {v}); }
            Value as_int(Value v) { return instruction(spv::OpBitcast, int_, {v}); }
            Value raw(Value v) { return instruction(spv::OpBitcast, uint_, {v}); }
            Value truth(Value v) { return instruction(spv::OpINotEqual, bool_, {v, literal(0)}); }
            Value boolean(Value v) { return instruction(spv::OpSelect, uint_, {v, literal(1), literal(0)}); }
            Value choose(Value predicate, Value yes, Value no, uint32_t type) {
                return instruction(spv::OpSelect, type, {predicate, yes, no});
            }
            Value u(spv::Op op, Value a, Value b) { return instruction(op, uint_, {a, b}); }
            Value fp(spv::Op op, Value a, Value b) { return instruction(op, float_, {a, b}); }
            Value cmp(spv::Op op, Value a, Value b) { return instruction(op, bool_, {a, b}); }
            Value add(Value a, Value b) { return fp(spv::OpFAdd, a, b); }
            Value sub(Value a, Value b) { return fp(spv::OpFSub, a, b); }
            Value mul(Value a, Value b) { return fp(spv::OpFMul, a, b); }
            Value div(Value a, Value b) { return fp(spv::OpFDiv, a, b); }
            Value neg(Value a) { return instruction(spv::OpFNegate, float_, {a}); }
            Value finite(Value a) {
                return cmp(spv::OpULessThan, u(spv::OpBitwiseAnd, raw(a), literal(0x7fffffffu)), literal(0x7f800000u));
            }
            Value ieee_minmax(Value x, Value y, bool maximum) {
                auto xb = raw(x), yb = raw(y);
                auto xm = u(spv::OpBitwiseAnd, xb, literal(0x7fffffff)), ym = u(spv::OpBitwiseAnd, yb, literal(0x7fffffff));
                auto zeros = instruction(spv::OpLogicalAnd, bool_, {cmp(spv::OpIEqual, xm, literal(0)), cmp(spv::OpIEqual, ym, literal(0))});
                auto ordered = choose(cmp(spv::OpFOrdLessThan, maximum ? x : y, maximum ? y : x), y, x, float_);
                auto signed_zero = as_float(u(maximum ? spv::OpBitwiseAnd : spv::OpBitwiseOr, xb, yb));
                auto result = choose(zeros, signed_zero, ordered, float_);
                result = choose(cmp(spv::OpUGreaterThan, ym, literal(0x7f800000)), y, result, float_);
                return choose(cmp(spv::OpUGreaterThan, xm, literal(0x7f800000)), x, result, float_);
            }
            Value round(Value x) {
                auto lower = ext(GLSLstd450Floor, {x}), fraction = sub(x, lower), next = add(lower, f(1));
                auto even = cmp(spv::OpFOrdEqual, fp(spv::OpFRem, lower, f(2)), f(0));
                auto result = choose(cmp(spv::OpFOrdLessThan, fraction, f(.5f)), lower,
                                     choose(cmp(spv::OpFOrdGreaterThan, fraction, f(.5f)), next, choose(even, lower, next, float_), float_), float_);
                auto sign = cmp(spv::OpUGreaterThanEqual, raw(x), literal(0x80000000));
                auto zero = instruction(spv::OpLogicalAnd, bool_, {cmp(spv::OpFOrdEqual, result, f(0)), sign});
                return choose(finite(x), choose(zero, f(-0.0f), result, float_), x, float_);
            }
            Value log1p(Value x) {
                return as_float(select(boolean(cmp(spv::OpFOrdEqual, x, f(0))), [&] { return raw(x); }, [&] {
                    auto special = instruction(spv::OpLogicalOr, bool_, {cmp(spv::OpFOrdLessThanEqual, x, f(-1)), instruction(spv::OpLogicalNot, bool_, {finite(x)})});
                    return select(boolean(special), [&] { return raw(ext(GLSLstd450Log, {add(f(1), x)})); }, [&] {
                        auto sum = add(f(1), x);
                        auto exponent_bits = u(spv::OpBitwiseAnd, u(spv::OpISub, raw(sum), literal(1061158912)), literal(uint32_t(-8388608)));
                        auto reduced = as_float(u(spv::OpISub, raw(x), exponent_bits));
                        auto scale = as_float(u(spv::OpISub, literal(1082130432), exponent_bits));
                        auto v = add(reduced, ext(GLSLstd450Fma, {f(.25f), scale, f(-1)}));
                        auto exponent = mul(instruction(spv::OpConvertSToF, float_, {as_int(exponent_bits)}), f(1.1920928955078125e-7f));
                        auto polynomial = f(-0.04534861445426941f);
                        for (float coefficient : {0.10546888411045074f, -0.13229703903198242f, 0.14491446316242218f, -0.16641564667224884f,
                                                  0.199888676404953f, -0.2500019669532776f, 0.33333510160446167f, -0.5f})
                            polynomial = ext(GLSLstd450Fma, {polynomial, v, f(coefficient)});
                        auto reduced_log = ext(GLSLstd450Fma, {mul(polynomial, v), v, v});
                        return raw(ext(GLSLstd450Fma, {exponent, f(0.6931471824645996f), reduced_log}));
                    }); }));
            }
            Value pointer(Value address) { return instruction(spv::OpConvertUToPtr, physical_uint_, {address}); }
            Value load_word(Value address) { return instruction(spv::OpLoad, uint_, {pointer(address), spv::MemoryAccessAlignedMask, 4}); }
            Value word_at(Value word) {
                if (direct_)
                    return instruction(spv::OpLoad, uint_, {instruction(spv::OpAccessChain, push_uint_, {push_, literal(0), word})});
                auto base = instruction(spv::OpLoad, ulong_, {instruction(spv::OpAccessChain, push_ulong_, {push_, literal(0)})});
                auto offset = instruction(spv::OpIMul, ulong_, {instruction(spv::OpUConvert, ulong_, {word}), constant(ulong_, 4)});
                return load_word(instruction(spv::OpIAdd, ulong_, {base, offset}));
            }
            Value address(uint32_t word, Value index, uint32_t width) {
                auto low = instruction(spv::OpUConvert, ulong_, {arguments_[word]});
                auto high = instruction(spv::OpUConvert, ulong_, {arguments_[word + 1]});
                auto base = instruction(spv::OpBitwiseOr, ulong_, {low, instruction(spv::OpShiftLeftLogical, ulong_, {high, constant(ulong_, 32)})});
                auto offset = instruction(spv::OpIMul, ulong_, {instruction(spv::OpUConvert, ulong_, {index}), constant(ulong_, width)});
                return instruction(spv::OpIAdd, ulong_, {base, offset});
            }

        public:
            explicit SpirvEmitter(uint32_t words) : direct_(words * 4 <= 128) {
                emit(header_, spv::OpCapability, {spv::CapabilityShader});
                emit(header_, spv::OpCapability, {spv::CapabilityInt64});
                emit(header_, spv::OpCapability, {spv::CapabilityPhysicalStorageBufferAddresses});
                glsl_ = id();
                string(header_, spv::OpExtInstImport, {glsl_}, "GLSL.std.450");
                emit(header_, spv::OpMemoryModel, {spv::AddressingModelPhysicalStorageBuffer64, spv::MemoryModelGLSL450});
                void_ = type(spv::OpTypeVoid);
                bool_ = type(spv::OpTypeBool);
                uint_ = type(spv::OpTypeInt, {32, 0});
                int_ = type(spv::OpTypeInt, {32, 1});
                float_ = type(spv::OpTypeFloat, {32});
                ulong_ = type(spv::OpTypeInt, {64, 0});
                uint3_ = type(spv::OpTypeVector, {uint_, 3});
                wide_ = type(spv::OpTypeStruct, {uint_, uint_});
                function_type_ = type(spv::OpTypeFunction, {void_});
                local_uint_ = type(spv::OpTypePointer, {spv::StorageClassFunction, uint_});
                physical_uint_ = type(spv::OpTypePointer, {spv::StorageClassPhysicalStorageBuffer, uint_});
                push_uint_ = type(spv::OpTypePointer, {spv::StorageClassPushConstant, uint_});
                push_ulong_ = type(spv::OpTypePointer, {spv::StorageClassPushConstant, ulong_});
                auto member = ulong_;
                if (direct_) {
                    member = type(spv::OpTypeArray, {uint_, literal(words)});
                    emit(annotations_, spv::OpDecorate, {member, spv::DecorationArrayStride, 4});
                }
                auto bindings = type(spv::OpTypeStruct, {member});
                emit(annotations_, spv::OpDecorate, {bindings, spv::DecorationBlock});
                emit(annotations_, spv::OpMemberDecorate, {bindings, 0, spv::DecorationOffset, 0});
                auto push_pointer = type(spv::OpTypePointer, {spv::StorageClassPushConstant, bindings});
                push_ = id();
                emit(globals_, spv::OpVariable, {push_pointer, push_, spv::StorageClassPushConstant});
                auto input_pointer = type(spv::OpTypePointer, {spv::StorageClassInput, uint3_});
                thread_ = id();
                groups_ = id();
                emit(globals_, spv::OpVariable, {input_pointer, thread_, spv::StorageClassInput});
                emit(globals_, spv::OpVariable, {input_pointer, groups_, spv::StorageClassInput});
                emit(annotations_, spv::OpDecorate, {thread_, spv::DecorationBuiltIn, spv::BuiltInGlobalInvocationId});
                emit(annotations_, spv::OpDecorate, {groups_, spv::DecorationBuiltIn, spv::BuiltInNumWorkgroups});
                entry_ = id();
                string(header_, spv::OpEntryPoint, {spv::ExecutionModelGLCompute, entry_}, "main", {push_, thread_, groups_});
                emit(header_, spv::OpExecutionMode, {entry_, spv::ExecutionModeLocalSize, 256, 1, 1});
                // Loaded once at entry so every use is dominated; unused words are dead code.
                for (uint32_t word = 0; word < words; ++word)
                    arguments_.push_back(word_at(literal(word)));
            }
            Value literal(uint32_t value) override { return constant(uint_, value); }
            Value math(ExprOp op, Value a, Value b, Value c) override {
                switch (op) {
                case ExprOp::AddInt: return u(spv::OpIAdd, a, b);
                case ExprOp::SubInt: return u(spv::OpISub, a, b);
                case ExprOp::MulInt: return u(spv::OpIMul, a, b);
                case ExprOp::BitAnd: return u(spv::OpBitwiseAnd, a, b);
                case ExprOp::BitOr: return u(spv::OpBitwiseOr, a, b);
                case ExprOp::BitXor: return u(spv::OpBitwiseXor, a, b);
                case ExprOp::BitNot: return instruction(spv::OpNot, uint_, {a});
                case ExprOp::ShiftLeft: return u(spv::OpShiftLeftLogical, a, u(spv::OpBitwiseAnd, b, literal(31)));
                case ExprOp::ShiftRight: return u(spv::OpShiftRightLogical, a, u(spv::OpBitwiseAnd, b, literal(31)));
                case ExprOp::EqualInt: return boolean(cmp(spv::OpIEqual, a, b));
                case ExprOp::NotEqualInt: return boolean(cmp(spv::OpINotEqual, a, b));
                case ExprOp::LessInt: return boolean(cmp(spv::OpSLessThan, as_int(a), as_int(b)));
                case ExprOp::LessEqualInt: return boolean(cmp(spv::OpSLessThanEqual, as_int(a), as_int(b)));
                case ExprOp::GreaterInt: return boolean(cmp(spv::OpSGreaterThan, as_int(a), as_int(b)));
                case ExprOp::GreaterEqualInt: return boolean(cmp(spv::OpSGreaterThanEqual, as_int(a), as_int(b)));
                case ExprOp::LessUInt: return boolean(cmp(spv::OpULessThan, a, b));
                case ExprOp::LessEqualUInt: return boolean(cmp(spv::OpULessThanEqual, a, b));
                case ExprOp::GreaterUInt: return boolean(cmp(spv::OpUGreaterThan, a, b));
                case ExprOp::GreaterEqualUInt: return boolean(cmp(spv::OpUGreaterThanEqual, a, b));
                case ExprOp::LogicalAnd: return boolean(instruction(spv::OpLogicalAnd, bool_, {truth(a), truth(b)}));
                case ExprOp::LogicalOr: return boolean(instruction(spv::OpLogicalOr, bool_, {truth(a), truth(b)}));
                case ExprOp::LogicalXor: return boolean(instruction(spv::OpLogicalNotEqual, bool_, {truth(a), truth(b)}));
                case ExprOp::LogicalNot: return boolean(cmp(spv::OpIEqual, a, literal(0)));
                case ExprOp::Select: return choose(truth(a), b, c, uint_);
                case ExprOp::UIntDiv:
                case ExprOp::UIntMod: {
                    auto divisor = choose(truth(b), b, literal(1), uint_);
                    auto value = u(op == ExprOp::UIntDiv ? spv::OpUDiv : spv::OpUMod, a, divisor);
                    return choose(truth(b), value, literal(0), uint_);
                }
                case ExprOp::IntDiv:
                case ExprOp::IntMod: {
                    auto an = cmp(spv::OpSLessThan, as_int(a), as_int(literal(0))), bn = cmp(spv::OpSLessThan, as_int(b), as_int(literal(0)));
                    auto aa = choose(an, u(spv::OpISub, literal(0), a), a, uint_);
                    auto bb = choose(bn, u(spv::OpISub, literal(0), b), b, uint_);
                    auto result = math(op == ExprOp::IntDiv ? ExprOp::UIntDiv : ExprOp::UIntMod, aa, bb, 0);
                    auto negative = op == ExprOp::IntMod ? an : instruction(spv::OpLogicalNotEqual, bool_, {an, bn});
                    return choose(negative, u(spv::OpISub, literal(0), result), result, uint_);
                }
                case ExprOp::CastFloat: return raw(instruction(spv::OpConvertSToF, float_, {as_int(a)}));
                case ExprOp::CastUnsignedFloat: return raw(instruction(spv::OpConvertUToF, float_, {a}));
                case ExprOp::CastInt: return raw(instruction(spv::OpConvertFToS, int_, {as_float(a)}));
                case ExprOp::CastUInt: return instruction(spv::OpConvertFToU, uint_, {as_float(a)});
                default: break;
                }
                const auto x = as_float(a);
                const auto y = expr_arity(op) > 1 ? as_float(b) : f(0);
                const auto z = expr_arity(op) > 2 ? as_float(c) : f(0);
                struct RoundingScope {
                    bool& flag;
                    bool saved;
                    ~RoundingScope() { flag = saved; }
                } rounding{rounded_, rounded_};
                rounded_ = op == ExprOp::Add || op == ExprOp::Sub || op == ExprOp::Mul || op == ExprOp::Div || op == ExprOp::Square;
                Value result;
                switch (op) {
                case ExprOp::Add: result = add(x, y); break;
                case ExprOp::Sub: result = sub(x, y); break;
                case ExprOp::Mul: result = mul(x, y); break;
                case ExprOp::Div: result = div(x, y); break;
                case ExprOp::Mod: result = fp(spv::OpFRem, x, y); break;
                case ExprOp::Pow: result = choose(cmp(spv::OpFOrdEqual, y, f(2)), mul(x, x), ext(GLSLstd450Pow, {x, y}), float_); break;
                case ExprOp::Min: result = ieee_minmax(x, y, false); break;
                case ExprOp::Max: result = ieee_minmax(x, y, true); break;
                case ExprOp::Atan2: result = ext(GLSLstd450Atan2, {x, y}); break;
                case ExprOp::Fma: result = ext(GLSLstd450Fma, {x, y, z}); break;
                case ExprOp::Clamp: result = choose(instruction(spv::OpIsNan, bool_, {x}), x, ext(GLSLstd450FMin, {ext(GLSLstd450FMax, {x, y}), z}), float_); break;
                case ExprOp::Equal: return boolean(cmp(spv::OpFOrdEqual, x, y));
                case ExprOp::NotEqual: return boolean(cmp(spv::OpFUnordNotEqual, x, y));
                case ExprOp::Less: return boolean(cmp(spv::OpFOrdLessThan, x, y));
                case ExprOp::LessEqual: return boolean(cmp(spv::OpFOrdLessThanEqual, x, y));
                case ExprOp::Greater: return boolean(cmp(spv::OpFOrdGreaterThan, x, y));
                case ExprOp::GreaterEqual: return boolean(cmp(spv::OpFOrdGreaterThanEqual, x, y));
                case ExprOp::IsNan: return boolean(instruction(spv::OpIsNan, bool_, {x}));
                case ExprOp::IsInf: return boolean(instruction(spv::OpIsInf, bool_, {x}));
                case ExprOp::IsFinite: return boolean(finite(x));
                case ExprOp::Abs: result = ext(GLSLstd450FAbs, {x}); break;
                case ExprOp::Neg: result = neg(x); break;
                case ExprOp::Exp: result = ext(GLSLstd450Exp, {x}); break;
                case ExprOp::Log: result = ext(GLSLstd450Log, {x}); break;
                case ExprOp::Sqrt: result = ext(GLSLstd450Sqrt, {x}); break;
                case ExprOp::Sigmoid: result = div(f(1), add(f(1), ext(GLSLstd450Exp, {neg(x)}))); break;
                case ExprOp::Relu: result = ieee_minmax(x, f(0), true); break;
                case ExprOp::Square: result = mul(x, x); break;
                case ExprOp::Tanh: result = ext(GLSLstd450Tanh, {x}); break;
                case ExprOp::Rsqrt: result = ext(GLSLstd450InverseSqrt, {x}); break;
                case ExprOp::Sign: result = instruction(spv::OpConvertSToF, float_, {as_int(u(spv::OpISub, boolean(cmp(spv::OpFOrdGreaterThan, x, f(0))), boolean(cmp(spv::OpFOrdLessThan, x, f(0)))))}); break;
                case ExprOp::Reciprocal: result = div(f(1), x); break;
                case ExprOp::Floor: result = ext(GLSLstd450Floor, {x}); break;
                case ExprOp::Ceil: result = ext(GLSLstd450Ceil, {x}); break;
                case ExprOp::Round: result = round(x); break;
                case ExprOp::Exp2: result = ext(GLSLstd450Exp2, {x}); break;
                case ExprOp::Log2: result = ext(GLSLstd450Log2, {x}); break;
                case ExprOp::Log10: result = mul(ext(GLSLstd450Log, {x}), f(0.4342944819032518f)); break;
                case ExprOp::Log1p: result = log1p(x); break;
                case ExprOp::Sin: result = ext(GLSLstd450Sin, {x}); break;
                case ExprOp::Cos: result = ext(GLSLstd450Cos, {x}); break;
                case ExprOp::Tan: result = ext(GLSLstd450Tan, {x}); break;
                case ExprOp::Asin:
                case ExprOp::Acos: {
                    auto bounded = ext(GLSLstd450FClamp, {x, f(-1), f(1)});
                    auto root = ext(GLSLstd450Sqrt, {ext(GLSLstd450FMax, {f(0), sub(f(1), mul(bounded, bounded))})});
                    result = op == ExprOp::Asin ? ext(GLSLstd450Atan2, {bounded, root}) : ext(GLSLstd450Atan2, {root, bounded});
                    break;
                }
                case ExprOp::Atan: result = ext(GLSLstd450Atan, {x}); break;
                case ExprOp::Sinh: result = ext(GLSLstd450Sinh, {x}); break;
                case ExprOp::Cosh: result = ext(GLSLstd450Cosh, {x}); break;
                case ExprOp::Gelu: {
                    auto inner = mul(f(std::sqrt(2.0f / 3.14159265358979323846f)), add(x, mul(mul(mul(f(.044715f), x), x), x)));
                    result = mul(mul(f(.5f), x), add(f(1), ext(GLSLstd450Tanh, {inner})));
                    break;
                }
                case ExprOp::Swish: result = div(x, add(f(1), ext(GLSLstd450Exp, {neg(x)}))); break;
                case ExprOp::Trunc: result = ext(GLSLstd450Trunc, {x}); break;
                default: throw std::invalid_argument("No SPIR-V lowering for expression opcode");
                }
                return raw(result);
            }
            Value mul_hi(Value a, Value b) override {
                return instruction(spv::OpCompositeExtract, uint_, {instruction(spv::OpUMulExtended, wide_, {a, b}), 1});
            }
            Value variable(Value initial) override {
                auto result = id();
                emit(variables_, spv::OpVariable, {local_uint_, result, spv::StorageClassFunction});
                assign(result, initial);
                return result;
            }
            Value read(Value slot) override { return instruction(spv::OpLoad, uint_, {slot}); }
            void assign(Value slot, Value value) override { emit(body_, spv::OpStore, {slot, value}); }
            void condition(Value predicate, const std::function<void()>& yes, const std::function<void()>& no) override {
                auto test = truth(predicate), then_label = id(), else_label = id(), end = id();
                emit(body_, spv::OpSelectionMerge, {end, spv::SelectionControlMaskNone});
                emit(body_, spv::OpBranchConditional, {test, then_label, else_label});
                emit(body_, spv::OpLabel, {then_label});
                yes();
                emit(body_, spv::OpBranch, {end});
                emit(body_, spv::OpLabel, {else_label});
                if (no)
                    no();
                emit(body_, spv::OpBranch, {end});
                emit(body_, spv::OpLabel, {end});
            }
            void loop(Value first, Value end, Value step, const std::function<void(Value)>& body) override {
                const auto slot = variable(first), head = id(), next = id(), run = id(), done = id();
                emit(body_, spv::OpBranch, {head});
                emit(body_, spv::OpLabel, {head});
                const auto index = read(slot), test = cmp(spv::OpULessThan, index, end);
                emit(body_, spv::OpLoopMerge, {done, next, spv::LoopControlDontUnrollMask});
                emit(body_, spv::OpBranchConditional, {test, run, done});
                emit(body_, spv::OpLabel, {run});
                body(index);
                emit(body_, spv::OpBranch, {next});
                emit(body_, spv::OpLabel, {next});
                assign(slot, u(spv::OpIAdd, index, step));
                emit(body_, spv::OpBranch, {head});
                emit(body_, spv::OpLabel, {done});
            }
            Value argument(Value word) override {
                const auto found = words_.find(word);
                return found != words_.end() && found->second < arguments_.size() ? arguments_[found->second] : word_at(word);
            }
            Value load(uint32_t word, Value index, DataType dtype) override {
                auto p = address(word, index, dtype_size(dtype));
                if (dtype_size(dtype) == 4)
                    return load_word(p);
                auto aligned = instruction(spv::OpBitwiseAnd, ulong_, {p, constant(ulong_, ~uint64_t(3))});
                auto shift = u(spv::OpIMul, u(spv::OpBitwiseAnd, instruction(spv::OpUConvert, uint_, {p}), literal(3)), literal(8));
                auto value = u(spv::OpBitwiseAnd, u(spv::OpShiftRightLogical, load_word(aligned), shift), literal(dtype == DataType::Float16 ? 65535 : 255));
                return dtype == DataType::Float16 ? half(value, false) : value;
            }
            std::array<Value, 2> load_pair(uint32_t, Value) override { throw std::logic_error("SPIR-V loads elements singly"); }
            Value output(uint32_t word, Value index) override { return load_word(address(word, index, 4)); }
            void store(uint32_t word, Value index, Value value, DataType dtype) override {
                if (dtype_size(dtype) != 4)
                    throw std::logic_error("SPIR-V outputs own complete words");
                emit(body_, spv::OpStore, {pointer(address(word, index, 4)), value, spv::MemoryAccessAlignedMask, 4});
            }
            Value half(Value value, bool pack) override {
                const auto band = [&](Value a, uint32_t b) { return u(spv::OpBitwiseAnd, a, literal(b)); };
                const auto bor = [&](Value a, Value b) { return u(spv::OpBitwiseOr, a, b); };
                const auto shr = [&](Value a, uint32_t b) { return u(spv::OpShiftRightLogical, a, literal(b)); };
                const auto shl = [&](Value a, uint32_t b) { return u(spv::OpShiftLeftLogical, a, literal(b)); };
                if (!pack) {
                    auto sign = shl(band(value, 0x8000), 16), exponent = band(shr(value, 10), 31), mantissa = band(value, 1023);
                    auto subnormal = bor(sign, raw(mul(instruction(spv::OpConvertUToF, float_, {mantissa}), f(5.9604644775390625e-8f))));
                    auto special = bor(bor(sign, literal(0x7f800000)), shl(mantissa, 13));
                    auto normal = bor(bor(sign, shl(u(spv::OpIAdd, exponent, literal(112)), 23)), shl(mantissa, 13));
                    return choose(cmp(spv::OpIEqual, exponent, literal(0)), subnormal,
                                  choose(cmp(spv::OpIEqual, exponent, literal(31)), special, normal, uint_), uint_);
                }
                auto sign = band(shr(value, 16), 0x8000), magnitude = band(value, 0x7fffffff);
                return select(boolean(cmp(spv::OpUGreaterThan, magnitude, literal(0x7f800000))), [&] {
                    auto payload = shr(band(magnitude, 0x7fffff), 13);
                    return bor(bor(sign, literal(0x7c00)), choose(cmp(spv::OpUGreaterThan, payload, literal(1)), payload, literal(1), uint_)); }, [&] { return select(boolean(cmp(spv::OpUGreaterThanEqual, magnitude, literal(0x477ff000))), [&] { return bor(sign, literal(0x7c00)); }, [&] { return select(boolean(cmp(spv::OpULessThan, magnitude, literal(0x38800000))), [&] { return select(boolean(cmp(spv::OpULessThan, magnitude, literal(0x33000000))), [&] { return sign; }, [&] {
                            auto shift = u(spv::OpISub, literal(126), shr(magnitude, 23));
                            auto significand = bor(band(magnitude, 0x7fffff), literal(0x800000));
                            auto truncated = u(spv::OpShiftRightLogical, significand, shift);
                            auto remainder = u(spv::OpBitwiseAnd, significand, u(spv::OpISub, u(spv::OpShiftLeftLogical, literal(1), shift), literal(1)));
                            auto halfway = u(spv::OpShiftLeftLogical, literal(1), u(spv::OpISub, shift, literal(1)));
                            auto tie = instruction(spv::OpLogicalAnd, bool_, {cmp(spv::OpIEqual, remainder, halfway), truth(band(truncated, 1))});
                            auto up = instruction(spv::OpLogicalOr, bool_, {cmp(spv::OpUGreaterThan, remainder, halfway), tie});
                            return bor(sign, u(spv::OpIAdd, truncated, boolean(up))); }); }, [&] {
                        auto rounded = u(spv::OpIAdd, u(spv::OpIAdd, magnitude, literal(0xfff)), band(shr(magnitude, 13), 1));
                        return bor(sign, shr(u(spv::OpISub, rounded, literal(0x38000000)), 13)); }); }); });
            }
            Value thread() override { return instruction(spv::OpCompositeExtract, uint_, {instruction(spv::OpLoad, uint3_, {thread_}), 0}); }
            Value grid_stride() override {
                return u(spv::OpIMul, instruction(spv::OpCompositeExtract, uint_, {instruction(spv::OpLoad, uint3_, {groups_}), 0}), literal(256));
            }
            bool cuda() const override { return false; }
            Words finish() {
                Words result{0x07230203, 0x00010500, 0, 0, 0};
                for (const auto* section : {&header_, &annotations_, &types_, &globals_})
                    result.insert(result.end(), section->begin(), section->end());
                emit(result, spv::OpFunction, {void_, entry_, spv::FunctionControlMaskNone, function_type_});
                emit(result, spv::OpLabel, {id()});
                result.insert(result.end(), variables_.begin(), variables_.end());
                result.insert(result.end(), body_.begin(), body_.end());
                emit(result, spv::OpReturn, {});
                emit(result, spv::OpFunctionEnd, {});
                result[3] = next_;
                return spirv::finalize(result);
            }
        };
    } // namespace
    std::vector<uint32_t> expression_spirv(const ExpressionProgram& program, const ExpressionSignature& signature) {
        SpirvEmitter emitter(expression_layout(program, signature).words);
        emit_expression(emitter, program, signature);
        return emitter.finish();
    }
} // namespace lfs::core::internal
