/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "internal/expression_emitter.hpp"
#include <algorithm>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>

namespace lfs::core::internal {
    namespace {
        ExprOp opcode(const ExprInstruction& ins) { return ExprOp(ins.op_dst_a_b & 255); }
        uint32_t aux_of(const ExprInstruction& ins) { return (ins.c_aux_policy >> 8) & 255; }
        uint32_t policy_of(const ExprInstruction& ins) { return (ins.c_aux_policy >> 16) & 255; }
        uint32_t gather_rank(const ExprOp op) {
            return op == ExprOp::Gather ? 1 : op == ExprOp::Gather2 ? 2
                                          : op == ExprOp::Gather3   ? 3
                                                                    : 0;
        }
    } // namespace

    uint32_t expression_packing(const ExpressionSignature& signature) {
        uint32_t packing = 1;
        for (uint32_t o = 0; o < signature.outputs; ++o)
            packing = std::max(packing, uint32_t(4 / dtype_size(signature.output[o].dtype)));
        return packing;
    }

    ExpressionLayout expression_layout(const ExpressionProgram& program, const ExpressionSignature& signature) {
        ExpressionLayout layout;
        const auto& arity = program.analysis().arity;
        uint32_t next = 0;
        for (uint32_t i = 0; i < signature.inputs; ++i)
            if (!signature.input[i].host) {
                layout.input[i] = next;
                next += 2;
            }
        for (uint32_t o = 0; o < signature.outputs; ++o) {
            layout.output[o] = next;
            next += 2;
        }
        layout.count = next++;
        int outer = -1;
        for (uint32_t d = 0; d < signature.rank; ++d) {
            layout.dims[d] = next++;
            if (outer < 0 && int(d) != signature.fold_dim)
                outer = int(d);
        }
        for (uint32_t d = 0; d < signature.rank; ++d)
            if (int(d) != signature.fold_dim && int(d) != outer) {
                layout.magic[d] = next++;
                layout.shift[d] = next++;
            }
        bool first_host = true;
        for (uint32_t i = 0; i < signature.inputs; ++i) {
            const auto& input = signature.input[i];
            if (input.host) {
                layout.input[i] = first_host ? ExpressionLayout::none : next++;
                first_host = false;
            }
            for (uint32_t d = 0; d < ExpressionProgram::max_rank; ++d) {
                layout.input_stride[i][d] = input.strides[d] == ExprStride::Strided ? next++ : ExpressionLayout::none;
                layout.bound[i][d] = d < arity[i] ? next++ : ExpressionLayout::none;
            }
        }
        for (uint32_t o = 0; o < signature.outputs; ++o)
            for (uint32_t d = 0; d < ExpressionProgram::max_rank; ++d)
                layout.output_stride[o][d] = signature.output[o].strides[d] == ExprStride::Strided ? next++ : ExpressionLayout::none;
        if (!first_host) {
            layout.bank = next;
            uint32_t scalars = 0;
            bool scalar_bank = true;
            for (uint32_t i = 0; i < signature.inputs; ++i)
                if (signature.input[i].host) {
                    ++scalars;
                    for (auto stride : signature.input[i].strides)
                        scalar_bank &= stride == ExprStride::Zero;
                }
            next += scalar_bank ? scalars : ExpressionProgram::max_parameters;
        }
        layout.words = next;
        if (layout.words > ExpressionLayout::max_words)
            throw std::logic_error("expression argument block exceeds its capacity");
        return layout;
    }

    std::string expression_key(const ExpressionProgram& program, const ExpressionSignature& signature) {
        std::string key = "expression-direct-v2:";
        key += expression_emitter_hash();
        auto append = [&](uint32_t word) { key.append(reinterpret_cast<const char*>(&word), sizeof(word)); };
        append(program.instructions().size());
        const auto instructions = program.instructions();
        key.append(reinterpret_cast<const char*>(instructions.data()), instructions.size_bytes());
        append(signature.rank);
        append(uint32_t(signature.fold_dim));
        append(signature.fold_length);
        append(uint32_t(signature.gather_in_range));
        append(signature.inputs);
        append(signature.outputs);
        const auto binding = [&](const ExpressionSignature::Binding& b) {
            uint32_t classes = 0;
            for (uint32_t d = 0; d < ExpressionProgram::max_rank; ++d)
                classes |= uint32_t(b.strides[d]) << (2 * d);
            append(uint32_t(b.dtype) | uint32_t(b.host) << 8 | uint32_t(b.linear) << 9 |
                   uint32_t(uint8_t(b.pair)) << 16 | classes << 24);
        };
        for (uint32_t i = 0; i < signature.inputs; ++i)
            binding(signature.input[i]);
        for (uint32_t o = 0; o < signature.outputs; ++o)
            binding(signature.output[o]);
        return key;
    }

    std::span<const std::array<uint32_t, 3>> expression_sources(const ExpressionProgram& program) {
        return program.analysis().sources;
    }

    std::vector<ExprPhase> expression_phases(const ExpressionProgram& program, const ExpressionSignature& signature) {
        const auto instructions = program.instructions();
        const auto sources = expression_sources(program);
        std::vector<ExprPhase> phases(instructions.size(), ExprPhase::Invariant);
        const int fold = signature.fold_dim;
        for (uint32_t pc = 0; pc < instructions.size(); ++pc) {
            const auto op = opcode(instructions[pc]);
            const auto aux = aux_of(instructions[pc]);
            auto phase = ExprPhase::Invariant;
            if (op == ExprOp::Fold)
                phase = ExprPhase::Folded;
            else if (op == ExprOp::Iota && int(aux) == fold)
                phase = ExprPhase::Variant;
            else if (op == ExprOp::Load && fold >= 0 && signature.input[aux].strides[fold] != ExprStride::Zero)
                phase = ExprPhase::Variant;
            for (uint32_t a = 0; a < expr_arity(op); ++a) {
                const auto operand = phases[sources[pc][a]];
                if (op == ExprOp::Fold) {
                    if (operand == ExprPhase::Folded)
                        throw std::invalid_argument("expression folds cannot nest");
                } else if (operand != ExprPhase::Invariant) {
                    if (phase != ExprPhase::Invariant && phase != operand)
                        throw std::invalid_argument("expression value depends on its fold dimension after the fold");
                    phase = operand;
                }
            }
            if (op == ExprOp::Store && phase == ExprPhase::Variant)
                throw std::invalid_argument("expression output varies along its fold dimension");
            phases[pc] = phase;
        }
        return phases;
    }

    bool expression_gather_in_range(const ExpressionProgram& program,
                                    const ExprShape& shape,
                                    const std::array<std::array<uint32_t, ExpressionProgram::max_rank>,
                                                     ExpressionProgram::max_inputs>& bounds) {
        if (!program.analysis().runtime_gathers)
            return false;
        const auto instructions = program.instructions();
        const auto sources = expression_sources(program);
        struct Range {
            int64_t low, high;
        };
        struct Constraint {
            uint32_t node;
            Range range;
        };
        using Key = std::tuple<uint32_t, uint32_t, int64_t, int64_t>;
        std::map<Key, std::optional<Range>> known;
        size_t work = 0;
        bool exhausted = false;
        const auto eval = [&](auto&& self, uint32_t pc, std::optional<Constraint> constraint) -> std::optional<Range> {
            if (exhausted)
                return {};
            if (constraint && constraint->node == pc)
                return constraint->range;
            const Key key{pc, constraint ? constraint->node + 1 : 0,
                          constraint ? constraint->range.low : 0, constraint ? constraint->range.high : 0};
            if (const auto found = known.find(key); found != known.end())
                return found->second;
            // Unproven bounds keep the runtime gather checks.
            if (++work > 4096) {
                exhausted = true;
                return {};
            }
            const auto ins = instructions[pc];
            const auto op = opcode(ins);
            const auto arg = [&](uint32_t n) { return self(self, sources[pc][n], constraint); };
            std::optional<Range> result;
            if (op == ExprOp::Immediate)
                result = Range{int32_t(ins.immediate), int32_t(ins.immediate)};
            else if (op == ExprOp::Iota && shape.dims[aux_of(ins)])
                result = Range{0, int64_t(shape.dims[aux_of(ins)]) - 1};
            else if (op == ExprOp::Extent)
                result = Range{shape.dims[aux_of(ins)], shape.dims[aux_of(ins)]};
            else if (op == ExprOp::AddInt || op == ExprOp::SubInt || op == ExprOp::MulInt) {
                const auto a = arg(0), b = arg(1);
                if (a && b) {
                    Range r{};
                    if (op == ExprOp::AddInt)
                        r = {a->low + b->low, a->high + b->high};
                    else if (op == ExprOp::SubInt)
                        r = {a->low - b->high, a->high - b->low};
                    else {
                        const int64_t products[]{a->low * b->low, a->low * b->high, a->high * b->low, a->high * b->high};
                        r = {*std::min_element(std::begin(products), std::end(products)), *std::max_element(std::begin(products), std::end(products))};
                    }
                    // Wrapping arithmetic cannot justify signed bounds elimination.
                    if (r.low >= INT32_MIN && r.high <= INT32_MAX)
                        result = r;
                }
            } else if (op == ExprOp::EqualInt || op == ExprOp::NotEqualInt) {
                const auto a = arg(0), b = arg(1);
                if (a && b) {
                    const bool disjoint = a->high < b->low || b->high < a->low;
                    const bool equal = a->low == a->high && b->low == b->high && a->low == b->low;
                    if (disjoint || equal) {
                        const int64_t value = op == ExprOp::EqualInt ? equal : !equal;
                        result = Range{value, value};
                    } else
                        result = Range{0, 1};
                }
            } else if (op == ExprOp::Select) {
                const auto predicate = arg(0);
                if (predicate && predicate->low == predicate->high)
                    result = arg(predicate->low ? 1 : 2);
                else {
                    auto yes_constraint = constraint, no_constraint = constraint;
                    const auto test = sources[pc][0];
                    if (!constraint && opcode(instructions[test]) == ExprOp::EqualInt) {
                        const auto node = sources[test][0];
                        const auto variable = self(self, node, {}), constant = self(self, sources[test][1], {});
                        if (variable && constant && constant->low == constant->high &&
                            constant->low >= variable->low && constant->high <= variable->high) {
                            yes_constraint = Constraint{node, *constant};
                            auto other = *variable;
                            if (constant->low == other.low)
                                ++other.low;
                            else if (constant->high == other.high)
                                --other.high;
                            if (other.low <= other.high)
                                no_constraint = Constraint{node, other};
                        }
                    }
                    const auto yes = self(self, sources[pc][1], yes_constraint);
                    const auto no = self(self, sources[pc][2], no_constraint);
                    if (yes && no)
                        result = Range{std::min(yes->low, no->low), std::max(yes->high, no->high)};
                }
            }
            known.emplace(key, result);
            return result;
        };
        bool gathered = false;
        for (uint32_t pc = 0; pc < instructions.size(); ++pc) {
            const auto rank = gather_rank(opcode(instructions[pc]));
            if (!rank || ExprOob(policy_of(instructions[pc])) == ExprOob::Checked)
                continue;
            gathered = true;
            for (uint32_t d = 0; d < rank; ++d) {
                const auto index = eval(eval, sources[pc][d], {});
                if (!index || index->low < 0 || uint64_t(index->high) >= bounds[aux_of(instructions[pc])][d])
                    return false;
            }
        }
        return gathered;
    }

    void emit_expression(ExpressionEmitter& e, const ExpressionProgram& program,
                         const ExpressionSignature& signature) {
        using V = ExpressionEmitter::Value;
        const auto layout = expression_layout(program, signature);
        const auto instructions = program.instructions();
        const auto sources = expression_sources(program);
        const auto phases = expression_phases(program, signature);
        const int fold = signature.fold_dim;
        const uint32_t fold_n = signature.fold_length;
        const auto k = [&](uint32_t n) { return e.literal(n); };
        const auto word = [&](uint32_t offset) { return e.argument(k(offset)); };
        const auto add = [&](V a, V b) { return e.math(ExprOp::AddInt, a, b); };
        const auto sub = [&](V a, V b) { return e.math(ExprOp::SubInt, a, b); };
        const auto mul = [&](V a, V b) { return e.math(ExprOp::MulInt, a, b); };
        const auto less = [&](V a, V b) { return e.math(ExprOp::LessUInt, a, b); };
        const auto equal = [&](V a, V b) { return e.math(ExprOp::EqualInt, a, b); };
        const auto bit_and = [&](V a, V b) { return e.math(ExprOp::BitAnd, a, b); };
        const auto bit_or = [&](V a, V b) { return e.math(ExprOp::BitOr, a, b); };
        const auto shl = [&](V a, V b) { return e.math(ExprOp::ShiftLeft, a, b); };
        const uint32_t packing = e.cuda() ? 1 : expression_packing(signature);
        std::vector<uint32_t> uses(instructions.size());
        for (uint32_t pc = 0; pc < instructions.size(); ++pc)
            for (uint32_t a = 0; a < expr_arity(opcode(instructions[pc])); ++a)
                ++uses[sources[pc][a]];
        std::vector<int> owner(instructions.size(), -1);
        std::vector<bool> gathered(instructions.size()), zero_select(instructions.size());
        for (uint32_t pc = 0; pc < instructions.size(); ++pc) {
            gathered[pc] = gather_rank(opcode(instructions[pc])) != 0;
            for (uint32_t a = 0; a < expr_arity(opcode(instructions[pc])); ++a)
                gathered[pc] = gathered[pc] || gathered[sources[pc][a]];
        }
        for (uint32_t pc = 0; pc < instructions.size(); ++pc) {
            const auto op = opcode(instructions[pc]);
            if (op != ExprOp::LogicalAnd && op != ExprOp::LogicalOr && op != ExprOp::Select)
                continue;
            std::function<void(uint32_t)> claim = [&](uint32_t node) {
                const auto kind = opcode(instructions[node]);
                if (uses[node] != 1 || owner[node] >= 0 || kind == ExprOp::LogicalAnd ||
                    kind == ExprOp::LogicalOr || kind == ExprOp::Select || kind == ExprOp::Fold ||
                    phases[node] != phases[pc])
                    return;
                owner[node] = int(pc);
                for (uint32_t a = 0; a < expr_arity(kind); ++a)
                    claim(sources[node][a]);
            };
            zero_select[pc] = op == ExprOp::Select && gathered[sources[pc][0]] &&
                              opcode(instructions[sources[pc][1]]) == ExprOp::Load &&
                              opcode(instructions[sources[pc][2]]) == ExprOp::Immediate &&
                              instructions[sources[pc][2]].immediate == 0;
            claim(sources[pc][zero_select[pc] ? 0 : 1]);
            if (op == ExprOp::Select)
                claim(sources[pc][2]);
        }
        std::array<bool, ExpressionProgram::max_rank> needed{};
        const auto need = [&](const ExpressionSignature::Binding& binding) {
            if (!binding.linear)
                for (uint32_t d = 0; d < signature.rank; ++d)
                    needed[d] = needed[d] || (int(d) != fold && binding.strides[d] != ExprStride::Zero);
        };
        for (const auto& ins : instructions) {
            if (opcode(ins) == ExprOp::Iota && int(aux_of(ins)) != fold)
                needed[aux_of(ins)] = true;
            if (opcode(ins) == ExprOp::Load)
                need(signature.input[aux_of(ins)]);
        }
        for (uint32_t o = 0; o < signature.outputs; ++o)
            need(signature.output[o]);
        int outer = -1;
        for (uint32_t d = 0; d < signature.rank && outer < 0; ++d)
            if (int(d) != fold)
                outer = int(d);
        int lowest = int(signature.rank);
        for (uint32_t d = 0; d < signature.rank && lowest == int(signature.rank); ++d)
            if (needed[d])
                lowest = int(d);

        const V count = word(layout.count);
        const auto write_packed = [&](uint32_t o, V begin, V packed) {
            const uint32_t width = dtype_size(signature.output[o].dtype) * 8, lanes = 32 / width;
            e.condition(less(begin, count), [&] {
                const auto word_index = e.math(ExprOp::UIntDiv, begin, k(lanes));
                const auto valid = sub(count, begin);
                const auto value = e.select(less(valid, k(lanes)), [&] {
                    const auto mask = sub(shl(k(1), mul(valid, k(width))), k(1));
                    return bit_or(bit_and(e.output(layout.output[o], word_index), e.math(ExprOp::BitNot, mask)), bit_and(packed, mask)); }, [&] { return packed; });
                e.store(layout.output[o], word_index, value);
            });
        };
        const V work = packing > 1 ? e.math(ExprOp::UIntDiv, add(count, k(packing - 1)), k(packing))
                                   : count;
        e.loop(e.thread(), work, e.grid_stride(), [&](V thread) {
            std::vector<std::array<V, 2>> packed(signature.outputs);
            for (uint32_t o = 0; o < signature.outputs; ++o)
                for (uint32_t group = 0; group < packing * dtype_size(signature.output[o].dtype) / 4; ++group)
                    if (dtype_size(signature.output[o].dtype) < 4)
                        packed[o][group] = e.variable(k(0));
            const auto emit_lane = [&](V lane) {
                const V element = add(mul(thread, k(packing)), lane);
                e.condition(less(element, count), [&] {
                    std::array<V, ExpressionProgram::max_rank> coords{};
                    V rest = element;
                    for (int d = int(signature.rank) - 1; d >= lowest; --d) {
                        if (d == fold)
                            continue;
                        if (d == outer) {
                            coords[d] = rest;
                            break;
                        }
                        // Exact for dividends below 2^31: q = (mulhi(n, magic) + n) >> shift.
                        const auto quotient = e.math(ExprOp::ShiftRight, add(e.mul_hi(rest, word(layout.magic[d])), rest),
                                                     word(layout.shift[d]));
                        if (needed[d])
                            coords[d] = sub(rest, mul(quotient, word(layout.dims[d])));
                        rest = quotient;
                    }
                    V fold_index = 0;
                    const auto coord = [&](uint32_t d) { return int(d) == fold ? fold_index : coords[d]; };
                    const auto strided = [&](const ExpressionSignature::Binding& binding, uint32_t d, V index, uint32_t stride) {
                        if (binding.strides[d] == ExprStride::Unit)
                            return index;
                        return binding.strides[d] == ExprStride::Strided ? mul(index, word(stride)) : k(0);
                    };
                    const auto view_index = [&](const ExpressionSignature::Binding& binding, const auto& strides) {
                        if (binding.linear)
                            return element;
                        V index = k(0);
                        for (uint32_t d = 0; d < signature.rank; ++d)
                            if (binding.strides[d] != ExprStride::Zero)
                                index = add(index, strided(binding, d, coord(d), strides[d]));
                        return index;
                    };
                    const auto read = [&](uint32_t i, V index) {
                        const auto& binding = signature.input[i];
                        if (!binding.host)
                            return e.load(layout.input[i], index, binding.dtype);
                        const auto start = layout.input[i] == ExpressionLayout::none ? k(layout.bank) : add(k(layout.bank), word(layout.input[i]));
                        return e.argument(add(start, index));
                    };
                    std::vector<V> values(instructions.size());
                    std::function<V(uint32_t)> node = [&](uint32_t pc) -> V {
                        if (values[pc])
                            return values[pc];
                        const auto ins = instructions[pc];
                        const auto op = opcode(ins);
                        const auto aux = aux_of(ins);
                        const auto arg = [&](uint32_t n) { return node(sources[pc][n]); };
                        V result = 0;
                        if (op == ExprOp::Immediate)
                            result = k(ins.immediate);
                        else if (op == ExprOp::Iota)
                            result = coord(aux);
                        else if (op == ExprOp::Extent)
                            result = int(aux) == fold && fold_n ? k(fold_n) : word(layout.dims[aux]);
                        else if (op == ExprOp::Load) {
                            const auto& binding = signature.input[aux];
                            const auto index = view_index(binding, layout.input_stride[aux]);
                            int low = binding.pair >= 0 ? int(aux) : -1;
                            for (uint32_t i = 0; i < signature.inputs && low < 0; ++i)
                                if (signature.input[i].pair == int(aux))
                                    low = int(i);
                            result = low >= 0 && e.cuda() ? e.load_pair(layout.input[low], index)[low == int(aux) ? 0 : 1]
                                                          : read(aux, index);
                        } else if (const auto rank = gather_rank(op)) {
                            const auto& binding = signature.input[aux];
                            const auto policy = ExprOob(policy_of(ins));
                            V index = k(0), valid = k(1);
                            for (uint32_t d = 0; d < rank; ++d) {
                                auto position = arg(d);
                                if (policy != ExprOob::Checked && !signature.gather_in_range) {
                                    const auto n = word(layout.bound[aux][d]);
                                    if (policy == ExprOob::Clamp) {
                                        position = e.math(ExprOp::Select, e.math(ExprOp::LessInt, position, k(0)), k(0), position);
                                        position = e.math(ExprOp::Select, less(position, n), position, sub(n, k(1)));
                                    } else if (policy == ExprOob::Wrap) {
                                        position = e.math(ExprOp::IntMod, position, n);
                                        position = e.math(ExprOp::Select, e.math(ExprOp::LessInt, position, k(0)), add(position, n), position);
                                    } else
                                        valid = d == 0 ? less(position, n) : bit_and(valid, less(position, n));
                                }
                                if (binding.strides[d] != ExprStride::Zero)
                                    index = add(index, strided(binding, d, position, layout.input_stride[aux][d]));
                            }
                            result = policy == ExprOob::Zero
                                         ? e.select(valid, [&] { return read(aux, index); }, [&] { return k(0); })
                                         : read(aux, index);
                        } else if (zero_select[pc]) {
                            const auto value = arg(1);
                            result = e.select(value, [&] { return e.math(ExprOp::Select, arg(0), value, k(0)); }, [&] { return k(0); });
                        } else if ((op == ExprOp::LogicalAnd || op == ExprOp::LogicalOr || op == ExprOp::Select) &&
                                   std::find(owner.begin(), owner.end(), int(pc)) != owner.end()) {
                            const auto a = arg(0);
                            const auto boolean = [&](V v) { return e.math(ExprOp::NotEqualInt, v, k(0)); };
                            result = e.select(a, [&] { return op == ExprOp::LogicalOr ? k(1) : op == ExprOp::Select ? arg(1)
                                                                                                                    : boolean(arg(1)); }, [&] { return op == ExprOp::LogicalAnd ? k(0) : op == ExprOp::Select ? arg(2)
                                                                                                                                                                                                                       : boolean(arg(1)); });
                        } else if (op == ExprOp::Fold || op == ExprOp::Store) {
                            throw std::logic_error("expression fold or store evaluated as a value");
                        } else {
                            const auto a = arg(0), b = expr_arity(op) > 1 ? arg(1) : k(0), c = expr_arity(op) > 2 ? arg(2) : k(0);
                            result = e.math(op, a, b, c);
                        }
                        return values[pc] = result;
                    };
                    if (fold >= 0) {
                        // Values defined inside the fold loop do not dominate the code after it.
                        for (uint32_t pc = 0; pc < instructions.size(); ++pc)
                            if (phases[pc] == ExprPhase::Invariant && owner[pc] < 0 && opcode(instructions[pc]) != ExprOp::Store)
                                node(pc);
                        std::vector<V> accumulators(instructions.size()), compensations(instructions.size());
                        for (uint32_t pc = 0; pc < instructions.size(); ++pc) {
                            if (opcode(instructions[pc]) != ExprOp::Fold)
                                continue;
                            const auto kind = ExprReduce(aux_of(instructions[pc]));
                            const auto type = DataType(instructions[pc].immediate);
                            const bool floating = type == DataType::Float32;
                            uint32_t initial = 0;
                            if (kind == ExprReduce::Min)
                                initial = floating ? 0x7f800000u : type == DataType::Int32 ? 0x7fffffffu
                                                                                           : ~0u;
                            if (kind == ExprReduce::Max)
                                initial = floating ? 0xff800000u : type == DataType::Int32 ? 0x80000000u
                                                                                           : 0u;
                            if (kind == ExprReduce::And)
                                initial = type == DataType::Bool ? 1u : ~0u;
                            accumulators[pc] = e.variable(k(initial));
                            if (kind == ExprReduce::Sum && floating)
                                compensations[pc] = e.variable(k(0));
                        }
                        const auto fold_body = [&](V i) {
                            fold_index = i;
                            if (fold_n)
                                for (uint32_t pc = 0; pc < instructions.size(); ++pc)
                                    if (phases[pc] == ExprPhase::Variant)
                                        values[pc] = 0;
                            for (uint32_t pc = 0; pc < instructions.size(); ++pc) {
                                if (opcode(instructions[pc]) != ExprOp::Fold) {
                                    if (phases[pc] == ExprPhase::Variant && owner[pc] < 0)
                                        node(pc);
                                    continue;
                                }
                                const auto kind = ExprReduce(aux_of(instructions[pc]));
                                const auto type = DataType(instructions[pc].immediate);
                                const bool floating = type == DataType::Float32;
                                auto value = node(sources[pc][0]), acc = e.read(accumulators[pc]);
                                switch (kind) {
                                case ExprReduce::Sum:
                                    if (floating) {
                                        // Compensated summation keeps the low-order bits a serial Float32 sum drops.
                                        const auto y = e.math(ExprOp::Sub, value, e.read(compensations[pc]));
                                        const auto t = e.math(ExprOp::Add, acc, y);
                                        const auto correction = e.math(ExprOp::Sub, e.math(ExprOp::Sub, t, acc), y);
                                        e.assign(compensations[pc], e.math(ExprOp::Select, e.math(ExprOp::IsFinite, t), correction, k(0)));
                                        value = t;
                                    } else
                                        value = add(acc, value);
                                    break;
                                case ExprReduce::And: value = bit_and(acc, value); break;
                                case ExprReduce::Or: value = bit_or(acc, value); break;
                                case ExprReduce::Xor: value = e.math(ExprOp::BitXor, acc, value); break;
                                case ExprReduce::Count: value = add(acc, e.math(ExprOp::NotEqualInt, value, k(0))); break;
                                default:
                                    if (floating)
                                        value = e.math(kind == ExprReduce::Min ? ExprOp::Min : ExprOp::Max, acc, value);
                                    else {
                                        const auto pred = e.math(type == DataType::Int32 ? ExprOp::LessInt : ExprOp::LessUInt, acc, value);
                                        value = kind == ExprReduce::Min ? e.math(ExprOp::Select, pred, acc, value)
                                                                        : e.math(ExprOp::Select, pred, value, acc);
                                    }
                                }
                                e.assign(accumulators[pc], value);
                            }
                        };
                        if (fold_n)
                            for (uint32_t i = 0; i < fold_n; ++i)
                                fold_body(k(i));
                        else
                            e.loop(k(0), word(layout.dims[fold]), k(1), fold_body);
                        for (uint32_t pc = 0; pc < instructions.size(); ++pc)
                            if (opcode(instructions[pc]) == ExprOp::Fold)
                                values[pc] = e.read(accumulators[pc]);
                    }
                    std::array<V, ExpressionProgram::max_outputs> output_values{};
                    for (uint32_t pc = 0; pc < instructions.size(); ++pc) {
                        if (opcode(instructions[pc]) != ExprOp::Store) {
                            if (opcode(instructions[pc]) != ExprOp::Fold &&
                                phases[pc] != ExprPhase::Variant && owner[pc] < 0)
                                node(pc);
                            continue;
                        }
                        output_values[aux_of(instructions[pc])] = node(sources[pc][0]);
                    }
                    // Positional aliases read the original inputs before any output is written.
                    for (uint32_t o = 0; o < signature.outputs; ++o) {
                        const auto& binding = signature.output[o];
                        auto value = output_values[o];
                        if (dtype_size(binding.dtype) == 4 || e.cuda())
                            e.store(layout.output[o], view_index(binding, layout.output_stride[o]), value, binding.dtype);
                        else {
                            const uint32_t width = dtype_size(binding.dtype) * 8, lanes = 32 / width;
                            value = binding.dtype == DataType::Float16 ? e.half(value, true) : bit_and(value, k(255));
                            const auto shifted = shl(value, mul(e.math(ExprOp::UIntMod, lane, k(lanes)), k(width)));
                            for (uint32_t group = 0; group < packing / lanes; ++group)
                                e.condition(equal(e.math(ExprOp::UIntDiv, lane, k(lanes)), k(group)), [&] {
                                    e.assign(packed[o][group], bit_or(e.read(packed[o][group]), shifted));
                                });
                        }
                    }
                });
            };
            if (packing > 1)
                e.loop(k(0), k(packing), k(1), emit_lane);
            else
                emit_lane(k(0));
            if (!e.cuda())
                for (uint32_t o = 0; o < signature.outputs; ++o) {
                    const uint32_t width = dtype_size(signature.output[o].dtype) * 8, lanes = 32 / width;
                    if (width == 32)
                        continue;
                    for (uint32_t group = 0; group < packing / lanes; ++group)
                        write_packed(o, add(mul(thread, k(packing)), k(group * lanes)), e.read(packed[o][group]));
                }
        });
    }
} // namespace lfs::core::internal
