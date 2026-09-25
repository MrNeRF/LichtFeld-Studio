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

    class ExpressionCodegen {
        using V = ExpressionEmitter::Value;

        ExpressionEmitter& e_;
        const ExpressionSignature& signature_;
        ExpressionLayout layout_;
        std::span<const ExprInstruction> instructions_;
        std::span<const std::array<uint32_t, 3>> sources_;
        std::vector<ExprPhase> phases_;
        int fold_;
        uint32_t fold_n_;
        uint32_t packing_;
        std::vector<int> owner_;
        std::vector<bool> gathered_, zero_select_;
        std::array<bool, ExpressionProgram::max_rank> needed_{};
        int outer_ = -1, lowest_ = 0;
        std::array<V, ExpressionProgram::max_rank> coords_{};
        V element_ = 0, fold_index_ = 0;
        std::vector<V> values_, accumulators_, compensations_;

        V k(uint32_t value) { return e_.literal(value); }
        V word(uint32_t offset) { return e_.argument(k(offset)); }
        V add(V a, V b) { return e_.math(ExprOp::AddInt, a, b); }
        V sub(V a, V b) { return e_.math(ExprOp::SubInt, a, b); }
        V mul(V a, V b) { return e_.math(ExprOp::MulInt, a, b); }
        V less(V a, V b) { return e_.math(ExprOp::LessUInt, a, b); }
        V equal(V a, V b) { return e_.math(ExprOp::EqualInt, a, b); }
        V bit_and(V a, V b) { return e_.math(ExprOp::BitAnd, a, b); }
        V bit_or(V a, V b) { return e_.math(ExprOp::BitOr, a, b); }
        V shl(V a, V b) { return e_.math(ExprOp::ShiftLeft, a, b); }

        void claim(uint32_t pc, uint32_t node, const std::vector<uint32_t>& uses) {
            const auto kind = opcode(instructions_[node]);
            if (uses[node] != 1 || owner_[node] >= 0 || kind == ExprOp::LogicalAnd ||
                kind == ExprOp::LogicalOr || kind == ExprOp::Select || kind == ExprOp::Fold ||
                phases_[node] != phases_[pc])
                return;
            owner_[node] = int(pc);
            for (uint32_t a = 0; a < expr_arity(kind); ++a)
                claim(pc, sources_[node][a], uses);
        }

        void initialize_analysis() {
            std::vector<uint32_t> uses(instructions_.size());
            for (uint32_t pc = 0; pc < instructions_.size(); ++pc)
                for (uint32_t a = 0; a < expr_arity(opcode(instructions_[pc])); ++a)
                    ++uses[sources_[pc][a]];
            owner_.assign(instructions_.size(), -1);
            gathered_.resize(instructions_.size());
            zero_select_.resize(instructions_.size());
            for (uint32_t pc = 0; pc < instructions_.size(); ++pc) {
                gathered_[pc] = gather_rank(opcode(instructions_[pc])) != 0;
                for (uint32_t a = 0; a < expr_arity(opcode(instructions_[pc])); ++a)
                    gathered_[pc] = gathered_[pc] || gathered_[sources_[pc][a]];
            }
            for (uint32_t pc = 0; pc < instructions_.size(); ++pc) {
                const auto op = opcode(instructions_[pc]);
                if (op != ExprOp::LogicalAnd && op != ExprOp::LogicalOr && op != ExprOp::Select)
                    continue;
                zero_select_[pc] = op == ExprOp::Select && gathered_[sources_[pc][0]] &&
                                   opcode(instructions_[sources_[pc][1]]) == ExprOp::Load &&
                                   opcode(instructions_[sources_[pc][2]]) == ExprOp::Immediate &&
                                   instructions_[sources_[pc][2]].immediate == 0;
                claim(pc, sources_[pc][zero_select_[pc] ? 0 : 1], uses);
                if (op == ExprOp::Select)
                    claim(pc, sources_[pc][2], uses);
            }
            for (const auto& ins : instructions_) {
                if (opcode(ins) == ExprOp::Iota && int(aux_of(ins)) != fold_)
                    needed_[aux_of(ins)] = true;
                if (opcode(ins) == ExprOp::Load)
                    need(signature_.input[aux_of(ins)]);
            }
            for (uint32_t o = 0; o < signature_.outputs; ++o)
                need(signature_.output[o]);
            for (uint32_t d = 0; d < signature_.rank && outer_ < 0; ++d)
                if (int(d) != fold_)
                    outer_ = int(d);
            lowest_ = int(signature_.rank);
            for (uint32_t d = 0; d < signature_.rank && lowest_ == int(signature_.rank); ++d)
                if (needed_[d])
                    lowest_ = int(d);
        }

        void need(const ExpressionSignature::Binding& binding) {
            if (!binding.linear)
                for (uint32_t d = 0; d < signature_.rank; ++d)
                    needed_[d] = needed_[d] || (int(d) != fold_ && binding.strides[d] != ExprStride::Zero);
        }

        V coord(uint32_t d) const { return int(d) == fold_ ? fold_index_ : coords_[d]; }

        V strided(const ExpressionSignature::Binding& binding, uint32_t d, V index, uint32_t stride) {
            if (binding.strides[d] == ExprStride::Unit)
                return index;
            return binding.strides[d] == ExprStride::Strided ? mul(index, word(stride)) : k(0);
        }

        V view_index(const ExpressionSignature::Binding& binding,
                     const std::array<uint32_t, ExpressionProgram::max_rank>& strides) {
            if (binding.linear)
                return element_;
            V index = k(0);
            for (uint32_t d = 0; d < signature_.rank; ++d)
                if (binding.strides[d] != ExprStride::Zero)
                    index = add(index, strided(binding, d, coord(d), strides[d]));
            return index;
        }

        V read(uint32_t i, V index) {
            const auto& binding = signature_.input[i];
            if (!binding.host)
                return e_.load(layout_.input[i], index, binding.dtype);
            const auto start = layout_.input[i] == ExpressionLayout::none ? k(layout_.bank) : add(k(layout_.bank), word(layout_.input[i]));
            return e_.argument(add(start, index));
        }

        V truth(V value) { return e_.math(ExprOp::NotEqualInt, value, k(0)); }

        V eval(uint32_t pc) {
            if (values_[pc])
                return values_[pc];
            const auto ins = instructions_[pc];
            const auto op = opcode(ins);
            const auto aux = aux_of(ins);
            V result = 0;
            if (op == ExprOp::Immediate)
                result = k(ins.immediate);
            else if (op == ExprOp::Iota)
                result = coord(aux);
            else if (op == ExprOp::Extent)
                result = int(aux) == fold_ && fold_n_ ? k(fold_n_) : word(layout_.dims[aux]);
            else if (op == ExprOp::Load) {
                const auto& binding = signature_.input[aux];
                const auto index = view_index(binding, layout_.input_stride[aux]);
                int low = binding.pair >= 0 ? int(aux) : -1;
                for (uint32_t i = 0; i < signature_.inputs && low < 0; ++i)
                    if (signature_.input[i].pair == int(aux))
                        low = int(i);
                result = low >= 0 && e_.cuda() ? e_.load_pair(layout_.input[low], index)[low == int(aux) ? 0 : 1]
                                               : read(aux, index);
            } else if (const auto rank = gather_rank(op)) {
                const auto& binding = signature_.input[aux];
                const auto policy = ExprOob(policy_of(ins));
                V index = k(0), valid = k(1);
                for (uint32_t d = 0; d < rank; ++d) {
                    auto position = eval(sources_[pc][d]);
                    if (policy != ExprOob::Checked && !signature_.gather_in_range) {
                        const auto n = word(layout_.bound[aux][d]);
                        if (policy == ExprOob::Clamp) {
                            position = e_.math(ExprOp::Select, e_.math(ExprOp::LessInt, position, k(0)), k(0), position);
                            position = e_.math(ExprOp::Select, less(position, n), position, sub(n, k(1)));
                        } else if (policy == ExprOob::Wrap) {
                            position = e_.math(ExprOp::IntMod, position, n);
                            position = e_.math(ExprOp::Select, e_.math(ExprOp::LessInt, position, k(0)), add(position, n), position);
                        } else
                            valid = d == 0 ? less(position, n) : bit_and(valid, less(position, n));
                    }
                    if (binding.strides[d] != ExprStride::Zero)
                        index = add(index, strided(binding, d, position, layout_.input_stride[aux][d]));
                }
                result = policy == ExprOob::Zero
                             ? e_.select(valid, [this, aux, index] { return read(aux, index); }, [this] { return k(0); })
                             : read(aux, index);
            } else if (zero_select_[pc]) {
                const auto value = eval(sources_[pc][1]);
                result = e_.select(value, [this, pc, value] { return e_.math(ExprOp::Select, eval(sources_[pc][0]), value, k(0)); }, [this] { return k(0); });
            } else if ((op == ExprOp::LogicalAnd || op == ExprOp::LogicalOr || op == ExprOp::Select) &&
                       std::find(owner_.begin(), owner_.end(), int(pc)) != owner_.end()) {
                const auto a = eval(sources_[pc][0]);
                result = e_.select(a, [this, pc, op] {
                    if (op == ExprOp::LogicalOr)
                        return k(1);
                    const auto value = eval(sources_[pc][1]);
                    return op == ExprOp::Select ? value : truth(value); }, [this, pc, op] {
                    if (op == ExprOp::LogicalAnd)
                        return k(0);
                    const auto value = eval(sources_[pc][op == ExprOp::Select ? 2 : 1]);
                    return op == ExprOp::Select ? value : truth(value); });
            } else if (op == ExprOp::Fold || op == ExprOp::Store) {
                throw std::logic_error("expression fold or store evaluated as a value");
            } else {
                const auto a = eval(sources_[pc][0]);
                const auto b = expr_arity(op) > 1 ? eval(sources_[pc][1]) : k(0);
                const auto c = expr_arity(op) > 2 ? eval(sources_[pc][2]) : k(0);
                result = e_.math(op, a, b, c);
            }
            return values_[pc] = result;
        }

        void fold_body(V i) {
            fold_index_ = i;
            if (fold_n_)
                for (uint32_t pc = 0; pc < instructions_.size(); ++pc)
                    if (phases_[pc] == ExprPhase::Variant)
                        values_[pc] = 0;
            for (uint32_t pc = 0; pc < instructions_.size(); ++pc) {
                if (opcode(instructions_[pc]) != ExprOp::Fold) {
                    if (phases_[pc] == ExprPhase::Variant && owner_[pc] < 0)
                        eval(pc);
                    continue;
                }
                const auto kind = ExprReduce(aux_of(instructions_[pc]));
                const auto type = DataType(instructions_[pc].immediate);
                const bool floating = type == DataType::Float32;
                auto value = eval(sources_[pc][0]), acc = e_.read(accumulators_[pc]);
                switch (kind) {
                case ExprReduce::Sum:
                    if (floating) {
                        // Compensated summation keeps the low-order bits a serial Float32 sum drops.
                        const auto y = e_.math(ExprOp::Sub, value, e_.read(compensations_[pc]));
                        const auto t = e_.math(ExprOp::Add, acc, y);
                        const auto correction = e_.math(ExprOp::Sub, e_.math(ExprOp::Sub, t, acc), y);
                        e_.assign(compensations_[pc], e_.math(ExprOp::Select, e_.math(ExprOp::IsFinite, t), correction, k(0)));
                        value = t;
                    } else
                        value = add(acc, value);
                    break;
                case ExprReduce::And: value = bit_and(acc, value); break;
                case ExprReduce::Or: value = bit_or(acc, value); break;
                case ExprReduce::Xor: value = e_.math(ExprOp::BitXor, acc, value); break;
                case ExprReduce::Count: value = add(acc, e_.math(ExprOp::NotEqualInt, value, k(0))); break;
                default:
                    if (floating)
                        value = e_.math(kind == ExprReduce::Min ? ExprOp::Min : ExprOp::Max, acc, value);
                    else {
                        const auto pred = e_.math(type == DataType::Int32 ? ExprOp::LessInt : ExprOp::LessUInt, acc, value);
                        value = kind == ExprReduce::Min ? e_.math(ExprOp::Select, pred, acc, value)
                                                        : e_.math(ExprOp::Select, pred, value, acc);
                    }
                }
                e_.assign(accumulators_[pc], value);
            }
        }

        void initialize_fold() {
            for (uint32_t pc = 0; pc < instructions_.size(); ++pc)
                if (phases_[pc] == ExprPhase::Invariant && owner_[pc] < 0 && opcode(instructions_[pc]) != ExprOp::Store)
                    eval(pc);
            accumulators_.resize(instructions_.size());
            compensations_.resize(instructions_.size());
            for (uint32_t pc = 0; pc < instructions_.size(); ++pc) {
                if (opcode(instructions_[pc]) != ExprOp::Fold)
                    continue;
                const auto kind = ExprReduce(aux_of(instructions_[pc]));
                const auto type = DataType(instructions_[pc].immediate);
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
                accumulators_[pc] = e_.variable(k(initial));
                if (kind == ExprReduce::Sum && floating)
                    compensations_[pc] = e_.variable(k(0));
            }
            if (fold_n_)
                for (uint32_t i = 0; i < fold_n_; ++i)
                    fold_body(k(i));
            else
                e_.loop(k(0), word(layout_.dims[fold_]), k(1), [this](V i) { fold_body(i); });
            for (uint32_t pc = 0; pc < instructions_.size(); ++pc)
                if (opcode(instructions_[pc]) == ExprOp::Fold)
                    values_[pc] = e_.read(accumulators_[pc]);
        }

        void emit_element(V lane, V element, std::vector<std::array<V, 2>>& packed) {
            element_ = element;
            fold_index_ = 0;
            coords_.fill(0);
            V rest = element;
            for (int d = int(signature_.rank) - 1; d >= lowest_; --d) {
                if (d == fold_)
                    continue;
                if (d == outer_) {
                    coords_[d] = rest;
                    break;
                }
                // Exact for dividends below 2^31: q = (mulhi(n, magic) + n) >> shift.
                const auto quotient = e_.math(ExprOp::ShiftRight, add(e_.mul_hi(rest, word(layout_.magic[d])), rest),
                                              word(layout_.shift[d]));
                if (needed_[d])
                    coords_[d] = sub(rest, mul(quotient, word(layout_.dims[d])));
                rest = quotient;
            }
            values_.assign(instructions_.size(), 0);
            if (fold_ >= 0)
                initialize_fold();
            std::array<V, ExpressionProgram::max_outputs> output_values{};
            for (uint32_t pc = 0; pc < instructions_.size(); ++pc) {
                if (opcode(instructions_[pc]) != ExprOp::Store) {
                    if (opcode(instructions_[pc]) != ExprOp::Fold &&
                        phases_[pc] != ExprPhase::Variant && owner_[pc] < 0)
                        eval(pc);
                    continue;
                }
                output_values[aux_of(instructions_[pc])] = eval(sources_[pc][0]);
            }
            // Positional aliases read the original inputs before any output is written.
            for (uint32_t o = 0; o < signature_.outputs; ++o) {
                const auto& binding = signature_.output[o];
                auto value = output_values[o];
                if (dtype_size(binding.dtype) == 4 || e_.cuda())
                    e_.store(layout_.output[o], view_index(binding, layout_.output_stride[o]), value, binding.dtype);
                else {
                    const uint32_t width = dtype_size(binding.dtype) * 8, lanes = 32 / width;
                    value = binding.dtype == DataType::Float16 ? e_.half(value, true) : bit_and(value, k(255));
                    const auto width_value = k(width);
                    const auto lane_in_word = e_.math(ExprOp::UIntMod, lane, k(lanes));
                    const auto shift = mul(lane_in_word, width_value);
                    const auto shifted = shl(value, shift);
                    for (uint32_t group = 0; group < packing_ / lanes; ++group) {
                        const auto group_index = k(group);
                        const auto word_in_group = e_.math(ExprOp::UIntDiv, lane, k(lanes));
                        const auto matches_group = equal(word_in_group, group_index);
                        e_.condition(matches_group, [this, &packed, o, group, shifted] {
                            const auto previous = e_.read(packed[o][group]);
                            const auto combined = bit_or(previous, shifted);
                            e_.assign(packed[o][group], combined);
                        });
                    }
                }
            }
        }

        void emit_lane(V thread, V lane, std::vector<std::array<V, 2>>& packed, V count) {
            const V element = add(mul(thread, k(packing_)), lane);
            e_.condition(less(element, count), [this, lane, element, &packed] {
                emit_element(lane, element, packed);
            });
        }

        void emit_thread(V thread, V count) {
            std::vector<std::array<V, 2>> packed(signature_.outputs);
            for (uint32_t o = 0; o < signature_.outputs; ++o)
                for (uint32_t group = 0; group < packing_ * dtype_size(signature_.output[o].dtype) / 4; ++group)
                    if (dtype_size(signature_.output[o].dtype) < 4)
                        packed[o][group] = e_.variable(k(0));
            if (packing_ > 1)
                e_.loop(k(0), k(packing_), k(1), [this, thread, count, &packed](V lane) {
                    emit_lane(thread, lane, packed, count);
                });
            else
                emit_lane(thread, k(0), packed, count);
            if (!e_.cuda())
                for (uint32_t o = 0; o < signature_.outputs; ++o) {
                    const uint32_t width = dtype_size(signature_.output[o].dtype) * 8, lanes = 32 / width;
                    if (width == 32)
                        continue;
                    for (uint32_t group = 0; group < packing_ / lanes; ++group) {
                        const auto begin = add(mul(thread, k(packing_)), k(group * lanes));
                        const auto packed_value = e_.read(packed[o][group]);
                        write_packed(o, begin, packed_value, count);
                    }
                }
        }

        void write_packed(uint32_t o, V begin, V packed, V count) {
            const uint32_t width = dtype_size(signature_.output[o].dtype) * 8, lanes = 32 / width;
            e_.condition(less(begin, count), [this, o, begin, packed, count, lanes, width] {
                const auto word_index = e_.math(ExprOp::UIntDiv, begin, k(lanes));
                const auto valid = sub(count, begin);
                const auto value = e_.select(less(valid, k(lanes)), [this, valid, packed, o, word_index, width] {
                    const auto existing = e_.output(layout_.output[o], word_index);
                    const auto sub_one = k(1);
                    const auto width_value = k(width);
                    const auto scaled_valid = mul(valid, width_value);
                    const auto shift_one = k(1);
                    const auto mask_bits = shl(shift_one, scaled_valid);
                    const auto mask = sub(mask_bits, sub_one);
                    const auto inserted = bit_and(packed, mask);
                    const auto inverse_mask = e_.math(ExprOp::BitNot, mask);
                    const auto preserved = bit_and(existing, inverse_mask);
                    return bit_or(preserved, inserted); }, [packed] { return packed; });
                e_.store(layout_.output[o], word_index, value);
            });
        }

    public:
        ExpressionCodegen(ExpressionEmitter& emitter, const ExpressionProgram& program,
                          const ExpressionSignature& signature)
            : e_(emitter), signature_(signature), layout_(expression_layout(program, signature)), instructions_(program.instructions()), sources_(expression_sources(program)), phases_(expression_phases(program, signature)), fold_(signature.fold_dim), fold_n_(signature.fold_length), packing_(emitter.cuda() ? 1 : expression_packing(signature)) {
            initialize_analysis();
        }

        void emit() {
            const V count = word(layout_.count);
            const V work = packing_ > 1 ? e_.math(ExprOp::UIntDiv, add(count, k(packing_ - 1)), k(packing_)) : count;
            e_.loop(e_.thread(), work, e_.grid_stride(), [this, count](V thread) { emit_thread(thread, count); });
        }
    };

    void emit_expression(ExpressionEmitter& e, const ExpressionProgram& program,
                         const ExpressionSignature& signature) {
        ExpressionCodegen(e, program, signature).emit();
    }
} // namespace lfs::core::internal
