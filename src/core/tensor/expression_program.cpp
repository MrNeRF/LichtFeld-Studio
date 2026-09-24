/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "backend/gpu_backend_ops.hpp"
#include "core/assert.hpp"
#include "internal/expression_runtime.hpp"
#include "internal/tensor_impl.hpp"
#ifdef LFS_TENSOR_VULKAN
#include "backend/vulkan/vk_context.hpp"
#include "backend/vulkan/vk_pipelines.hpp"
#endif

#include <algorithm>
#include <bit>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace lfs::core::internal {

    ExpressionBuilder::Value ExpressionBuilder::emit(const ExprOp op,
                                                     const Value a, const Value b,
                                                     const Value c, const uint8_t aux,
                                                     const uint32_t immediate,
                                                     const ExprOob policy) {
        const Node node{op, {a, b, c}, aux, immediate, policy};
        for (uint32_t i = 0; i < expr_arity(op); ++i)
            LFS_ASSERT_MSG(node.operands[i] < nodes_.size() &&
                               nodes_[node.operands[i]].op != ExprOp::Store,
                           "expression operand is not a defined value");
        nodes_.push_back(node);
        return static_cast<Value>(nodes_.size() - 1);
    }

    ExpressionProgram ExpressionBuilder::build() const {
        std::vector<bool> live(nodes_.size(), false);
        for (size_t end = nodes_.size(); end > 0; --end) {
            const size_t i = end - 1;
            live[i] = live[i] || nodes_[i].op == ExprOp::Store;
            if (live[i])
                for (uint32_t arg = 0; arg < expr_arity(nodes_[i].op); ++arg)
                    live[nodes_[i].operands[arg]] = true;
        }
        std::vector<size_t> last_use(nodes_.size(), 0);
        for (size_t i = 0; i < nodes_.size(); ++i)
            if (live[i])
                for (uint32_t arg = 0; arg < expr_arity(nodes_[i].op); ++arg)
                    last_use[nodes_[i].operands[arg]] = i;
        std::vector<uint8_t> slots(nodes_.size(), 0);
        std::array<bool, ExpressionProgram::max_registers> used{};
        ExpressionProgram result;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            const Node& node = nodes_[i];
            if (!live[i])
                continue;
            std::array<uint8_t, 3> args{};
            for (uint32_t arg = 0; arg < expr_arity(node.op); ++arg) {
                args[arg] = slots[node.operands[arg]];
                if (last_use[node.operands[arg]] == i)
                    used[args[arg]] = false;
            }
            uint8_t dst = 0;
            if (node.op != ExprOp::Store) {
                const auto free = std::find(used.begin(), used.end(), false);
                if (free == used.end())
                    throw std::length_error("expression register pressure exceeds limit");
                dst = static_cast<uint8_t>(free - used.begin());
                used[dst] = true;
                slots[i] = dst;
            }
            result.emit(node.op, dst, args[0], args[1], args[2],
                        node.aux, node.immediate, node.policy);
        }
        result.analyze();
        return result;
    }

    std::vector<uint32_t> ExpressionProgram::compact_inputs() {
        std::array<uint8_t, max_inputs> remap{};
        remap.fill(uint8_t(max_inputs));
        std::vector<uint32_t> retained;
        for (auto& ins : instructions_) {
            const auto op = ExprOp(ins.op_dst_a_b & 255u);
            if (op != ExprOp::Load && op != ExprOp::Gather && op != ExprOp::Gather2 && op != ExprOp::Gather3)
                continue;
            const uint32_t old = (ins.c_aux_policy >> 8) & 255u;
            if (remap[old] == max_inputs) {
                remap[old] = uint8_t(retained.size());
                retained.push_back(old);
            }
            ins.c_aux_policy = (ins.c_aux_policy & ~0xff00u) | (uint32_t(remap[old]) << 8);
        }
        analysis_ = {};
        analyze();
        return retained;
    }

    void ExpressionProgram::emit(const ExprOp op, const uint8_t dst,
                                 const uint8_t a, const uint8_t b,
                                 const uint8_t c, const uint8_t aux,
                                 const uint32_t immediate, const ExprOob policy) {
        if (instructions_.size() >= max_instructions)
            throw std::length_error("expression instruction limit exceeded");
        LFS_ASSERT_MSG(dst < max_registers && a < max_registers &&
                           b < max_registers && c < max_registers,
                       "expression register limit exceeded");
        instructions_.push_back({
            expr_pack(op, dst, a, b),
            uint32_t(c) | (uint32_t(aux) << 8) |
                (uint32_t(policy) << 16),
            immediate,
        });
    }

    void ExpressionProgram::analyze() {
        const auto require = [](bool valid, const char* message) {
            if (!valid)
                throw std::invalid_argument(message);
        };
        auto& facts = analysis_;
        std::array<uint32_t, max_registers> definitions{};
        for (uint32_t pc = 0; pc < instructions_.size(); ++pc) {
            const auto ins = instructions_[pc];
            const auto op = ExprOp(ins.op_dst_a_b & 255u);
            const uint32_t aux = (ins.c_aux_policy >> 8) & 255u;
            const uint32_t policy = (ins.c_aux_policy >> 16) & 255u;
            require(uint32_t(op) <= uint32_t(ExprOp::CastUnsignedFloat), "invalid expression opcode");
            const uint32_t registers[]{(ins.op_dst_a_b >> 16) & 255u, ins.op_dst_a_b >> 24, ins.c_aux_policy & 255u};
            std::array<uint32_t, 3> sources{};
            for (uint32_t a = 0; a < expr_arity(op); ++a)
                sources[a] = definitions[registers[a]];
            facts.sources.push_back(sources);
            if (op != ExprOp::Store)
                definitions[(ins.op_dst_a_b >> 8) & 255u] = pc;
            const uint32_t gathered = op == ExprOp::Gather ? 1 : op == ExprOp::Gather2 ? 2
                                                             : op == ExprOp::Gather3   ? 3
                                                                                       : 0;
            if (op == ExprOp::Load || gathered) {
                require(aux < max_inputs, "expression input out of range");
                facts.inputs = std::max(facts.inputs, aux + 1);
                facts.loaded[aux] = facts.loaded[aux] || op == ExprOp::Load;
                facts.arity[aux] = std::max(facts.arity[aux], gathered);
            }
            if (gathered) {
                require(policy <= uint32_t(ExprOob::Checked), "invalid expression gather policy");
                facts.runtime_gathers = facts.runtime_gathers || policy != uint32_t(ExprOob::Checked);
                for (uint32_t d = 0; d < gathered && policy == uint32_t(ExprOob::Checked); ++d) {
                    const auto index = instructions_[sources[d]];
                    require((index.op_dst_a_b & 255u) == uint32_t(ExprOp::Immediate) && index.immediate < uint32_t(INT32_MAX),
                            "checked expression gathers take nonnegative immediate indices below int32 max");
                    facts.checked_bounds[aux][d] = std::max(facts.checked_bounds[aux][d], index.immediate + 1);
                }
            }
            if (op == ExprOp::Store) {
                require(aux < max_outputs && !(facts.stores & (1u << aux)), "expression output must have one store");
                facts.stores |= 1u << aux;
            }
            if (op == ExprOp::Iota || op == ExprOp::Extent) {
                require(aux < max_rank, "expression dimension out of range");
                facts.rank = std::max(facts.rank, aux + 1);
            }
            if (op == ExprOp::Fold) {
                const auto type = DataType(ins.immediate);
                require(aux <= uint32_t(ExprReduce::Count) && policy < max_rank && (facts.fold < 0 || int(policy) == facts.fold) &&
                            (type == DataType::Float32 || type == DataType::Int32 || type == DataType::UInt32 ||
                             (type == DataType::Bool && aux >= uint32_t(ExprReduce::And))),
                        "expression folds share one dimension and a valid accumulator");
                facts.fold = int(policy);
                facts.rank = std::max(facts.rank, policy + 1);
            }
        }
    }

    namespace {
        void require(const bool condition, const char* message) {
            if (!condition)
                throw std::invalid_argument(message);
        }

        uint64_t device_address(const StorageRef& storage) {
            if (storage.backend == GpuBackend::CUDA)
                return reinterpret_cast<uintptr_t>(storage.data) + storage.byte_offset;
            require(storage.meta != nullptr && storage.meta->gpu_descriptor.base_address != 0,
                    "expression storage has no device address");
            return storage.meta->gpu_descriptor.base_address + storage.byte_offset;
        }

        ExprStride stride_class(const int32_t stride) {
            return stride == 0 ? ExprStride::Zero : stride == 1 ? ExprStride::Unit
                                                                : ExprStride::Strided;
        }

        // Contiguous over the domain without the fold dimension and constant along it.
        bool linear(const std::array<int32_t, ExpressionProgram::max_rank>& strides,
                    const ExprShape& shape, const int fold) {
            int64_t running = 1;
            for (int d = int(shape.rank) - 1; d >= 0; --d) {
                if (d == fold) {
                    if (strides[d] != 0)
                        return false;
                    continue;
                }
                if (shape.dims[d] > 1 && strides[d] != running)
                    return false;
                running *= shape.dims[d];
            }
            return true;
        }

        struct Extent {
            uint64_t begin = 0, end = 0;
            [[nodiscard]] bool overlaps(const Extent& other) const { return begin < other.end && other.begin < end; }
        };

        // The element sets of two views with equal strides are disjoint when their offset is
        // not a multiple of the strides' common divisor.
        bool interleaved(const uint64_t a, const uint64_t b,
                         const std::array<int32_t, ExpressionProgram::max_rank>& strides,
                         const ExprShape& shape, const int fold) {
            uint64_t divisor = 0;
            for (uint32_t d = 0; d < shape.rank; ++d)
                if (int(d) != fold && shape.dims[d] > 1)
                    divisor = std::gcd(divisor, uint64_t(strides[d]));
            const uint64_t delta = a > b ? a - b : b - a;
            return divisor != 0 && delta % 4 == 0 && (delta / 4) % divisor != 0;
        }

        ExpressionLaunch describe(const ExpressionProgram& program,
                                  const std::span<const ExprInput> inputs,
                                  const std::span<const ExprOutput> outputs,
                                  const ExprShape& shape) {
            constexpr auto rank_limit = ExpressionProgram::max_rank;
            const auto instructions = program.instructions();
            require(!instructions.empty() && inputs.size() <= ExpressionProgram::max_inputs &&
                        !outputs.empty() && outputs.size() <= ExpressionProgram::max_outputs &&
                        shape.rank > 0 && shape.rank <= rank_limit,
                    "invalid expression program size");
            const auto& analysis = program.analysis();
            require(inputs.size() >= analysis.inputs && shape.rank >= analysis.rank &&
                        analysis.stores == (1u << outputs.size()) - 1,
                    "expression bindings or domain do not match its program");
            const auto& loaded = analysis.loaded;
            const auto& arity = analysis.arity;
            const int fold = analysis.fold;

            ExpressionLaunch launch;
            launch.program = &program;
            auto& signature = launch.signature;
            signature.rank = shape.rank;
            signature.fold_dim = fold;
            if (fold >= 0 && shape.dims[fold] > 0 && shape.dims[fold] <= 64)
                signature.fold_length = shape.dims[fold];
            signature.inputs = uint32_t(inputs.size());
            signature.outputs = uint32_t(outputs.size());
            uint64_t count = 1, domain_count = 1;
            for (uint32_t d = 0; d < shape.rank; ++d) {
                require(shape.dims[d] <= uint32_t(INT32_MAX), "expression dimension exceeds int32");
                domain_count *= shape.dims[d];
                require(domain_count <= uint32_t(INT32_MAX), "expression domain product exceeds int32");
                if (int(d) == fold) {
                    require(shape.dims[d] <= ExpressionProgram::max_fold, "expression fold exceeds its length limit");
                    continue;
                }
                require(shape.dims[d] > 0, "expression dimensions must be nonzero");
                count *= shape.dims[d];
                require(count <= std::numeric_limits<int32_t>::max(), "expression output count exceeds int32");
            }
            launch.count = uint32_t(count);
            const auto backend = outputs[0].storage.backend;
            const auto supported = [](const DataType dtype) {
                return dtype == DataType::Float32 || dtype == DataType::Float16 || dtype == DataType::Int32 ||
                       dtype == DataType::UInt32 || dtype == DataType::UInt8 || dtype == DataType::Bool;
            };

            std::array<std::array<int32_t, rank_limit>, ExpressionProgram::max_inputs> strides{};
            std::array<Extent, ExpressionProgram::max_inputs> read_extents{};
            uint32_t bank_words = 0;
            for (uint32_t i = 0; i < inputs.size(); ++i) {
                const auto& input = inputs[i];
                auto& binding = signature.input[i];
                binding.dtype = input.storage.dtype;
                binding.host = input.storage.flags & STORAGE_REF_HOST_MEMORY;
                require(supported(binding.dtype) && (binding.host ? dtype_size(binding.dtype) == 4 : input.storage.backend == backend),
                        "expression input backend or dtype invalid");
                require(input.storage.data && input.storage.byte_offset <= INT32_MAX &&
                            input.storage.byte_offset % dtype_size(binding.dtype) == 0,
                        "expression input address or byte offset invalid");
                const uint32_t used = std::max(loaded[i] ? shape.rank : 0u, arity[i]);
                uint64_t elements = 1, last = 0, source_last = 0;
                int64_t running = 1;
                for (int d = int(rank_limit) - 1; d >= 0; --d) {
                    require(input.dims[d] > 0 && input.dims[d] <= uint32_t(INT32_MAX) && input.strides[d] >= 0, "expression input dimensions must be nonzero and strides nonnegative");
                    auto stride = binding.host ? (input.dims[d] == 1 ? 0 : int32_t(running)) : input.strides[d];
                    if (binding.host)
                        binding.strides[d] = stride_class(stride);
                    running *= input.dims[d];
                    require(running <= INT32_MAX, "expression input product exceeds int32");
                    if (uint32_t(d) >= used) {
                        require(used == 0 || input.dims[d] == 1, "expression input has dimensions its program does not index");
                        stride = 0;
                    }
                    if (loaded[i] && uint32_t(d) < shape.rank && stride != 0)
                        require(input.dims[d] == shape.dims[d], "expression input does not match the domain");
                    strides[i][d] = stride;
                    elements *= input.dims[d];
                    last += uint64_t(input.dims[d] - 1) * uint64_t(stride);
                    source_last += uint64_t(input.dims[d] - 1) * uint64_t(input.strides[d]);
                    require(last <= INT32_MAX && source_last <= INT32_MAX, "expression input offsets exceed int32");
                }
                require(last <= std::numeric_limits<int32_t>::max(), "expression input offsets exceed int32");
                binding.linear = loaded[i] && linear(strides[i], shape, fold);
                if (!binding.host)
                    for (uint32_t d = 0; d < rank_limit; ++d)
                        binding.strides[d] = binding.linear && arity[i] == 0 ? ExprStride::Zero : stride_class(strides[i][d]);
                if (binding.host) {
                    bank_words += uint32_t(elements);
                    require(bank_words <= ExpressionProgram::max_parameters,
                            "expression host parameters exceed the 64-word launch bank");
                    continue;
                }
                const uint64_t address = device_address(input.storage);
                read_extents[i] = {address, address + (last + 1) * dtype_size(binding.dtype)};
                launch.reads[launch.read_count++] = input.storage;
            }
            for (uint32_t i = 0; i < inputs.size(); ++i)
                for (uint32_t d = 0; d < rank_limit; ++d)
                    require(analysis.checked_bounds[i][d] <= inputs[i].dims[d],
                            "checked expression index out of range");

            std::array<std::array<int32_t, rank_limit>, ExpressionProgram::max_outputs> output_strides{};
            std::array<Extent, ExpressionProgram::max_outputs> write_extents{};
            for (uint32_t o = 0; o < outputs.size(); ++o) {
                const auto& output = outputs[o];
                auto& binding = signature.output[o];
                binding.dtype = output.storage.dtype;
                require(supported(binding.dtype) && output.storage.backend == backend &&
                            !(output.storage.flags & STORAGE_REF_HOST_MEMORY),
                        "expression output backend or dtype invalid");
                require(output.storage.data && output.storage.byte_offset <= INT32_MAX &&
                            output.storage.byte_offset % dtype_size(binding.dtype) == 0,
                        "expression output address or byte offset invalid");
                std::array<uint32_t, rank_limit> order{};
                uint32_t active = 0;
                uint64_t last = 0;
                for (uint32_t d = 0; d < rank_limit; ++d) {
                    const bool domain = d < shape.rank && int(d) != fold;
                    const int32_t stride = domain ? output.strides[d] : 0;
                    require(stride >= 0 && (!domain || shape.dims[d] == 1 || stride > 0),
                            "expression outputs cannot be expanded");
                    output_strides[o][d] = stride;
                    if (domain && shape.dims[d] > 1)
                        order[active++] = d;
                    if (domain)
                        last += uint64_t(shape.dims[d] - 1) * uint64_t(stride);
                }
                std::sort(order.begin(), order.begin() + active, [&](uint32_t a, uint32_t b) { return output_strides[o][a] < output_strides[o][b]; });
                uint64_t span = 1;
                for (uint32_t j = 0; j < active; ++j) {
                    require(uint64_t(output_strides[o][order[j]]) >= span, "expression output elements overlap");
                    span += uint64_t(shape.dims[order[j]] - 1) * uint64_t(output_strides[o][order[j]]);
                }
                require(last <= std::numeric_limits<int32_t>::max(), "expression output offsets exceed int32");
                binding.linear = linear(output_strides[o], shape, fold);
                for (uint32_t d = 0; d < rank_limit; ++d)
                    binding.strides[d] = binding.linear ? ExprStride::Zero : stride_class(output_strides[o][d]);
                const uint64_t address = device_address(output.storage);
                const bool packed = dtype_size(binding.dtype) < 4;
                require(!packed || (binding.linear && address % 4 == 0),
                        "expression packed output must be word aligned and contiguous");
                uint64_t end = address + (last + 1) * dtype_size(binding.dtype);
                if (packed) {
                    end = (end + 3) & ~uint64_t(3);
                    const auto* meta = output.storage.meta;
                    require(meta && output.storage.byte_offset <= meta->gpu_descriptor.byte_size &&
                                end - address <= meta->gpu_descriptor.byte_size - output.storage.byte_offset,
                            "expression packed output requires capacity for its full tail word");
                }
                write_extents[o] = {address, end};
                for (uint32_t other = 0; other < o; ++other) {
                    const bool same = output_strides[o] == output_strides[other] && !packed &&
                                      dtype_size(signature.output[other].dtype) == 4;
                    require(!write_extents[o].overlaps(write_extents[other]) ||
                                (same && interleaved(address, write_extents[other].begin, output_strides[o], shape, fold)),
                            "expression outputs overlap");
                }
                launch.writes[launch.write_count++] = output.storage;
            }
            for (uint32_t i = 0; i < inputs.size(); ++i) {
                if (signature.input[i].host)
                    continue;
                for (uint32_t o = 0; o < outputs.size(); ++o) {
                    if (!read_extents[i].overlaps(write_extents[o]))
                        continue;
                    bool positional = loaded[i] && arity[i] == 0 && (fold < 0 || strides[i][fold] == 0);
                    for (uint32_t d = 0; d < shape.rank; ++d)
                        positional = positional && (int(d) == fold || shape.dims[d] == 1 || strides[i][d] == output_strides[o][d]);
                    const bool identical = read_extents[i].begin == write_extents[o].begin &&
                                           dtype_size(signature.input[i].dtype) == dtype_size(signature.output[o].dtype);
                    require(positional && (identical || (dtype_size(signature.input[i].dtype) == 4 &&
                                                         dtype_size(signature.output[o].dtype) == 4 &&
                                                         interleaved(read_extents[i].begin, write_extents[o].begin, output_strides[o], shape, fold))),
                            "expression input aliases an output at other elements");
                }
            }
            if (backend == GpuBackend::CUDA) {
                std::array<bool, ExpressionProgram::max_inputs> paired{};
                for (uint32_t i = 0; i < inputs.size(); ++i)
                    for (uint32_t j = 0; j < inputs.size() && !paired[i]; ++j) {
                        const auto& low = signature.input[i];
                        const auto& high = signature.input[j];
                        if (i == j || paired[j] || !loaded[i] || !loaded[j] || low.host || high.host || low.linear ||
                            low.dtype != high.dtype || dtype_size(low.dtype) != 4 ||
                            inputs[i].storage.data != inputs[j].storage.data || strides[i] != strides[j] ||
                            read_extents[i].begin % 8 != 0 || read_extents[j].begin != read_extents[i].begin + 4 ||
                            std::any_of(strides[i].begin(), strides[i].end(), [](int32_t s) { return s % 2 != 0; }))
                            continue;
                        signature.input[i].pair = int8_t(j);
                        paired[i] = paired[j] = true;
                    }
            }

            std::array<std::array<uint32_t, rank_limit>, ExpressionProgram::max_inputs> gather_bounds{};
            for (uint32_t i = 0; i < inputs.size(); ++i)
                gather_bounds[i] = inputs[i].dims;
            signature.gather_in_range = expression_gather_in_range(program, shape, gather_bounds);

            const auto layout = expression_layout(program, signature);
            auto& words = launch.arguments;
            launch.words = layout.words;
            const auto address = [&](uint32_t word, uint64_t value) {
                words[word] = uint32_t(value);
                words[word + 1] = uint32_t(value >> 32);
            };
            for (uint32_t i = 0; i < inputs.size(); ++i)
                if (!signature.input[i].host)
                    address(layout.input[i], read_extents[i].begin);
            for (uint32_t o = 0; o < outputs.size(); ++o)
                address(layout.output[o], write_extents[o].begin);
            words[layout.count] = launch.count;
            for (uint32_t d = 0; d < shape.rank; ++d) {
                words[layout.dims[d]] = shape.dims[d];
                if (layout.magic[d] == ExpressionLayout::none)
                    continue;
                const uint64_t divisor = shape.dims[d];
                const uint32_t shift = divisor == 1 ? 0 : 32 - std::countl_zero(uint32_t(divisor - 1));
                words[layout.magic[d]] = uint32_t(((uint64_t(1) << 32) * ((uint64_t(1) << shift) - divisor)) / divisor + 1);
                words[layout.shift[d]] = shift;
            }
            uint32_t bank = 0;
            for (uint32_t i = 0; i < inputs.size(); ++i) {
                for (uint32_t d = 0; d < rank_limit; ++d) {
                    if (layout.input_stride[i][d] != ExpressionLayout::none)
                        words[layout.input_stride[i][d]] = uint32_t(strides[i][d]);
                    if (layout.bound[i][d] != ExpressionLayout::none)
                        words[layout.bound[i][d]] = inputs[i].dims[d];
                }
                if (!signature.input[i].host)
                    continue;
                if (layout.input[i] != ExpressionLayout::none)
                    words[layout.input[i]] = bank;
                const auto& dims = inputs[i].dims;
                const auto* source = static_cast<const uint32_t*>(inputs[i].storage.data) + inputs[i].storage.byte_offset / 4;
                for (uint32_t a = 0; a < dims[0]; ++a)
                    for (uint32_t b = 0; b < dims[1]; ++b)
                        for (uint32_t c = 0; c < dims[2]; ++c)
                            for (uint32_t e = 0; e < dims[3]; ++e)
                                words[layout.bank + bank++] = source[a * inputs[i].strides[0] + b * inputs[i].strides[1] +
                                                                     c * inputs[i].strides[2] + e * inputs[i].strides[3]];
            }
            for (uint32_t o = 0; o < outputs.size(); ++o)
                for (uint32_t d = 0; d < rank_limit; ++d)
                    if (layout.output_stride[o][d] != ExpressionLayout::none)
                        words[layout.output_stride[o][d]] = uint32_t(output_strides[o][d]);
            return launch;
        }
    } // namespace

    void run_expression(const ExpressionProgram& program,
                        const std::span<const ExprInput> inputs,
                        const std::span<const ExprOutput> outputs,
                        const ExprShape& shape, const ExecContext context) {
        const auto launch = describe(program, inputs, outputs, shape);
        if (launch.signature.fold_dim >= 0)
            (void)expression_phases(program, launch.signature);
        backend_ops(outputs[0].storage.backend).compiled_expression(launch, context);
    }

    void prepare_expression(const ExpressionProgram& program,
                            const std::span<const ExprInput> inputs,
                            const std::span<const ExprOutput> outputs,
                            const ExprShape& shape) {
        auto launch = describe(program, inputs, outputs, shape);
        if (launch.signature.fold_dim >= 0)
            (void)expression_phases(program, launch.signature);
        launch.prepare_only = true;
        backend_ops(outputs[0].storage.backend).compiled_expression(launch, {});
    }

    ExpressionCacheStats expression_cache_stats(const GpuBackend backend) {
        if (backend == GpuBackend::CUDA)
#if LFS_HAS_CUDA
            return cuda_expression_cache().stats();
#else
            throw std::runtime_error("CUDA expression backend is unavailable");
#endif
#ifdef LFS_TENSOR_VULKAN
        return acquire_vulkan_context()->pipelines().expressions().stats();
#else
        throw std::runtime_error("Vulkan expression backend is unavailable");
#endif
    }

} // namespace lfs::core::internal
