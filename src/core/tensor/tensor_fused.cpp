/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_fused.hpp"
#include "internal/expression_program.hpp"
#include "internal/tensor_impl.hpp"

#include <bit>
#include <cmath>
#include <format>
#include <limits>
#include <map>
#include <tuple>

namespace lfs::core::fused {
    using internal::ExpressionProgram;
    using internal::ExprOob;
    using internal::ExprOp;

    namespace {
        [[noreturn]] void fail(const std::string& message) { throw TensorError("fused kernel: " + message); }

        bool integral(const DataType dtype) { return dtype == DataType::Int32 || dtype == DataType::UInt32; }

        DataType value_type(const DataType dtype) {
            switch (dtype) {
            case DataType::Float32:
            case DataType::Float16: return DataType::Float32;
            case DataType::Int32: return DataType::Int32;
            case DataType::UInt32:
            case DataType::UInt8: return DataType::UInt32;
            case DataType::Bool: return DataType::Bool;
            default: fail(std::format("unsupported input dtype {}", dtype_name(dtype)));
            }
        }
    } // namespace

    struct Builder::State {
        size_t rank;
        internal::ExpressionBuilder ir;
        std::map<std::tuple<ExprOp, uint32_t, uint32_t, uint32_t, uint8_t, uint32_t, ExprOob>, uint32_t> nodes;
        struct Declared {
            DataType dtype;
            size_t rank;
        };
        std::vector<Declared> inputs;
        // A program input: the gather view of an input, or a load mapping its dimensions
        // onto domain dimensions.
        struct Binding {
            uint32_t input;
            bool gather;
            std::array<int8_t, ExpressionProgram::max_rank> dims;
            bool operator==(const Binding&) const = default;
        };
        std::vector<Binding> bindings;
        std::vector<DataType> outputs;
        int fold = -1;
    };

    struct Kernel::Compiled {
        ExpressionProgram program;
        size_t rank;
        int fold;
        std::vector<Builder::State::Declared> inputs;
        std::vector<Builder::State::Binding> bindings;
        std::vector<DataType> outputs;
    };

    struct Nodes {
        static Builder::State& state(Builder* builder) { return *builder->state_; }

        static Expr make(Builder* builder, const ExprOp op, const DataType dtype,
                         std::array<uint32_t, 3> operands = {}, const uint8_t aux = 0,
                         const uint32_t immediate = 0, const ExprOob policy = ExprOob::Zero) {
            auto& s = state(builder);
            for (uint32_t a = internal::expr_arity(op); a < 3; ++a)
                operands[a] = 0;
            const auto key = std::tuple{op, operands[0], operands[1], operands[2], aux, immediate, policy};
            if (const auto found = s.nodes.find(key); found != s.nodes.end())
                return Expr(builder, found->second, dtype);
            const auto id = s.ir.emit(op, operands[0], operands[1], operands[2], aux, immediate, policy);
            s.nodes.emplace(key, id);
            return Expr(builder, id, dtype);
        }

        static Expr constant(Builder* builder, const DataType dtype, const uint32_t bits) {
            return make(builder, ExprOp::Immediate, dtype, {}, 0, bits);
        }

        static Expr scalar(Builder* builder, const DataType dtype, const double value) {
            const auto whole = [&](const double low, const double high) {
                if (value != std::trunc(value) || value < low || value > high)
                    fail(std::format("constant {} does not fit {}", value, dtype_name(dtype)));
            };
            switch (dtype) {
            case DataType::Float32: return constant(builder, dtype, std::bit_cast<uint32_t>(float(value)));
            case DataType::Int32: whole(std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()); return constant(builder, dtype, uint32_t(int32_t(value)));
            case DataType::UInt32: whole(0, std::numeric_limits<uint32_t>::max()); return constant(builder, dtype, uint32_t(value));
            default: whole(0, 1); return constant(builder, dtype, uint32_t(value));
            }
        }

        // Converts scalar operands to the type of the expression operands, which must agree.
        static std::vector<Expr> unify(std::initializer_list<const Operand*> operands, const char* operation) {
            const Expr* reference = nullptr;
            for (const auto* operand : operands)
                if (operand->expr_) {
                    if (reference && (operand->expr_->builder_ != reference->builder_ || operand->expr_->dtype_ != reference->dtype_))
                        fail(std::format("{} needs operands of one kernel and type, got {} and {}", operation,
                                         dtype_name(reference->dtype_), dtype_name(operand->expr_->dtype_)));
                    reference = reference ? reference : &*operand->expr_;
                }
            if (!reference)
                fail(std::format("{} needs an expression operand", operation));
            std::vector<Expr> result;
            for (const auto* operand : operands)
                result.push_back(operand->expr_ ? *operand->expr_ : scalar(reference->builder_, reference->dtype_, operand->scalar_));
            return result;
        }

        static Expr op(const ExprOp code, const DataType dtype, std::initializer_list<Expr> operands) {
            std::array<uint32_t, 3> ids{};
            size_t i = 0;
            for (const auto& operand : operands)
                ids[i++] = operand.id_;
            return make(operands.begin()->builder_, code, dtype, ids);
        }

        static Expr binary(const Operand& a, const Operand& b, const char* name,
                           const ExprOp floating, const ExprOp signed_op, const ExprOp unsigned_op, const bool predicate) {
            const auto values = unify({&a, &b}, name);
            const auto type = values[0].dtype_;
            if (type == DataType::Bool)
                fail(std::format("{} does not accept bool", name));
            const auto code = type == DataType::Float32 ? floating : type == DataType::Int32 ? signed_op
                                                                                             : unsigned_op;
            return op(code, predicate ? DataType::Bool : type, {values[0], values[1]});
        }

        static Expr bitwise(const Operand& a, const Operand& b, const char* name, const ExprOp code, const bool shift) {
            const auto values = unify({&a, &b}, name);
            const auto type = values[0].dtype_;
            if (!integral(type) && (shift || type != DataType::Bool))
                fail(std::format("{} does not accept {}", name, dtype_name(type)));
            return op(code, type, {values[0], values[1]});
        }

        static Expr floating(const Expr& a, const ExprOp code, const DataType result = DataType::Float32) {
            if (a.dtype_ != DataType::Float32)
                fail(std::format("math functions take Float32, got {}", dtype_name(a.dtype_)));
            return op(code, result, {a});
        }

        static Expr floating(std::initializer_list<const Operand*> operands, const ExprOp code, const char* name) {
            const auto values = unify(operands, name);
            if (values[0].dtype_ != DataType::Float32)
                fail(std::format("{} takes Float32, got {}", name, dtype_name(values[0].dtype_)));
            std::array<uint32_t, 3> ids{};
            for (size_t i = 0; i < values.size(); ++i)
                ids[i] = values[i].id_;
            return make(values[0].builder_, code, DataType::Float32, ids);
        }

        static Expr choose(const Expr& condition, const Expr& yes, const Expr& no) {
            return make(yes.builder_, ExprOp::Select, yes.dtype_, {condition.id_, yes.id_, no.id_});
        }

        static Expr select(const Operand& condition, const Operand& yes, const Operand& no) {
            if (!yes.expr_ && !no.expr_) {
                if (!condition.expr_ || condition.expr_->dtype_ != DataType::Bool)
                    fail("where needs a Bool condition expression");
                auto* builder = condition.expr_->builder_;
                const auto type = yes.scalar_type_ == DataType::Float32 || no.scalar_type_ == DataType::Float32
                                      ? DataType::Float32
                                      : yes.scalar_type_;
                return choose(*condition.expr_, scalar(builder, type, yes.scalar_), scalar(builder, type, no.scalar_));
            }
            const auto values = unify({&yes, &no}, "where");
            return choose(lift(condition, values[0], DataType::Bool), values[0], values[1]);
        }

        static Expr retype(const Expr& a, const DataType dtype) { return Expr(a.builder_, a.id_, dtype); }

        static Expr lift(const Operand& operand, const Expr& like, const DataType dtype) {
            if (!operand.expr_)
                return scalar(like.builder_, dtype, operand.scalar_);
            if (operand.expr_->builder_ != like.builder_ || operand.expr_->dtype_ != dtype)
                fail(std::format("expected a {} expression of the same kernel", dtype_name(dtype)));
            return *operand.expr_;
        }

        static uint32_t binding(Builder* builder, const Builder::State::Binding& binding) {
            auto& s = state(builder);
            for (uint32_t i = 0; i < s.bindings.size(); ++i)
                if (s.bindings[i] == binding)
                    return i;
            if (s.bindings.size() == ExpressionProgram::max_inputs)
                fail(std::format("a kernel reads at most {} input views", ExpressionProgram::max_inputs));
            s.bindings.push_back(binding);
            return uint32_t(s.bindings.size() - 1);
        }

        static Expr gather(Builder* builder, const uint32_t input, std::vector<Expr> index, const ExprOob policy) {
            const auto& declared = state(builder).inputs[input];
            if (index.size() != declared.rank || index.empty() || index.size() > 3)
                fail(std::format("gather takes one index per input dimension (1 to 3), got {} for rank {}", index.size(), declared.rank));
            for (auto& i : index) {
                if (i.builder_ != builder || !integral(i.dtype_))
                    fail("gather indices are Int32 or UInt32 expressions of the same kernel");
                i = retype(i, DataType::Int32);
            }
            const auto slot = binding(builder, {input, true, {-1, -1, -1, -1}});
            const auto code = index.size() == 1 ? ExprOp::Gather : index.size() == 2 ? ExprOp::Gather2
                                                                                     : ExprOp::Gather3;
            std::array<uint32_t, 3> ids{};
            for (size_t i = 0; i < index.size(); ++i)
                ids[i] = index[i].id_;
            return make(builder, code, value_type(declared.dtype), ids, uint8_t(slot), 0, policy);
        }
    };

    Expr Expr::cast(const DataType dtype) const {
        if (dtype == dtype_)
            return *this;
        switch (dtype) {
        case DataType::Float32:
            return Nodes::op(dtype_ == DataType::UInt32 ? ExprOp::CastUnsignedFloat : ExprOp::CastFloat, dtype, {*this});
        case DataType::Int32:
        case DataType::UInt32:
            if (dtype_ == DataType::Float32)
                return Nodes::op(dtype == DataType::Int32 ? ExprOp::CastInt : ExprOp::CastUInt, dtype, {*this});
            return Nodes::retype(*this, dtype);
        case DataType::Bool:
            return Nodes::op(dtype_ == DataType::Float32 ? ExprOp::NotEqual : ExprOp::NotEqualInt, dtype,
                             {*this, Nodes::scalar(builder_, dtype_, 0)});
        default: fail(std::format("expressions cannot hold {}", dtype_name(dtype)));
        }
    }

    Expr Input::load(std::initializer_list<size_t> dims) const {
        auto& s = Nodes::state(builder_);
        const auto& declared = s.inputs[index_];
        Builder::State::Binding binding{index_, false, {-1, -1, -1, -1}};
        if (dims.size() == 0 && declared.rank > s.rank)
            fail(std::format("a rank {} input cannot load over a rank {} domain", declared.rank, s.rank));
        if (dims.size() != 0 && dims.size() != declared.rank)
            fail(std::format("load maps {} dimensions of a rank {} input", dims.size(), declared.rank));
        std::array<bool, ExpressionProgram::max_rank> used{};
        for (size_t i = 0; i < declared.rank; ++i) {
            const size_t dim = dims.size() ? dims.begin()[i] : s.rank - declared.rank + i;
            if (dim >= s.rank || used[dim])
                fail("load maps input dimensions onto distinct domain dimensions");
            used[dim] = true;
            binding.dims[i] = int8_t(dim);
        }
        const auto slot = Nodes::binding(builder_, binding);
        return Nodes::make(builder_, ExprOp::Load, value_type(declared.dtype), {}, uint8_t(slot));
    }

    Expr Input::gather(std::initializer_list<Expr> index, const Bounds bounds) const {
        return Nodes::gather(builder_, index_, index, ExprOob(bounds));
    }

    Expr Input::at(std::initializer_list<uint32_t> index) const {
        std::vector<Expr> constants;
        for (const auto i : index) {
            if (i > uint32_t(std::numeric_limits<int32_t>::max()))
                fail("constant index exceeds int32");
            constants.push_back(Nodes::constant(builder_, DataType::Int32, i));
        }
        return Nodes::gather(builder_, index_, constants, ExprOob::Checked);
    }

    Builder::Builder(const size_t rank) : state_(std::make_unique<State>()) {
        if (rank == 0 || rank > ExpressionProgram::max_rank)
            fail(std::format("domain rank must be 1 to {}", ExpressionProgram::max_rank));
        state_->rank = rank;
    }

    Builder::~Builder() = default;

    Input Builder::input(const DataType dtype, const size_t rank) {
        value_type(dtype);
        if (rank == 0 || rank > ExpressionProgram::max_rank)
            fail(std::format("input rank must be 1 to {}", ExpressionProgram::max_rank));
        state_->inputs.push_back({dtype, rank});
        return Input(this, uint32_t(state_->inputs.size() - 1));
    }

    Expr Builder::iota(const size_t dim) {
        if (dim >= state_->rank)
            fail("iota dimension outside the domain");
        return Nodes::make(this, ExprOp::Iota, DataType::Int32, {}, uint8_t(dim));
    }

    Expr Builder::extent(const size_t dim) {
        if (dim >= state_->rank)
            fail("extent dimension outside the domain");
        return Nodes::make(this, ExprOp::Extent, DataType::Int32, {}, uint8_t(dim));
    }

    Expr Builder::constant(const float value) { return Nodes::constant(this, DataType::Float32, std::bit_cast<uint32_t>(value)); }
    Expr Builder::constant(const int32_t value) { return Nodes::constant(this, DataType::Int32, uint32_t(value)); }
    Expr Builder::constant(const uint32_t value) { return Nodes::constant(this, DataType::UInt32, value); }
    Expr Builder::constant(const bool value) { return Nodes::constant(this, DataType::Bool, value); }

    Expr Builder::fold(const Fold kind, const Expr& value, const size_t dim) {
        if (value.builder_ != this)
            fail("fold operand belongs to another builder");
        if (dim >= state_->rank || (state_->fold >= 0 && state_->fold != int(dim)))
            fail("folds of a kernel reduce one domain dimension");
        auto operand = value;
        if (kind == Fold::Count && operand.dtype() == DataType::Float32)
            operand = operand.cast(DataType::Bool);
        const auto type = operand.dtype();
        if ((kind <= Fold::Max && type == DataType::Bool) || (kind >= Fold::And && kind <= Fold::Xor && type == DataType::Float32))
            fail(std::format("fold does not accept {}", dtype_name(type)));
        state_->fold = int(dim);
        return Nodes::make(this, ExprOp::Fold, kind == Fold::Count ? DataType::Int32 : type, {operand.id_},
                           uint8_t(kind), uint32_t(type), ExprOob(dim));
    }

    void Builder::output(const Expr& value, const DataType dtype) {
        const auto type = value.dtype();
        const bool valid = type == DataType::Float32 ? dtype == DataType::Float32 || dtype == DataType::Float16
                           : integral(type)          ? integral(dtype) || dtype == DataType::UInt8
                                                     : dtype == DataType::Bool || dtype == DataType::UInt8;
        if (!valid || value.builder_ != this)
            fail(std::format("a {} value cannot be stored as {}", dtype_name(type), dtype_name(dtype)));
        if (state_->outputs.size() == ExpressionProgram::max_outputs)
            fail(std::format("a kernel writes at most {} outputs", ExpressionProgram::max_outputs));
        state_->ir.emit(ExprOp::Store, value.id_, 0, 0, uint8_t(state_->outputs.size()));
        state_->outputs.push_back(dtype);
    }

    Kernel::Kernel(const Builder& builder) {
        const auto& s = *builder.state_;
        if (s.outputs.empty())
            fail("a kernel needs an output");
        auto program = s.ir.build();
        const auto retained = program.compact_inputs();
        std::vector<Builder::State::Binding> bindings;
        for (const auto index : retained)
            bindings.push_back(s.bindings[index]);
        int fold = -1;
        for (const auto& instruction : program.instructions())
            if (ExprOp(instruction.op_dst_a_b & 255u) == ExprOp::Fold)
                fold = int((instruction.c_aux_policy >> 16) & 255u);
        compiled_ = std::make_shared<const Compiled>(Compiled{std::move(program), s.rank, fold, s.inputs, std::move(bindings), s.outputs});
    }

    Kernel::~Kernel() = default;

    namespace {
        std::vector<size_t> output_shape(const std::vector<size_t>& domain, const int fold) {
            uint64_t count = 1;
            for (size_t d = 0; d < domain.size(); ++d) {
                if (domain[d] == 0 || domain[d] > size_t(INT32_MAX) ||
                    (int(d) == fold && domain[d] > ExpressionProgram::max_fold))
                    fail("domain dimensions must be nonzero, fit int32, and folds must not exceed 4096");
                count *= domain[d];
                if (count > INT32_MAX)
                    fail("domain product exceeds int32");
            }
            std::vector<size_t> shape;
            for (size_t d = 0; d < domain.size(); ++d)
                if (int(d) != fold)
                    shape.push_back(domain[d]);
            return shape.empty() ? std::vector<size_t>{1} : shape;
        }

        uint32_t checked(const size_t value) {
            if (value > size_t(std::numeric_limits<int32_t>::max()))
                fail("tensor sizes and strides must fit int32");
            return uint32_t(value);
        }

        std::vector<Tensor> allocate(const std::vector<size_t>& shape, const std::vector<Tensor>& inputs,
                                     const std::vector<DataType>& dtypes) {
            std::optional<GpuBackend> backend;
            for (const auto& input : inputs)
                if (!backend && input.is_valid() && input.device() == Device::GPU)
                    backend = gpu_backend_of(input);
            GpuBackendScope scope(backend.value_or(default_gpu_backend()));
            std::vector<Tensor> outputs;
            size_t count = 1;
            for (const auto extent : shape)
                count *= extent;
            for (const auto dtype : dtypes) {
                const size_t lanes = 4 / dtype_size(dtype);
                if (count % lanes)
                    outputs.push_back(Tensor::empty({(count + lanes - 1) / lanes * lanes}, Device::GPU, dtype)
                                          .slice(0, 0, count)
                                          .reshape(TensorShape(shape)));
                else
                    outputs.push_back(Tensor::empty(TensorShape(shape), Device::GPU, dtype));
            }
            return outputs;
        }
    } // namespace

    void Kernel::launch(const std::vector<size_t>& domain, const std::vector<Tensor>& inputs,
                        const std::vector<Tensor>& outputs, const bool run) const {
        const auto& kernel = *compiled_;
        if (domain.size() != kernel.rank)
            fail(std::format("domain rank {} differs from the kernel's {}", domain.size(), kernel.rank));
        if (inputs.size() != kernel.inputs.size() || outputs.size() != kernel.outputs.size())
            fail(std::format("kernel takes {} inputs and {} outputs, got {} and {}", kernel.inputs.size(),
                             kernel.outputs.size(), inputs.size(), outputs.size()));
        internal::ExprShape shape;
        shape.rank = uint32_t(kernel.rank);
        for (size_t d = 0; d < domain.size(); ++d) {
            shape.dims[d] = checked(domain[d]);
        }
        std::optional<GpuBackend> backend;
        const auto place = [&](const Tensor& tensor) {
            pin_operands({&tensor});
            if (tensor.device() != Device::GPU)
                return;
            if (backend && gpu_backend_of(tensor) != backend)
                fail("kernel tensors share one GPU backend");
            backend = gpu_backend_of(tensor);
        };
        std::vector<bool> live(kernel.inputs.size());
        for (const auto& binding : kernel.bindings)
            live[binding.input] = true;
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto& tensor = inputs[i];
            if (!tensor.is_valid() || tensor.dtype() != kernel.inputs[i].dtype || tensor.ndim() != kernel.inputs[i].rank)
                fail(std::format("input {} must be a {} tensor of rank {}", i, dtype_name(kernel.inputs[i].dtype), kernel.inputs[i].rank));
            if (live[i])
                place(tensor);
        }
        const auto expected = output_shape(domain, kernel.fold);
        for (size_t o = 0; o < outputs.size(); ++o) {
            const auto& tensor = outputs[o];
            bool matches = tensor.is_valid() && tensor.device() == Device::GPU && tensor.dtype() == kernel.outputs[o] &&
                           tensor.ndim() == expected.size();
            for (size_t d = 0; matches && d < expected.size(); ++d)
                matches = tensor.size(d) == expected[d];
            if (!matches)
                fail(std::format("output {} must be a {} GPU tensor shaped like the domain without the fold dimension",
                                 o, dtype_name(kernel.outputs[o])));
            place(tensor);
        }

        const cudaStream_t stream = outputs[0].stream();
        std::vector<internal::ExprInput> bindings;
        for (const auto& binding : kernel.bindings) {
            const auto& tensor = inputs[binding.input];
            internal::ExprInput input;
            if (tensor.device() == Device::GPU) {
                (void)prepare_inputs_for_stream({&tensor}, stream);
                input.storage = internal::storage_ref(tensor);
            } else {
                input.storage = internal::raw_storage_ref(const_cast<void*>(tensor.data_ptr()), tensor.dtype());
                input.storage.flags |= internal::STORAGE_REF_HOST_MEMORY;
            }
            for (size_t i = 0; i < tensor.ndim(); ++i) {
                const size_t dim = binding.gather ? i : size_t(binding.dims[i]);
                if (!binding.gather && tensor.size(i) != 1 && tensor.size(i) != domain[dim])
                    fail(std::format("input {} dimension {} has extent {} but the domain has {}", binding.input, i,
                                     tensor.size(i), domain[dim]));
                input.dims[dim] = checked(tensor.size(i));
                input.strides[dim] = !binding.gather && tensor.size(i) == 1 ? 0 : int32_t(checked(tensor.stride(i)));
            }
            bindings.push_back(input);
        }
        std::vector<internal::ExprOutput> targets;
        for (const auto& tensor : outputs) {
            if (run) {
                Tensor written = tensor;
                internal::preserve_lazy_snapshots_before_write(written);
            }
            (void)prepare_inputs_for_stream({&tensor}, stream);
            internal::ExprOutput output;
            output.storage = internal::storage_ref(tensor);
            for (size_t d = 0, j = 0; d < domain.size(); ++d)
                if (int(d) != kernel.fold && j < tensor.ndim())
                    output.strides[d] = int32_t(checked(tensor.stride(j++)));
            targets.push_back(output);
        }
        if (run) {
            internal::run_expression(kernel.program, bindings, targets, shape, internal::ExecContext{stream});
            if (*backend == GpuBackend::CUDA)
                for (const auto& output : outputs)
                    if (output.stream() != stream)
                        internal::backend_ops(*backend).bridge(internal::ExecContext{stream}, internal::ExecContext{output.stream()});
        } else
            internal::prepare_expression(kernel.program, bindings, targets, shape);
    }

    std::vector<Tensor> Kernel::operator()(const std::vector<size_t>& domain, const std::vector<Tensor>& inputs) const {
        std::vector<Tensor> live_inputs;
        for (const auto& binding : compiled_->bindings)
            live_inputs.push_back(inputs.at(binding.input));
        auto outputs = allocate(output_shape(domain, compiled_->fold), live_inputs, compiled_->outputs);
        launch(domain, inputs, outputs, true);
        return outputs;
    }

    void Kernel::operator()(const std::vector<size_t>& domain, const std::vector<Tensor>& inputs,
                            const std::vector<Tensor>& outputs) const {
        launch(domain, inputs, outputs, true);
    }

    void Kernel::prepare(const std::vector<size_t>& domain, const std::vector<Tensor>& inputs,
                         const std::vector<Tensor>& outputs) const {
        std::vector<Tensor> live_inputs;
        for (const auto& binding : compiled_->bindings)
            live_inputs.push_back(inputs.at(binding.input));
        launch(domain, inputs, outputs.empty() ? allocate(output_shape(domain, compiled_->fold), live_inputs, compiled_->outputs) : outputs, false);
    }

    Expr operator+(const Operand& a, const Operand& b) { return Nodes::binary(a, b, "+", ExprOp::Add, ExprOp::AddInt, ExprOp::AddInt, false); }
    Expr operator-(const Operand& a, const Operand& b) { return Nodes::binary(a, b, "-", ExprOp::Sub, ExprOp::SubInt, ExprOp::SubInt, false); }
    Expr operator*(const Operand& a, const Operand& b) { return Nodes::binary(a, b, "*", ExprOp::Mul, ExprOp::MulInt, ExprOp::MulInt, false); }
    Expr operator/(const Operand& a, const Operand& b) { return Nodes::binary(a, b, "/", ExprOp::Div, ExprOp::IntDiv, ExprOp::UIntDiv, false); }
    Expr operator%(const Operand& a, const Operand& b) { return Nodes::binary(a, b, "%", ExprOp::Mod, ExprOp::IntMod, ExprOp::UIntMod, false); }
    Expr operator-(const Expr& a) {
        if (a.dtype() == DataType::Float32)
            return Nodes::floating(a, ExprOp::Neg);
        return 0 - a;
    }
    Expr operator<(const Operand& a, const Operand& b) { return Nodes::binary(a, b, "<", ExprOp::Less, ExprOp::LessInt, ExprOp::LessUInt, true); }
    Expr operator<=(const Operand& a, const Operand& b) { return Nodes::binary(a, b, "<=", ExprOp::LessEqual, ExprOp::LessEqualInt, ExprOp::LessEqualUInt, true); }
    Expr operator>(const Operand& a, const Operand& b) { return Nodes::binary(a, b, ">", ExprOp::Greater, ExprOp::GreaterInt, ExprOp::GreaterUInt, true); }
    Expr operator>=(const Operand& a, const Operand& b) { return Nodes::binary(a, b, ">=", ExprOp::GreaterEqual, ExprOp::GreaterEqualInt, ExprOp::GreaterEqualUInt, true); }
    Expr operator==(const Operand& a, const Operand& b) {
        const auto values = Nodes::unify({&a, &b}, "==");
        return Nodes::op(values[0].dtype() == DataType::Float32 ? ExprOp::Equal : ExprOp::EqualInt, DataType::Bool, {values[0], values[1]});
    }
    Expr operator!=(const Operand& a, const Operand& b) {
        const auto values = Nodes::unify({&a, &b}, "!=");
        return Nodes::op(values[0].dtype() == DataType::Float32 ? ExprOp::NotEqual : ExprOp::NotEqualInt, DataType::Bool, {values[0], values[1]});
    }
    Expr operator&&(const Operand& a, const Operand& b) {
        const auto values = Nodes::unify({&a, &b}, "&&");
        if (values[0].dtype() != DataType::Bool)
            fail("&& takes Bool");
        return Nodes::op(ExprOp::LogicalAnd, DataType::Bool, {values[0], values[1]});
    }
    Expr operator||(const Operand& a, const Operand& b) {
        const auto values = Nodes::unify({&a, &b}, "||");
        if (values[0].dtype() != DataType::Bool)
            fail("|| takes Bool");
        return Nodes::op(ExprOp::LogicalOr, DataType::Bool, {values[0], values[1]});
    }
    Expr operator!(const Expr& a) {
        if (a.dtype() != DataType::Bool)
            fail("! takes Bool");
        return Nodes::op(ExprOp::LogicalNot, DataType::Bool, {a});
    }
    Expr operator&(const Operand& a, const Operand& b) { return Nodes::bitwise(a, b, "&", ExprOp::BitAnd, false); }
    Expr operator|(const Operand& a, const Operand& b) { return Nodes::bitwise(a, b, "|", ExprOp::BitOr, false); }
    Expr operator^(const Operand& a, const Operand& b) { return Nodes::bitwise(a, b, "^", ExprOp::BitXor, false); }
    Expr operator~(const Expr& a) {
        if (a.dtype() == DataType::Bool)
            return !a;
        if (!integral(a.dtype()))
            fail("~ takes Int32, UInt32 or Bool");
        return Nodes::op(ExprOp::BitNot, a.dtype(), {a});
    }
    Expr operator<<(const Operand& a, const Operand& b) { return Nodes::bitwise(a, b, "<<", ExprOp::ShiftLeft, true); }
    Expr operator>>(const Operand& a, const Operand& b) { return Nodes::bitwise(a, b, ">>", ExprOp::ShiftRight, true); }

    Expr where(const Operand& condition, const Operand& yes, const Operand& no) {
        return Nodes::select(condition, yes, no);
    }

    Expr min(const Operand& a, const Operand& b) {
        const auto values = Nodes::unify({&a, &b}, "min");
        if (values[0].dtype() == DataType::Float32)
            return Nodes::op(ExprOp::Min, DataType::Float32, {values[0], values[1]});
        return Nodes::choose(values[1] < values[0], values[1], values[0]);
    }

    Expr max(const Operand& a, const Operand& b) {
        const auto values = Nodes::unify({&a, &b}, "max");
        if (values[0].dtype() == DataType::Float32)
            return Nodes::op(ExprOp::Max, DataType::Float32, {values[0], values[1]});
        return Nodes::choose(values[0] < values[1], values[1], values[0]);
    }

    Expr clamp(const Operand& value, const Operand& low, const Operand& high) {
        const auto values = Nodes::unify({&value, &low, &high}, "clamp");
        if (values[0].dtype() == DataType::Float32)
            return Nodes::op(ExprOp::Clamp, DataType::Float32, {values[0], values[1], values[2]});
        return max(min(values[0], values[2]), values[1]);
    }

    Expr abs(const Expr& a) {
        if (a.dtype() == DataType::Float32)
            return Nodes::floating(a, ExprOp::Abs);
        if (a.dtype() != DataType::Int32)
            fail("abs takes Float32 or Int32");
        return Nodes::choose(a < 0, -a, a);
    }

    Expr fma(const Operand& a, const Operand& b, const Operand& c) { return Nodes::floating({&a, &b, &c}, ExprOp::Fma, "fma"); }
    Expr pow(const Operand& a, const Operand& b) { return Nodes::floating({&a, &b}, ExprOp::Pow, "pow"); }
    Expr atan2(const Operand& y, const Operand& x) { return Nodes::floating({&y, &x}, ExprOp::Atan2, "atan2"); }

#define LFS_FUSED_UNARY(name, code, result) \
    Expr name(const Expr& a) { return Nodes::floating(a, ExprOp::code, DataType::result); }
    LFS_FUSED_UNARY(exp, Exp, Float32)
    LFS_FUSED_UNARY(log, Log, Float32)
    LFS_FUSED_UNARY(sqrt, Sqrt, Float32)
    LFS_FUSED_UNARY(sigmoid, Sigmoid, Float32)
    LFS_FUSED_UNARY(relu, Relu, Float32)
    LFS_FUSED_UNARY(square, Square, Float32)
    LFS_FUSED_UNARY(tanh, Tanh, Float32)
    LFS_FUSED_UNARY(rsqrt, Rsqrt, Float32)
    LFS_FUSED_UNARY(sign, Sign, Float32)
    LFS_FUSED_UNARY(reciprocal, Reciprocal, Float32)
    LFS_FUSED_UNARY(floor, Floor, Float32)
    LFS_FUSED_UNARY(ceil, Ceil, Float32)
    LFS_FUSED_UNARY(round, Round, Float32)
    LFS_FUSED_UNARY(exp2, Exp2, Float32)
    LFS_FUSED_UNARY(log2, Log2, Float32)
    LFS_FUSED_UNARY(log10, Log10, Float32)
    LFS_FUSED_UNARY(log1p, Log1p, Float32)
    LFS_FUSED_UNARY(sin, Sin, Float32)
    LFS_FUSED_UNARY(cos, Cos, Float32)
    LFS_FUSED_UNARY(tan, Tan, Float32)
    LFS_FUSED_UNARY(asin, Asin, Float32)
    LFS_FUSED_UNARY(acos, Acos, Float32)
    LFS_FUSED_UNARY(atan, Atan, Float32)
    LFS_FUSED_UNARY(sinh, Sinh, Float32)
    LFS_FUSED_UNARY(cosh, Cosh, Float32)
    LFS_FUSED_UNARY(gelu, Gelu, Float32)
    LFS_FUSED_UNARY(swish, Swish, Float32)
    LFS_FUSED_UNARY(trunc, Trunc, Float32)
    LFS_FUSED_UNARY(isnan, IsNan, Bool)
    LFS_FUSED_UNARY(isinf, IsInf, Bool)
    LFS_FUSED_UNARY(isfinite, IsFinite, Bool)
#undef LFS_FUSED_UNARY
} // namespace lfs::core::fused
