/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/detail/tensor_impl.hpp"

#include "core/cuda_error.hpp"
#include "core/cuda_types.hpp"
#include "core/detail/lazy_executor.hpp"
#include "core/detail/tensor_broadcast.hpp"
#include "core/detail/tensor_cpu_apply.hpp"
#include "core/detail/tensor_dtype_dispatch.hpp"

#include <format>

namespace lfs::core {

    LazyExprState::~LazyExprState() noexcept {
        if (node_id != 0 && !materializer_unregistered) {
            try {
                internal::lazy_executor_unregister_deferred_materializer(node_id);
            } catch (...) {
            }
        }
    }

} // namespace lfs::core

namespace lfs::core {
    void Tensor::propagate_view_meta(Tensor& view) const {
        const_cast<Tensor*>(this)->ensure_storage_meta();
        view.storage_meta_ = storage_meta_;
        view.ensure_state();
        if (state_) {
            view.state_->stream = state_->stream;
        }
        view.view_generation_snapshot_ =
            storage_meta_->generation.load(std::memory_order_relaxed);
    }

    const Tensor& Tensor::contiguous_read(Tensor& materialized) const {
        // Deferred placeholders are stamped contiguous even when the
        // materializer returns a broadcast or expand view. Resolve that
        // before trusting the flag, matching contiguous().
        materialize_if_deferred();
        if (is_contiguous() && !has_zero_stride()) {
            return *this;
        }

        materialized = contiguous();
        LFS_ASSERT_MSG(materialized.is_contiguous(),
                       "non-contiguous read materialization must produce dense storage");
        return materialized;
    }

    bool Tensor::shares_storage_with(const Tensor& other) const {
        if (storage_meta_ && other.storage_meta_ && storage_meta_ == other.storage_meta_) {
            return true;
        }
        return data_owner_ && other.data_owner_ &&
               !data_owner_.owner_before(other.data_owner_) &&
               !other.data_owner_.owner_before(data_owner_);
    }

    // Generic functor-based binary operation (zero enum overhead)
    template <typename SrcT, typename OutT, typename Op>
    Tensor Tensor::binary_op_generic(const Tensor& other, Op op) const {
        validate_binary_op(other, false, true);

        Tensor lhs_materialized;
        Tensor rhs_materialized;
        const Tensor& lhs_dense = contiguous_read(lhs_materialized);
        const Tensor& rhs_dense = other.contiguous_read(rhs_materialized);
        if (&lhs_dense != this || &rhs_dense != &other) {
            return lhs_dense.binary_op_generic<SrcT, OutT>(rhs_dense, op);
        }

        auto broadcast_shape = this->broadcast_shape(other.shape());
        if (!broadcast::can_broadcast(shape_.dims(), other.shape_.dims())) {
            throw std::runtime_error(
                "Incompatible shapes for broadcasting: " + shape_.str() + " vs " + other.shape_.str());
        }

        // Determine output dtype from template parameter
        DataType out_dtype;
        if constexpr (std::is_same_v<OutT, unsigned char>) {
            out_dtype = DataType::Bool;
        } else if constexpr (std::is_same_v<OutT, float>) {
            out_dtype = DataType::Float32;
        } else if constexpr (std::is_same_v<OutT, int>) {
            out_dtype = DataType::Int32;
        } else {
            out_dtype = DataType::Float32; // fallback
        }

        auto result = internal::allocate_like(*this, broadcast_shape, out_dtype);

        bool a_needs_broadcast = (shape_ != broadcast_shape);
        bool b_needs_broadcast = (other.shape() != broadcast_shape);

        if (!a_needs_broadcast && !b_needs_broadcast) {
            // Element-wise operation without broadcasting
            if (device_ == Device::GPU) {
                pin_operands({this, &other});
                internal::backend_ops_for(*this).binary(
                    internal::pointwise_program(dtype_, out_dtype, op),
                    internal::storage_ref(*this), internal::storage_ref(other),
                    internal::storage_ref(result), result.numel(),
                    internal::ExecContext{result.stream()});
                // No sync - tensor operation
            } else {
                pin_operands({this, &other});
                apply_binary_cpu(ptr<SrcT>(), other.ptr<SrcT>(), result.ptr<OutT>(),
                                 result.numel(), op);
            }
        } else {
            // Broadcasting needed
            pin_operands({this, &other});

            if (device_ == Device::GPU) {
                internal::backend_ops_for(*this).broadcast_binary(
                    internal::pointwise_program(dtype_, out_dtype, op),
                    internal::storage_ref(*this), internal::strided_layout(*this),
                    internal::storage_ref(other), internal::strided_layout(other),
                    internal::storage_ref(result), internal::strided_layout(result),
                    internal::ExecContext{result.stream()});
                // No sync - tensor operation
            } else {
                // CPU broadcasting: expand may return zero-stride views;
                // linear apply_binary_cpu needs dense storage — materialize.
                auto a_broadcast = a_needs_broadcast
                                       ? broadcast_to(broadcast_shape).contiguous()
                                       : clone();
                auto b_broadcast = b_needs_broadcast
                                       ? other.broadcast_to(broadcast_shape).contiguous()
                                       : other.clone();
                pin_operands({&a_broadcast, &b_broadcast});
                apply_binary_cpu(a_broadcast.ptr<SrcT>(), b_broadcast.ptr<SrcT>(),
                                 result.ptr<OutT>(), result.numel(), op);
            }
        }

        return result;
    }

    // Generic functor-based in-place scalar operation (zero enum overhead)
    template <typename Op>
    Tensor& Tensor::scalar_op_inplace_generic(float scalar, Op op) {
        preserve_lazy_snapshots_before_write();
        validate_unary_op();
        tensor_contract::require_dtype(
            *this, DataType::Float32, "in-place scalar operation", "input",
            LFS_SOURCE_SITE_CURRENT());
        reject_inplace_on_zero_stride("in-place scalar op");

        if (!is_contiguous()) {
            return mutate_logical_view(
                [&](Tensor& materialized) {
                    materialized.scalar_op_inplace_generic(scalar, op);
                });
        }

        if (device_ == Device::GPU) {
            auto program = internal::pointwise_program(dtype_, dtype_, op);
            program.scalar = internal::scalar_operand(scalar);
            internal::backend_ops_for(*this).scalar(
                program, internal::storage_ref(*this), internal::storage_ref(*this),
                numel(), internal::ExecContext{stream()});
            // No sync - tensor operation
        } else {
            // CPU implementation
            float* dst = ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                dst[i] = op(dst[i], scalar);
            }
        }

        return *this;
    }

    // Generic functor-based in-place binary operation (zero enum overhead)
    template <typename SrcT, typename Op>
    Tensor& Tensor::binary_op_inplace_generic(const Tensor& other, Op op) {
        preserve_lazy_snapshots_before_write();
        // CRITICAL: In-place operations MUST have matching shapes - throw on mismatch!
        if (!is_valid() || !other.is_valid()) {
            throw std::runtime_error("In-place binary op on invalid tensor");
        }
        if (shape_ != other.shape()) {
            throw std::runtime_error(
                "In-place binary op shape mismatch: " + shape_.str() + " vs " + other.shape_.str() +
                " (in-place ops require exact shape match)");
        }
        if (device_ != other.device()) {
            throw std::runtime_error(
                std::string("In-place binary op device mismatch: ") +
                (device_ == Device::GPU ? "CUDA" : "CPU") + " vs " +
                (other.device() == Device::GPU ? "CUDA" : "CPU"));
        }
        internal::require_same_gpu_backend(
            *this, other, "in-place binary operation");
        tensor_contract::require_dtype(
            other, dtype_, "in-place binary operation", "source",
            LFS_SOURCE_SITE_CURRENT());
        tensor_contract::require_dtype(
            *this, DataType::Float32, "in-place binary operation", "destination",
            LFS_SOURCE_SITE_CURRENT());
        reject_inplace_on_zero_stride("in-place binary op");

        if (!is_contiguous()) {
            return mutate_logical_view(
                [&](Tensor& materialized) {
                    materialized.binary_op_inplace_generic<SrcT>(other, op);
                });
        }

        Tensor other_materialized;
        const Tensor& other_dense = other.contiguous_read(other_materialized);

        if (device_ == Device::GPU) {
            pin_operands({this, &other_dense});
            internal::backend_ops_for(*this).binary(
                internal::pointwise_program(dtype_, dtype_, op),
                internal::storage_ref(*this), internal::storage_ref(other_dense),
                internal::storage_ref(*this), numel(),
                internal::ExecContext{stream()});
            // No sync - tensor operation
        } else {
            // CPU implementation
            pin_operands({this, &other_dense});
            apply_binary_cpu(ptr<SrcT>(), other_dense.ptr<SrcT>(), ptr<SrcT>(),
                             numel(), op);
        }

        return *this;
    }

    // Validation helpers - throw on error
    void Tensor::validate_binary_op(const Tensor& other, bool require_same_shape, bool require_same_device) const {
        if (!is_valid() || !other.is_valid()) {
            throw std::runtime_error("Binary operation on invalid tensor");
        }
        if (require_same_device && device_ != other.device()) {
            throw std::runtime_error("Tensors must be on same device");
        }
        if (require_same_device) {
            internal::require_same_gpu_backend(*this, other, "binary operation");
        }
        if (require_same_shape && shape_ != other.shape()) {
            throw std::runtime_error("Shape mismatch: " + shape_.str() + " vs " + other.shape_.str());
        }
        if (!require_same_shape && shape_ != other.shape()) {
            const auto& a = shape_.dims();
            const auto& b = other.shape_.dims();
            if (!broadcast::can_broadcast(a, b)) {
                throw std::runtime_error("Incompatible shapes for broadcasting: " + shape_.str() + " vs " + other.shape_.str());
            }
        }
    }

    // Map add/sub/mul/div functors to LazyPointwiseOpKind tensor-binary stages.
    // Returns nullopt for ops that are not fusable (pow, maximum, ...).
    template <typename Op>
    std::optional<internal::PointwiseOp> Tensor::tensor_binary_fusion_kind() {
        if constexpr (std::is_same_v<Op, ops::add_op>) {
            return internal::PointwiseOp::AddTensor;
        } else if constexpr (std::is_same_v<Op, ops::sub_op>) {
            return internal::PointwiseOp::SubTensor;
        } else if constexpr (std::is_same_v<Op, ops::mul_op>) {
            return internal::PointwiseOp::MulTensor;
        } else if constexpr (std::is_same_v<Op, ops::div_op>) {
            return internal::PointwiseOp::DivTensor;
        } else {
            return std::nullopt;
        }
    }

    // Helper for binary operations with automatic type promotion.
    // Single same-shape contiguous ops stay on the eager fast path. When a
    // fusable chain can form from the size heuristic or deferred LHS,
    // seed or extend a pointwise fusion recipe with a tensor-binary stage so
    // mul+add and mul→reduce collapse to one fused launch.
    template <typename Op>
    Tensor Tensor::binary_op_with_promotion(const Tensor& other, Op op,
                                            bool true_division) const {
        validate_binary_op(other, false, true);

        // Determine promoted dtype for the result
        DataType result_dtype = promote_dtypes(dtype_, other.dtype());
        if (true_division && result_dtype != DataType::Float32 &&
            result_dtype != DataType::Float16) {
            result_dtype = DataType::Float32;
        }
        LFS_ASSERT_MSG(result_dtype != DataType::Bool,
                       "arithmetic on two Bool tensors is unsupported; use a logical operation");

        const auto fusion_kind = tensor_binary_fusion_kind<Op>();
        const bool same_shape_contig_f32 =
            result_dtype == DataType::Float32 &&
            dtype_ == DataType::Float32 && other.dtype() == DataType::Float32 &&
            shape_ == other.shape() &&
            is_contiguous() && other.is_contiguous() &&
            device_ == other.device() &&
            numel() > 0;

        // binary fusion: seed/extend deferred chain for large same-shape
        // float32 binaries (or whenever LHS is already a deferred fusion node).
        // Single non-deferred ops below the size threshold remain eager.
        //
        // Use the raw byte threshold here, not
        // lazy_size_heuristic_should_defer(). That helper treats the test
        // override "size heuristic off" as "always defer" so small unaries
        // still form chains in IR tests — applying it to binaries would
        // regress (tiny a.add(b) would become deferred).
        const bool lhs_deferred = is_deferred();
        const size_t nbytes = numel() * sizeof(float);
        const bool size_wants_defer =
            nbytes >= internal::lazy_executor_size_heuristic_threshold();
        if (fusion_kind.has_value() &&
            same_shape_contig_f32 &&
            internal::lazy_executor_pointwise_fusion_enabled() &&
            (lhs_deferred || size_wants_defer)) {
            Tensor lhs_source = *this;
            Tensor rhs_operand = other;
            // Materializer fallback: direct vectorized launch (no re-entry into fusion).
            const Device dev = device_;
            const TensorShape shp = shape_;
            // Stamp stream hint like TensorExpr::operator Tensor so deferred
            // large binaries keep cross-stream ordering (D4 / stream tests).
            const cudaStream_t stream_hint = [&] {
                if (dev != Device::GPU) {
                    return static_cast<cudaStream_t>(nullptr);
                }
                if (const cudaStream_t current = getCurrentCUDAStream()) {
                    return current;
                }
                if (const cudaStream_t lhs_stream = lhs_source.stream()) {
                    return lhs_stream;
                }
                return rhs_operand.stream();
            }();
            Tensor result = make_deferred_expr_tensor(
                shp, dev, DataType::Float32, internal::gpu_backend_tag(*this),
                [lhs_source, rhs_operand, op, shp, dev, stream_hint]() mutable {
                    lhs_source.materialize_if_deferred();
                    rhs_operand.materialize_if_deferred();
                    std::optional<CUDAStreamGuard> execution_guard;
                    if (dev == Device::GPU) {
                        execution_guard.emplace(prepare_inputs_for_stream(
                            {&lhs_source, &rhs_operand}, stream_hint));
                    }
                    Tensor out = internal::allocate_like(
                        lhs_source, shp, DataType::Float32);
                    if (dev == Device::GPU) {
                        pin_operands({&lhs_source, &rhs_operand});
                        internal::backend_ops_for(lhs_source).binary(internal::pointwise_program(DataType::Float32, DataType::Float32, op), internal::storage_ref(lhs_source), internal::storage_ref(rhs_operand), internal::storage_ref(out), out.numel(), internal::ExecContext{out.stream()});
                        tensor_ops::record_tensor_kernel_launch(1);
                    } else {
                        // LFS-CENSUS-OK(unpinned-multi-capture): host tensors; the CPU loop runs synchronously on this thread.
                        apply_binary_cpu(lhs_source.ptr<float>(), rhs_operand.ptr<float>(),
                                         out.ptr<float>(), out.numel(), op);
                    }
                    return out;
                },
                {lazy_expr_id(), other.lazy_expr_id()});
            if (dev == Device::GPU) {
                result.set_stream(stream_hint);
            }

            if (result.is_valid() && result.is_deferred() && result.state_ &&
                result.state_->lazy) {
                const uint64_t result_node_id = result.lazy_expr_id();
                if (result_node_id != 0) {
                    internal::LazyPointwiseOp fusion_op;
                    fusion_op.kind = *fusion_kind;
                    fusion_op.scalar = 0.0f;
                    fusion_op.rhs = internal::lazy_executor_snapshot_operand(rhs_operand);
                    internal::lazy_executor_register_pointwise_fusion_op(
                        result_node_id,
                        lazy_expr_id(),
                        lhs_source,
                        std::move(fusion_op),
                        result.state_->lazy);
                }
            }
            return result;
        }

        // fast path: same shape/device/dtype, contiguous, non-deferred
        // skip BinaryExpr + TensorLeaf heap cells and launch directly.
        // Match BinaryExpr evaluator: prepare stream first, then empty, so the
        // result tensor inherits the prepared execution stream.
        const bool can_fast_path =
            dtype_ == result_dtype && other.dtype() == result_dtype &&
            shape_ == other.shape() &&
            is_contiguous() && other.is_contiguous() &&
            !is_deferred() && !other.is_deferred() &&
            (result_dtype == DataType::Float32 || result_dtype == DataType::Float16 ||
             result_dtype == DataType::Int32 || result_dtype == DataType::Int64 ||
             result_dtype == DataType::UInt8);

        if (can_fast_path) {
            std::optional<CUDAStreamGuard> execution_guard;
            if (device_ == Device::GPU && numel() > 0) {
                execution_guard.emplace(prepare_inputs_for_stream({this, &other}));
            }
            Tensor result = internal::allocate_like(*this, shape_, result_dtype);
            if (numel() > 0) {
                pin_operands({this, &other});
                if (device_ == Device::GPU) {
                    switch (result_dtype) {
                    case DataType::Float32:
                        internal::backend_ops_for(*this).binary(
                            internal::pointwise_program(dtype_, result_dtype, op),
                            internal::storage_ref(*this), internal::storage_ref(other),
                            internal::storage_ref(result), result.numel(),
                            internal::ExecContext{result.stream()});
                        tensor_ops::record_tensor_kernel_launch(1);
                        break;
                    case DataType::Float16:
                    case DataType::Int32:
                    case DataType::Int64:
                    case DataType::UInt8:
                        internal::backend_ops_for(*this).binary(
                            internal::pointwise_program(dtype_, result_dtype, op),
                            internal::storage_ref(*this), internal::storage_ref(other),
                            internal::storage_ref(result), result.numel(),
                            internal::ExecContext{result.stream()});
                        break;
                    default:
                        break; // unreachable given can_fast_path dtype filter
                    }
                } else {
                    switch (result_dtype) {
                    case DataType::Float32:
                        apply_binary_cpu(ptr<float>(), other.ptr<float>(), result.ptr<float>(),
                                         result.numel(), op);
                        break;
                    case DataType::Float16: {
                        const detail::tensor_half_t* left_ptr = ptr<detail::tensor_half_t>();
                        const detail::tensor_half_t* right_ptr = other.ptr<detail::tensor_half_t>();
                        detail::tensor_half_t* out_ptr = result.ptr<detail::tensor_half_t>();
                        const size_t n = result.numel();
                        for (size_t i = 0; i < n; ++i) {
                            out_ptr[i] = detail::tensor_float_to_half(
                                op(detail::tensor_half_to_float(left_ptr[i]), detail::tensor_half_to_float(right_ptr[i])));
                        }
                        break;
                    }
                    case DataType::Int32:
                        apply_binary_cpu(ptr<int>(), other.ptr<int>(), result.ptr<int>(),
                                         result.numel(), op);
                        break;
                    case DataType::Int64:
                        apply_binary_cpu(ptr<int64_t>(), other.ptr<int64_t>(),
                                         result.ptr<int64_t>(), result.numel(), op);
                        break;
                    case DataType::UInt8:
                        apply_binary_cpu(ptr<uint8_t>(), other.ptr<uint8_t>(),
                                         result.ptr<uint8_t>(), result.numel(), op);
                        break;
                    default:
                        break;
                    }
                }
            }
            internal::lazy_ir_record_binary(*this, other, result, "binary");
            return result;
        }

        // Convert operands to result dtype if needed (promotion / broadcast path)
        const Tensor& lhs = (dtype_ == result_dtype) ? *this : this->to(result_dtype);
        const Tensor& rhs = (other.dtype() == result_dtype) ? other : other.to(result_dtype);

        // Compute broadcast shape
        auto broadcast_shape = lhs.broadcast_shape(rhs.shape());

        // Evaluate eagerly for non-fusable / broadcast / promotion cases
        auto expr = BinaryExpr<TensorLeaf, TensorLeaf, Op>(
            TensorLeaf(lhs), TensorLeaf(rhs), op,
            broadcast_shape, lhs.device(), result_dtype);
        Tensor result = expr.eval();
        internal::lazy_ir_record_binary(lhs, rhs, result, "binary");
        return result;
    }

    // Helper for comparison operations with automatic type promotion
    // Promotes operand types for comparison, but always returns Bool.
    // Eager for the same reasons as binary_op_with_promotion.
    template <typename Op>
    Tensor Tensor::comparison_op_with_promotion(const Tensor& other, Op op) const {
        validate_binary_op(other, false, true);

        // Promote operand types for comparison
        DataType compare_dtype = promote_dtypes(dtype_, other.dtype());

        // Convert operands to common dtype for comparison
        const Tensor& lhs = (dtype_ == compare_dtype) ? *this : this->to(compare_dtype);
        const Tensor& rhs = (other.dtype() == compare_dtype) ? other : other.to(compare_dtype);

        // Compute broadcast shape
        auto broadcast_shape = lhs.broadcast_shape(rhs.shape());

        auto expr = BinaryExpr<TensorLeaf, TensorLeaf, Op>(
            TensorLeaf(lhs), TensorLeaf(rhs), op,
            broadcast_shape, lhs.device(), DataType::Bool);
        Tensor result = expr.eval();
        internal::lazy_ir_record_binary(lhs, rhs, result, "comparison");
        return result;
    }

    Tensor Tensor::eq(const Tensor& other) const {
        return comparison_op_with_promotion(other, ops::equal_op{});
    }

    Tensor Tensor::ne(const Tensor& other) const {
        return comparison_op_with_promotion(other, ops::not_equal_op{});
    }

    Tensor Tensor::lt(const Tensor& other) const {
        return comparison_op_with_promotion(other, ops::less_op{});
    }

    Tensor Tensor::le(const Tensor& other) const {
        return comparison_op_with_promotion(other, ops::less_equal_op{});
    }

    Tensor Tensor::gt(const Tensor& other) const {
        return comparison_op_with_promotion(other, ops::greater_op{});
    }

    Tensor Tensor::ge(const Tensor& other) const {
        return comparison_op_with_promotion(other, ops::greater_equal_op{});
    }

    void Tensor::validate_ternary_op(const Tensor& b, const Tensor& c) const {
        if (!is_valid() || !b.is_valid() || !c.is_valid()) {
            throw std::runtime_error("Ternary operation on invalid tensor");
        }
        if (device_ != b.device() || device_ != c.device()) {
            throw std::runtime_error("All tensors must be on same device");
        }
        internal::require_same_gpu_backend(*this, b, "ternary operation");
        internal::require_same_gpu_backend(*this, c, "ternary operation");
    }

    // Helper to ensure tensor is on same device
    Tensor Tensor::ensure_same_device(const Tensor& other) const {
        if (other.device() == device_) {
            internal::require_same_gpu_backend(*this, other, "device conversion");
            return other;
        }
        if (device_ == Device::GPU) {
            return internal::copy_to_backend(other, gpu_backend_of(*this).value());
        }
        return other.to(device_);
    }

    void Tensor::link_deferred_result_to_inputs(Tensor& result,
                                                std::initializer_list<uint64_t> candidate_input_ids) {
        if (!result.is_valid() || !result.has_lazy_expr()) {
            return;
        }
        const uint64_t result_node_id = result.lazy_expr_id();
        if (result_node_id == 0) {
            return;
        }

        std::vector<uint64_t> input_ids;
        input_ids.reserve(candidate_input_ids.size());
        for (uint64_t input_id : candidate_input_ids) {
            if (input_id == 0) {
                continue;
            }
            if (std::find(input_ids.begin(), input_ids.end(), input_id) == input_ids.end()) {
                input_ids.push_back(input_id);
            }
        }

        if (!input_ids.empty()) {
            internal::lazy_ir_set_node_inputs(result_node_id, input_ids);
        }
    }

    // Helper to create view with shared ownership
    Tensor Tensor::create_view(const TensorShape& new_shape) const {
        // Per-handle deferred: do not key off shared lazy alone
        // a materialized sibling may still hold lazy->result for other handles.
        if (is_deferred()) {
            const uint64_t source_id = lazy_expr_id();
            Tensor source = *this;
            const cudaStream_t source_stream = source.stream();
            TensorShape deferred_shape = new_shape;
            std::vector<uint64_t> deferred_inputs;
            if (source_id != 0) {
                deferred_inputs.push_back(source_id);
            }
            Tensor view = make_deferred_expr_tensor(
                deferred_shape, device_, dtype_, internal::gpu_backend_tag(*this),
                [source = std::move(source), deferred_shape]() mutable {
                    Tensor materialized = source;
                    materialized.materialize_if_deferred();
                    return materialized.create_view(deferred_shape);
                },
                std::move(deferred_inputs));
            view.set_stream(source_stream);
            return view;
        }

        // If tensor is not contiguous, we cannot create a simple reshape view
        // We must materialize it first
        if (!is_contiguous_) {
            // Make contiguous copy, then reshape
            return contiguous().create_view(new_shape);
        }

        // For contiguous tensors, we can create a view with the new shape
        Tensor view(data_, new_shape, device_, dtype_);
        view.data_owner_ = data_owner_;
        view.storage_offset_ = storage_offset_;
        view.is_view_ = true;
        view.is_contiguous_ = true;
        propagate_view_meta(view);
        return view;
    }

    Tensor Tensor::create_strided_view(const TensorShape& new_shape,
                                       RankedDims new_strides) const {
        LFS_ASSERT_MSG(new_strides.size() == new_shape.rank(),
                       "strided view shape and stride ranks must match");
        LFS_ASSERT_MSG(new_shape.elements() == numel(),
                       "metadata-only view must preserve the logical element count");

        if (is_deferred()) {
            const bool identity_shape = new_shape == shape_;
            const bool identity_strides =
                new_strides == strides_ || new_strides == new_shape.strides();
            if (identity_shape && identity_strides) {
                return create_view(new_shape);
            }
            materialize_if_deferred();
        }

        Tensor view;
        view.data_ = data_;
        view.data_owner_ = data_owner_;
        view.shape_ = new_shape;
        view.strides_ = std::move(new_strides);
        view.storage_offset_ = storage_offset_;
        view.device_ = device_;
        view.dtype_ = dtype_;
        view.is_view_ = true;
        view.id_ = profiling_enabled_ ? next_id_++ : 0;

        size_t expected_stride = 1;
        view.is_contiguous_ = true;
        for (int dimension = static_cast<int>(new_shape.rank()) - 1;
             dimension >= 0; --dimension) {
            if (view.strides_[static_cast<size_t>(dimension)] != expected_stride) {
                view.is_contiguous_ = false;
                break;
            }
            expected_stride *= new_shape[static_cast<size_t>(dimension)];
        }
        propagate_view_meta(view);
        return view;
    }

    /// Zero-copy expand / broadcast_to view. Logical numel may grow; broadcast
    // dims receive stride 0. Shares storage with *this.
    Tensor Tensor::create_broadcast_view(const TensorShape& target_shape,
                                         std::vector<size_t> new_strides) const {
        LFS_ASSERT_MSG(new_strides.size() == target_shape.rank(),
                       "broadcast view shape and stride ranks must match");
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid(),
                       "broadcast view requires a valid source tensor");

        Tensor view;
        view.data_ = data_;
        view.data_owner_ = data_owner_;
        view.shape_ = target_shape;
        view.strides_ = std::move(new_strides);
        view.storage_offset_ = storage_offset_;
        view.device_ = device_;
        view.dtype_ = dtype_;
        view.is_view_ = true;
        view.id_ = profiling_enabled_ ? next_id_++ : 0;

        size_t expected_stride = 1;
        view.is_contiguous_ = true;
        for (int dimension = static_cast<int>(target_shape.rank()) - 1;
             dimension >= 0; --dimension) {
            if (view.strides_[static_cast<size_t>(dimension)] != expected_stride) {
                view.is_contiguous_ = false;
                break;
            }
            expected_stride *= target_shape[static_cast<size_t>(dimension)];
        }
        // size-1 dims may legally have any stride in contiguous layout; if any
        // stride is 0 and rank>0 with elements, mark non-contiguous.
        if (view.is_contiguous_) {
            for (size_t s : view.strides_) {
                if (s == 0 && target_shape.elements() > 1) {
                    view.is_contiguous_ = false;
                    break;
                }
            }
        }
        propagate_view_meta(view);
        return view;
    }

    /// Reject in-place mutation of expand/broadcast views (shared cells).
    void Tensor::reject_inplace_on_zero_stride(const char* op_name) const {
        if (has_zero_stride()) {
            throw std::runtime_error(
                std::string(op_name) +
                " is not supported on zero-stride expand/broadcast views "
                "(materialize with contiguous() first)");
        }
    }

    LazyTelemetrySnapshot Tensor::lazy_telemetry_snapshot() {
        return internal::lazy_telemetry_snapshot();
    }

    void Tensor::reset_lazy_telemetry() {
        internal::reset_lazy_telemetry();
    }

    void Tensor::clear_lazy_ir_for_testing() {
        internal::clear_lazy_ir_for_testing();
    }

    std::optional<internal::LazyExprDebugInfo> Tensor::lazy_expr_info() const {
        if (const uint64_t node_id = lazy_expr_id(); node_id != 0) {
            return internal::lazy_ir_node_info(node_id);
        }
        return std::nullopt;
    }

    // ============= BINARY OPERATIONS (Template-based) =============

    // Float32 multiply-add with one rounding, including broadcast operands.

    // Arithmetic operations

    // New functor-based overloads for Tensor (zero enum overhead, lazy evaluation)
    // Now with automatic type promotion for mixed-dtype operations
    Tensor Tensor::add(const Tensor& other) const {
        return binary_op_with_promotion(other, ops::add_op{});
    }

    Tensor Tensor::sub(const Tensor& other) const {
        return binary_op_with_promotion(other, ops::sub_op{});
    }

    Tensor Tensor::mul(const Tensor& other) const {
        return binary_op_with_promotion(other, ops::mul_op{});
    }

    Tensor Tensor::div(const Tensor& other) const {
        return binary_op_with_promotion(other, ops::div_op{}, true);
    }

    Tensor Tensor::pow(const Tensor& other) const {
        return binary_op_with_promotion(other, ops::pow_op{});
    }

    Tensor Tensor::mod(const Tensor& other) const {
        if ((dtype_ == DataType::Int32 || dtype_ == DataType::Int64 ||
             dtype_ == DataType::UInt8 || dtype_ == DataType::Bool) &&
            (other.dtype_ == DataType::Int32 || other.dtype_ == DataType::Int64 ||
             other.dtype_ == DataType::UInt8 || other.dtype_ == DataType::Bool)) {
            // Counted where the divisor lives; only the count comes back.
            const size_t nonzero = other.count_nonzero();
            LFS_ASSERT_MSG(nonzero == other.numel(),
                           std::format("integer modulo divisor has {} zero elements out of {}",
                                       other.numel() - nonzero, other.numel()));
        }
        return binary_op_with_promotion(other, ops::mod_op{});
    }

    Tensor Tensor::maximum(const Tensor& other) const {
        return binary_op_with_promotion(other, ops::maximum_op{});
    }

    Tensor Tensor::minimum(const Tensor& other) const {
        return binary_op_with_promotion(other, ops::minimum_op{});
    }

    template <typename T>
    auto Tensor::validated_scalar_operand(
        const T& value,
        const std::string_view operation,
        const std::initializer_list<DataType> allowed_dtypes) const {
        validate_unary_op();
        tensor_contract::require_dtype(
            *this, allowed_dtypes, operation, "input", LFS_SOURCE_SITE_CURRENT());
        if constexpr (std::is_floating_point_v<T>) {
            return static_cast<float>(value);
        } else {
            LFS_ASSERT_MSG(std::in_range<int32_t>(value),
                           std::string(operation) + " integer scalar is outside Int32 range");
            if (dtype_ == DataType::Int32 && operation == "mod") {
                LFS_ASSERT_MSG(value != 0,
                               "integer modulo divisor must not be zero");
            }
            if (dtype_ == DataType::Int32 && operation == "pow") {
                LFS_ASSERT_MSG(value >= 0,
                               "integer power does not accept a negative integer exponent");
            }
            return static_cast<int32_t>(value);
        }
    }

    // Macro for scalar binary operations (lazy evaluation with scalar_right_op)

    // Logical operations (Tensor only, Bool -> Bool)
    Tensor Tensor::logical_and(const Tensor& other) const {
        LFS_ASSERT_MSG(dtype_ == DataType::Bool && other.dtype() == DataType::Bool,
                       "logical_and requires Bool tensors");
        return comparison_op_with_promotion(other, ops::logical_and_op{});
    }

    Tensor Tensor::logical_or(const Tensor& other) const {
        LFS_ASSERT_MSG(dtype_ == DataType::Bool && other.dtype() == DataType::Bool,
                       "logical_or requires Bool tensors");
        return comparison_op_with_promotion(other, ops::logical_or_op{});
    }

    Tensor Tensor::logical_xor(const Tensor& other) const {
        LFS_ASSERT_MSG(dtype_ == DataType::Bool && other.dtype() == DataType::Bool,
                       "logical_xor requires Bool tensors");
        return comparison_op_with_promotion(other, ops::logical_xor_op{});
    }

    // Scalar reduce operations - use direct CUB path for CUDA Float32 contiguous tensors
    float Tensor::sum_scalar() const {
        if (device_ == Device::GPU && dtype_ == DataType::Float32 && is_contiguous_) {
            return internal::backend_ops_for(*this).sum_scalar(
                internal::storage_ref(*this), numel(), internal::ExecContext{stream()});
        }
        auto result = sum();
        if (dtype_ == DataType::Bool) {
            return static_cast<float>(result.item<int64_t>());
        }
        return result.item<float>();
    }

    float Tensor::mean_scalar() const {
        if (device_ == Device::GPU && dtype_ == DataType::Float32 && is_contiguous_) {
            return internal::backend_ops_for(*this).mean_scalar(
                internal::storage_ref(*this), numel(), internal::ExecContext{stream()});
        }
        return mean().item();
    }

    float Tensor::min_scalar() const {
        if (device_ == Device::GPU && dtype_ == DataType::Float32 && is_contiguous_) {
            return internal::backend_ops_for(*this).min_scalar(
                internal::storage_ref(*this), numel(), internal::ExecContext{stream()});
        }
        return min().item();
    }

    float Tensor::max_scalar() const {
        if (device_ == Device::GPU && dtype_ == DataType::Float32 && is_contiguous_) {
            return internal::backend_ops_for(*this).max_scalar(
                internal::storage_ref(*this), numel(), internal::ExecContext{stream()});
        }
        return max().item();
    }

    template <typename T>
    T Tensor::item() const {
        materialize_if_deferred();
        if (!is_valid()) {
            throw std::runtime_error("item<T>() called on invalid tensor");
        }
        if (numel() != 1) {
            throw std::runtime_error(
                "item<T>() requires single-element tensor, got " + std::to_string(numel()) + " elements");
        }

        const auto read_storage_value = [this]<typename StorageT>() {
            StorageT value{};
            internal::read_scalar(*this, 0, &value, sizeof(StorageT));
            return value;
        };

        if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int>) {
            if (dtype_ == DataType::Int64) {
                const int64_t value = read_storage_value.template operator()<int64_t>();
                LFS_ASSERT_MSG(
                    value >= static_cast<int64_t>(std::numeric_limits<T>::min()) &&
                        value <= static_cast<int64_t>(std::numeric_limits<T>::max()),
                    "item<int>() cannot narrow an out-of-range Int64 scalar");
                return static_cast<T>(value);
            }
        }

        // Validate that template type T matches tensor's dtype
        // Note: unsigned char can be used for both Bool and UInt8 (since uint8_t is typedef of unsigned char)
        bool dtype_matches = false;

        if constexpr (std::is_same_v<T, float>) {
            dtype_matches = (dtype_ == DataType::Float32);
        } else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int>) {
            dtype_matches = (dtype_ == DataType::Int32);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            dtype_matches = (dtype_ == DataType::Int64);
        } else if constexpr (std::is_same_v<T, unsigned char> || std::is_same_v<T, uint8_t> || std::is_same_v<T, bool>) {
            // unsigned char/uint8_t can be used for both Bool and UInt8
            dtype_matches = (dtype_ == DataType::Bool || dtype_ == DataType::UInt8);
        }
        // Note: __half check omitted as it's only available in CUDA compilation units
        // Float16 tensors should be accessed through .to(Float32).item<float>()

        if (!dtype_matches) {
            throw std::runtime_error(
                std::string("item<T>(): dtype mismatch - tensor is ") + dtype_name(dtype_) +
                ", but requested incompatible type T");
        }

        return read_storage_value.template operator()<T>();
    }

    // In-place operations (Template-based, direct functor dispatch - zero enum overhead!)
    template <typename T>
    Tensor& Tensor::add_(const T& other) {
        if constexpr (std::is_same_v<T, Tensor>) {
            return binary_op_inplace_generic(other, ops::add_op{});
        } else {
            return scalar_op_inplace_generic(static_cast<float>(other), ops::add_op{});
        }
    }

    template <typename T>
    Tensor& Tensor::sub_(const T& other) {
        if constexpr (std::is_same_v<T, Tensor>) {
            return binary_op_inplace_generic(other, ops::sub_op{});
        } else {
            return scalar_op_inplace_generic(static_cast<float>(other), ops::sub_op{});
        }
    }

    template <typename T>
    Tensor& Tensor::mul_(const T& other) {
        if constexpr (std::is_same_v<T, Tensor>) {
            return binary_op_inplace_generic(other, ops::mul_op{});
        } else {
            return scalar_op_inplace_generic(static_cast<float>(other), ops::mul_op{});
        }
    }

    template <typename T>
    Tensor& Tensor::div_(const T& other) {
        if constexpr (std::is_same_v<T, Tensor>) {
            return binary_op_inplace_generic(other, ops::div_op{});
        } else {
            return scalar_op_inplace_generic(static_cast<float>(other), ops::div_op{});
        }
    }

#define LFS_DEFINE_UNARY_OP(name, op_type)                                                                         \
    Tensor Tensor::name() const {                                                                                  \
        validate_unary_op();                                                                                       \
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 || dtype_ == DataType::Int32,                                   \
                       ::lfs::core::detail::format_cuda_safe("{} requires Float32 or Int32 input "                 \
                                                             "(operation={}, input_dtype={}({}), input_shape={}, " \
                                                             "input_device={})",                                   \
                                                             #name, #name, dtype_name(dtype_),                     \
                                                             static_cast<int>(dtype_), shape_.str(),               \
                                                             device_name(device_)));                               \
        const DataType result_dtype =                                                                              \
            dtype_ == DataType::Int32 && !ops::supports_int32_v<ops::op_type>                                      \
                ? DataType::Float32                                                                                \
                : dtype_;                                                                                          \
        if (numel() == 0) {                                                                                        \
            return internal::allocate_like(*this, shape_, result_dtype);                                           \
        }                                                                                                          \
        Tensor result = UnaryExpr<TensorLeaf, ops::op_type>(                                                       \
            TensorLeaf(*this), ops::op_type{}, shape_, device_, result_dtype);                                     \
        link_deferred_result_to_inputs(result, {lazy_expr_id()});                                                  \
        return result;                                                                                             \
    }

#define LFS_DEFINE_UNARY_OP_FUSABLE(name, op_type, fusion_kind)                           \
    Tensor Tensor::name() const {                                                         \
        validate_unary_op();                                                              \
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 || dtype_ == DataType::Int32,          \
                       #name " currently supports only Float32 and Int32");               \
        const DataType result_dtype =                                                     \
            dtype_ == DataType::Int32 && !ops::supports_int32_v<ops::op_type>             \
                ? DataType::Float32                                                       \
                : dtype_;                                                                 \
        if (numel() == 0) {                                                               \
            return internal::allocate_like(*this, shape_, result_dtype);                  \
        }                                                                                 \
        Tensor result = UnaryExpr<TensorLeaf, ops::op_type>(                              \
            TensorLeaf(*this), ops::op_type{}, shape_, device_, result_dtype);            \
        link_deferred_result_to_inputs(result, {lazy_expr_id()});                         \
        if (dtype_ == DataType::Float32 &&                                                \
            result.is_valid() && result.is_deferred()) {                                  \
            const uint64_t result_node_id = result.lazy_expr_id();                        \
            if (result_node_id != 0 && result.state_) {                                   \
                LFS_ASSERT_MSG(result.state_->lazy != nullptr,                            \
                               "fusable unary result requires shared lazy state");        \
                internal::lazy_executor_register_pointwise_fusion_op(                     \
                    result_node_id,                                                       \
                    lazy_expr_id(),                                                       \
                    *this,                                                                \
                    internal::LazyPointwiseOp{internal::LazyPointwiseOpKind::fusion_kind, \
                                              0.0f},                                      \
                    result.state_->lazy);                                                 \
            }                                                                             \
        }                                                                                 \
        return result;                                                                    \
    }

    // Macro for unary ops that return Bool dtype (isnan, isinf, etc.)
#define LFS_DEFINE_UNARY_OP_BOOL(name, op_type)                                    \
    Tensor Tensor::name() const {                                                  \
        validate_unary_op();                                                       \
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 || dtype_ == DataType::Int32 || \
                           dtype_ == DataType::UInt8 || dtype_ == DataType::Bool,  \
                       #name " encountered an unsupported dtype");                 \
        if (numel() == 0) {                                                        \
            return internal::allocate_like(*this, shape_, DataType::Bool);         \
        }                                                                          \
        Tensor result = UnaryExpr<TensorLeaf, ops::op_type>(                       \
            TensorLeaf(*this), ops::op_type{}, shape_, device_, DataType::Bool);   \
        link_deferred_result_to_inputs(result, {lazy_expr_id()});                  \
        return result;                                                             \
    }

    // Arithmetic unary operations
    LFS_DEFINE_UNARY_OP_FUSABLE(neg, neg_op, Neg)
    LFS_DEFINE_UNARY_OP_FUSABLE(abs, abs_op, Abs)
    LFS_DEFINE_UNARY_OP_FUSABLE(sign, sign_op, Sign)
    LFS_DEFINE_UNARY_OP_FUSABLE(reciprocal, reciprocal_op, Reciprocal)

    // Exponential and logarithmic
    LFS_DEFINE_UNARY_OP_FUSABLE(exp, exp_op, Exp)
    LFS_DEFINE_UNARY_OP(exp2, exp2_op)
    LFS_DEFINE_UNARY_OP_FUSABLE(log, log_op, Log)
    LFS_DEFINE_UNARY_OP(log2, log2_op)
    LFS_DEFINE_UNARY_OP(log10, log10_op)
    LFS_DEFINE_UNARY_OP(log1p, log1p_op)

    // Power and roots
    LFS_DEFINE_UNARY_OP_FUSABLE(sqrt, sqrt_op, Sqrt)
    LFS_DEFINE_UNARY_OP_FUSABLE(rsqrt, rsqrt_op, Rsqrt)
    LFS_DEFINE_UNARY_OP_FUSABLE(square, square_op, Square)

    // Trigonometric
    LFS_DEFINE_UNARY_OP(sin, sin_op)
    LFS_DEFINE_UNARY_OP(cos, cos_op)
    LFS_DEFINE_UNARY_OP(tan, tan_op)
    LFS_DEFINE_UNARY_OP(asin, asin_op)
    LFS_DEFINE_UNARY_OP(acos, acos_op)
    LFS_DEFINE_UNARY_OP(atan, atan_op)

    // Hyperbolic
    LFS_DEFINE_UNARY_OP(sinh, sinh_op)
    LFS_DEFINE_UNARY_OP(cosh, cosh_op)
    LFS_DEFINE_UNARY_OP_FUSABLE(tanh, tanh_op, Tanh)

    // Activation functions
    LFS_DEFINE_UNARY_OP_FUSABLE(sigmoid, sigmoid_op, Sigmoid)
    LFS_DEFINE_UNARY_OP_FUSABLE(relu, relu_op, Relu)
    LFS_DEFINE_UNARY_OP(gelu, gelu_op)
    LFS_DEFINE_UNARY_OP(swish, swish_op)

    // Rounding
    LFS_DEFINE_UNARY_OP_FUSABLE(floor, floor_op, Floor)
    LFS_DEFINE_UNARY_OP_FUSABLE(ceil, ceil_op, Ceil)
    LFS_DEFINE_UNARY_OP_FUSABLE(round, round_op, Round)
    LFS_DEFINE_UNARY_OP(trunc, trunc_op)

    // Boolean predicates (return Bool dtype)
    LFS_DEFINE_UNARY_OP_BOOL(isnan, isnan_op)
    LFS_DEFINE_UNARY_OP_BOOL(isinf, isinf_op)
    LFS_DEFINE_UNARY_OP_BOOL(isfinite, isfinite_op)
    LFS_DEFINE_UNARY_OP_BOOL(logical_not, logical_not_op)

#undef LFS_DEFINE_UNARY_OP
#undef LFS_DEFINE_UNARY_OP_FUSABLE
#undef LFS_DEFINE_UNARY_OP_BOOL
#define LFS_DEFINE_SCALAR_BINARY_OP_FUSABLE(name, op_type, fusion_kind, true_division)     \
    template <typename T, typename>                                                        \
    Tensor Tensor::name(const T& other) const {                                            \
        const auto scalar_value = validated_scalar_operand(                                \
            other, #name, {DataType::Float32, DataType::Int32});                           \
        DataType result_dtype = promote_dtypes(dtype_, scalar_operand_dtype<T>());         \
        if constexpr (true_division) {                                                     \
            result_dtype = DataType::Float32;                                              \
        }                                                                                  \
        if (numel() == 0) {                                                                \
            return internal::allocate_like(*this, shape_, result_dtype);                   \
        }                                                                                  \
        Tensor scalar_input = dtype_ == result_dtype ? *this : to(result_dtype);           \
        using Scalar = std::remove_cv_t<decltype(scalar_value)>;                           \
        Tensor result = UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::op_type, Scalar>>( \
            TensorLeaf(scalar_input),                                                      \
            ops::scalar_right_op<ops::op_type, Scalar>(scalar_value),                      \
            shape_, device_, result_dtype);                                                \
        link_deferred_result_to_inputs(result, {lazy_expr_id()});                          \
        if (result_dtype == DataType::Float32 &&                                           \
            result.is_valid() && result.is_deferred()) {                                   \
            const uint64_t result_node_id = result.lazy_expr_id();                         \
            if (result_node_id != 0 && result.state_) {                                    \
                LFS_ASSERT_MSG(result.state_->lazy != nullptr,                             \
                               "fusable scalar result requires shared lazy state");        \
                internal::lazy_executor_register_pointwise_fusion_op(                      \
                    result_node_id,                                                        \
                    lazy_expr_id(),                                                        \
                    scalar_input,                                                          \
                    internal::LazyPointwiseOp{internal::LazyPointwiseOpKind::fusion_kind,  \
                                              static_cast<float>(scalar_value)},           \
                    result.state_->lazy);                                                  \
            }                                                                              \
        }                                                                                  \
        return result;                                                                     \
    }

#define LFS_DEFINE_SCALAR_BINARY_OP(name, op_type, true_division)                          \
    template <typename T, typename>                                                        \
    Tensor Tensor::name(const T& other) const {                                            \
        const auto scalar_value = validated_scalar_operand(                                \
            other, #name, {DataType::Float32, DataType::Int32});                           \
        DataType result_dtype = promote_dtypes(dtype_, scalar_operand_dtype<T>());         \
        if constexpr (true_division) {                                                     \
            result_dtype = DataType::Float32;                                              \
        }                                                                                  \
        if (numel() == 0) {                                                                \
            return internal::allocate_like(*this, shape_, result_dtype);                   \
        }                                                                                  \
        Tensor scalar_input = dtype_ == result_dtype ? *this : to(result_dtype);           \
        using Scalar = std::remove_cv_t<decltype(scalar_value)>;                           \
        Tensor result = UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::op_type, Scalar>>( \
            TensorLeaf(scalar_input),                                                      \
            ops::scalar_right_op<ops::op_type, Scalar>(scalar_value),                      \
            shape_, device_, result_dtype);                                                \
        link_deferred_result_to_inputs(result, {lazy_expr_id()});                          \
        return result;                                                                     \
    }

    LFS_DEFINE_SCALAR_BINARY_OP_FUSABLE(add, add_op, AddScalar, false)
    LFS_DEFINE_SCALAR_BINARY_OP_FUSABLE(sub, sub_op, SubScalar, false)
    LFS_DEFINE_SCALAR_BINARY_OP_FUSABLE(mul, mul_op, MulScalar, false)
    LFS_DEFINE_SCALAR_BINARY_OP_FUSABLE(div, div_op, DivScalar, true)
    LFS_DEFINE_SCALAR_BINARY_OP(pow, pow_op, false)
    LFS_DEFINE_SCALAR_BINARY_OP(mod, mod_op, false)
    LFS_DEFINE_SCALAR_BINARY_OP(maximum, maximum_op, false)
    LFS_DEFINE_SCALAR_BINARY_OP(minimum, minimum_op, false)

#undef LFS_DEFINE_SCALAR_BINARY_OP
#undef LFS_DEFINE_SCALAR_BINARY_OP_FUSABLE
#define LFS_DEFINE_SCALAR_CMP_OP(name, op_type)                                           \
    template <typename T, typename>                                                       \
    Tensor Tensor::name(const T& other) const {                                           \
        const auto scalar_value = validated_scalar_operand(                               \
            other, #name,                                                                 \
            {DataType::Float32, DataType::Int32, DataType::UInt8, DataType::Bool});       \
        const DataType compare_dtype = promote_dtypes(dtype_, scalar_operand_dtype<T>()); \
        if (numel() == 0) {                                                               \
            return internal::allocate_like(*this, shape_, DataType::Bool);                \
        }                                                                                 \
        Tensor scalar_input = dtype_ == compare_dtype ? *this : to(compare_dtype);        \
        using Scalar = std::remove_cv_t<decltype(scalar_value)>;                          \
        return UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::op_type, Scalar>>(         \
            TensorLeaf(scalar_input),                                                     \
            ops::scalar_right_op<ops::op_type, Scalar>(scalar_value),                     \
            shape_, device_, DataType::Bool);                                             \
    }

    LFS_DEFINE_SCALAR_CMP_OP(eq, equal_op)
    LFS_DEFINE_SCALAR_CMP_OP(ne, not_equal_op)
    LFS_DEFINE_SCALAR_CMP_OP(lt, less_op)
    LFS_DEFINE_SCALAR_CMP_OP(le, less_equal_op)
    LFS_DEFINE_SCALAR_CMP_OP(gt, greater_op)
    LFS_DEFINE_SCALAR_CMP_OP(ge, greater_equal_op)

#undef LFS_DEFINE_SCALAR_CMP_OP

    // Fixed arithmetic types callers pass to the scalar templates.

    template LFS_CORE_API Tensor Tensor::add<float>(const float&) const;
    template LFS_CORE_API Tensor Tensor::sub<float>(const float&) const;
    template LFS_CORE_API Tensor Tensor::mul<float>(const float&) const;
    template LFS_CORE_API Tensor Tensor::div<float>(const float&) const;
    template LFS_CORE_API Tensor Tensor::pow<float>(const float&) const;
    template LFS_CORE_API Tensor Tensor::mod<float>(const float&) const;
    template LFS_CORE_API Tensor Tensor::maximum<float>(const float&) const;
    template LFS_CORE_API Tensor Tensor::minimum<float>(const float&) const;
    template LFS_CORE_API Tensor Tensor::eq<float>(const float&) const;
    template LFS_CORE_API Tensor Tensor::ne<float>(const float&) const;
    template LFS_CORE_API Tensor Tensor::lt<float>(const float&) const;
    template LFS_CORE_API Tensor Tensor::le<float>(const float&) const;
    template LFS_CORE_API Tensor Tensor::gt<float>(const float&) const;
    template LFS_CORE_API Tensor Tensor::ge<float>(const float&) const;
    template LFS_CORE_API Tensor& Tensor::add_<float>(const float&);
    template LFS_CORE_API Tensor& Tensor::sub_<float>(const float&);
    template LFS_CORE_API Tensor& Tensor::mul_<float>(const float&);
    template LFS_CORE_API Tensor& Tensor::div_<float>(const float&);
    template LFS_CORE_API Tensor Tensor::add<double>(const double&) const;
    template LFS_CORE_API Tensor Tensor::sub<double>(const double&) const;
    template LFS_CORE_API Tensor Tensor::mul<double>(const double&) const;
    template LFS_CORE_API Tensor Tensor::div<double>(const double&) const;
    template LFS_CORE_API Tensor Tensor::pow<double>(const double&) const;
    template LFS_CORE_API Tensor Tensor::mod<double>(const double&) const;
    template LFS_CORE_API Tensor Tensor::maximum<double>(const double&) const;
    template LFS_CORE_API Tensor Tensor::minimum<double>(const double&) const;
    template LFS_CORE_API Tensor Tensor::eq<double>(const double&) const;
    template LFS_CORE_API Tensor Tensor::ne<double>(const double&) const;
    template LFS_CORE_API Tensor Tensor::lt<double>(const double&) const;
    template LFS_CORE_API Tensor Tensor::le<double>(const double&) const;
    template LFS_CORE_API Tensor Tensor::gt<double>(const double&) const;
    template LFS_CORE_API Tensor Tensor::ge<double>(const double&) const;
    template LFS_CORE_API Tensor& Tensor::add_<double>(const double&);
    template LFS_CORE_API Tensor& Tensor::sub_<double>(const double&);
    template LFS_CORE_API Tensor& Tensor::mul_<double>(const double&);
    template LFS_CORE_API Tensor& Tensor::div_<double>(const double&);
    template LFS_CORE_API Tensor Tensor::add<int>(const int&) const;
    template LFS_CORE_API Tensor Tensor::sub<int>(const int&) const;
    template LFS_CORE_API Tensor Tensor::mul<int>(const int&) const;
    template LFS_CORE_API Tensor Tensor::div<int>(const int&) const;
    template LFS_CORE_API Tensor Tensor::pow<int>(const int&) const;
    template LFS_CORE_API Tensor Tensor::mod<int>(const int&) const;
    template LFS_CORE_API Tensor Tensor::maximum<int>(const int&) const;
    template LFS_CORE_API Tensor Tensor::minimum<int>(const int&) const;
    template LFS_CORE_API Tensor Tensor::eq<int>(const int&) const;
    template LFS_CORE_API Tensor Tensor::ne<int>(const int&) const;
    template LFS_CORE_API Tensor Tensor::lt<int>(const int&) const;
    template LFS_CORE_API Tensor Tensor::le<int>(const int&) const;
    template LFS_CORE_API Tensor Tensor::gt<int>(const int&) const;
    template LFS_CORE_API Tensor Tensor::ge<int>(const int&) const;
    template LFS_CORE_API Tensor& Tensor::add_<int>(const int&);
    template LFS_CORE_API Tensor& Tensor::sub_<int>(const int&);
    template LFS_CORE_API Tensor& Tensor::mul_<int>(const int&);
    template LFS_CORE_API Tensor& Tensor::div_<int>(const int&);
    template LFS_CORE_API Tensor Tensor::add<long>(const long&) const;
    template LFS_CORE_API Tensor Tensor::sub<long>(const long&) const;
    template LFS_CORE_API Tensor Tensor::mul<long>(const long&) const;
    template LFS_CORE_API Tensor Tensor::div<long>(const long&) const;
    template LFS_CORE_API Tensor Tensor::pow<long>(const long&) const;
    template LFS_CORE_API Tensor Tensor::mod<long>(const long&) const;
    template LFS_CORE_API Tensor Tensor::maximum<long>(const long&) const;
    template LFS_CORE_API Tensor Tensor::minimum<long>(const long&) const;
    template LFS_CORE_API Tensor Tensor::eq<long>(const long&) const;
    template LFS_CORE_API Tensor Tensor::ne<long>(const long&) const;
    template LFS_CORE_API Tensor Tensor::lt<long>(const long&) const;
    template LFS_CORE_API Tensor Tensor::le<long>(const long&) const;
    template LFS_CORE_API Tensor Tensor::gt<long>(const long&) const;
    template LFS_CORE_API Tensor Tensor::ge<long>(const long&) const;
    template LFS_CORE_API Tensor& Tensor::add_<long>(const long&);
    template LFS_CORE_API Tensor& Tensor::sub_<long>(const long&);
    template LFS_CORE_API Tensor& Tensor::mul_<long>(const long&);
    template LFS_CORE_API Tensor& Tensor::div_<long>(const long&);
    template LFS_CORE_API Tensor Tensor::add<long long>(const long long&) const;
    template LFS_CORE_API Tensor Tensor::sub<long long>(const long long&) const;
    template LFS_CORE_API Tensor Tensor::mul<long long>(const long long&) const;
    template LFS_CORE_API Tensor Tensor::div<long long>(const long long&) const;
    template LFS_CORE_API Tensor Tensor::pow<long long>(const long long&) const;
    template LFS_CORE_API Tensor Tensor::mod<long long>(const long long&) const;
    template LFS_CORE_API Tensor Tensor::maximum<long long>(const long long&) const;
    template LFS_CORE_API Tensor Tensor::minimum<long long>(const long long&) const;
    template LFS_CORE_API Tensor Tensor::eq<long long>(const long long&) const;
    template LFS_CORE_API Tensor Tensor::ne<long long>(const long long&) const;
    template LFS_CORE_API Tensor Tensor::lt<long long>(const long long&) const;
    template LFS_CORE_API Tensor Tensor::le<long long>(const long long&) const;
    template LFS_CORE_API Tensor Tensor::gt<long long>(const long long&) const;
    template LFS_CORE_API Tensor Tensor::ge<long long>(const long long&) const;
    template LFS_CORE_API Tensor& Tensor::add_<long long>(const long long&);
    template LFS_CORE_API Tensor& Tensor::sub_<long long>(const long long&);
    template LFS_CORE_API Tensor& Tensor::mul_<long long>(const long long&);
    template LFS_CORE_API Tensor& Tensor::div_<long long>(const long long&);
    template LFS_CORE_API Tensor Tensor::add<unsigned>(const unsigned&) const;
    template LFS_CORE_API Tensor Tensor::sub<unsigned>(const unsigned&) const;
    template LFS_CORE_API Tensor Tensor::mul<unsigned>(const unsigned&) const;
    template LFS_CORE_API Tensor Tensor::div<unsigned>(const unsigned&) const;
    template LFS_CORE_API Tensor Tensor::pow<unsigned>(const unsigned&) const;
    template LFS_CORE_API Tensor Tensor::mod<unsigned>(const unsigned&) const;
    template LFS_CORE_API Tensor Tensor::maximum<unsigned>(const unsigned&) const;
    template LFS_CORE_API Tensor Tensor::minimum<unsigned>(const unsigned&) const;
    template LFS_CORE_API Tensor Tensor::eq<unsigned>(const unsigned&) const;
    template LFS_CORE_API Tensor Tensor::ne<unsigned>(const unsigned&) const;
    template LFS_CORE_API Tensor Tensor::lt<unsigned>(const unsigned&) const;
    template LFS_CORE_API Tensor Tensor::le<unsigned>(const unsigned&) const;
    template LFS_CORE_API Tensor Tensor::gt<unsigned>(const unsigned&) const;
    template LFS_CORE_API Tensor Tensor::ge<unsigned>(const unsigned&) const;
    template LFS_CORE_API Tensor& Tensor::add_<unsigned>(const unsigned&);
    template LFS_CORE_API Tensor& Tensor::sub_<unsigned>(const unsigned&);
    template LFS_CORE_API Tensor& Tensor::mul_<unsigned>(const unsigned&);
    template LFS_CORE_API Tensor& Tensor::div_<unsigned>(const unsigned&);
    template LFS_CORE_API Tensor Tensor::add<unsigned long>(const unsigned long&) const;
    template LFS_CORE_API Tensor Tensor::sub<unsigned long>(const unsigned long&) const;
    template LFS_CORE_API Tensor Tensor::mul<unsigned long>(const unsigned long&) const;
    template LFS_CORE_API Tensor Tensor::div<unsigned long>(const unsigned long&) const;
    template LFS_CORE_API Tensor Tensor::pow<unsigned long>(const unsigned long&) const;
    template LFS_CORE_API Tensor Tensor::mod<unsigned long>(const unsigned long&) const;
    template LFS_CORE_API Tensor Tensor::maximum<unsigned long>(const unsigned long&) const;
    template LFS_CORE_API Tensor Tensor::minimum<unsigned long>(const unsigned long&) const;
    template LFS_CORE_API Tensor Tensor::eq<unsigned long>(const unsigned long&) const;
    template LFS_CORE_API Tensor Tensor::ne<unsigned long>(const unsigned long&) const;
    template LFS_CORE_API Tensor Tensor::lt<unsigned long>(const unsigned long&) const;
    template LFS_CORE_API Tensor Tensor::le<unsigned long>(const unsigned long&) const;
    template LFS_CORE_API Tensor Tensor::gt<unsigned long>(const unsigned long&) const;
    template LFS_CORE_API Tensor Tensor::ge<unsigned long>(const unsigned long&) const;
    template LFS_CORE_API Tensor& Tensor::add_<unsigned long>(const unsigned long&);
    template LFS_CORE_API Tensor& Tensor::sub_<unsigned long>(const unsigned long&);
    template LFS_CORE_API Tensor& Tensor::mul_<unsigned long>(const unsigned long&);
    template LFS_CORE_API Tensor& Tensor::div_<unsigned long>(const unsigned long&);
    template LFS_CORE_API Tensor Tensor::add<unsigned long long>(const unsigned long long&) const;
    template LFS_CORE_API Tensor Tensor::sub<unsigned long long>(const unsigned long long&) const;
    template LFS_CORE_API Tensor Tensor::mul<unsigned long long>(const unsigned long long&) const;
    template LFS_CORE_API Tensor Tensor::div<unsigned long long>(const unsigned long long&) const;
    template LFS_CORE_API Tensor Tensor::pow<unsigned long long>(const unsigned long long&) const;
    template LFS_CORE_API Tensor Tensor::mod<unsigned long long>(const unsigned long long&) const;
    template LFS_CORE_API Tensor Tensor::maximum<unsigned long long>(const unsigned long long&) const;
    template LFS_CORE_API Tensor Tensor::minimum<unsigned long long>(const unsigned long long&) const;
    template LFS_CORE_API Tensor Tensor::eq<unsigned long long>(const unsigned long long&) const;
    template LFS_CORE_API Tensor Tensor::ne<unsigned long long>(const unsigned long long&) const;
    template LFS_CORE_API Tensor Tensor::lt<unsigned long long>(const unsigned long long&) const;
    template LFS_CORE_API Tensor Tensor::le<unsigned long long>(const unsigned long long&) const;
    template LFS_CORE_API Tensor Tensor::gt<unsigned long long>(const unsigned long long&) const;
    template LFS_CORE_API Tensor Tensor::ge<unsigned long long>(const unsigned long long&) const;
    template LFS_CORE_API Tensor& Tensor::add_<unsigned long long>(const unsigned long long&);
    template LFS_CORE_API Tensor& Tensor::sub_<unsigned long long>(const unsigned long long&);
    template LFS_CORE_API Tensor& Tensor::mul_<unsigned long long>(const unsigned long long&);
    template LFS_CORE_API Tensor& Tensor::div_<unsigned long long>(const unsigned long long&);
    template LFS_CORE_API Tensor Tensor::add<short>(const short&) const;
    template LFS_CORE_API Tensor Tensor::sub<short>(const short&) const;
    template LFS_CORE_API Tensor Tensor::mul<short>(const short&) const;
    template LFS_CORE_API Tensor Tensor::div<short>(const short&) const;
    template LFS_CORE_API Tensor Tensor::pow<short>(const short&) const;
    template LFS_CORE_API Tensor Tensor::mod<short>(const short&) const;
    template LFS_CORE_API Tensor Tensor::maximum<short>(const short&) const;
    template LFS_CORE_API Tensor Tensor::minimum<short>(const short&) const;
    template LFS_CORE_API Tensor Tensor::eq<short>(const short&) const;
    template LFS_CORE_API Tensor Tensor::ne<short>(const short&) const;
    template LFS_CORE_API Tensor Tensor::lt<short>(const short&) const;
    template LFS_CORE_API Tensor Tensor::le<short>(const short&) const;
    template LFS_CORE_API Tensor Tensor::gt<short>(const short&) const;
    template LFS_CORE_API Tensor Tensor::ge<short>(const short&) const;
    template LFS_CORE_API Tensor& Tensor::add_<short>(const short&);
    template LFS_CORE_API Tensor& Tensor::sub_<short>(const short&);
    template LFS_CORE_API Tensor& Tensor::mul_<short>(const short&);
    template LFS_CORE_API Tensor& Tensor::div_<short>(const short&);
    template LFS_CORE_API Tensor Tensor::add<unsigned short>(const unsigned short&) const;
    template LFS_CORE_API Tensor Tensor::sub<unsigned short>(const unsigned short&) const;
    template LFS_CORE_API Tensor Tensor::mul<unsigned short>(const unsigned short&) const;
    template LFS_CORE_API Tensor Tensor::div<unsigned short>(const unsigned short&) const;
    template LFS_CORE_API Tensor Tensor::pow<unsigned short>(const unsigned short&) const;
    template LFS_CORE_API Tensor Tensor::mod<unsigned short>(const unsigned short&) const;
    template LFS_CORE_API Tensor Tensor::maximum<unsigned short>(const unsigned short&) const;
    template LFS_CORE_API Tensor Tensor::minimum<unsigned short>(const unsigned short&) const;
    template LFS_CORE_API Tensor Tensor::eq<unsigned short>(const unsigned short&) const;
    template LFS_CORE_API Tensor Tensor::ne<unsigned short>(const unsigned short&) const;
    template LFS_CORE_API Tensor Tensor::lt<unsigned short>(const unsigned short&) const;
    template LFS_CORE_API Tensor Tensor::le<unsigned short>(const unsigned short&) const;
    template LFS_CORE_API Tensor Tensor::gt<unsigned short>(const unsigned short&) const;
    template LFS_CORE_API Tensor Tensor::ge<unsigned short>(const unsigned short&) const;
    template LFS_CORE_API Tensor& Tensor::add_<unsigned short>(const unsigned short&);
    template LFS_CORE_API Tensor& Tensor::sub_<unsigned short>(const unsigned short&);
    template LFS_CORE_API Tensor& Tensor::mul_<unsigned short>(const unsigned short&);
    template LFS_CORE_API Tensor& Tensor::div_<unsigned short>(const unsigned short&);
    template LFS_CORE_API Tensor Tensor::add<unsigned char>(const unsigned char&) const;
    template LFS_CORE_API Tensor Tensor::sub<unsigned char>(const unsigned char&) const;
    template LFS_CORE_API Tensor Tensor::mul<unsigned char>(const unsigned char&) const;
    template LFS_CORE_API Tensor Tensor::div<unsigned char>(const unsigned char&) const;
    template LFS_CORE_API Tensor Tensor::pow<unsigned char>(const unsigned char&) const;
    template LFS_CORE_API Tensor Tensor::mod<unsigned char>(const unsigned char&) const;
    template LFS_CORE_API Tensor Tensor::maximum<unsigned char>(const unsigned char&) const;
    template LFS_CORE_API Tensor Tensor::minimum<unsigned char>(const unsigned char&) const;
    template LFS_CORE_API Tensor Tensor::eq<unsigned char>(const unsigned char&) const;
    template LFS_CORE_API Tensor Tensor::ne<unsigned char>(const unsigned char&) const;
    template LFS_CORE_API Tensor Tensor::lt<unsigned char>(const unsigned char&) const;
    template LFS_CORE_API Tensor Tensor::le<unsigned char>(const unsigned char&) const;
    template LFS_CORE_API Tensor Tensor::gt<unsigned char>(const unsigned char&) const;
    template LFS_CORE_API Tensor Tensor::ge<unsigned char>(const unsigned char&) const;
    template LFS_CORE_API Tensor& Tensor::add_<unsigned char>(const unsigned char&);
    template LFS_CORE_API Tensor& Tensor::sub_<unsigned char>(const unsigned char&);
    template LFS_CORE_API Tensor& Tensor::mul_<unsigned char>(const unsigned char&);
    template LFS_CORE_API Tensor& Tensor::div_<unsigned char>(const unsigned char&);
    template LFS_CORE_API Tensor Tensor::add<signed char>(const signed char&) const;
    template LFS_CORE_API Tensor Tensor::sub<signed char>(const signed char&) const;
    template LFS_CORE_API Tensor Tensor::mul<signed char>(const signed char&) const;
    template LFS_CORE_API Tensor Tensor::div<signed char>(const signed char&) const;
    template LFS_CORE_API Tensor Tensor::pow<signed char>(const signed char&) const;
    template LFS_CORE_API Tensor Tensor::mod<signed char>(const signed char&) const;
    template LFS_CORE_API Tensor Tensor::maximum<signed char>(const signed char&) const;
    template LFS_CORE_API Tensor Tensor::minimum<signed char>(const signed char&) const;
    template LFS_CORE_API Tensor Tensor::eq<signed char>(const signed char&) const;
    template LFS_CORE_API Tensor Tensor::ne<signed char>(const signed char&) const;
    template LFS_CORE_API Tensor Tensor::lt<signed char>(const signed char&) const;
    template LFS_CORE_API Tensor Tensor::le<signed char>(const signed char&) const;
    template LFS_CORE_API Tensor Tensor::gt<signed char>(const signed char&) const;
    template LFS_CORE_API Tensor Tensor::ge<signed char>(const signed char&) const;
    template LFS_CORE_API Tensor& Tensor::add_<signed char>(const signed char&);
    template LFS_CORE_API Tensor& Tensor::sub_<signed char>(const signed char&);
    template LFS_CORE_API Tensor& Tensor::mul_<signed char>(const signed char&);
    template LFS_CORE_API Tensor& Tensor::div_<signed char>(const signed char&);
    template LFS_CORE_API Tensor& Tensor::add_<Tensor>(const Tensor&);
    template LFS_CORE_API Tensor& Tensor::sub_<Tensor>(const Tensor&);
    template LFS_CORE_API Tensor& Tensor::mul_<Tensor>(const Tensor&);
    template LFS_CORE_API Tensor& Tensor::div_<Tensor>(const Tensor&);
    template LFS_CORE_API float Tensor::item<float>() const;
    template LFS_CORE_API double Tensor::item<double>() const;
    template LFS_CORE_API int Tensor::item<int>() const;
    template LFS_CORE_API long Tensor::item<long>() const;
    template LFS_CORE_API long long Tensor::item<long long>() const;
    template LFS_CORE_API unsigned char Tensor::item<unsigned char>() const;
    template LFS_CORE_API bool Tensor::item<bool>() const;

} // namespace lfs::core
