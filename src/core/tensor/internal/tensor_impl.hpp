/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor_fwd.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <concepts>
#include <cstring>
#include <cuda_runtime.h>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "lazy_config.hpp"
#include "lazy_executor.hpp"
#include "lazy_ir.hpp"
#include "tensor_functors.hpp"
#include "tensor_ops.hpp"

#include "core/export.hpp"

namespace lfs::core {

    class TensorError;
    class TensorIndexer;
    class MaskedTensorProxy;
    class TensorRowProxy;

    // ============================================================================
    // Type Promotion System
    // ============================================================================
    // Determines the result dtype for binary operations between different types.
    // Follows PyTorch/NumPy conventions:
    //   - Bool promotes to any numeric type
    //   - Integer promotes to Float
    //   - Smaller types promote to larger types
    //   - Float16 + Float32 → Float32
    // ============================================================================

    constexpr DataType promote_dtypes(DataType lhs, DataType rhs) {
        // Same types - no promotion needed
        if (lhs == rhs)
            return lhs;

        // Bool promotes to any other type
        if (lhs == DataType::Bool)
            return rhs;
        if (rhs == DataType::Bool)
            return lhs;

        // Type promotion table for different type combinations
        // Order of precedence: Float32 > Float16 > Int64 > Int32 > UInt8

        // Float32 is the highest - anything with Float32 becomes Float32
        if (lhs == DataType::Float32 || rhs == DataType::Float32) {
            return DataType::Float32;
        }

        // Float16 with any integer becomes Float16
        if (lhs == DataType::Float16 || rhs == DataType::Float16) {
            return DataType::Float16;
        }

        // Int64 is the largest integer type
        if (lhs == DataType::Int64 || rhs == DataType::Int64) {
            return DataType::Int64;
        }

        // Int32 with UInt8 becomes Int32
        if (lhs == DataType::Int32 || rhs == DataType::Int32) {
            return DataType::Int32;
        }

        // Only UInt8 remains
        return DataType::UInt8;
    }

    enum class BoundaryMode : uint8_t {
        Assert = 0,
        Clamp = 1,
        Wrap = 2
    };

    enum class ScatterMode : uint8_t {
        None = 0,
        Add = 1,
        Multiply = 2,
        Max = 3,
        Min = 4
    };

    enum class ReduceOp : uint8_t {
        Sum = 0,
        Mean = 1,
        Max = 2,
        Min = 3,
        Prod = 4,
        Any = 5,
        All = 6,
        Std = 7,
        Var = 8,
        Argmax = 9,
        Argmin = 10,
        CountNonzero = 11,
        Norm = 12
    };

    enum class MovementOp : uint8_t {
        Reshape = 0,
        Permute = 1,
        Expand = 2,
        Pad = 3,
        Shrink = 4,
        Flip = 5,
        Transpose = 6,
        Squeeze = 7,
        Unsqueeze = 8,
        Flatten = 9,
        Cat = 10,
        Stack = 11,
        Slice = 12
    };

    enum class LoadOp : uint8_t {
        Empty = 0,
        Const = 1,
        Arange = 2,
        Random = 3,
        Eye = 4,
        FromCPU = 5,
        FromCUDA = 6,
        Normal = 7,
        Randint = 8,
        Bernoulli = 9,
        Multinomial = 10
    };

    class LFS_CORE_API TensorShape {
    private:
        std::vector<size_t> dims_;
        size_t total_elements_ = 1;

    public:
        TensorShape() = default;
        TensorShape(std::initializer_list<size_t> dims) : dims_(dims) {
            compute_total();
        }
        explicit TensorShape(const std::vector<size_t>& dims) : dims_(dims) {
            compute_total();
        }
        explicit TensorShape(std::span<const size_t> dims) : dims_(dims.begin(), dims.end()) {
            compute_total();
        }

        size_t rank() const { return dims_.size(); }
        size_t operator[](size_t i) const {
            if (i >= dims_.size()) {
                throw std::out_of_range(
                    "Shape index " + std::to_string(i) + " out of range for rank " + std::to_string(dims_.size()));
            }
            return dims_[i];
        }
        size_t elements() const { return total_elements_; }
        const std::vector<size_t>& dims() const { return dims_; }

        // Calculate strides for row-major layout
        std::vector<size_t> strides() const {
            if (dims_.empty())
                return {};

            std::vector<size_t> result(dims_.size());
            result.back() = 1;
            for (int i = static_cast<int>(dims_.size()) - 2; i >= 0; --i) {
                result[i] = result[i + 1] * dims_[i + 1];
            }
            return result;
        }

        bool operator==(const TensorShape& other) const { return dims_ == other.dims_; }
        bool operator!=(const TensorShape& other) const { return !(*this == other); }

        std::string str() const;

    private:
        void compute_total() {
            if (dims_.empty()) {
                total_elements_ = 1;
            } else {
                total_elements_ = 1;
                for (auto d : dims_) {
                    total_elements_ *= d;
                }
            }
        }
    };

    struct MovementArgs {
        std::variant<
            std::monostate,
            std::vector<int>,
            std::pair<int, int>,
            std::vector<std::pair<int, int>>,
            int,
            void*,
            std::pair<void*, int>>
            args;
    };

    struct LoadArgs {
        TensorShape shape;
        Device device = Device::CUDA;
        DataType dtype = DataType::Float32;
        bool use_pinned = true;
        std::variant<
            std::monostate,
            float,
            std::tuple<float, float, float>,
            std::pair<float, float>,
            std::pair<int, int>,
            void*,
            std::pair<void*, bool>>
            args;
    };

    struct ReduceArgs {
        std::vector<int> axes;
        bool keepdim = false;
        bool unbiased = true;
        std::variant<
            std::monostate,
            float>
            args;
    };

    class LFS_CORE_API RandomGenerator {
    public:
        static RandomGenerator& instance();
        void manual_seed(uint64_t seed);
        uint64_t get_seed() const { return seed_; }
        void* get_generator(Device device);
        uint64_t get_next_cuda_seed();
        uint64_t get_next_cuda_offset(); // Get next offset for cuRAND generator
        void* get_impl() { return impl_; }
        const void* get_impl() const { return impl_; }

    private:
        RandomGenerator();
        ~RandomGenerator();
        uint64_t seed_;
        void* impl_ = nullptr;
        std::mt19937_64 cpu_generator_;
        RandomGenerator(const RandomGenerator&) = delete;
        RandomGenerator& operator=(const RandomGenerator&) = delete;
    };

    struct StorageMeta {
        std::atomic<uint64_t> generation{0};
    };

} // namespace lfs::core

// Include expression template declarations (forward declarations only)
#include "tensor_expr.hpp"

namespace lfs::core {

    class LFS_CORE_API Tensor {
    private:
        struct TensorState {
            // Capacity management for in-place growth (like std::vector)
            size_t capacity = 0;
            size_t logical_size = 0;

            // Cached alignment flags (computed once on allocation)
            bool is_aligned_16 = false;  // 16-byte alignment for float4 vectorization
            bool is_aligned_128 = false; // 128-byte alignment for cache line optimization

            // CUDA stream for async execution (assigned round-robin from StreamPool)
            cudaStream_t stream = nullptr;

            // Debug tracking - when true, operations on this tensor are logged
            bool tracked = false;
            std::string name; // Optional name for identification in traces

            // Deferred expression materialization (lazy mode = on)
            bool has_deferred_expr = false;
            bool materializing_deferred_expr = false;
            uint64_t deferred_expr_node_id = 0;
            std::function<Tensor()> deferred_materializer;
        };

        void* data_ = nullptr;
        std::shared_ptr<void> data_owner_;
        std::shared_ptr<TensorState> state_ = std::make_shared<TensorState>();
        TensorShape shape_;
        std::vector<size_t> strides_; // Stride for each dimension (in elements)
        size_t storage_offset_ = 0;   // Offset from data_ (in elements)
        bool is_contiguous_ = true;   // True if memory layout is C-contiguous
        Device device_ = Device::CPU;
        DataType dtype_ = DataType::Float32;
        bool is_view_ = false;

        std::shared_ptr<StorageMeta> storage_meta_;
        uint64_t view_generation_snapshot_ = 0;

        mutable size_t id_ = 0;
        static std::atomic<size_t> next_id_;
        static inline bool profiling_enabled_ = false;

        void materialize_if_deferred();
        void materialize_if_deferred() const {
            const_cast<Tensor*>(this)->materialize_if_deferred();
        }

        static Tensor make_deferred_expr_tensor(TensorShape shape,
                                                Device device,
                                                DataType dtype,
                                                std::function<Tensor()> materializer,
                                                std::vector<uint64_t> lazy_input_ids = {});

        // Compute alignment flags for vectorization
        void compute_alignment() {
            if (data_ != nullptr) {
                auto addr = reinterpret_cast<uintptr_t>(data_);
                state_->is_aligned_16 = (addr % 16) == 0;
                state_->is_aligned_128 = (addr % 128) == 0;
            } else {
                state_->is_aligned_16 = false;
                state_->is_aligned_128 = false;
            }
        }

        void init_storage_meta() {
            storage_meta_ = std::make_shared<StorageMeta>();
        }

        void ensure_storage_meta() {
            if (!storage_meta_) {
                storage_meta_ = std::make_shared<StorageMeta>();
            }
        }

        void bump_storage_generation() {
            if (storage_meta_) {
                storage_meta_->generation.fetch_add(1, std::memory_order_relaxed);
            }
        }

        void assert_view_not_stale() const {
            if (is_view_ && storage_meta_ &&
                view_generation_snapshot_ != storage_meta_->generation.load(std::memory_order_relaxed)) {
                throw std::runtime_error("Attempted to access a stale tensor view after storage reallocation");
            }
        }

        void propagate_view_meta(Tensor& view) const {
            const_cast<Tensor*>(this)->ensure_storage_meta();
            view.storage_meta_ = storage_meta_;
            view.state_->stream = state_->stream;
            view.view_generation_snapshot_ =
                storage_meta_->generation.load(std::memory_order_relaxed);
        }

        // Generic functor-based binary operation (zero enum overhead)
        template <typename SrcT, typename OutT, typename Op>
        Tensor binary_op_generic(const Tensor& other, Op op) const {
            validate_binary_op(other, false, true);

            auto broadcast_shape = this->broadcast_shape(other.shape());
            if (broadcast_shape.rank() == 0) {
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

            auto result = Tensor::empty(broadcast_shape, device_, out_dtype);

            bool a_needs_broadcast = (shape_ != broadcast_shape);
            bool b_needs_broadcast = (other.shape() != broadcast_shape);

            if (!a_needs_broadcast && !b_needs_broadcast) {
                // Element-wise operation without broadcasting
                if (device_ == Device::CUDA) {
                    tensor_ops::launch_binary_op_generic(
                        ptr<SrcT>(), other.ptr<SrcT>(), result.ptr<OutT>(),
                        result.numel(), op, result.stream());
                    // No sync - tensor operation
                } else {
                    apply_binary_cpu(ptr<SrcT>(), other.ptr<SrcT>(), result.ptr<OutT>(),
                                     result.numel(), op);
                }
            } else {
                // Broadcasting needed
                auto a_shape = shape_.dims();
                auto b_shape = other.shape().dims();
                auto c_shape = broadcast_shape.dims();

                if (device_ == Device::CUDA) {
                    tensor_ops::launch_broadcast_binary(
                        ptr<SrcT>(), other.ptr<SrcT>(), result.ptr<OutT>(),
                        a_shape.data(), b_shape.data(), c_shape.data(),
                        a_shape.size(), b_shape.size(), c_shape.size(),
                        result.numel(), op, result.stream());
                    // No sync - tensor operation
                } else {
                    // CPU broadcasting: materialize broadcasts first
                    auto a_broadcast = a_needs_broadcast ? broadcast_to(broadcast_shape) : clone();
                    auto b_broadcast = b_needs_broadcast ? other.broadcast_to(broadcast_shape) : other.clone();
                    apply_binary_cpu(a_broadcast.ptr<SrcT>(), b_broadcast.ptr<SrcT>(),
                                     result.ptr<OutT>(), result.numel(), op);
                }
            }

            return result;
        }

        // Generic functor-based scalar operation (zero enum overhead)
        template <typename Op>
        Tensor scalar_op_generic(float scalar, Op op, DataType out_dtype = DataType::Float32) const {
            validate_unary_op();

            auto result = Tensor::empty(shape_, device_, out_dtype);

            if (device_ == Device::CUDA) {
                // Handle different input tensor dtypes
                if (dtype_ == DataType::Int32) {
                    int scalar_int = static_cast<int>(scalar);
                    if (out_dtype == DataType::Bool) {
                        tensor_ops::launch_scalar_op_generic(
                            ptr<int>(), scalar_int, result.ptr<unsigned char>(),
                            numel(), op, result.stream());
                    } else if (out_dtype == DataType::Int32) {
                        tensor_ops::launch_scalar_op_generic(
                            ptr<int>(), scalar_int, result.ptr<int>(),
                            numel(), op, result.stream());
                    }
                } else { // Float32
                    if (out_dtype == DataType::Bool) {
                        tensor_ops::launch_scalar_op_generic(
                            ptr<float>(), scalar, result.ptr<unsigned char>(),
                            numel(), op, result.stream());
                    } else {
                        tensor_ops::launch_scalar_op_generic(
                            ptr<float>(), scalar, result.ptr<float>(),
                            numel(), op, result.stream());
                    }
                }
                // No sync needed - operations are async
            } else {
                // CPU implementation
                if (dtype_ == DataType::Int32) {
                    const int* src = ptr<int>();
                    int scalar_int = static_cast<int>(scalar);
                    if (out_dtype == DataType::Bool) {
                        unsigned char* dst = result.ptr<unsigned char>();
                        for (size_t i = 0; i < numel(); ++i) {
                            dst[i] = op(src[i], scalar_int);
                        }
                    } else {
                        int* dst = result.ptr<int>();
                        for (size_t i = 0; i < numel(); ++i) {
                            dst[i] = op(src[i], scalar_int);
                        }
                    }
                } else { // Float32
                    const float* src = ptr<float>();
                    if (out_dtype == DataType::Bool) {
                        unsigned char* dst = result.ptr<unsigned char>();
                        for (size_t i = 0; i < numel(); ++i) {
                            dst[i] = op(src[i], scalar);
                        }
                    } else {
                        float* dst = result.ptr<float>();
                        apply_unary_cpu(src, dst, numel(), ops::scalar_right_op<Op, float>(scalar));
                    }
                }
            }

            return result;
        }

        // Generic functor-based in-place scalar operation (zero enum overhead)
        template <typename Op>
        Tensor& scalar_op_inplace_generic(float scalar, Op op) {
            validate_unary_op();

            if (device_ == Device::CUDA) {
                tensor_ops::launch_scalar_op_generic(
                    ptr<float>(), scalar, ptr<float>(),
                    numel(), op, stream());
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
        template <typename SrcT = float, typename Op>
        Tensor& binary_op_inplace_generic(const Tensor& other, Op op) {
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
                    (device_ == Device::CUDA ? "CUDA" : "CPU") + " vs " +
                    (other.device() == Device::CUDA ? "CUDA" : "CPU"));
            }

            if (device_ == Device::CUDA) {
                tensor_ops::launch_binary_op_generic(
                    ptr<SrcT>(), other.ptr<SrcT>(), ptr<SrcT>(),
                    numel(), op, stream());
                // No sync - tensor operation
            } else {
                // CPU implementation
                apply_binary_cpu(ptr<SrcT>(), other.ptr<SrcT>(), ptr<SrcT>(),
                                 numel(), op);
            }

            return *this;
        }

        std::pair<Tensor, Tensor> _broadcasted(const Tensor& other, bool match_dtype = true) const;

        int resolve_dim(int dim) const {
            if (!is_valid())
                return -1;
            return dim < 0 ? static_cast<int>(shape_.rank()) + dim : dim;
        }

        // Validation helpers - throw on error
        void validate_binary_op(const Tensor& other, bool require_same_shape = false, bool require_same_device = false) const {
            if (!is_valid() || !other.is_valid()) {
                throw std::runtime_error("Binary operation on invalid tensor");
            }
            if (require_same_device && device_ != other.device()) {
                throw std::runtime_error("Tensors must be on same device");
            }
            if (require_same_shape && shape_ != other.shape()) {
                throw std::runtime_error("Shape mismatch: " + shape_.str() + " vs " + other.shape_.str());
            }
            if (!require_same_shape && shape_ != other.shape()) {
                const auto& a = shape_.dims();
                const auto& b = other.shape_.dims();
                size_t max_rank = std::max(a.size(), b.size());
                bool compatible = true;
                for (size_t i = 0; i < max_rank && compatible; ++i) {
                    size_t da = (i < a.size()) ? a[a.size() - 1 - i] : 1;
                    size_t db = (i < b.size()) ? b[b.size() - 1 - i] : 1;
                    if (da == 0 && db == 0)
                        continue;
                    compatible = (da == db || da == 1 || db == 1) && da != 0 && db != 0;
                }
                if (!compatible) {
                    throw std::runtime_error("Incompatible shapes for broadcasting: " + shape_.str() + " vs " + other.shape_.str());
                }
            }
        }

        // Helper for binary operations with automatic type promotion
        // Promotes types, converts operands if needed, and creates BinaryExpr
        template <typename Op>
        Tensor binary_op_with_promotion(const Tensor& other, Op op) const {
            validate_binary_op(other, false, true);

            // Determine promoted dtype for the result
            DataType result_dtype = promote_dtypes(dtype_, other.dtype());

            // Convert operands to result dtype if needed
            const Tensor& lhs = (dtype_ == result_dtype) ? *this : this->to(result_dtype);
            const Tensor& rhs = (other.dtype() == result_dtype) ? other : other.to(result_dtype);

            // Compute broadcast shape
            auto broadcast_shape = lhs.broadcast_shape(rhs.shape());

            // Create and return the binary expression with promoted dtype
            Tensor result = BinaryExpr<TensorLeaf, TensorLeaf, Op>(
                TensorLeaf(lhs), TensorLeaf(rhs), op,
                broadcast_shape, lhs.device(), result_dtype);
            link_deferred_result_to_inputs(result, {lhs.lazy_expr_id(), rhs.lazy_expr_id()});
            return result;
        }

        // Helper for comparison operations with automatic type promotion
        // Promotes operand types for comparison, but always returns Bool
        template <typename Op>
        Tensor comparison_op_with_promotion(const Tensor& other, Op op) const {
            validate_binary_op(other, false, true);

            // Promote operand types for comparison
            DataType compare_dtype = promote_dtypes(dtype_, other.dtype());

            // Convert operands to common dtype for comparison
            const Tensor& lhs = (dtype_ == compare_dtype) ? *this : this->to(compare_dtype);
            const Tensor& rhs = (other.dtype() == compare_dtype) ? other : other.to(compare_dtype);

            // Compute broadcast shape
            auto broadcast_shape = lhs.broadcast_shape(rhs.shape());

            // Return Bool tensor (comparison result)
            Tensor result = BinaryExpr<TensorLeaf, TensorLeaf, Op>(
                TensorLeaf(lhs), TensorLeaf(rhs), op,
                broadcast_shape, lhs.device(), DataType::Bool);
            link_deferred_result_to_inputs(result, {lhs.lazy_expr_id(), rhs.lazy_expr_id()});
            return result;
        }

        void validate_unary_op() const {
            if (!is_valid()) {
                throw std::runtime_error("Unary operation on invalid tensor");
            }
        }

        void validate_ternary_op(const Tensor& b, const Tensor& c) const {
            if (!is_valid() || !b.is_valid() || !c.is_valid()) {
                throw std::runtime_error("Ternary operation on invalid tensor");
            }
            if (device_ != b.device() || device_ != c.device()) {
                throw std::runtime_error("All tensors must be on same device");
            }
        }

        // Helper to ensure tensor is on same device
        Tensor ensure_same_device(const Tensor& other) const {
            return (other.device() == device_) ? other : other.to(device_);
        }

        static void link_deferred_result_to_inputs(Tensor& result,
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
        Tensor create_view(const TensorShape& new_shape) const {
            if (state_ && state_->has_deferred_expr) {
                const uint64_t source_id = lazy_expr_id();
                Tensor source = *this;
                TensorShape deferred_shape = new_shape;
                std::vector<uint64_t> deferred_inputs;
                if (source_id != 0) {
                    deferred_inputs.push_back(source_id);
                }
                return make_deferred_expr_tensor(
                    deferred_shape, device_, dtype_,
                    [source = std::move(source), deferred_shape]() mutable {
                        Tensor materialized = source;
                        materialized.materialize_if_deferred();
                        return materialized.create_view(deferred_shape);
                    },
                    std::move(deferred_inputs));
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

        std::vector<size_t> resolve_dims(std::span<const int> dims) const;
        bool is_contiguous_slice(const std::vector<size_t>& starts,
                                 const std::vector<size_t>& ends) const;
        size_t calculate_offset(const std::vector<size_t>& indices) const;
        Tensor copy_slice(const std::vector<size_t>& starts,
                          const std::vector<size_t>& ends,
                          const std::vector<size_t>& new_shape) const;

    public:
        Tensor() = default;
        Tensor(void* data, TensorShape shape, Device device, DataType dtype);

        // Copy constructor and assignment - SHALLOW COPY (LibTorch behavior)
        Tensor(const Tensor& other);
        Tensor& operator=(const Tensor& other);

        // Move constructor and assignment
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;

        ~Tensor();

        // ============= Multi-dimensional accessor =============
        template <typename T, size_t N>
        class TensorAccessor {
        private:
            T* data_;
            std::array<size_t, N> sizes_;
            std::array<size_t, N> strides_;

        public:
            TensorAccessor(T* data, const std::array<size_t, N>& sizes)
                : data_(data),
                  sizes_(sizes) {
                strides_[N - 1] = 1;
                if constexpr (N > 1) {
                    for (size_t i = N - 1; i > 0; --i) {
                        strides_[i - 1] = strides_[i] * sizes_[i];
                    }
                }
            }

            template <typename... Indices>
            T& operator()(Indices... indices) {
                static_assert(sizeof...(Indices) == N, "Wrong number of indices");
                std::array<size_t, N> idx_array{static_cast<size_t>(indices)...};
                size_t offset = 0;
                for (size_t i = 0; i < N; ++i) {
                    offset += idx_array[i] * strides_[i];
                }
                return data_[offset];
            }

            const std::array<size_t, N>& sizes() const { return sizes_; }
        };

        template <typename T, size_t N>
        TensorAccessor<T, N> accessor() {
            if (device_ != Device::CPU) {
                throw std::runtime_error("accessor() only works on CPU tensors");
            }
            if (!is_valid() || shape_.rank() != N) {
                throw std::runtime_error(
                    "accessor() dimension mismatch: tensor has " + std::to_string(shape_.rank()) +
                    " dims, requested " + std::to_string(N));
            }

            if (!is_contiguous()) {
                throw std::runtime_error("accessor() only works on contiguous tensors");
            }

            std::array<size_t, N> sizes;
            for (size_t i = 0; i < N; ++i) {
                sizes[i] = shape_[i];
            }
            return TensorAccessor<T, N>(ptr<T>(), sizes);
        }

        // ============= Array-like indexing operator[] =============
        TensorRowProxy operator[](size_t index);
        const TensorRowProxy operator[](size_t index) const;

        // ============= CORE UNIFIED OPERATIONS =============
        static Tensor load(LoadOp op, const LoadArgs& args);
        Tensor movement(MovementOp op, const MovementArgs& args) const;
        Tensor reduce(ReduceOp op, const ReduceArgs& args = {}) const;
        // Internal helper for where() operation
        Tensor ternary(const Tensor& b, const Tensor& c) const;

        // ============= FACTORY METHODS =============
        static Tensor empty(TensorShape shape, Device device = Device::CUDA,
                            DataType dtype = DataType::Float32, bool use_pinned = true);
        static Tensor empty_unpinned(TensorShape shape, DataType dtype = DataType::Float32);
        static Tensor zeros(TensorShape shape, Device device = Device::CUDA,
                            DataType dtype = DataType::Float32);
        static Tensor zeros_direct(TensorShape shape, size_t capacity, Device device = Device::CUDA);
        static Tensor ones(TensorShape shape, Device device = Device::CUDA,
                           DataType dtype = DataType::Float32);
        static Tensor full(TensorShape shape, float value, Device device = Device::CUDA,
                           DataType dtype = DataType::Float32);
        static Tensor full_bool(TensorShape shape, bool value, Device device = Device::CUDA);
        static Tensor zeros_bool(TensorShape shape, Device device = Device::CUDA);
        static Tensor ones_bool(TensorShape shape, Device device = Device::CUDA);
        static Tensor rand(TensorShape shape, Device device = Device::CUDA,
                           DataType dtype = DataType::Float32);
        static Tensor randn(TensorShape shape, Device device = Device::CUDA,
                            DataType dtype = DataType::Float32);
        static Tensor uniform(TensorShape shape, float low = 0.0f, float high = 1.0f,
                              Device device = Device::CUDA, DataType dtype = DataType::Float32);
        static Tensor normal(TensorShape shape, float mean = 0.0f, float std = 1.0f,
                             Device device = Device::CUDA, DataType dtype = DataType::Float32);
        static Tensor randint(TensorShape shape, int low, int high,
                              Device device = Device::CUDA, DataType dtype = DataType::Int32);
        static Tensor bernoulli(TensorShape shape, float p = 0.5f,
                                Device device = Device::CUDA, DataType dtype = DataType::Float32);
        static Tensor multinomial(const Tensor& weights, int num_samples,
                                  bool replacement = false);
        static Tensor arange(float end);
        static Tensor arange(float start, float end, float step = 1.0f);
        static Tensor linspace(float start, float end, size_t steps, Device device = Device::CUDA);
        static Tensor eye(size_t n, Device device = Device::CUDA);
        static Tensor eye(size_t m, size_t n, Device device = Device::CUDA);
        static Tensor diag(const Tensor& diagonal);

        static Tensor from_blob(void* data, TensorShape shape, Device device, DataType dtype) {
            return Tensor(data, shape, device, dtype);
        }

        static Tensor from_vector(const std::vector<float>& data, TensorShape shape,
                                  Device device = Device::CUDA);
        static Tensor from_vector(const std::vector<int>& data, TensorShape shape,
                                  Device device = Device::CUDA);
        static Tensor from_vector(const std::vector<bool>& data, TensorShape shape,
                                  Device device = Device::CUDA);

        // Initializer list overloads for convenience
        static Tensor from_vector(std::initializer_list<float> data, TensorShape shape,
                                  Device device = Device::CUDA) {
            return from_vector(std::vector<float>(data), shape, device);
        }

        static Tensor from_vector(std::initializer_list<int> data, TensorShape shape,
                                  Device device = Device::CUDA) {
            return from_vector(std::vector<int>(data), shape, device);
        }

        static Tensor from_vector(std::initializer_list<bool> data, TensorShape shape,
                                  Device device = Device::CUDA) {
            return from_vector(std::vector<bool>(data), shape, device);
        }

        // ============= LIKE OPERATIONS =============
        static Tensor zeros_like(const Tensor& other) {
            return zeros(other.shape(), other.device(), other.dtype());
        }

        static Tensor ones_like(const Tensor& other) {
            return ones(other.shape(), other.device(), other.dtype());
        }

        static Tensor ones_like(const Tensor& other, DataType dtype) {
            return ones(other.shape(), other.device(), dtype);
        }

        static Tensor rand_like(const Tensor& other) {
            return rand(other.shape(), other.device(), other.dtype());
        }

        static Tensor randn_like(const Tensor& other) {
            return randn(other.shape(), other.device(), other.dtype());
        }

        static Tensor empty_like(const Tensor& other) {
            auto result = empty(other.shape(), other.device(), other.dtype());
            result.set_stream(other.stream());
            return result;
        }

        static Tensor full_like(const Tensor& other, float value) {
            auto result = full(other.shape(), value, other.device(), other.dtype());
            result.set_stream(other.stream());
            return result;
        }

        // ============= COMBINING TENSORS =============
        static Tensor cat(const std::vector<Tensor>& tensors, int dim = 0);
        static Tensor stack(const std::vector<Tensor>& tensors, int dim = 0);

        // ============= CONDITIONAL =============
        static Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y);

        // ============= GLOBAL CONFIGURATION =============
        static void manual_seed(uint64_t seed) {
            RandomGenerator::instance().manual_seed(seed);
        }

        static void enable_profiling(bool enable) { profiling_enabled_ = enable; }
        static LazyTelemetrySnapshot lazy_telemetry_snapshot() {
            return internal::lazy_telemetry_snapshot();
        }
        static void reset_lazy_telemetry() {
            internal::reset_lazy_telemetry();
        }
        static void clear_lazy_ir_for_testing() {
            internal::clear_lazy_ir_for_testing();
        }

        static void trim_memory_pool();
        static void shutdown_memory_pool();
        static void set_memory_pool_iteration(int iteration);
        static void print_memory_pool_stats();

        void set_bool(std::initializer_list<size_t> indices, bool value);
        bool get_bool(std::initializer_list<size_t> indices) const;
        void set_bool(std::span<const size_t> indices, bool value);
        bool get_bool(std::span<const size_t> indices) const;

        // Data access - FIXED: Handle invalid tensors safely
        template <typename T>
        T* ptr() {
            materialize_if_deferred();
            if (!is_valid()) {
                return nullptr;
            }
            assert_view_not_stale();
            char* data_ptr = static_cast<char*>(data_) + storage_offset_ * dtype_size(dtype_);
            return static_cast<T*>(static_cast<void*>(data_ptr));
        }

        template <typename T>
        const T* ptr() const {
            materialize_if_deferred();
            if (!is_valid()) {
                return nullptr;
            }
            assert_view_not_stale();
            const char* data_ptr = static_cast<const char*>(data_) + storage_offset_ * dtype_size(dtype_);
            return static_cast<const T*>(static_cast<const void*>(data_ptr));
        }

        void* data_ptr() {
            materialize_if_deferred();
            if (!is_valid()) {
                return nullptr;
            }
            assert_view_not_stale();
            return static_cast<char*>(data_) + storage_offset_ * dtype_size(dtype_);
        }
        const void* data_ptr() const {
            materialize_if_deferred();
            if (!is_valid()) {
                return nullptr;
            }
            assert_view_not_stale();
            return static_cast<const char*>(data_) + storage_offset_ * dtype_size(dtype_);
        }

        // Base of allocation (for memory management only)
        void* storage_ptr() {
            materialize_if_deferred();
            if (!is_valid()) {
                return nullptr;
            }
            return data_;
        }
        const void* storage_ptr() const {
            materialize_if_deferred();
            if (!is_valid()) {
                return nullptr;
            }
            return data_;
        }

        // Properties - FIXED: Check validity before accessing shape
        const TensorShape& shape() const { return shape_; }
        Device device() const { return device_; }
        DataType dtype() const { return dtype_; }
        bool owns_memory() const { return static_cast<bool>(data_owner_) && !is_view_; }
        bool is_view() const { return is_view_; }
        bool is_empty() const { return !is_valid() || numel() == 0; }
        bool has_lazy_expr() const {
            return (state_ && state_->has_deferred_expr) || internal::tensor_has_lazy_expr(*this);
        }
        uint64_t lazy_expr_id() const {
            if (state_ && state_->has_deferred_expr && state_->deferred_expr_node_id != 0) {
                return state_->deferred_expr_node_id;
            }
            return internal::tensor_lazy_expr_id(*this);
        }
        std::optional<internal::LazyExprDebugInfo> lazy_expr_info() const {
            if (const uint64_t node_id = lazy_expr_id(); node_id != 0) {
                return internal::lazy_ir_node_info(node_id);
            }
            return std::nullopt;
        }
        size_t debug_id() const { return id_; }

        bool is_valid() const {
            return static_cast<bool>(data_owner_) || is_view_ || (state_ && state_->has_deferred_expr);
        }

        // CRITICAL: All size queries must check validity first
        size_t numel() const {
            return is_valid() ? shape_.elements() : 0;
        }

        size_t bytes() const {
            return numel() * dtype_size(dtype_);
        }

        size_t ndim() const {
            return is_valid() ? shape_.rank() : 0;
        }

        // Alignment accessors (cached flags computed on allocation)
        bool is_aligned_16() const { return state_->is_aligned_16; }
        bool is_aligned_128() const { return state_->is_aligned_128; }

        // Stream accessor (for async CUDA operations)
        cudaStream_t stream() const { return state_->stream; }
        void set_stream(cudaStream_t stream) { state_->stream = stream; }

        // Debug tracking - mark tensor to trace all operations it's involved in
        bool is_tracked() const { return state_->tracked; }
        Tensor& set_tracked(bool tracked = true) {
            state_->tracked = tracked;
            return *this;
        }
        Tensor& track() { return set_tracked(true); } // Convenience alias
        Tensor& untrack() { return set_tracked(false); }

        // Optional name for identifying tensors in traces
        const std::string& name() const { return state_->name; }
        Tensor& set_name(std::string name) {
            state_->name = std::move(name);
            return *this;
        }

        size_t size(size_t dim) const {
            if (!is_valid())
                return 0;
            if (dim >= shape_.rank()) {
                throw std::out_of_range(
                    "Dimension " + std::to_string(dim) + " out of range for rank " + std::to_string(shape_.rank()));
            }
            return shape_[dim];
        }

        // Capacity management (for in-place growth like std::vector)
        // capacity() returns the reserved capacity along dimension 0 (0 = no reservation)
        // logical_size() returns the logical size along dimension 0 (same as shape()[0])
        size_t capacity() const { return state_->capacity; }
        size_t logical_size() const { return state_->logical_size; }

        // reserve() pre-allocates memory for future growth along dimension 0
        // Supports multi-dimensional tensors: [N, D1, D2, ...] reserves N "rows"
        void reserve(size_t new_capacity);

        // Memory operations
        Tensor clone() const;      // Deep copy
        Tensor contiguous() const; // Materialize to contiguous if strided
        Tensor to(Device device, cudaStream_t stream = nullptr) const;
        Tensor to(DataType dtype) const;
        bool is_contiguous() const { return is_contiguous_; }

        // Stride operations (Phase 4: Zero-copy views)
        const std::vector<size_t>& strides() const { return strides_; }
        size_t stride(size_t dim) const {
            if (dim >= strides_.size())
                return 0;
            return strides_[dim];
        }
        size_t storage_offset() const { return storage_offset_; }

        Tensor cpu() const { return to(Device::CPU); }
        Tensor cuda() const { return to(Device::CUDA); }

        // ============= SHAPE OPERATIONS =============
        Tensor reshape(std::span<const int> sizes) const {
            MovementArgs args;
            args.args = std::vector<int>(sizes.begin(), sizes.end());
            return movement(MovementOp::Reshape, args);
        }
        Tensor reshape(std::initializer_list<int> sizes) const {
            return reshape(std::span<const int>(sizes));
        }
        Tensor reshape(TensorShape new_shape) const;

        Tensor view(std::span<const int> sizes) const { return reshape(sizes); }
        Tensor view(std::initializer_list<int> sizes) const { return reshape(sizes); }
        Tensor view(TensorShape new_shape) const { return reshape(new_shape); }

        Tensor squeeze(std::optional<int> dim = std::nullopt) const {
            MovementArgs args;
            args.args = dim.value_or(std::numeric_limits<int>::min());
            return movement(MovementOp::Squeeze, args);
        }

        Tensor squeeze(int dim) const {
            MovementArgs args;
            args.args = dim;
            return movement(MovementOp::Squeeze, args);
        }

        Tensor unsqueeze(int dim) const {
            MovementArgs args;
            args.args = dim;
            return movement(MovementOp::Unsqueeze, args);
        }

        Tensor expand(std::span<const int> sizes) const {
            MovementArgs args;
            args.args = std::vector<int>(sizes.begin(), sizes.end());
            return movement(MovementOp::Expand, args);
        }
        Tensor expand(std::initializer_list<int> sizes) const {
            return expand(std::span<const int>(sizes));
        }
        Tensor expand(const TensorShape& target_shape) const;

        Tensor flatten(int start_dim = 0, int end_dim = -1) const {
            MovementArgs args;
            args.args = std::pair<int, int>{start_dim, end_dim};
            return movement(MovementOp::Flatten, args);
        }

        Tensor permute(std::span<const int> axes) const;
        Tensor permute(std::initializer_list<int> axes) const {
            return permute(std::span<const int>(axes));
        }

        Tensor transpose(int dim1 = -2, int dim2 = -1) const {
            MovementArgs args;
            args.args = std::pair<int, int>{dim1, dim2};
            return movement(MovementOp::Transpose, args);
        }
        Tensor t() const;

        Tensor slice(std::span<const std::pair<int, int>> ranges) const;
        Tensor slice(std::initializer_list<std::pair<int, int>> ranges) const {
            return slice(std::span<const std::pair<int, int>>(ranges));
        }
        Tensor slice(size_t dim, size_t start, size_t end) const;

        Tensor cat(const Tensor& other, int dim = 0) const;

        // Broadcasting
        Tensor broadcast_to(const TensorShape& target_shape) const;
        bool can_broadcast_to(const TensorShape& target) const;
        TensorShape broadcast_shape(const TensorShape& other) const;

        // ============= UNARY OPERATIONS (LAZY EVALUATION) =============
        // Macro to define unary operations with lazy evaluation via expression templates
#define LFS_DEFINE_UNARY_OP(name, op_type)                               \
    Tensor name() const {                                                \
        if (!is_valid() || numel() == 0) {                               \
            if (!is_valid())                                             \
                return Tensor();                                         \
            return Tensor::empty(shape_, device_, dtype_);               \
        }                                                                \
        Tensor result = UnaryExpr<TensorLeaf, ops::op_type>(             \
            TensorLeaf(*this), ops::op_type{}, shape_, device_, dtype_); \
        link_deferred_result_to_inputs(result, {lazy_expr_id()});        \
        return result;                                                   \
    }

#define LFS_DEFINE_UNARY_OP_FUSABLE(name, op_type, fusion_kind)                           \
    Tensor name() const {                                                                 \
        if (!is_valid() || numel() == 0) {                                                \
            if (!is_valid())                                                              \
                return Tensor();                                                          \
            return Tensor::empty(shape_, device_, dtype_);                                \
        }                                                                                 \
        Tensor result = UnaryExpr<TensorLeaf, ops::op_type>(                              \
            TensorLeaf(*this), ops::op_type{}, shape_, device_, dtype_);                  \
        link_deferred_result_to_inputs(result, {lazy_expr_id()});                         \
        if (dtype_ == DataType::Float32 &&                                                \
            result.is_valid() && result.has_lazy_expr()) {                                \
            const uint64_t result_node_id = result.lazy_expr_id();                        \
            if (result_node_id != 0 && result.state_) {                                   \
                internal::lazy_executor_register_pointwise_fusion_op(                     \
                    result_node_id,                                                       \
                    lazy_expr_id(),                                                       \
                    *this,                                                                \
                    internal::LazyPointwiseOp{internal::LazyPointwiseOpKind::fusion_kind, \
                                              0.0f},                                      \
                    std::weak_ptr<void>(result.state_));                                  \
            }                                                                             \
        }                                                                                 \
        return result;                                                                    \
    }

        // Macro for unary ops that return Bool dtype (isnan, isinf, etc.)
#define LFS_DEFINE_UNARY_OP_BOOL(name, op_type)                                  \
    Tensor name() const {                                                        \
        if (!is_valid() || numel() == 0) {                                       \
            if (!is_valid())                                                     \
                return Tensor();                                                 \
            return Tensor::empty(shape_, device_, DataType::Bool);               \
        }                                                                        \
        Tensor result = UnaryExpr<TensorLeaf, ops::op_type>(                     \
            TensorLeaf(*this), ops::op_type{}, shape_, device_, DataType::Bool); \
        link_deferred_result_to_inputs(result, {lazy_expr_id()});                \
        return result;                                                           \
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

        Tensor normalize(int dim = -1, float eps = 1e-12f) const;
        Tensor logit(float eps = 1e-7f) const;

        // ============= BINARY OPERATIONS (Template-based) =============

        // Arithmetic operations

        // New functor-based overloads for Tensor (zero enum overhead, lazy evaluation)
        // Now with automatic type promotion for mixed-dtype operations
        Tensor add(const Tensor& other) const {
            return binary_op_with_promotion(other, ops::add_op{});
        }

        Tensor sub(const Tensor& other) const {
            return binary_op_with_promotion(other, ops::sub_op{});
        }

        Tensor mul(const Tensor& other) const {
            return binary_op_with_promotion(other, ops::mul_op{});
        }

        Tensor div(const Tensor& other) const {
            return binary_op_with_promotion(other, ops::div_op{});
        }

        Tensor pow(const Tensor& other) const {
            return binary_op_with_promotion(other, ops::pow_op{});
        }

        Tensor mod(const Tensor& other) const {
            return binary_op_with_promotion(other, ops::mod_op{});
        }

        Tensor maximum(const Tensor& other) const {
            return binary_op_with_promotion(other, ops::maximum_op{});
        }

        Tensor minimum(const Tensor& other) const {
            return binary_op_with_promotion(other, ops::minimum_op{});
        }

        // Macro for scalar binary operations (lazy evaluation with scalar_right_op)
#define LFS_DEFINE_SCALAR_BINARY_OP_FUSABLE(name, op_type, fusion_kind)                   \
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>           \
    Tensor name(const T& other) const {                                                   \
        if (!is_valid() || numel() == 0) {                                                \
            if (!is_valid())                                                              \
                return Tensor();                                                          \
            return Tensor::empty(shape_, device_, dtype_);                                \
        }                                                                                 \
        const float scalar_value = static_cast<float>(other);                             \
        Tensor result = UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::op_type, float>>( \
            TensorLeaf(*this), ops::scalar_right_op<ops::op_type, float>(scalar_value),   \
            shape_, device_, dtype_);                                                     \
        link_deferred_result_to_inputs(result, {lazy_expr_id()});                         \
        if (dtype_ == DataType::Float32 &&                                                \
            result.is_valid() && result.has_lazy_expr()) {                                \
            const uint64_t result_node_id = result.lazy_expr_id();                        \
            if (result_node_id != 0 && result.state_) {                                   \
                internal::lazy_executor_register_pointwise_fusion_op(                     \
                    result_node_id,                                                       \
                    lazy_expr_id(),                                                       \
                    *this,                                                                \
                    internal::LazyPointwiseOp{internal::LazyPointwiseOpKind::fusion_kind, \
                                              scalar_value},                              \
                    std::weak_ptr<void>(result.state_));                                  \
            }                                                                             \
        }                                                                                 \
        return result;                                                                    \
    }

#define LFS_DEFINE_SCALAR_BINARY_OP(name, op_type)                                                   \
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>                      \
    Tensor name(const T& other) const {                                                              \
        if (!is_valid() || numel() == 0) {                                                           \
            if (!is_valid())                                                                         \
                return Tensor();                                                                     \
            return Tensor::empty(shape_, device_, dtype_);                                           \
        }                                                                                            \
        Tensor result = UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::op_type, float>>(            \
            TensorLeaf(*this), ops::scalar_right_op<ops::op_type, float>(static_cast<float>(other)), \
            shape_, device_, dtype_);                                                                \
        link_deferred_result_to_inputs(result, {lazy_expr_id()});                                    \
        return result;                                                                               \
    }

        LFS_DEFINE_SCALAR_BINARY_OP_FUSABLE(add, add_op, AddScalar)
        LFS_DEFINE_SCALAR_BINARY_OP_FUSABLE(sub, sub_op, SubScalar)
        LFS_DEFINE_SCALAR_BINARY_OP_FUSABLE(mul, mul_op, MulScalar)
        LFS_DEFINE_SCALAR_BINARY_OP_FUSABLE(div, div_op, DivScalar)
        LFS_DEFINE_SCALAR_BINARY_OP(pow, pow_op)
        LFS_DEFINE_SCALAR_BINARY_OP(mod, mod_op)
        LFS_DEFINE_SCALAR_BINARY_OP(maximum, maximum_op)
        LFS_DEFINE_SCALAR_BINARY_OP(minimum, minimum_op)

#undef LFS_DEFINE_SCALAR_BINARY_OP
#undef LFS_DEFINE_SCALAR_BINARY_OP_FUSABLE

        // Comparison operations (return Bool tensors)

        // Functor-based overloads for Tensor (zero enum overhead)
        Tensor eq(const Tensor& other) const {
            return comparison_op_with_promotion(other, ops::equal_op{});
        }

        Tensor ne(const Tensor& other) const {
            return comparison_op_with_promotion(other, ops::not_equal_op{});
        }

        Tensor lt(const Tensor& other) const {
            return comparison_op_with_promotion(other, ops::less_op{});
        }

        Tensor le(const Tensor& other) const {
            return comparison_op_with_promotion(other, ops::less_equal_op{});
        }

        Tensor gt(const Tensor& other) const {
            return comparison_op_with_promotion(other, ops::greater_op{});
        }

        Tensor ge(const Tensor& other) const {
            return comparison_op_with_promotion(other, ops::greater_equal_op{});
        }

        // Macro for scalar comparison operations (return Bool dtype)
#define LFS_DEFINE_SCALAR_CMP_OP(name, op_type)                                                      \
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>                      \
    Tensor name(const T& other) const {                                                              \
        if (!is_valid() || numel() == 0) {                                                           \
            if (!is_valid())                                                                         \
                return Tensor();                                                                     \
            return Tensor::empty(shape_, device_, DataType::Bool);                                   \
        }                                                                                            \
        return UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::op_type, float>>(                     \
            TensorLeaf(*this), ops::scalar_right_op<ops::op_type, float>(static_cast<float>(other)), \
            shape_, device_, DataType::Bool);                                                        \
    }

        LFS_DEFINE_SCALAR_CMP_OP(eq, equal_op)
        LFS_DEFINE_SCALAR_CMP_OP(ne, not_equal_op)
        LFS_DEFINE_SCALAR_CMP_OP(lt, less_op)
        LFS_DEFINE_SCALAR_CMP_OP(le, less_equal_op)
        LFS_DEFINE_SCALAR_CMP_OP(gt, greater_op)
        LFS_DEFINE_SCALAR_CMP_OP(ge, greater_equal_op)

#undef LFS_DEFINE_SCALAR_CMP_OP

        // Logical operations (Tensor only, Bool -> Bool)
        Tensor logical_and(const Tensor& other) const {
            return comparison_op_with_promotion(other, ops::logical_and_op{});
        }

        Tensor logical_or(const Tensor& other) const {
            return comparison_op_with_promotion(other, ops::logical_or_op{});
        }

        Tensor logical_xor(const Tensor& other) const {
            return comparison_op_with_promotion(other, ops::logical_xor_op{});
        }

        // ============= REDUCE OPERATIONS =============
        Tensor sum(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Sum, args);
        }

        Tensor sum(std::initializer_list<int> axes, bool keepdim = false) const {
            return sum(std::span<const int>(axes), keepdim);
        }

        Tensor sum(int dim, bool keepdim = false) const {
            std::vector<int> axes = {dim};
            return sum(std::span<const int>(axes), keepdim);
        }

        Tensor mean(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Mean, args);
        }

        Tensor mean(std::initializer_list<int> axes, bool keepdim = false) const {
            return mean(std::span<const int>(axes), keepdim);
        }

        Tensor mean(int dim, bool keepdim = false) const {
            std::vector<int> axes = {dim};
            return mean(std::span<const int>(axes), keepdim);
        }

        Tensor max(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Max, args);
        }

        Tensor max(std::initializer_list<int> axes, bool keepdim = false) const {
            return max(std::span<const int>(axes), keepdim);
        }

        Tensor max(int dim, bool keepdim = false) const {
            std::vector<int> axes = {dim};
            return max(std::span<const int>(axes), keepdim);
        }

        Tensor min(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Min, args);
        }

        Tensor min(std::initializer_list<int> axes, bool keepdim = false) const {
            return min(std::span<const int>(axes), keepdim);
        }

        Tensor min(int dim, bool keepdim = false) const {
            std::vector<int> axes = {dim};
            return min(std::span<const int>(axes), keepdim);
        }

        Tensor prod(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Prod, args);
        }

        Tensor prod(std::initializer_list<int> axes, bool keepdim = false) const {
            return prod(std::span<const int>(axes), keepdim);
        }

        Tensor prod(int dim, bool keepdim = false) const {
            std::vector<int> axes = {dim};
            return prod(std::span<const int>(axes), keepdim);
        }

        Tensor any(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Any, args);
        }

        Tensor any(int dim, bool keepdim = false) const {
            std::vector<int> axes = {dim};
            return any(std::span<const int>(axes), keepdim);
        }

        Tensor all(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::All, args);
        }

        Tensor all(int dim, bool keepdim = false) const {
            std::vector<int> axes = {dim};
            return all(std::span<const int>(axes), keepdim);
        }

        Tensor std(std::span<const int> axes = {}, bool keepdim = false, bool unbiased = true) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            args.unbiased = unbiased;
            return reduce(ReduceOp::Std, args);
        }

        Tensor std(std::initializer_list<int> axes, bool keepdim = false, bool unbiased = true) const {
            return std(std::span<const int>(axes), keepdim, unbiased);
        }

        Tensor std(int dim, bool keepdim = false, bool unbiased = true) const {
            std::vector<int> axes = {dim};
            return std(std::span<const int>(axes), keepdim, unbiased);
        }

        Tensor var(std::span<const int> axes = {}, bool keepdim = false, bool unbiased = true) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            args.unbiased = unbiased;
            return reduce(ReduceOp::Var, args);
        }

        Tensor var(std::initializer_list<int> axes, bool keepdim = false, bool unbiased = true) const {
            return var(std::span<const int>(axes), keepdim, unbiased);
        }

        Tensor var(int dim, bool keepdim = false, bool unbiased = true) const {
            std::vector<int> axes = {dim};
            return var(std::span<const int>(axes), keepdim, unbiased);
        }

        Tensor argmax(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Argmax, args);
        }

        Tensor argmin(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Argmin, args);
        }

        Tensor cumsum(int dim = 0) const;

        // Scalar reduce operations - use direct CUB path for CUDA Float32 contiguous tensors
        float sum_scalar() const {
            if (device_ == Device::CUDA && dtype_ == DataType::Float32 && is_contiguous_) {
                return tensor_ops::direct_sum_scalar(ptr<float>(), numel(), stream());
            }
            auto result = sum();
            if (dtype_ == DataType::Bool) {
                return static_cast<float>(result.item<int64_t>());
            }
            return result.item<float>();
        }

        float mean_scalar() const {
            if (device_ == Device::CUDA && dtype_ == DataType::Float32 && is_contiguous_) {
                return tensor_ops::direct_mean_scalar(ptr<float>(), numel(), stream());
            }
            return mean().item();
        }

        float min_scalar() const {
            if (device_ == Device::CUDA && dtype_ == DataType::Float32 && is_contiguous_) {
                return tensor_ops::direct_min_scalar(ptr<float>(), numel(), stream());
            }
            return min().item();
        }

        float max_scalar() const {
            if (device_ == Device::CUDA && dtype_ == DataType::Float32 && is_contiguous_) {
                return tensor_ops::direct_max_scalar(ptr<float>(), numel(), stream());
            }
            return max().item();
        }
        float std_scalar(bool unbiased = true) const { return std({}, false, unbiased).item(); }
        float var_scalar(bool unbiased = true) const { return var({}, false, unbiased).item(); }
        std::pair<float, float> minmax() const { return {min_scalar(), max_scalar()}; }

        float norm(float p = 2.0f) const;
        Tensor norm(float p, std::span<const int> dims, bool keepdim = false) const;
        Tensor norm(float p, std::initializer_list<int> dims, bool keepdim = false) const {
            return norm(p, std::span<const int>(dims), keepdim);
        }

        // Convenience methods
        Tensor norm(float p, int dim, bool keepdim = false) const {
            std::vector<int> dims_vec = {dim};
            return norm(p, std::span<const int>(dims_vec), keepdim);
        }

        float item() const;

        template <typename T>
        T item() const {
            materialize_if_deferred();
            if (!is_valid()) {
                throw std::runtime_error("item<T>() called on invalid tensor");
            }
            if (numel() != 1) {
                throw std::runtime_error(
                    "item<T>() requires single-element tensor, got " + std::to_string(numel()) + " elements");
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

            T value{};
            // Account for storage offset (important for sliced tensors)
            const char* data_ptr = static_cast<const char*>(data_) + storage_offset_ * dtype_size(dtype_);

            if (device_ == Device::CUDA) {
                cudaMemcpy(&value, data_ptr, sizeof(T), cudaMemcpyDeviceToHost);
            } else {
                value = *static_cast<const T*>(static_cast<const void*>(data_ptr));
            }
            return value;
        }

        size_t count_nonzero() const;

        // ============= TERNARY OPERATIONS =============
        Tensor where(const Tensor& condition, const Tensor& other) const {
            return condition.ternary(*this, other);
        }

        Tensor clamp(float min_val, float max_val) const;

        Tensor clamp_min(float min) const {
            return clamp(min, std::numeric_limits<float>::max());
        }

        Tensor clamp_max(float max) const {
            return clamp(std::numeric_limits<float>::lowest(), max);
        }

        Tensor& clamp_(float min_val, float max_val);
        Tensor& clamp_min_(float min);
        Tensor& clamp_max_(float max);

        // In-place operations (Template-based, direct functor dispatch - zero enum overhead!)
        template <typename T>
        Tensor& add_(const T& other) {
            if constexpr (std::is_same_v<T, Tensor>) {
                return binary_op_inplace_generic(other, ops::add_op{});
            } else {
                return scalar_op_inplace_generic(static_cast<float>(other), ops::add_op{});
            }
        }

        template <typename T>
        Tensor& sub_(const T& other) {
            if constexpr (std::is_same_v<T, Tensor>) {
                return binary_op_inplace_generic(other, ops::sub_op{});
            } else {
                return scalar_op_inplace_generic(static_cast<float>(other), ops::sub_op{});
            }
        }

        template <typename T>
        Tensor& mul_(const T& other) {
            if constexpr (std::is_same_v<T, Tensor>) {
                return binary_op_inplace_generic(other, ops::mul_op{});
            } else {
                return scalar_op_inplace_generic(static_cast<float>(other), ops::mul_op{});
            }
        }

        template <typename T>
        Tensor& div_(const T& other) {
            if constexpr (std::is_same_v<T, Tensor>) {
                return binary_op_inplace_generic(other, ops::div_op{});
            } else {
                return scalar_op_inplace_generic(static_cast<float>(other), ops::div_op{});
            }
        }

        // Matrix operations
        Tensor mm(const Tensor& other) const;
        Tensor bmm(const Tensor& other) const;
        Tensor matmul(const Tensor& other) const;
        Tensor dot(const Tensor& other) const;

        // Neural network operations
        // Conv1x1: per-pixel linear transform [N,C_in,H,W] -> [N,C_out,H,W]
        Tensor conv1x1(const Tensor& weight) const;
        Tensor conv1x1(const Tensor& weight, const Tensor& bias) const;

        // MaxPool2d: window-based max [N,C,H,W] -> [N,C,H/stride,W/stride]
        Tensor max_pool2d(int kernel_size, int stride = -1, int padding = 0) const;

        // AdaptiveAvgPool2d: pool to fixed output size [N,C,H,W] -> [N,C,out_h,out_w]
        Tensor adaptive_avg_pool2d(int output_h, int output_w) const;

        // Linear: fully connected layer [...,in] -> [...,out]
        Tensor linear(const Tensor& weight) const;
        Tensor linear(const Tensor& weight, const Tensor& bias) const;

        // Fused operations for performance
        // Conv1x1 + bias + ReLU in single pass (avoids intermediate allocations)
        Tensor conv1x1_bias_relu(const Tensor& weight, const Tensor& bias) const;
        // Linear + bias + ReLU in single pass
        Tensor linear_bias_relu(const Tensor& weight, const Tensor& bias) const;

        // _out variants that write into pre-allocated output tensors (zero allocation)
        void conv1x1_bias_out(const Tensor& weight, const Tensor& bias, Tensor& output) const;
        void conv1x1_bias_relu_out(const Tensor& weight, const Tensor& bias, Tensor& output) const;
        void relu_out(Tensor& output) const;
        void max_pool2d_out(int kernel_size, int stride, int padding, Tensor& output) const;
        void adaptive_avg_pool2d_out(int output_h, int output_w, Tensor& output) const;
        void linear_bias_relu_out(const Tensor& weight, const Tensor& bias, Tensor& output) const;
        void linear_out(const Tensor& weight, const Tensor& bias, Tensor& output) const;

        // Masking operations
        Tensor masked_select(const Tensor& mask) const;
        Tensor& masked_fill_(const Tensor& mask, float value);
        Tensor masked_fill(const Tensor& mask, float value) const;

        // Indexing operations
        Tensor index_select(int dim, const Tensor& indices) const;
        Tensor gather(int dim, const Tensor& indices) const;
        Tensor take(const Tensor& indices) const;

        /**
         * Append gathered elements in-place to the end of this tensor along dimension 0.
         * This is a fused operation that combines index_select + cat without allocating
         * intermediate tensors.
         *
         * Requirements:
         * - This tensor must have capacity_ > 0 (pre-allocated with reserve())
         * - capacity_ must be sufficient to hold logical_size_ + indices.numel()
         * - Only works for dim=0 (appending along first dimension)
         *
         * Example:
         *   auto param = Tensor::randn({1000, 3}, Device::CUDA);
         *   param.reserve(2000);  // Pre-allocate capacity
         *   auto indices = Tensor::from_vector({0, 5, 10}, {3}, Device::CUDA);
         *   param.append_gather(indices);  // Now param.shape() = {1003, 3}
         *
         * This is equivalent to:
         *   param = Tensor::cat({param, param.index_select(0, indices)}, 0);
         * but without the index_select allocation and extra memcpy.
         *
         * @param indices 1D tensor of indices to gather (same device as this tensor)
         * @return reference to this tensor (for chaining)
         */
        Tensor& append_gather(const Tensor& indices);

        /**
         * Append zeros in-place to the end of this tensor along dimension 0.
         * This is more efficient than cat() with zeros() as it avoids allocating
         * intermediate tensors when capacity is available.
         *
         * Requirements:
         * - This tensor must have capacity_ > 0 (pre-allocated with reserve())
         * - capacity_ must be sufficient to hold logical_size_ + n_rows
         * - Only works for dim=0 (appending along first dimension)
         *
         * @param n_rows Number of zero rows to append
         * @return reference to this tensor (for chaining)
         */
        Tensor& append_zeros(size_t n_rows);

        // Lazy indexing operations (returns expression template)
        auto gather_lazy(const Tensor& indices) const -> PermutationExpr<TensorLeaf, TensorLeaf>;

        Tensor nonzero() const;
        std::vector<Tensor> nonzero_split() const;

        Tensor& scatter_(int dim, const Tensor& indices, const Tensor& src,
                         ScatterMode mode = ScatterMode::None);
        Tensor& scatter_(int dim, const Tensor& indices, float value,
                         ScatterMode mode = ScatterMode::None);
        Tensor& index_fill_(int dim, const Tensor& indices, float value);
        Tensor& index_copy_(int dim, const Tensor& indices, const Tensor& src);
        Tensor& index_add_(int dim, const Tensor& indices, const Tensor& src);
        Tensor& index_put_(const Tensor& indices, const Tensor& values);
        Tensor& index_put_(const std::vector<Tensor>& indices, const Tensor& values);

        Tensor index_select(int dim, const Tensor& indices, BoundaryMode mode) const;
        Tensor gather(int dim, const Tensor& indices, BoundaryMode mode) const;

        TensorIndexer operator[](const Tensor& indices);
        TensorIndexer operator[](const std::vector<Tensor>& indices);
        MaskedTensorProxy operator[](const Tensor& mask) const;

        float& at(std::initializer_list<size_t> indices);
        float at(std::initializer_list<size_t> indices) const;

        // ============= ADVANCED OPERATIONS =============

        // Pairwise distance
        Tensor cdist(const Tensor& other, float p = 2.0f) const;

        // Min/max with indices
        std::pair<Tensor, Tensor> min_with_indices(int dim = -1, bool keepdim = false) const;
        std::pair<Tensor, Tensor> max_with_indices(int dim = -1, bool keepdim = false) const;

        /**
         * Sort the tensor along a given dimension.
         *
         * Returns a pair of tensors:
         * - values: Sorted values (same dtype as input)
         * - indices: Int64 tensor containing the indices that would sort the input
         *
         * Example:
         *   auto t = Tensor::from_vector({3.0f, 1.0f, 2.0f}, {3}, Device::CPU);
         *   auto [sorted_vals, sorted_idx] = t.sort(0, false);
         *   // sorted_vals: [1.0, 2.0, 3.0] (Float32)
         *   // sorted_idx:  [1, 0, 2]       (Int64)
         *
         * @param dim Dimension to sort along (default: -1, last dimension)
         * @param descending If true, sort in descending order (default: false)
         * @return Pair of (sorted_values, indices). Indices are always Int64 dtype.
         */
        std::pair<Tensor, Tensor> sort(int dim = -1, bool descending = false) const;

        // Scalar boolean reductions
        bool any_scalar() const;
        bool all_scalar() const;

        // ============= OPERATOR OVERLOADS (Template-based) =============

        // Addition
        template <typename T>
        auto operator+(const T& other) const { return add(other); }

        // Subtraction
        template <typename T>
        auto operator-(const T& other) const { return sub(other); }

        // Multiplication
        template <typename T>
        auto operator*(const T& other) const { return mul(other); }

        // Division
        template <typename T>
        auto operator/(const T& other) const { return div(other); }

        // Modulo
        template <typename T>
        Tensor operator%(const T& other) const { return mod(other); }

        // Negation
        auto operator-() const { return neg(); }

        // Comparison operators
        template <typename T>
        Tensor operator==(const T& other) const { return eq(other); }

        template <typename T>
        Tensor operator!=(const T& other) const { return ne(other); }

        template <typename T>
        Tensor operator<(const T& other) const { return lt(other); }

        template <typename T>
        Tensor operator<=(const T& other) const { return le(other); }

        template <typename T>
        Tensor operator>(const T& other) const { return gt(other); }

        template <typename T>
        Tensor operator>=(const T& other) const { return ge(other); }

        // Logical operators (Tensor only)
        Tensor operator&&(const Tensor& other) const { return logical_and(other); }
        Tensor operator||(const Tensor& other) const { return logical_or(other); }
        Tensor operator!() const { return logical_not(); }

        Tensor operator~() const;
        Tensor operator|(const Tensor& other) const;

        // Other in-place operations
        Tensor& zero_();
        Tensor& fill_(float value);
        Tensor& fill_(float value, cudaStream_t stream); // Stream-aware version (no sync)
        Tensor& copy_from(const Tensor& other);
        Tensor& copy_(const Tensor& src) { return copy_from(src); }
        Tensor& uniform_(float low = 0.0f, float high = 1.0f);
        Tensor& normal_(float mean = 0.0f, float std = 1.0f);

        std::optional<Tensor> try_reshape(TensorShape shape) const;

        static std::vector<Tensor> split_batch(const Tensor& tensor, size_t batch_size);

        // Utility template methods
        template <typename Func>
        Tensor& inplace(Func&& func) {
            func(*this);
            return *this;
        }

        template <typename Func>
        Tensor apply(Func&& func) const {
            return func(*this);
        }

        template <typename Func>
        Tensor timed(const std::string& name, Func&& func) const {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = func(*this);
            auto end = std::chrono::high_resolution_clock::now();
            if (profiling_enabled_) {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                // Note: logging moved to .cpp - use profile_callback_ if set
                (void)name;
                (void)duration;
            }
            return result;
        }

        // Validation & assertions
        Tensor& assert_shape(TensorShape expected, const std::string& msg = "");
        Tensor& assert_device(Device expected);
        Tensor& assert_dtype(DataType expected);
        Tensor& assert_finite();

        // Comparison operations
        bool has_nan() const;
        bool has_inf() const;
        bool all_close(const Tensor& other, float rtol = 1e-5f, float atol = 1e-8f) const;

        // Utility functions
        std::string str() const;
        std::vector<float> to_vector() const;
        std::vector<uint8_t> to_vector_uint8() const;
        std::vector<int64_t> to_vector_int64() const;

        std::vector<int> to_vector_int() const;
        std::vector<bool> to_vector_bool() const;
        std::vector<float> debug_values(size_t max_values = 100) const;

        void dump_diagnostic(const std::string& filename) const;
        void log_info(const std::string& name = "") const;
        void print_formatted(const std::string& name = "", size_t max_per_dim = 10) const;

        // ============= TENSOR OPTIONS =============
        struct TensorOptions {
            Device device = Device::CUDA;
            DataType dtype = DataType::Float32;

            TensorOptions() = default;
            TensorOptions(Device dev) : device(dev) {}
            TensorOptions(DataType dt) : dtype(dt) {}
            TensorOptions(Device dev, DataType dt) : device(dev),
                                                     dtype(dt) {}
        };

        TensorOptions options() const {
            return TensorOptions{device_, dtype_};
        }

    private:
        void print_1d(size_t max_elem = 10) const;
        void print_2d(size_t max_per_dim = 10) const;
        friend class TensorIndexer;
        friend class MaskedTensorProxy;
        friend class TensorRowProxy;
        template <typename Derived>
        friend class TensorExpr;
    };

    // ============= TensorRowProxy for operator[] =============
    // Implementations in tensor_row_proxy.cpp (except template methods)
    class LFS_CORE_API TensorRowProxy {
    private:
        Tensor* tensor_;
        size_t row_index_;
        mutable float cuda_staging_ = 0.0f;
        mutable size_t cuda_staging_linear_idx_ = 0;
        mutable bool cuda_staging_pending_write_ = false;
        void flush_cuda_staging() const;

    public:
        TensorRowProxy(Tensor* tensor, size_t row_index)
            : tensor_(tensor),
              row_index_(row_index) {
            if (tensor_ && tensor_->is_valid() && row_index_ >= tensor_->shape()[0]) {
                throw std::out_of_range(
                    "Row index " + std::to_string(row_index_) + " out of bounds for dimension 0 with size " +
                    std::to_string(tensor_->shape()[0]));
            }
        }
        ~TensorRowProxy();

        // 2D Access: tensor[i][j]
        float& operator[](size_t col_index);
        float operator[](size_t col_index) const;

        // 1D Access: Extract Value
        float item() const;
        operator float() const;

        // Template version for type specification (must stay in header)
        template <typename T = float>
        T item_as() const {
            if (!tensor_ || !tensor_->is_valid()) {
                throw std::runtime_error("TensorRowProxy::item_as(): invalid tensor pointer");
            }
            flush_cuda_staging();

            // Handle 2D tensors with shape [N, 1] (like nonzero() output)
            if (tensor_->shape().rank() == 2 && tensor_->shape()[1] == 1) {
                Tensor row_tensor = static_cast<Tensor>(*this);
                return row_tensor.item<T>();
            }

            // Standard 1D case
            if (tensor_->shape().rank() != 1) {
                throw std::runtime_error(
                    "TensorRowProxy::item_as(): only valid for 1D or [N,1] tensors, got rank " +
                    std::to_string(tensor_->shape().rank()));
            }

            if (row_index_ >= tensor_->numel()) {
                throw std::out_of_range(
                    "TensorRowProxy::item_as(): index " + std::to_string(row_index_) +
                    " out of bounds for size " + std::to_string(tensor_->numel()));
            }

            if (tensor_->device() == Device::CUDA) {
                T value{};
                size_t type_size = dtype_size(tensor_->dtype());
                const void* src_ptr = static_cast<const char*>(tensor_->data_ptr()) + row_index_ * type_size;
                cudaError_t err = cudaMemcpy(&value, src_ptr, sizeof(T), cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("CUDA memcpy failed in TensorRowProxy::item_as(): ") + cudaGetErrorString(err));
                }
                return value;
            } else {
                if (tensor_->dtype() == DataType::Float32) {
                    if constexpr (std::is_same_v<T, float>) {
                        return static_cast<T>(tensor_->ptr<float>()[row_index_]);
                    } else if constexpr (std::is_same_v<T, int>) {
                        return static_cast<T>(tensor_->ptr<float>()[row_index_]);
                    } else if constexpr (std::is_same_v<T, int64_t>) {
                        return static_cast<T>(tensor_->ptr<float>()[row_index_]);
                    }
                } else if (tensor_->dtype() == DataType::Int32) {
                    if constexpr (std::is_same_v<T, int>) {
                        return static_cast<T>(tensor_->ptr<int>()[row_index_]);
                    } else if constexpr (std::is_same_v<T, float>) {
                        return static_cast<T>(tensor_->ptr<int>()[row_index_]);
                    } else if constexpr (std::is_same_v<T, int64_t>) {
                        return static_cast<T>(tensor_->ptr<int>()[row_index_]);
                    }
                } else if (tensor_->dtype() == DataType::Int64) {
                    const int64_t* data = reinterpret_cast<const int64_t*>(tensor_->data_ptr());
                    if constexpr (std::is_same_v<T, int64_t>) {
                        return data[row_index_];
                    } else if constexpr (std::is_same_v<T, int>) {
                        return static_cast<T>(data[row_index_]);
                    } else if constexpr (std::is_same_v<T, float>) {
                        return static_cast<T>(data[row_index_]);
                    }
                } else if (tensor_->dtype() == DataType::Bool) {
                    const unsigned char* data = tensor_->ptr<unsigned char>();
                    if constexpr (std::is_same_v<T, bool>) {
                        return data[row_index_] != 0;
                    } else if constexpr (std::is_same_v<T, float>) {
                        return data[row_index_] ? 1.0f : 0.0f;
                    } else if constexpr (std::is_same_v<T, int>) {
                        return data[row_index_] ? 1 : 0;
                    } else if constexpr (std::is_same_v<T, int64_t>) {
                        return data[row_index_] ? 1LL : 0LL;
                    }
                }
                throw std::runtime_error("Unsupported dtype/type combination for item_as()");
            }
        }

        // Specialized item_as for common types
        int item_int() const { return item_as<int>(); }
        int64_t item_int64() const { return item_as<int64_t>(); }

        // Conversion to Tensor
        operator Tensor() const;

        // Assignment Operators
        TensorRowProxy& operator=(const TensorRowProxy& other);
        TensorRowProxy& operator=(const Tensor& other);
        TensorRowProxy& operator=(float value);

        // Arithmetic Operations with TensorRowProxy
        Tensor operator-(const TensorRowProxy& other) const;
        Tensor operator+(const TensorRowProxy& other) const;
        Tensor operator*(const TensorRowProxy& other) const;
        Tensor operator/(const TensorRowProxy& other) const;

        // Arithmetic Operations with Scalars
        Tensor operator-(float scalar) const;
        Tensor operator+(float scalar) const;
        Tensor operator*(float scalar) const;
        Tensor operator/(float scalar) const;

        // Unary Operations
        Tensor operator-() const;
        Tensor pow(float exponent) const;
        Tensor sqrt() const;
        Tensor abs() const;
        Tensor neg() const;
        Tensor sum() const;
        Tensor mean() const;
        Tensor square() const;
    };

    // Implementation of Tensor::operator[]
    inline TensorRowProxy Tensor::operator[](size_t index) {
        if (!is_valid()) {
            throw std::runtime_error("operator[] on invalid tensor");
        }
        if (index >= shape_[0]) {
            throw std::out_of_range(
                "Index " + std::to_string(index) + " out of bounds for dimension 0 with size " +
                std::to_string(shape_[0]));
        }
        return TensorRowProxy(this, index);
    }

    inline const TensorRowProxy Tensor::operator[](size_t index) const {
        if (!is_valid()) {
            throw std::runtime_error("operator[] on invalid tensor");
        }
        if (index >= shape_[0]) {
            throw std::out_of_range(
                "Index " + std::to_string(index) + " out of bounds for dimension 0 with size " +
                std::to_string(shape_[0]));
        }
        return TensorRowProxy(const_cast<Tensor*>(this), index);
    }

    // Helper classes
    class LFS_CORE_API MaskedTensorProxy {
    private:
        const Tensor* tensor_;
        Tensor mask_;

    public:
        MaskedTensorProxy(const Tensor* tensor, Tensor mask)
            : tensor_(tensor),
              mask_(std::move(mask)) {}

        void operator=(float value);
        void operator=(const Tensor& other);
        operator Tensor() const;
    };

    class LFS_CORE_API TensorIndexer {
    private:
        Tensor* tensor_;
        std::vector<Tensor> indices_;

    public:
        TensorIndexer(Tensor* tensor, std::vector<Tensor> indices)
            : tensor_(tensor),
              indices_(std::move(indices)) {}

        void operator=(float value);
        void operator=(const Tensor& other);
        operator Tensor() const;
    };

    class LFS_CORE_API TensorError : public std::runtime_error {
    public:
        TensorError(const std::string& msg, const Tensor* t = nullptr);
        const std::string& tensor_info() const { return tensor_info_; }

    private:
        std::string tensor_info_;
    };

    // Memory info
    class LFS_CORE_API MemoryInfo {
    public:
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        size_t allocated_bytes = 0;
        int device_id = -1;

        static MemoryInfo cuda();
        static MemoryInfo cpu();

        void log() const;
    };

    // ========================================================================
    // Inline implementation of lazy gather operation
    // ========================================================================

    inline auto Tensor::gather_lazy(const Tensor& indices) const -> PermutationExpr<TensorLeaf, TensorLeaf> {
        if (!is_valid() || !indices.is_valid()) {
            throw std::runtime_error("gather_lazy: invalid tensor or indices");
        }

        // Create expression that will lazily gather elements
        return PermutationExpr<TensorLeaf, TensorLeaf>(
            TensorLeaf(*this),
            TensorLeaf(indices),
            indices.shape(), // Output shape matches indices shape
            device_,
            dtype_);
    }

} // namespace lfs::core

// Include expression template implementations at the very end
// This ensures all Tensor definitions are complete before templates are instantiated
#include "tensor_expr_impl.hpp"
