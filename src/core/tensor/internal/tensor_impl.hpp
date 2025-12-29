/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor_fwd.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstring>
#include <cuda_runtime.h>
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

#include "tensor_functors.hpp"
#include "tensor_ops.hpp"

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

    class TensorShape {
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

    class RandomGenerator {
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

} // namespace lfs::core

// Include expression template declarations (forward declarations only)
#include "tensor_expr.hpp"

namespace lfs::core {

    class Tensor {
    private:
        void* data_ = nullptr;
        std::shared_ptr<void> data_owner_;
        TensorShape shape_;
        std::vector<size_t> strides_; // Stride for each dimension (in elements)
        size_t storage_offset_ = 0;   // Offset from data_ (in elements)
        bool is_contiguous_ = true;   // True if memory layout is C-contiguous
        Device device_ = Device::CPU;
        DataType dtype_ = DataType::Float32;
        bool is_view_ = false;

        // Capacity management for in-place growth (like std::vector)
        // capacity_ is the number of "rows" (dim 0) that can fit in the allocated buffer
        // logical_size_ is the current logical number of rows (same as shape_[0])
        // When capacity_ > 0, the buffer is larger than needed to allow in-place growth
        size_t capacity_ = 0;     // Reserved capacity along dimension 0 (0 = no reservation)
        size_t logical_size_ = 0; // Logical size along dimension 0 (same as shape_[0])

        // Cached alignment flags (computed once on allocation)
        bool is_aligned_16_ = false;  // 16-byte alignment for float4 vectorization
        bool is_aligned_128_ = false; // 128-byte alignment for cache line optimization

        // CUDA stream for async execution (assigned round-robin from StreamPool)
        cudaStream_t stream_ = nullptr;

        mutable size_t id_ = 0;
        static std::atomic<size_t> next_id_;
        static inline bool profiling_enabled_ = false;

        // Debug tracking - when true, operations on this tensor are logged
        bool tracked_ = false;
        std::string name_; // Optional name for identification in traces

        // Compute alignment flags for vectorization
        void compute_alignment() {
            if (data_ != nullptr) {
                auto addr = reinterpret_cast<uintptr_t>(data_);
                is_aligned_16_ = (addr % 16) == 0;
                is_aligned_128_ = (addr % 128) == 0;
            } else {
                is_aligned_16_ = false;
                is_aligned_128_ = false;
            }
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
                            numel(), op, nullptr);
                    } else if (out_dtype == DataType::Int32) {
                        tensor_ops::launch_scalar_op_generic(
                            ptr<int>(), scalar_int, result.ptr<int>(),
                            numel(), op, nullptr);
                    }
                } else { // Float32
                    if (out_dtype == DataType::Bool) {
                        tensor_ops::launch_scalar_op_generic(
                            ptr<float>(), scalar, result.ptr<unsigned char>(),
                            numel(), op, nullptr);
                    } else {
                        tensor_ops::launch_scalar_op_generic(
                            ptr<float>(), scalar, result.ptr<float>(),
                            numel(), op, nullptr);
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
                    numel(), op, nullptr);
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
                    numel(), op, nullptr);
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
            // Check if broadcasting is valid (even when require_same_shape is false)
            if (!require_same_shape && shape_ != other.shape()) {
                // Compute broadcast shape and validate
                auto bcast_shape = broadcast_shape(other.shape());
                if (bcast_shape.rank() == 0 || bcast_shape.elements() == 0) {
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
            return BinaryExpr<TensorLeaf, TensorLeaf, Op>(
                TensorLeaf(lhs), TensorLeaf(rhs), op,
                broadcast_shape, lhs.device(), result_dtype);
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
            return BinaryExpr<TensorLeaf, TensorLeaf, Op>(
                TensorLeaf(lhs), TensorLeaf(rhs), op,
                broadcast_shape, lhs.device(), DataType::Bool);
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

        // Helper to create view with shared ownership
        Tensor create_view(const TensorShape& new_shape) const {
            // If tensor is not contiguous, we cannot create a simple reshape view
            // We must materialize it first
            if (!is_contiguous_) {
                // Make contiguous copy, then reshape
                return contiguous().create_view(new_shape);
            }

            // For contiguous tensors, we can create a view with the new shape
            Tensor view(data_, new_shape, device_, dtype_);
            view.data_owner_ = data_owner_;
            view.storage_offset_ = storage_offset_; // Preserve storage offset
            view.is_view_ = true;
            view.is_contiguous_ = true; // Reshaped contiguous tensor is still contiguous
            // Strides are automatically set to contiguous by constructor
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
            return empty(other.shape(), other.device(), other.dtype());
        }

        static Tensor full_like(const Tensor& other, float value) {
            return full(other.shape(), value, other.device(), other.dtype());
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

        void set_bool(std::initializer_list<size_t> indices, bool value);
        bool get_bool(std::initializer_list<size_t> indices) const;
        void set_bool(std::span<const size_t> indices, bool value);
        bool get_bool(std::span<const size_t> indices) const;

        // Data access - FIXED: Handle invalid tensors safely
        template <typename T>
        T* ptr() {
            if (!is_valid()) {
                return nullptr;
            }
            // Account for storage offset (important for sliced/strided tensors)
            char* data_ptr = static_cast<char*>(data_) + storage_offset_ * dtype_size(dtype_);
            return static_cast<T*>(static_cast<void*>(data_ptr));
        }

        template <typename T>
        const T* ptr() const {
            if (!is_valid()) {
                return nullptr;
            }
            // Account for storage offset (important for sliced/strided tensors)
            const char* data_ptr = static_cast<const char*>(data_) + storage_offset_ * dtype_size(dtype_);
            return static_cast<const T*>(static_cast<const void*>(data_ptr));
        }

        // Pointer to tensor data (accounts for storage_offset)
        void* data_ptr() noexcept {
            return static_cast<char*>(data_) + storage_offset_ * dtype_size(dtype_);
        }
        const void* data_ptr() const noexcept {
            return static_cast<const char*>(data_) + storage_offset_ * dtype_size(dtype_);
        }

        // Base of allocation (for memory management only)
        void* storage_ptr() noexcept { return data_; }
        const void* storage_ptr() const noexcept { return data_; }

        // Properties - FIXED: Check validity before accessing shape
        const TensorShape& shape() const { return shape_; }
        Device device() const { return device_; }
        DataType dtype() const { return dtype_; }
        bool owns_memory() const { return static_cast<bool>(data_owner_) && !is_view_; }
        bool is_view() const { return is_view_; }
        bool is_empty() const { return !is_valid() || numel() == 0; }

        // CRITICAL: Check data presence, not any flag
        bool is_valid() const {
            return static_cast<bool>(data_owner_) || is_view_;
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
        bool is_aligned_16() const { return is_aligned_16_; }
        bool is_aligned_128() const { return is_aligned_128_; }

        // Stream accessor (for async CUDA operations)
        cudaStream_t stream() const { return stream_; }
        void set_stream(cudaStream_t stream) { stream_ = stream; }

        // Debug tracking - mark tensor to trace all operations it's involved in
        bool is_tracked() const { return tracked_; }
        Tensor& set_tracked(bool tracked = true) {
            tracked_ = tracked;
            return *this;
        }
        Tensor& track() { return set_tracked(true); } // Convenience alias
        Tensor& untrack() { return set_tracked(false); }

        // Optional name for identifying tensors in traces
        const std::string& name() const { return name_; }
        Tensor& set_name(std::string name) {
            name_ = std::move(name);
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
        size_t capacity() const { return capacity_; }
        size_t logical_size() const { return logical_size_; }

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
        return UnaryExpr<TensorLeaf, ops::op_type>(                      \
            TensorLeaf(*this), ops::op_type{}, shape_, device_, dtype_); \
    }

        // Macro for unary ops that return Bool dtype (isnan, isinf, etc.)
#define LFS_DEFINE_UNARY_OP_BOOL(name, op_type)                                  \
    Tensor name() const {                                                        \
        if (!is_valid() || numel() == 0) {                                       \
            if (!is_valid())                                                     \
                return Tensor();                                                 \
            return Tensor::empty(shape_, device_, DataType::Bool);               \
        }                                                                        \
        return UnaryExpr<TensorLeaf, ops::op_type>(                              \
            TensorLeaf(*this), ops::op_type{}, shape_, device_, DataType::Bool); \
    }

        // Arithmetic unary operations
        LFS_DEFINE_UNARY_OP(neg, neg_op)
        LFS_DEFINE_UNARY_OP(abs, abs_op)
        LFS_DEFINE_UNARY_OP(sign, sign_op)
        LFS_DEFINE_UNARY_OP(reciprocal, reciprocal_op)

        // Exponential and logarithmic
        LFS_DEFINE_UNARY_OP(exp, exp_op)
        LFS_DEFINE_UNARY_OP(exp2, exp2_op)
        LFS_DEFINE_UNARY_OP(log, log_op)
        LFS_DEFINE_UNARY_OP(log2, log2_op)
        LFS_DEFINE_UNARY_OP(log10, log10_op)
        LFS_DEFINE_UNARY_OP(log1p, log1p_op)

        // Power and roots
        LFS_DEFINE_UNARY_OP(sqrt, sqrt_op)
        LFS_DEFINE_UNARY_OP(rsqrt, rsqrt_op)
        LFS_DEFINE_UNARY_OP(square, square_op)

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
        LFS_DEFINE_UNARY_OP(tanh, tanh_op)

        // Activation functions
        LFS_DEFINE_UNARY_OP(sigmoid, sigmoid_op)
        LFS_DEFINE_UNARY_OP(relu, relu_op)
        LFS_DEFINE_UNARY_OP(gelu, gelu_op)
        LFS_DEFINE_UNARY_OP(swish, swish_op)

        // Rounding
        LFS_DEFINE_UNARY_OP(floor, floor_op)
        LFS_DEFINE_UNARY_OP(ceil, ceil_op)
        LFS_DEFINE_UNARY_OP(round, round_op)
        LFS_DEFINE_UNARY_OP(trunc, trunc_op)

        // Boolean predicates (return Bool dtype)
        LFS_DEFINE_UNARY_OP_BOOL(isnan, isnan_op)
        LFS_DEFINE_UNARY_OP_BOOL(isinf, isinf_op)
        LFS_DEFINE_UNARY_OP_BOOL(isfinite, isfinite_op)
        LFS_DEFINE_UNARY_OP_BOOL(logical_not, logical_not_op)

#undef LFS_DEFINE_UNARY_OP
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
#define LFS_DEFINE_SCALAR_BINARY_OP(name, op_type)                                                   \
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>                      \
    Tensor name(const T& other) const {                                                              \
        if (!is_valid() || numel() == 0) {                                                           \
            if (!is_valid())                                                                         \
                return Tensor();                                                                     \
            return Tensor::empty(shape_, device_, dtype_);                                           \
        }                                                                                            \
        return UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::op_type, float>>(                     \
            TensorLeaf(*this), ops::scalar_right_op<ops::op_type, float>(static_cast<float>(other)), \
            shape_, device_, dtype_);                                                                \
    }

        LFS_DEFINE_SCALAR_BINARY_OP(add, add_op)
        LFS_DEFINE_SCALAR_BINARY_OP(sub, sub_op)
        LFS_DEFINE_SCALAR_BINARY_OP(mul, mul_op)
        LFS_DEFINE_SCALAR_BINARY_OP(div, div_op)
        LFS_DEFINE_SCALAR_BINARY_OP(pow, pow_op)
        LFS_DEFINE_SCALAR_BINARY_OP(mod, mod_op)
        LFS_DEFINE_SCALAR_BINARY_OP(maximum, maximum_op)
        LFS_DEFINE_SCALAR_BINARY_OP(minimum, minimum_op)

#undef LFS_DEFINE_SCALAR_BINARY_OP

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

        Tensor all(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::All, args);
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

        // Scalar reduce operations
        float sum_scalar() const {
            // Bool reduction is now properly handled by launch_reduce_op_bool()
            // It returns Int64 for Bool dtype (PyTorch behavior)
            auto result = sum();
            if (dtype_ == DataType::Bool) {
                return static_cast<float>(result.item<int64_t>());
            }
            return result.item<float>();
        }
        float mean_scalar() const { return mean().item(); }
        float min_scalar() const { return min().item(); }
        float max_scalar() const { return max().item(); }
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
    };

    // ============= TensorRowProxy for operator[] =============
    // Implementations in tensor_row_proxy.cpp (except template methods)
    class TensorRowProxy {
    private:
        Tensor* tensor_;
        size_t row_index_;

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
    class MaskedTensorProxy {
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

    class TensorIndexer {
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

    class TensorError : public std::runtime_error {
    public:
        TensorError(const std::string& msg, const Tensor* t = nullptr);
        const std::string& tensor_info() const { return tensor_info_; }

    private:
        std::string tensor_info_;
    };

    // Memory info
    class MemoryInfo {
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
