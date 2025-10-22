/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cuda_runtime.h>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "tensor_functors.hpp"
#include "tensor_ops.hpp"

namespace lfs::core {

    // Forward declarations
    class Tensor;
    class TensorError;
    class TensorIndexer;
    class MaskedTensorProxy;
    class TensorRowProxy;

    enum class Device : uint8_t {
        CPU = 0,
        CUDA = 1
    };

    enum class DataType : uint8_t {
        Float32 = 0,
        Float16 = 1,
        Int32 = 2,
        Int64 = 3,
        UInt8 = 4,
        Bool = 5
    };

    constexpr size_t dtype_size(DataType dtype) {
        switch (dtype) {
        case DataType::Float32: return 4;
        case DataType::Float16: return 2;
        case DataType::Int32: return 4;
        case DataType::Int64: return 8;
        case DataType::UInt8: return 1;
        case DataType::Bool: return 1;
        default: return 0;
        }
    }

    inline const char* dtype_name(DataType dtype) {
        switch (dtype) {
        case DataType::Float32: return "float32";
        case DataType::Float16: return "float16";
        case DataType::Int32: return "int32";
        case DataType::Int64: return "int64";
        case DataType::UInt8: return "uint8";
        case DataType::Bool: return "bool";
        default: return "unknown";
        }
    }

    inline const char* device_name(Device device) {
        return device == Device::CPU ? "cpu" : "cuda";
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
        size_t total_elements_ = 0;

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
                LOG_ERROR("Shape index {} out of range for rank {}", i, dims_.size());
                return 0;
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

        mutable size_t id_ = 0;
        static std::atomic<size_t> next_id_;
        static inline bool profiling_enabled_ = false;

        // Generic functor-based binary operation (zero enum overhead)
        template <typename SrcT, typename OutT, typename Op>
        Tensor binary_op_generic(const Tensor& other, Op op) const {
            if (!validate_binary_op(other, false, true)) {
                return Tensor();
            }

            auto broadcast_shape = this->broadcast_shape(other.shape());
            if (broadcast_shape.rank() == 0) {
                LOG_ERROR("Incompatible shapes for broadcasting: {} vs {}",
                          shape_.str(), other.shape_.str());
                return Tensor();
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
                        result.numel(), op, nullptr);
                    cudaDeviceSynchronize();
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
                        result.numel(), op, nullptr);
                    cudaDeviceSynchronize();
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
            if (!validate_unary_op()) {
                return Tensor();
            }

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
            if (!validate_unary_op()) {
                return *this;
            }

            if (device_ == Device::CUDA) {
                tensor_ops::launch_scalar_op_generic(
                    ptr<float>(), scalar, ptr<float>(),
                    numel(), op, nullptr);
                cudaDeviceSynchronize();
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
            if (!validate_binary_op(other, true)) {
                return *this;
            }

            if (device_ == Device::CUDA) {
                tensor_ops::launch_binary_op_generic(
                    ptr<SrcT>(), other.ptr<SrcT>(), ptr<SrcT>(),
                    numel(), op, nullptr);
                cudaDeviceSynchronize();
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

        // Validation helpers
        bool validate_binary_op(const Tensor& other, bool require_same_shape = false, bool require_same_device = false) const {
            if (!is_valid() || !other.is_valid()) {
                LOG_ERROR("Operation on invalid tensor");
                return false;
            }
            if (require_same_device && device_ != other.device()) {
                LOG_ERROR("Tensors must be on same device");
                return false;
            }
            if (require_same_shape && shape_ != other.shape()) {
                LOG_ERROR("Shape mismatch: {} vs {}", shape_.str(), other.shape_.str());
                return false;
            }
            // Check if broadcasting is valid (even when require_same_shape is false)
            if (!require_same_shape && shape_ != other.shape()) {
                // Compute broadcast shape and validate
                auto bcast_shape = broadcast_shape(other.shape());
                if (bcast_shape.rank() == 0 || bcast_shape.elements() == 0) {
                    LOG_ERROR("Incompatible shapes for broadcasting: {} vs {}", shape_.str(), other.shape_.str());
                    return false;
                }
            }
            return true;
        }

        bool validate_unary_op() const {
            if (!is_valid()) {
                LOG_ERROR("Unary operation on invalid tensor");
                return false;
            }
            return true;
        }

        bool validate_ternary_op(const Tensor& b, const Tensor& c) const {
            if (!is_valid() || !b.is_valid() || !c.is_valid()) {
                LOG_ERROR("Ternary operation on invalid tensor");
                return false;
            }
            if (device_ != b.device() || device_ != c.device()) {
                LOG_ERROR("All tensors must be on same device");
                return false;
            }
            return true;
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
                LOG_ERROR("accessor() only works on CPU tensors");
                return TensorAccessor<T, N>(nullptr, std::array<size_t, N>{});
            }
            if (!is_valid() || shape_.rank() != N) {
                LOG_ERROR("accessor() dimension mismatch: tensor has {} dims, requested {}",
                          shape_.rank(), N);
                return TensorAccessor<T, N>(nullptr, std::array<size_t, N>{});
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
                            DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = std::monostate{};
            return load(LoadOp::Empty, args);
        }

        static Tensor zeros(TensorShape shape, Device device = Device::CUDA,
                            DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = 0.0f;
            return load(LoadOp::Const, args);
        }

        static Tensor ones(TensorShape shape, Device device = Device::CUDA,
                           DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = 1.0f;
            return load(LoadOp::Const, args);
        }

        static Tensor full(TensorShape shape, float value, Device device = Device::CUDA,
                           DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = value;
            return load(LoadOp::Const, args);
        }

        static Tensor full_bool(TensorShape shape, bool value, Device device = Device::CUDA) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = DataType::Bool;
            args.args = value ? 1.0f : 0.0f;
            return load(LoadOp::Const, args);
        }

        static Tensor zeros_bool(TensorShape shape, Device device = Device::CUDA) {
            return full_bool(shape, false, device);
        }

        static Tensor ones_bool(TensorShape shape, Device device = Device::CUDA) {
            return full_bool(shape, true, device);
        }

        static Tensor rand(TensorShape shape, Device device = Device::CUDA,
                           DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = std::pair<float, float>{0.0f, 1.0f};
            return load(LoadOp::Random, args);
        }

        static Tensor randn(TensorShape shape, Device device = Device::CUDA,
                            DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = std::pair<float, float>{0.0f, 1.0f};
            return load(LoadOp::Normal, args);
        }

        static Tensor uniform(TensorShape shape, float low = 0.0f, float high = 1.0f,
                              Device device = Device::CUDA, DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = std::pair<float, float>{low, high};
            return load(LoadOp::Random, args);
        }

        static Tensor normal(TensorShape shape, float mean = 0.0f, float std = 1.0f,
                             Device device = Device::CUDA, DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = std::pair<float, float>{mean, std};
            return load(LoadOp::Normal, args);
        }

        static Tensor randint(TensorShape shape, int low, int high,
                              Device device = Device::CUDA, DataType dtype = DataType::Int32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = std::pair<int, int>{low, high};
            return load(LoadOp::Randint, args);
        }

        static Tensor bernoulli(TensorShape shape, float p = 0.5f,
                                Device device = Device::CUDA, DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = p;
            return load(LoadOp::Bernoulli, args);
        }

        static Tensor multinomial(const Tensor& weights, int num_samples,
                                  bool replacement = false);

        static Tensor arange(float end) {
            LoadArgs args;
            args.shape = TensorShape{};
            args.device = Device::CUDA;
            args.dtype = DataType::Float32;
            args.args = std::tuple<float, float, float>{0.0f, end, 1.0f};
            return load(LoadOp::Arange, args);
        }

        static Tensor arange(float start, float end, float step = 1.0f) {
            LoadArgs args;
            args.shape = TensorShape{};
            args.device = Device::CUDA;
            args.dtype = DataType::Float32;
            args.args = std::tuple<float, float, float>{start, end, step};
            return load(LoadOp::Arange, args);
        }

        static Tensor linspace(float start, float end, size_t steps, Device device = Device::CUDA);

        static Tensor eye(size_t n, Device device = Device::CUDA) {
            LoadArgs args;
            args.shape = TensorShape{n, n};
            args.device = device;
            args.dtype = DataType::Float32;
            args.args = std::monostate{};
            return load(LoadOp::Eye, args);
        }

        static Tensor eye(size_t m, size_t n, Device device = Device::CUDA) {
            LoadArgs args;
            args.shape = TensorShape{m, n};
            args.device = device;
            args.dtype = DataType::Float32;
            args.args = std::monostate{};
            return load(LoadOp::Eye, args);
        }

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

        void* raw_ptr() { return data_; }
        const void* raw_ptr() const { return data_; }

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

        size_t size(size_t dim) const {
            if (!is_valid())
                return 0;
            if (dim >= shape_.rank()) {
                LOG_ERROR("Dimension {} out of range for rank {}", dim, shape_.rank());
                return 0;
            }
            return shape_[dim];
        }

        // Memory operations
        Tensor clone() const;      // Deep copy
        Tensor contiguous() const; // Materialize to contiguous if strided
        Tensor to(Device device) const;
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
        Tensor neg() const {
            // Edge cases: return eager Tensor for invalid or empty tensors
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            // Lazy evaluation: return expression template (implicitly converts to Tensor)
            return UnaryExpr<TensorLeaf, ops::neg_op>(
                TensorLeaf(*this), ops::neg_op{}, shape_, device_, dtype_);
        }

        Tensor abs() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::abs_op>(
                TensorLeaf(*this), ops::abs_op{}, shape_, device_, dtype_);
        }

        Tensor sign() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::sign_op>(
                TensorLeaf(*this), ops::sign_op{}, shape_, device_, dtype_);
        }

        Tensor reciprocal() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::reciprocal_op>(
                TensorLeaf(*this), ops::reciprocal_op{}, shape_, device_, dtype_);
        }

        Tensor exp() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::exp_op>(
                TensorLeaf(*this), ops::exp_op{}, shape_, device_, dtype_);
        }

        Tensor exp2() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::exp2_op>(
                TensorLeaf(*this), ops::exp2_op{}, shape_, device_, dtype_);
        }

        Tensor log() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::log_op>(
                TensorLeaf(*this), ops::log_op{}, shape_, device_, dtype_);
        }

        Tensor log2() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::log2_op>(
                TensorLeaf(*this), ops::log2_op{}, shape_, device_, dtype_);
        }

        Tensor log10() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::log10_op>(
                TensorLeaf(*this), ops::log10_op{}, shape_, device_, dtype_);
        }

        Tensor log1p() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::log1p_op>(
                TensorLeaf(*this), ops::log1p_op{}, shape_, device_, dtype_);
        }

        Tensor sqrt() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::sqrt_op>(
                TensorLeaf(*this), ops::sqrt_op{}, shape_, device_, dtype_);
        }

        Tensor rsqrt() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::rsqrt_op>(
                TensorLeaf(*this), ops::rsqrt_op{}, shape_, device_, dtype_);
        }

        Tensor square() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::square_op>(
                TensorLeaf(*this), ops::square_op{}, shape_, device_, dtype_);
        }
        Tensor sin() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::sin_op>(
                TensorLeaf(*this), ops::sin_op{}, shape_, device_, dtype_);
        }

        Tensor cos() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::cos_op>(
                TensorLeaf(*this), ops::cos_op{}, shape_, device_, dtype_);
        }

        Tensor tan() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::tan_op>(
                TensorLeaf(*this), ops::tan_op{}, shape_, device_, dtype_);
        }

        Tensor asin() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::asin_op>(
                TensorLeaf(*this), ops::asin_op{}, shape_, device_, dtype_);
        }

        Tensor acos() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::acos_op>(
                TensorLeaf(*this), ops::acos_op{}, shape_, device_, dtype_);
        }

        Tensor atan() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::atan_op>(
                TensorLeaf(*this), ops::atan_op{}, shape_, device_, dtype_);
        }

        Tensor sinh() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::sinh_op>(
                TensorLeaf(*this), ops::sinh_op{}, shape_, device_, dtype_);
        }

        Tensor cosh() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::cosh_op>(
                TensorLeaf(*this), ops::cosh_op{}, shape_, device_, dtype_);
        }

        Tensor tanh() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::tanh_op>(
                TensorLeaf(*this), ops::tanh_op{}, shape_, device_, dtype_);
        }

        Tensor sigmoid() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::sigmoid_op>(
                TensorLeaf(*this), ops::sigmoid_op{}, shape_, device_, dtype_);
        }

        Tensor relu() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::relu_op>(
                TensorLeaf(*this), ops::relu_op{}, shape_, device_, dtype_);
        }

        Tensor gelu() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::gelu_op>(
                TensorLeaf(*this), ops::gelu_op{}, shape_, device_, dtype_);
        }

        Tensor swish() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::swish_op>(
                TensorLeaf(*this), ops::swish_op{}, shape_, device_, dtype_);
        }
        Tensor floor() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::floor_op>(
                TensorLeaf(*this), ops::floor_op{}, shape_, device_, dtype_);
        }

        Tensor ceil() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::ceil_op>(
                TensorLeaf(*this), ops::ceil_op{}, shape_, device_, dtype_);
        }

        Tensor round() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::round_op>(
                TensorLeaf(*this), ops::round_op{}, shape_, device_, dtype_);
        }

        Tensor trunc() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::trunc_op>(
                TensorLeaf(*this), ops::trunc_op{}, shape_, device_, dtype_);
        }

        Tensor isnan() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            return UnaryExpr<TensorLeaf, ops::isnan_op>(
                TensorLeaf(*this), ops::isnan_op{}, shape_, device_, DataType::Bool);
        }

        Tensor isinf() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            return UnaryExpr<TensorLeaf, ops::isinf_op>(
                TensorLeaf(*this), ops::isinf_op{}, shape_, device_, DataType::Bool);
        }

        Tensor isfinite() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            return UnaryExpr<TensorLeaf, ops::isfinite_op>(
                TensorLeaf(*this), ops::isfinite_op{}, shape_, device_, DataType::Bool);
        }

        Tensor logical_not() const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            return UnaryExpr<TensorLeaf, ops::logical_not_op>(
                TensorLeaf(*this), ops::logical_not_op{}, shape_, device_, DataType::Bool);
        }

        Tensor normalize(int dim = -1, float eps = 1e-12f) const;
        Tensor logit(float eps = 1e-7f) const;

        // ============= BINARY OPERATIONS (Template-based) =============

        // Arithmetic operations

        // New functor-based overloads for Tensor (zero enum overhead, lazy evaluation)
        Tensor add(const Tensor& other) const {
            if (!validate_binary_op(other, false, true)) {
                return Tensor();
            }
            auto broadcast_shape = this->broadcast_shape(other.shape());
            return BinaryExpr<TensorLeaf, TensorLeaf, ops::add_op>(
                TensorLeaf(*this), TensorLeaf(other), ops::add_op{},
                broadcast_shape, device_, dtype_);
        }

        Tensor sub(const Tensor& other) const {
            if (!validate_binary_op(other, false, true)) {
                return Tensor();
            }
            auto broadcast_shape = this->broadcast_shape(other.shape());
            return BinaryExpr<TensorLeaf, TensorLeaf, ops::sub_op>(
                TensorLeaf(*this), TensorLeaf(other), ops::sub_op{},
                broadcast_shape, device_, dtype_);
        }

        Tensor mul(const Tensor& other) const {
            if (!validate_binary_op(other, false, true)) {
                return Tensor();
            }
            auto broadcast_shape = this->broadcast_shape(other.shape());
            return BinaryExpr<TensorLeaf, TensorLeaf, ops::mul_op>(
                TensorLeaf(*this), TensorLeaf(other), ops::mul_op{},
                broadcast_shape, device_, dtype_);
        }

        Tensor div(const Tensor& other) const {
            if (!validate_binary_op(other, false, true)) {
                return Tensor();
            }
            auto broadcast_shape = this->broadcast_shape(other.shape());
            return BinaryExpr<TensorLeaf, TensorLeaf, ops::div_op>(
                TensorLeaf(*this), TensorLeaf(other), ops::div_op{},
                broadcast_shape, device_, dtype_);
        }

        Tensor pow(const Tensor& other) const {
            if (!is_valid() || numel() == 0 || !other.is_valid() || other.numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            auto result_shape = this->broadcast_shape(other.shape());
            return BinaryExpr<TensorLeaf, TensorLeaf, ops::pow_op>(
                TensorLeaf(*this), TensorLeaf(other), ops::pow_op{},
                result_shape, device_, dtype_);
        }

        Tensor mod(const Tensor& other) const {
            if (!is_valid() || numel() == 0 || !other.is_valid() || other.numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            auto result_shape = this->broadcast_shape(other.shape());
            return BinaryExpr<TensorLeaf, TensorLeaf, ops::mod_op>(
                TensorLeaf(*this), TensorLeaf(other), ops::mod_op{},
                result_shape, device_, dtype_);
        }

        Tensor maximum(const Tensor& other) const {
            if (!is_valid() || numel() == 0 || !other.is_valid() || other.numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            auto result_shape = this->broadcast_shape(other.shape());
            return BinaryExpr<TensorLeaf, TensorLeaf, ops::maximum_op>(
                TensorLeaf(*this), TensorLeaf(other), ops::maximum_op{},
                result_shape, device_, dtype_);
        }

        Tensor minimum(const Tensor& other) const {
            if (!is_valid() || numel() == 0 || !other.is_valid() || other.numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            auto result_shape = this->broadcast_shape(other.shape());
            return BinaryExpr<TensorLeaf, TensorLeaf, ops::minimum_op>(
                TensorLeaf(*this), TensorLeaf(other), ops::minimum_op{},
                result_shape, device_, dtype_);
        }

        // Template versions for scalars (lazy evaluation with scalar_right_op)
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor add(const T& other) const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::add_op, float>>(
                TensorLeaf(*this), ops::scalar_right_op<ops::add_op, float>(static_cast<float>(other)),
                shape_, device_, dtype_);
        }

        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor sub(const T& other) const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::sub_op, float>>(
                TensorLeaf(*this), ops::scalar_right_op<ops::sub_op, float>(static_cast<float>(other)),
                shape_, device_, dtype_);
        }

        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor mul(const T& other) const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::mul_op, float>>(
                TensorLeaf(*this), ops::scalar_right_op<ops::mul_op, float>(static_cast<float>(other)),
                shape_, device_, dtype_);
        }

        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor div(const T& other) const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::div_op, float>>(
                TensorLeaf(*this), ops::scalar_right_op<ops::div_op, float>(static_cast<float>(other)),
                shape_, device_, dtype_);
        }

        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor pow(const T& other) const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::pow_op, float>>(
                TensorLeaf(*this), ops::scalar_right_op<ops::pow_op, float>(static_cast<float>(other)),
                shape_, device_, dtype_);
        }

        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor mod(const T& other) const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::mod_op, float>>(
                TensorLeaf(*this), ops::scalar_right_op<ops::mod_op, float>(static_cast<float>(other)),
                shape_, device_, dtype_);
        }

        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor maximum(const T& other) const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::maximum_op, float>>(
                TensorLeaf(*this), ops::scalar_right_op<ops::maximum_op, float>(static_cast<float>(other)),
                shape_, device_, dtype_);
        }

        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor minimum(const T& other) const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, dtype_);
            }
            return UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::minimum_op, float>>(
                TensorLeaf(*this), ops::scalar_right_op<ops::minimum_op, float>(static_cast<float>(other)),
                shape_, device_, dtype_);
        }

        // Comparison operations (return Bool tensors)

        // Functor-based overloads for Tensor (zero enum overhead)
        Tensor eq(const Tensor& other) const {
            if (!is_valid() || numel() == 0 || !other.is_valid() || other.numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            auto result_shape = this->broadcast_shape(other.shape());
            return BinaryExpr<TensorLeaf, TensorLeaf, ops::equal_op>(
                TensorLeaf(*this), TensorLeaf(other), ops::equal_op{},
                result_shape, device_, DataType::Bool);
        }

        Tensor ne(const Tensor& other) const {
            if (!is_valid() || numel() == 0 || !other.is_valid() || other.numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            auto result_shape = this->broadcast_shape(other.shape());
            return BinaryExpr<TensorLeaf, TensorLeaf, ops::not_equal_op>(
                TensorLeaf(*this), TensorLeaf(other), ops::not_equal_op{},
                result_shape, device_, DataType::Bool);
        }

        Tensor lt(const Tensor& other) const {
            if (!is_valid() || numel() == 0 || !other.is_valid() || other.numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            auto result_shape = this->broadcast_shape(other.shape());
            return BinaryExpr<TensorLeaf, TensorLeaf, ops::less_op>(
                TensorLeaf(*this), TensorLeaf(other), ops::less_op{},
                result_shape, device_, DataType::Bool);
        }

        Tensor le(const Tensor& other) const {
            if (!is_valid() || numel() == 0 || !other.is_valid() || other.numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            auto result_shape = this->broadcast_shape(other.shape());
            return BinaryExpr<TensorLeaf, TensorLeaf, ops::less_equal_op>(
                TensorLeaf(*this), TensorLeaf(other), ops::less_equal_op{},
                result_shape, device_, DataType::Bool);
        }

        Tensor gt(const Tensor& other) const {
            if (!is_valid() || numel() == 0 || !other.is_valid() || other.numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            auto result_shape = this->broadcast_shape(other.shape());
            return BinaryExpr<TensorLeaf, TensorLeaf, ops::greater_op>(
                TensorLeaf(*this), TensorLeaf(other), ops::greater_op{},
                result_shape, device_, DataType::Bool);
        }

        Tensor ge(const Tensor& other) const {
            if (!is_valid() || numel() == 0 || !other.is_valid() || other.numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            auto result_shape = this->broadcast_shape(other.shape());
            return BinaryExpr<TensorLeaf, TensorLeaf, ops::greater_equal_op>(
                TensorLeaf(*this), TensorLeaf(other), ops::greater_equal_op{},
                result_shape, device_, DataType::Bool);
        }

        // Template versions for scalars (direct functor calls - zero enum overhead!)
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor eq(const T& other) const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            return UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::equal_op, float>>(
                TensorLeaf(*this), ops::scalar_right_op<ops::equal_op, float>(static_cast<float>(other)),
                shape_, device_, DataType::Bool);
        }

        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor ne(const T& other) const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            return UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::not_equal_op, float>>(
                TensorLeaf(*this), ops::scalar_right_op<ops::not_equal_op, float>(static_cast<float>(other)),
                shape_, device_, DataType::Bool);
        }

        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor lt(const T& other) const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            return UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::less_op, float>>(
                TensorLeaf(*this), ops::scalar_right_op<ops::less_op, float>(static_cast<float>(other)),
                shape_, device_, DataType::Bool);
        }

        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor le(const T& other) const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            return UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::less_equal_op, float>>(
                TensorLeaf(*this), ops::scalar_right_op<ops::less_equal_op, float>(static_cast<float>(other)),
                shape_, device_, DataType::Bool);
        }

        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor gt(const T& other) const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            return UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::greater_op, float>>(
                TensorLeaf(*this), ops::scalar_right_op<ops::greater_op, float>(static_cast<float>(other)),
                shape_, device_, DataType::Bool);
        }

        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor ge(const T& other) const {
            if (!is_valid() || numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            return UnaryExpr<TensorLeaf, ops::scalar_right_op<ops::greater_equal_op, float>>(
                TensorLeaf(*this), ops::scalar_right_op<ops::greater_equal_op, float>(static_cast<float>(other)),
                shape_, device_, DataType::Bool);
        }

        // Logical operations (Tensor only, Bool -> Bool)
        Tensor logical_and(const Tensor& other) const {
            if (!is_valid() || numel() == 0 || !other.is_valid() || other.numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            auto result_shape = this->broadcast_shape(other.shape());
            return BinaryExpr<TensorLeaf, TensorLeaf, ops::logical_and_op>(
                TensorLeaf(*this), TensorLeaf(other), ops::logical_and_op{},
                result_shape, device_, DataType::Bool);
        }

        Tensor logical_or(const Tensor& other) const {
            if (!is_valid() || numel() == 0 || !other.is_valid() || other.numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            auto result_shape = this->broadcast_shape(other.shape());
            return BinaryExpr<TensorLeaf, TensorLeaf, ops::logical_or_op>(
                TensorLeaf(*this), TensorLeaf(other), ops::logical_or_op{},
                result_shape, device_, DataType::Bool);
        }

        Tensor logical_xor(const Tensor& other) const {
            if (!is_valid() || numel() == 0 || !other.is_valid() || other.numel() == 0) {
                if (!is_valid())
                    return Tensor();
                return Tensor::empty(shape_, device_, DataType::Bool);
            }
            auto result_shape = this->broadcast_shape(other.shape());
            return BinaryExpr<TensorLeaf, TensorLeaf, ops::logical_xor_op>(
                TensorLeaf(*this), TensorLeaf(other), ops::logical_xor_op{},
                result_shape, device_, DataType::Bool);
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
        float sum_scalar() const { return sum().item(); }
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
            if (!is_valid() || numel() != 1) {
                LOG_ERROR("item<T>() requires a valid single-element tensor");
                return T{};
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
                LOG_INFO("{}: {} µs", name, duration.count());
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
    class TensorRowProxy {
    private:
        Tensor* tensor_;
        size_t row_index_;

    public:
        TensorRowProxy(Tensor* tensor, size_t row_index)
            : tensor_(tensor),
              row_index_(row_index) {
            if (tensor_ && tensor_->is_valid() && row_index_ >= tensor_->shape()[0]) {
                LOG_ERROR("Row index {} out of bounds for dimension 0 with size {}",
                          row_index_, tensor_->shape()[0]);
            }
        }

        // ============= 2D Access: tensor[i][j] =============

        float& operator[](size_t col_index) {
            if (!tensor_ || !tensor_->is_valid()) {
                LOG_ERROR("TensorRowProxy: invalid tensor pointer");
                static float dummy = 0.0f;
                return dummy;
            }

            if (tensor_->shape().rank() < 2) {
                LOG_ERROR("TensorRowProxy: Cannot use tensor[i][j] on {}-D tensor with shape {}. "
                          "Use tensor[i] for 1-D access, or tensor.unsqueeze() to add dimensions.",
                          tensor_->shape().rank(), tensor_->shape().str());
                static float dummy = 0.0f;
                return dummy;
            }

            if (col_index >= tensor_->shape()[1]) {
                LOG_ERROR("Column index {} out of bounds for dimension 1 with size {}",
                          col_index, tensor_->shape()[1]);
                static float dummy = 0.0f;
                return dummy;
            }

            if (tensor_->device() != Device::CPU) {
                // For CUDA tensors in read context, we need to provide a workaround
                // Store the value in a thread-local static so the reference remains valid
                thread_local static float cuda_read_value = 0.0f;

                size_t linear_idx = row_index_ * tensor_->shape()[1] + col_index;

                cudaError_t err = cudaMemcpy(
                    &cuda_read_value,
                    tensor_->ptr<float>() + linear_idx,
                    sizeof(float),
                    cudaMemcpyDeviceToHost);

                if (err != cudaSuccess) {
                    LOG_ERROR("CUDA memcpy failed in TensorRowProxy::operator[]: {}",
                              cudaGetErrorString(err));
                    cuda_read_value = 0.0f;
                }

                return cuda_read_value;
            }

            return tensor_->at({row_index_, col_index});
        }

        float operator[](size_t col_index) const {
            if (!tensor_ || !tensor_->is_valid()) {
                LOG_ERROR("TensorRowProxy: invalid tensor pointer");
                return 0.0f;
            }

            if (tensor_->shape().rank() < 2) {
                LOG_ERROR("TensorRowProxy: tensor rank {} < 2", tensor_->shape().rank());
                return 0.0f;
            }

            if (col_index >= tensor_->shape()[1]) {
                LOG_ERROR("Column index {} out of bounds for dimension 1 with size {}",
                          col_index, tensor_->shape()[1]);
                return 0.0f;
            }

            // Calculate linear index: row_index * num_cols + col_index
            size_t linear_idx = row_index_ * tensor_->shape()[1] + col_index;

            if (tensor_->device() == Device::CUDA) {
                float value = 0.0f;
                cudaError_t err = cudaMemcpy(
                    &value,
                    tensor_->ptr<float>() + linear_idx,
                    sizeof(float),
                    cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    LOG_ERROR("CUDA memcpy failed in TensorRowProxy::operator[]: {}",
                              cudaGetErrorString(err));
                    return 0.0f;
                }
                return value;
            } else {
                return tensor_->ptr<float>()[linear_idx];
            }
        }

        // ============= 1D Access: Extract Value =============

        // Explicit item extraction (no ambiguity, always works)
        float item() const {
            if (!tensor_ || !tensor_->is_valid()) {
                LOG_ERROR("TensorRowProxy::item(): invalid tensor pointer");
                return 0.0f;
            }

            // Handle 2D tensors with shape [N, 1] (like nonzero() output)
            if (tensor_->shape().rank() == 2 && tensor_->shape()[1] == 1) {
                // Convert to actual Tensor row first
                Tensor row_tensor = static_cast<Tensor>(*this);
                return row_tensor.item();
            }

            // Standard 1D case
            if (tensor_->shape().rank() != 1) {
                LOG_ERROR("TensorRowProxy::item(): only valid for 1D tensors, got rank {}",
                          tensor_->shape().rank());
                return 0.0f;
            }

            if (row_index_ >= tensor_->numel()) {
                LOG_ERROR("TensorRowProxy::item(): index {} out of bounds for size {}",
                          row_index_, tensor_->numel());
                return 0.0f;
            }

            if (tensor_->device() == Device::CUDA) {
                float value = 0.0f;
                cudaError_t err = cudaMemcpy(
                    &value,
                    tensor_->ptr<float>() + row_index_,
                    sizeof(float),
                    cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    LOG_ERROR("CUDA memcpy failed in TensorRowProxy::item(): {}",
                              cudaGetErrorString(err));
                    return 0.0f;
                }
                return value;
            } else {
                return tensor_->ptr<float>()[row_index_];
            }
        }

        operator float() const {
            if (!tensor_ || !tensor_->is_valid()) {
                LOG_ERROR("TensorRowProxy: invalid tensor pointer in float conversion");
                return 0.0f;
            }

            // Allow implicit conversion for 1D tensors OR 2D [N,1] tensors (like nonzero output)
            if (tensor_->shape().rank() == 1) {
                return item();
            } else if (tensor_->shape().rank() == 2 && tensor_->shape()[1] == 1) {
                // Special case for [N, 1] shaped tensors
                return item();
            } else {
                LOG_ERROR("Implicit float conversion only valid for 1D or [N,1] tensors, got rank {} with shape {}. Use .item() or convert to Tensor first.",
                          tensor_->shape().rank(), tensor_->shape().str());
                return 0.0f;
            }
        }

        // Template version for type specification
        template <typename T = float>
        T item_as() const {
            if (!tensor_ || !tensor_->is_valid()) {
                LOG_ERROR("TensorRowProxy::item_as(): invalid tensor pointer");
                return T{};
            }

            // Handle 2D tensors with shape [N, 1] (like nonzero() output)
            if (tensor_->shape().rank() == 2 && tensor_->shape()[1] == 1) {
                // Convert to actual Tensor row first, then extract item
                Tensor row_tensor = static_cast<Tensor>(*this);
                return row_tensor.item<T>();
            }

            // Standard 1D case
            if (tensor_->shape().rank() != 1) {
                LOG_ERROR("TensorRowProxy::item_as(): only valid for 1D or [N,1] tensors, got rank {}",
                          tensor_->shape().rank());
                return T{};
            }

            if (row_index_ >= tensor_->numel()) {
                LOG_ERROR("TensorRowProxy::item_as(): index {} out of bounds for size {}",
                          row_index_, tensor_->numel());
                return T{};
            }

            if (tensor_->device() == Device::CUDA) {
                T value{};

                // Calculate proper offset based on dtype
                size_t type_size = dtype_size(tensor_->dtype());
                const void* src_ptr = static_cast<const char*>(tensor_->raw_ptr()) + row_index_ * type_size;

                cudaError_t err = cudaMemcpy(
                    &value,
                    src_ptr,
                    sizeof(T),
                    cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    LOG_ERROR("CUDA memcpy failed in TensorRowProxy::item_as(): {}",
                              cudaGetErrorString(err));
                    return T{};
                }
                return value;
            } else {
                // Direct pointer access for CPU tensors
                // Properly cast based on tensor's actual dtype
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
                    // Proper Int64 handling
                    const int64_t* data = reinterpret_cast<const int64_t*>(tensor_->raw_ptr());
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

                LOG_ERROR("Unsupported dtype/type combination for item_as()");
                return T{};
            }
        }

        // Specialized item_as for common types
        int item_int() const { return item_as<int>(); }
        int64_t item_int64() const { return item_as<int64_t>(); }

        // ============= Conversion to Tensor =============

        operator Tensor() const {
            if (!tensor_ || !tensor_->is_valid()) {
                LOG_ERROR("TensorRowProxy: invalid tensor pointer");
                return Tensor();
            }

            if (tensor_->shape().rank() > 1) {
                // For multi-dimensional tensors, manually copy the row data
                // This ensures we get a true independent copy, not a view

                // Calculate the shape of one row (all dims except first)
                std::vector<size_t> row_shape;
                for (size_t i = 1; i < tensor_->shape().rank(); ++i) {
                    row_shape.push_back(tensor_->shape()[i]);
                }

                // Create new independent tensor with correct shape
                auto result = Tensor::empty(TensorShape(row_shape),
                                            tensor_->device(),
                                            tensor_->dtype());

                // Calculate how many elements in one row
                size_t row_elements = 1;
                for (size_t i = 1; i < tensor_->shape().rank(); ++i) {
                    row_elements *= tensor_->shape()[i];
                }

                // Calculate byte offset to the start of this row
                size_t byte_offset = row_index_ * row_elements * dtype_size(tensor_->dtype());
                size_t copy_bytes = row_elements * dtype_size(tensor_->dtype());

                // Perform deep copy of row data
                if (tensor_->device() == Device::CUDA) {
                    cudaError_t err = cudaMemcpy(
                        result.raw_ptr(),
                        static_cast<const char*>(tensor_->raw_ptr()) + byte_offset,
                        copy_bytes,
                        cudaMemcpyDeviceToDevice);
                    if (err != cudaSuccess) {
                        LOG_ERROR("CUDA memcpy failed in TensorRowProxy tensor conversion: {}",
                                  cudaGetErrorString(err));
                        return Tensor();
                    }
                } else {
                    std::memcpy(
                        result.raw_ptr(),
                        static_cast<const char*>(tensor_->raw_ptr()) + byte_offset,
                        copy_bytes);
                }

                return result;
            }

            // For 1D tensors, return a scalar tensor
            float val = item();

            auto result = Tensor::empty({1}, tensor_->device(), tensor_->dtype());

            if (tensor_->device() == Device::CUDA) {
                cudaMemcpy(result.raw_ptr(), &val, sizeof(float), cudaMemcpyHostToDevice);
            } else {
                *result.ptr<float>() = val;
            }

            return result.squeeze();
        }

        // ============= Assignment Operators =============

        TensorRowProxy& operator=(const TensorRowProxy& other) {
            if (this == &other) {
                return *this; // Self-assignment
            }

            // Convert other proxy to an independent Tensor first
            // This is crucial for cases like: tensor[4] = tensor[0]
            // Without this, we'd just copy the proxy metadata (pointer + index)
            // instead of copying the actual data
            Tensor other_copy = other;    // Triggers operator Tensor()
            return operator=(other_copy); // Use the Tensor assignment operator
        }

        // Assignment from Tensor
        TensorRowProxy& operator=(const Tensor& other) {
            if (!tensor_ || !tensor_->is_valid()) {
                return *this;
            }

            if (tensor_->shape().rank() > 1) {
                // Multi-dimensional: assign entire row slice

                // Calculate expected shape (all dims except first)
                std::vector<size_t> slice_shape;
                for (size_t i = 1; i < tensor_->shape().rank(); ++i) {
                    slice_shape.push_back(tensor_->shape()[i]);
                }
                TensorShape expected_shape(slice_shape);

                // Validate shape match
                if (other.shape() != expected_shape) {
                    LOG_ERROR("Shape mismatch in row assignment: expected {}, got {}",
                              expected_shape.str(), other.shape().str());
                    return *this;
                }

                // Calculate number of elements per row
                size_t row_elements = 1;
                for (size_t i = 1; i < tensor_->shape().rank(); ++i) {
                    row_elements *= tensor_->shape()[i];
                }

                // Calculate byte offset for this row
                size_t byte_offset = row_index_ * row_elements * dtype_size(tensor_->dtype());
                size_t copy_bytes = row_elements * dtype_size(tensor_->dtype());

                // CRITICAL: Always clone to ensure independence from source
                // This prevents issues when assigning from another row of the same tensor
                auto other_copy = (other.device() == tensor_->device())
                                      ? other.clone()
                                      : other.to(tensor_->device());

                // Copy data based on device
                if (tensor_->device() == Device::CUDA) {
                    cudaError_t err = cudaMemcpy(
                        static_cast<char*>(tensor_->raw_ptr()) + byte_offset,
                        other_copy.raw_ptr(),
                        copy_bytes,
                        cudaMemcpyDeviceToDevice);
                    if (err != cudaSuccess) {
                        LOG_ERROR("CUDA memcpy failed in row assignment: {}",
                                  cudaGetErrorString(err));
                    }
                } else {
                    std::memcpy(
                        static_cast<char*>(tensor_->raw_ptr()) + byte_offset,
                        other_copy.raw_ptr(),
                        copy_bytes);
                }
            } else {
                // 1D: assign single element
                if (other.numel() != 1) {
                    LOG_ERROR("Cannot assign tensor with {} elements to single position",
                              other.numel());
                    return *this;
                }

                float val = other.item();

                if (tensor_->device() == Device::CUDA) {
                    cudaMemcpy(
                        tensor_->ptr<float>() + row_index_,
                        &val,
                        sizeof(float),
                        cudaMemcpyHostToDevice);
                } else {
                    tensor_->ptr<float>()[row_index_] = val;
                }
            }
            return *this;
        }

        // Assignment from scalar (for 1D tensors)
        TensorRowProxy& operator=(float value) {
            if (!tensor_ || !tensor_->is_valid()) {
                return *this;
            }

            if (tensor_->shape().rank() != 1) {
                LOG_ERROR("Float assignment only valid for 1D tensors, got rank {}",
                          tensor_->shape().rank());
                return *this;
            }

            if (row_index_ >= tensor_->numel()) {
                LOG_ERROR("Index {} out of bounds for size {}", row_index_, tensor_->numel());
                return *this;
            }

            if (tensor_->device() == Device::CUDA) {
                cudaMemcpy(
                    tensor_->ptr<float>() + row_index_,
                    &value,
                    sizeof(float),
                    cudaMemcpyHostToDevice);
            } else {
                tensor_->ptr<float>()[row_index_] = value;
            }
            return *this;
        }

        // ============= Arithmetic Operations with TensorRowProxy =============

        Tensor operator-(const TensorRowProxy& other) const {
            return Tensor(*this).sub(Tensor(other));
        }

        Tensor operator+(const TensorRowProxy& other) const {
            return Tensor(*this).add(Tensor(other));
        }

        Tensor operator*(const TensorRowProxy& other) const {
            return Tensor(*this).mul(Tensor(other));
        }

        Tensor operator/(const TensorRowProxy& other) const {
            return Tensor(*this).div(Tensor(other));
        }

        // ============= Arithmetic Operations with Scalars =============

        Tensor operator-(float scalar) const {
            return Tensor(*this).sub(scalar);
        }

        Tensor operator+(float scalar) const {
            return Tensor(*this).add(scalar);
        }

        Tensor operator*(float scalar) const {
            return Tensor(*this).mul(scalar);
        }

        Tensor operator/(float scalar) const {
            return Tensor(*this).div(scalar);
        }

        // ============= Unary Operations =============

        Tensor operator-() const {
            return Tensor(*this).neg();
        }

        Tensor pow(float exponent) const {
            return Tensor(*this).pow(exponent);
        }

        Tensor sqrt() const {
            return Tensor(*this).sqrt();
        }

        Tensor abs() const {
            return Tensor(*this).abs();
        }

        Tensor neg() const {
            return Tensor(*this).neg();
        }

        Tensor sum() const {
            return Tensor(*this).sum();
        }

        Tensor mean() const {
            return Tensor(*this).mean();
        }

        Tensor square() const {
            return Tensor(*this).square();
        }
    };

    // Implementation of Tensor::operator[]
    inline TensorRowProxy Tensor::operator[](size_t index) {
        if (!is_valid()) {
            LOG_ERROR("operator[] on invalid tensor");
        }
        if (is_valid() && index >= shape_[0]) {
            LOG_ERROR("Index {} out of bounds for dimension 0 with size {}", index, shape_[0]);
        }
        return TensorRowProxy(this, index);
    }

    inline const TensorRowProxy Tensor::operator[](size_t index) const {
        if (!is_valid()) {
            LOG_ERROR("operator[] on invalid tensor");
        }
        if (is_valid() && index >= shape_[0]) {
            LOG_ERROR("Index {} out of bounds for dimension 0 with size {}", index, shape_[0]);
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