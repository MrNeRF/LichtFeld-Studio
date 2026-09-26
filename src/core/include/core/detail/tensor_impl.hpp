/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/assert.hpp"
#include "core/cuda_safe_format.hpp"
#include "core/cuda_types.hpp"
#include "core/detail/tensor_half.hpp"
#include "core/gpu_backend_fwd.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <concepts>
#include <cstring>
#include <deque>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "core/tensor_cuda_interop.hpp"
#include "descriptors.hpp"
#include "gpu_backend_ops.hpp"
#include "lazy_config.hpp"
#include "lazy_ir.hpp"
#include "pointwise_lowering.hpp"
// Functors and expression types stay visible because caller translation units
// compose map() operations and instantiate lazy expression return types.
#include "tensor_functors.hpp"

#include "core/export.hpp"

namespace lfs::core::tensor_ops {
    LFS_CORE_API void record_tensor_kernel_launch(uint64_t n) noexcept;
}

namespace lfs::core {

    struct LazyExprState;
    inline void pin_operands(std::initializer_list<const Tensor*> tensors);

    namespace detail {
        constexpr size_t tensor_logical_rank(const size_t physical_rank) {
            return physical_rank == 0 ? 1 : physical_rank;
        }

        constexpr int resolve_tensor_dim(const int dim, const size_t physical_rank) {
            return dim < 0
                       ? static_cast<int>(tensor_logical_rank(physical_rank)) + dim
                       : dim;
        }

        constexpr bool tensor_dim_is_valid(const int resolved_dim,
                                           const size_t physical_rank) {
            return resolved_dim >= 0 &&
                   resolved_dim < static_cast<int>(tensor_logical_rank(physical_rank));
        }

        template <typename T>
        constexpr const char* tensor_cpp_type_name() {
            using Value = std::remove_cv_t<T>;
            if constexpr (std::is_void_v<Value>)
                return "void";
            else if constexpr (std::is_same_v<Value, float>)
                return "float";
            else if constexpr (std::is_same_v<Value, detail::tensor_half_t>)
                return "__half";
            else if constexpr (std::is_same_v<Value, int> || std::is_same_v<Value, int32_t>)
                return "int32";
            else if constexpr (std::is_same_v<Value, uint32_t>)
                return "uint32";
            else if constexpr (std::is_same_v<Value, int64_t>)
                return "int64";
            else if constexpr (std::is_same_v<Value, bool>)
                return "bool";
            else if constexpr (std::is_same_v<Value, unsigned char> ||
                               std::is_same_v<Value, uint8_t>)
                return "uint8";
            else
                return "unsupported";
        }
    } // namespace detail

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
        Flip = 5,
        Transpose = 6,
        Squeeze = 7,
        Unsqueeze = 8,
        Flatten = 9,
        Cat = 10,
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

    // Stack-resident ranked size list (rank ≤ MAX_TENSOR_RANK). Replaces heap
    // std::vector for TensorShape dims and Tensor strides.
    struct RankedDims {
        std::array<size_t, MAX_TENSOR_RANK> values{};
        size_t rank = 0;

        RankedDims() = default;
        RankedDims(std::initializer_list<size_t> dims) { assign(dims); }
        explicit RankedDims(const std::vector<size_t>& dims) { assign(dims); }
        explicit RankedDims(std::span<const size_t> dims) { assign(dims); }

        size_t size() const noexcept { return rank; }
        bool empty() const noexcept { return rank == 0; }
        size_t& operator[](size_t i) noexcept { return values[i]; }
        size_t operator[](size_t i) const noexcept { return values[i]; }
        size_t* data() noexcept { return values.data(); }
        const size_t* data() const noexcept { return values.data(); }

        size_t* begin() noexcept { return values.data(); }
        size_t* end() noexcept { return values.data() + rank; }
        const size_t* begin() const noexcept { return values.data(); }
        const size_t* end() const noexcept { return values.data() + rank; }
        const size_t* cbegin() const noexcept { return begin(); }
        const size_t* cend() const noexcept { return end(); }

        void clear() noexcept { rank = 0; }

        void assign(std::span<const size_t> dims) {
            LFS_ASSERT_MSG(dims.size() <= MAX_TENSOR_RANK,
                           "Tensor rank exceeds MAX_TENSOR_RANK");
            rank = dims.size();
            if (rank > 0) {
                std::copy_n(dims.begin(), rank, values.begin());
            }
        }
        void assign(const std::vector<size_t>& dims) {
            assign(std::span<const size_t>(dims.data(), dims.size()));
        }
        void assign(std::initializer_list<size_t> dims) {
            assign(std::span<const size_t>(dims.begin(), dims.size()));
        }

        RankedDims& operator=(std::span<const size_t> dims) {
            assign(dims);
            return *this;
        }
        RankedDims& operator=(const std::vector<size_t>& dims) {
            assign(dims);
            return *this;
        }
        RankedDims& operator=(std::initializer_list<size_t> dims) {
            assign(dims);
            return *this;
        }

        operator std::span<const size_t>() const noexcept {
            return {values.data(), rank};
        }
        // Enables `std::vector<size_t> v = shape.dims()` without a heap on the
        // shape itself; only the destination vector allocates.
        operator std::vector<size_t>() const {
            return std::vector<size_t>(begin(), end());
        }

        bool operator==(const RankedDims& other) const noexcept {
            return rank == other.rank &&
                   std::equal(begin(), end(), other.begin());
        }
        bool operator!=(const RankedDims& other) const noexcept {
            return !(*this == other);
        }
        bool operator==(const std::vector<size_t>& other) const noexcept {
            return rank == other.size() &&
                   std::equal(begin(), end(), other.begin());
        }
        bool operator!=(const std::vector<size_t>& other) const noexcept {
            return !(*this == other);
        }
    };

    inline bool operator==(const std::vector<size_t>& lhs, const RankedDims& rhs) noexcept {
        return rhs == lhs;
    }
    inline bool operator!=(const std::vector<size_t>& lhs, const RankedDims& rhs) noexcept {
        return rhs != lhs;
    }

    class LFS_CORE_API TensorShape {
    private:
        RankedDims dims_;
        size_t total_elements_ = 1;

    public:
        TensorShape() = default;
        TensorShape(std::initializer_list<size_t> dims) : dims_(dims) {
            compute_total();
        }
        explicit TensorShape(const std::vector<size_t>& dims) : dims_(dims) {
            compute_total();
        }
        explicit TensorShape(std::span<const size_t> dims) : dims_(dims) {
            compute_total();
        }
        explicit TensorShape(const RankedDims& dims) : dims_(dims) {
            compute_total();
        }

        size_t rank() const { return dims_.rank; }
        size_t operator[](size_t i) const {
            if (i >= dims_.rank) {
                throw std::out_of_range(
                    "Shape index " + std::to_string(i) + " out of range for rank " +
                    std::to_string(dims_.rank));
            }
            return dims_[i];
        }
        size_t elements() const { return total_elements_; }
        // Span-compatible view into stack storage (no heap).
        const RankedDims& dims() const { return dims_; }

        // Row-major contiguous strides — stack only, no heap allocation.
        RankedDims strides() const {
            RankedDims result;
            if (dims_.empty()) {
                return result;
            }
            result.rank = dims_.rank;
            result[dims_.rank - 1] = 1;
            for (int i = static_cast<int>(dims_.rank) - 2; i >= 0; --i) {
                result[static_cast<size_t>(i)] =
                    result[static_cast<size_t>(i + 1)] * dims_[static_cast<size_t>(i + 1)];
            }
            return result;
        }

        bool operator==(const TensorShape& other) const { return dims_ == other.dims_; }
        bool operator!=(const TensorShape& other) const { return !(*this == other); }

        std::string str() const;

    private:
        void compute_total() {
            LFS_ASSERT_MSG(dims_.rank <= MAX_TENSOR_RANK,
                           "Tensor rank exceeds MAX_TENSOR_RANK");
            if (dims_.empty()) {
                total_elements_ = 1;
            } else {
                total_elements_ = 1;
                for (size_t i = 0; i < dims_.rank; ++i) {
                    const size_t d = dims_[i];
                    LFS_ASSERT_MSG(d == 0 || total_elements_ <= std::numeric_limits<size_t>::max() / d,
                                   "TensorShape element count overflow");
                    total_elements_ *= d;
                }
            }
        }
    };

    namespace tensor_contract {

        LFS_CORE_API void require_valid(
            const Tensor& tensor,
            std::string_view operation,
            std::string_view role,
            SourceSite location);

        LFS_CORE_API void require_same_device(
            const Tensor& reference,
            const Tensor& other,
            std::string_view operation,
            std::string_view reference_role,
            std::string_view other_role,
            SourceSite location);

        LFS_CORE_API void require_dtype(
            const Tensor& tensor,
            DataType expected,
            std::string_view operation,
            std::string_view role,
            SourceSite location);

        LFS_CORE_API void require_dtype(
            const Tensor& tensor,
            std::initializer_list<DataType> expected,
            std::string_view operation,
            std::string_view role,
            SourceSite location);

        LFS_CORE_API void require_shape(
            const Tensor& reference,
            const Tensor& other,
            std::string_view operation,
            std::string_view reference_role,
            std::string_view other_role,
            SourceSite location);

        LFS_CORE_API void require_shape(
            const Tensor& tensor,
            const TensorShape& expected,
            std::string_view operation,
            std::string_view role,
            SourceSite location);

    } // namespace tensor_contract

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
        Device device = Device::GPU;
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
        void generate_cuda_normal(float* output, size_t count, float mean, float std,
                                  cudaStream_t stream);
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

    enum class StorageAccountingKind : uint8_t {
        CudaDirect,
        CudaExternal,
        VulkanOwned,
        VulkanExternal,
        MetalOwned,
    };

    struct GpuStorageDescriptor {
        uint64_t native_buffer;
        uint64_t native_allocation;
        uint64_t native_context;
        uint64_t base_address;
        uint64_t byte_size;
        StorageAccountingKind accounting_kind;
    };
    static_assert(std::is_trivial_v<GpuStorageDescriptor>);
    static_assert(std::is_standard_layout_v<GpuStorageDescriptor>);

    struct StorageMeta {
        GpuBackend backend = GpuBackend::CUDA;
        GpuStorageDescriptor gpu_descriptor{};
        std::atomic<uint64_t> pending_recorder{0};
        std::atomic<uint64_t> pending_value{0};
        std::atomic<uint64_t> generation{0};
        std::atomic<uint32_t> pending_lazy_snapshots{0};
        std::mutex lazy_snapshot_mutex;
        std::vector<std::weak_ptr<Tensor>> lazy_snapshots;
        std::string external_kind;
        std::shared_ptr<void> external_owner;
        mutable std::mutex vulkan_interop_mutex;
        mutable std::unordered_map<void*, std::shared_ptr<void>> vulkan_interop_owners;
        // Exportable packed-SoA provenance. When set, bind sites
        // re-resolve the device pointer through the live control block instead of
        // trusting the baked data_ pointer across a capacity grow.
        // exportable_control holds shared_ptr<SplatExportableStorage::Control>.
        std::shared_ptr<void> exportable_control;
        std::uint32_t exportable_region = 0;
        std::uint64_t exportable_bound_generation = 0;
    };

    namespace internal {
        // Synchronous host copies run on the legacy default stream, which a
        // non-blocking home stream neither waits for nor is waited on by. A
        // download must follow the home stream's producer; an upload from
        // pageable memory returns while its DMA is still in flight, so the home
        // stream must wait for it. Both orderings are event edges; the host
        // never waits on the home stream, so a gated stream cannot deadlock.
        LFS_LOCAL_SYMBOL void order_legacy_after_home(const Tensor& tensor);
        LFS_LOCAL_SYMBOL void order_home_after_legacy(const Tensor& tensor);
        LFS_LOCAL_SYMBOL GpuBackend resolve_new_gpu_storage_backend();
        // Cache trimming for every GPU backend that has a live context; a
        // backend that was never initialized is not brought up to be trimmed.
        LFS_LOCAL_SYMBOL void trim_live_gpu_backends();
        LFS_LOCAL_SYMBOL void trim_live_gpu_backends_if_reserved_unused_exceeds(size_t threshold_bytes);
        LFS_CORE_API Tensor allocate_like(const Tensor& input,
                                          const TensorShape& shape,
                                          DataType dtype);
        LFS_CORE_API Tensor allocate_like(const Tensor& input,
                                          const TensorShape& shape,
                                          DataType dtype,
                                          float value);
        LFS_CORE_API Tensor allocate_zeros_like(const Tensor& input,
                                                const TensorShape& shape,
                                                DataType dtype);
        LFS_CORE_API Tensor allocate_ones_like(const Tensor& input,
                                               const TensorShape& shape,
                                               DataType dtype);
        LFS_CORE_API Tensor allocate_rand_like(const Tensor& input,
                                               const TensorShape& shape,
                                               DataType dtype);
        LFS_CORE_API Tensor allocate_randn_like(const Tensor& input,
                                                const TensorShape& shape,
                                                DataType dtype);
        LFS_CORE_API Tensor copy_to_backend(const Tensor& source, GpuBackend target);
        [[noreturn]] LFS_CORE_API void throw_gpu_backend_mismatch(const Tensor& reference,
                                                                  const Tensor& other,
                                                                  std::string_view operation);
        inline void require_same_gpu_backend(const Tensor& reference,
                                             const Tensor& other,
                                             std::string_view operation);
        LFS_CORE_API void read_scalar(const Tensor& tensor, size_t element_index,
                                      void* output, size_t bytes);
        LFS_CORE_API void preserve_lazy_snapshots_before_write(Tensor& tensor);
        LFS_CORE_API std::shared_ptr<Tensor> lazy_executor_snapshot_operand(const Tensor& source);
        inline GpuBackend gpu_backend_tag(const Tensor& tensor);
        inline bool shares_storage(const Tensor& left, const Tensor& right);
    } // namespace internal

} // namespace lfs::core

// Include expression template declarations (forward declarations only)
#include "tensor_expr.hpp"

namespace lfs::core {

    struct TensorVulkanBuffer;
    namespace internal {
        std::optional<TensorVulkanBuffer> native_vulkan_buffer(const Tensor& tensor);
    }
    LFS_CORE_API std::optional<TensorVulkanBuffer> tensor_vulkan_buffer(const Tensor& tensor);

    class LFS_CORE_API Tensor {
    private:
        friend struct internal::LazyIrTensorAccess;
        friend class TensorLeaf;
        friend void pin_operands(std::initializer_list<const Tensor*> tensors);
        friend bool internal::shares_storage(const Tensor& left, const Tensor& right);
        friend std::shared_ptr<Tensor> internal::lazy_executor_snapshot_operand(
            const Tensor& source);
        friend Tensor broadcast_to(const Tensor& src, const TensorShape& target);
        friend LFS_CORE_API std::optional<GpuBackend> gpu_backend_of(const Tensor& tensor);
        friend LFS_CORE_API std::optional<TensorVulkanBuffer> tensor_vulkan_buffer(const Tensor& tensor);
        friend std::optional<TensorVulkanBuffer> internal::native_vulkan_buffer(const Tensor& tensor);
        friend internal::StorageRef internal::storage_ref(const Tensor& tensor);
        friend void internal::require_same_gpu_backend(const Tensor& reference,
                                                       const Tensor& other,
                                                       std::string_view operation);
        friend void internal::preserve_lazy_snapshots_before_write(Tensor& tensor);
        friend GpuBackend internal::gpu_backend_tag(const Tensor& tensor);

        struct TensorState {
            // Capacity management for in-place growth (like std::vector)
            size_t capacity = 0;
            size_t logical_size = 0;

            // Cached alignment flags (computed once on allocation)

            // Home stream: the stream the tensor's most recent enqueued write is
            // ordered on. Atomic so cross-thread readers see untorn values;
            // cross-thread *use* still requires host-side ordering plus
            // sync_to_stream/record_stream for the allocator.
            struct StreamHandle {
                using ValueType = cudaStream_t;
                std::atomic<ValueType> value{nullptr};

                StreamHandle() = default;
                StreamHandle(const StreamHandle& other)
                    : value(other.value.load(std::memory_order_relaxed)) {}
                StreamHandle& operator=(const StreamHandle& other) {
                    value.store(other.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
                    return *this;
                }
                StreamHandle& operator=(cudaStream_t stream) {
                    value.store(stream, std::memory_order_relaxed);
                    return *this;
                }
                operator ValueType() const {
                    return value.load(std::memory_order_relaxed);
                }
            };
            StreamHandle stream;

            // Debug tracking - when true, operations on this tensor are logged
            bool tracked = false;
            std::string name; // Optional name for identification in traces

            std::shared_ptr<LazyExprState> lazy;
        };

        void* data_ = nullptr;
        std::shared_ptr<void> data_owner_;
        // shared handle state (stream/name/lazy/capacity). Default empty
        // tensors keep a null state (no heap cell) until first mutation/use.
        std::shared_ptr<TensorState> state_;
        TensorShape shape_;
        RankedDims strides_;        // Stride per dim (stack; rank matches shape_)
        size_t storage_offset_ = 0; // Offset from data_ (in elements)
        bool is_contiguous_ = true; // True if memory layout is C-contiguous
        Device device_ = Device::CPU;
        DataType dtype_ = DataType::Float32;
        bool is_view_ = false;

        void ensure_state() {
            if (!state_) {
                state_ = std::make_shared<TensorState>();
            }
        }

        std::shared_ptr<StorageMeta> storage_meta_;
        uint64_t view_generation_snapshot_ = 0;

        mutable size_t id_ = 0;
        mutable bool lazy_ir_registered_ = false;
        static std::atomic<size_t> next_id_;
        static inline bool profiling_enabled_ = false;

        void materialize_deferred_slow();
        void materialize_if_deferred() {
            // is_deferred() is per-handle: shared TensorState may still hold
            // lazy->result for sibling copies after this handle materializes.
            if (is_deferred()) [[unlikely]] {
                materialize_deferred_slow();
            }
        }
        void materialize_if_deferred() const {
            if (is_deferred()) [[unlikely]] {
                const_cast<Tensor*>(this)->materialize_deferred_slow();
            }
        }

        /// Materialize zero-stride expand/broadcast views before raw-pointer escape.
        ///
        /// Raw-pointer escape is the materialization boundary:
        /// - Op paths are firewalled via contiguous_read / the allowlist.
        /// - That firewall does NOT cover raw-pointer escapes: callers hand
        ///   ptr()+numel() or data_ptr()+bytes() to flat memcpy/kernels (e.g.
        ///   cudaMemcpy HostToDevice of scaling [N,3] from an expand of [N,1]).
        /// - Materializing at the escape boundary preserves zero-copy expansion
        ///   for allowlisted ops while making every
        ///   flat-buffer consumer safe. storage_ptr() stays non-materializing
        ///   (allocation base for lifetime / sharing checks only).
        ///
        /// Do not assign `contiguous()` back to `*this`: expand views
        /// set is_view_=true, so operator= takes the view deep-copy path (copy_from),
        /// which re-enters data_ptr() → infinite recursion. Rebind fields like
        /// materialize_deferred_slow instead (implemented in tensor.cpp).
        void materialize_zero_stride_for_raw_ptr_escape();
        void materialize_zero_stride_for_raw_ptr_escape() const {
            // has_zero_stride is cheap; avoid a virtual-ish hop when dense.
            if (has_zero_stride()) [[unlikely]] {
                const_cast<Tensor*>(this)->materialize_zero_stride_for_raw_ptr_escape();
            }
        }

        /// Reject CPU-tagged tensors whose storage is CUDA device memory (and
        /// the inverse) when escaping via raw pointers. Mismatched tagging
        /// produces cudaMemcpy "invalid argument" with src/dst in the same
        /// address region and is otherwise hard to diagnose.
        void assert_device_storage_matches_tag() const;

        // The backend tags the deferred result before any validator can observe it.
        static Tensor make_deferred_expr_tensor(TensorShape shape,
                                                Device device,
                                                DataType dtype,
                                                GpuBackend backend,
                                                std::function<Tensor()> materializer);
        static Tensor make_deferred_expr_tensor(TensorShape shape,
                                                Device device,
                                                DataType dtype,
                                                GpuBackend backend,
                                                std::function<Tensor()> materializer,
                                                std::vector<uint64_t> lazy_input_ids);

        void compute_alignment() {}

        void init_storage_meta() {
            storage_meta_ = std::make_shared<StorageMeta>();
        }

        template <typename Deleter>
        void adopt_storage(void* data, Deleter&& deleter) {
            using StoredDeleter = std::decay_t<Deleter>;
            struct StorageOwner {
                void* data;
                StoredDeleter deleter;
                StorageMeta meta;

                StorageOwner(void* data_value, StoredDeleter&& deleter_value)
                    : data(data_value),
                      deleter(std::move(deleter_value)) {}

                ~StorageOwner() { deleter(data); }
            };

            auto owner = std::make_shared<StorageOwner>(
                data, StoredDeleter(std::forward<Deleter>(deleter)));
            data_owner_ = std::shared_ptr<void>(owner, data);
            storage_meta_ = std::shared_ptr<StorageMeta>(owner, &owner->meta);
            if (device_ == Device::GPU) {
                storage_meta_->backend = GpuBackend::CUDA;
            }
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

        std::shared_ptr<Tensor> create_lazy_snapshot() const;
        void register_lazy_snapshot_cell(const std::shared_ptr<Tensor>& snapshot) const;
        void preserve_lazy_snapshots_before_write();
        void replace_lazy_snapshot_storage(Tensor&& replacement);

        bool has_external_storage() const {
            return storage_meta_ && !storage_meta_->external_kind.empty();
        }

        static size_t storage_allocation_bytes(const TensorShape& shape,
                                               size_t capacity,
                                               DataType dtype);
        static void record_storage_allocation(StorageAccountingKind kind, size_t bytes);
        static void record_storage_deallocation(StorageAccountingKind kind, size_t bytes);

        void assert_view_not_stale() const {
            if (is_view_ && storage_meta_ &&
                view_generation_snapshot_ != storage_meta_->generation.load(std::memory_order_relaxed)) {
                throw std::runtime_error("Attempted to access a stale tensor view after storage reallocation");
            }
        }

        void propagate_view_meta(Tensor& view) const;

        const Tensor& contiguous_read(Tensor& materialized) const;

        template <typename Mutation>
        Tensor& mutate_logical_view(Mutation&& mutation) {
            LFS_ASSERT_MSG(!is_contiguous(),
                           "logical-view writeback is only valid for non-contiguous destinations");
            Tensor materialized = contiguous();
            LFS_ASSERT_MSG(materialized.is_contiguous(),
                           "logical-view write staging must produce dense storage");
            std::forward<Mutation>(mutation)(materialized);
            return copy_from(materialized);
        }

        bool shares_storage_with(const Tensor& other) const;

        // Generic functor-based binary operation (zero enum overhead)
        template <typename SrcT, typename OutT, typename Op>
        Tensor binary_op_generic(const Tensor& other, Op op) const;

        // Generic functor-based in-place scalar operation (zero enum overhead)
        template <typename Op>
        Tensor& scalar_op_inplace_generic(float scalar, Op op);

        // Generic functor-based in-place binary operation (zero enum overhead)
        template <typename SrcT = float, typename Op>
        Tensor& binary_op_inplace_generic(const Tensor& other, Op op);

        std::pair<Tensor, Tensor> _broadcasted(const Tensor& other, bool match_dtype = true) const;

        int resolve_dim(int dim) const {
            if (!is_valid())
                return -1;
            return detail::resolve_tensor_dim(dim, shape_.rank());
        }

        // Validation helpers - throw on error
        void validate_binary_op(const Tensor& other, bool require_same_shape = false, bool require_same_device = false) const;

        // Map add/sub/mul/div functors to LazyPointwiseOpKind tensor-binary stages.
        // Returns nullopt for ops that are not fusable (pow, maximum, ...).
        template <typename Op>
        static std::optional<internal::PointwiseOp> tensor_binary_fusion_kind();

        // Helper for binary operations with automatic type promotion.
        // Single same-shape contiguous ops stay on the eager fast path. When a
        // fusable chain can form from the size heuristic or deferred LHS,
        // seed or extend a pointwise fusion recipe with a tensor-binary stage so
        // mul+add and mul→reduce collapse to one fused launch.
        template <typename Op>
        Tensor binary_op_with_promotion(const Tensor& other, Op op,
                                        bool true_division = false) const;

        // Helper for comparison operations with automatic type promotion
        // Promotes operand types for comparison, but always returns Bool.
        // Eager for the same reasons as binary_op_with_promotion.
        template <typename Op>
        Tensor comparison_op_with_promotion(const Tensor& other, Op op) const;

        void validate_unary_op() const {
            if (!is_valid()) {
                throw std::runtime_error("Unary operation on invalid tensor");
            }
        }

        void validate_ternary_op(const Tensor& b, const Tensor& c) const;

        // Helper to ensure tensor is on same device
        Tensor ensure_same_device(const Tensor& other) const;

        static void link_deferred_result_to_inputs(Tensor& result,
                                                   std::initializer_list<uint64_t> candidate_input_ids);

        // Helper to create view with shared ownership
        Tensor create_view(const TensorShape& new_shape) const;

        Tensor create_strided_view(const TensorShape& new_shape,
                                   RankedDims new_strides) const;

        // Accept vector-built strides from older call sites without forcing heap
        // storage on the resulting view.
        Tensor create_strided_view(const TensorShape& new_shape,
                                   const std::vector<size_t>& new_strides) const {
            return create_strided_view(new_shape, RankedDims(new_strides));
        }

        /// Zero-copy expand / broadcast_to view. Logical numel may grow; broadcast
        // dims receive stride 0. Shares storage with *this.
        Tensor create_broadcast_view(const TensorShape& target_shape,
                                     std::vector<size_t> new_strides) const;

        /// Reject in-place mutation of expand/broadcast views (shared cells).
        void reject_inplace_on_zero_stride(const char* op_name) const;

        std::vector<size_t> resolve_dims(std::span<const int> dims) const;
        size_t calculate_offset(const std::vector<size_t>& indices) const;

    public:
        Tensor() = default;
        // Non-owning CUDA storage must stamp its producer stream; lifetime remains caller-owned.
        Tensor(void* data, TensorShape shape, Device device, DataType dtype,
               cudaStream_t home_stream = nullptr);

        // Copy constructor and assignment - SHALLOW COPY (LibTorch behavior)
        Tensor(const Tensor& other);
        Tensor& operator=(const Tensor& other);

        // Move constructor and assignment
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other);

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
                static_assert(N > 0, "TensorAccessor requires at least one dimension");
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
                    LFS_ASSERT_MSG(idx_array[i] < sizes_[i],
                                   "TensorAccessor index is out of bounds");
                    offset += idx_array[i] * strides_[i];
                }
                return data_[offset];
            }

            const std::array<size_t, N>& sizes() const { return sizes_; }
        };

        template <typename T, size_t N>
        TensorAccessor<T, N> accessor() {
            static_assert(N > 0, "accessor() requires at least one dimension");
            LFS_ASSERT_MSG(is_valid(),
                           "accessor() requires a valid tensor");
            LFS_ASSERT_MSG(device_ == Device::CPU,
                           "accessor() only works on CPU tensors");
            LFS_ASSERT_MSG(shape_.rank() == N,
                           "accessor() rank does not match the requested accessor rank");
            LFS_ASSERT_MSG(is_contiguous(),
                           "accessor() only works on contiguous tensors");

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
        Tensor reduce(ReduceOp op) const;
        Tensor reduce(ReduceOp op, const ReduceArgs& args) const;
        // Internal helper for where() operation
        Tensor ternary(const Tensor& b, const Tensor& c) const;

        // ============= FACTORY METHODS =============
        static Tensor empty(TensorShape shape, Device device = Device::GPU,
                            DataType dtype = DataType::Float32, bool use_pinned = true);
        // Allocate ordinary pageable host storage. Unlike the default CPU path,
        // this never consults PinnedMemoryAllocator or cudaHostAlloc.
        static Tensor empty_pageable_host(TensorShape shape,
                                          DataType dtype = DataType::Float32);
        static Tensor empty_unpinned(TensorShape shape, DataType dtype = DataType::Float32);
        static Tensor zeros(TensorShape shape, Device device = Device::GPU,
                            DataType dtype = DataType::Float32);
        static Tensor zeros_direct(TensorShape shape, size_t capacity, Device device = Device::GPU,
                                   DataType dtype = DataType::Float32);
        // Uninitialized CUDA storage of exactly the requested size, stream-ordered
        // and outside the size buckets. For large buffers retained across steps,
        // where bucket rounding would be permanent waste.
        static Tensor empty_exact(TensorShape shape, DataType dtype = DataType::Float32);
        static Tensor ones(TensorShape shape, Device device = Device::GPU,
                           DataType dtype = DataType::Float32);
        static Tensor full(TensorShape shape, float value, Device device = Device::GPU,
                           DataType dtype = DataType::Float32);
        static Tensor full_bool(TensorShape shape, bool value, Device device = Device::GPU);
        static Tensor zeros_bool(TensorShape shape, Device device = Device::GPU);
        static Tensor ones_bool(TensorShape shape, Device device = Device::GPU);
        static Tensor rand(TensorShape shape, Device device = Device::GPU,
                           DataType dtype = DataType::Float32);
        static Tensor randn(TensorShape shape, Device device = Device::GPU,
                            DataType dtype = DataType::Float32);
        static Tensor uniform(TensorShape shape, float low = 0.0f, float high = 1.0f,
                              Device device = Device::GPU, DataType dtype = DataType::Float32);
        static Tensor normal(TensorShape shape, float mean = 0.0f, float std = 1.0f,
                             Device device = Device::GPU, DataType dtype = DataType::Float32);
        static Tensor randint(TensorShape shape, int low, int high,
                              Device device = Device::GPU, DataType dtype = DataType::Int32);
        static Tensor bernoulli(TensorShape shape, float p = 0.5f,
                                Device device = Device::GPU, DataType dtype = DataType::Float32);
        static Tensor multinomial(const Tensor& weights, int num_samples,
                                  bool replacement = false);
        static Tensor arange(float end);
        static Tensor arange(float start, float end, float step = 1.0f);
        static Tensor linspace(float start, float end, size_t steps, Device device = Device::GPU);
        static Tensor eye(size_t n, Device device = Device::GPU);
        static Tensor eye(size_t m, size_t n, Device device = Device::GPU);
        static Tensor diag(const Tensor& diagonal);

        static Tensor from_blob(void* data, TensorShape shape, Device device, DataType dtype,
                                cudaStream_t home_stream = nullptr) {
            LFS_ASSERT_MSG(data != nullptr || shape.elements() == 0,
                           "from_blob received null data for a non-empty tensor");
            return Tensor(data, shape, device, dtype, home_stream);
        }
        static Tensor from_external_owner(void* data,
                                          TensorShape shape,
                                          Device device,
                                          DataType dtype,
                                          std::shared_ptr<void> owner);
        static Tensor from_external_owner(void* data,
                                          TensorShape shape,
                                          Device device,
                                          DataType dtype,
                                          std::shared_ptr<void> owner,
                                          size_t capacity);
        static Tensor from_external_owner(void* data,
                                          TensorShape shape,
                                          Device device,
                                          DataType dtype,
                                          std::shared_ptr<void> owner,
                                          size_t capacity,
                                          cudaStream_t stream);
        static Tensor from_external_owner(void* data,
                                          TensorShape shape,
                                          Device device,
                                          DataType dtype,
                                          std::shared_ptr<void> owner,
                                          size_t capacity,
                                          cudaStream_t stream,
                                          std::string external_kind);

        static Tensor from_vector(const std::vector<float>& data, TensorShape shape,
                                  Device device = Device::GPU);
        static Tensor from_vector(const std::vector<int>& data, TensorShape shape,
                                  Device device = Device::GPU);
        static Tensor from_vector(const std::vector<bool>& data, TensorShape shape,
                                  Device device = Device::GPU);

        // Initializer list overloads for convenience
        static Tensor from_vector(std::initializer_list<float> data, TensorShape shape,
                                  Device device = Device::GPU) {
            return from_vector(std::vector<float>(data), shape, device);
        }

        static Tensor from_vector(std::initializer_list<int> data, TensorShape shape,
                                  Device device = Device::GPU) {
            return from_vector(std::vector<int>(data), shape, device);
        }

        static Tensor from_vector(std::initializer_list<bool> data, TensorShape shape,
                                  Device device = Device::GPU) {
            return from_vector(std::vector<bool>(data), shape, device);
        }

        // ============= LIKE OPERATIONS =============
        static Tensor zeros_like(const Tensor& other) {
            tensor_contract::require_valid(
                other, "zeros_like", "template", LFS_SOURCE_SITE_CURRENT());
            return internal::allocate_zeros_like(other, other.shape(), other.dtype());
        }

        static Tensor ones_like(const Tensor& other) {
            tensor_contract::require_valid(
                other, "ones_like", "template", LFS_SOURCE_SITE_CURRENT());
            return internal::allocate_ones_like(other, other.shape(), other.dtype());
        }

        static Tensor ones_like(const Tensor& other, DataType dtype) {
            tensor_contract::require_valid(
                other, "ones_like", "template", LFS_SOURCE_SITE_CURRENT());
            return internal::allocate_ones_like(other, other.shape(), dtype);
        }

        static Tensor rand_like(const Tensor& other) {
            tensor_contract::require_valid(
                other, "rand_like", "template", LFS_SOURCE_SITE_CURRENT());
            return internal::allocate_rand_like(other, other.shape(), other.dtype());
        }

        static Tensor randn_like(const Tensor& other) {
            tensor_contract::require_valid(
                other, "randn_like", "template", LFS_SOURCE_SITE_CURRENT());
            return internal::allocate_randn_like(other, other.shape(), other.dtype());
        }

        static Tensor empty_like(const Tensor& other, const TensorShape& shape, DataType dtype);

        static Tensor empty_like(const Tensor& other) {
            tensor_contract::require_valid(
                other, "empty_like", "template", LFS_SOURCE_SITE_CURRENT());
            auto result = internal::allocate_like(other, other.shape(), other.dtype());
            result.set_stream(other.stream());
            return result;
        }

        static Tensor full_like(const Tensor& other, float value) {
            tensor_contract::require_valid(
                other, "full_like", "template", LFS_SOURCE_SITE_CURRENT());
            auto result = internal::allocate_like(
                other, other.shape(), other.dtype(), value);
            result.set_stream(other.stream());
            return result;
        }

        // ============= COMBINING TENSORS =============
        static Tensor cat(const std::vector<Tensor>& tensors, int dim = 0);
        static Tensor stack(const std::vector<Tensor>& tensors, int dim = 0);

        // ============= CONDITIONAL =============
        static Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y);

        // ============= GLOBAL CONFIGURATION =============
        // Resets the CPU mt19937_64 stream and GPU seed/normal offsets.
        // CPU draws advance independently of GPU draws.
        static void manual_seed(uint64_t seed) {
            RandomGenerator::instance().manual_seed(seed);
        }

        static void enable_profiling(bool enable) { profiling_enabled_ = enable; }
        static LazyTelemetrySnapshot lazy_telemetry_snapshot();

        static void reset_lazy_telemetry();

        static void clear_lazy_ir_for_testing();

        static void trim_memory_pool();
        static void trim_memory_pool_if_reserved_unused_exceeds(size_t threshold_bytes);
        // CUDA device pool only; leaves the pinned host cache intact.
        static void trim_device_memory_pool();
        static void shutdown_memory_pool();
        static void set_memory_pool_iteration(int iteration);

        void set_bool(std::initializer_list<size_t> indices, bool value);
        bool get_bool(std::initializer_list<size_t> indices) const;
        void set_bool(std::span<const size_t> indices, bool value);
        bool get_bool(std::span<const size_t> indices) const;

        template <typename T>
        T* ptr() {
            preserve_lazy_snapshots_before_write();
            return const_cast<T*>(std::as_const(*this).template ptr<T>());
        }

        template <typename T>
        const T* ptr() const {
            materialize_if_deferred();
            // raw ptr() is a flat-buffer escape — materialize stride-0 views.
            materialize_zero_stride_for_raw_ptr_escape();
            tensor_contract::require_valid(
                *this, "ptr<T>", "tensor", LFS_SOURCE_SITE_CURRENT());
            using Value = std::remove_cv_t<T>;
            if constexpr (!std::is_void_v<Value>) {
                const bool dtype_matches =
                    (std::is_same_v<Value, float> && dtype_ == DataType::Float32) ||
                    (std::is_same_v<Value, detail::tensor_half_t> && dtype_ == DataType::Float16) ||
                    ((std::is_same_v<Value, int> || std::is_same_v<Value, int32_t> ||
                      std::is_same_v<Value, uint32_t>) &&
                     dtype_ == DataType::Int32) ||
                    (std::is_same_v<Value, uint32_t> && dtype_ == DataType::UInt32) ||
                    (std::is_same_v<Value, int64_t> && dtype_ == DataType::Int64) ||
                    ((std::is_same_v<Value, bool> || std::is_same_v<Value, unsigned char> ||
                      std::is_same_v<Value, uint8_t>) &&
                     (dtype_ == DataType::Bool || dtype_ == DataType::UInt8));
                LFS_ASSERT_MSG(dtype_matches,
                               "ptr<T>() type does not match tensor dtype");
            }
            assert_view_not_stale();
            LFS_ASSERT_MSG(data_ != nullptr || numel() == 0,
                           "ptr<T>() found null storage for a non-empty tensor");
            assert_device_storage_matches_tag();
            const char* data_ptr = static_cast<const char*>(data_) + storage_offset_ * dtype_size(dtype_);
            return static_cast<const T*>(static_cast<const void*>(data_ptr));
        }

        void* data_ptr() {
            materialize_if_deferred();
            // raw data_ptr() is a flat-buffer escape — materialize stride-0 views.
            materialize_zero_stride_for_raw_ptr_escape();
            preserve_lazy_snapshots_before_write();
            tensor_contract::require_valid(
                *this, "data_ptr", "tensor", LFS_SOURCE_SITE_CURRENT());
            assert_view_not_stale();
            LFS_ASSERT_MSG(data_ != nullptr || numel() == 0,
                           "data_ptr() found null storage for a non-empty tensor");
            assert_device_storage_matches_tag();
            return static_cast<char*>(data_) + storage_offset_ * dtype_size(dtype_);
        }
        const void* data_ptr() const {
            materialize_if_deferred();
            // raw data_ptr() is a flat-buffer escape — materialize stride-0 views.
            materialize_zero_stride_for_raw_ptr_escape();
            tensor_contract::require_valid(
                *this, "data_ptr", "tensor", LFS_SOURCE_SITE_CURRENT());
            assert_view_not_stale();
            LFS_ASSERT_MSG(data_ != nullptr || numel() == 0,
                           "data_ptr() found null storage for a non-empty tensor");
            assert_device_storage_matches_tag();
            return static_cast<const char*>(data_) + storage_offset_ * dtype_size(dtype_);
        }

        // Base of allocation (for memory management / storage-identity only).
        // Intentionally does NOT materialize zero-stride views: callers that need
        // a dense flat buffer of numel() elements must use ptr()/data_ptr().
        void* storage_ptr() {
            materialize_if_deferred();
            preserve_lazy_snapshots_before_write();
            tensor_contract::require_valid(
                *this, "storage_ptr", "tensor", LFS_SOURCE_SITE_CURRENT());
            return data_;
        }
        const void* storage_ptr() const {
            materialize_if_deferred();
            tensor_contract::require_valid(
                *this, "storage_ptr", "tensor", LFS_SOURCE_SITE_CURRENT());
            return data_;
        }

        // Properties - FIXED: Check validity before accessing shape
        // Hot field read used throughout kernels and tensor expressions.
        const TensorShape& shape() const { return shape_; }
        Device device() const { return device_; }
        DataType dtype() const { return dtype_; }
        bool owns_memory() const { return static_cast<bool>(data_owner_) && !is_view_; }
        bool is_view() const { return is_view_; }
        bool is_external_storage() const {
            return storage_meta_ && static_cast<bool>(storage_meta_->external_owner);
        }
        bool is_empty() const { return !is_valid() || numel() == 0; }
        // Local deferred flag only — never takes the global IR mutex. Eager IR
        // nodes (debug map) are NOT reported here; use lazy_expr_id() / IR APIs.
        // Per-handle: after materialize, this handle has storage so it is no
        // longer deferred even if shared TensorState still retains lazy->result
        // for sibling copies.
        bool has_lazy_expr() const { return is_deferred(); }
        bool is_deferred() const {
            return state_ && state_->lazy && data_ == nullptr && !data_owner_ && !is_view_;
        }
        uint64_t lazy_expr_id() const;
        std::optional<internal::LazyExprDebugInfo> lazy_expr_info() const;

        size_t debug_id() const { return id_; }

        bool is_valid() const {
            return static_cast<bool>(data_owner_) || is_view_ || (state_ && state_->lazy);
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

        // Home stream: where this tensor's pending writes are ordered. Frees route
        // here; reads from other streams must be recorded (record_stream) or
        // bridged + recorded (sync_to_stream).
        cudaStream_t stream() const {
            return state_ ? static_cast<cudaStream_t>(state_->stream) : nullptr;
        }

        // Declarative re-homing: future writes happen on `stream`. The old home
        // becomes a recorded use so the eventual free stays ordered after it.
        void set_stream(cudaStream_t stream);

        // Marks a read of this tensor on `stream` (other than its home) so the
        // allocator defers recycling until that stream passes the read.
        void record_stream(cudaStream_t stream) const;

        // Orders `execution_stream` after this tensor's pending work, then records
        // the use. The standard prologue for consuming a tensor on another stream.
        void sync_to_stream(cudaStream_t execution_stream) const;

        // Debug tracking - mark tensor to trace all operations it's involved in
        bool is_tracked() const { return state_ && state_->tracked; }
        Tensor& set_tracked(bool tracked = true) {
            ensure_state();
            state_->tracked = tracked;
            return *this;
        }
        Tensor& track() { return set_tracked(true); } // Convenience alias
        Tensor& untrack() { return set_tracked(false); }

        // Optional name for identifying tensors in traces. Also forwarded to the
        // VRAM profiler so the underlying allocation is labelled with this name.
        const std::string& name() const {
            static const std::string kEmpty;
            return state_ ? state_->name : kEmpty;
        }
        Tensor& set_name(std::string name) {
            ensure_state();
            state_->name = std::move(name);
            relabel_allocation_for_profiler();
            return *this;
        }

    private:
        void relabel_allocation_for_profiler();

    public:
        size_t size(size_t dim) const {
            LFS_ASSERT_MSG(is_valid(),
                           "size() called on an invalid tensor");
            if (dim >= shape_.rank()) {
                throw std::out_of_range(
                    "Dimension " + std::to_string(dim) + " out of range for rank " + std::to_string(shape_.rank()));
            }
            return shape_[dim];
        }

        // Capacity management (for in-place growth like std::vector)
        // capacity() returns the reserved capacity along dimension 0 (0 = no reservation)
        // logical_size() returns the logical size along dimension 0 (same as shape()[0])
        size_t capacity() const { return state_ ? state_->capacity : 0; }
        size_t logical_size() const {
            if (state_ && state_->capacity > 0) {
                return state_->logical_size;
            }
            return shape_.rank() > 0 ? shape_[0] : 0;
        }
        std::string external_storage_kind() const {
            return storage_meta_ ? storage_meta_->external_kind : std::string{};
        }
        std::shared_ptr<void> external_storage_owner() const {
            return storage_meta_ ? storage_meta_->external_owner : nullptr;
        }

        // Exportable SoA provenance (stamped by SplatExportableStorage allocators).
        void set_exportable_provenance(std::shared_ptr<void> control,
                                       std::uint32_t region,
                                       std::uint64_t bound_generation) {
            ensure_storage_meta();
            storage_meta_->exportable_control = std::move(control);
            storage_meta_->exportable_region = region;
            storage_meta_->exportable_bound_generation = bound_generation;
        }
        [[nodiscard]] bool has_exportable_provenance() const noexcept {
            return storage_meta_ && static_cast<bool>(storage_meta_->exportable_control);
        }
        [[nodiscard]] std::shared_ptr<void> exportable_control() const noexcept {
            return storage_meta_ ? storage_meta_->exportable_control : nullptr;
        }
        [[nodiscard]] std::uint32_t exportable_region() const noexcept {
            return storage_meta_ ? storage_meta_->exportable_region : 0u;
        }
        [[nodiscard]] std::uint64_t exportable_bound_generation() const noexcept {
            return storage_meta_ ? storage_meta_->exportable_bound_generation : 0u;
        }
        static std::string storage_memory_summary();
        static std::size_t cuda_direct_storage_live_bytes();
        static void log_storage_memory();
        static void log_storage_memory(std::string_view label);

        // reserve() pre-allocates memory for future growth along dimension 0
        // Supports multi-dimensional tensors: [N, D1, D2, ...] reserves N "rows"
        void reserve(size_t new_capacity);

        // Memory operations
        Tensor clone() const;      // Deep copy
        Tensor contiguous() const; // Materialize to contiguous if strided
        Tensor to(Device device, cudaStream_t stream = nullptr) const;
        // Synchronous export-oriented copy to ordinary pageable host memory.
        Tensor to_pageable_host(cudaStream_t stream = nullptr) const;
        Tensor to(DataType dtype) const;
        Tensor to(GpuBackend backend) const;
        bool is_contiguous() const { return is_contiguous_; }

        // Stride operations (Zero-copy views) — stack RankedDims
        const RankedDims& strides() const { return strides_; }
        size_t stride(size_t dim) const {
            LFS_ASSERT_MSG(is_valid(),
                           "stride() called on an invalid tensor");
            LFS_ASSERT_MSG(dim < strides_.size(),
                           "stride dimension is out of range");
            return strides_[dim];
        }
        size_t storage_offset() const { return storage_offset_; }

        /// True if any dimension with size > 1 has stride 0 (expand / broadcast_to).
        /// Such views share storage cells across logical indices; in-place mutation
        // is rejected and non-allowlisted kernels must materialize first.
        ///
        /// stride of 0 (product of later dims includes 0), e.g. shN [N,0,3] →
        /// strides [0,3,1]. That is NOT an expand view — only size>1 with stride 0
        /// is a true broadcast. Size-1 dims with stride 0 are also expand-like but
        /// do not change flat numel layout on their own.
        bool has_zero_stride() const {
            if (!is_valid() || numel() == 0) {
                // Empty tensors (e.g. [N,0,3]) are never expand/broadcast views.
                // Their row-major strides may still contain 0 from a zero-size dim.
                return false;
            }
            const size_t rank = strides_.size();
            for (size_t i = 0; i < rank; ++i) {
                if (strides_[i] == 0 && shape_[i] > 1) {
                    return true;
                }
            }
            return false;
        }

        Tensor cpu() const { return to(Device::CPU); }
        Tensor gpu() const { return to(Device::GPU); }
        Tensor cuda() const { return gpu(); }
        bool is_gpu() const { return device() == Device::GPU; }
        bool is_cuda() const { return is_gpu(); }

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
        Tensor view_as(DataType dtype) const;

        Tensor view(std::span<const int> sizes) const { return reshape(sizes); }
        Tensor view(std::initializer_list<int> sizes) const { return reshape(sizes); }
        Tensor view(TensorShape new_shape) const { return reshape(new_shape); }

        Tensor squeeze() const {
            return squeeze(std::optional<int>{});
        }

        Tensor squeeze(std::optional<int> dim) const {
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
        Tensor neg() const;
        Tensor abs() const;
        Tensor sign() const;
        Tensor reciprocal() const;
        Tensor exp() const;
        Tensor exp2() const;
        Tensor log() const;
        Tensor log2() const;
        Tensor log10() const;
        Tensor log1p() const;
        Tensor sqrt() const;
        Tensor rsqrt() const;
        Tensor square() const;
        Tensor sin() const;
        Tensor cos() const;
        Tensor tan() const;
        Tensor asin() const;
        Tensor acos() const;
        Tensor atan() const;
        Tensor sinh() const;
        Tensor cosh() const;
        Tensor tanh() const;
        Tensor sigmoid() const;
        Tensor relu() const;
        Tensor gelu() const;
        Tensor swish() const;
        Tensor floor() const;
        Tensor ceil() const;
        Tensor round() const;
        Tensor trunc() const;
        Tensor isnan() const;
        Tensor isinf() const;
        Tensor isfinite() const;
        Tensor logical_not() const;

        Tensor normalize(int dim = -1, float eps = 1e-12f) const;
        Tensor logit(float eps = 1e-7f) const;

        // ============= BINARY OPERATIONS (Template-based) =============

        // Float32 multiply-add with one rounding, including broadcast operands.

        // Arithmetic operations

        // New functor-based overloads for Tensor (zero enum overhead, lazy evaluation)
        // Now with automatic type promotion for mixed-dtype operations
        Tensor add(const Tensor& other) const;

        Tensor sub(const Tensor& other) const;

        Tensor mul(const Tensor& other) const;

        Tensor div(const Tensor& other) const;

        Tensor pow(const Tensor& other) const;

        Tensor mod(const Tensor& other) const;

        Tensor maximum(const Tensor& other) const;

        Tensor minimum(const Tensor& other) const;

    private:
        template <typename T>
        auto validated_scalar_operand(
            const T& value,
            const std::string_view operation,
            const std::initializer_list<DataType> allowed_dtypes) const;

        template <typename T>
        static constexpr DataType scalar_operand_dtype() {
            return std::is_floating_point_v<T> ? DataType::Float32 : DataType::Int32;
        }

    public:
        // Macro for scalar binary operations (lazy evaluation with scalar_right_op)
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor add(const T& other) const;
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor sub(const T& other) const;
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor mul(const T& other) const;
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor div(const T& other) const;
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor pow(const T& other) const;
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor mod(const T& other) const;
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor maximum(const T& other) const;
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor minimum(const T& other) const;

        // Comparison operations (return Bool tensors)

        // Functor-based overloads for Tensor (zero enum overhead)
        Tensor eq(const Tensor& other) const;

        Tensor ne(const Tensor& other) const;

        Tensor lt(const Tensor& other) const;

        Tensor le(const Tensor& other) const;

        Tensor gt(const Tensor& other) const;

        Tensor ge(const Tensor& other) const;

        // Macro for scalar comparison operations (return Bool dtype)
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor eq(const T& other) const;
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor ne(const T& other) const;
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor lt(const T& other) const;
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor le(const T& other) const;
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor gt(const T& other) const;
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor ge(const T& other) const;

        // Logical operations (Tensor only, Bool -> Bool)
        Tensor logical_and(const Tensor& other) const;

        Tensor logical_or(const Tensor& other) const;

        Tensor logical_xor(const Tensor& other) const;

        // Keep non-zero values in this mask (selection groups) while clearing
        // entries whose corresponding live-mask value is zero. This is an
        // in-place CUDA operation and deliberately does not allocate a result.
        Tensor& and_live_(const Tensor& live_mask);

        // ============= REDUCE OPERATIONS =============
        Tensor sum() const {
            return sum(std::span<const int>{}, false);
        }

        Tensor sum(std::span<const int> axes, bool keepdim = false) const {
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

        Tensor mean() const {
            return mean(std::span<const int>{}, false);
        }

        Tensor mean(std::span<const int> axes, bool keepdim = false) const {
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

        Tensor max() const {
            return max(std::span<const int>{}, false);
        }

        Tensor max(std::span<const int> axes, bool keepdim = false) const {
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

        Tensor min() const {
            return min(std::span<const int>{}, false);
        }

        Tensor min(std::span<const int> axes, bool keepdim = false) const {
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

        Tensor prod() const {
            return prod(std::span<const int>{}, false);
        }

        Tensor prod(std::span<const int> axes, bool keepdim = false) const {
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

        Tensor any() const {
            return any(std::span<const int>{}, false);
        }

        Tensor any(std::span<const int> axes, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Any, args);
        }

        Tensor any(int dim, bool keepdim = false) const {
            std::vector<int> axes = {dim};
            return any(std::span<const int>(axes), keepdim);
        }

        Tensor all() const {
            return all(std::span<const int>{}, false);
        }

        Tensor all(std::span<const int> axes, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::All, args);
        }

        Tensor all(int dim, bool keepdim = false) const {
            std::vector<int> axes = {dim};
            return all(std::span<const int>(axes), keepdim);
        }

        Tensor std() const {
            return std(std::span<const int>{}, false, true);
        }

        Tensor std(std::span<const int> axes, bool keepdim = false, bool unbiased = true) const {
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

        Tensor var() const {
            return var(std::span<const int>{}, false, true);
        }

        Tensor var(std::span<const int> axes, bool keepdim = false, bool unbiased = true) const {
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

        Tensor argmax() const {
            return argmax(std::span<const int>{}, false);
        }

        Tensor argmax(std::span<const int> axes, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Argmax, args);
        }

        Tensor argmin() const {
            return argmin(std::span<const int>{}, false);
        }

        Tensor argmin(std::span<const int> axes, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Argmin, args);
        }

        Tensor cumsum(int dim = 0) const;

        // Scalar reduce operations - use direct CUB path for CUDA Float32 contiguous tensors
        float sum_scalar() const;

        float mean_scalar() const;

        float min_scalar() const;

        float max_scalar() const;

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
        T item() const;

        size_t count_nonzero() const;

        // ============= TERNARY OPERATIONS =============
        Tensor where(const Tensor& condition, const Tensor& other) const {
            return condition.ternary(*this, other);
        }

        Tensor clamp(float min_val, float max_val) const;

        Tensor clamp_min(float min) const {
            return clamp(min, dtype_ == DataType::Int32 ? std::numeric_limits<float>::infinity()
                                                        : std::numeric_limits<float>::max());
        }

        Tensor clamp_max(float max) const {
            return clamp(dtype_ == DataType::Int32 ? -std::numeric_limits<float>::infinity()
                                                   : std::numeric_limits<float>::lowest(),
                         max);
        }

        Tensor& clamp_(float min_val, float max_val);
        Tensor& clamp_min_(float min);
        Tensor& clamp_max_(float max);

        // In-place operations (Template-based, direct functor dispatch - zero enum overhead!)
        template <typename T>
        Tensor& add_(const T& other);

        template <typename T>
        Tensor& sub_(const T& other);

        template <typename T>
        Tensor& mul_(const T& other);

        template <typename T>
        Tensor& div_(const T& other);

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
        // Linear + bias + ReLU in single pass

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
         *   auto param = Tensor::randn({1000, 3}, Device::GPU);
         *   param.reserve(2000);  // Pre-allocate capacity
         *   auto indices = Tensor::from_vector({0, 5, 10}, {3}, Device::GPU);
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
        // Gather rows along `dim` into a caller-provided output (no allocation),
        // letting the caller control the output's storage (e.g. a Vulkan-external
        // backing block). `out` must already be sized [..., indices.numel(), ...]
        // and share this tensor's dtype/device; `indices` must be 1-D integer.
        void index_select_into(Tensor& out, int dim, const Tensor& indices, BoundaryMode mode) const;
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
         *   auto sorted = t.sort(0, false);
         *   auto& sorted_vals = sorted.first;
         *   auto& sorted_idx = sorted.second;
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
        Tensor& assert_shape(TensorShape expected);
        Tensor& assert_shape(TensorShape expected, const std::string& msg);
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

        void print_formatted() const;
        void print_formatted(const std::string& name, size_t max_per_dim = 10) const;

        // ============= TENSOR OPTIONS =============
        struct TensorOptions {
            Device device = Device::GPU;
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

    inline void pin_operands(std::initializer_list<const Tensor*> tensors) {
        const Tensor* backend_reference = nullptr;
        for (const Tensor* tensor : tensors) {
            if (tensor) {
                tensor->materialize_if_deferred();
                if (tensor->device() == Device::GPU) {
                    if (backend_reference == nullptr) {
                        backend_reference = tensor;
                    } else {
                        internal::require_same_gpu_backend(
                            *backend_reference, *tensor, "tensor operation");
                    }
                }
            }
        }
    }

    struct LazyExprState {
        ~LazyExprState() noexcept;

        std::mutex gate;
        std::atomic<std::thread::id> gate_owner{};
        uint64_t node_id = 0;
        std::function<Tensor()> materializer;
        Tensor result;
        bool materializer_unregistered = false;
        // The backend the materializer will allocate on, derived from the inputs;
        // a deferred tensor has no storage yet, so validators read this instead.
        GpuBackend backend = GpuBackend::CUDA;
    };

    inline uint64_t Tensor::lazy_expr_id() const {
        // Only the still-pending handle reports the deferred node id.
        if (is_deferred() && state_->lazy->node_id != 0) {
            return state_->lazy->node_id;
        }
        return internal::tensor_lazy_expr_id(*this);
    }

    // ============= TensorRowProxy for operator[] =============
    // Implementations in tensor_row_proxy.cpp (except template methods)
    class LFS_CORE_API TensorRowProxy {
    private:
        struct CudaStagingSlot {
            float value = 0.0f;
            size_t linear_index = 0;
        };

        Tensor* tensor_;
        size_t row_index_;
        mutable std::deque<CudaStagingSlot> cuda_staging_slots_;
        void flush_cuda_staging() const;

    public:
        TensorRowProxy(Tensor* tensor, size_t row_index)
            : tensor_(tensor),
              row_index_(row_index) {
            LFS_ASSERT_MSG(tensor_ != nullptr && tensor_->is_valid(),
                           "TensorRowProxy requires a valid tensor");
            LFS_ASSERT_MSG(tensor_->ndim() > 0,
                           "TensorRowProxy requires a tensor with at least one dimension");
            LFS_ASSERT_MSG(row_index_ < tensor_->shape()[0],
                           "TensorRowProxy row index is out of bounds");
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
            static_assert(std::is_arithmetic_v<T>,
                          "TensorRowProxy::item_as<T>() requires an arithmetic type");
            LFS_ASSERT_MSG(tensor_ != nullptr && tensor_->is_valid(),
                           "TensorRowProxy::item_as() requires a valid tensor");
            flush_cuda_staging();

            // Handle 2D tensors with shape [N, 1] (like nonzero() output)
            if (tensor_->shape().rank() == 2 && tensor_->shape()[1] == 1) {
                Tensor row_tensor = static_cast<Tensor>(*this);
                return row_tensor.item<T>();
            }

            // Standard 1D case
            LFS_ASSERT_MSG(tensor_->shape().rank() == 1,
                           "TensorRowProxy::item_as() requires a 1D or [N,1] tensor");
            LFS_ASSERT_MSG(row_index_ < tensor_->numel(),
                           "TensorRowProxy::item_as() index is out of bounds");

            const size_t linear_index = row_index_ * tensor_->stride(0);
            const auto copy_and_convert = [&]<typename Stored>() -> T {
                Stored value{};
                internal::read_scalar(*tensor_, linear_index, &value, sizeof(Stored));
                return static_cast<T>(value);
            };

            switch (tensor_->dtype()) {
            case DataType::Float32:
                return copy_and_convert.template operator()<float>();
            case DataType::Int32:
                return copy_and_convert.template operator()<int32_t>();
            case DataType::UInt32:
                return copy_and_convert.template operator()<uint32_t>();
            case DataType::Int64:
                return copy_and_convert.template operator()<int64_t>();
            case DataType::UInt8:
            case DataType::Bool:
                return copy_and_convert.template operator()<uint8_t>();
            case DataType::Float16:
                LFS_ASSERT_MSG(false,
                               "TensorRowProxy::item_as() does not support Float16");
            }
            return T{};
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
        LFS_ASSERT_MSG(is_valid(),
                       "operator[] requires a valid tensor");
        LFS_ASSERT_MSG(ndim() > 0,
                       "operator[] requires a tensor with at least one dimension");
        LFS_ASSERT_MSG(index < shape_[0],
                       "operator[] index is out of bounds");
        return TensorRowProxy(this, index);
    }

    inline const TensorRowProxy Tensor::operator[](size_t index) const {
        LFS_ASSERT_MSG(is_valid(),
                       "operator[] requires a valid tensor");
        LFS_ASSERT_MSG(ndim() > 0,
                       "operator[] requires a tensor with at least one dimension");
        LFS_ASSERT_MSG(index < shape_[0],
                       "operator[] index is out of bounds");
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

    namespace internal {

        inline void require_same_gpu_backend(const Tensor& reference,
                                             const Tensor& other,
                                             const std::string_view operation) {
            if (reference.device_ != Device::GPU || other.device_ != Device::GPU) {
                return;
            }
            if (gpu_backend_tag(reference) != gpu_backend_tag(other)) {
                throw_gpu_backend_mismatch(reference, other, operation);
            }
        }

        // Backend of a GPU tensor: the storage tag when storage exists, the lazy
        // state's tag for a deferred tensor that has not materialized yet.
        inline GpuBackend gpu_backend_tag(const Tensor& tensor) {
            if (tensor.storage_meta_) {
                return tensor.storage_meta_->backend;
            }
            if (tensor.state_ && tensor.state_->lazy) {
                return tensor.state_->lazy->backend;
            }
            return GpuBackend::CUDA;
        }

        // Fused-chain operands cross the facade as raw addresses; deriving them from
        // the StorageRef skips the pointer classification ptr<T>() performs per call.
        inline const float* chain_operand_address(const StorageRef& storage) {
            return reinterpret_cast<const float*>(
                static_cast<const unsigned char*>(storage.data) + storage.byte_offset);
        }

        inline StorageRef storage_ref(const Tensor& tensor) {
            // Facade callers replace ptr<T>() raw-pointer escapes. Preserve its
            // deferred-materialization boundary before reading the storage base.
            tensor.materialize_if_deferred();
            LFS_ASSERT_MSG(tensor.is_valid(), "storage_ref requires a valid tensor");
            tensor.assert_view_not_stale();
            LFS_ASSERT_MSG(tensor.device_ == Device::GPU,
                           "storage_ref requires GPU storage");
            LFS_ASSERT_MSG(tensor.data_ != nullptr || tensor.numel() == 0,
                           "storage_ref found null storage for a non-empty tensor");
            return StorageRef{
                .backend = tensor.storage_meta_ ? tensor.storage_meta_->backend
                                                : GpuBackend::CUDA,
                .data = tensor.data_,
                .byte_offset = tensor.storage_offset_ * dtype_size(tensor.dtype_),
                .dtype = tensor.dtype_,
                .meta = tensor.storage_meta_.get(),
            };
        }

        inline StridedLayout strided_layout(const Tensor& tensor) {
            LFS_ASSERT_MSG(tensor.is_valid(), "strided_layout requires a valid tensor");
            LFS_ASSERT_MSG(tensor.ndim() <= MAX_TENSOR_RANK,
                           "strided layout rank exceeds MAX_TENSOR_RANK");
            StridedLayout layout{};
            layout.rank = tensor.ndim();
            layout.element_count = tensor.numel();
            for (size_t i = 0; i < layout.rank; ++i) {
                layout.dims[i] = tensor.shape()[i];
                layout.strides[i] = tensor.stride(i);
            }
            return layout;
        }

        // The storage refs are taken before the adapter is chosen: a deferred
        // input materializes inside storage_ref, and only the materialized storage
        // carries the backend that must dispatch it.
        template <class Functor>
        inline void run_pointwise_unary(const Tensor& input, const Tensor& output,
                                        const Functor& functor,
                                        const ExecContext context) {
            const StorageRef in = storage_ref(input);
            const StorageRef out = storage_ref(output);
            backend_ops(in.backend).unary(pointwise_program(input.dtype(), output.dtype(), functor), in, out, output.numel(), context);
        }

        template <class Functor>
        inline void run_pointwise_binary(const Tensor& lhs, const Tensor& rhs,
                                         const Tensor& output, const Functor& functor,
                                         const ExecContext context) {
            const StorageRef left = storage_ref(lhs);
            const StorageRef right = storage_ref(rhs);
            const StorageRef out = storage_ref(output);
            backend_ops(left.backend).binary(pointwise_program(lhs.dtype(), output.dtype(), functor), left, right, out, output.numel(), context);
        }

        template <class Functor>
        inline void run_pointwise_broadcast(
            const Tensor& lhs, const Tensor& rhs, const Tensor& output,
            const Functor& functor, const ExecContext context) {
            const StorageRef left = storage_ref(lhs);
            const StorageRef right = storage_ref(rhs);
            const StorageRef out = storage_ref(output);
            backend_ops(left.backend).broadcast_binary(pointwise_program(lhs.dtype(), output.dtype(), functor), left, strided_layout(lhs), right, strided_layout(rhs), out, strided_layout(output), context);
        }

    } // namespace internal

    // Memory info
    class LFS_CORE_API MemoryInfo {
    public:
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        size_t allocated_bytes = 0;
        int device_id = -1;
        size_t pool_used_current = 0;
        size_t pool_reserved_current = 0;
        size_t pool_used_high = 0;
        size_t pool_reserved_high = 0;

        static MemoryInfo cuda();
        static MemoryInfo cpu();

        void log() const;
    };

    // ========================================================================
    // Inline implementation of lazy gather operation
    // ========================================================================

    inline auto Tensor::gather_lazy(const Tensor& indices) const -> PermutationExpr<TensorLeaf, TensorLeaf> {
        LFS_ASSERT_MSG(is_valid() && indices.is_valid(),
                       "gather_lazy requires valid tensors");
        LFS_ASSERT_MSG(indices.dtype() == DataType::Int32,
                       "gather_lazy indices must be Int32");
        LFS_ASSERT_MSG(indices.device() == device_,
                       "gather_lazy indices must be on the input device");
        internal::require_same_gpu_backend(*this, indices, "gather_lazy");

        // Create expression that will lazily gather elements
        return PermutationExpr<TensorLeaf, TensorLeaf>(
            TensorLeaf(*this),
            TensorLeaf(indices),
            indices.shape(), // Output shape matches indices shape
            device_,
            dtype_);
    }

    // Parallel first-touch for a large ordinary (pageable) host allocation.
    // The caller must have allocated the storage with empty_pageable_host() or
    // another pageable allocator.
    LFS_CORE_API void prefault_pageable_host_memory(void* data, size_t bytes);

} // namespace lfs::core

// Include expression template implementations at the very end
// This ensures all Tensor definitions are complete before templates are instantiated
#include "tensor_expr_impl.hpp"

namespace lfs::core::internal {
    inline bool shares_storage(const Tensor& left, const Tensor& right) {
        return left.shares_storage_with(right);
    }
} // namespace lfs::core::internal
