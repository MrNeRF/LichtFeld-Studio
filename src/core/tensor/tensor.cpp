/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/checked_arithmetic.hpp"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/tensor_trace.hpp"
#include "internal/cuda_event_pool.hpp"
#include "internal/cuda_memory_guard.hpp"
#include "internal/cuda_stream_context.hpp"
#include "internal/lazy_executor.hpp"
#include "internal/memory_pool.hpp"
#include "internal/tensor_broadcast.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <format>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <print>
#include <utility>

// SIMD intrinsics for CPU optimization
#if defined(__AVX2__)
#include <immintrin.h>
#endif

// OpenMP for multi-threading
#ifdef _OPENMP
#include <omp.h>
#endif

namespace lfs::core {

    std::atomic<size_t> Tensor::next_id_{1};

    namespace {
        struct StorageAccountingCounter {
            std::atomic<uint64_t> live_bytes{0};
            std::atomic<uint64_t> live_allocations{0};
            std::atomic<uint64_t> total_bytes{0};
            std::atomic<uint64_t> total_allocations{0};
        };

        struct StorageAccountingState {
            StorageAccountingCounter cuda_direct;
            StorageAccountingCounter vulkan_external;
        };

        StorageAccountingState& storage_accounting_state() {
            static StorageAccountingState state;
            return state;
        }

        void add_counter(StorageAccountingCounter& counter, const size_t bytes) {
            if (bytes == 0) {
                return;
            }
            counter.live_bytes.fetch_add(bytes, std::memory_order_relaxed);
            counter.live_allocations.fetch_add(1, std::memory_order_relaxed);
            counter.total_bytes.fetch_add(bytes, std::memory_order_relaxed);
            counter.total_allocations.fetch_add(1, std::memory_order_relaxed);
        }

        void subtract_counter(StorageAccountingCounter& counter, const size_t bytes) {
            if (bytes == 0) {
                return;
            }

            const uint64_t current = counter.live_bytes.load(std::memory_order_relaxed);
            if (current >= bytes) {
                counter.live_bytes.fetch_sub(bytes, std::memory_order_relaxed);
            } else {
                counter.live_bytes.store(0, std::memory_order_relaxed);
            }

            const uint64_t allocations = counter.live_allocations.load(std::memory_order_relaxed);
            if (allocations > 0) {
                counter.live_allocations.fetch_sub(1, std::memory_order_relaxed);
            }
        }

        double mib(const uint64_t bytes) {
            return static_cast<double>(bytes) / (1024.0 * 1024.0);
        }

        [[nodiscard]] bool is_supported_device(const Device device) {
            return device == Device::CPU || device == Device::CUDA;
        }

        [[nodiscard]] bool is_supported_dtype(const DataType dtype) {
            return dtype_size(dtype) != 0;
        }

        struct NamedTensorContractOperand {
            std::string_view role;
            const Tensor* tensor;
        };

        std::string format_tensor_contract(
            const std::string_view operation,
            const std::string_view requirement,
            const std::initializer_list<NamedTensorContractOperand> tensors,
            const std::string_view context = {}) {
            std::string message(operation);
            message += ": ";
            message += requirement;
            if (!context.empty()) {
                message += " [";
                message += context;
                message += ']';
            }
            message += " (";
            bool first = true;
            for (const auto& [role, tensor] : tensors) {
                if (!first) {
                    message += ", ";
                }
                first = false;
                message += role;
                message += '=';
                message += tensor->str();
            }
            message += ')';
            return message;
        }

        std::string expected_dtypes(const std::initializer_list<DataType> expected) {
            std::string result = "expected=";
            bool first = true;
            for (const DataType dtype : expected) {
                if (!first) {
                    result += '|';
                }
                first = false;
                result += dtype_name(dtype);
            }
            return result;
        }

    } // namespace

    namespace tensor_contract {

        void require_valid(const Tensor& tensor,
                           const std::string_view operation,
                           const std::string_view role,
                           const SourceSite location) {
            if (!tensor.is_valid()) [[unlikely]] {
                detail::assertion_failed(
                    "LFS boundary contract", "tensor.is_valid()",
                    format_tensor_contract(operation, "invalid tensor", {{role, &tensor}}),
                    location);
            }
        }

        void require_same_device(const Tensor& reference,
                                 const Tensor& other,
                                 const std::string_view operation,
                                 const std::string_view reference_role,
                                 const std::string_view other_role,
                                 const SourceSite location) {
            if (reference.device() != other.device()) [[unlikely]] {
                detail::assertion_failed(
                    "LFS boundary contract", "tensor devices match",
                    format_tensor_contract(
                        operation, "device mismatch",
                        {{reference_role, &reference}, {other_role, &other}}),
                    location);
            }
        }

        void require_dtype(const Tensor& tensor,
                           const DataType expected,
                           const std::string_view operation,
                           const std::string_view role,
                           const SourceSite location) {
            require_dtype(tensor, {expected}, operation, role, location);
        }

        void require_dtype(const Tensor& tensor,
                           const std::initializer_list<DataType> expected,
                           const std::string_view operation,
                           const std::string_view role,
                           const SourceSite location) {
            if (std::find(expected.begin(), expected.end(), tensor.dtype()) == expected.end()) [[unlikely]] {
                detail::assertion_failed(
                    "LFS boundary contract", "tensor dtype is allowed",
                    format_tensor_contract(
                        operation, "dtype mismatch", {{role, &tensor}}, expected_dtypes(expected)),
                    location);
            }
        }

        void require_shape(const Tensor& reference,
                           const Tensor& other,
                           const std::string_view operation,
                           const std::string_view reference_role,
                           const std::string_view other_role,
                           const SourceSite location) {
            if (reference.shape() != other.shape()) [[unlikely]] {
                detail::assertion_failed(
                    "LFS boundary contract", "tensor shapes match",
                    format_tensor_contract(
                        operation, "shape mismatch",
                        {{reference_role, &reference}, {other_role, &other}}),
                    location);
            }
        }

        void require_shape(const Tensor& tensor,
                           const TensorShape& expected,
                           const std::string_view operation,
                           const std::string_view role,
                           const SourceSite location) {
            if (tensor.shape() != expected) [[unlikely]] {
                const std::string context = "expected=" + expected.str();
                detail::assertion_failed(
                    "LFS boundary contract", "tensor shape matches expected shape",
                    format_tensor_contract(
                        operation, "shape mismatch", {{role, &tensor}}, context),
                    location);
            }
        }

    } // namespace tensor_contract

    size_t Tensor::storage_allocation_bytes(const TensorShape& shape,
                                            const size_t capacity,
                                            const DataType dtype) {
        LFS_ASSERT_MSG(is_supported_dtype(dtype),
                       "tensor storage allocation received an invalid dtype");
        if (shape.rank() == 0) {
            LFS_ASSERT_MSG(capacity == 0,
                           "scalar tensor storage cannot have row capacity");
            return checked_product(shape.elements(), dtype_size(dtype),
                                   "scalar tensor storage");
        }

        size_t row_elements = 1;
        for (size_t i = 1; i < shape.rank(); ++i) {
            row_elements = checked_product(row_elements, shape[i], "tensor row");
        }

        const size_t rows = capacity == 0 ? shape[0] : capacity;
        LFS_ASSERT_MSG(rows >= shape[0],
                       "tensor capacity cannot be smaller than its logical row count");
        const size_t elements = checked_product(rows, row_elements,
                                                "tensor storage element count");
        return checked_product(elements, dtype_size(dtype),
                               "tensor storage byte count");
    }

    void Tensor::record_storage_allocation(const StorageAccountingKind kind,
                                           const size_t bytes) {
        auto& state = storage_accounting_state();
        switch (kind) {
        case StorageAccountingKind::CudaDirect:
            add_counter(state.cuda_direct, bytes);
            break;
        case StorageAccountingKind::VulkanExternal:
            add_counter(state.vulkan_external, bytes);
            break;
        }
    }

    void Tensor::record_storage_deallocation(const StorageAccountingKind kind,
                                             const size_t bytes) {
        auto& state = storage_accounting_state();
        switch (kind) {
        case StorageAccountingKind::CudaDirect:
            subtract_counter(state.cuda_direct, bytes);
            break;
        case StorageAccountingKind::VulkanExternal:
            subtract_counter(state.vulkan_external, bytes);
            break;
        }
    }

    std::string Tensor::storage_memory_summary() {
        const auto& state = storage_accounting_state();
        std::ostringstream oss;
        const auto append = [&oss](std::string_view label, const StorageAccountingCounter& counter) {
            const uint64_t live_bytes = counter.live_bytes.load(std::memory_order_relaxed);
            const uint64_t live_allocations = counter.live_allocations.load(std::memory_order_relaxed);
            const uint64_t total_bytes = counter.total_bytes.load(std::memory_order_relaxed);
            const uint64_t total_allocations = counter.total_allocations.load(std::memory_order_relaxed);
            oss << label << ": live=" << std::fixed << std::setprecision(2) << mib(live_bytes)
                << " MiB/" << live_allocations << " allocs, total=" << mib(total_bytes)
                << " MiB/" << total_allocations << " allocs";
        };

        oss << "Tensor storage accounting: ";
        append("cuda_direct", state.cuda_direct);
        oss << "; ";
        append("vulkan_external", state.vulkan_external);
        return oss.str();
    }

    void Tensor::log_storage_memory() {
        log_storage_memory({});
    }

    void Tensor::log_storage_memory(const std::string_view label) {
        if (label.empty()) {
            LOG_INFO("{}", storage_memory_summary());
        } else {
            LOG_INFO("{} - {}", label, storage_memory_summary());
        }
    }

    // TensorLeaf implementation
    TensorLeaf::TensorLeaf(Tensor tensor)
        : tensor_ptr_(std::make_shared<Tensor>(std::move(tensor))) {}

    Tensor TensorLeaf::eval_impl() const {
        // Materialize non-contiguous or offset tensors
        if (tensor_ptr_->storage_offset() != 0 || !tensor_ptr_->is_contiguous()) {
            return tensor_ptr_->contiguous();
        }
        return *tensor_ptr_;
    }

    const TensorShape& TensorLeaf::shape_impl() const { return tensor_ptr_->shape(); }
    Device TensorLeaf::device_impl() const { return tensor_ptr_->device(); }
    DataType TensorLeaf::dtype_impl() const { return tensor_ptr_->dtype(); }
    cudaStream_t TensorLeaf::stream_hint_impl() const { return tensor_ptr_ ? tensor_ptr_->stream() : nullptr; }

    Tensor Tensor::make_deferred_expr_tensor(TensorShape shape,
                                             const Device device,
                                             const DataType dtype,
                                             std::function<Tensor()> materializer) {
        return make_deferred_expr_tensor(
            std::move(shape), device, dtype, std::move(materializer), {});
    }

    Tensor Tensor::make_deferred_expr_tensor(TensorShape shape,
                                             Device device,
                                             DataType dtype,
                                             std::function<Tensor()> materializer,
                                             std::vector<uint64_t> lazy_input_ids) {
        if (!materializer) {
            throw std::runtime_error("make_deferred_expr_tensor: materializer is empty");
        }

        Tensor deferred;
        deferred.shape_ = std::move(shape);
        deferred.strides_ = deferred.shape_.strides();
        deferred.storage_offset_ = 0;
        deferred.is_contiguous_ = true;
        deferred.device_ = device;
        deferred.dtype_ = dtype;
        deferred.is_view_ = false;
        deferred.data_ = nullptr;
        deferred.data_owner_.reset();
        deferred.state_->has_deferred_expr = true;
        deferred.state_->materializing_deferred_expr = false;
        deferred.state_->deferred_expr_node_id = 0;
        deferred.state_->deferred_materializer = std::move(materializer);
        deferred.id_ = next_id_++;
        deferred.compute_alignment();

        if (internal::lazy_ir_active()) {
            deferred.state_->deferred_expr_node_id =
                internal::lazy_ir_record_deferred(deferred, "deferred_expr", lazy_input_ids);
        }
        if (deferred.state_->deferred_expr_node_id != 0) {
            internal::lazy_executor_register_deferred_materializer(
                deferred.state_->deferred_expr_node_id,
                deferred.state_->deferred_materializer,
                std::weak_ptr<void>(deferred.state_));
        }

        return deferred;
    }

    void Tensor::materialize_if_deferred() {
        if (!state_ || !state_->has_deferred_expr) {
            return;
        }
        const uint64_t lazy_node_id = lazy_expr_id();
        Tensor materialized;

        if (lazy_node_id != 0) {
            Tensor cached_materialized;
            if (internal::lazy_executor_lookup_cached_materialization(lazy_node_id, cached_materialized)) {
                materialized = std::move(cached_materialized);
            }
        }

        if (!materialized.is_valid()) {
            if (state_->materializing_deferred_expr) {
                throw std::runtime_error("Recursive deferred tensor materialization detected");
            }
            if (!state_->deferred_materializer) {
                throw std::runtime_error("Deferred tensor has no materializer");
            }

            state_->materializing_deferred_expr = true;
            auto materializer = std::move(state_->deferred_materializer);
            state_->deferred_materializer = {};

            try {
                materialized = internal::lazy_planner_execute_plan_for_tensor(*this, materializer);
            } catch (...) {
                state_->deferred_materializer = std::move(materializer);
                state_->materializing_deferred_expr = false;
                throw;
            }
            state_->materializing_deferred_expr = false;

            if (lazy_node_id != 0) {
                internal::lazy_executor_cache_materialization(lazy_node_id, materialized);
            }
        }

        if (!materialized.is_valid()) {
            throw std::runtime_error("Deferred tensor materializer returned invalid tensor");
        }

        const size_t preserved_id = id_;
        const bool preserved_tracked = state_->tracked;
        const std::string preserved_name = state_->name;
        const cudaStream_t preserved_stream = state_->stream;

        data_ = materialized.data_;
        data_owner_ = std::move(materialized.data_owner_);
        shape_ = std::move(materialized.shape_);
        strides_ = std::move(materialized.strides_);
        storage_offset_ = materialized.storage_offset_;
        is_contiguous_ = materialized.is_contiguous_;
        device_ = materialized.device_;
        dtype_ = materialized.dtype_;
        is_view_ = materialized.is_view_;
        storage_meta_ = std::move(materialized.storage_meta_);
#ifndef NDEBUG
        view_generation_snapshot_ = materialized.view_generation_snapshot_;
#endif

        state_ = std::move(materialized.state_);
        if (!state_) {
            state_ = std::make_shared<TensorState>();
        }
        state_->has_deferred_expr = false;
        state_->materializing_deferred_expr = false;
        if (lazy_node_id != 0) {
            internal::lazy_executor_unregister_deferred_materializer(lazy_node_id);
        }
        state_->deferred_expr_node_id = 0;
        state_->deferred_materializer = {};
        state_->tracked = state_->tracked || preserved_tracked;
        if (!preserved_name.empty()) {
            state_->name = preserved_name;
        }
        // Keep the actual materialization stream when one exists. The deferred hint is
        // only a fallback for materializers that do not stamp stream metadata.
        if (state_->stream == nullptr && preserved_stream) {
            state_->stream = preserved_stream;
        }

        id_ = preserved_id;
        compute_alignment();

        materialized.data_ = nullptr;
        materialized.storage_offset_ = 0;
        materialized.is_view_ = false;
    }

    // ============= Helper Functions =============

    // Check if strides represent contiguous memory layout (row-major)
    static bool check_contiguous(const TensorShape& shape, const std::vector<size_t>& strides) {
        if (strides.empty())
            return true;
        if (strides.size() != shape.rank())
            return false;

        // Check if strides match row-major contiguous layout
        size_t expected_stride = 1;
        for (int i = static_cast<int>(shape.rank()) - 1; i >= 0; --i) {
            if (strides[i] != expected_stride)
                return false;
            expected_stride *= shape[i];
        }
        return true;
    }

    // ============= Constructors & Destructor =============

    Tensor::Tensor(void* data, TensorShape shape, Device device, DataType dtype)
        : data_(data),
          data_owner_(nullptr), // Non-owning
          state_(std::make_shared<TensorState>()),
          shape_(shape),
          strides_(shape.strides()),
          storage_offset_(0),
          is_contiguous_(true),
          device_(device),
          dtype_(dtype),
          is_view_(true),
          id_(next_id_++) {

        LFS_ASSERT_MSG(is_supported_device(device_),
                       "Tensor constructor received an invalid device");
        LFS_ASSERT_MSG(is_supported_dtype(dtype_),
                       "Tensor constructor received an invalid dtype");
        LFS_ASSERT_MSG(data_ != nullptr || shape_.elements() == 0,
                       "Tensor constructor received null storage for a non-empty tensor");

        compute_alignment();

        if (profiling_enabled_) {
            LOG_DEBUG("Created tensor #{} (non-owning view): shape={}, device={}, dtype={}",
                      id_, shape_.str(), device_name(device_), dtype_name(dtype_));
        }
    }

    // ============= Copy Constructor - SHALLOW COPY (LibTorch behavior) =============
    Tensor::Tensor(const Tensor& other)
        : data_(other.data_),
          data_owner_(other.data_owner_),
          state_(std::make_shared<TensorState>(*other.state_)),
          shape_(other.shape_),
          strides_(other.strides_),
          storage_offset_(other.storage_offset_),
          is_contiguous_(other.is_contiguous_),
          device_(other.device_),
          dtype_(other.dtype_),
          is_view_(other.is_view_),
          storage_meta_(other.storage_meta_),
#ifndef NDEBUG
          view_generation_snapshot_(other.view_generation_snapshot_),
#endif
          id_(next_id_++),
          lazy_ir_registered_(false) {

        if (profiling_enabled_) {
            LOG_DEBUG("Shallow copy: tensor #{} from #{}: shape={}, device={}, dtype={}, refcount={}",
                      id_, other.id_, shape_.str(), device_name(device_), dtype_name(dtype_),
                      data_owner_ ? data_owner_.use_count() : 0);
        }
    }

    // ============= Copy Assignment - Context-aware (Shallow or Deep) =============
    Tensor& Tensor::operator=(const Tensor& other) {
        if (this == &other) {
            return *this;
        }
        // PyTorch semantics: slice/view assignment does deep copy, regular assignment does shallow copy
        // Example: t1[0:5] = t2  -> deep copy into the slice
        //          t1 = t2        -> shallow copy (both point to same data)

        // If LHS is a view/slice and shapes match, do deep copy
        if (is_view_ && is_valid() && other.is_valid() &&
            shape_ == other.shape_ && dtype_ == other.dtype_) {

            // Deep copy the data, accounting for storage offsets
            if (numel() > 0) {
                // Calculate actual data pointers accounting for storage offsets
                void* dst_ptr = static_cast<char*>(data_) + storage_offset_ * dtype_size(dtype_);
                const void* src_ptr = other.data_ptr();

                if (device_ == Device::CUDA && other.device_ == Device::CUDA) {
                    const cudaStream_t copy_stream = stream();
                    other.sync_to_stream(copy_stream);
                    LFS_CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, bytes(), cudaMemcpyDeviceToDevice, copy_stream));
                    record_stream(copy_stream);
                    other.record_stream(copy_stream);
                } else if (device_ == Device::CUDA && other.device_ == Device::CPU) {
                    const cudaStream_t copy_stream = stream();
                    LFS_CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, bytes(), cudaMemcpyHostToDevice, copy_stream));
                    record_stream(copy_stream);
                    other.record_stream(copy_stream);
                } else if (device_ == Device::CPU && other.device_ == Device::CUDA) {
                    const cudaStream_t copy_stream = stream();
                    other.sync_to_stream(copy_stream);
                    LFS_CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, bytes(), cudaMemcpyDeviceToHost, copy_stream));
                    record_stream(copy_stream);
                    other.record_stream(copy_stream);
                } else {
                    std::memcpy(dst_ptr, src_ptr, bytes());
                }
                // No sync after copy - operations are async like LibTorch
                // User can sync explicitly if needed via cudaDeviceSynchronize()
            }

            if (profiling_enabled_) {
                LOG_DEBUG("Deep copy assign (view): tensor #{} from #{}: shape={}, device={}, dtype={}, offset={}, src_offset={}",
                          id_, other.id_, shape_.str(), device_name(device_), dtype_name(dtype_),
                          storage_offset_, other.storage_offset_);
            }

            return *this;
        }

        if (lazy_ir_registered_) {
            internal::lazy_ir_unregister_tensor(id_);
            lazy_ir_registered_ = false;
        }

        const size_t old_capacity = state_ ? state_->capacity : 0;
        const size_t new_capacity = other.state_ ? other.state_->capacity : 0;
        void* const old_data = data_;
        void* const new_data = other.data_;

        data_ = other.data_;
        data_owner_ = other.data_owner_;
        shape_ = other.shape_;
        strides_ = other.strides_;
        storage_offset_ = other.storage_offset_;
        is_contiguous_ = other.is_contiguous_;
        device_ = other.device_;
        dtype_ = other.dtype_;
        is_view_ = other.is_view_;
        storage_meta_ = other.storage_meta_;
#ifndef NDEBUG
        view_generation_snapshot_ = other.view_generation_snapshot_;
#endif
        if (old_capacity > 1000000 && new_capacity < old_capacity) {
            LOG_WARN("Tensor assignment reduced capacity: old_capacity={} -> new_capacity={}, old_data={}, new_data={}",
                     old_capacity, new_capacity, old_data, new_data);
        }
        state_ = std::make_shared<TensorState>(*other.state_);
        id_ = next_id_++;

        if (profiling_enabled_) {
            LOG_DEBUG("Shallow assign: tensor #{} from #{}: shape={}, device={}, dtype={}, refcount={}",
                      id_, other.id_, shape_.str(), device_name(device_), dtype_name(dtype_),
                      data_owner_ ? data_owner_.use_count() : 0);
        }

        return *this;
    }

    // ============= Move Constructor =============
    Tensor::Tensor(Tensor&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          data_owner_(std::move(other.data_owner_)),
          state_(std::move(other.state_)),
          shape_(std::move(other.shape_)),
          strides_(std::move(other.strides_)),
          storage_offset_(std::exchange(other.storage_offset_, 0)),
          is_contiguous_(std::exchange(other.is_contiguous_, true)),
          device_(other.device_),
          dtype_(other.dtype_),
          is_view_(std::exchange(other.is_view_, false)),
          storage_meta_(std::move(other.storage_meta_)),
#ifndef NDEBUG
          view_generation_snapshot_(other.view_generation_snapshot_),
#endif
          id_(std::exchange(other.id_, 0)),
          lazy_ir_registered_(std::exchange(other.lazy_ir_registered_, false)) {

        if (!state_) {
            state_ = std::make_shared<TensorState>();
        }
        if (!other.state_) {
            other.state_ = std::make_shared<TensorState>();
        }

        if (profiling_enabled_) {
            LOG_DEBUG("Move constructed: tensor #{} (moved-from is now invalid)", id_);
        }
    }

    // ============= Move Assignment =============
    Tensor& Tensor::operator=(Tensor&& other) {
        if (this != &other) {
            // PyTorch semantics: slice/view assignment does deep copy even for rvalues
            // This handles: t1.slice(0, 0, 5) = t2.slice(0, 5, 10)
            // where the RHS is a temporary view

            if (is_view_ && is_valid() && other.is_valid() &&
                shape_ == other.shape_ && dtype_ == other.dtype_) {

                // Deep copy the data from the temporary view
                if (numel() > 0) {
                    void* dst_ptr = static_cast<char*>(data_) + storage_offset_ * dtype_size(dtype_);
                    const void* src_ptr = other.data_ptr();

                    if (device_ == Device::CUDA && other.device_ == Device::CUDA) {
                        const cudaStream_t copy_stream = stream();
                        other.sync_to_stream(copy_stream);
                        LFS_CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, bytes(), cudaMemcpyDeviceToDevice, copy_stream));
                        record_stream(copy_stream);
                        other.record_stream(copy_stream);
                    } else if (device_ == Device::CUDA && other.device_ == Device::CPU) {
                        const cudaStream_t copy_stream = stream();
                        LFS_CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, bytes(), cudaMemcpyHostToDevice, copy_stream));
                        record_stream(copy_stream);
                        other.record_stream(copy_stream);
                    } else if (device_ == Device::CPU && other.device_ == Device::CUDA) {
                        const cudaStream_t copy_stream = stream();
                        other.sync_to_stream(copy_stream);
                        LFS_CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, bytes(), cudaMemcpyDeviceToHost, copy_stream));
                        record_stream(copy_stream);
                        other.record_stream(copy_stream);
                    } else {
                        std::memcpy(dst_ptr, src_ptr, bytes());
                    }
                    // No sync after copy - operations are async like LibTorch
                }

                if (profiling_enabled_) {
                    LOG_DEBUG("Deep copy move assign (view): tensor #{} from #{}: shape={}, device={}, dtype={}",
                              id_, other.id_, shape_.str(), device_name(device_), dtype_name(dtype_));
                }

                return *this;
            }

            if (lazy_ir_registered_) {
                internal::lazy_ir_unregister_tensor(id_);
            }

            data_ = std::exchange(other.data_, nullptr);
            data_owner_ = std::move(other.data_owner_);
            state_ = std::move(other.state_);
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            storage_offset_ = std::exchange(other.storage_offset_, 0);
            is_contiguous_ = std::exchange(other.is_contiguous_, true);
            device_ = other.device_;
            dtype_ = other.dtype_;
            is_view_ = std::exchange(other.is_view_, false);
            storage_meta_ = std::move(other.storage_meta_);
#ifndef NDEBUG
            view_generation_snapshot_ = other.view_generation_snapshot_;
#endif
            id_ = std::exchange(other.id_, 0);
            lazy_ir_registered_ = std::exchange(other.lazy_ir_registered_, false);
            if (!state_) {
                state_ = std::make_shared<TensorState>();
            }
            if (!other.state_) {
                other.state_ = std::make_shared<TensorState>();
            }

            if (profiling_enabled_) {
                LOG_DEBUG("Move assigned: tensor #{} (moved-from is now invalid)", id_);
            }
        }
        return *this;
    }

    CudaMemoryPool& CudaMemoryPool::instance() {
        static CudaMemoryPool pool;
        return pool;
    }

    namespace {
        thread_local std::string g_pool_pending_label;
    }

    CudaMemoryPool::LabelGuard::LabelGuard(std::string_view label)
        : previous_(std::move(g_pool_pending_label)),
          active_(!label.empty()) {
        if (active_) {
            g_pool_pending_label.assign(label);
        }
    }

    CudaMemoryPool::LabelGuard::~LabelGuard() {
        g_pool_pending_label = std::move(previous_);
    }

    std::string_view CudaMemoryPool::current_label() noexcept {
        return g_pool_pending_label;
    }

    void Tensor::set_stream(cudaStream_t stream) {
        LFS_ASSERT_MSG(is_valid(),
                       "set_stream requires a valid tensor");
        if (device_ == Device::CUDA && data_owner_ && state_->stream != stream) {
            // Rehoming changes where future writes occur. Preserve prior writes
            // from the old home before changing allocator ownership metadata.
            bridgeStreams(state_->stream, stream);
            CudaMemoryPool::instance().rehome_stream(data_owner_.get(), stream);
        }
        state_->stream = stream;
    }

    void Tensor::record_stream(cudaStream_t stream) const {
        LFS_ASSERT_MSG(is_valid(),
                       "record_stream requires a valid tensor");
        if (!data_owner_) {
            return;
        }
        if (device_ == Device::CUDA) {
            CudaMemoryPool::instance().record_stream(data_owner_.get(), stream);
        } else {
            PinnedMemoryAllocator::instance().record_stream(data_owner_.get(), stream);
        }
    }

    void Tensor::sync_to_stream(cudaStream_t execution_stream) const {
        LFS_ASSERT_MSG(is_valid(),
                       "sync_to_stream requires a valid tensor");
        if (device_ != Device::CUDA) {
            return;
        }
        const cudaStream_t home = stream();
        if (execution_stream == home) {
            return;
        }
        bridgeStreams(home, execution_stream);
        record_stream(execution_stream);
    }

    void Tensor::relabel_allocation_for_profiler() {
        if (device_ != Device::CUDA || data_ == nullptr || state_->name.empty()) {
            return;
        }
        try {
            lfs::diagnostics::VramProfiler::instance().relabelAllocation(data_, state_->name);
        } catch (...) {
            // Diagnostics must never throw out of tensor operations.
        }
    }

    void Tensor::trim_memory_pool() {
        CudaMemoryPool::instance().trim_cached_memory();
        PinnedMemoryAllocator::instance().empty_cache();
    }

    void Tensor::shutdown_memory_pool() {
        CudaMemoryPool::instance().shutdown();
    }

    void Tensor::set_memory_pool_iteration(int iteration) {
        CudaMemoryPool::instance().set_iteration(iteration);
    }

    void Tensor::print_memory_pool_stats() {
        CudaMemoryPool::instance().print_stats();
    }

    // ============= Destructor =============
    Tensor::~Tensor() {
        if (data_owner_ && profiling_enabled_) {
            LOG_DEBUG("Destroying tensor #{}: shape={}, device={}, refcount={}",
                      id_, shape_.str(), device_name(device_), data_owner_.use_count());
        }
        if (lazy_ir_registered_) {
            internal::lazy_ir_unregister_tensor(id_);
        }
        // shared_ptr automatically handles memory cleanup when refcount reaches 0
    }

    // ============= Deep Copy (explicit) =============
    Tensor Tensor::clone() const {
        LFS_CUDA_BREADCRUMB_STREAM("tensor.clone", stream());
        debug::OpTraceGuard trace("clone", *this, LFS_SOURCE_SITE_CURRENT());

        LFS_ASSERT_MSG(is_valid(),
                       "clone requires a valid tensor");

        if (numel() == 0) {
            // Return empty tensor with same shape and properties
            return empty(shape_, device_, dtype_);
        }

        // If not contiguous, materialize first then clone
        if (!is_contiguous_) {
            return contiguous();
        }

        // Create new tensor with same properties
        auto result = empty(shape_, device_, dtype_);

        // Copy data using data_ptr() which accounts for storage_offset
        size_t num_bytes = this->bytes();

        if (device_ == Device::CUDA) {
            LFS_CUDA_CHECK_MSG(
                cudaMemcpy(result.data_ptr(), data_ptr(), num_bytes,
                           cudaMemcpyDeviceToDevice),
                "tensor clone (bytes={}, source_shape={}, source_pointer={}, "
                "destination_pointer={})",
                num_bytes, shape_.str(), data_ptr(), result.data_ptr());
        } else {
            std::memcpy(result.data_ptr(), data_ptr(), num_bytes);
        }

        if (profiling_enabled_) {
            LOG_DEBUG("Deep clone: tensor #{} from #{}: copied {} bytes",
                      result.id_, id_, num_bytes);
        }

        return result;
    }

    // ============= Contiguous (materializes non-contiguous tensors) =============
    Tensor Tensor::contiguous() const {
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid(),
                       "contiguous requires a valid tensor");

        // Already contiguous? Just return shallow copy
        if (is_contiguous_) {
            return *this;
        }

        // Need to materialize strided view into contiguous layout
        auto result = empty(shape_, device_, dtype_);

        if (numel() == 0) {
            return result;
        }

        if (device_ == Device::CUDA) {
            const char* src_base = static_cast<const char*>(data_) + storage_offset_ * dtype_size(dtype_);
            const size_t rank = shape_.rank();

            if (rank >= 2 && rank <= 4) {
                tensor_ops::launch_strided_copy_immediate(
                    src_base, result.data_, shape_.dims(), strides_, numel(), dtype_, result.stream());
            } else {
                CudaDeviceMemory<size_t> d_shape(rank);
                CudaDeviceMemory<size_t> d_strides(rank);
                LFS_ASSERT_MSG(d_shape.valid() && d_strides.valid(),
                               "contiguous failed to allocate CUDA shape metadata");
                LFS_CUDA_CHECK(d_shape.copy_from_host(shape_.dims().data(), rank));
                LFS_CUDA_CHECK(d_strides.copy_from_host(strides_.data(), rank));
                tensor_ops::launch_strided_copy(
                    src_base, result.data_, d_shape.get(), d_strides.get(), rank,
                    numel(), dtype_, result.stream());
            }
        } else {
            // CPU strided copy - optimized for common cases with SIMD + multi-threading
            const char* src_base = static_cast<const char*>(data_) + storage_offset_ * dtype_size(dtype_);
            char* dst = static_cast<char*>(result.data_);
            size_t elem_size = dtype_size(dtype_);

            // Fast path for 3D tensors (common case: HWC→CHW image permute)
            // MULTI-THREADED + VECTORIZED with AVX2
            if (shape_.rank() == 3 && elem_size == sizeof(float)) {
                const float* src = reinterpret_cast<const float*>(src_base);
                float* dst_float = reinterpret_cast<float*>(dst);

                size_t dim0 = shape_[0];
                size_t dim1 = shape_[1];
                size_t dim2 = shape_[2];
                size_t stride0 = strides_[0];
                size_t stride1 = strides_[1];
                size_t stride2 = strides_[2];

                // Use multi-threading for medium/large tensors (>100KB)
                // Small tensors: thread overhead not worth it
                bool use_parallel = bytes() > 100 * 1024;

#if defined(__AVX2__)
                constexpr size_t SIMD_WIDTH = 8; // AVX2 processes 8 floats

// MULTI-THREADED: Parallelize over BOTH dim0 and dim1 using collapse(2)
// For HWC→CHW permute ([H, W, C] → [C, H, W]), this means:
//   dim0 = C (typically 3), dim1 = H (typically 720+), dim2 = W (typically 820+)
// Using collapse(2) distributes work across C×H iterations (e.g., 3×720=2160 iterations)
// This ensures good work distribution even when dim0 is small.
// collapse() requires OpenMP 3.0 (2008)
#if _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (use_parallel) schedule(static)
#else
#pragma omp parallel for if (use_parallel) schedule(static)
#endif
                for (size_t i0 = 0; i0 < dim0; ++i0) {
                    for (size_t i1 = 0; i1 < dim1; ++i1) {
                        size_t i2 = 0;

                        // FAST PATH 1: stride2 == 1 (contiguous) - vectorized load
                        if (stride2 == 1) {
                            for (; i2 + SIMD_WIDTH <= dim2; i2 += SIMD_WIDTH) {
                                size_t src_idx = i0 * stride0 + i1 * stride1 + i2;
                                size_t dst_idx = i0 * dim1 * dim2 + i1 * dim2 + i2;

                                __m256 vec = _mm256_loadu_ps(&src[src_idx]);
                                _mm256_storeu_ps(&dst_float[dst_idx], vec);
                            }
                        }
                        // FAST PATH 2: Small stride - unrolled scalar (HWC→CHW case)
                        else if (stride2 <= 4) {
                            // Unroll by 4 for instruction-level parallelism
                            for (; i2 + 4 <= dim2; i2 += 4) {
                                size_t src_idx0 = i0 * stride0 + i1 * stride1 + i2 * stride2;
                                size_t dst_idx0 = i0 * dim1 * dim2 + i1 * dim2 + i2;

                                dst_float[dst_idx0] = src[src_idx0];
                                dst_float[dst_idx0 + 1] = src[src_idx0 + stride2];
                                dst_float[dst_idx0 + 2] = src[src_idx0 + 2 * stride2];
                                dst_float[dst_idx0 + 3] = src[src_idx0 + 3 * stride2];
                            }
                        }

                        // Scalar tail for remaining elements
                        for (; i2 < dim2; ++i2) {
                            size_t src_idx = i0 * stride0 + i1 * stride1 + i2 * stride2;
                            size_t dst_idx = i0 * dim1 * dim2 + i1 * dim2 + i2;
                            dst_float[dst_idx] = src[src_idx];
                        }
                    }
                }
#else
// Fallback: scalar version (no SIMD), but still multi-threaded with collapse(2)
#if _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (use_parallel) schedule(static)
#else
#pragma omp parallel for if (use_parallel) schedule(static)
#endif
                for (int64_t i0 = 0; i0 < static_cast<int64_t>(dim0); ++i0) {
                    for (int64_t i1 = 0; i1 < static_cast<int64_t>(dim1); ++i1) {
                        for (size_t i2 = 0; i2 < dim2; ++i2) {
                            size_t src_idx = i0 * stride0 + i1 * stride1 + i2 * stride2;
                            size_t dst_idx = i0 * dim1 * dim2 + i1 * dim2 + i2;
                            dst_float[dst_idx] = src[src_idx];
                        }
                    }
                }
#endif
            }
            // Fast path for 2D tensors (matrix transpose)
            // MULTI-THREADED + VECTORIZED with AVX2
            else if (shape_.rank() == 2 && elem_size == sizeof(float)) {
                const float* src = reinterpret_cast<const float*>(src_base);
                float* dst_float = reinterpret_cast<float*>(dst);

                size_t rows = shape_[0];
                size_t cols = shape_[1];
                size_t stride0 = strides_[0];
                size_t stride1 = strides_[1];

                bool use_parallel = bytes() > 100 * 1024;

#if defined(__AVX2__)
                constexpr size_t SIMD_WIDTH = 8;

// MULTI-THREADED: Each thread processes rows
#pragma omp parallel for if (use_parallel) schedule(static)
                for (size_t i = 0; i < rows; ++i) {
                    size_t j = 0;

                    // Vectorize if stride1 == 1 (contiguous in column direction)
                    if (stride1 == 1) {
                        for (; j + SIMD_WIDTH <= cols; j += SIMD_WIDTH) {
                            size_t src_idx = i * stride0 + j;
                            size_t dst_idx = i * cols + j;

                            __m256 vec = _mm256_loadu_ps(&src[src_idx]);
                            _mm256_storeu_ps(&dst_float[dst_idx], vec);
                        }
                    }

                    // Scalar tail
                    for (; j < cols; ++j) {
                        size_t src_idx = i * stride0 + j * stride1;
                        size_t dst_idx = i * cols + j;
                        dst_float[dst_idx] = src[src_idx];
                    }
                }
#else
// Fallback: scalar version, but still multi-threaded
#pragma omp parallel for if (use_parallel) schedule(static)
                for (int64_t i = 0; i < static_cast<int64_t>(rows); ++i) {
                    for (size_t j = 0; j < cols; ++j) {
                        size_t src_idx = i * stride0 + j * stride1;
                        size_t dst_idx = i * cols + j;
                        dst_float[dst_idx] = src[src_idx];
                    }
                }
#endif
            }
            // Generic fallback for arbitrary ranks and types
            else {
                std::vector<size_t> indices(shape_.rank(), 0);

                for (size_t i = 0; i < numel(); ++i) {
                    // Calculate source offset using strides
                    size_t src_offset = 0;
                    for (size_t d = 0; d < shape_.rank(); ++d) {
                        src_offset += indices[d] * strides_[d];
                    }

                    // Copy element
                    std::memcpy(dst + i * elem_size, src_base + src_offset * elem_size, elem_size);

                    // Increment indices (row-major)
                    for (int d = static_cast<int>(shape_.rank()) - 1; d >= 0; --d) {
                        indices[d]++;
                        if (indices[d] < shape_[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
        }

        return result;
    }

    // ============= Device Transfer =============
    Tensor Tensor::to(Device device, cudaStream_t stream) const {
        LFS_CUDA_BREADCRUMB_STREAM("tensor.device_copy", stream);
        materialize_if_deferred();
        const char* op_name = (device == Device::CUDA) ? "to_cuda" : "to_cpu";
        debug::OpTraceGuard trace(op_name, *this, LFS_SOURCE_SITE_CURRENT());

        LFS_ASSERT_MSG(is_valid(),
                       "device transfer requires a valid tensor");

        if (device_ == device) {
            return clone();
        }

        // OPTIMIZATION: Handle non-contiguous tensor transfers intelligently
        // NEW: Use GPU-side strided upload kernel for CPU→GPU transfers!
        // This eliminates CPU-side materialization entirely.
        //
        // Performance comparison:
        // OLD: CPU AVX2 materialize (2-3ms) + Upload (1-2ms) = ~4-5ms
        // NEW: GPU strided upload (direct from pinned host memory) = ~1-2ms
        //
        // The GPU can directly read from pinned host memory via PCIe while
        // simultaneously rearranging the layout - no CPU work needed!
        if (!is_contiguous_) {
            if (device_ == Device::CUDA && device == Device::CPU) {
                // GPU→CPU: Materialize on GPU FIRST (GPU kernel is faster)
                LOG_DEBUG("GPU→CPU: materializing on GPU before download");
                return contiguous().to(device, stream);
            } else if (device_ == Device::CPU && device == Device::CUDA) {
                // CPU→GPU: Use fused strided upload kernel!
                LOG_DEBUG("CPU→GPU non-contiguous: using fused strided upload kernel (rank={})", shape_.rank());

                auto t = empty(shape_, Device::CUDA, dtype_);
                const cudaStream_t transfer_stream = stream ? stream : t.stream();
                if (t.stream() != transfer_stream) {
                    t.set_stream(transfer_stream);
                }
                if (numel() == 0) {
                    return t;
                }

                // Account for storage offset
                const char* src = static_cast<const char*>(data_) + storage_offset_ * dtype_size(dtype_);

                // FAST PATHS: Use optimized kernels with immediate parameters (no device memory allocation!)
                // This eliminates 2x cudaMalloc + 2x cudaMemcpy overhead (~0.5-1ms saved per upload)
                // ASYNC: Kernel launches asynchronously - CUDA runtime handles synchronization automatically
                if (shape_.rank() == 3) {
                    LOG_DEBUG("Using optimized rank-{} strided upload (no metadata allocation)", shape_.rank());

                    // Pass host pointers directly - the launcher will use them as immediate kernel parameters
                    tensor_ops::launch_strided_upload(
                        src,
                        t.data_,
                        shape_.dims().data(), // Host memory pointer
                        strides_.data(),      // Host memory pointer
                        shape_.rank(),
                        numel(),
                        dtype_,
                        transfer_stream);

                    // The host owner can retire immediately after this call. Register
                    // the actual reader stream with the pinned allocator before return.
                    record_stream(transfer_stream);
                    LOG_DEBUG("Optimized rank-{} strided upload launched (async): {} elements", shape_.rank(), numel());
                    return t;
                }

                // GENERIC PATH: For rank > 3, allocate device memory for metadata
                LOG_DEBUG("Using generic strided upload (requires metadata allocation for rank={})", shape_.rank());

                size_t* d_shape;
                size_t* d_strides;
                LFS_CUDA_CHECK(cudaMalloc(&d_shape, shape_.rank() * sizeof(size_t)));
                LFS_CUDA_CHECK(cudaMalloc(&d_strides, shape_.rank() * sizeof(size_t)));

                LFS_CUDA_CHECK(cudaMemcpy(d_shape, shape_.dims().data(),
                                          shape_.rank() * sizeof(size_t),
                                          cudaMemcpyHostToDevice));
                LFS_CUDA_CHECK(cudaMemcpy(d_strides, strides_.data(),
                                          shape_.rank() * sizeof(size_t),
                                          cudaMemcpyHostToDevice));

                // Launch generic strided upload kernel
                tensor_ops::launch_strided_upload(
                    src,
                    t.data_,
                    d_shape,
                    d_strides,
                    shape_.rank(),
                    numel(),
                    dtype_,
                    transfer_stream);

                // Free metadata immediately (kernel has already captured the data)
                // NO SYNC: Kernel launches asynchronously for better PCIe overlap
                LFS_CUDA_CHECK(cudaFree(d_shape));
                LFS_CUDA_CHECK(cudaFree(d_strides));

                record_stream(transfer_stream);

                LOG_DEBUG("Generic strided upload launched (async): {} elements", numel());
                return t;
            }
        }

        auto t = empty(shape_, device, dtype_);
        if (numel() == 0) {
            return t;
        }

        // Account for storage offset
        const char* src = static_cast<const char*>(data_) + storage_offset_ * dtype_size(dtype_);

        // LOG_DEBUG("to(Device): storage_offset_={}, dtype_size={}, src_offset_bytes={}, bytes_to_copy={}",
        //           storage_offset_, dtype_size(dtype_), storage_offset_ * dtype_size(dtype_), bytes());

        if (device_ == Device::CPU && device == Device::CUDA) {
            // Use cudaMemcpyAsync with pinned memory for maximum PCIe bandwidth (~7-11 GB/s)
            // CPU tensor now uses pinned memory (allocated via PinnedMemoryAllocator)

            // PROFILING: Track H2D uploads to identify bottlenecks
            static std::atomic<uint64_t> h2d_counter{0};
            static std::atomic<uint64_t> h2d_bytes{0};
            static std::atomic<uint64_t> h2d_tiny{0};   // < 64 bytes (metadata)
            static std::atomic<uint64_t> h2d_small{0};  // 64B - 1KB
            static std::atomic<uint64_t> h2d_medium{0}; // 1KB - 1MB
            static std::atomic<uint64_t> h2d_large{0};  // > 1MB (images)

            uint64_t current_count = h2d_counter.fetch_add(1, std::memory_order_relaxed) + 1;
            uint64_t upload_bytes = bytes();
            h2d_bytes.fetch_add(upload_bytes, std::memory_order_relaxed);

            // Categorize by size
            if (upload_bytes < 64) {
                h2d_tiny.fetch_add(1, std::memory_order_relaxed);
            } else if (upload_bytes < 1024) {
                h2d_small.fetch_add(1, std::memory_order_relaxed);
            } else if (upload_bytes < 1024 * 1024) {
                h2d_medium.fetch_add(1, std::memory_order_relaxed);
            } else {
                h2d_large.fetch_add(1, std::memory_order_relaxed);
            }

            // Log details for TINY uploads to understand the bottleneck
            /*
            if (upload_bytes < 64) {
                std::string shape_str = "[";
                for (size_t i = 0; i < shape_.rank(); ++i) {
                    if (i > 0) shape_str += ", ";
                    shape_str += std::to_string(shape_.dims()[i]);
                }
                shape_str += "]";

                if (current_count <= 100) {
                    // First 100 tiny uploads
                    LOG_WARN("TINY H2D Upload #{}: {} bytes, shape={}",
                             current_count, upload_bytes, shape_str);
                } else if (current_count >= 500 && current_count <= 510) {
                    // Tiny uploads DURING TRAINING (after initialization) - these are the bottleneck!
                    LOG_WARN("TRAINING TINY H2D Upload #{}: {} bytes, shape={} - THIS IS THE BOTTLENECK!",
                             current_count, upload_bytes, shape_str);
                }

                // SPECIFIC TRACKING FOR 32-BYTE UPLOADS (8 floats)
                if (upload_bytes == 32) {
                    static std::atomic<uint64_t> h2d_32byte_counter{0};
                    uint64_t count_32 = h2d_32byte_counter.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (count_32 <= 20 || (count_32 >= 400 && count_32 <= 420)) {
                        LOG_ERROR("32-BYTE H2D Upload #{}: shape={}, dtype={}, numel={}",
                                 count_32, shape_str, static_cast<int>(dtype_), numel());
                    }
                }

                // SPECIFIC TRACKING FOR 4-BYTE UPLOADS (single scalars)
                if (upload_bytes == 4) {
                    static std::atomic<uint64_t> h2d_4byte_counter{0};
                    uint64_t count_4 = h2d_4byte_counter.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (count_4 <= 20 || (count_4 >= 500 && count_4 <= 520)) {
                        LOG_ERROR("4-BYTE H2D Upload #{}: shape={}, dtype={}, numel={}",
                                 count_4, shape_str, static_cast<int>(dtype_), numel());
                    }
                }
            } else if (current_count % 1000 == 0) {
                LOG_WARN("H2D Upload #{}: {:.1f} MB total | Tiny(<64B): {}, Small(64B-1KB): {}, Medium(1KB-1MB): {}, Large(>1MB): {}",
                         current_count, h2d_bytes.load() / 1024.0 / 1024.0,
                         h2d_tiny.load(), h2d_small.load(), h2d_medium.load(), h2d_large.load());
            }
            */

            cudaStream_t transfer_stream = stream ? stream : t.stream();
            if (t.stream() != transfer_stream) {
                t.set_stream(transfer_stream);
            }
            LFS_CUDA_CHECK(cudaMemcpyAsync(t.data_, src, bytes(), cudaMemcpyHostToDevice, transfer_stream));

            record_stream(transfer_stream);

            // If stream is provided, caller is responsible for sync.
            // Otherwise wait only on transfer_stream — the H2D is the only work
            // we submitted, and a full cudaDeviceSynchronize was draining
            // unrelated GPU work (Vulkan compute, other CUDA streams), turning
            // sub-KB uploads into multi-ms calls during concurrent rendering.
            if (!stream) {
                LFS_CUDA_CHECK(cudaStreamSynchronize(transfer_stream));
            }
        } else if (device_ == Device::CUDA && device == Device::CPU) {
            // Order the transfer after the source's producing stream without
            // draining unrelated CUDA work. Explicit-stream calls remain async.
            const cudaStream_t transfer_stream = stream ? stream : nullptr;
            sync_to_stream(transfer_stream);
            if (stream) {
                t.set_stream(transfer_stream);
            }
            LFS_CUDA_CHECK(cudaMemcpyAsync(t.data_, src, bytes(), cudaMemcpyDeviceToHost, transfer_stream));
            record_stream(transfer_stream);
            t.record_stream(transfer_stream);

            if (!stream) {
                LFS_CUDA_CHECK(cudaStreamSynchronize(transfer_stream));
            }
        }

        return t;
    }

    // ============= Type Conversion =============
    Tensor Tensor::to(DataType dtype) const {
        LFS_CUDA_BREADCRUMB_STREAM("tensor.dtype_convert", stream());
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid(),
                       "dtype conversion requires a valid tensor");

        if (dtype_ == dtype) {
            return clone();
        }

        // If not contiguous, materialize first
        if (!is_contiguous_) {
            return contiguous().to(dtype);
        }

// Macro for type conversions using launch_convert_type
#define CONVERT_DTYPE_CUDA(FROM_TYPE, TO_TYPE, FROM_DTYPE, TO_DTYPE)                                                                         \
    if (dtype_ == FROM_DTYPE && dtype == TO_DTYPE) {                                                                                         \
        auto result = empty(shape_, device_, TO_DTYPE);                                                                                      \
        if (numel() == 0)                                                                                                                    \
            return result;                                                                                                                   \
        if (device_ == Device::CUDA) {                                                                                                       \
            tensor_ops::launch_convert_type<FROM_TYPE, TO_TYPE>(                                                                             \
                ptr<FROM_TYPE>(), result.ptr<TO_TYPE>(), numel(), result.stream());                                                          \
            /* No sync - tensor-to-tensor operation */                                                                                       \
            return result;                                                                                                                   \
        }                                                                                                                                    \
        /* CPU fallback */                                                                                                                   \
        const FROM_TYPE* src = ptr<FROM_TYPE>();                                                                                             \
        TO_TYPE* dst = result.ptr<TO_TYPE>();                                                                                                \
        for (size_t i = 0; i < numel(); ++i) {                                                                                               \
            if constexpr (std::is_same_v<FROM_TYPE, float> && std::is_same_v<TO_TYPE, uint8_t>) {                                            \
                dst[i] = static_cast<uint8_t>(std::round(std::clamp(static_cast<float>(src[i]), 0.0f, 255.0f)));                             \
            } else if constexpr (std::is_same_v<FROM_TYPE, int> && std::is_same_v<TO_TYPE, uint8_t>) {                                       \
                dst[i] = static_cast<uint8_t>(std::clamp(static_cast<int>(src[i]), 0, 255));                                                 \
            } else if constexpr (std::is_same_v<FROM_TYPE, int64_t> && std::is_same_v<TO_TYPE, uint8_t>) {                                   \
                dst[i] = static_cast<uint8_t>(std::clamp(static_cast<int64_t>(src[i]), static_cast<int64_t>(0), static_cast<int64_t>(255))); \
            } else {                                                                                                                         \
                dst[i] = static_cast<TO_TYPE>(src[i]);                                                                                       \
            }                                                                                                                                \
        }                                                                                                                                    \
        return result;                                                                                                                       \
    }

        // Bool <-> Float32 (manual - can't use launch_convert_type due to uint8_t conflict)
        if (dtype_ == DataType::Bool && dtype == DataType::Float32) {
            auto result = empty(shape_, device_, DataType::Float32);
            if (numel() == 0)
                return result;

            if (device_ == Device::CUDA) {
                // Use generic conversion (unsigned char -> float)
                tensor_ops::launch_convert_type<unsigned char, float>(
                    ptr<unsigned char>(), result.ptr<float>(), numel(), result.stream());
                // No sync - tensor-to-tensor GPU operation
            } else {
                const unsigned char* src = ptr<unsigned char>();
                float* dst = result.ptr<float>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = static_cast<float>(src[i]);
                }
            }
            return result;
        }

        if (dtype_ == DataType::Float32 && dtype == DataType::Bool) {
            auto result = empty(shape_, device_, DataType::Bool);
            if (numel() == 0)
                return result;

            if (device_ == Device::CUDA) {
                // Can't use launch_convert_type - need custom != 0 logic
                auto result_cpu = empty(shape_, Device::CPU, DataType::Bool);
                std::vector<float> temp(numel());
                LFS_CUDA_CHECK(cudaMemcpy(temp.data(), ptr<float>(), bytes(), cudaMemcpyDeviceToHost));

                unsigned char* dst_cpu = result_cpu.ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst_cpu[i] = (temp[i] != 0.0f) ? 1 : 0;
                }

                LFS_CUDA_CHECK(cudaMemcpy(result.ptr<unsigned char>(), dst_cpu, numel(), cudaMemcpyHostToDevice));
            } else {
                const float* src = ptr<float>();
                unsigned char* dst = result.ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = (src[i] != 0.0f) ? 1 : 0;
                }
            }
            return result;
        }

        // Float32 <-> Int32
        if (dtype_ == DataType::Float32 && dtype == DataType::Int32) {
            auto result = empty(shape_, device_, DataType::Int32);
            if (numel() == 0)
                return result;

            if (device_ == Device::CUDA) {
                tensor_ops::launch_convert_type<float, int>(
                    ptr<float>(), result.ptr<int>(), numel(), result.stream());
                // No sync - tensor-to-tensor GPU operation
            } else {
                const float* src = ptr<float>();
                int* dst = result.ptr<int>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = static_cast<int>(src[i]);
                }
            }
            return result;
        }
        CONVERT_DTYPE_CUDA(int, float, DataType::Int32, DataType::Float32)

        // UInt8 conversions
        CONVERT_DTYPE_CUDA(float, uint8_t, DataType::Float32, DataType::UInt8)
        CONVERT_DTYPE_CUDA(uint8_t, float, DataType::UInt8, DataType::Float32)
        CONVERT_DTYPE_CUDA(int, uint8_t, DataType::Int32, DataType::UInt8)
        CONVERT_DTYPE_CUDA(uint8_t, int, DataType::UInt8, DataType::Int32)

        // Bool -> UInt8: Bool storage is already normalized to 0 or 1.
        if (dtype_ == DataType::Bool && dtype == DataType::UInt8) {
            auto result = empty(shape_, device_, dtype);
            if (numel() > 0) {
                if (device_ == Device::CUDA) {
                    LFS_CUDA_CHECK(cudaMemcpy(const_cast<void*>(result.data_ptr()), data_ptr(), bytes(), cudaMemcpyDeviceToDevice));
                } else {
                    std::memcpy(const_cast<void*>(result.data_ptr()), data_ptr(), bytes());
                }
            }
            return result;
        }

        // UInt8 -> Bool follows Torch's nonzero semantics rather than merely
        // reinterpreting the byte storage.
        if (dtype_ == DataType::UInt8 && dtype == DataType::Bool) {
            auto result = empty(shape_, device_, DataType::Bool);
            if (numel() == 0)
                return result;

            if (device_ == Device::CUDA) {
                tensor_ops::launch_convert_type<uint8_t, bool>(
                    ptr<uint8_t>(), result.ptr<bool>(), numel(), result.stream());
            } else {
                const uint8_t* src = ptr<uint8_t>();
                uint8_t* dst = result.ptr<uint8_t>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = src[i] != 0 ? 1 : 0;
                }
            }
            return result;
        }

        // Bool <-> Int32: Manual conversion (bool != 0 logic)
        if (dtype_ == DataType::Int32 && dtype == DataType::Bool) {
            auto result = empty(shape_, device_, DataType::Bool);
            if (numel() == 0)
                return result;

            if (device_ == Device::CUDA) {
                // Copy to CPU, convert, copy back
                auto result_cpu = empty(shape_, Device::CPU, DataType::Bool);
                std::vector<int> temp(numel());
                LFS_CUDA_CHECK(cudaMemcpy(temp.data(), ptr<int>(), bytes(), cudaMemcpyDeviceToHost));

                unsigned char* dst_cpu = result_cpu.ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst_cpu[i] = (temp[i] != 0) ? 1 : 0;
                }

                LFS_CUDA_CHECK(cudaMemcpy(result.ptr<unsigned char>(), result_cpu.ptr<unsigned char>(),
                                          numel() * sizeof(unsigned char), cudaMemcpyHostToDevice));
            } else {
                const int* src = ptr<int>();
                unsigned char* dst = result.ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = (src[i] != 0) ? 1 : 0;
                }
            }
            return result;
        }

        if (dtype_ == DataType::Bool && dtype == DataType::Int32) {
            auto result = empty(shape_, device_, DataType::Int32);
            if (numel() == 0)
                return result;

            if (device_ == Device::CUDA) {
                // Use generic conversion (unsigned char -> int)
                tensor_ops::launch_convert_type<unsigned char, int>(
                    ptr<unsigned char>(), result.ptr<int>(), numel(), result.stream());
                // No sync - tensor-to-tensor GPU operation
            } else {
                const unsigned char* src = ptr<unsigned char>();
                int* dst = result.ptr<int>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = static_cast<int>(src[i]);
                }
            }
            return result;
        }

        // Bool -> Int64
        if (dtype_ == DataType::Bool && dtype == DataType::Int64) {
            auto result = empty(shape_, device_, DataType::Int64);
            if (numel() == 0)
                return result;

            if (device_ == Device::CUDA) {
                // Use generic conversion (unsigned char -> int64_t)
                tensor_ops::launch_convert_type<unsigned char, int64_t>(
                    ptr<unsigned char>(), result.ptr<int64_t>(), numel(), result.stream());
                // No sync - tensor-to-tensor GPU operation
            } else {
                const unsigned char* src = ptr<unsigned char>();
                int64_t* dst = result.ptr<int64_t>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = static_cast<int64_t>(src[i]);
                }
            }
            return result;
        }

        // Int64 -> Bool
        if (dtype_ == DataType::Int64 && dtype == DataType::Bool) {
            auto result = empty(shape_, device_, DataType::Bool);
            if (numel() == 0)
                return result;

            if (device_ == Device::CUDA) {
                // Copy to CPU, convert, copy back
                auto result_cpu = empty(shape_, Device::CPU, DataType::Bool);
                std::vector<int64_t> temp(numel());
                LFS_CUDA_CHECK(cudaMemcpy(temp.data(), ptr<int64_t>(), bytes(), cudaMemcpyDeviceToHost));

                unsigned char* dst_cpu = result_cpu.ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst_cpu[i] = (temp[i] != 0) ? 1 : 0;
                }

                LFS_CUDA_CHECK(cudaMemcpy(result.ptr<unsigned char>(), result_cpu.ptr<unsigned char>(),
                                          numel() * sizeof(unsigned char), cudaMemcpyHostToDevice));
            } else {
                const int64_t* src = ptr<int64_t>();
                unsigned char* dst = result.ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = (src[i] != 0) ? 1 : 0;
                }
            }
            return result;
        }

        CONVERT_DTYPE_CUDA(int64_t, uint8_t, DataType::Int64, DataType::UInt8)
        CONVERT_DTYPE_CUDA(uint8_t, int64_t, DataType::UInt8, DataType::Int64)

        // Bool -> Float16
        if (dtype_ == DataType::Bool && dtype == DataType::Float16) {
            auto result = empty(shape_, device_, DataType::Float16);
            if (numel() == 0)
                return result;

            if (device_ == Device::CUDA) {
                // Use generic conversion (unsigned char -> __half)
                tensor_ops::launch_convert_type<unsigned char, __half>(
                    ptr<unsigned char>(), result.ptr<__half>(), numel(), result.stream());
                // No sync - tensor-to-tensor GPU operation
            } else {
                const unsigned char* src = ptr<unsigned char>();
                __half* dst = result.ptr<__half>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = __float2half(static_cast<float>(src[i]));
                }
            }
            return result;
        }

        // Float16 -> Bool
        if (dtype_ == DataType::Float16 && dtype == DataType::Bool) {
            auto result = empty(shape_, device_, DataType::Bool);
            if (numel() == 0)
                return result;

            if (device_ == Device::CUDA) {
                // Copy to CPU, convert, copy back
                auto result_cpu = empty(shape_, Device::CPU, DataType::Bool);
                std::vector<__half> temp(numel());
                LFS_CUDA_CHECK(cudaMemcpy(temp.data(), ptr<__half>(), bytes(), cudaMemcpyDeviceToHost));
                LFS_CUDA_CHECK(cudaDeviceSynchronize()); // Ensure copy completes

                unsigned char* dst_cpu = result_cpu.ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst_cpu[i] = (__half2float(temp[i]) != 0.0f) ? 1 : 0;
                }

                LFS_CUDA_CHECK(cudaMemcpy(result.ptr<unsigned char>(), result_cpu.ptr<unsigned char>(),
                                          numel() * sizeof(unsigned char), cudaMemcpyHostToDevice));
                LFS_CUDA_CHECK(cudaDeviceSynchronize()); // Ensure copy completes
            } else {
                const __half* src = ptr<__half>();
                unsigned char* dst = result.ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = (__half2float(src[i]) != 0.0f) ? 1 : 0;
                }
            }
            return result;
        }

        // Int64 conversions
        CONVERT_DTYPE_CUDA(int64_t, float, DataType::Int64, DataType::Float32)
        CONVERT_DTYPE_CUDA(float, int64_t, DataType::Float32, DataType::Int64)
        CONVERT_DTYPE_CUDA(int, int64_t, DataType::Int32, DataType::Int64)

        // Int64 -> Int32: CRITICAL SYNCHRONIZATION for item() reads
        // Without sync, item<int>() may read before conversion completes, getting garbage
        if (dtype_ == DataType::Int64 && dtype == DataType::Int32) {
            auto result = empty(shape_, device_, DataType::Int32);
            if (numel() == 0)
                return result;

            if (device_ == Device::CUDA) {
                tensor_ops::launch_convert_type<int64_t, int>(
                    ptr<int64_t>(), result.ptr<int>(), numel(), result.stream());
                // CRITICAL: Sync to ensure conversion completes before item() reads
                LFS_CUDA_CHECK(cudaDeviceSynchronize());
            } else {
                const int64_t* src = ptr<int64_t>();
                int* dst = result.ptr<int>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = static_cast<int>(src[i]);
                }
            }
            return result;
        }

        // Float16 conversions
        CONVERT_DTYPE_CUDA(float, __half, DataType::Float32, DataType::Float16)
        CONVERT_DTYPE_CUDA(__half, float, DataType::Float16, DataType::Float32)
        CONVERT_DTYPE_CUDA(int, __half, DataType::Int32, DataType::Float16)
        CONVERT_DTYPE_CUDA(__half, int, DataType::Float16, DataType::Int32)
        CONVERT_DTYPE_CUDA(int64_t, __half, DataType::Int64, DataType::Float16)
        CONVERT_DTYPE_CUDA(__half, int64_t, DataType::Float16, DataType::Int64)
        CONVERT_DTYPE_CUDA(uint8_t, __half, DataType::UInt8, DataType::Float16)
        CONVERT_DTYPE_CUDA(__half, uint8_t, DataType::Float16, DataType::UInt8)

#undef CONVERT_DTYPE_CUDA

        LFS_ASSERT_MSG(false,
                       std::format("dtype conversion from {} to {} is unsupported",
                                   dtype_name(dtype_), dtype_name(dtype)));
    }

    // ============= In-place Operations =============

    Tensor& Tensor::zero_() {
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid(),
                       "zero_ requires a valid tensor");
        if (numel() == 0) {
            return *this;
        }

        // Account for storage offset
        char* dest = static_cast<char*>(data_) + storage_offset_ * dtype_size(dtype_);

        if (device_ == Device::CUDA) {
            LFS_CUDA_CHECK(cudaMemset(dest, 0, bytes()));
        } else {
            std::memset(dest, 0, bytes());
        }

        return *this;
    }

    Tensor& Tensor::fill_(float value) {
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid(),
                       "fill_ requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 || dtype_ == DataType::Int32 || dtype_ == DataType::Bool,
                       "fill_ currently supports only Float32, Int32, and Bool");
        LFS_ASSERT_MSG(std::isfinite(value),
                       "fill_ value must be finite");
        if (numel() == 0) {
            return *this;
        }

        // CRITICAL FIX: For non-contiguous tensors (from slice/view operations),
        // we must respect strides and fill only the elements in the view
        if (!is_contiguous()) {
            // For non-contiguous tensors, iterate and use operator[] which respects strides
            const size_t n = numel();

            // For CUDA non-contiguous tensors: use CUDA kernel that respects strides
            if (device_ == Device::CUDA) {
                // Use CUDA kernel for strided fill (much faster than element-by-element cudaMemcpy)
                if (dtype_ == DataType::Float32) {
                    tensor_ops::launch_fill_strided<float>(
                        static_cast<float*>(data_), value, shape_.dims(), strides_, storage_offset_, n, stream());
                } else if (dtype_ == DataType::Int32) {
                    int int_val = static_cast<int>(value);
                    tensor_ops::launch_fill_strided<int>(
                        static_cast<int*>(data_), int_val, shape_.dims(), strides_, storage_offset_, n, stream());
                } else if (dtype_ == DataType::Bool) {
                    unsigned char bool_val = (value != 0.0f) ? 1 : 0;
                    tensor_ops::launch_fill_strided<unsigned char>(
                        static_cast<unsigned char*>(data_), bool_val, shape_.dims(), strides_, storage_offset_, n, stream());
                }
                // Sync for the no-stream overload (maintains original behavior)
                LFS_CUDA_CHECK(cudaDeviceSynchronize());
                return *this;
            }

            // CPU non-contiguous: manually compute offsets using strides
            std::vector<size_t> indices(ndim(), 0);
            for (size_t linear_idx = 0; linear_idx < n; ++linear_idx) {
                // Convert linear index to multi-dimensional indices
                size_t remaining = linear_idx;
                for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
                    indices[d] = remaining % shape_[d];
                    remaining /= shape_[d];
                }

                // Calculate physical offset and set value
                size_t offset = storage_offset_;
                for (size_t d = 0; d < ndim(); ++d) {
                    offset += indices[d] * strides_[d];
                }

                if (dtype_ == DataType::Float32) {
                    static_cast<float*>(data_)[offset] = value;
                } else if (dtype_ == DataType::Int32) {
                    static_cast<int*>(data_)[offset] = static_cast<int>(value);
                } else if (dtype_ == DataType::Bool) {
                    static_cast<unsigned char*>(data_)[offset] = (value != 0.0f) ? 1 : 0;
                }
            }
            return *this;
        }

        // For contiguous tensors, use the fast path with memcpy
        // Account for storage offset
        void* dest = static_cast<char*>(data_) + storage_offset_ * dtype_size(dtype_);

        // Handle Bool dtype
        if (dtype_ == DataType::Bool) {
            unsigned char bool_val = (value != 0.0f) ? 1 : 0;
            if (device_ == Device::CUDA) {
                std::vector<unsigned char> temp(numel(), bool_val);
                LFS_CUDA_CHECK(cudaMemcpy(dest, temp.data(), bytes(), cudaMemcpyHostToDevice));
            } else {
                unsigned char* data = static_cast<unsigned char*>(dest);
                std::fill(data, data + numel(), bool_val);
            }
            return *this;
        }

        // Handle Int32 dtype
        if (dtype_ == DataType::Int32) {
            int int_val = static_cast<int>(value);
            if (device_ == Device::CUDA) {
                std::vector<int> temp(numel(), int_val);
                LFS_CUDA_CHECK(cudaMemcpy(dest, temp.data(), bytes(), cudaMemcpyHostToDevice));
            } else {
                int* data = static_cast<int*>(dest);
                std::fill(data, data + numel(), int_val);
            }
            return *this;
        }

        // Handle Float32 dtype (original code)
        if (device_ == Device::CUDA) {
            std::vector<float> temp(numel(), value);
            LFS_CUDA_CHECK(cudaMemcpy(dest, temp.data(), bytes(), cudaMemcpyHostToDevice));
        } else {
            float* data = static_cast<float*>(dest);
            std::fill(data, data + numel(), value);
        }

        return *this;
    }

    Tensor& Tensor::fill_(float value, cudaStream_t stream) {
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid(),
                       "stream-aware fill_ requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 || dtype_ == DataType::Int32 || dtype_ == DataType::Bool,
                       "stream-aware fill_ currently supports only Float32, Int32, and Bool");
        LFS_ASSERT_MSG(std::isfinite(value),
                       "stream-aware fill_ value must be finite");
        if (numel() == 0) {
            return *this;
        }

        // Only CUDA tensors benefit from stream-aware fill
        if (device_ != Device::CUDA) {
            return fill_(value); // Fall back to sync version for CPU
        }

        const size_t n = numel();

        // Non-contiguous tensors: use strided kernel with stream
        if (!is_contiguous()) {
            if (dtype_ == DataType::Float32) {
                tensor_ops::launch_fill_strided<float>(
                    static_cast<float*>(data_), value, shape_.dims(), strides_, storage_offset_, n, stream);
            } else if (dtype_ == DataType::Int32) {
                int int_val = static_cast<int>(value);
                tensor_ops::launch_fill_strided<int>(
                    static_cast<int*>(data_), int_val, shape_.dims(), strides_, storage_offset_, n, stream);
            } else if (dtype_ == DataType::Bool) {
                unsigned char bool_val = (value != 0.0f) ? 1 : 0;
                tensor_ops::launch_fill_strided<unsigned char>(
                    static_cast<unsigned char*>(data_), bool_val, shape_.dims(), strides_, storage_offset_, n, stream);
            }
            return *this;
        }

        // Contiguous tensors: use cudaMemsetAsync for zeros, or strided kernel for non-zero
        void* dest = static_cast<char*>(data_) + storage_offset_ * dtype_size(dtype_);

        if (value == 0.0f) {
            // Fast path: use cudaMemsetAsync for zeros
            LFS_CUDA_CHECK(cudaMemsetAsync(dest, 0, bytes(), stream));
        } else {
            // Non-zero fill: use kernel (treat as 1D strided with stride=1)
            std::vector<size_t> shape_1d = {n};
            std::vector<size_t> strides_1d = {1};
            if (dtype_ == DataType::Float32) {
                tensor_ops::launch_fill_strided<float>(
                    static_cast<float*>(dest), value, shape_1d, strides_1d, 0, n, stream);
            } else if (dtype_ == DataType::Int32) {
                tensor_ops::launch_fill_strided<int>(
                    static_cast<int*>(dest), static_cast<int>(value), shape_1d, strides_1d, 0, n, stream);
            } else if (dtype_ == DataType::Bool) {
                tensor_ops::launch_fill_strided<unsigned char>(
                    static_cast<unsigned char*>(dest), (value != 0.0f) ? (unsigned char)1 : (unsigned char)0,
                    shape_1d, strides_1d, 0, n, stream);
            }
        }

        return *this;
    }

    Tensor& Tensor::copy_from(const Tensor& other) {
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid() && other.is_valid(),
                       "copy_from requires valid tensors");
        LFS_ASSERT_MSG(shape_ == other.shape_,
                       std::format("copy_from shape mismatch: {} vs {}", shape_.str(), other.shape_.str()));

        // Type conversion path
        if (dtype_ != other.dtype_) {
            // Fused int32→float32 strided scatter
            if (!is_contiguous() && other.is_contiguous() &&
                device_ == Device::CUDA && other.device_ == Device::CUDA &&
                dtype_ == DataType::Float32 && other.dtype_ == DataType::Int32 && ndim() == 2) {
                const size_t shape_arr[2] = {static_cast<size_t>(size(0)), static_cast<size_t>(size(1))};
                const size_t strides_arr[2] = {strides_[0], strides_[1]};
                tensor_ops::launch_strided_scatter_int32_to_float32(
                    other.data_ptr(), data_ptr(), shape_arr, strides_arr, 2, numel(), stream());
                return *this;
            }
            // Convert then copy
            const Tensor src = other.is_contiguous() ? other : other.contiguous();
            return copy_from(src.to(dtype_));
        }

        if (numel() == 0)
            return *this;

        const bool dst_contig = is_contiguous();
        const bool src_contig = other.is_contiguous();

        // Both contiguous: direct memcpy
        if (dst_contig && src_contig) {
            if (device_ == Device::CUDA && other.device_ == Device::CUDA) {
                LFS_CUDA_CHECK(cudaMemcpy(data_ptr(), other.data_ptr(), bytes(), cudaMemcpyDeviceToDevice));
            } else if (device_ == Device::CUDA && other.device_ == Device::CPU) {
                LFS_CUDA_CHECK(cudaMemcpy(data_ptr(), other.data_ptr(), bytes(), cudaMemcpyHostToDevice));
            } else if (device_ == Device::CPU && other.device_ == Device::CUDA) {
                LFS_CUDA_CHECK(cudaMemcpy(data_ptr(), other.data_ptr(), bytes(), cudaMemcpyDeviceToHost));
            } else {
                std::memcpy(data_ptr(), other.data_ptr(), bytes());
            }
            return *this;
        }

        // Strided destination, contiguous source (CUDA)
        if (!dst_contig && src_contig && device_ == Device::CUDA && other.device_ == Device::CUDA) {
            const size_t rank = ndim();
            std::vector<size_t> shape_vec(rank);
            for (size_t i = 0; i < rank; ++i)
                shape_vec[i] = static_cast<size_t>(size(i));

            if (rank >= 2 && rank <= 4) {
                tensor_ops::launch_strided_scatter_immediate(
                    other.data_ptr(), data_ptr(), shape_vec, strides_, numel(), dtype_, stream());
            } else {
                size_t* d_shape = nullptr;
                size_t* d_strides = nullptr;
                LFS_CUDA_CHECK(cudaMalloc(&d_shape, rank * sizeof(size_t)));
                LFS_CUDA_CHECK(cudaMalloc(&d_strides, rank * sizeof(size_t)));
                LFS_CUDA_CHECK(cudaMemcpy(d_shape, shape_vec.data(), rank * sizeof(size_t), cudaMemcpyHostToDevice));
                LFS_CUDA_CHECK(cudaMemcpy(d_strides, strides_.data(), rank * sizeof(size_t), cudaMemcpyHostToDevice));
                tensor_ops::launch_strided_scatter(
                    other.data_ptr(), data_ptr(), d_shape, d_strides, rank, numel(), dtype_, stream());
                LFS_CUDA_CHECK(cudaFree(d_shape));
                LFS_CUDA_CHECK(cudaFree(d_strides));
            }
            return *this;
        }

        // Strided destination, contiguous source (cross-device: move src to dst device first)
        if (!dst_contig && src_contig && device_ != other.device_) {
            return copy_from(other.to(device_));
        }

        // Fallback: materialize non-contiguous source
        if (!src_contig) {
            return copy_from(other.contiguous());
        }

        return *this;
    }

    // ============= Shape Operations =============

    Tensor Tensor::cat(const Tensor& other, int dim) const {
        std::vector<Tensor> tensors;
        tensors.push_back(*this);
        tensors.push_back(other);
        return Tensor::cat(tensors, dim);
    }

    // ============= Broadcasting =============

    Tensor Tensor::broadcast_to(const TensorShape& target_shape) const {
        LFS_ASSERT_MSG(is_valid(),
                       "broadcast_to requires a valid tensor");
        LFS_ASSERT_MSG(can_broadcast_to(target_shape),
                       std::format("cannot broadcast shape {} to {}", shape_.str(), target_shape.str()));
        if (state_ && state_->has_deferred_expr) {
            const uint64_t source_id = lazy_expr_id();
            Tensor source = *this;
            TensorShape deferred_shape = target_shape;
            std::vector<uint64_t> deferred_inputs;
            if (source_id != 0) {
                deferred_inputs.push_back(source_id);
            }
            Tensor deferred = make_deferred_expr_tensor(
                deferred_shape, device_, dtype_,
                [source = std::move(source), deferred_shape]() mutable {
                    Tensor materialized = source;
                    materialized.materialize_if_deferred();
                    return lfs::core::broadcast_to(materialized, deferred_shape);
                },
                std::move(deferred_inputs));
            deferred.set_stream(source.stream());
            return deferred;
        }
        return lfs::core::broadcast_to(*this, target_shape);
    }

    bool Tensor::can_broadcast_to(const TensorShape& target) const {
        auto result = broadcast::shape(shape_.dims(), target.dims());
        return !result.empty() && result == target.dims();
    }

    TensorShape Tensor::broadcast_shape(const TensorShape& other) const {
        auto result = broadcast::shape(shape_.dims(), other.dims());
        return result.empty() ? TensorShape() : TensorShape(result);
    }

    // ============= Special operations =============

    Tensor Tensor::normalize(int dim, float eps) const {
        LFS_ASSERT_MSG(is_valid(),
                       "normalize requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32,
                       "normalize currently supports only Float32");
        LFS_ASSERT_MSG(std::isfinite(eps) && eps > 0.0f,
                       "normalize epsilon must be finite and positive");
        if (dim != -1) {
            const int resolved = resolve_dim(dim);
            LFS_ASSERT_MSG(resolved >= 0 && resolved < static_cast<int>(shape_.rank()),
                           "normalize dimension is out of range");
        }

        if (dim == -1) {
            auto m = mean();
            auto s = std({}, false, false).add(eps);
            return sub(m).div(s);
        }
        std::vector<int> axes = {dim};
        auto m = mean(axes, true);
        auto s = std(axes, true, false).add(eps);
        return sub(m).div(s);
    }

    Tensor Tensor::logit(float eps) const {
        LFS_ASSERT_MSG(is_valid(),
                       "logit requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32,
                       "logit currently supports only Float32");
        LFS_ASSERT_MSG(std::isfinite(eps) && eps > 0.0f && eps < 0.5f,
                       "logit epsilon must be finite and in (0, 0.5)");

        auto x_clamped = clamp(eps, 1.0f - eps);
        auto one_minus_x = full(shape_, 1.0f, device_, dtype_).sub(x_clamped);
        return x_clamped.div(one_minus_x).log();
    }

    // ============= Bitwise Operations =============

    Tensor Tensor::operator~() const {
        LFS_ASSERT_MSG(is_valid(),
                       "bitwise NOT requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::Bool,
                       "bitwise NOT requires Bool dtype");

        // Use the new functor-based logical_not() method
        return logical_not();
    }

    Tensor Tensor::operator|(const Tensor& other) const {
        LFS_ASSERT_MSG(is_valid() && other.is_valid(),
                       "bitwise OR requires valid tensors");
        LFS_ASSERT_MSG(dtype_ == DataType::Bool && other.dtype() == DataType::Bool,
                       "bitwise OR requires Bool tensors");
        LFS_ASSERT_MSG(device_ == other.device(),
                       "bitwise OR requires tensors on the same device");

        return logical_or(other);
    }

    // ============= Clamp Operations =============

    Tensor& Tensor::clamp_(float min_val, float max_val) {
        LFS_ASSERT_MSG(is_valid(),
                       "clamp_ requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 || dtype_ == DataType::Int32,
                       "clamp_ currently supports only Float32 and Int32");
        LFS_ASSERT_MSG(!std::isnan(min_val) && !std::isnan(max_val) && min_val <= max_val,
                       "clamp_ bounds must not be NaN and must be ordered");
        if (numel() == 0) {
            return *this;
        }

        if (device_ == Device::CUDA) {
            if (dtype_ == DataType::Float32) {
                tensor_ops::launch_clamp_scalar(ptr<float>(), min_val, max_val, numel(), stream());
            } else if (dtype_ == DataType::Int32) {
                tensor_ops::launch_clamp_scalar_int(ptr<int>(),
                                                    static_cast<int>(min_val),
                                                    static_cast<int>(max_val),
                                                    numel(), stream());
            }
        } else {
            if (dtype_ == DataType::Float32) {
                float* data = ptr<float>();
                for (size_t i = 0; i < numel(); ++i) {
                    if (!std::isnan(data[i])) {
                        data[i] = std::clamp(data[i], min_val, max_val);
                    }
                }
            } else if (dtype_ == DataType::Int32) {
                int* data = ptr<int>();
                int min_int = static_cast<int>(min_val);
                int max_int = static_cast<int>(max_val);
                for (size_t i = 0; i < numel(); ++i) {
                    data[i] = std::clamp(data[i], min_int, max_int);
                }
            }
        }

        return *this;
    }

    Tensor& Tensor::clamp_min_(float min) {
        return clamp_(min, std::numeric_limits<float>::max());
    }

    Tensor& Tensor::clamp_max_(float max) {
        return clamp_(std::numeric_limits<float>::lowest(), max);
    }

    // ============= Cumulative sum =============

    Tensor Tensor::cumsum(int dim) const {
        LFS_ASSERT_MSG(is_valid(),
                       "cumsum requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 || dtype_ == DataType::Int32,
                       "cumsum currently supports only Float32 and Int32");

        dim = resolve_dim(dim);
        LFS_ASSERT_MSG(dim >= 0 && dim < static_cast<int>(shape_.rank()),
                       "cumsum dimension is out of range");

        auto result = clone();

        if (device_ == Device::CUDA) {
            tensor_ops::launch_cumsum(result.data_ptr(), shape_.dims().data(),
                                      shape_.rank(), dim, dtype_, result.stream());
            // No sync - returns tensor, not API boundary
        } else {
            if (dtype_ == DataType::Float32) {
                float* data = result.ptr<float>();

                auto strides = shape_.strides();
                size_t dim_stride = strides[dim];
                size_t dim_size = shape_[dim];
                size_t total = numel();

                for (size_t idx = 0; idx < total; ++idx) {
                    size_t coord_along_dim = (idx / dim_stride) % dim_size;

                    if (coord_along_dim == 0)
                        continue;

                    data[idx] += data[idx - dim_stride];
                }
            } else if (dtype_ == DataType::Int32) {
                int* data = result.ptr<int>();

                auto strides = shape_.strides();
                size_t dim_stride = strides[dim];
                size_t dim_size = shape_[dim];
                size_t total = numel();

                for (size_t idx = 0; idx < total; ++idx) {
                    size_t coord_along_dim = (idx / dim_stride) % dim_size;
                    if (coord_along_dim == 0)
                        continue;
                    data[idx] += data[idx - dim_stride];
                }
            }
        }

        return result;
    }

    // ============= TensorShape Implementation =============

    std::string TensorShape::str() const {
        if (dims_.empty()) {
            return "[]";
        }
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < dims_.size(); ++i) {
            if (i > 0)
                oss << ", ";
            oss << dims_[i];
        }
        oss << "]";
        return oss.str();
    }

    // ============= Tensor String Representation =============

    std::string Tensor::str() const {
        std::ostringstream oss;
        oss << "Tensor(";
        if (!is_valid()) {
            oss << "invalid";
        } else {
            oss << "shape=" << shape_.str();
            oss << ", device=" << device_name(device_);
            oss << ", dtype=" << dtype_name(dtype_);
            if (data_owner_) {
                oss << ", owned, refcount=" << data_owner_.use_count();
            } else {
                oss << ", view";
            }
            if (has_external_storage()) {
                oss << ", external=" << external_storage_kind();
            }
        }
        oss << ")";
        return oss.str();
    }

    // ============= Debug Functions =============

    void Tensor::log_info() const {
        log_info({});
    }

    void Tensor::log_info(const std::string& name) const {
        const std::string& prefix = name.empty() ? "Tensor" : name;

        if (!is_valid()) {
            LOG_INFO("{}: {}", prefix, str());
            return;
        }

        auto values = debug_values(10);

        std::ostringstream oss;
        oss << str() << "\n  Values: [";

        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0)
                oss << ", ";
            oss << std::fixed << std::setprecision(4) << values[i];
            if (i == 9 && numel() > 10) {
                oss << ", ... (" << numel() - 10 << " more)";
            }
        }
        oss << "]";

        LOG_INFO("{}: {}", prefix, oss.str());
    }

    void Tensor::print_formatted() const {
        print_formatted({}, 10);
    }

    void Tensor::print_formatted(const std::string& name, size_t max_per_dim) const {
        std::println("\n=== {} ===", name.empty() ? "Tensor" : name);
        std::println("{}", str());

        if (!is_valid()) {
            std::println("  (invalid tensor)");
            return;
        }

        if (shape_.rank() == 1) {
            print_1d(max_per_dim);
        } else if (shape_.rank() == 2) {
            print_2d(max_per_dim);
        } else {
            std::println("  [Higher dimensional tensor - showing first slice]");
            auto first_slice = slice(0, 0, 1);
            first_slice.squeeze().print_2d(max_per_dim);
        }
    }

    void Tensor::print_1d(size_t max_elem) const {
        if (!is_valid())
            return;

        auto values = debug_values(std::min(max_elem, numel()));
        std::print("  [");

        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0)
                std::print(", ");
            std::print("{:8.4f}", values[i]);
        }

        if (numel() > max_elem) {
            std::print(", ... ({} more)", numel() - max_elem);
        }
        std::println("]");
    }

    void Tensor::print_2d(size_t max_per_dim) const {
        if (!is_valid() || shape_.rank() != 2)
            return;

        size_t rows = std::min(max_per_dim, shape_[0]);
        size_t cols = std::min(max_per_dim, shape_[1]);

        auto values = debug_values(shape_[0] * shape_[1]);

        for (size_t i = 0; i < rows; ++i) {
            std::print("  [");
            for (size_t j = 0; j < cols; ++j) {
                if (j > 0)
                    std::print(", ");
                size_t idx = i * shape_[1] + j;
                std::print("{:8.4f}", values[idx]);
            }
            if (shape_[1] > cols) {
                std::print(", ... ({} more)", shape_[1] - cols);
            }
            std::print("]");

            if (i == rows - 1 && shape_[0] > rows) {
                std::print("  ... ({} more rows)", shape_[0] - rows);
            }
            std::println("");
        }
    }

    // ============= Utility Functions =============

    std::optional<Tensor> Tensor::try_reshape(TensorShape shape) const {
        if (!is_valid()) {
            return std::nullopt;
        }

        if (shape.elements() != numel()) {
            return std::nullopt;
        }

        return reshape(shape);
    }

    std::vector<Tensor> Tensor::split_batch(const Tensor& tensor, size_t batch_size) {
        std::vector<Tensor> batches;
        LFS_ASSERT_MSG(tensor.is_valid(),
                       "split_batch requires a valid tensor");
        LFS_ASSERT_MSG(tensor.shape().rank() > 0,
                       "split_batch requires at least one tensor dimension");
        LFS_ASSERT_MSG(batch_size > 0,
                       "split_batch batch size must be positive");

        size_t total_size = tensor.shape()[0];
        size_t num_batches = (total_size + batch_size - 1) / batch_size;

        for (size_t i = 0; i < num_batches; ++i) {
            size_t start = i * batch_size;
            size_t end = std::min(start + batch_size, total_size);
            batches.push_back(tensor.slice(0, start, end));
        }

        return batches;
    }

    float Tensor::item() const {
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid(),
                       "item requires a valid tensor");
        LFS_ASSERT_MSG(numel() == 1,
                       "item requires exactly one element");

        const void* raw_ptr = data_ptr();
        LFS_ASSERT_MSG(raw_ptr != nullptr,
                       "item found null tensor storage");
        float value = 0.0f;

        // Sync before reading from GPU
        if (device_ == Device::CUDA) {
            // API BOUNDARY: Sync before reading value from GPU
            LFS_CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Handle different dtypes
        switch (dtype_) {
        case DataType::Float32: {
            float temp;
            if (device_ == Device::CUDA) {
                LFS_CUDA_CHECK(cudaMemcpy(&temp, raw_ptr, sizeof(float), cudaMemcpyDeviceToHost));
            } else {
                temp = *static_cast<const float*>(raw_ptr);
            }
            value = temp;
            break;
        }
        case DataType::Int64: {
            int64_t temp;
            if (device_ == Device::CUDA) {
                LFS_CUDA_CHECK(cudaMemcpy(&temp, raw_ptr, sizeof(int64_t), cudaMemcpyDeviceToHost));
            } else {
                temp = *static_cast<const int64_t*>(raw_ptr);
            }
            value = static_cast<float>(temp);
            break;
        }
        case DataType::Int32: {
            int32_t temp;
            if (device_ == Device::CUDA) {
                LFS_CUDA_CHECK(cudaMemcpy(&temp, raw_ptr, sizeof(int32_t), cudaMemcpyDeviceToHost));
            } else {
                temp = *static_cast<const int32_t*>(raw_ptr);
            }
            value = static_cast<float>(temp);
            break;
        }
        case DataType::Bool: {
            unsigned char temp;
            if (device_ == Device::CUDA) {
                LFS_CUDA_CHECK(cudaMemcpy(&temp, raw_ptr, sizeof(unsigned char), cudaMemcpyDeviceToHost));
            } else {
                temp = *static_cast<const unsigned char*>(raw_ptr);
            }
            value = static_cast<float>(temp);
            break;
        }
        case DataType::UInt8: {
            uint8_t temp;
            if (device_ == Device::CUDA) {
                LFS_CUDA_CHECK(cudaMemcpy(&temp, raw_ptr, sizeof(uint8_t), cudaMemcpyDeviceToHost));
            } else {
                temp = *static_cast<const uint8_t*>(raw_ptr);
            }
            value = static_cast<float>(temp);
            break;
        }
        default:
            LFS_ASSERT_MSG(false,
                           "item encountered an unsupported dtype");
        }

        return value;
    }

    std::vector<float> Tensor::debug_values(size_t max_values) const {
        materialize_if_deferred();
        std::vector<float> values;

        LFS_ASSERT_MSG(is_valid(),
                       "debug_values requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32,
                       "debug_values currently supports only Float32");
        if (numel() == 0 || max_values == 0) {
            return values;
        }

        if (!is_contiguous_)
            return contiguous().debug_values(max_values);

        size_t n = std::min(max_values, numel());
        values.resize(n);

        if (device_ == Device::CUDA) {
            // API BOUNDARY: Sync before reading from GPU
            LFS_CUDA_CHECK(cudaDeviceSynchronize());
            LFS_CUDA_CHECK(cudaMemcpy(values.data(), data_ptr(), n * sizeof(float),
                                      cudaMemcpyDeviceToHost));
        } else {
            const float* data = ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                values[i] = data[i];
            }
        }

        return values;
    }

    std::vector<float> Tensor::to_vector() const {
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid(),
                       "to_vector requires a valid tensor");

        if (numel() == 0) {
            return {};
        }

        // If tensor is not contiguous, materialize it first
        if (!is_contiguous_) {
            return contiguous().to_vector();
        }

        // Handle Bool dtype by converting to float
        if (dtype_ == DataType::Bool) {
            auto float_tensor = to(DataType::Float32);
            return float_tensor.to_vector();
        }

        // Handle Int64 dtype by converting to float
        if (dtype_ == DataType::Int64) {
            auto float_tensor = to(DataType::Float32);
            return float_tensor.to_vector();
        }

        // Handle Int32 dtype by converting to float
        if (dtype_ == DataType::Int32) {
            auto float_tensor = to(DataType::Float32);
            return float_tensor.to_vector();
        }

        // Handle UInt8 dtype by converting to float
        if (dtype_ == DataType::UInt8) {
            auto float_tensor = to(DataType::Float32);
            return float_tensor.to_vector();
        }

        LFS_ASSERT_MSG(dtype_ == DataType::Float32,
                       "to_vector encountered an unsupported dtype");

        std::vector<float> result(numel());

        // Use data_ptr() which accounts for storage_offset
        const void* src = data_ptr();

        if (device_ == Device::CUDA) {
            // API BOUNDARY: Sync before reading from GPU
            LFS_CUDA_CHECK(cudaDeviceSynchronize());
            LFS_CUDA_CHECK(cudaMemcpy(result.data(), src, bytes(), cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(result.data(), src, bytes());
        }

        return result;
    }

    std::vector<int64_t> Tensor::to_vector_int64() const {
        materialize_if_deferred();
        LOG_DEBUG("to_vector_int64() called");
        LOG_DEBUG("  dtype: {}", dtype_name(dtype_));
        LOG_DEBUG("  device: {}", device_name(device_));
        LOG_DEBUG("  numel: {}", numel());
        LOG_DEBUG("  is_valid: {}", is_valid());

        LFS_ASSERT_MSG(is_valid(),
                       "to_vector_int64 requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::Int64,
                       "to_vector_int64 requires Int64 dtype");

        if (numel() == 0) {
            LOG_DEBUG("Empty tensor, returning empty vector");
            return {};
        }

        LOG_DEBUG("Creating result vector of size {}", numel());
        std::vector<int64_t> result(numel());

        if (device_ == Device::CUDA) {
            LOG_DEBUG("Copying from CUDA to CPU, bytes: {}", bytes());
            LFS_CUDA_CHECK(cudaMemcpy(result.data(), data_ptr(), bytes(), cudaMemcpyDeviceToHost));
            LOG_DEBUG("CUDA copy complete");
        } else {
            LOG_DEBUG("Copying from CPU memory, bytes: {}", bytes());
            std::memcpy(result.data(), data_ptr(), bytes());
            LOG_DEBUG("CPU copy complete");
        }

        LOG_DEBUG("to_vector_int64() complete, returning {} elements", result.size());
        return result;
    }

    std::vector<int> Tensor::to_vector_int() const {
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid(),
                       "to_vector_int requires a valid tensor");

        if (numel() == 0) {
            return {};
        }

        // Handle Bool dtype by converting to int
        if (dtype_ == DataType::Bool) {
            auto int_tensor = to(DataType::Int32);
            return int_tensor.to_vector_int();
        }

        // Handle Int64 by converting to Int32
        if (dtype_ == DataType::Int64) {
            auto int_tensor = to(DataType::Int32);
            return int_tensor.to_vector_int();
        }

        LFS_ASSERT_MSG(dtype_ == DataType::Int32,
                       "to_vector_int supports only Int32, Int64, and Bool");

        std::vector<int> result(numel());

        if (device_ == Device::CUDA) {
            LFS_CUDA_CHECK(cudaMemcpy(result.data(), data_ptr(), bytes(), cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(result.data(), data_ptr(), bytes());
        }

        return result;
    }

    std::vector<bool> Tensor::to_vector_bool() const {
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid(),
                       "to_vector_bool requires a valid tensor");

        // Support both Bool and UInt8 dtypes (UInt8 can be used as byte array)
        LFS_ASSERT_MSG(dtype_ == DataType::Bool || dtype_ == DataType::UInt8,
                       "to_vector_bool requires Bool or UInt8 dtype");

        if (numel() == 0) {
            return {};
        }

        std::vector<bool> result(numel());

        if (device_ == Device::CUDA) {
            std::vector<unsigned char> temp(numel());
            LFS_CUDA_CHECK(cudaMemcpy(temp.data(), data_ptr(), bytes(), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < numel(); ++i) {
                result[i] = temp[i] != 0;
            }
        } else {
            const unsigned char* data = ptr<unsigned char>();
            for (size_t i = 0; i < numel(); ++i) {
                result[i] = data[i] != 0;
            }
        }

        return result;
    }

    std::vector<uint8_t> Tensor::to_vector_uint8() const {
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid(),
                       "to_vector_uint8 requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::UInt8 || dtype_ == DataType::Bool,
                       "to_vector_uint8 requires UInt8 or Bool dtype");

        if (numel() == 0) {
            return {};
        }

        // Handle UInt8 dtype directly
        if (dtype_ == DataType::UInt8) {
            std::vector<uint8_t> result(numel());

            if (device_ == Device::CUDA) {
                if (stream()) {
                    LFS_CUDA_CHECK(cudaStreamSynchronize(stream()));
                } else {
                    LFS_CUDA_CHECK(cudaDeviceSynchronize());
                }
                LFS_CUDA_CHECK(cudaMemcpy(result.data(), data_ptr(), bytes(), cudaMemcpyDeviceToHost));
            } else {
                std::memcpy(result.data(), data_ptr(), bytes());
            }

            return result;
        }

        // Handle Bool dtype (convert to uint8)
        if (dtype_ == DataType::Bool) {
            std::vector<uint8_t> result(numel());

            if (device_ == Device::CUDA) {
                if (stream()) {
                    LFS_CUDA_CHECK(cudaStreamSynchronize(stream()));
                } else {
                    LFS_CUDA_CHECK(cudaDeviceSynchronize());
                }
                LFS_CUDA_CHECK(cudaMemcpy(result.data(), data_ptr(), bytes(), cudaMemcpyDeviceToHost));
            } else {
                const unsigned char* src = ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    result[i] = src[i];
                }
            }

            return result;
        }

        LFS_ASSERT_MSG(false,
                       "to_vector_uint8 reached an unsupported dtype");
    }

    void Tensor::dump_diagnostic(const std::string& filename) const {
        std::ofstream file;
        if (!lfs::core::open_file_for_write(lfs::core::utf8_to_path(filename), file)) {
            LOG_ERROR("Failed to open diagnostic dump file: {}", filename);
            return;
        }
        file << "=== Tensor Diagnostic Dump ===\n";
        file << std::format("Info: {}\n", str());
        file << std::format("Memory address: {}\n", data_);

        if (is_valid()) {
            file << std::format("Bytes: {}\n", bytes());

            if (device_ == Device::CPU || numel() < 10000) {
                auto values = to_vector();
                file << std::format("Values ({} total):\n", values.size());
                for (size_t i = 0; i < std::min(size_t(1000), values.size()); ++i) {
                    file << std::format("[{}]: {}\n", i, values[i]);
                }
            }
        } else {
            file << "Tensor is invalid\n";
        }

        file.close();
        LOG_INFO("Diagnostic dump saved to {}", filename);
    }

    // ============= Validation & Assertions =============

    Tensor& Tensor::assert_shape(TensorShape expected) {
        return assert_shape(std::move(expected), {});
    }

    Tensor& Tensor::assert_shape(TensorShape expected, const std::string& msg) {
        if (!is_valid()) {
            std::string error_msg = "Cannot assert shape on invalid tensor";
            throw TensorError(error_msg, this);
        }

        if (shape_ != expected) {
            std::string error_msg = msg.empty() ? "Shape assertion failed: expected " + expected.str() + " but got " + shape_.str() : msg;
            throw TensorError(error_msg, this);
        }
        return *this;
    }

    Tensor& Tensor::assert_device(Device expected) {
        if (!is_valid()) {
            std::string error_msg = "Cannot assert device on invalid tensor";
            throw TensorError(error_msg, this);
        }

        if (device_ != expected) {
            std::string error_msg = "Device assertion failed: expected " +
                                    std::string(device_name(expected)) + " but got " +
                                    std::string(device_name(device_));
            throw TensorError(error_msg, this);
        }
        return *this;
    }

    Tensor& Tensor::assert_dtype(DataType expected) {
        if (!is_valid()) {
            std::string error_msg = "Cannot assert dtype on invalid tensor";
            throw TensorError(error_msg, this);
        }

        if (dtype_ != expected) {
            std::string error_msg = "DataType assertion failed: expected " +
                                    std::string(dtype_name(expected)) + " but got " +
                                    std::string(dtype_name(dtype_));
            throw TensorError(error_msg, this);
        }
        return *this;
    }

    Tensor& Tensor::assert_finite() {
        if (!is_valid()) {
            std::string error_msg = "Cannot assert finite on invalid tensor";
            throw TensorError(error_msg, this);
        }

        if (has_nan() || has_inf()) {
            std::string error_msg = "Tensor contains NaN or Inf values";
            throw TensorError(error_msg, this);
        }
        return *this;
    }

    // ============= Comparison Utilities =============

    bool Tensor::has_nan() const {
        LFS_ASSERT_MSG(is_valid(),
                       "has_nan requires a valid tensor");
        if (numel() == 0) {
            return false;
        }

        // Use fast GPU check for CUDA tensors (only transfers 1 int back)
        if (device_ == Device::CUDA && dtype_ == DataType::Float32) {
            return tensor_ops::has_nan_gpu(ptr<float>(), numel(), stream());
        }

        // CPU fallback
        auto values = to_vector();
        return std::any_of(values.begin(), values.end(),
                           [](float x) { return std::isnan(x); });
    }

    bool Tensor::has_inf() const {
        LFS_ASSERT_MSG(is_valid(),
                       "has_inf requires a valid tensor");
        if (numel() == 0) {
            return false;
        }

        // Use fast GPU check for CUDA tensors (only transfers 1 int back)
        if (device_ == Device::CUDA && dtype_ == DataType::Float32) {
            return tensor_ops::has_inf_gpu(ptr<float>(), numel(), stream());
        }

        // CPU fallback
        auto values = to_vector();
        return std::any_of(values.begin(), values.end(),
                           [](float x) { return std::isinf(x); });
    }

    bool Tensor::all_close(const Tensor& other, float rtol, float atol) const {
        LFS_ASSERT_MSG(is_valid() && other.is_valid(),
                       "all_close requires valid tensors");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 && other.dtype_ == DataType::Float32,
                       "all_close currently supports only Float32 tensors");
        LFS_ASSERT_MSG(std::isfinite(rtol) && std::isfinite(atol) &&
                           rtol >= 0.0f && atol >= 0.0f,
                       "all_close tolerances must be finite and non-negative");

        if (shape_ != other.shape_ || dtype_ != other.dtype_) {
            return false;
        }

        if (numel() == 0) {
            return true;
        }

        const float* a_data = nullptr;
        const float* b_data = nullptr;

        Tensor a_temp, b_temp;

        if (device_ == Device::CUDA) {
            a_temp = to(Device::CPU);
            a_data = a_temp.ptr<float>();
        } else {
            a_data = ptr<float>();
        }

        if (other.device() == Device::CUDA) {
            b_temp = other.to(Device::CPU);
            b_data = b_temp.ptr<float>();
        } else {
            b_data = other.ptr<float>();
        }

        if (!a_data || !b_data) {
            return false;
        }

        for (size_t i = 0; i < numel(); ++i) {
            float diff = std::abs(a_data[i] - b_data[i]);
            float tol = atol + rtol * std::abs(b_data[i]);
            if (diff > tol) {
                return false;
            }
        }

        return true;
    }

    // ============= Capacity Management =============

    void Tensor::reserve(size_t new_capacity) {
        LFS_CUDA_BREADCRUMB_STREAM("tensor.reserve", stream());
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid(),
                       "reserve requires a valid tensor");
        LFS_ASSERT_MSG(is_supported_device(device_),
                       "reserve encountered an invalid device");
        LFS_ASSERT_MSG(is_supported_dtype(dtype_),
                       "reserve encountered an invalid dtype");
        // Validate tensor state
        if (!data_owner_) {
            throw TensorError(
                std::format(
                    "reserve({}) requires an owning tensor (tensor='{}', id={}, is_view={}, "
                    "capacity={}, shape={})",
                    new_capacity, name().empty() ? "<unnamed>" : name(), id_, is_view_,
                    state_->capacity, shape_.str()),
                this);
        }
        if (shape_.rank() == 0) {
            throw TensorError("reserve() requires tensor with at least 1 dimension", this);
        }

        // Get current size (use shape_[0], not logical_size_ which may be stale/uninitialized)
        const size_t current_rows = shape_[0];

        // If already at or above requested capacity, just update logical_size_ and return
        if (state_->capacity >= new_capacity) {
            // Make sure logical_size_ is correct (in case reserve was called after resize)
            state_->logical_size = current_rows;
            LOG_DEBUG("Tensor #{}: reserve({}) skipped, already have capacity {} (current size {})",
                      id_, new_capacity, state_->capacity, current_rows);
            return;
        }

        if (has_external_storage()) {
            throw TensorError(std::format(
                                  "reserve({}) would reallocate externally-owned tensor storage '{}'; "
                                  "growth must be handled by the storage owner",
                                  new_capacity,
                                  external_storage_kind()),
                              this);
        }

        LOG_DEBUG("Tensor #{}: reserve({}) starting (current size {}, capacity {})",
                  id_, new_capacity, current_rows, state_->capacity);

        // Calculate sizes
        size_t row_size = 1;
        for (size_t i = 1; i < shape_.rank(); ++i) {
            row_size = checked_product(row_size, shape_[i], "reserve row");
        }
        if (row_size == 0) {
            state_->capacity = new_capacity;
            state_->logical_size = current_rows;
            return;
        }
        const size_t new_total_elements =
            checked_product(new_capacity, row_size, "reserve element count");
        const size_t element_size = dtype_size(dtype_);
        const size_t new_bytes =
            checked_product(new_total_elements, element_size, "reserve byte count");

        LOG_DEBUG("  Allocating: {} rows × {} elements/row × {} bytes/elem = {} MB",
                  new_capacity, row_size, element_size, new_bytes / (1024.0 * 1024.0));

        // Keep the installed owner untouched until allocation, copy, and all
        // replacement metadata have succeeded.
        void* const old_data = data_;

        // Allocate new buffer
        void* new_data = nullptr;
        if (device_ == Device::CUDA) {
            new_data = allocate_cuda_storage(
                new_bytes, stream(), CudaStorageMode::Direct,
                "Tensor reserve CUDA allocation failed", "tensor.reserve");
            LFS_ASSERT_MSG(new_data != nullptr,
                           "reserve CPU allocation failed");
            LOG_DEBUG("  ✓ CUDA allocation succeeded: {} MB at {}", new_bytes / (1024.0 * 1024.0), new_data);
        } else {
            new_data = std::malloc(new_bytes);
            if (new_data == nullptr) {
                throw std::runtime_error(std::format(
                    "reserve CPU allocation failed (data_pointer={}, bytes={}, "
                    "requested_capacity={}, row_size={}, tensor_shape={})",
                    new_data, new_bytes, new_capacity, row_size, shape_.str()));
            }
            LOG_DEBUG("  ✓ CPU allocation succeeded: {} MB at {}", new_bytes / (1024.0 * 1024.0), new_data);
        }

        std::shared_ptr<void> new_owner;
        if (device_ == Device::CUDA) {
            record_storage_allocation(StorageAccountingKind::CudaDirect, new_bytes);
            new_owner = std::shared_ptr<void>(new_data, [bytes = new_bytes](void* ptr) {
                const cudaError_t status = cudaFree(ptr);
                if (status != cudaSuccess) {
                    ensure_cuda_success(
                        status, "cudaFree(tensor reserve storage)", {},
                        LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                }
                Tensor::record_storage_deallocation(StorageAccountingKind::CudaDirect, bytes);
            });
        } else {
            new_owner = std::shared_ptr<void>(new_data, [](void* ptr) {
                std::free(ptr);
            });
        }
        auto new_storage_meta = std::make_shared<StorageMeta>();

        // Copy existing data
        if (old_data && numel() > 0) {
            const size_t copy_bytes =
                checked_product(numel(), element_size, "reserve copy byte count");
            if (device_ == Device::CUDA) {
                const cudaError_t status =
                    cudaMemcpy(new_data, old_data, copy_bytes, cudaMemcpyDeviceToDevice);
                if (status != cudaSuccess) {
                    ensure_cuda_success(
                        status, "cudaMemcpy(tensor reserve)",
                        std::format("bytes={}, source_pointer={}, destination_pointer={}, "
                                    "tensor_shape={}, requested_capacity={}",
                                    copy_bytes, old_data, new_data, shape_.str(), new_capacity),
                        LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                    (void)cudaGetLastError();
                    throw std::runtime_error("Tensor reserve CUDA copy failed");
                }
            } else {
                std::memcpy(new_data, old_data, copy_bytes);
            }
        }

        bump_storage_generation();

        data_ = new_data;
        data_owner_ = std::move(new_owner);
        storage_meta_ = std::move(new_storage_meta);
        state_->capacity = new_capacity;
        state_->logical_size = current_rows;

        LOG_DEBUG("✓ Tensor #{}: reserve({}) SUCCEEDED - capacity now {}, size {} ({:.1f}% utilization)",
                  id_, new_capacity, state_->capacity, current_rows, 100.0 * current_rows / state_->capacity);
    }

    // ============= Error Classes =============

    TensorError::TensorError(const std::string& msg, const Tensor* t)
        : std::runtime_error(msg),
          tensor_info_(t ? t->str() : "") {}

    Tensor Tensor::zeros_direct(TensorShape shape, size_t capacity, Device device, DataType dtype) {
        LFS_ASSERT_MSG(device == Device::CUDA,
                       "zeros_direct currently supports only CUDA tensors");
        LFS_ASSERT_MSG(is_supported_dtype(dtype),
                       "zeros_direct received an invalid dtype");
        LFS_ASSERT_MSG(shape.rank() > 0,
                       "zeros_direct requires at least one dimension");

        const size_t current_size = shape[0];
        LFS_ASSERT_MSG(capacity >= current_size,
                       std::format("zeros_direct capacity {} is smaller than logical row count {}",
                                   capacity, current_size));
        size_t row_size = 1;
        for (size_t i = 1; i < shape.rank(); i++) {
            row_size = checked_product(row_size, shape[i], "zeros_direct row");
        }

        const size_t total_elements =
            checked_product(capacity, row_size, "zeros_direct element count");
        const size_t total_bytes =
            checked_product(total_elements, dtype_size(dtype),
                            "zeros_direct byte count");

        if (total_bytes == 0) {
            Tensor t;
            t.data_ = nullptr;
            static char dummy_owner = 0;
            t.data_owner_ = std::shared_ptr<void>(&dummy_owner, [](void*) {});
            t.shape_ = shape;
            t.strides_ = shape.strides();
            t.storage_offset_ = 0;
            t.device_ = device;
            t.dtype_ = dtype;
            t.state_->capacity = capacity;
            t.state_->logical_size = current_size;
            t.id_ = next_id_++;
            return t;
        }

        void* data_ptr = allocate_cuda_storage(
            total_bytes, nullptr, CudaStorageMode::Direct,
            "cudaMalloc(zeros_direct)", "tensor.zeros_direct");

        // Zero full capacity
        cudaError_t err = cudaMemset(data_ptr, 0, total_bytes);
        if (err != cudaSuccess) {
            const cudaError_t cleanup_status = cudaFree(data_ptr);
            if (cleanup_status != cudaSuccess) {
                ensure_cuda_success(
                    cleanup_status, "cudaFree(failed zeros_direct allocation)", {},
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
            }
            LFS_ENSURE_CUDA_SUCCESS_MSG(
                err, "cudaMemset(zeros_direct)", std::format("bytes={}", total_bytes));
        }

        // Create tensor with custom deleter
        Tensor t;
        t.data_ = data_ptr;
        record_storage_allocation(StorageAccountingKind::CudaDirect, total_bytes);
        t.data_owner_ = std::shared_ptr<void>(data_ptr, [bytes = total_bytes](void* ptr) {
            if (ptr) {
                const cudaError_t status = cudaFree(ptr);
                if (status != cudaSuccess) {
                    ensure_cuda_success(
                        status, "cudaFree(zeros_direct storage)", {},
                        LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                }
                Tensor::record_storage_deallocation(StorageAccountingKind::CudaDirect, bytes);
            }
        });
        t.shape_ = shape;
        t.strides_ = shape.strides();
        t.storage_offset_ = 0;
        t.device_ = device;
        t.dtype_ = dtype;
        t.state_->capacity = capacity;
        t.state_->logical_size = current_size;
        t.id_ = next_id_++;

        return t;
    }

} // namespace lfs::core
