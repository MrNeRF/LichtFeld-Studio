/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor_trace.hpp"
#include "internal/tensor_broadcast.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <print>

// SIMD intrinsics for CPU optimization
#if defined(__AVX2__)
#include <immintrin.h>
#endif

// OpenMP for multi-threading
#ifdef _OPENMP
#include <omp.h>
#endif

#define CHECK_CUDA(call)                              \
    do {                                              \
        cudaError_t error = call;                     \
        if (error != cudaSuccess) {                   \
            LOG_ERROR("CUDA error at {}:{} - {}: {}", \
                      __FILE__, __LINE__,             \
                      cudaGetErrorName(error),        \
                      cudaGetErrorString(error));     \
        }                                             \
    } while (0)

namespace lfs::core {

    std::atomic<size_t> Tensor::next_id_{1};

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
          shape_(shape),
          strides_(shape.strides()), // Initialize to contiguous strides
          storage_offset_(0),
          is_contiguous_(true),
          device_(device),
          dtype_(dtype),
          is_view_(true),   // This is a view
          stream_(nullptr), // Use default stream (stream 0) for all operations
          id_(next_id_++) {

        compute_alignment(); // Compute alignment flags

        if (profiling_enabled_) {
            LOG_DEBUG("Created tensor #{} (non-owning view): shape={}, device={}, dtype={}",
                      id_, shape_.str(), device_name(device_), dtype_name(dtype_));
        }
    }

    // ============= Copy Constructor - SHALLOW COPY (LibTorch behavior) =============
    Tensor::Tensor(const Tensor& other)
        : data_(other.data_),             // Share the pointer
          data_owner_(other.data_owner_), // Share ownership via shared_ptr!
          shape_(other.shape_),
          strides_(other.strides_), // Copy stride information
          storage_offset_(other.storage_offset_),
          is_contiguous_(other.is_contiguous_),
          device_(other.device_),
          dtype_(other.dtype_),
          is_view_(other.is_view_),
          is_aligned_16_(other.is_aligned_16_), // Copy alignment flags
          is_aligned_128_(other.is_aligned_128_),
          stream_(other.stream_),     // Copy stream assignment
          capacity_(other.capacity_), // Copy capacity management
          logical_size_(other.logical_size_),
          id_(next_id_++) {

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
                const void* src_ptr = static_cast<const char*>(other.data_) +
                                      other.storage_offset_ * dtype_size(other.dtype_);

                if (device_ == Device::CUDA && other.device_ == Device::CUDA) {
                    CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr, bytes(), cudaMemcpyDeviceToDevice, stream_));
                } else if (device_ == Device::CUDA && other.device_ == Device::CPU) {
                    CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr, bytes(), cudaMemcpyHostToDevice, stream_));
                } else if (device_ == Device::CPU && other.device_ == Device::CUDA) {
                    // GPU→CPU requires sync before copy to ensure data is ready
                    if (other.stream_) {
                        cudaStreamSynchronize(other.stream_);
                    } else {
                        cudaDeviceSynchronize();
                    }
                    CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr, bytes(), cudaMemcpyDeviceToHost, stream_));
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

        // Otherwise do shallow copy - share data via shared_ptr (LibTorch behavior)
        data_ = other.data_;
        data_owner_ = other.data_owner_; // shared_ptr handles refcounting automatically
        shape_ = other.shape_;
        strides_ = other.strides_; // Copy stride information
        storage_offset_ = other.storage_offset_;
        is_contiguous_ = other.is_contiguous_;
        device_ = other.device_;
        dtype_ = other.dtype_;
        is_view_ = other.is_view_;
        is_aligned_16_ = other.is_aligned_16_; // Copy alignment flags
        is_aligned_128_ = other.is_aligned_128_;
        stream_ = other.stream_; // Copy stream assignment
        // capacity_ and logical_size_ apply to the specific buffer allocation
        // When we're doing shallow copy (sharing buffer), we share these too
        if (capacity_ != other.capacity_ && capacity_ > 1000000) {
            LOG_WARN("Assignment operator: LOSING CAPACITY! this.capacity={} → other.capacity={}, this.data_={}, other.data_={}",
                     capacity_, other.capacity_, data_, other.data_);
        }
        capacity_ = other.capacity_;
        logical_size_ = other.logical_size_;
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
          shape_(std::move(other.shape_)),
          strides_(std::move(other.strides_)),
          storage_offset_(std::exchange(other.storage_offset_, 0)),
          is_contiguous_(std::exchange(other.is_contiguous_, true)),
          device_(other.device_),
          dtype_(other.dtype_),
          is_view_(std::exchange(other.is_view_, false)),
          is_aligned_16_(std::exchange(other.is_aligned_16_, false)),
          is_aligned_128_(std::exchange(other.is_aligned_128_, false)),
          stream_(std::exchange(other.stream_, nullptr)),
          capacity_(std::exchange(other.capacity_, 0)),
          logical_size_(std::exchange(other.logical_size_, 0)),
          id_(other.id_) {

        if (profiling_enabled_) {
            LOG_DEBUG("Move constructed: tensor #{} (moved-from is now invalid)", id_);
        }
    }

    // ============= Move Assignment =============
    Tensor& Tensor::operator=(Tensor&& other) noexcept {
        if (this != &other) {
            // PyTorch semantics: slice/view assignment does deep copy even for rvalues
            // This handles: t1.slice(0, 0, 5) = t2.slice(0, 5, 10)
            // where the RHS is a temporary view

            if (is_view_ && is_valid() && other.is_valid() &&
                shape_ == other.shape_ && dtype_ == other.dtype_) {

                // Deep copy the data from the temporary view
                if (numel() > 0) {
                    void* dst_ptr = static_cast<char*>(data_) + storage_offset_ * dtype_size(dtype_);
                    const void* src_ptr = static_cast<const char*>(other.data_) +
                                          other.storage_offset_ * dtype_size(other.dtype_);

                    if (device_ == Device::CUDA && other.device_ == Device::CUDA) {
                        CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr, bytes(), cudaMemcpyDeviceToDevice, stream_));
                    } else if (device_ == Device::CUDA && other.device_ == Device::CPU) {
                        CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr, bytes(), cudaMemcpyHostToDevice, stream_));
                    } else if (device_ == Device::CPU && other.device_ == Device::CUDA) {
                        // GPU→CPU requires sync before copy to ensure data is ready
                        if (other.stream_) {
                            cudaStreamSynchronize(other.stream_);
                        } else {
                            cudaDeviceSynchronize();
                        }
                        CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr, bytes(), cudaMemcpyDeviceToHost, stream_));
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

            // Normal move assignment - transfer ownership
            data_ = std::exchange(other.data_, nullptr);
            data_owner_ = std::move(other.data_owner_);
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            storage_offset_ = std::exchange(other.storage_offset_, 0);
            is_contiguous_ = std::exchange(other.is_contiguous_, true);
            device_ = other.device_;
            dtype_ = other.dtype_;
            is_view_ = std::exchange(other.is_view_, false);
            is_aligned_16_ = std::exchange(other.is_aligned_16_, false);
            is_aligned_128_ = std::exchange(other.is_aligned_128_, false);
            stream_ = std::exchange(other.stream_, nullptr);
            capacity_ = std::exchange(other.capacity_, 0);
            logical_size_ = std::exchange(other.logical_size_, 0);
            id_ = other.id_;

            if (profiling_enabled_) {
                LOG_DEBUG("Move assigned: tensor #{} (moved-from is now invalid)", id_);
            }
        }
        return *this;
    }

    // ============= Destructor =============
    Tensor::~Tensor() {
        if (data_owner_ && profiling_enabled_) {
            LOG_DEBUG("Destroying tensor #{}: shape={}, device={}, refcount={}",
                      id_, shape_.str(), device_name(device_), data_owner_.use_count());
        }
        // shared_ptr automatically handles memory cleanup when refcount reaches 0
    }

    // ============= Deep Copy (explicit) =============
    Tensor Tensor::clone() const {
        debug::OpTraceGuard trace("clone", *this);

        if (!is_valid()) {
            LOG_ERROR("Cannot clone invalid tensor");
            return Tensor();
        }

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
            cudaError_t err = cudaMemcpy(result.data_ptr(), data_ptr(), num_bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                LOG_ERROR("CUDA memcpy failed in clone(): {}", cudaGetErrorString(err));
                return Tensor();
            }
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
        if (!is_valid()) {
            LOG_ERROR("Cannot make invalid tensor contiguous");
            return Tensor();
        }

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
                    src_base, result.data_, shape_.dims(), strides_, numel(), dtype_, nullptr);
            } else {
                size_t* d_shape = nullptr;
                size_t* d_strides = nullptr;
                cudaMalloc(&d_shape, rank * sizeof(size_t));
                cudaMalloc(&d_strides, rank * sizeof(size_t));
                cudaMemcpy(d_shape, shape_.dims().data(), rank * sizeof(size_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_strides, strides_.data(), rank * sizeof(size_t), cudaMemcpyHostToDevice);
                tensor_ops::launch_strided_copy(
                    src_base, result.data_, d_shape, d_strides, rank, numel(), dtype_, nullptr);
                cudaFree(d_shape);
                cudaFree(d_strides);
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
// This ensures good work distribution even when dim0 is small
#pragma omp parallel for collapse(2) if (use_parallel) schedule(static)
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
#pragma omp parallel for collapse(2) if (use_parallel) schedule(static)
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
        const char* op_name = (device == Device::CUDA) ? "to_cuda" : "to_cpu";
        debug::OpTraceGuard trace(op_name, *this);

        if (!is_valid()) {
            LOG_ERROR("Cannot transfer invalid tensor to device");
            return Tensor();
        }

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
                return contiguous().to(device);
            } else if (device_ == Device::CPU && device == Device::CUDA) {
                // CPU→GPU: Use fused strided upload kernel!
                LOG_DEBUG("CPU→GPU non-contiguous: using fused strided upload kernel (rank={})", shape_.rank());

                auto t = empty(shape_, Device::CUDA, dtype_);
                if (numel() == 0) {
                    return t;
                }

                // Account for storage offset
                const char* src = static_cast<const char*>(data_) + storage_offset_ * dtype_size(dtype_);

                // FAST PATHS: Use optimized kernels with immediate parameters (no device memory allocation!)
                // This eliminates 2x cudaMalloc + 2x cudaMemcpy overhead (~0.5-1ms saved per upload)
                // ASYNC: Kernel launches asynchronously - CUDA runtime handles synchronization automatically
                if (shape_.rank() <= 3) {
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
                        nullptr // Default stream
                    );

                    // NO SYNC: Let CUDA runtime handle synchronization when data is accessed
                    // The pinned memory (src) is hopefully safe because:
                    // 1. It's managed by shared_ptr in data_owner_
                    // 2. Stream-aware allocator tracks the stream and won't reuse until complete
                    // Update source tensor's stream for deallocator
                    const_cast<Tensor*>(this)->stream_ = nullptr; // Used default stream
                    LOG_DEBUG("Optimized rank-{} strided upload launched (async): {} elements", shape_.rank(), numel());
                    return t;
                }

                // GENERIC PATH: For rank > 3, allocate device memory for metadata
                LOG_DEBUG("Using generic strided upload (requires metadata allocation for rank={})", shape_.rank());

                size_t* d_shape;
                size_t* d_strides;
                CHECK_CUDA(cudaMalloc(&d_shape, shape_.rank() * sizeof(size_t)));
                CHECK_CUDA(cudaMalloc(&d_strides, shape_.rank() * sizeof(size_t)));

                CHECK_CUDA(cudaMemcpy(d_shape, shape_.dims().data(),
                                      shape_.rank() * sizeof(size_t),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_strides, strides_.data(),
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
                    nullptr // Default stream
                );

                // Free metadata immediately (kernel has already captured the data)
                // NO SYNC: Kernel launches asynchronously for better PCIe overlap
                CHECK_CUDA(cudaFree(d_shape));
                CHECK_CUDA(cudaFree(d_strides));

                // CRITICAL: Update source tensor's stream for deallocator
                const_cast<Tensor*>(this)->stream_ = nullptr; // Used default stream

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

            // Use provided stream, or tensor's stream, or default stream
            cudaStream_t transfer_stream = stream ? stream : (stream_ ? stream_ : 0);
            CHECK_CUDA(cudaMemcpyAsync(t.data_, src, bytes(), cudaMemcpyHostToDevice, transfer_stream));

            // CRITICAL: Update source tensor's stream so deallocator knows which stream used this memory
            // This is needed for the stream-aware pinned memory allocator
            if (stream) {
                const_cast<Tensor*>(this)->stream_ = stream;
            }

            // If no stream provided, sync - but use stream sync if tensor has a stream
            if (!stream) {
                if (stream_) {
                    cudaStreamSynchronize(stream_);
                } else {
                    CHECK_CUDA(cudaDeviceSynchronize());
                }
            }
        } else if (device_ == Device::CUDA && device == Device::CPU) {
            // API BOUNDARY: Sync before GPU→CPU transfer - use stream sync when possible
            if (stream) {
                cudaStreamSynchronize(stream);
            } else if (stream_) {
                cudaStreamSynchronize(stream_);
            } else {
                cudaDeviceSynchronize();
            }
            // Async transfer for GPU→CPU as well (destination is pinned)
            cudaStream_t transfer_stream = stream ? stream : (stream_ ? stream_ : 0);
            CHECK_CUDA(cudaMemcpyAsync(t.data_, src, bytes(), cudaMemcpyDeviceToHost, transfer_stream));

            // Sync after transfer - but use stream sync if available
            if (!stream) {
                if (stream_) {
                    cudaStreamSynchronize(stream_);
                } else {
                    CHECK_CUDA(cudaDeviceSynchronize());
                }
            }
        }

        return t;
    }

    // ============= Type Conversion =============
    Tensor Tensor::to(DataType dtype) const {
        if (!is_valid()) {
            LOG_ERROR("Cannot convert invalid tensor to different dtype");
            return Tensor();
        }

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
        result.set_stream(stream_);                                                                                                          \
        if (numel() == 0)                                                                                                                    \
            return result;                                                                                                                   \
        if (device_ == Device::CUDA) {                                                                                                       \
            tensor_ops::launch_convert_type<FROM_TYPE, TO_TYPE>(                                                                             \
                ptr<FROM_TYPE>(), result.ptr<TO_TYPE>(), numel(), stream_);                                                                  \
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
            result.set_stream(stream_);
            if (numel() == 0)
                return result;

            if (device_ == Device::CUDA) {
                // Use generic conversion (unsigned char -> float)
                tensor_ops::launch_convert_type<unsigned char, float>(
                    ptr<unsigned char>(), result.ptr<float>(), numel(), stream_);
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
                CHECK_CUDA(cudaMemcpy(temp.data(), ptr<float>(), bytes(), cudaMemcpyDeviceToHost));

                unsigned char* dst_cpu = result_cpu.ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst_cpu[i] = (temp[i] != 0.0f) ? 1 : 0;
                }

                CHECK_CUDA(cudaMemcpy(result.ptr<unsigned char>(), dst_cpu, numel(), cudaMemcpyHostToDevice));
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
        // DEBUG: Add logging for Float32->Int32 conversion
        if (dtype_ == DataType::Float32 && dtype == DataType::Int32) {
            auto result = empty(shape_, device_, DataType::Int32);
            result.set_stream(stream_);
            if (numel() == 0)
                return result;

            // Read source value before conversion (for debugging)
            if (numel() == 1 && device_ == Device::CUDA) {
                float src_val;
                cudaMemcpy(&src_val, ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);
                printf("[to Float32->Int32] Source value: %f\n", src_val);
            }

            if (device_ == Device::CUDA) {
                tensor_ops::launch_convert_type<float, int>(
                    ptr<float>(), result.ptr<int>(), numel(), stream_);
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

        // Bool <-> UInt8: Same underlying storage, just reinterpret dtype
        if ((dtype_ == DataType::Bool && dtype == DataType::UInt8) ||
            (dtype_ == DataType::UInt8 && dtype == DataType::Bool)) {
            auto result = empty(shape_, device_, dtype);
            if (numel() > 0) {
                if (device_ == Device::CUDA) {
                    CHECK_CUDA(cudaMemcpy(const_cast<void*>(result.data_ptr()), data_ptr(), bytes(), cudaMemcpyDeviceToDevice));
                } else {
                    std::memcpy(const_cast<void*>(result.data_ptr()), data_ptr(), bytes());
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
                CHECK_CUDA(cudaMemcpy(temp.data(), ptr<int>(), bytes(), cudaMemcpyDeviceToHost));

                unsigned char* dst_cpu = result_cpu.ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst_cpu[i] = (temp[i] != 0) ? 1 : 0;
                }

                CHECK_CUDA(cudaMemcpy(result.ptr<unsigned char>(), result_cpu.ptr<unsigned char>(),
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
            result.set_stream(stream_);
            if (numel() == 0)
                return result;

            if (device_ == Device::CUDA) {
                // Use generic conversion (unsigned char -> int)
                tensor_ops::launch_convert_type<unsigned char, int>(
                    ptr<unsigned char>(), result.ptr<int>(), numel(), stream_);
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
            result.set_stream(stream_);
            if (numel() == 0)
                return result;

            if (device_ == Device::CUDA) {
                // Use generic conversion (unsigned char -> int64_t)
                tensor_ops::launch_convert_type<unsigned char, int64_t>(
                    ptr<unsigned char>(), result.ptr<int64_t>(), numel(), stream_);
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
                CHECK_CUDA(cudaMemcpy(temp.data(), ptr<int64_t>(), bytes(), cudaMemcpyDeviceToHost));

                unsigned char* dst_cpu = result_cpu.ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst_cpu[i] = (temp[i] != 0) ? 1 : 0;
                }

                CHECK_CUDA(cudaMemcpy(result.ptr<unsigned char>(), result_cpu.ptr<unsigned char>(),
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
            result.set_stream(stream_);
            if (numel() == 0)
                return result;

            if (device_ == Device::CUDA) {
                // Use generic conversion (unsigned char -> __half)
                tensor_ops::launch_convert_type<unsigned char, __half>(
                    ptr<unsigned char>(), result.ptr<__half>(), numel(), stream_);
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
                CHECK_CUDA(cudaMemcpy(temp.data(), ptr<__half>(), bytes(), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaDeviceSynchronize()); // Ensure copy completes

                unsigned char* dst_cpu = result_cpu.ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst_cpu[i] = (__half2float(temp[i]) != 0.0f) ? 1 : 0;
                }

                CHECK_CUDA(cudaMemcpy(result.ptr<unsigned char>(), result_cpu.ptr<unsigned char>(),
                                      numel() * sizeof(unsigned char), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaDeviceSynchronize()); // Ensure copy completes
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
            result.set_stream(stream_);
            if (numel() == 0)
                return result;

            if (device_ == Device::CUDA) {
                tensor_ops::launch_convert_type<int64_t, int>(
                    ptr<int64_t>(), result.ptr<int>(), numel(), stream_);
                // CRITICAL: Sync to ensure conversion completes before item() reads
                cudaStreamSynchronize(stream_);
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

        LOG_ERROR("Type conversion from {} to {} not implemented",
                  dtype_name(dtype_), dtype_name(dtype));
        return Tensor();
    }

    // ============= In-place Operations =============

    Tensor& Tensor::zero_() {
        if (!is_valid() || numel() == 0) {
            return *this;
        }

        // Account for storage offset
        char* dest = static_cast<char*>(data_) + storage_offset_ * dtype_size(dtype_);

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemset(dest, 0, bytes()));
        } else {
            std::memset(dest, 0, bytes());
        }

        return *this;
    }

    Tensor& Tensor::fill_(float value) {
        if (!is_valid() || numel() == 0) {
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
                // Use tensor's stream if available for async operation
                if (dtype_ == DataType::Float32) {
                    tensor_ops::launch_fill_strided<float>(
                        static_cast<float*>(data_), value, shape_.dims(), strides_, storage_offset_, n, stream_);
                } else if (dtype_ == DataType::Int32) {
                    int int_val = static_cast<int>(value);
                    tensor_ops::launch_fill_strided<int>(
                        static_cast<int*>(data_), int_val, shape_.dims(), strides_, storage_offset_, n, stream_);
                } else if (dtype_ == DataType::Bool) {
                    unsigned char bool_val = (value != 0.0f) ? 1 : 0;
                    tensor_ops::launch_fill_strided<unsigned char>(
                        static_cast<unsigned char*>(data_), bool_val, shape_.dims(), strides_, storage_offset_, n, stream_);
                }
                // Only sync if no stream (maintains original behavior for non-stream tensors)
                if (!stream_) {
                    CHECK_CUDA(cudaDeviceSynchronize());
                }
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

        // CUDA: use GPU kernel directly (avoids CPU temp array + memcpy)
        if (device_ == Device::CUDA) {
            const size_t n = numel();
            const std::vector<size_t> shape_1d = {n};
            const std::vector<size_t> strides_1d = {1};

            if (dtype_ == DataType::Float32) {
                tensor_ops::launch_fill_strided<float>(
                    static_cast<float*>(dest), value, shape_1d, strides_1d, 0, n, stream_);
            } else if (dtype_ == DataType::Int32) {
                tensor_ops::launch_fill_strided<int>(
                    static_cast<int*>(dest), static_cast<int>(value), shape_1d, strides_1d, 0, n, stream_);
            } else if (dtype_ == DataType::Bool) {
                const auto bool_val = static_cast<unsigned char>(value != 0.0f);
                tensor_ops::launch_fill_strided<unsigned char>(
                    static_cast<unsigned char*>(dest), bool_val, shape_1d, strides_1d, 0, n, stream_);
            }
            if (!stream_) {
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            return *this;
        }

        // CPU: use std::fill
        const size_t n = numel();
        if (dtype_ == DataType::Bool) {
            const auto bool_val = static_cast<unsigned char>(value != 0.0f);
            std::fill(static_cast<unsigned char*>(dest), static_cast<unsigned char*>(dest) + n, bool_val);
        } else if (dtype_ == DataType::Int32) {
            const auto int_val = static_cast<int>(value);
            std::fill(static_cast<int*>(dest), static_cast<int*>(dest) + n, int_val);
        } else {
            std::fill(static_cast<float*>(dest), static_cast<float*>(dest) + n, value);
        }

        return *this;
    }

    Tensor& Tensor::fill_(float value, cudaStream_t stream) {
        if (!is_valid() || numel() == 0) {
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
            CHECK_CUDA(cudaMemsetAsync(dest, 0, bytes(), stream));
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
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensors for copy_from");
            return *this;
        }
        if (shape_ != other.shape_) {
            LOG_ERROR("Shape mismatch in copy_from: {} vs {}", shape_.str(), other.shape_.str());
            return *this;
        }

        // Type conversion path
        if (dtype_ != other.dtype_) {
            // Fused int32→float32 strided scatter
            if (!is_contiguous() && other.is_contiguous() &&
                device_ == Device::CUDA && other.device_ == Device::CUDA &&
                dtype_ == DataType::Float32 && other.dtype_ == DataType::Int32 && ndim() == 2) {
                const size_t shape_arr[2] = {static_cast<size_t>(size(0)), static_cast<size_t>(size(1))};
                const size_t strides_arr[2] = {strides_[0], strides_[1]};
                tensor_ops::launch_strided_scatter_int32_to_float32(
                    other.data_ptr(), data_ptr(), shape_arr, strides_arr, 2, numel(), nullptr);
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
                CHECK_CUDA(cudaMemcpy(data_ptr(), other.data_ptr(), bytes(), cudaMemcpyDeviceToDevice));
            } else if (device_ == Device::CUDA && other.device_ == Device::CPU) {
                CHECK_CUDA(cudaMemcpy(data_ptr(), other.data_ptr(), bytes(), cudaMemcpyHostToDevice));
            } else if (device_ == Device::CPU && other.device_ == Device::CUDA) {
                CHECK_CUDA(cudaMemcpy(data_ptr(), other.data_ptr(), bytes(), cudaMemcpyDeviceToHost));
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
                    other.data_ptr(), data_ptr(), shape_vec, strides_, numel(), dtype_, nullptr);
            } else {
                size_t* d_shape = nullptr;
                size_t* d_strides = nullptr;
                CHECK_CUDA(cudaMalloc(&d_shape, rank * sizeof(size_t)));
                CHECK_CUDA(cudaMalloc(&d_strides, rank * sizeof(size_t)));
                CHECK_CUDA(cudaMemcpy(d_shape, shape_vec.data(), rank * sizeof(size_t), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_strides, strides_.data(), rank * sizeof(size_t), cudaMemcpyHostToDevice));
                tensor_ops::launch_strided_scatter(
                    other.data_ptr(), data_ptr(), d_shape, d_strides, rank, numel(), dtype_, nullptr);
                CHECK_CUDA(cudaFree(d_shape));
                CHECK_CUDA(cudaFree(d_strides));
            }
            return *this;
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
        if (!is_valid()) {
            LOG_ERROR("Cannot normalize invalid tensor");
            return Tensor();
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
        if (!is_valid()) {
            LOG_ERROR("Cannot compute logit of invalid tensor");
            return Tensor();
        }

        auto x_clamped = clamp(eps, 1.0f - eps);
        auto one_minus_x = full(shape_, 1.0f, device_, dtype_).sub(x_clamped);
        return x_clamped.div(one_minus_x).log();
    }

    // ============= Bitwise Operations =============

    Tensor Tensor::operator~() const {
        if (!is_valid()) {
            LOG_ERROR("Bitwise NOT on invalid tensor");
            return Tensor();
        }

        if (dtype_ != DataType::Bool) {
            LOG_ERROR("Bitwise NOT only works on boolean tensors");
            return Tensor();
        }

        // Use the new functor-based logical_not() method
        return logical_not();
    }

    Tensor Tensor::operator|(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Bitwise OR on invalid tensor");
            return Tensor();
        }

        if (dtype_ != DataType::Bool || other.dtype() != DataType::Bool) {
            LOG_ERROR("Bitwise OR only works on boolean tensors");
            return Tensor();
        }

        return logical_or(other);
    }

    // ============= Clamp Operations =============

    Tensor& Tensor::clamp_(float min_val, float max_val) {
        if (!is_valid() || numel() == 0) {
            return *this;
        }

        if (device_ == Device::CUDA) {
            if (dtype_ == DataType::Float32) {
                tensor_ops::launch_clamp_scalar(ptr<float>(), min_val, max_val, numel(), stream_);
            } else if (dtype_ == DataType::Int32) {
                tensor_ops::launch_clamp_scalar_int(ptr<int>(),
                                                    static_cast<int>(min_val),
                                                    static_cast<int>(max_val),
                                                    numel(), stream_);
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
        if (!is_valid()) {
            LOG_ERROR("cumsum on invalid tensor");
            return Tensor();
        }

        dim = resolve_dim(dim);
        if (dim < 0 || dim >= static_cast<int>(shape_.rank())) {
            LOG_ERROR("Invalid dimension for cumsum: {}", dim);
            return Tensor();
        }

        auto result = clone();

        if (device_ == Device::CUDA) {
            tensor_ops::launch_cumsum(result.data_ptr(), shape_.dims().data(),
                                      shape_.rank(), dim, dtype_, nullptr);
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
        }
        oss << ")";
        return oss.str();
    }

    // ============= Debug Functions =============

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

        if (!tensor.is_valid() || tensor.shape().rank() == 0) {
            return batches;
        }

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
        if (!is_valid() || numel() != 1) {
            LOG_ERROR("item() requires a valid single-element tensor");
            return 0.0f;
        }

        // Account for storage offset (important for sliced tensors)
        const char* data_ptr = static_cast<const char*>(data_) + storage_offset_ * dtype_size(dtype_);
        float value = 0.0f;

        // Sync before reading from GPU - use stream sync if available (much faster!)
        if (device_ == Device::CUDA) {
            if (stream_) {
                cudaStreamSynchronize(stream_);
            } else {
                cudaDeviceSynchronize();
            }
        }

        // Handle different dtypes
        switch (dtype_) {
        case DataType::Float32: {
            float temp;
            if (device_ == Device::CUDA) {
                CHECK_CUDA(cudaMemcpy(&temp, data_ptr, sizeof(float), cudaMemcpyDeviceToHost));
            } else {
                temp = *static_cast<const float*>(static_cast<const void*>(data_ptr));
            }
            value = temp;
            break;
        }
        case DataType::Int64: {
            int64_t temp;
            if (device_ == Device::CUDA) {
                CHECK_CUDA(cudaMemcpy(&temp, data_ptr, sizeof(int64_t), cudaMemcpyDeviceToHost));
            } else {
                temp = *static_cast<const int64_t*>(static_cast<const void*>(data_ptr));
            }
            value = static_cast<float>(temp);
            break;
        }
        case DataType::Int32: {
            int32_t temp;
            if (device_ == Device::CUDA) {
                CHECK_CUDA(cudaMemcpy(&temp, data_ptr, sizeof(int32_t), cudaMemcpyDeviceToHost));
            } else {
                temp = *static_cast<const int32_t*>(static_cast<const void*>(data_ptr));
            }
            value = static_cast<float>(temp);
            break;
        }
        case DataType::Bool: {
            unsigned char temp;
            if (device_ == Device::CUDA) {
                CHECK_CUDA(cudaMemcpy(&temp, data_ptr, sizeof(unsigned char), cudaMemcpyDeviceToHost));
            } else {
                temp = *static_cast<const unsigned char*>(static_cast<const void*>(data_ptr));
            }
            value = static_cast<float>(temp);
            break;
        }
        case DataType::UInt8: {
            uint8_t temp;
            if (device_ == Device::CUDA) {
                CHECK_CUDA(cudaMemcpy(&temp, data_ptr, sizeof(uint8_t), cudaMemcpyDeviceToHost));
            } else {
                temp = *static_cast<const uint8_t*>(static_cast<const void*>(data_ptr));
            }
            value = static_cast<float>(temp);
            break;
        }
        default:
            LOG_ERROR("item() does not support dtype {}", static_cast<int>(dtype_));
            return 0.0f;
        }

        return value;
    }

    std::vector<float> Tensor::debug_values(size_t max_values) const {
        std::vector<float> values;

        if (!is_valid() || dtype_ != DataType::Float32 || numel() == 0) {
            return values;
        }

        if (!is_contiguous_)
            return contiguous().debug_values(max_values);

        size_t n = std::min(max_values, numel());
        values.resize(n);

        if (device_ == Device::CUDA) {
            // Sync before reading from GPU - use stream sync if available
            if (stream_) {
                cudaStreamSynchronize(stream_);
            } else {
                cudaDeviceSynchronize();
            }
            CHECK_CUDA(cudaMemcpy(values.data(), data_, n * sizeof(float),
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
        if (!is_valid()) {
            LOG_ERROR("to_vector on invalid tensor");
            return {};
        }

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

        if (dtype_ != DataType::Float32) {
            LOG_ERROR("to_vector only supports float32, int32, int64, uint8 and bool tensors, got {}",
                      dtype_name(dtype_));
            return {};
        }

        std::vector<float> result(numel());

        // Use data_ptr() which accounts for storage_offset
        const void* src = data_ptr();

        if (device_ == Device::CUDA) {
            // Sync before reading from GPU - use stream sync if available
            if (stream_) {
                cudaStreamSynchronize(stream_);
            } else {
                cudaDeviceSynchronize();
            }
            CHECK_CUDA(cudaMemcpy(result.data(), src, bytes(), cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(result.data(), src, bytes());
        }

        return result;
    }

    std::vector<int64_t> Tensor::to_vector_int64() const {
        LOG_DEBUG("to_vector_int64() called");
        LOG_DEBUG("  dtype: {}", dtype_name(dtype_));
        LOG_DEBUG("  device: {}", device_name(device_));
        LOG_DEBUG("  numel: {}", numel());
        LOG_DEBUG("  is_valid: {}", is_valid());

        if (!is_valid()) {
            LOG_ERROR("to_vector_int64() on invalid tensor");
            return {};
        }

        if (dtype_ != DataType::Int64) {
            LOG_ERROR("to_vector_int64() requires Int64 tensor, got {}", dtype_name(dtype_));
            return {};
        }

        if (numel() == 0) {
            LOG_DEBUG("Empty tensor, returning empty vector");
            return {};
        }

        LOG_DEBUG("Creating result vector of size {}", numel());
        std::vector<int64_t> result(numel());

        if (device_ == Device::CUDA) {
            LOG_DEBUG("Copying from CUDA to CPU, bytes: {}", bytes());
            CHECK_CUDA(cudaMemcpy(result.data(), data_ptr(), bytes(), cudaMemcpyDeviceToHost));
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
        if (!is_valid()) {
            LOG_ERROR("to_vector_int on invalid tensor");
            return {};
        }

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

        if (dtype_ != DataType::Int32) {
            LOG_ERROR("to_vector_int only supports int32, int64, and bool tensors, got {}",
                      dtype_name(dtype_));
            return {};
        }

        std::vector<int> result(numel());

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(result.data(), data_ptr(), bytes(), cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(result.data(), data_ptr(), bytes());
        }

        return result;
    }

    std::vector<bool> Tensor::to_vector_bool() const {
        if (!is_valid()) {
            LOG_ERROR("to_vector_bool only supports valid bool tensors");
            return {};
        }

        // Support both Bool and UInt8 dtypes (UInt8 can be used as byte array)
        if (dtype_ != DataType::Bool && dtype_ != DataType::UInt8) {
            LOG_ERROR("to_vector_bool only supports bool and uint8 tensors, got {}",
                      dtype_name(dtype_));
            return {};
        }

        if (numel() == 0) {
            return {};
        }

        std::vector<bool> result(numel());

        if (device_ == Device::CUDA) {
            std::vector<unsigned char> temp(numel());
            CHECK_CUDA(cudaMemcpy(temp.data(), data_, bytes(), cudaMemcpyDeviceToHost));
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
        if (!is_valid()) {
            LOG_ERROR("to_vector_uint8 on invalid tensor");
            return {};
        }

        if (numel() == 0) {
            return {};
        }

        // Handle UInt8 dtype directly
        if (dtype_ == DataType::UInt8) {
            std::vector<uint8_t> result(numel());

            if (device_ == Device::CUDA) {
                CHECK_CUDA(cudaMemcpy(result.data(), data_ptr(), bytes(), cudaMemcpyDeviceToHost));
            } else {
                std::memcpy(result.data(), data_ptr(), bytes());
            }

            return result;
        }

        // Handle Bool dtype (convert to uint8)
        if (dtype_ == DataType::Bool) {
            std::vector<uint8_t> result(numel());

            if (device_ == Device::CUDA) {
                CHECK_CUDA(cudaMemcpy(result.data(), data_ptr(), bytes(), cudaMemcpyDeviceToHost));
            } else {
                const unsigned char* src = ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    result[i] = src[i];
                }
            }

            return result;
        }

        // For other types, log error
        LOG_ERROR("to_vector_uint8 only supports uint8 and bool tensors directly, got {}. "
                  "Convert to UInt8 first using .to(DataType::UInt8)",
                  dtype_name(dtype_));
        return {};
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

    Tensor& Tensor::assert_shape(TensorShape expected, const std::string& msg) {
        if (!is_valid()) {
            std::string error_msg = "Cannot assert shape on invalid tensor";
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg, this);
        }

        if (shape_ != expected) {
            std::string error_msg = msg.empty() ? "Shape assertion failed: expected " + expected.str() + " but got " + shape_.str() : msg;
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg, this);
        }
        return *this;
    }

    Tensor& Tensor::assert_device(Device expected) {
        if (!is_valid()) {
            std::string error_msg = "Cannot assert device on invalid tensor";
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg, this);
        }

        if (device_ != expected) {
            std::string error_msg = "Device assertion failed: expected " +
                                    std::string(device_name(expected)) + " but got " +
                                    std::string(device_name(device_));
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg, this);
        }
        return *this;
    }

    Tensor& Tensor::assert_dtype(DataType expected) {
        if (!is_valid()) {
            std::string error_msg = "Cannot assert dtype on invalid tensor";
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg, this);
        }

        if (dtype_ != expected) {
            std::string error_msg = "DataType assertion failed: expected " +
                                    std::string(dtype_name(expected)) + " but got " +
                                    std::string(dtype_name(dtype_));
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg, this);
        }
        return *this;
    }

    Tensor& Tensor::assert_finite() {
        if (!is_valid()) {
            std::string error_msg = "Cannot assert finite on invalid tensor";
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg, this);
        }

        if (has_nan() || has_inf()) {
            std::string error_msg = "Tensor contains NaN or Inf values";
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg, this);
        }
        return *this;
    }

    // ============= Comparison Utilities =============

    bool Tensor::has_nan() const {
        if (!is_valid() || numel() == 0) {
            return false;
        }

        // Use fast GPU check for CUDA tensors (only transfers 1 int back)
        if (device_ == Device::CUDA && dtype_ == DataType::Float32) {
            return tensor_ops::has_nan_or_inf_gpu(ptr<float>(), numel(), nullptr);
        }

        // CPU fallback
        auto values = to_vector();
        return std::any_of(values.begin(), values.end(),
                           [](float x) { return std::isnan(x); });
    }

    bool Tensor::has_inf() const {
        if (!is_valid() || numel() == 0) {
            return false;
        }

        // Use fast GPU check for CUDA tensors (only transfers 1 int back)
        // Note: has_nan_or_inf_gpu checks both NaN and Inf
        if (device_ == Device::CUDA && dtype_ == DataType::Float32) {
            return tensor_ops::has_nan_or_inf_gpu(ptr<float>(), numel(), nullptr);
        }

        // CPU fallback
        auto values = to_vector();
        return std::any_of(values.begin(), values.end(),
                           [](float x) { return std::isinf(x); });
    }

    bool Tensor::all_close(const Tensor& other, float rtol, float atol) const {
        if (!is_valid() || !other.is_valid()) {
            return false;
        }

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
        // Validate tensor state
        if (!data_owner_) {
            LOG_ERROR("reserve({}) failed on tensor '{}' (id={}): null data_owner_, is_view_={}, capacity_={}, shape={}",
                      new_capacity, name_.empty() ? "<unnamed>" : name_, id_, is_view_, capacity_, shape_.str());
            throw TensorError("reserve() requires an owning tensor (not a view)", this);
        }
        if (shape_.rank() == 0) {
            throw TensorError("reserve() requires tensor with at least 1 dimension", this);
        }

        // Get current size (use shape_[0], not logical_size_ which may be stale/uninitialized)
        const size_t current_rows = shape_[0];

        // If already at or above requested capacity, just update logical_size_ and return
        if (capacity_ >= new_capacity) {
            // Make sure logical_size_ is correct (in case reserve was called after resize)
            logical_size_ = current_rows;
            LOG_DEBUG("Tensor #{}: reserve({}) skipped, already have capacity {} (current size {})",
                      id_, new_capacity, capacity_, current_rows);
            return;
        }

        LOG_DEBUG("Tensor #{}: reserve({}) starting (current size {}, capacity {})",
                  id_, new_capacity, current_rows, capacity_);

        // Calculate sizes
        const size_t row_size = numel() / current_rows; // elements per "row"
        const size_t new_total_elements = new_capacity * row_size;
        const size_t element_size = dtype_size(dtype_);
        const size_t new_bytes = new_total_elements * element_size;

        LOG_DEBUG("  Allocating: {} rows × {} elements/row × {} bytes/elem = {} MB",
                  new_capacity, row_size, element_size, new_bytes / (1024.0 * 1024.0));

        // First, explicitly release the old buffer to avoid double allocation
        // This ensures the old buffer is freed BEFORE we allocate the new one
        void* old_data = data_;
        std::shared_ptr<void> old_owner = data_owner_; // Keep reference temporarily

        // Allocate new buffer
        void* new_data = nullptr;
        try {
            if (device_ == Device::CUDA) {
                CHECK_CUDA(cudaMalloc(&new_data, new_bytes));
                LOG_DEBUG("  ✓ CUDA allocation succeeded: {} MB at {}", new_bytes / (1024.0 * 1024.0), new_data);
            } else {
                new_data = std::malloc(new_bytes);
                if (!new_data) {
                    throw TensorError("Failed to allocate CPU memory for reserve()", this);
                }
                LOG_DEBUG("  ✓ CPU allocation succeeded: {} MB at {}", new_bytes / (1024.0 * 1024.0), new_data);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("  ✗ Allocation failed: {}", e.what());
            throw;
        }

        // Copy existing data
        if (old_data && numel() > 0) {
            const size_t copy_bytes = numel() * element_size;
            if (device_ == Device::CUDA) {
                CHECK_CUDA(cudaMemcpy(new_data, old_data, copy_bytes, cudaMemcpyDeviceToDevice));
            } else {
                std::memcpy(new_data, old_data, copy_bytes);
            }
        }

        // Update tensor state with new buffer
        data_ = new_data;
        data_owner_ = std::shared_ptr<void>(new_data, [device = device_](void* ptr) {
            if (device == Device::CUDA) {
                cudaFree(ptr);
            } else {
                std::free(ptr);
            }
        });
        capacity_ = new_capacity;
        logical_size_ = current_rows; // Set to actual current size (from shape_[0])

        // Explicitly release old buffer AFTER copy is complete
        // This ensures we don't have both buffers alive at the same time
        old_owner.reset(); // Decrement ref count, potentially freeing old buffer immediately

        LOG_DEBUG("✓ Tensor #{}: reserve({}) SUCCEEDED - capacity now {}, size {} ({:.1f}% utilization)",
                  id_, new_capacity, capacity_, current_rows, 100.0 * current_rows / capacity_);
    }

    // ============= Error Classes =============

    TensorError::TensorError(const std::string& msg, const Tensor* t)
        : std::runtime_error(msg),
          tensor_info_(t ? t->str() : "") {}

    Tensor Tensor::zeros_direct(TensorShape shape, size_t capacity, Device device) {
        if (device != Device::CUDA) {
            throw TensorError("zeros_direct only supports CUDA device");
        }

        // Rank-0 tensor (empty)
        if (shape.rank() == 0) {
            static char DUMMY_OWNER = 0;
            Tensor t;
            t.data_ = nullptr;
            t.data_owner_ = std::shared_ptr<void>(&DUMMY_OWNER, [](void*) {});
            t.shape_ = shape;
            t.strides_ = {};
            t.storage_offset_ = 0;
            t.device_ = device;
            t.dtype_ = DataType::Float32;
            t.capacity_ = 0;
            t.logical_size_ = 0;
            t.id_ = next_id_++;
            return t;
        }

        const size_t current_size = shape[0];
        size_t row_size = 1;
        for (size_t i = 1; i < shape.rank(); i++) {
            row_size *= shape[i];
        }

        const size_t total_elements = capacity * row_size;
        const size_t total_bytes = total_elements * sizeof(float);

        // Handle zero-size tensors (shN with sh-degree 0 has shape [N, 0, 3])
        if (total_bytes == 0) {
            Tensor t;
            t.data_ = nullptr;
            // Dummy owner allows reserve() to recognize this as owning tensor
            static char dummy_owner = 0;
            t.data_owner_ = std::shared_ptr<void>(&dummy_owner, [](void*) {});
            t.shape_ = shape;
            t.strides_ = shape.strides();
            t.storage_offset_ = 0;
            t.device_ = device;
            t.dtype_ = DataType::Float32;
            t.capacity_ = capacity;
            t.logical_size_ = current_size;
            t.id_ = next_id_++;
            return t;
        }

        // Direct cudaMalloc bypassing pool
        void* data_ptr = nullptr;
        cudaError_t err = cudaMalloc(&data_ptr, total_bytes);
        if (err != cudaSuccess) {
            throw TensorError("cudaMalloc failed in zeros_direct: " + std::string(cudaGetErrorString(err)));
        }

        // Zero full capacity
        err = cudaMemset(data_ptr, 0, total_bytes);
        if (err != cudaSuccess) {
            cudaFree(data_ptr);
            throw TensorError("cudaMemset failed in zeros_direct: " + std::string(cudaGetErrorString(err)));
        }

        // Create tensor with custom deleter
        Tensor t;
        t.data_ = data_ptr;
        t.data_owner_ = std::shared_ptr<void>(data_ptr, [](void* ptr) {
            if (ptr)
                cudaFree(ptr);
        });
        t.shape_ = shape;
        t.strides_ = shape.strides();
        t.storage_offset_ = 0;
        t.device_ = device;
        t.dtype_ = DataType::Float32;
        t.capacity_ = capacity;
        t.logical_size_ = current_size;
        t.id_ = next_id_++;

        return t;
    }

} // namespace lfs::core
