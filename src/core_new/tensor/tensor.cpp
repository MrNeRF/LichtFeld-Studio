/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_broadcast.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <cstring>
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
          is_view_(true), // This is a view
          id_(next_id_++) {

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
          id_(next_id_++) {

        if (profiling_enabled_) {
            LOG_DEBUG("Shallow copy: tensor #{} from #{}: shape={}, device={}, dtype={}, refcount={}",
                      id_, other.id_, shape_.str(), device_name(device_), dtype_name(dtype_),
                      data_owner_ ? data_owner_.use_count() : 0);
        }
    }

    // ============= Copy Assignment - SHALLOW COPY (LibTorch behavior) =============
    Tensor& Tensor::operator=(const Tensor& other) {
        if (this == &other) {
            return *this;
        }

        // Shallow copy - share data via shared_ptr
        data_ = other.data_;
        data_owner_ = other.data_owner_; // shared_ptr handles refcounting automatically
        shape_ = other.shape_;
        strides_ = other.strides_; // Copy stride information
        storage_offset_ = other.storage_offset_;
        is_contiguous_ = other.is_contiguous_;
        device_ = other.device_;
        dtype_ = other.dtype_;
        is_view_ = other.is_view_;
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
          id_(other.id_) {

        if (profiling_enabled_) {
            LOG_DEBUG("Move constructed: tensor #{} (moved-from is now invalid)", id_);
        }
    }

    // ============= Move Assignment =============
    Tensor& Tensor::operator=(Tensor&& other) noexcept {
        if (this != &other) {
            data_ = std::exchange(other.data_, nullptr);
            data_owner_ = std::move(other.data_owner_);
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            storage_offset_ = std::exchange(other.storage_offset_, 0);
            is_contiguous_ = std::exchange(other.is_contiguous_, true);
            device_ = other.device_;
            dtype_ = other.dtype_;
            is_view_ = std::exchange(other.is_view_, false);
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

        // Copy data (accounting for storage offset)
        size_t bytes = this->bytes();
        const char* src = static_cast<const char*>(data_) + storage_offset_ * dtype_size(dtype_);

        if (device_ == Device::CUDA) {
            cudaError_t err = cudaMemcpy(result.data_, src, bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                LOG_ERROR("CUDA memcpy failed in clone(): {}", cudaGetErrorString(err));
                return Tensor();
            }
        } else {
            std::memcpy(result.data_, src, bytes);
        }

        if (profiling_enabled_) {
            LOG_DEBUG("Deep clone: tensor #{} from #{}: copied {} bytes",
                      result.id_, id_, bytes);
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
            // Launch strided copy kernel for CUDA tensors
            const char* src_base = static_cast<const char*>(data_) + storage_offset_ * dtype_size(dtype_);

            // Allocate device memory for shape and strides
            size_t* d_shape;
            size_t* d_strides;
            cudaMalloc(&d_shape, shape_.rank() * sizeof(size_t));
            cudaMalloc(&d_strides, shape_.rank() * sizeof(size_t));

            cudaMemcpy(d_shape, shape_.dims().data(), shape_.rank() * sizeof(size_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_strides, strides_.data(), shape_.rank() * sizeof(size_t), cudaMemcpyHostToDevice);

            tensor_ops::launch_strided_copy(
                src_base,
                result.data_,
                d_shape,
                d_strides,
                shape_.rank(),
                numel(),
                dtype_,
                nullptr);

            cudaDeviceSynchronize();
            cudaFree(d_shape);
            cudaFree(d_strides);
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
    Tensor Tensor::to(Device device) const {
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

                // FAST PATH: Rank-3 uses optimized kernel with immediate parameters (no device memory allocation!)
                // This eliminates 2x cudaMalloc + 2x cudaMemcpy overhead (~0.5-1ms saved)
                // ASYNC: Kernel launches asynchronously - CUDA runtime handles synchronization automatically
                if (shape_.rank() == 3) {
                    LOG_DEBUG("Using optimized rank-3 strided upload (no metadata allocation)");

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
                    // The pinned memory (src) is safe because:
                    // 1. It's managed by shared_ptr in data_owner_
                    // 2. cudaFreeHost() blocks until kernel completes
                    LOG_DEBUG("Optimized rank-3 strided upload launched (async): {} elements", numel());
                    return t;
                }

                // GENERIC PATH: For rank != 3, allocate device memory for metadata
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

        LOG_DEBUG("to(Device): storage_offset_={}, dtype_size={}, src_offset_bytes={}, bytes_to_copy={}",
                  storage_offset_, dtype_size(dtype_), storage_offset_ * dtype_size(dtype_), bytes());

        if (device_ == Device::CPU && device == Device::CUDA) {
            // Use cudaMemcpyAsync with pinned memory for maximum PCIe bandwidth (~7-11 GB/s)
            // CPU tensor now uses pinned memory (allocated via PinnedMemoryAllocator)
            CHECK_CUDA(cudaMemcpyAsync(t.data_, src, bytes(), cudaMemcpyHostToDevice, 0));
            CHECK_CUDA(cudaDeviceSynchronize()); // Ensure transfer completes before returning
        } else if (device_ == Device::CUDA && device == Device::CPU) {
            // Async transfer for GPU→CPU as well (destination is pinned)
            CHECK_CUDA(cudaMemcpyAsync(t.data_, src, bytes(), cudaMemcpyDeviceToHost, 0));
            CHECK_CUDA(cudaDeviceSynchronize()); // Ensure transfer completes before returning
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
        if (numel() == 0)                                                                                                                    \
            return result;                                                                                                                   \
        if (device_ == Device::CUDA) {                                                                                                       \
            tensor_ops::launch_convert_type<FROM_TYPE, TO_TYPE>(                                                                             \
                ptr<FROM_TYPE>(), result.ptr<TO_TYPE>(), numel(), 0);                                                                        \
            CHECK_CUDA(cudaDeviceSynchronize());                                                                                             \
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
                    ptr<unsigned char>(), result.ptr<float>(), numel(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
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
        CONVERT_DTYPE_CUDA(float, int, DataType::Float32, DataType::Int32)
        CONVERT_DTYPE_CUDA(int, float, DataType::Int32, DataType::Float32)

        // UInt8 conversions
        CONVERT_DTYPE_CUDA(float, uint8_t, DataType::Float32, DataType::UInt8)
        CONVERT_DTYPE_CUDA(uint8_t, float, DataType::UInt8, DataType::Float32)
        CONVERT_DTYPE_CUDA(int, uint8_t, DataType::Int32, DataType::UInt8)
        CONVERT_DTYPE_CUDA(uint8_t, int, DataType::UInt8, DataType::Int32)

        // Bool <-> UInt8: Same underlying type (unsigned char), just clone
        if (dtype_ == DataType::Bool && dtype == DataType::UInt8) {
            return clone();
        }
        if (dtype_ == DataType::UInt8 && dtype == DataType::Bool) {
            return clone();
        }

        CONVERT_DTYPE_CUDA(int64_t, uint8_t, DataType::Int64, DataType::UInt8)
        CONVERT_DTYPE_CUDA(uint8_t, int64_t, DataType::UInt8, DataType::Int64)

        // Int64 conversions
        CONVERT_DTYPE_CUDA(int64_t, float, DataType::Int64, DataType::Float32)
        CONVERT_DTYPE_CUDA(float, int64_t, DataType::Float32, DataType::Int64)
        CONVERT_DTYPE_CUDA(int, int64_t, DataType::Int32, DataType::Int64)
        CONVERT_DTYPE_CUDA(int64_t, int, DataType::Int64, DataType::Int32)

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

        // Account for storage offset
        void* dest = static_cast<char*>(data_) + storage_offset_ * dtype_size(dtype_);

        if (device_ == Device::CUDA) {
            std::vector<float> temp(numel(), value);
            CHECK_CUDA(cudaMemcpy(dest, temp.data(), bytes(), cudaMemcpyHostToDevice));
        } else {
            float* data = static_cast<float*>(dest);
            std::fill(data, data + numel(), value);
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

        if (dtype_ != other.dtype_) {
            LOG_ERROR("Dtype mismatch in copy_from");
            return *this;
        }

        if (numel() == 0) {
            return *this;
        }

        if (device_ == Device::CUDA && other.device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(data_, other.data_, bytes(), cudaMemcpyDeviceToDevice));
        } else if (device_ == Device::CUDA && other.device_ == Device::CPU) {
            CHECK_CUDA(cudaMemcpy(data_, other.data_, bytes(), cudaMemcpyHostToDevice));
        } else if (device_ == Device::CPU && other.device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(data_, other.data_, bytes(), cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(data_, other.data_, bytes());
        }

        return *this;
    }

    // ============= Shape Operations =============

    Tensor Tensor::cat(const Tensor& other, int dim) const {
        std::vector<Tensor> tensors;
        tensors.push_back(clone());
        tensors.push_back(other.clone());
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
                tensor_ops::launch_clamp_scalar(ptr<float>(), min_val, max_val, numel(), 0);
            } else if (dtype_ == DataType::Int32) {
                tensor_ops::launch_clamp_scalar_int(ptr<int>(),
                                                    static_cast<int>(min_val),
                                                    static_cast<int>(max_val),
                                                    numel(), 0);
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
            tensor_ops::launch_cumsum(result.raw_ptr(), shape_.dims().data(),
                                      shape_.rank(), dim, dtype_, nullptr);
            cudaDeviceSynchronize();
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

        if (dtype_ != DataType::Float32) {
            LOG_ERROR("item() only supports float32 tensors");
            return 0.0f;
        }

        float value = 0.0f;
        // Account for storage offset (important for sliced tensors)
        const char* data_ptr = static_cast<const char*>(data_) + storage_offset_ * dtype_size(dtype_);

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(&value, data_ptr, sizeof(float), cudaMemcpyDeviceToHost));
        } else {
            value = *static_cast<const float*>(static_cast<const void*>(data_ptr));
        }

        return value;
    }

    std::vector<float> Tensor::debug_values(size_t max_values) const {
        std::vector<float> values;

        if (!is_valid() || dtype_ != DataType::Float32 || numel() == 0) {
            return values;
        }

        size_t n = std::min(max_values, numel());
        values.resize(n);

        if (device_ == Device::CUDA) {
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

        // Account for storage offset
        const char* src = static_cast<const char*>(data_) + storage_offset_ * dtype_size(dtype_);

        if (device_ == Device::CUDA) {
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
            CHECK_CUDA(cudaMemcpy(result.data(), data_, bytes(), cudaMemcpyDeviceToHost));
            LOG_DEBUG("CUDA copy complete");
        } else {
            LOG_DEBUG("Copying from CPU memory, bytes: {}", bytes());
            std::memcpy(result.data(), data_, bytes());
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

        if (dtype_ != DataType::Int32) {
            LOG_ERROR("to_vector_int only supports int32 and bool tensors, got {}",
                      dtype_name(dtype_));
            return {};
        }

        std::vector<int> result(numel());

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(result.data(), data_, bytes(), cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(result.data(), data_, bytes());
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
                CHECK_CUDA(cudaMemcpy(result.data(), data_, bytes(), cudaMemcpyDeviceToHost));
            } else {
                std::memcpy(result.data(), data_, bytes());
            }

            return result;
        }

        // Handle Bool dtype (convert to uint8)
        if (dtype_ == DataType::Bool) {
            std::vector<uint8_t> result(numel());

            if (device_ == Device::CUDA) {
                CHECK_CUDA(cudaMemcpy(result.data(), data_, bytes(), cudaMemcpyDeviceToHost));
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
        std::ofstream file(filename);
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

        auto values = to_vector();
        return std::any_of(values.begin(), values.end(),
                           [](float x) { return std::isnan(x); });
    }

    bool Tensor::has_inf() const {
        if (!is_valid() || numel() == 0) {
            return false;
        }

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

    // ============= Error Classes =============

    TensorError::TensorError(const std::string& msg, const Tensor* t)
        : std::runtime_error(msg),
          tensor_info_(t ? t->str() : "") {}

#undef CHECK_CUDA

} // namespace lfs::core