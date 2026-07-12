/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/assert.hpp"
#include "core/cuda_error.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <format>
#include <limits>
#include <string>
#include <string_view>

// cuda.h defines CUDA_VERSION which GLM needs to detect CUDA version properly
// Must be included before any GLM headers
#include <cuda.h>
#include <cuda_runtime.h>

#include <glm/gtc/type_ptr.hpp>

//
// Camera Types (at global scope for compatibility with Cameras.cuh)
//
#ifndef _GSPLAT_CAMERA_MODEL_TYPE_DEFINED
#define _GSPLAT_CAMERA_MODEL_TYPE_DEFINED
enum CameraModelType {
    PINHOLE = 0,
    ORTHO = 1,
    FISHEYE = 2,
    EQUIRECTANGULAR = 3,
    THIN_PRISM_FISHEYE = 4
};
#endif

#ifndef _GSPLAT_SHUTTER_TYPE_DEFINED
#define _GSPLAT_SHUTTER_TYPE_DEFINED
enum class ShutterType {
    ROLLING_TOP_TO_BOTTOM,
    ROLLING_LEFT_TO_RIGHT,
    ROLLING_BOTTOM_TO_TOP,
    ROLLING_RIGHT_TO_LEFT,
    GLOBAL
};
#endif

#ifndef _GSPLAT_UNSCENTED_TRANSFORM_PARAMS_DEFINED
#define _GSPLAT_UNSCENTED_TRANSFORM_PARAMS_DEFINED
struct UnscentedTransformParameters {
    float alpha = 0.1f;
    float beta = 2.f;
    float kappa = 0.f;
    float in_image_margin_factor = 0.1f;
    bool require_all_sigma_points_valid = true;
};
#endif

namespace gsplat_lfs {

// Redundant pointer validation is debug-only; public tensor/device contracts are
// established by the caller before this low-level backend boundary.
#ifdef DEBUG
    inline void debug_validate_cuda_pointer(const void* pointer, const std::string_view name) {
        lfs::core::validate_cuda_device_pointer(pointer, name);
    }
#else
    inline void debug_validate_cuda_pointer(const void*, std::string_view) {}
#endif

    inline size_t checked_multiply(const size_t lhs,
                                   const size_t rhs,
                                   const std::string_view quantity) {
        LFS_ASSERT_MSG(
            lhs == 0 || rhs <= std::numeric_limits<size_t>::max() / lhs,
            std::format("{} size overflow: {} * {}", quantity, lhs, rhs));
        return lhs * rhs;
    }

    inline size_t checked_bytes(const size_t count,
                                const size_t element_size,
                                const std::string_view allocation) {
        return checked_multiply(count, element_size, allocation);
    }

    void set_cuda_allocation_failure_for_testing(bool fail);

#ifdef LFS_ENABLE_CUDA_FAILURE_INJECTION
    bool cuda_allocation_failure_is_forced();
#endif

    inline void maybe_inject_cuda_allocation_failure(const std::string_view label) {
#ifdef LFS_ENABLE_CUDA_FAILURE_INJECTION
        if (cuda_allocation_failure_is_forced()) {
            LFS_ASSERT_MSG(false, std::format("CUDA allocation for '{}' failed (injected)", label));
        }
#else
        (void)label;
#endif
    }

    inline void checked_cuda_malloc(void** ptr,
                                    const size_t bytes,
                                    const std::string_view label) {
        LFS_ASSERT(ptr != nullptr);
        LFS_ASSERT_MSG(bytes > 0, "CUDA allocation requires a nonzero size");
        maybe_inject_cuda_allocation_failure(label);
        LFS_CUDA_CHECK_MSG(cudaMalloc(ptr, bytes),
                           "gsplat CUDA allocation '{}' ({} bytes)", label, bytes);
        LFS_ASSERT_MSG(*ptr != nullptr,
                       std::format("CUDA allocation for '{}' returned null ({} bytes)", label, bytes));
    }

    class StreamOrderedDeviceBuffer {
    public:
        StreamOrderedDeviceBuffer() = default;

        StreamOrderedDeviceBuffer(const size_t bytes,
                                  const cudaStream_t stream,
                                  const std::string_view label) {
            allocate(bytes, stream, label);
        }

        ~StreamOrderedDeviceBuffer() {
            reset();
        }

        StreamOrderedDeviceBuffer(const StreamOrderedDeviceBuffer&) = delete;
        StreamOrderedDeviceBuffer& operator=(const StreamOrderedDeviceBuffer&) = delete;
        StreamOrderedDeviceBuffer(StreamOrderedDeviceBuffer&&) = delete;
        StreamOrderedDeviceBuffer& operator=(StreamOrderedDeviceBuffer&&) = delete;

        void allocate(const size_t bytes,
                      const cudaStream_t stream,
                      const std::string_view label) {
            LFS_ASSERT_MSG(ptr_ == nullptr, "Stream-ordered CUDA buffer cannot be allocated twice");
            LFS_ASSERT_MSG(bytes > 0, "Stream-ordered CUDA buffer requires a nonzero size");
            maybe_inject_cuda_allocation_failure(label);

            void* ptr = nullptr;
#if CUDART_VERSION >= 11020
            LFS_CUDA_CHECK_MSG(cudaMallocAsync(&ptr, bytes, stream),
                               "gsplat stream-ordered allocation '{}' ({} bytes)", label, bytes);
#else
            LFS_CUDA_CHECK_MSG(cudaMalloc(&ptr, bytes),
                               "gsplat allocation '{}' ({} bytes)", label, bytes);
#endif
            LFS_ASSERT_MSG(ptr != nullptr,
                           std::format("CUDA allocation for '{}' returned null ({} bytes)", label, bytes));
            ptr_ = ptr;
            stream_ = stream;
            bytes_ = bytes;
        }

        void reset() noexcept {
            if (!ptr_) {
                return;
            }
#if CUDART_VERSION >= 11020
            const cudaError_t status = cudaFreeAsync(ptr_, stream_);
#else
            const cudaError_t status = cudaFree(ptr_);
#endif
            if (status != cudaSuccess) {
                lfs::core::ensure_cuda_success(
                    status, "gsplat stream-ordered buffer free",
                    std::format("ptr={}, bytes={}", ptr_, bytes_),
                    std::source_location::current(),
                    lfs::core::CudaFailureDisposition::LogOnly);
                cudaGetLastError();
            }
            ptr_ = nullptr;
            bytes_ = 0;
        }

        void* get() const noexcept { return ptr_; }

        template <typename T>
        T* as() const noexcept {
            return static_cast<T*>(ptr_);
        }

        size_t size() const noexcept { return bytes_; }

    private:
        void* ptr_ = nullptr;
        cudaStream_t stream_ = nullptr;
        size_t bytes_ = 0;
    };

    template <typename Operation>
    void run_cub_operation(const std::string_view name,
                           const cudaStream_t stream,
                           Operation&& operation) {
        size_t workspace_bytes = 0;
        LFS_CUDA_CHECK_MSG(operation(nullptr, workspace_bytes),
                           "{} workspace query", name);
        LFS_ASSERT_MSG(
            workspace_bytes > 0,
            std::format("{} returned an empty workspace for a nonempty operation", name));

        StreamOrderedDeviceBuffer workspace(
            workspace_bytes, stream, "rasterizer.gsplat.cub_workspace");
        LFS_ASSERT_MSG(
            workspace.get() != nullptr,
            std::format("{} cannot execute with null workspace ({} bytes)", name, workspace_bytes));
        LFS_CUDA_CHECK_MSG(operation(workspace.get(), workspace_bytes),
                           "gsplat CUB workspace operation: {}", name);
    }

    //
    // Convenience typedefs for CUDA types
    //
    using vec2 = glm::vec<2, float>;
    using vec3 = glm::vec<3, float>;
    using vec4 = glm::vec<4, float>;
    using mat2 = glm::mat<2, 2, float>;
    using mat3 = glm::mat<3, 3, float>;
    using mat4 = glm::mat<4, 4, float>;
    using mat3x2 = glm::mat<3, 2, float>;

#define N_THREADS_PACKED 256
#define ALPHA_THRESHOLD  (1.f / 255.f)

} // namespace gsplat_lfs
