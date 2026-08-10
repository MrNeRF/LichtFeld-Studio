/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/alloc_counter.hpp"
#include "core/assert.hpp"
#include "core/checked_arithmetic.hpp"
#include "core/cuda_allocation.hpp"
#include "core/cuda_error.hpp"
#include "core/crash_handler.hpp"
#include "core/memory_pressure.hpp"

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>
#include <utility>

// cuda.h defines CUDA_VERSION which GLM needs to detect CUDA version properly
// Must be included before any GLM headers
#include <cuda.h>
#include <cuda_runtime.h>

#include <glm/gtc/type_ptr.hpp>

#ifndef LFS_CUDA_FAILURE_INJECTION_ENABLED
#define LFS_CUDA_FAILURE_INJECTION_ENABLED 0
#endif

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

    void* allocate_exact_async_storage(size_t bytes, cudaStream_t stream, const char* label);
    void release_exact_async_storage(void* ptr, cudaStream_t stream) noexcept;
    void record_exact_async_stream(void* ptr, cudaStream_t stream) noexcept;

// Redundant pointer validation is debug-only; public tensor/device contracts are
// established by the caller before this low-level backend boundary.
#ifdef DEBUG_BUILD
    inline void debug_validate_cuda_pointer(const void* pointer, const std::string_view name) {
        LFS_VALIDATE_CUDA_DEVICE_POINTER(pointer, name);
    }
#else
    inline void debug_validate_cuda_pointer(const void*, std::string_view) {}
#endif

    inline size_t checked_multiply(const size_t lhs,
                                   const size_t rhs,
                                   const std::string_view quantity) {
        return lfs::core::checked_product(lhs, rhs, quantity);
    }

    inline size_t checked_bytes(const size_t count,
                                const size_t element_size,
                                const std::string_view allocation) {
        return lfs::core::checked_product(count, element_size, allocation);
    }

#if LFS_CUDA_FAILURE_INJECTION_ENABLED
    void set_cuda_allocation_failure_for_testing(bool fail);
    bool cuda_allocation_failure_is_forced();

    inline void maybe_inject_cuda_allocation_failure(const std::string_view label) {
        if (cuda_allocation_failure_is_forced()) [[unlikely]] {
            LFS_ASSERT_MSG(
                false,
                lfs::core::detail::format_cuda_safe(
                    "CUDA allocation for '{}' failed (injected)", label));
        }
    }
#else
    inline void maybe_inject_cuda_allocation_failure(std::string_view) noexcept {}
#endif

    // Exact, stream-ordered ownership for gsplat's retained workspaces. The
    // exact tier bypasses bucket rounding while the pool still records every
    // allocation in the VRAM ledger and routes release through the serving
    // stream. A reused workspace records any newly participating stream.
    class ExactAsyncDeviceBuffer {
    public:
        ExactAsyncDeviceBuffer() = default;
        explicit ExactAsyncDeviceBuffer(const char* label) : label_(label) {}
        ExactAsyncDeviceBuffer(const size_t bytes, const cudaStream_t stream, const char* label)
            : label_(label) {
            allocate_exact(bytes, stream);
        }
        ExactAsyncDeviceBuffer(const ExactAsyncDeviceBuffer&) = delete;
        ExactAsyncDeviceBuffer& operator=(const ExactAsyncDeviceBuffer&) = delete;
        ExactAsyncDeviceBuffer(ExactAsyncDeviceBuffer&& other) noexcept
            : ptr_(std::exchange(other.ptr_, nullptr)),
              bytes_(std::exchange(other.bytes_, 0)),
              stream_(std::exchange(other.stream_, nullptr)),
              label_(other.label_) {}
        ExactAsyncDeviceBuffer& operator=(ExactAsyncDeviceBuffer&& other) noexcept {
            if (this != &other) {
                reset();
                ptr_ = std::exchange(other.ptr_, nullptr);
                bytes_ = std::exchange(other.bytes_, 0);
                stream_ = std::exchange(other.stream_, nullptr);
                label_ = other.label_;
            }
            return *this;
        }
        ~ExactAsyncDeviceBuffer() { reset(); }

        void bind_stream(cudaStream_t stream) {
            if (ptr_ && stream != stream_) {
                record_exact_async_stream(ptr_, stream);
            }
            stream_ = stream;
        }

        void allocate_exact(const size_t bytes, const cudaStream_t stream) {
            LFS_ASSERT_MSG(ptr_ == nullptr && bytes_ == 0,
                           "gsplat exact workspace must be empty before allocation");
            LFS_ASSERT_MSG(bytes > 0, "gsplat exact workspace cannot allocate zero bytes");
            maybe_inject_cuda_allocation_failure(label_);
            ptr_ = allocate_exact_async_storage(bytes, stream, label_);
            LFS_ASSERT_MSG(ptr_ != nullptr, "gsplat exact workspace allocation returned null");
            bytes_ = bytes;
            stream_ = stream;
        }

        void reset() noexcept {
            if (!ptr_) {
                return;
            }
            void* const ptr = std::exchange(ptr_, nullptr);
            bytes_ = 0;
            if (lfs::core::gpu_process_teardown_started()) {
                stream_ = nullptr;
                return;
            }
            release_exact_async_storage(ptr, stream_);
            stream_ = nullptr;
        }

        [[nodiscard]] void* get() const noexcept { return ptr_; }
        template <typename T> [[nodiscard]] T* as() const noexcept {
            return static_cast<T*>(ptr_);
        }
        [[nodiscard]] size_t size() const noexcept { return bytes_; }
        [[nodiscard]] explicit operator bool() const noexcept { return ptr_ != nullptr; }

    private:
        void* ptr_ = nullptr;
        size_t bytes_ = 0;
        cudaStream_t stream_ = nullptr;
        const char* label_ = "rasterizer.gsplat.workspace";
    };

    using DirectDeviceBuffer = ExactAsyncDeviceBuffer;
    using StreamOrderedDeviceBuffer = ExactAsyncDeviceBuffer;

    /// Grow-only CUB temp storage shared by gsplat scan/sort (thread-local).
    /// Returns a workspace of at least @p bytes; never shrinks until release.
    [[nodiscard]] void* ensure_gsplat_cub_workspace(size_t bytes, cudaStream_t stream);
    bool release_gsplat_cub_workspace() noexcept;

    /// Grow-only intermediate for fused SH backward color grads (thread-local).
    /// Replaces per-backward StreamOrderedDeviceBuffer malloc/free of C×N×ch floats.
    [[nodiscard]] void* ensure_gsplat_color_grad_workspace(size_t bytes, cudaStream_t stream);
    bool release_gsplat_color_grad_workspace() noexcept;

    /// Run a CUB query+execute pair against the pooled high-water workspace
    /// instead of malloc/free every call (legacy CudaCubWorkspace path).
    template <typename Operation>
    void run_cub_operation(const std::string_view name,
                           const cudaStream_t stream,
                           Operation&& operation) {
        size_t workspace_bytes = 0;
        LFS_CUDA_CHECK_MSG(operation(nullptr, workspace_bytes),
                           "{} workspace query", name);
        LFS_ASSERT_MSG(
            workspace_bytes > 0,
            std::string(name) + " returned an empty workspace for a nonempty operation");
        void* const workspace = ensure_gsplat_cub_workspace(workspace_bytes, stream);
        size_t storage_bytes = workspace_bytes;
        LFS_CUDA_CHECK_MSG(operation(workspace, storage_bytes),
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
