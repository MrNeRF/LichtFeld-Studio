/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/assert.hpp"
#include "core/checked_arithmetic.hpp"
#include "core/cuda_allocation.hpp"
#include "core/cuda_error.hpp"

#include <cstdint>
#include <cstdio>
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

    struct GsplatAllocationHooks {
        void before_allocate(const std::string_view label) const {
            maybe_inject_cuda_allocation_failure(label);
        }

        void after_allocate(void*, size_t, std::string_view) const noexcept {}
        void before_deallocate(void*) const noexcept {}
    };

    struct GsplatCubWorkspaceTraits {
        static constexpr std::string_view allocation_label = "rasterizer.gsplat.cub_workspace";
        static constexpr std::string_view diagnostic_scope = "gsplat";
    };

    using DirectDeviceBuffer = lfs::core::UniqueCudaAllocation<
        lfs::core::DirectCudaAllocator, GsplatAllocationHooks>;
    using StreamOrderedDeviceBuffer = lfs::core::UniqueCudaAllocation<
        lfs::core::StreamOrderedCudaAllocator, GsplatAllocationHooks>;
    using GsplatCubWorkspace = lfs::core::CudaCubWorkspace<
        StreamOrderedDeviceBuffer, GsplatCubWorkspaceTraits>;

    template <typename Operation>
    void run_cub_operation(const std::string_view name,
                           const cudaStream_t stream,
                           Operation&& operation) {
        lfs::core::run_cub_operation<GsplatCubWorkspace>(
            name, stream, std::forward<Operation>(operation));
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
