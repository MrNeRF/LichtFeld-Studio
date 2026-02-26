/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>

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

namespace gsplat_fwd {

// Validation macros (no-ops in release, enabled in debug)
#ifdef DEBUG
#define GSPLAT_CHECK_CUDA_PTR(ptr, name)                         \
    do {                                                         \
        if ((ptr) == nullptr) {                                  \
            fprintf(stderr, "GSPLAT ERROR: %s is null\n", name); \
        }                                                        \
    } while (false)
#else
#define GSPLAT_CHECK_CUDA_PTR(ptr, name) ((void)0)
#endif

// CUB wrapper that handles temporary storage allocation
// Uses cudaMalloc instead of PyTorch caching allocator
#define CUB_WRAPPER_LFS(func, ...)                                            \
    do {                                                                      \
        size_t temp_storage_bytes = 0;                                        \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                       \
        void* temp_storage = nullptr;                                         \
        cudaError_t _cub_err = cudaMalloc(&temp_storage, temp_storage_bytes); \
        assert(_cub_err == cudaSuccess && "CUB temp alloc failed");           \
        func(temp_storage, temp_storage_bytes, __VA_ARGS__);                  \
        cudaFree(temp_storage);                                               \
    } while (false)

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

} // namespace gsplat_fwd
