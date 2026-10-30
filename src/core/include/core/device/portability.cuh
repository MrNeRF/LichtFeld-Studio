/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#if defined(__HIPCC__)
#include <cassert>
#include <hip/hip_runtime.h>
#define LFS_DEBUG_ASSERT_MSG(condition, ...) assert(condition)
#include <hip/hip_cooperative_groups.h>
#define LFS_DEVICE_COMPILER 1
#define LFS_HOST_DEVICE     __host__ __device__
#define LFS_NO_CONTRACT     _Pragma("clang fp contract(off)")
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#define LFS_DEVICE_COMPILER 1
#define LFS_HOST_DEVICE     __host__ __device__
#define LFS_NO_CONTRACT
#else
#define LFS_NO_CONTRACT
#define LFS_DEVICE_COMPILER 0
#define LFS_HOST_DEVICE
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define LFS_DEVICE_CODE 1
#else
#define LFS_DEVICE_CODE 0
#endif

#if LFS_DEVICE_COMPILER
#define LFS_FORCEINLINE __forceinline__
#elif defined(_MSC_VER)
#define LFS_FORCEINLINE __forceinline
#else
#define LFS_FORCEINLINE inline __attribute__((always_inline))
#endif

#if LFS_DEVICE_COMPILER
namespace lfs::core::device {
#if defined(__HIPCC__)
    __device__ inline int32_t componentRoot(int32_t* parent, int32_t node) {
        int32_t current = __hip_atomic_load(parent + node, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        int32_t previous = node;
        for (;;) {
            const int32_t ancestor = __hip_atomic_load(parent + current, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            if (current == ancestor)
                return current;
            atomicMin(parent + previous, ancestor);
            previous = current;
            current = ancestor;
        }
    }
    __device__ inline int32_t finalComponentRoot(const int32_t* parent, int32_t node) {
        for (;;) {
            const int32_t ancestor = __hip_atomic_load(parent + node, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            if (ancestor == node)
                return node;
            node = ancestor;
        }
    }
#endif
    __device__ __forceinline__ int component_load(const int* value) {
#if defined(__HIPCC__)
        return __hip_atomic_load(value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
        return *value;
#endif
    }
    __device__ __forceinline__ void component_store(int* value, int result) {
#if defined(__HIPCC__)
        atomicExch(value, result);
#else
        *value = result;
#endif
    }
    __device__ __forceinline__ float add_rn(float a, float b) {
#if defined(__HIPCC__)
        volatile float result = a + b;
        return result;
#else
        return __fadd_rn(a, b);
#endif
    }
    __device__ __forceinline__ float sub_rn(float a, float b) {
#if defined(__HIPCC__)
        volatile float result = a - b;
        return result;
#else
        return __fsub_rn(a, b);
#endif
    }
    __device__ __forceinline__ float mul_rn(float a, float b) {
#if defined(__HIPCC__)
        volatile float result = a * b;
        return result;
#else
        return __fmul_rn(a, b);
#endif
    }
    __device__ __forceinline__ float div_rn(float a, float b) {
#if defined(__HIPCC__)
        volatile float result = a / b;
        return result;
#else
        return __fdiv_rn(a, b);
#endif
    }
} // namespace lfs::core::device
#endif
