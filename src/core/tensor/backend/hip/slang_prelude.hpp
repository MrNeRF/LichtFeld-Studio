/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/device/portability.cuh"
#include <hip/hip_fp16.h>
// Supply Slang's CUDA-target ABI with HIP types, leaving compiler output intact.
#define SLANG_CUDA_PRELUDE_H
#define SLANG_INFINITY INFINITY
namespace hip_slang {

    using uint = unsigned;
    using ulonglong = unsigned long long;
    using ::make_float2;
    using ::make_float3;
    __device__ inline float3 make_float3(float x) { return ::make_float3(x, x, x); }
    __device__ inline float2 make_float2(float x) { return ::make_float2(x, x); }
    template <class T, int N>
    struct FixedArray {
        T data[N];
        __device__ T& operator[](int i) { return data[i]; }
        __device__ const T& operator[](int i) const { return data[i]; }
    };
    __device__ inline uint F32_asuint(float x) { return __float_as_uint(x); }
    __device__ inline float F32_fma(float x, float y, float z) { return fmaf(x, y, z); }
    __device__ inline float F32_min(float x, float y) { return fminf(x, y); }
    __device__ inline float F32_max(float x, float y) { return fmaxf(x, y); }
    __device__ inline float F32_abs(float x) { return fabsf(x); }
    __device__ inline float F32_exp(float x) { return expf(x); }
    __device__ inline float F32_log(float x) { return logf(x); }
    __device__ inline float F32_sqrt(float x) { return sqrtf(x); }
    __device__ inline float F32_pow(float x, float y) { return powf(x, y); }
    __device__ inline float F32_round(float x) { return roundf(x); }
    __device__ inline bool F32_isfinite(float x) { return isfinite(x); }
    __device__ inline bool F32_isinf(float x) { return isinf(x); }
    __device__ inline bool F32_isnan(float x) { return isnan(x); }
    __device__ inline float U32_asfloat(uint x) { return __uint_as_float(x); }
    __device__ inline uint U32_min(uint x, uint y) { return x < y ? x : y; }
    __device__ inline int I32_max(int x, int y) { return x > y ? x : y; }
    __device__ inline int I32_min(int x, int y) { return x < y ? x : y; }
    __device__ inline float F32_ceil(float x) { return ceilf(x); }
    __device__ inline float F32_asin(float x) { return asinf(x); }
    __device__ inline float F32_atan2(float x, float y) { return atan2f(x, y); }
    struct bool3 {
        bool x, y, z;
    };
    __device__ inline bool3 operator>=(float3 a, float3 b) { return {a.x >= b.x, a.y >= b.y, a.z >= b.z}; }
    __device__ inline bool3 operator<=(float3 a, float3 b) { return {a.x <= b.x, a.y <= b.y, a.z <= b.z}; }
    __device__ inline bool* _slang_vector_get_element_ptr(bool3* p, int i) { return i == 0 ? &p->x : i == 1 ? &p->y
                                                                                                            : &p->z; }
    __device__ inline bool _slang_vector_get_element(bool3 p, int i) { return i == 0 ? p.x : i == 1 ? p.y
                                                                                                    : p.z; }
    __device__ inline float* _slang_vector_get_element_ptr(float2* p, int i) { return i == 0 ? &p->x : &p->y; }
    __device__ inline float _slang_vector_get_element(float2 p, int i) { return i == 0 ? p.x : p.y; }
    __device__ inline uint* _slang_vector_get_element_ptr(uint3* p, int i) { return i == 0 ? &p->x : i == 1 ? &p->y
                                                                                                            : &p->z; }
    __device__ inline uint _slang_vector_get_element(uint3 p, int i) { return i == 0 ? p.x : i == 1 ? p.y
                                                                                                    : p.z; }
    __device__ inline float* _slang_vector_get_element_ptr(float3* p, int i) { return i == 0 ? &p->x : i == 1 ? &p->y
                                                                                                              : &p->z; }
    __device__ inline float _slang_vector_get_element(float3 p, int i) { return i == 0 ? p.x : i == 1 ? p.y
                                                                                                      : p.z; }

    __device__ inline float F32_floor(float x) { return floorf(x); }
    __device__ inline float F32_tanh(float x) { return tanhf(x); }
    __device__ inline uint3 lfs_dispatch_groups() { return make_uint3(gridDim.x, gridDim.y, gridDim.z); }
    __device__ inline uint3 operator*(decltype(blockIdx) a, decltype(blockDim) b) { return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z); }
    __device__ inline uint3 operator+(uint3 a, decltype(threadIdx) b) { return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z); }
    __device__ inline float lfs_rounded_add(float a, float b) { LFS_NO_CONTRACT return a + b; }
    __device__ inline float lfs_rounded_sub(float a, float b) { LFS_NO_CONTRACT return a - b; }
    __device__ inline float lfs_rounded_mul(float a, float b) { LFS_NO_CONTRACT return a * b; }
    __device__ inline float lfs_rounded_div(float a, float b) { LFS_NO_CONTRACT return a / b; }
} // namespace hip_slang
