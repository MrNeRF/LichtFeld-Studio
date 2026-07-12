/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda_error.hpp"

#ifdef CUDA_DEBUG_SYNC
#define CUDA_KERNEL_CHECK(name)                                                             \
    do {                                                                                    \
        ::lfs::core::ensure_cuda_success(                                                   \
            cudaGetLastError(), "cudaGetLastError()", (name),                               \
            std::source_location::current(), ::lfs::core::CudaFailureDisposition::LogOnly); \
        ::lfs::core::ensure_cuda_success(                                                   \
            cudaDeviceSynchronize(), "cudaDeviceSynchronize()", (name),                     \
            std::source_location::current(), ::lfs::core::CudaFailureDisposition::LogOnly); \
    } while (false)

#define CUDA_KERNEL_LAUNCH(kernel, grid, block, shared, stream, ...) \
    do {                                                             \
        kernel<<<grid, block, shared, stream>>>(__VA_ARGS__);        \
        CUDA_KERNEL_CHECK(#kernel);                                  \
    } while (0)
#elif defined(DEBUG_BUILD)
#define CUDA_KERNEL_CHECK(name)                           \
    ::lfs::core::ensure_cuda_success(                     \
        cudaGetLastError(), "cudaGetLastError()", (name), \
        std::source_location::current(), ::lfs::core::CudaFailureDisposition::LogOnly)

#define CUDA_KERNEL_LAUNCH(kernel, grid, block, shared, stream, ...) \
    do {                                                             \
        kernel<<<grid, block, shared, stream>>>(__VA_ARGS__);        \
        CUDA_KERNEL_CHECK(#kernel);                                  \
    } while (0)
#else
#define CUDA_KERNEL_CHECK(name) ((void)0)

#define CUDA_KERNEL_LAUNCH(kernel, grid, block, shared, stream, ...) \
    kernel<<<grid, block, shared, stream>>>(__VA_ARGS__)
#endif

#define CUDA_KERNEL(kernel, grid, block, ...) \
    CUDA_KERNEL_LAUNCH(kernel, grid, block, 0, 0, __VA_ARGS__)
