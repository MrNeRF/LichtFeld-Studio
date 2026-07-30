/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "core/exportable_storage.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <utility>

using namespace lfs::core;

TEST(ExportableStorageTest, ImmediateDestroyLeavesCudaUsable) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr std::size_t BLOCK_BYTES = 1 << 20;
    auto block_result = allocateExportableDeviceBlock(BLOCK_BYTES, 0, false);
    if (!block_result) {
        FAIL() << block_result.error();
    }

    auto block = std::move(*block_result);
    ASSERT_NE(block, nullptr);
    ASSERT_NE(block->device_ptr, nullptr);
    block.reset();

    constexpr std::size_t PROBE_BYTES = 4096;
    void* probe = nullptr;
    LFS_CUDA_CHECK_MSG_ARGS(
        cudaMalloc(&probe, PROBE_BYTES),
        reinterpret_cast<uintptr_t>(probe),
        0,
        PROBE_BYTES,
        "allocating unrelated CUDA probe dst={} src={} bytes={}",
        probe,
        static_cast<const void*>(nullptr),
        PROBE_BYTES);
    LFS_CUDA_CHECK_MSG_ARGS(
        cudaMemset(probe, 0xa5, PROBE_BYTES),
        reinterpret_cast<uintptr_t>(probe),
        0,
        PROBE_BYTES,
        "writing unrelated CUDA probe dst={} src={} bytes={}",
        probe,
        static_cast<const void*>(nullptr),
        PROBE_BYTES);
    LFS_CUDA_CHECK_MSG_ARGS(
        cudaDeviceSynchronize(),
        reinterpret_cast<uintptr_t>(probe),
        0,
        PROBE_BYTES,
        "synchronizing unrelated CUDA probe dst={} src={} bytes={}",
        probe,
        static_cast<const void*>(nullptr),
        PROBE_BYTES);
    LFS_CUDA_CHECK_MSG_ARGS(
        cudaFree(probe),
        0,
        reinterpret_cast<uintptr_t>(probe),
        PROBE_BYTES,
        "freeing unrelated CUDA probe dst={} src={} bytes={}",
        static_cast<const void*>(nullptr),
        probe,
        PROBE_BYTES);
}
