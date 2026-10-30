/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

// Forced include for a separate, opt-in test module. Never included by the
// normal runtime: the production module has no error-injection entry points.
#include "runtime_api.h"
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

namespace lfs_hip_test {
    inline int fault = 0, pinned = 0, events = 0, device_allocations = 0;
    inline bool fail(int operation) {
        if (fault != operation)
            return false;
        fault = 0;
        return true;
    }
} // namespace lfs_hip_test

#ifdef LFS_HIP_TEST_EXPORTS
extern "C" LFS_HIP_API void lfs_hip_test_fail_once(int operation) {
    lfs_hip_test::fault = operation;
}
extern "C" LFS_HIP_API int lfs_hip_test_live_resources() {
    // Check each category independently, including accidental negative counts.
    return (lfs_hip_test::pinned != 0 ? 1 : 0) | (lfs_hip_test::events != 0 ? 2 : 0) | (lfs_hip_test::device_allocations != 0 ? 4 : 0);
}

#endif

static hipError_t lfs_hip_test_host_malloc(void** pointer, size_t bytes, unsigned flags) {
    if (lfs_hip_test::fail(1))
        return hipErrorOutOfMemory;
    const auto result = hipHostMalloc(pointer, bytes, flags);
    if (result == hipSuccess)
        ++lfs_hip_test::pinned;
    return result;
}
static hipError_t lfs_hip_test_host_free(void* pointer) {
    if (lfs_hip_test::fail(2))
        return hipErrorUnknown;
    const auto result = hipHostFree(pointer);
    if (result == hipSuccess)
        --lfs_hip_test::pinned;
    return result;
}
static hipError_t lfs_hip_test_event_record(hipEvent_t event, hipStream_t stream) {
    if (lfs_hip_test::fail(3))
        return hipErrorUnknown;
    return hipEventRecord(event, stream);
}
static hipError_t lfs_hip_test_event_destroy(hipEvent_t event) {
    if (lfs_hip_test::fail(4))
        return hipErrorUnknown;
    const auto result = hipEventDestroy(event);
    if (result == hipSuccess)
        --lfs_hip_test::events;
    return result;
}
static hipError_t lfs_hip_test_event_create(hipEvent_t* event, unsigned flags) {
    if (lfs_hip_test::fail(5))
        return hipErrorOutOfMemory;
    const auto result = hipEventCreateWithFlags(event, flags);
    if (result == hipSuccess)
        ++lfs_hip_test::events;
    return result;
}
static hipError_t lfs_hip_test_plain_event_create(hipEvent_t* event) {
    return lfs_hip_test_event_create(event, hipEventDefault);
}
static hipError_t lfs_hip_test_copy(void* destination, const void* source, size_t bytes, hipMemcpyKind kind, hipStream_t stream) {
    if (lfs_hip_test::fail(6))
        return hipErrorUnknown;
    return hipMemcpyAsync(destination, source, bytes, kind, stream);
}
static hipError_t lfs_hip_test_stream_synchronize(hipStream_t stream) {
    if (lfs_hip_test::fail(7))
        return hipErrorUnknown;
    return hipStreamSynchronize(stream);
}

#define hipHostMalloc           lfs_hip_test_host_malloc
#define hipHostFree             lfs_hip_test_host_free
#define hipEventRecord          lfs_hip_test_event_record
#define hipEventDestroy         lfs_hip_test_event_destroy
#define hipEventCreateWithFlags lfs_hip_test_event_create
#define hipEventCreate          lfs_hip_test_plain_event_create
#define hipMemcpyAsync          lfs_hip_test_copy
#define hipStreamSynchronize    lfs_hip_test_stream_synchronize

// Instantiate SDK templates before replacing allocation calls in our module.
template <typename T>
static hipError_t lfs_hip_test_device_malloc(T** pointer, size_t bytes) {
    if (lfs_hip_test::fail(8))
        return hipErrorOutOfMemory;
    const auto result = hipMalloc(pointer, bytes);
    if (result == hipSuccess)
        ++lfs_hip_test::device_allocations;
    return result;
}
static hipError_t lfs_hip_test_device_free(void* pointer) {
    if (lfs_hip_test::fail(9))
        return hipErrorUnknown;
    const auto result = hipFree(pointer);
    if (result == hipSuccess && pointer)
        --lfs_hip_test::device_allocations;
    return result;
}
#define hipMalloc lfs_hip_test_device_malloc
#define hipFree   lfs_hip_test_device_free

static hipError_t lfs_hip_test_memset_async(void* pointer, int value, size_t bytes, hipStream_t stream) {
    if (lfs_hip_test::fail(10))
        return hipErrorUnknown;
    return hipMemsetAsync(pointer, value, bytes, stream);
}
#define hipMemsetAsync lfs_hip_test_memset_async
