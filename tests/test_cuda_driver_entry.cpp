/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/driver_entry.hpp"
#include <gtest/gtest.h>
#include <string_view>

// This executable deliberately does not link CUDART or the application. Only
// its fake lookup boundary is replaced; the production resolver runs unchanged.
namespace {
    struct RuntimeState {
        int driver_version = CUDA_VERSION;
        cudaError_t driver_result = cudaSuccess;
        cudaError_t lookup_result = cudaSuccess;
        cudaDriverEntryPointQueryResult query = cudaDriverEntryPointSuccess;
        bool null_entry = false;
        int driver_queries = 0;
        int lookups = 0;
        int cleared_errors = 0;
        unsigned int requested_version = 0;
    } runtime;

    CUresult CUDAAPI granularity(size_t* bytes, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) {
        *bytes = 65536;
        return CUDA_SUCCESS;
    }

    class CudaDriverEntryContracts : public ::testing::Test {
        void SetUp() override { runtime = {}; }
    };

    auto cached_granularity() {
        return LFS_CUDA_DRIVER(cuMemGetAllocationGranularity);
    }
} // namespace

extern "C" cudaError_t CUDARTAPI cudaDriverGetVersion(int* version) {
    ++runtime.driver_queries;
    *version = runtime.driver_version;
    return runtime.driver_result;
}

extern "C" cudaError_t CUDARTAPI cudaGetLastError() {
    ++runtime.cleared_errors;
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion(
    const char* name, void** entry, unsigned int version, unsigned long long flags,
    cudaDriverEntryPointQueryResult* query) {
    ++runtime.lookups;
    runtime.requested_version = version;
    EXPECT_EQ(flags, cudaEnableDefault);
    EXPECT_EQ(std::string_view(name), "cuMemGetAllocationGranularity");
    *entry = nullptr;
    if (version > static_cast<unsigned int>(runtime.driver_version))
        return cudaErrorInvalidValue;
    if (runtime.lookup_result != cudaSuccess)
        return runtime.lookup_result;
    if (query)
        *query = runtime.query;
    if (!runtime.null_entry)
        *entry = reinterpret_cast<void*>(&granularity);
    return cudaSuccess;
}

TEST_F(CudaDriverEntryContracts, NewerToolkitResolvesVmmOnOlderDriver) {
    runtime.driver_version = CUDA_VERSION - 10;
    const auto entry = lfs::core::cuda_driver::resolve<decltype(&cuMemGetAllocationGranularity)>("cuMemGetAllocationGranularity");
    size_t bytes = 0;
    EXPECT_EQ(entry(&bytes, nullptr, CU_MEM_ALLOC_GRANULARITY_MINIMUM), CUDA_SUCCESS);
    EXPECT_EQ(bytes, 65536);
    EXPECT_EQ(runtime.requested_version, runtime.driver_version);
    EXPECT_EQ(runtime.cleared_errors, 0);
}

TEST_F(CudaDriverEntryContracts, NewerDriverKeepsToolkitAbi) {
    runtime.driver_version = CUDA_VERSION + 10;
    const auto entry = lfs::core::cuda_driver::resolve<decltype(&cuMemGetAllocationGranularity)>("cuMemGetAllocationGranularity");
    size_t bytes = 0;
    EXPECT_EQ(entry(&bytes, nullptr, CU_MEM_ALLOC_GRANULARITY_MINIMUM), CUDA_SUCCESS);
    EXPECT_EQ(runtime.requested_version, CUDA_VERSION);
}

TEST_F(CudaDriverEntryContracts, EqualVersionsResolveAndCallSiteCachesEntry) {
    (void)cached_granularity();
    const int initial_queries = runtime.driver_queries;
    const int initial_lookups = runtime.lookups;
    EXPECT_LE(initial_queries, 1);
    EXPECT_EQ(initial_queries, initial_lookups);
    for (int i = 0; i < 100; ++i) {
        size_t bytes = 0;
        EXPECT_EQ(cached_granularity()(&bytes, nullptr, CU_MEM_ALLOC_GRANULARITY_MINIMUM), CUDA_SUCCESS);
        EXPECT_EQ(bytes, 65536);
    }
    EXPECT_EQ(runtime.driver_queries, initial_queries);
    EXPECT_EQ(runtime.lookups, initial_lookups);
}

TEST_F(CudaDriverEntryContracts, DriverQueryFailureDoesNotAttemptLookup) {
    runtime.driver_result = cudaErrorInsufficientDriver;
    const auto entry = lfs::core::cuda_driver::resolve<decltype(&cuMemGetAllocationGranularity)>("cuMemGetAllocationGranularity");
    size_t bytes = 0;
    EXPECT_EQ(entry(&bytes, nullptr, CU_MEM_ALLOC_GRANULARITY_MINIMUM), CUDA_ERROR_NOT_INITIALIZED);
    EXPECT_EQ(runtime.lookups, 0);
    EXPECT_EQ(runtime.cleared_errors, 1);
}

TEST_F(CudaDriverEntryContracts, MissingDriverVersionDoesNotAttemptLookup) {
    runtime.driver_version = 0;
    const auto entry = lfs::core::cuda_driver::resolve<decltype(&cuMemGetAllocationGranularity)>("cuMemGetAllocationGranularity");
    size_t bytes = 0;
    EXPECT_EQ(entry(&bytes, nullptr, CU_MEM_ALLOC_GRANULARITY_MINIMUM), CUDA_ERROR_NOT_INITIALIZED);
    EXPECT_EQ(runtime.lookups, 0);
}

TEST_F(CudaDriverEntryContracts, LookupFailureClearsErrorAndReturnsUnavailable) {
    runtime.lookup_result = cudaErrorInvalidValue;
    const auto entry = lfs::core::cuda_driver::resolve<decltype(&cuMemGetAllocationGranularity)>("cuMemGetAllocationGranularity");
    size_t bytes = 0;
    EXPECT_EQ(entry(&bytes, nullptr, CU_MEM_ALLOC_GRANULARITY_MINIMUM), CUDA_ERROR_NOT_INITIALIZED);
    EXPECT_EQ(runtime.cleared_errors, 1);
}

TEST_F(CudaDriverEntryContracts, SuccessfulLookupWithNullPointerIsUnavailable) {
    runtime.null_entry = true;
    const auto entry = lfs::core::cuda_driver::resolve<decltype(&cuMemGetAllocationGranularity)>("cuMemGetAllocationGranularity");
    size_t bytes = 0;
    EXPECT_EQ(entry(&bytes, nullptr, CU_MEM_ALLOC_GRANULARITY_MINIMUM), CUDA_ERROR_NOT_INITIALIZED);
}

TEST_F(CudaDriverEntryContracts, IncompatibleSymbolStatusRejectsEvenNonNullPointer) {
    runtime.query = cudaDriverEntryPointVersionNotSufficent;
    const auto entry = lfs::core::cuda_driver::resolve<decltype(&cuMemGetAllocationGranularity)>("cuMemGetAllocationGranularity");
    size_t bytes = 0;
    EXPECT_EQ(entry(&bytes, nullptr, CU_MEM_ALLOC_GRANULARITY_MINIMUM), CUDA_ERROR_NOT_INITIALIZED);
}
