/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/driver_entry.hpp"
#include <gtest/gtest.h>

namespace {
    class CudaDriverEntryLive : public ::testing::Test {
        void SetUp() override {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
                GTEST_SKIP() << "CUDA device unavailable";
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
        }
    };
} // namespace

TEST_F(CudaDriverEntryLive, VmmAllocationEntryResolvesOnInstalledDriver) {
    int supports_vmm = 0;
    ASSERT_EQ(LFS_CUDA_DRIVER(cuDeviceGetAttribute)(
                  &supports_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, 0),
              CUDA_SUCCESS);
    if (!supports_vmm)
        GTEST_SKIP() << "Device does not support VMM";
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    size_t bytes = 0;
    ASSERT_EQ(LFS_CUDA_DRIVER(cuMemGetAllocationGranularity)(
                  &bytes, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
              CUDA_SUCCESS);
    EXPECT_GT(bytes, 0);
}

TEST_F(CudaDriverEntryLive, IndependentLazyEntriesAndMissingSymbol) {
    int count = -1, version = -1, runtime_version = -1;
    ASSERT_EQ(LFS_CUDA_DRIVER(cuDeviceGetCount)(&count), CUDA_SUCCESS);
    ASSERT_EQ(LFS_CUDA_DRIVER(cuDriverGetVersion)(&version), CUDA_SUCCESS);
    ASSERT_EQ(cudaDriverGetVersion(&runtime_version), cudaSuccess);
    EXPECT_GT(count, 0);
    EXPECT_EQ(version, runtime_version);
    // These entries share a C signature but must not share a cached symbol.
    EXPECT_NE(count, version);
    const auto missing = lfs::core::cuda_driver::resolve<decltype(&cuDeviceGetCount)>("lfs_missing_cuda_test_entry");
    EXPECT_EQ(missing(&count), CUDA_ERROR_NOT_INITIALIZED);
    ASSERT_EQ(LFS_CUDA_DRIVER(cuDriverGetVersion)(&version), CUDA_SUCCESS);
    EXPECT_EQ(version, runtime_version);
}
