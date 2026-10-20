/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda/driver_entry.hpp"
#include "core/detail/gpu_backend_ops.hpp"
#include "core/gpu_device_info.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_cuda_interop.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

TEST(CudaDeviceSelection, AutoPreservesWorkerDeviceAndExplicitSelectionBindsNewWorkers) {
    using namespace lfs::core;
    const auto devices = enumerate_cuda_hardware();
    if (devices.empty())
        GTEST_SKIP() << "CUDA unavailable";
    ASSERT_TRUE(set_tensor_backend_options({}));
    std::vector<std::thread> workers;
    for (const auto& device : devices) {
        workers.emplace_back([ordinal = device.index] {
            ASSERT_EQ(cudaSetDevice(ordinal), cudaSuccess);
            (void)internal::backend_ops(GpuBackend::CUDA);
            int current = -1;
            ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
            EXPECT_EQ(current, int(ordinal));
        });
    }
    for (auto& worker : workers)
        worker.join();
    workers.clear();

    const auto selected = std::find_if(devices.rbegin(), devices.rend(), [](const auto& device) {
        return device.tensor_unavailable_reason.empty();
    });
    ASSERT_NE(selected, devices.rend());
    TensorBackendOptions options;
    options.cuda_device = selected->uuid;
    ASSERT_TRUE(set_tensor_backend_options(options));
    ASSERT_TRUE(set_default_gpu_backend(GpuBackend::CUDA));
    for (int i = 0; i < 4; ++i) {
        workers.emplace_back([&, i] {
            // On multi-CUDA machines this starts some workers on another card.
            ASSERT_EQ(cudaSetDevice(devices[i % devices.size()].index), cudaSuccess);
            (void)internal::backend_ops(GpuBackend::CUDA);
            int current = -1;
            ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
            EXPECT_EQ(current, int(selected->index));
            for (int call = 0; call < 100; ++call) {
                (void)getCurrentCUDAStream();
                (void)internal::backend_ops(GpuBackend::CUDA);
            }
            ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
            EXPECT_EQ(current, int(selected->index));
            // Explicit entry scopes must rebind even after an external switch.
            ASSERT_EQ(cudaSetDevice(devices.front().index), cudaSuccess);
            const GpuBackendScope scope(GpuBackend::CUDA);
            const auto tensor = Tensor::ones({8}, Device::GPU);
            cudaPointerAttributes attributes{};
            ASSERT_EQ(cudaPointerGetAttributes(&attributes, tensor.data_ptr()), cudaSuccess);
            EXPECT_EQ(attributes.device, int(selected->index));
            EXPECT_EQ(tensor.cpu().ptr<float>()[0], 1.f);
        });
    }
    for (auto& worker : workers)
        worker.join();
}

#if defined(__linux__)
TEST(CudaDriverEntry, IndependentLazyEntriesAndMissingSymbol) {
    if (lfs::core::enumerate_cuda_hardware().empty())
        GTEST_SKIP() << "CUDA unavailable";
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
#endif
