/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "training/kernels/cuda_scratch.hpp"
#include "training/kernels/densification_kernels.hpp"
#include <algorithm>
#include <cub/cub.cuh>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::training;

TEST(MedianSelectionPerformance, RelativeToSort) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA unavailable";
    const lfs::core::GpuBackendScope backend(lfs::core::GpuBackend::CUDA);
    constexpr size_t n = 65536;
    constexpr int repetitions = 30;
    std::vector<float> values(n);
    for (size_t i = 0; i < n; ++i)
        values[i] = 0.25f + static_cast<float>((i * 7919) % n) / n;
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    lfs::training::cuda_scratch::DeviceBuffer source(n * sizeof(float), stream, "test.median.values");
    lfs::training::cuda_scratch::DeviceBuffer input(n * sizeof(float), stream, "test.median.values");
    lfs::training::cuda_scratch::DeviceBuffer sorted(n * sizeof(float), stream, "test.median.values");
    ASSERT_EQ(cudaMemcpyAsync(source.get(), values.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream), cudaSuccess);
    size_t scratch_bytes = 0;
    ASSERT_EQ(cub::DeviceRadixSort::SortKeys(nullptr, scratch_bytes, source.as<float>(), sorted.as<float>(), n, 0, 32, stream), cudaSuccess);
    lfs::training::cuda_scratch::DeviceBuffer scratch(scratch_bytes, stream, "test.median.sort");
    cudaEvent_t start, stop;
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);
    auto launch_workload = [&](bool select) {
        if (select) {
            LFS_CUDA_CHECK(cudaMemcpyAsync(input.as<float>(), source.as<float>(), n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            kernels::launch_normalize_by_positive_median(input.as<float>(), n, stream);
        } else {
            LFS_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(scratch.get(), scratch_bytes,
                                                          source.as<float>(), sorted.as<float>(), n, 0, 32, stream));
        }
    };
    launch_workload(true);
    launch_workload(false);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    // Graph replay excludes host allocation and dispatch gaps from both timings.
    cudaGraph_t graphs[2];
    cudaGraphExec_t executable[2];
    for (int select = 0; select < 2; ++select) {
        ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
        launch_workload(select != 0);
        ASSERT_EQ(cudaStreamEndCapture(stream, &graphs[select]), cudaSuccess);
        ASSERT_EQ(cudaGraphInstantiate(&executable[select], graphs[select], nullptr, nullptr, 0), cudaSuccess);
    }
    auto measure = [&](bool select) {
        LFS_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < repetitions; ++i)
            LFS_CUDA_CHECK(cudaGraphLaunch(executable[select], stream));
        LFS_CUDA_CHECK(cudaEventRecord(stop, stream));
        LFS_CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        LFS_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };
    measure(true);
    measure(false);
    std::vector<float> ratios;
    for (int trial = 0; trial < 7; ++trial) {
        // Alternate order to limit clock/thermal drift; compare in one process.
        float selection_ms, sort_ms;
        if (trial % 2 == 0) {
            selection_ms = measure(true);
            sort_ms = measure(false);
        } else {
            sort_ms = measure(false);
            selection_ms = measure(true);
        }
        ratios.push_back(selection_ms / sort_ms);
    }
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(start), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(stop), cudaSuccess);
    std::sort(ratios.begin(), ratios.end());
    // Selection includes normalization and input reset. Allow twice the cost
    // of a full GPU sort, with no hardware-specific absolute time limit.
    EXPECT_LT(ratios[ratios.size() / 2], 2.0f);
    RecordProperty("median_selection_to_sort_ratio", ratios[ratios.size() / 2]);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (int i = 0; i < 2; ++i) {
        ASSERT_EQ(cudaGraphExecDestroy(executable[i]), cudaSuccess);
        ASSERT_EQ(cudaGraphDestroy(graphs[i]), cudaSuccess);
    }
    source.reset();
    input.reset();
    sorted.reset();
    scratch.reset();
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
