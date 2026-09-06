/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor_backend.hpp"
#include "rendering/selection_ops.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <tuple>
#include <vector>

namespace {

    using namespace lfs::core;

#define SKIP_IF_BACKEND_UNAVAILABLE(backend)               \
    do {                                                   \
        const GpuBackend skip_backend = (backend);         \
        if (!gpu_backend_available(skip_backend)) {        \
            GTEST_SKIP() << gpu_backend_name(skip_backend) \
                         << " backend unavailable";        \
        }                                                  \
    } while (false)

    void expect_bytes_equal(const Tensor& actual, const Tensor& expected, const char* const what) {
        const Tensor actual_cpu = actual.cpu().contiguous();
        const Tensor expected_cpu = expected.cpu().contiguous();
        ASSERT_TRUE(actual_cpu.is_valid()) << what;
        ASSERT_TRUE(expected_cpu.is_valid()) << what;
        ASSERT_EQ(actual_cpu.dtype(), expected_cpu.dtype()) << what;
        ASSERT_EQ(actual_cpu.numel(), expected_cpu.numel()) << what;
        ASSERT_EQ(actual_cpu.bytes(), expected_cpu.bytes()) << what;
        EXPECT_EQ(std::memcmp(actual_cpu.data_ptr(), expected_cpu.data_ptr(), actual_cpu.bytes()), 0)
            << what;
    }

    void expect_full_matches_cpu(const TensorShape& shape,
                                 const float value,
                                 const DataType dtype) {
        const Tensor cpu = Tensor::full(shape, value, Device::CPU, dtype);
        const Tensor gpu = Tensor::full(shape, value, Device::CUDA, dtype);
        expect_bytes_equal(gpu, cpu, "full");
    }

    void run_full_suite() {
        expect_full_matches_cpu({8}, 1.0f, DataType::Float32);
        expect_full_matches_cpu({8}, -1.0f, DataType::Float32);
        expect_full_matches_cpu({8}, 3.5f, DataType::Float32);
        expect_full_matches_cpu({8}, 0.0f, DataType::Float32);
        expect_full_matches_cpu({8}, 1.0f, DataType::Float16);
        expect_full_matches_cpu({8}, -1.0f, DataType::Float16);
        expect_full_matches_cpu({8}, 3.5f, DataType::Float16);
        expect_full_matches_cpu({8}, 65504.0f, DataType::Float16);
        expect_full_matches_cpu({8}, 1.0f, DataType::Int32);
        expect_full_matches_cpu({8}, -1.0f, DataType::Int32);
        expect_full_matches_cpu({8}, 3.5f, DataType::Int32);
        expect_full_matches_cpu({8}, 1.0f, DataType::Int64);
        expect_full_matches_cpu({8}, -1.0f, DataType::Int64);
        expect_full_matches_cpu({8}, 1099511627776.0f, DataType::Int64);
        expect_full_matches_cpu({8}, 255.0f, DataType::UInt8);
        expect_full_matches_cpu({8}, 1.0f, DataType::UInt8);
        expect_full_matches_cpu({8}, 0.0f, DataType::UInt8);

        expect_bytes_equal(Tensor::ones({16}, Device::CUDA, DataType::Int32),
                           Tensor::ones({16}, Device::CPU, DataType::Int32),
                           "ones Int32");
        expect_bytes_equal(Tensor::zeros({16}, Device::CUDA, DataType::Int64),
                           Tensor::zeros({16}, Device::CPU, DataType::Int64),
                           "zeros Int64");
        expect_bytes_equal(Tensor::full_bool({16}, true, Device::CUDA),
                           Tensor::full_bool({16}, true, Device::CPU),
                           "full_bool true");
        expect_bytes_equal(Tensor::full_bool({16}, false, Device::CUDA),
                           Tensor::full_bool({16}, false, Device::CPU),
                           "full_bool false");
        expect_bytes_equal(Tensor::ones({16}, Device::CUDA, DataType::Float16),
                           Tensor::ones({16}, Device::CPU, DataType::Float16),
                           "ones Float16");
    }

    Tensor load_arange(const float start,
                       const float end,
                       const float step,
                       const Device device,
                       const DataType dtype) {
        LoadArgs args;
        args.device = device;
        args.dtype = dtype;
        args.args = std::tuple<float, float, float>{start, end, step};
        return Tensor::load(LoadOp::Arange, args);
    }

    void expect_arange_matches_cpu(const float start,
                                   const float end,
                                   const float step,
                                   const DataType dtype) {
        const Tensor cpu = load_arange(start, end, step, Device::CPU, dtype);
        const Tensor gpu = load_arange(start, end, step, Device::CUDA, dtype);
        char what[160];
        std::snprintf(what, sizeof(what), "arange start=%g end=%g step=%g dtype=%d count=%zu",
                      static_cast<double>(start), static_cast<double>(end),
                      static_cast<double>(step), static_cast<int>(dtype), cpu.numel());
        expect_bytes_equal(gpu, cpu, what);
    }

    void run_arange_suite() {
        expect_arange_matches_cpu(0.0f, 16.0f, 1.0f, DataType::Float32);
        expect_arange_matches_cpu(0.0f, 87040.0f, 1.0f, DataType::Float32);
        expect_arange_matches_cpu(-4.0f, 4.0f, 0.5f, DataType::Float32);
        expect_arange_matches_cpu(10.0f, -10.0f, -1.25f, DataType::Float32);
        expect_arange_matches_cpu(-3.0f, 8701.0f, 0.1f, DataType::Float32);
        expect_arange_matches_cpu(0.5f, 26112.5f, 0.3f, DataType::Float32);
        expect_arange_matches_cpu(100.25f, -60827.75f, -0.7f, DataType::Float32);
        expect_arange_matches_cpu(0.0f, 87040.0f / 3.0f, 1.0f / 3.0f, DataType::Float32);
        expect_arange_matches_cpu(0.0f, 16.0f, 1.0f, DataType::Int32);
        expect_arange_matches_cpu(0.0f, 87040.0f, 1.0f, DataType::Int32);
        expect_arange_matches_cpu(-20.0f, 20.0f, 3.0f, DataType::Int32);
        expect_arange_matches_cpu(50.0f, -50.0f, -7.0f, DataType::Int32);
        expect_arange_matches_cpu(-3.0f, 8701.0f, 0.1f, DataType::Int32);
        expect_arange_matches_cpu(0.5f, 26112.5f, 0.3f, DataType::Int32);
        expect_arange_matches_cpu(100.25f, -60827.75f, -0.7f, DataType::Int32);
        expect_arange_matches_cpu(0.0f, 87040.0f / 3.0f, 1.0f / 3.0f, DataType::Int32);
        expect_bytes_equal(Tensor::arange(0.0f, 87040.0f),
                           load_arange(0.0f, 87040.0f, 1.0f, Device::CPU, DataType::Float32),
                           "public arange");
    }

    void sync_tensor(const Tensor& tensor) {
        internal::backend_ops_for(tensor).synchronize_stream(
            internal::ExecContext{tensor.stream()});
    }

    double elapsed_us(const std::chrono::steady_clock::time_point start,
                      const std::chrono::steady_clock::time_point stop) {
        return std::chrono::duration<double, std::micro>(stop - start).count();
    }

    void print_sample_stats(const char* const backend,
                            const char* const what,
                            const std::vector<double>& samples) {
        if (samples.empty()) {
            return;
        }
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const double first = samples.front();
        const double min_us = sorted.front();
        const double median_us = sorted[sorted.size() / 2];
        std::printf("TensorAsyncConsts bench-detail %s %s first=%.3f us min=%.3f us median=%.3f us n=%zu\n",
                    backend, what, first, min_us, median_us, samples.size());
    }

    void print_factory_bench(const char* const backend) {
        constexpr size_t kCount = 87040;
        constexpr int kIters = 200;
        double ones_us = 0.0;
        double bool_us = 0.0;
        double arange_us = 0.0;
        std::vector<double> ones_samples;
        std::vector<double> bool_samples;
        std::vector<double> arange_samples;
        ones_samples.reserve(static_cast<size_t>(kIters));
        bool_samples.reserve(static_cast<size_t>(kIters));
        arange_samples.reserve(static_cast<size_t>(kIters));
        for (int iter = 0; iter < kIters; ++iter) {
            const auto t0 = std::chrono::steady_clock::now();
            Tensor ones = Tensor::ones({kCount}, Device::CUDA, DataType::Int32);
            sync_tensor(ones);
            const auto t1 = std::chrono::steady_clock::now();
            Tensor flags = Tensor::full_bool({kCount}, true, Device::CUDA);
            sync_tensor(flags);
            const auto t2 = std::chrono::steady_clock::now();
            Tensor seq = Tensor::arange(0.0f, static_cast<float>(kCount));
            sync_tensor(seq);
            const auto t3 = std::chrono::steady_clock::now();
            const double ones_iter = elapsed_us(t0, t1);
            const double bool_iter = elapsed_us(t1, t2);
            const double arange_iter = elapsed_us(t2, t3);
            ones_us += ones_iter;
            bool_us += bool_iter;
            arange_us += arange_iter;
            ones_samples.push_back(ones_iter);
            bool_samples.push_back(bool_iter);
            arange_samples.push_back(arange_iter);
        }
        std::printf(
            "TensorAsyncConsts bench %s ones_int32=%.3f us full_bool=%.3f us arange=%.3f us\n",
            backend,
            ones_us / kIters,
            bool_us / kIters,
            arange_us / kIters);
        print_sample_stats(backend, "ones_int32", ones_samples);
        print_sample_stats(backend, "full_bool", bool_samples);
        print_sample_stats(backend, "arange", arange_samples);
    }

    Tensor random_group_mask(const size_t n, const uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, 7);
        Tensor cpu = Tensor::empty({n}, Device::CPU, DataType::UInt8);
        auto* const data = cpu.ptr<uint8_t>();
        for (size_t i = 0; i < n; ++i) {
            data[i] = static_cast<uint8_t>(dist(rng));
        }
        return cpu.to(Device::CUDA);
    }

    void print_hover_bench(const char* const backend) {
        constexpr size_t kCount = 87040;
        constexpr int kIters = 100;
        Tensor mask = random_group_mask(kCount, 20260906);
        Tensor scratch;
        std::array<int, 257> host{};
        double us = 0.0;
        for (int iter = 0; iter < kIters; ++iter) {
            const auto t0 = std::chrono::steady_clock::now();
            lfs::rendering::count_selection_groups_async(mask, scratch);
            lfs::rendering::SelectionCountTicket ticket;
            lfs::rendering::enqueue_selection_group_count_read(
                scratch, host.data(), nullptr, &ticket);
            while (!lfs::rendering::poll_selection_group_count_readback(ticket, host.data())) {
            }
            const auto t1 = std::chrono::steady_clock::now();
            us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        std::printf("TensorAsyncConsts hover %s %.3f us/call\n", backend, us / kIters);
    }

} // namespace

TEST(TensorAsyncConsts, FullMatchesCpuOnDefaultBackend) {
    SKIP_IF_BACKEND_UNAVAILABLE(default_gpu_backend());
    GpuBackendScope scope(GpuBackend::CUDA);
    run_full_suite();
}

TEST(TensorAsyncConsts, FullMatchesCpuOnVulkan) {
    SKIP_IF_BACKEND_UNAVAILABLE(GpuBackend::Vulkan);
    GpuBackendScope scope(GpuBackend::Vulkan);
    run_full_suite();
}

TEST(TensorAsyncConsts, ArangeMatchesCpuOnDefaultBackend) {
    SKIP_IF_BACKEND_UNAVAILABLE(default_gpu_backend());
    GpuBackendScope scope(GpuBackend::CUDA);
    run_arange_suite();
}

TEST(TensorAsyncConsts, ArangeMatchesCpuOnVulkan) {
    SKIP_IF_BACKEND_UNAVAILABLE(GpuBackend::Vulkan);
    GpuBackendScope scope(GpuBackend::Vulkan);
    run_arange_suite();
}

TEST(TensorAsyncConsts, TrimLeavesInFlightWorkReadable) {
    SKIP_IF_BACKEND_UNAVAILABLE(default_gpu_backend());
    GpuBackendScope scope(GpuBackend::CUDA);
    Tensor live = Tensor::full({4096}, 3.5f, Device::CUDA, DataType::Float32);
    Tensor::trim_memory_pool();
    expect_bytes_equal(live, Tensor::full({4096}, 3.5f, Device::CPU, DataType::Float32),
                       "cuda trim live");
}

TEST(TensorAsyncConsts, TrimLeavesInFlightWorkReadableOnVulkan) {
    SKIP_IF_BACKEND_UNAVAILABLE(GpuBackend::Vulkan);
    GpuBackendScope scope(GpuBackend::Vulkan);
    Tensor live = Tensor::ones({87040}, Device::CUDA, DataType::Int32);
    Tensor::trim_memory_pool();
    expect_bytes_equal(live, Tensor::ones({87040}, Device::CPU, DataType::Int32),
                       "vulkan trim live");
}

TEST(TensorAsyncConsts, FactoryMicrobenchDefaultBackend) {
    SKIP_IF_BACKEND_UNAVAILABLE(default_gpu_backend());
    GpuBackendScope scope(GpuBackend::CUDA);
    print_factory_bench("cuda");
}

TEST(TensorAsyncConsts, FactoryMicrobenchVulkan) {
    SKIP_IF_BACKEND_UNAVAILABLE(GpuBackend::Vulkan);
    GpuBackendScope scope(GpuBackend::Vulkan);
    print_factory_bench("vulkan");
}

TEST(TensorAsyncConsts, HoverMicrobenchVulkan) {
    SKIP_IF_BACKEND_UNAVAILABLE(GpuBackend::Vulkan);
    GpuBackendScope scope(GpuBackend::Vulkan);
    print_hover_bench("vulkan");
}

TEST(SelectionGroupCount, AsyncReadbackMatchesCountSelectionGroupsOnVulkan) {
    SKIP_IF_BACKEND_UNAVAILABLE(GpuBackend::Vulkan);
    GpuBackendScope scope(GpuBackend::Vulkan);
    constexpr size_t n = 87040;
    Tensor mask = random_group_mask(n, 20260906);
    Tensor sync_scratch;
    const auto expected = lfs::rendering::count_selection_groups(mask, sync_scratch);

    Tensor scratch;
    lfs::rendering::count_selection_groups_async(mask, scratch);
    std::array<int, 257> host{};
    lfs::rendering::SelectionCountTicket ticket;
    lfs::rendering::enqueue_selection_group_count_read(
        scratch, host.data(), nullptr, &ticket);
    while (!lfs::rendering::poll_selection_group_count_readback(ticket, host.data())) {
    }
    for (size_t group = 0; group < 256; ++group) {
        EXPECT_EQ(static_cast<size_t>(std::max(host[group], 0)), expected[group])
            << "group " << group;
    }
}
