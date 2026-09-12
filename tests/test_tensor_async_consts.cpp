/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "rendering/selection_ops.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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
        const Tensor gpu = Tensor::full(shape, value, Device::GPU, dtype);
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

        expect_bytes_equal(Tensor::ones({16}, Device::GPU, DataType::Int32),
                           Tensor::ones({16}, Device::CPU, DataType::Int32),
                           "ones Int32");
        expect_bytes_equal(Tensor::zeros({16}, Device::GPU, DataType::Int64),
                           Tensor::zeros({16}, Device::CPU, DataType::Int64),
                           "zeros Int64");
        expect_bytes_equal(Tensor::full_bool({16}, true, Device::GPU),
                           Tensor::full_bool({16}, true, Device::CPU),
                           "full_bool true");
        expect_bytes_equal(Tensor::full_bool({16}, false, Device::GPU),
                           Tensor::full_bool({16}, false, Device::CPU),
                           "full_bool false");
        expect_bytes_equal(Tensor::ones({16}, Device::GPU, DataType::Float16),
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
        const Tensor gpu = load_arange(start, end, step, Device::GPU, dtype);
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

    Tensor random_group_mask(const size_t n, const uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, 7);
        Tensor cpu = Tensor::empty({n}, Device::CPU, DataType::UInt8);
        auto* const data = cpu.ptr<uint8_t>();
        for (size_t i = 0; i < n; ++i) {
            data[i] = static_cast<uint8_t>(dist(rng));
        }
        return cpu.to(Device::GPU);
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
    Tensor live = Tensor::full({4096}, 3.5f, Device::GPU, DataType::Float32);
    Tensor::trim_memory_pool();
    expect_bytes_equal(live, Tensor::full({4096}, 3.5f, Device::CPU, DataType::Float32),
                       "cuda trim live");
}

TEST(TensorAsyncConsts, TrimLeavesInFlightWorkReadableOnVulkan) {
    SKIP_IF_BACKEND_UNAVAILABLE(GpuBackend::Vulkan);
    GpuBackendScope scope(GpuBackend::Vulkan);
    Tensor live = Tensor::ones({87040}, Device::GPU, DataType::Int32);
    Tensor::trim_memory_pool();
    expect_bytes_equal(live, Tensor::ones({87040}, Device::CPU, DataType::Int32),
                       "vulkan trim live");
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
