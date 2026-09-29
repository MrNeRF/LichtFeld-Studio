/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_execution.hpp"
#include "lfs/training/ops/pair_sort_vulkan.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;

    struct Pair {
        uint64_t key;
        uint32_t value;
    };

    std::pair<Tensor, Tensor> upload_pairs(const std::vector<Pair>& pairs,
                                           const bool wide_keys) {
        std::vector<uint32_t> values(pairs.size());
        for (size_t index = 0; index < pairs.size(); ++index) {
            values[index] = pairs[index].value;
        }
        Tensor device_keys;
        if (wide_keys) {
            std::vector<int64_t> wide(pairs.size());
            std::transform(pairs.begin(), pairs.end(), wide.begin(),
                           [](const Pair& pair) { return std::bit_cast<int64_t>(pair.key); });
            const Tensor host_keys = Tensor::from_blob(
                wide.data(), {pairs.size()}, Device::CPU, DataType::Int64);
            device_keys = host_keys.to(Device::GPU);
        } else {
            std::vector<uint32_t> narrow(pairs.size());
            std::transform(pairs.begin(), pairs.end(), narrow.begin(),
                           [](const Pair& pair) { return static_cast<uint32_t>(pair.key); });
            const Tensor host_keys = Tensor::from_blob(
                narrow.data(), {pairs.size()}, Device::CPU, DataType::UInt32);
            device_keys = host_keys.to(Device::GPU);
        }
        Tensor host_values = Tensor::from_blob(values.data(), {pairs.size()}, Device::CPU, DataType::UInt32);
        return {std::move(device_keys), host_values.to(Device::GPU)};
    }

    uint64_t window_key(const uint64_t key, const uint32_t begin_bit,
                        const uint32_t end_bit) {
        const uint32_t width = end_bit - begin_bit;
        const uint64_t mask = width == 64 ? ~uint64_t{0} : (uint64_t{1} << width) - 1;
        return (key >> begin_bit) & mask;
    }

    void verify_pairs(std::vector<Pair> input, const bool wide_keys,
                      const uint32_t begin_bit, const uint32_t end_bit,
                      const bool report_rate = false) {
        const uint32_t count = static_cast<uint32_t>(input.size());
        auto pair_less = [&](const Pair& left, const Pair& right) {
            return window_key(left.key, begin_bit, end_bit) <
                   window_key(right.key, begin_bit, end_bit);
        };

        auto [keys_a, values_a] = upload_pairs(input, wide_keys);
        std::stable_sort(input.begin(), input.end(), pair_less);
        Tensor keys_b = Tensor::empty({count}, Device::GPU,
                                      wide_keys ? DataType::Int64 : DataType::UInt32);
        Tensor values_b = Tensor::empty({count}, Device::GPU, DataType::UInt32);
        if (report_rate) {
            const Tensor original_keys = keys_a.clone(), original_values = values_a.clone();
            const auto warmup_start = std::chrono::steady_clock::now();
            do {
                keys_a.copy_(original_keys);
                values_a.copy_(original_values);
                (void)lfs::training::vulkan_pair_sort({&keys_a, &keys_b, &values_a, &values_b},
                                                      count, begin_bit, end_bit, wide_keys);
                lfs::core::TensorExecutionTarget::current().wait();
            } while (std::chrono::steady_clock::now() - warmup_start < std::chrono::milliseconds(500));
            keys_a.copy_(original_keys);
            values_a.copy_(original_values);
            lfs::core::TensorExecutionTarget::current().wait();
        }
        std::vector<lfs::training::PairSortPassTimings> pass_timings;
        const auto start = std::chrono::steady_clock::now();
        const bool in_a = lfs::training::vulkan_pair_sort(
            {&keys_a, &keys_b, &values_a, &values_b}, count,
            begin_bit, end_bit, wide_keys, report_rate ? &pass_timings : nullptr);
        lfs::core::TensorExecutionTarget::current().wait();
        const auto stop = std::chrono::steady_clock::now();
        const Tensor& sorted_keys = in_a ? keys_a : keys_b;
        const Tensor& sorted_values = in_a ? values_a : values_b;
        const Tensor host_keys = sorted_keys.to(Device::CPU);
        const Tensor host_values = sorted_values.to(Device::CPU);
        const auto* values = static_cast<const uint32_t*>(host_values.data_ptr());
        ASSERT_NE(values, nullptr);
        if (wide_keys) {
            const auto* keys = static_cast<const int64_t*>(host_keys.data_ptr());
            ASSERT_NE(keys, nullptr);
            for (size_t index = 0; index < input.size(); ++index) {
                if (std::bit_cast<uint64_t>(keys[index]) != input[index].key ||
                    values[index] != input[index].value) {
                    ADD_FAILURE() << "pair mismatch at " << index << ": key "
                                  << std::bit_cast<uint64_t>(keys[index]) << " vs "
                                  << input[index].key << ", payload " << values[index]
                                  << " vs " << input[index].value << " (window "
                                  << begin_bit << ".." << end_bit << ", wide="
                                  << wide_keys << ')';
                    return;
                }
            }
        } else {
            const auto* keys = static_cast<const uint32_t*>(host_keys.data_ptr());
            ASSERT_NE(keys, nullptr);
            for (size_t index = 0; index < input.size(); ++index) {
                if (keys[index] != static_cast<uint32_t>(input[index].key) ||
                    values[index] != input[index].value) {
                    ADD_FAILURE() << "pair mismatch at " << index << ": key "
                                  << keys[index] << " vs "
                                  << static_cast<uint32_t>(input[index].key)
                                  << ", payload " << values[index] << " vs "
                                  << input[index].value << " (window " << begin_bit
                                  << ".." << end_bit << ", wide=" << wide_keys << ')';
                    return;
                }
            }
        }
        if (report_rate) {
            const double elapsed = std::chrono::duration<double>(stop - start).count();
            double gpu_elapsed_ms = 0.0;
            for (const auto& timing : pass_timings) {
                gpu_elapsed_ms += timing.histogram_ms + timing.partition_scan_ms +
                                  timing.histogram_reduce_ms + timing.digit_base_scan_ms +
                                  timing.scatter_ms;
            }
            const double measured_ms = gpu_elapsed_ms > 0.0 ? gpu_elapsed_ms : elapsed * 1000.0;
            std::cout << "PAIR_SORT " << count << " pairs " << measured_ms << " gpu_ms "
                      << elapsed * 1000.0 << " host_ms "
                      << (static_cast<double>(count) / (measured_ms / 1000.0) /
                          1'000'000'000.0)
                      << " Gpairs/s key_bits=" << (wide_keys ? 64 : 32) << " window=" << begin_bit << ".." << end_bit << "\n";
            for (size_t pass = 0; pass < pass_timings.size(); ++pass) {
                const auto& timing = pass_timings[pass];
                std::cout << "PAIR_PASS " << count << ' ' << pass << " histogram "
                          << timing.histogram_ms << " ms histogram_reduce "
                          << timing.histogram_reduce_ms << " ms digit_base_scan "
                          << timing.digit_base_scan_ms << " ms partition_scan "
                          << timing.partition_scan_ms << " ms scatter "
                          << timing.scatter_ms << " ms\n";
            }
        }
    }

    uint32_t ordered_float(const float value) {
        const uint32_t bits = std::bit_cast<uint32_t>(value);
        return (bits & 0x8000'0000u) != 0 ? ~bits : (bits ^ 0x8000'0000u);
    }

    uint32_t fast_depth_key(const uint32_t tile_id, const float camera_z,
                            const uint32_t tile_count) {
        const uint32_t tile_bits = tile_count <= 1
                                       ? 0
                                       : 32u - std::countl_zero(tile_count - 1);
        const uint32_t depth_bits = std::min(23u, 32u - tile_bits);
        if (depth_bits == 0)
            return tile_id << depth_bits;
        const float transformed = std::clamp((2.0f * camera_z + 1.0f) /
                                                 (camera_z + 1.0f),
                                             1.0f, std::nextafter(2.0f, 1.0f));
        const uint32_t mantissa = std::bit_cast<uint32_t>(transformed) & 0x007f'ffffu;
        const uint32_t depth = mantissa >> (23u - depth_bits);
        return (tile_id << depth_bits) | depth;
    }

    std::vector<Pair> make_fast_pairs(const uint32_t count) {
        constexpr uint32_t tile_count = 1u << 16;
        std::mt19937 generator(0x1939u + count);
        std::uniform_int_distribution<uint32_t> tile(0, tile_count - 1);
        std::uniform_real_distribution<float> depth(0.01f, 1.0e6f);
        std::vector<Pair> pairs(count);
        for (uint32_t index = 0; index < count; ++index) {
            pairs[index] = {fast_depth_key(tile(generator), depth(generator), tile_count), index};
        }
        return pairs;
    }

    std::vector<Pair> make_random_pairs(const uint32_t count, const bool wide_keys) {
        std::mt19937_64 generator(0x9e37'79b9'7f4a'7c15ull + count + wide_keys);
        std::vector<Pair> pairs(count);
        for (uint32_t index = 0; index < count; ++index) {
            uint64_t key = generator();
            if (!wide_keys)
                key = static_cast<uint32_t>(key);
            pairs[index] = {key, index};
        }
        return pairs;
    }
} // namespace

TEST(TrainingVulkanPairSort, SortsWindowsStablyForThirtyTwoAndSixtyFourBitKeys) {
    if (!lfs::core::gpu_backend_available(GpuBackend::Vulkan))
        GTEST_SKIP() << "Vulkan tensor backend unavailable";
    const lfs::core::GpuBackendScope scope(GpuBackend::Vulkan);
    const std::vector<Pair> fast_pairs{{0xffff'ffffu, 0}, {0, 1}, {1, 2}, {0x7fff'ffffu, 3}, {1, 4}};
    verify_pairs(fast_pairs, false, 0, 32);
    verify_pairs(fast_pairs, false, 4, 28);
    verify_pairs({{0, 0}, {0x8000'0000u, 1}, {0x7fff'ffffu, 2}, {0x8000'0000u, 3}},
                 false, 31, 32);
    verify_pairs({{(7ull << 32) | 0x3f80'0000u, 0},
                  {(2ull << 32) | 0x4000'0000u, 1},
                  {(7ull << 32) | 0x3f00'0000u, 2},
                  {(2ull << 32) | 0x3f80'0000u, 3},
                  {(7ull << 32) | 0x3f80'0000u, 4}},
                 true, 0, 64);
    verify_pairs({{0x0000'0001'0000'0000ull, 0},
                  {0x0000'0002'0000'0000ull, 1},
                  {0x0000'0001'0000'0000ull, 2},
                  {0x0000'0000'0000'0001ull, 3}},
                 true, 32, 48);
    const std::vector<float> samples{-3.0f, -0.0f, 0.0f, 1.0f, 8.0f};
    std::vector<Pair> ascending;
    std::vector<Pair> descending;
    for (uint32_t index = 0; index < samples.size(); ++index) {
        const uint32_t key = ordered_float(samples[index]);
        ascending.push_back({key, index});
        descending.push_back({~key, index});
    }
    verify_pairs(ascending, false, 0, 32);
    verify_pairs(descending, false, 0, 32);
}

TEST(TrainingVulkanPairSort, CoversEmptySingletonAndPartitionBoundaries) {
    if (!lfs::core::gpu_backend_available(GpuBackend::Vulkan))
        GTEST_SKIP() << "Vulkan tensor backend unavailable";
    const lfs::core::GpuBackendScope scope(GpuBackend::Vulkan);
    Tensor empty_keys_a = Tensor::empty({0}, Device::GPU, DataType::UInt32);
    Tensor empty_keys_b = Tensor::empty({0}, Device::GPU, DataType::UInt32);
    Tensor empty_values_a = Tensor::empty({0}, Device::GPU, DataType::UInt32);
    Tensor empty_values_b = Tensor::empty({0}, Device::GPU, DataType::UInt32);
    EXPECT_TRUE(lfs::training::vulkan_pair_sort(
        {&empty_keys_a, &empty_keys_b, &empty_values_a, &empty_values_b}, 0, 0, 32, false));
    verify_pairs({{7, 0}}, false, 0, 32);
    for (const uint32_t count : {127u, 128u, 2047u, 2048u, 2049u, 4097u}) {
        std::vector<Pair> pairs(count);
        for (uint32_t index = 0; index < count; ++index) {
            pairs[index] = {static_cast<uint32_t>((index * 37u) % 19u), index};
        }
        verify_pairs(std::move(pairs), false, 0, 5);
    }
    for (const uint32_t count : {100u, 2047u, 2048u, 2049u, 4097u}) {
        verify_pairs(make_random_pairs(count, true), true, 0, 64);
        verify_pairs(make_random_pairs(count, true), true, 11, 49);
    }
}

TEST(TrainingVulkanPairSort, MatchesFastQuantizedKeysAndReportsThroughput) {
    if (!lfs::core::gpu_backend_available(GpuBackend::Vulkan))
        GTEST_SKIP() << "Vulkan tensor backend unavailable";
    const lfs::core::GpuBackendScope scope(GpuBackend::Vulkan);
    verify_pairs(make_fast_pairs(1'000'000), false, 0, 32, true);
    verify_pairs(make_fast_pairs(10'000'000), false, 0, 32, true);
    constexpr std::array<uint32_t, 4> sizes{1'000'000, 4'000'000, 10'000'000, 32'000'000};
    for (const uint32_t size : sizes) {
        for (const bool wide_keys : {false, true}) {
            const uint32_t end_bit = wide_keys ? 64u : 32u;
            const uint32_t begin_bit = wide_keys ? 11u : 5u;
            const uint32_t partial_end = wide_keys ? 49u : 26u;
            verify_pairs(make_random_pairs(size, wide_keys), wide_keys, 0, end_bit, true);
            verify_pairs(make_random_pairs(size, wide_keys), wide_keys,
                         begin_bit, partial_end, true);
        }
    }
}

TEST(TrainingVulkanPairSort, FastKeyThroughput) {
    if (!lfs::core::gpu_backend_available(GpuBackend::Vulkan))
        GTEST_SKIP();
    const lfs::core::GpuBackendScope scope(GpuBackend::Vulkan);
    verify_pairs(make_fast_pairs(10'000'000), false, 0, 32, true);
}
