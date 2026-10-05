/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/backend/facade_trace.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor_backend.hpp"
#include <array>
#include <chrono>
#include <limits>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

using namespace lfs::core;

TEST(TensorOrderingTest, Float32SortReturnsValuesAndSourceIndices) {
    const auto input = Tensor::from_vector(
        std::vector<float>{3.0f, 1.0f, 4.0f, 2.0f}, {4}, Device::GPU);

    const auto [ascending, ascending_indices] = input.sort(0);
    EXPECT_EQ(ascending_indices.dtype(), DataType::Int64);
    EXPECT_EQ(ascending.cpu().to_vector(), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
    EXPECT_EQ(ascending_indices.cpu().to_vector_int64(),
              (std::vector<int64_t>{1, 3, 0, 2}));

    const auto [descending, descending_indices] = input.sort(0, true);
    EXPECT_EQ(descending_indices.dtype(), DataType::Int64);
    EXPECT_EQ(descending.cpu().to_vector(), (std::vector<float>{4.0f, 3.0f, 2.0f, 1.0f}));
    EXPECT_EQ(descending_indices.cpu().to_vector_int64(),
              (std::vector<int64_t>{2, 0, 3, 1}));
}

TEST(TensorOrderingTest, SortKeepsTheOrderOfEqualKeysOnEveryDevice) {
    // Short lines sort in one block, long ones across blocks.
    for (const size_t count : {size_t(1000), size_t(200000)}) {
        std::vector<float> keys(count);
        for (size_t i = 0; i < count; ++i)
            keys[i] = static_cast<float>((i * 7919) % 13);
        for (const auto device : {Device::CPU, Device::GPU}) {
            for (const bool descending : {false, true}) {
                const auto order = Tensor::from_vector(keys, {count}, device).sort(0, descending).second.cpu().to_vector_int64();
                ASSERT_EQ(order.size(), count);
                for (size_t i = 1; i < count; ++i) {
                    const float previous = keys[static_cast<size_t>(order[i - 1])];
                    const float current = keys[static_cast<size_t>(order[i])];
                    ASSERT_TRUE(descending ? previous >= current : previous <= current) << i;
                    if (previous == current)
                        ASSERT_LT(order[i - 1], order[i]) << "count " << count << " position " << i;
                }
            }
        }
    }
}

TEST(TensorOrderingTest, CpuSortPreservesInt64IndicesAcrossRanks) {
    const auto matrix = Tensor::from_vector(
        std::vector<float>{5.0f, 2.0f, 8.0f,
                           1.0f, 9.0f, 3.0f,
                           7.0f, 4.0f, 6.0f},
        {3, 3}, Device::CPU);
    const auto [column_values, column_indices] = matrix.sort(0);
    EXPECT_EQ(column_values.to_vector(),
              (std::vector<float>{1.0f, 2.0f, 3.0f,
                                  5.0f, 4.0f, 6.0f,
                                  7.0f, 9.0f, 8.0f}));
    EXPECT_EQ(column_indices.dtype(), DataType::Int64);
    EXPECT_EQ(column_indices.to_vector_int64(),
              (std::vector<int64_t>{1, 0, 1, 0, 2, 2, 2, 1, 0}));

    const auto volume = Tensor::from_vector(
        std::vector<float>{3.0f, 1.0f, 2.0f,
                           6.0f, 4.0f, 5.0f,
                           9.0f, 7.0f, 8.0f,
                           0.0f, -2.0f, -1.0f},
        {2, 2, 3}, Device::CPU);
    const auto [values, indices] = volume.sort(2);
    EXPECT_EQ(values.to_vector(),
              (std::vector<float>{1.0f, 2.0f, 3.0f,
                                  4.0f, 5.0f, 6.0f,
                                  7.0f, 8.0f, 9.0f,
                                  -2.0f, -1.0f, 0.0f}));
    EXPECT_EQ(indices.shape(), volume.shape());
    EXPECT_EQ(indices.dtype(), DataType::Int64);
    EXPECT_EQ(indices.to_vector_int64(),
              (std::vector<int64_t>{1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0}));
}

TEST(TensorOrderingTest, Float32SortAlongSecondDimension) {
    const auto input = Tensor::from_vector(
        std::vector<float>{3.0f, 1.0f, 2.0f, 4.0f, 6.0f, 5.0f},
        {2, 3}, Device::GPU);

    const auto [values, indices] = input.sort(1, true);
    EXPECT_EQ(values.cpu().to_vector(),
              (std::vector<float>{3.0f, 2.0f, 1.0f, 6.0f, 5.0f, 4.0f}));
    EXPECT_EQ(indices.cpu().to_vector_int64(),
              (std::vector<int64_t>{0, 2, 1, 1, 2, 0}));
}

TEST(TensorOrderingTest, MinMaxWithIndicesReturnValuesAndLocations) {
    for (const auto device : {Device::CPU, Device::GPU}) {
        const auto input = Tensor::from_vector(
            std::vector<float>{3.0f, 1.0f, 2.0f, 4.0f, 6.0f, 5.0f},
            {2, 3}, device);

        const auto [min_values, min_indices] = input.min_with_indices(1);
        const auto [max_values, max_indices] = input.max_with_indices(1);

        EXPECT_EQ(min_values.cpu().to_vector(), (std::vector<float>{1.0f, 4.0f}));
        EXPECT_EQ(min_indices.dtype(), DataType::Int64);
        EXPECT_EQ(min_indices.cpu().to_vector_int64(), (std::vector<int64_t>{1, 0}));
        EXPECT_EQ(max_values.cpu().to_vector(), (std::vector<float>{3.0f, 6.0f}));
        EXPECT_EQ(max_indices.dtype(), DataType::Int64);
        EXPECT_EQ(max_indices.cpu().to_vector_int64(), (std::vector<int64_t>{0, 1}));
    }
}

TEST(TensorDistanceTest, CdistL1AndL2HaveExactValues) {
    const auto lhs = Tensor::from_vector(
        std::vector<float>{0.0f, 0.0f, 3.0f, 4.0f}, {2, 2}, Device::GPU);
    const auto rhs = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, -2.0f, 0.0f}, {2, 2}, Device::GPU);

    const auto l1 = lhs.cdist(rhs, 1.0f).cpu().to_vector();
    EXPECT_EQ(l1, (std::vector<float>{3.0f, 2.0f, 4.0f, 9.0f}));

    const auto l2 = lhs.cdist(rhs, 2.0f).cpu().to_vector();
    ASSERT_EQ(l2.size(), 4u);
    EXPECT_NEAR(l2[0], std::sqrt(5.0f), 1e-5f);
    EXPECT_NEAR(l2[1], 2.0f, 1e-5f);
    EXPECT_NEAR(l2[2], std::sqrt(8.0f), 1e-5f);
    EXPECT_NEAR(l2[3], std::sqrt(41.0f), 1e-5f);
}

TEST(TensorOrderingTest, ArgExtremeKernelMatchesCpu) {
    struct Case {
        std::vector<size_t> shape;
        int axis;
    };
    const std::vector<Case> cases = {
        {{1, 1000003}, 1},
        {{3000, 700}, 1},
        {{5, 10000, 3}, 1},
        {{10000, 4}, 0},
        {{37}, 0},
        {{4, 1, 7}, 1},
        {{3, 4097}, 1}};
    for (const auto& [shape, axis] : cases) {
        SCOPED_TRACE(TensorShape(shape).str());
        size_t count = 1;
        for (const size_t extent : shape)
            count *= extent;
        std::vector<float> data(count);
        for (size_t i = 0; i < count; ++i) {
            data[i] = static_cast<float>(static_cast<int>(i * 7919 % 101) - 50);
            if (i % 7 == 0)
                data[i] = i % 2 ? -0.0f : 0.0f;
            if (i % 997 == 0)
                data[i] = std::numeric_limits<float>::quiet_NaN();
            if (i % 991 == 0)
                data[i] = i % 2 ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        }
        const Tensor cpu = Tensor::from_vector(data, TensorShape(shape), Device::CPU);
        const Tensor gpu = cpu.to(Device::GPU);
        for (const bool maximum : {false, true}) {
            for (const bool keepdim : {false, true}) {
                const auto [expected_values, expected_indices] = maximum ? cpu.max_with_indices(axis, keepdim)
                                                                         : cpu.min_with_indices(axis, keepdim);
                internal::facade_trace_enable_for_testing(true);
                const auto before = internal::facade_trace_snapshot_for_testing();
                const auto [values, indices] = maximum ? gpu.max_with_indices(axis, keepdim)
                                                       : gpu.min_with_indices(axis, keepdim);
                const auto after = internal::facade_trace_snapshot_for_testing();
                internal::facade_trace_enable_for_testing(false);
                const size_t entry = static_cast<size_t>(internal::FacadeEntry::arg_extreme);
                EXPECT_EQ(after[entry] - before[entry], 1u);
                EXPECT_EQ(indices.shape(), expected_indices.shape());
                EXPECT_EQ(indices.cpu().to_vector_int64(), expected_indices.to_vector_int64());
                const auto found = values.cpu().to_vector();
                const auto expected = expected_values.to_vector();
                ASSERT_EQ(found.size(), expected.size());
                for (size_t i = 0; i < found.size(); ++i) {
                    if (std::isnan(expected[i]))
                        ASSERT_TRUE(std::isnan(found[i])) << i;
                    else {
                        ASSERT_EQ(found[i], expected[i]) << i;
                        if (found[i] == 0.0f)
                            EXPECT_EQ(std::signbit(found[i]), std::signbit(expected[i]));
                    }
                }
            }
        }
    }
    // Offset and transposed views must materialize before kernel dispatch.
    const Tensor view = Tensor::from_vector(std::vector<float>{9, 2, 2, -0.0f, 0.0f, 3, 1, 8, 8, 4, 5, 6},
                                            {4, 3}, Device::GPU)
                            .slice(0, 1, 4)
                            .transpose(0, 1);
    EXPECT_EQ(view.argmax(std::array{1}).cpu().to_vector_int64(), view.cpu().argmax(std::array{1}).to_vector_int64());
    EXPECT_EQ(view.argmin(std::array{0}).cpu().to_vector_int64(), view.cpu().argmin(std::array{0}).to_vector_int64());
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const Tensor ties = Tensor::from_vector(std::vector<float>{-0.f, 0.f, -0.f, 0.f,
                                                               nan, 1.f, nan, nan,
                                                               nan, 1.f, 2.f, 3.f,
                                                               1.f, 3.f, 3.f, 1.f},
                                            {4, 4}, Device::GPU);
    const auto [max_values, max_indices] = ties.max_with_indices(1);
    const auto [min_values, min_indices] = ties.min_with_indices(1);
    EXPECT_EQ(max_indices.cpu().to_vector_int64(), (std::vector<int64_t>{0, 2, 0, 1}));
    EXPECT_EQ(min_indices.cpu().to_vector_int64(), (std::vector<int64_t>{0, 2, 0, 0}));
    EXPECT_TRUE(std::signbit(max_values.cpu().to_vector()[0]));
    EXPECT_TRUE(std::signbit(min_values.cpu().to_vector()[0]));
    EXPECT_EQ(ties.argmax().cpu().to_vector_int64(), (std::vector<int64_t>{4}));
    EXPECT_EQ(ties.argmin().cpu().to_vector_int64(), (std::vector<int64_t>{4}));
}

TEST(TensorOrderingTest, DISABLED_ArgExtremeTiming) {
    for (const auto& shape : std::vector<std::vector<size_t>>{{65536, 112}, {1, 4194304}, {65536, 448}}) {
        const Tensor x = Tensor::rand(TensorShape(shape), Device::GPU);
        auto& backend = internal::backend_ops_for(x);
        for (int i = 0; i < 3; ++i)
            (void)x.max_with_indices(1);
        backend.synchronize_device();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i)
            (void)x.max_with_indices(1);
        backend.synchronize_device();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 20;
        std::cout << "arg_extreme " << TensorShape(shape).str() << " " << ms << " ms\n";
    }
}

TEST(TensorOrderingTest, DISABLED_SortTiming) {
    for (const size_t count : {size_t(1) << 20, size_t(5) << 20}) {
        const Tensor random = Tensor::rand({count}, Device::GPU);
        // Small non-negative integers as biased floats, the shape of label keys.
        const Tensor labels = (random * 1048576.0f).floor() + 8388608.0f;
        auto& backend = internal::backend_ops_for(random);
        for (const auto* keys : {&random, &labels}) {
            for (int i = 0; i < 3; ++i)
                (void)keys->sort(0);
            backend.synchronize_device();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 10; ++i)
                (void)keys->sort(0);
            backend.synchronize_device();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 10;
            std::cout << "sort " << count << (keys == &random ? " uniform " : " labels ") << ms << " ms\n";
        }
    }
}

TEST(TensorOrderingTest, DISABLED_GroupingPrimitiveTiming) {
    const size_t count = size_t(5) << 20;
    const Tensor random = Tensor::rand({count}, Device::GPU);
    // Node evaluation holds freed buffers for reuse; measure the same way, once the backend is live.
    Tensor::hold_freed_memory();
    const Tensor ints = (random * 1000.0f).to(DataType::Int32);
    const Tensor mask = random.gt(0.5f);
    const Tensor order = random.sort(0).second.to(DataType::Int32);
    auto& backend = internal::backend_ops_for(random);
    const auto time = [&](const char* name, auto&& run) {
        for (int i = 0; i < 3; ++i)
            run();
        backend.synchronize_device();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 10; ++i)
            run();
        backend.synchronize_device();
        std::cout << name << " " << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 10 << " ms\n";
    };
    time("cumsum_int32", [&] { (void)ints.cumsum(0); });
    time("nonzero", [&] { (void)mask.nonzero(); });
    time("index_select_int32", [&] { (void)ints.index_select(0, order); });
    time("index_copy_int32", [&] { auto out = Tensor::zeros({count}, Device::GPU, DataType::Int32); out.index_copy_(0, order, ints); });
    time("item", [&] { (void)ints.slice(0, 0, 1).item<int>(); });
    time("any_item", [&] { (void)mask.any().item<bool>(); });
    Tensor::release_freed_memory();
}
