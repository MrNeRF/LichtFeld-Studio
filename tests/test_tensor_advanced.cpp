/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/pinned_memory_allocator.hpp"
#include "core/tensor.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace lfs::core;

TEST(TensorAdvancedTest, LinspaceIncludesEndpointsAndZeroStepsAreEmpty) {
    const auto values = Tensor::linspace(-1.0f, 1.0f, 5, Device::CUDA).cpu().to_vector();
    EXPECT_EQ(values, (std::vector<float>{-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}));
    const auto empty = Tensor::linspace(0.0f, 1.0f, 0, Device::CUDA);
    EXPECT_EQ(empty.shape(), TensorShape({0}));
    EXPECT_TRUE(empty.is_empty());
}

TEST(TensorAdvancedTest, ArangeCreatesOnRequestedDevice) {
    const auto host = Tensor::arange(0.0f, 2.0f, 0.5f, Device::CPU);
    EXPECT_EQ(host.device(), Device::CPU);
    EXPECT_EQ(host.to_vector(), (std::vector<float>{0.0f, 0.5f, 1.0f, 1.5f}));

    const auto empty_host = Tensor::arange(3.0f, -2.0f, 1.0f, Device::CPU);
    EXPECT_EQ(empty_host.device(), Device::CPU);
    EXPECT_EQ(empty_host.shape(), TensorShape({0}));

    const auto device = Tensor::arange(0.0f, 2.0f, 0.5f, Device::CUDA);
    EXPECT_EQ(device.device(), Device::CUDA);
    EXPECT_EQ(device.cpu().to_vector(), host.to_vector());
}

namespace {
    constexpr int64_t kFloat32IntegerLimit = int64_t{1} << 24;

    std::vector<int64_t> integer_values(const Tensor& tensor) {
        const auto host = tensor.cpu();
        if (host.dtype() == DataType::Int64)
            return host.to_vector_int64();
        const auto values = host.to_vector_int();
        return {values.begin(), values.end()};
    }

    void expect_integer_range(const Tensor& rows, const DataType dtype, const Device device,
                              const std::vector<int64_t>& expected) {
        ASSERT_EQ(rows.dtype(), dtype);
        ASSERT_EQ(rows.device(), device);
        ASSERT_EQ(rows.shape(), TensorShape({expected.size()}));
        EXPECT_EQ(integer_values(rows), expected);
    }

    const char* device_label(const Device device) {
        return device == Device::CUDA ? "cuda" : "cpu";
    }

    Tensor arange_from_float_bounds(const float start, const float end, const float step, const Device device) {
        LoadArgs args;
        args.shape = TensorShape{};
        args.device = device;
        args.dtype = DataType::Int32;
        args.args = std::tuple<float, float, float>{start, end, step};
        return Tensor::load(LoadOp::Arange, args);
    }

    template <typename Step>
    constexpr bool integer_arange_takes_step = requires(Step step) {
        Tensor::arange(int64_t{0}, int64_t{4}, step, Device::CPU, DataType::Int64);
    };
} // namespace

// Implicit narrowing would turn a step of 1.5 into 1.
static_assert(integer_arange_takes_step<int> && integer_arange_takes_step<int64_t> &&
              integer_arange_takes_step<size_t>);
static_assert(!integer_arange_takes_step<float> && !integer_arange_takes_step<double>);

// Through float32 the range [B, B+1) at B = 2^24 comes out empty and [B, B+4) as B + {0, 0, 2, 4}.
TEST(TensorAdvancedTest, IntegerArangeIsExactPastFloat32Precision) {
    constexpr int64_t b = kFloat32IntegerLimit;
    for (const DataType dtype : {DataType::Int32, DataType::Int64}) {
        for (const Device device : {Device::CPU, Device::CUDA}) {
            SCOPED_TRACE(::testing::Message() << dtype_name(dtype) << " on " << device_label(device));
            expect_integer_range(Tensor::arange(b, b + 1, 1, device, dtype), dtype, device, {b});
            expect_integer_range(Tensor::arange(b, b + 4, 1, device, dtype), dtype, device, {b, b + 1, b + 2, b + 3});
            expect_integer_range(Tensor::arange(b - 2, b + 3, 1, device, dtype), dtype, device,
                                 {b - 2, b - 1, b, b + 1, b + 2});
            expect_integer_range(Tensor::arange(3 * b + 1, 3 * b + 4, 1, device, dtype), dtype, device,
                                 {3 * b + 1, 3 * b + 2, 3 * b + 3});
            expect_integer_range(Tensor::arange(b + 5, b - 4, -3, device, dtype), dtype, device,
                                 {b + 5, b + 2, b - 1});
        }
    }
}

TEST(TensorAdvancedTest, IntegerArangeWritesByteResultsDirectly) {
    constexpr int64_t end = std::numeric_limits<int64_t>::max();
    for (const auto device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device_label(device));
        const auto before = PinnedMemoryAllocator::instance().get_stats();
        const auto bytes = Tensor::arange(end - 3, end, 1, device, DataType::UInt8);
        if (device == Device::CPU) {
            const auto after = PinnedMemoryAllocator::instance().get_stats();
            EXPECT_EQ((after.cache_hits - before.cache_hits) + (after.cache_misses - before.cache_misses), 1u);
        }
        EXPECT_EQ(bytes.device(), device);
        ASSERT_EQ(bytes.dtype(), DataType::UInt8);
        EXPECT_EQ(bytes.bytes(), 3u);
        EXPECT_EQ(bytes.cpu().to_vector_uint8(), (std::vector<uint8_t>{252, 253, 254}));
        const auto descending = Tensor::arange(1, -3, -1, device, DataType::UInt8);
        EXPECT_EQ(descending.cpu().to_vector_uint8(), (std::vector<uint8_t>{1, 0, 255, 254}));
    }
}

TEST(TensorAdvancedTest, IntegerArangeStepsAwayEmptyAndRejectsInvalidRanges) {
    for (const DataType dtype : {DataType::Int32, DataType::Int64}) {
        for (const Device device : {Device::CPU, Device::CUDA}) {
            SCOPED_TRACE(::testing::Message() << dtype_name(dtype) << " on " << device_label(device));
            expect_integer_range(Tensor::arange(int64_t{5}, int64_t{2}, 1, device, dtype), dtype, device, {});
            expect_integer_range(Tensor::arange(int64_t{2}, int64_t{5}, -1, device, dtype), dtype, device, {});
            expect_integer_range(Tensor::arange(int64_t{3}, int64_t{3}, 1, device, dtype), dtype, device, {});
            EXPECT_THROW(Tensor::arange(int64_t{0}, int64_t{4}, 0, device, dtype), std::runtime_error);
        }
    }
    EXPECT_THROW(Tensor::arange(int64_t{0}, int64_t{4}, 1, Device::CPU, DataType::Float32), std::runtime_error);
    constexpr int64_t int32_max = std::numeric_limits<int32_t>::max();
    expect_integer_range(Tensor::arange(int32_max, int32_max + 1, 1, Device::CPU, DataType::Int32), DataType::Int32,
                         Device::CPU, {int32_max});
    EXPECT_THROW(Tensor::arange(int32_max, int32_max + 2, 1, Device::CPU, DataType::Int32), std::runtime_error);
    EXPECT_THROW(Tensor::arange(int32_max - 1, int32_max + 2, 1, Device::CPU, DataType::Int32), std::runtime_error);
    expect_integer_range(Tensor::arange(int32_max - 1, int32_max + 2, 1, Device::CUDA, DataType::Int64),
                         DataType::Int64, Device::CUDA, {int32_max - 1, int32_max, int32_max + 1});

    constexpr int64_t int64_min = std::numeric_limits<int64_t>::min();
    constexpr int64_t int64_max = std::numeric_limits<int64_t>::max();
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device_label(device));
        expect_integer_range(Tensor::arange(int64_min, int64_max, int64_max, device, DataType::Int64),
                             DataType::Int64, device, {int64_min, -1, int64_max - 1});
        expect_integer_range(Tensor::arange(int64_max, int64_min, int64_min, device, DataType::Int64),
                             DataType::Int64, device, {int64_max, -1});
        EXPECT_THROW(Tensor::arange(int64_min, int64_max, 1, device, DataType::Int64), std::runtime_error);
    }
}

// Int32 values from whole float bounds were summed in float32: B + 1 came out as B.
TEST(TensorAdvancedTest, Int32ArangeFromWholeFloatBoundsIsExactPastFloat32Precision) {
    constexpr float start = static_cast<float>(kFloat32IntegerLimit);
    constexpr size_t count = 10;
    std::vector<int64_t> expected(count);
    for (size_t i = 0; i < count; ++i)
        expected[i] = kFloat32IntegerLimit + static_cast<int64_t>(i);
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device_label(device));
        expect_integer_range(arange_from_float_bounds(start, start + static_cast<float>(count), 1.0f, device),
                             DataType::Int32, device, expected);
    }
}

// Fractional bounds keep the float32 sum: 0.9f + 0.1f rounds to 1.0f, the double sum 0.99999998 truncates to 0.
TEST(TensorAdvancedTest, Int32ArangeFromFractionalFloatBoundsKeepsTheFloatSum) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device_label(device));
        expect_integer_range(arange_from_float_bounds(0.9f, 1.5f, 0.1f, device), DataType::Int32, device,
                             {0, 1, 1, 1, 1, 1, 1});
        expect_integer_range(arange_from_float_bounds(2.5f, -1.0f, -0.5f, device), DataType::Int32, device,
                             {2, 2, 1, 1, 0, 0, 0});
    }
}

TEST(TensorAdvancedTest, FractionalInt32ArangeEndsAtIntMax) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device_label(device));
        expect_integer_range(arange_from_float_bounds(2147483520.0f, 2147483648.0f, 63.5f, device),
                             DataType::Int32, device, {2147483520, 2147483520, 2147483647});
    }
}

TEST(TensorAdvancedTest, StackPreservesValuesAndRejectsEmptyInput) {
    const auto first = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f}, {2}, Device::CUDA);
    const auto second = Tensor::from_vector(
        std::vector<float>{3.0f, 4.0f}, {2}, Device::CUDA);
    const auto result = Tensor::stack({first, second}, 0);

    EXPECT_EQ(result.shape(), TensorShape({2, 2}));
    EXPECT_EQ(result.cpu().to_vector(), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
    EXPECT_THROW(Tensor::stack({}, 0), std::runtime_error);
}

TEST(TensorAdvancedTest, TryReshapeDistinguishesCompatibleShapes) {
    const auto input = Tensor::arange(12.0f).to(Device::CUDA);

    const auto compatible = input.try_reshape({3, 4});
    const auto incompatible = input.try_reshape({5, 3});

    ASSERT_TRUE(compatible.has_value());
    EXPECT_EQ(compatible->shape(), TensorShape({3, 4}));
    EXPECT_EQ(compatible->cpu().to_vector(), input.cpu().to_vector());
    EXPECT_FALSE(incompatible.has_value());
}

TEST(TensorAdvancedTest, SplitBatchPreservesTailAndValues) {
    const auto input = Tensor::arange(1000.0f).to(Device::CUDA).reshape({100, 10});

    const auto batches = Tensor::split_batch(input, 32);

    ASSERT_EQ(batches.size(), 4u);
    EXPECT_EQ(batches[0].shape(), TensorShape({32, 10}));
    EXPECT_EQ(batches[1].shape(), TensorShape({32, 10}));
    EXPECT_EQ(batches[2].shape(), TensorShape({32, 10}));
    EXPECT_EQ(batches[3].shape(), TensorShape({4, 10}));
    EXPECT_EQ(Tensor::cat(batches, 0).cpu().to_vector(), input.cpu().to_vector());
}

TEST(TensorAdvancedTest, ApplyAndInplaceChainsHaveDistinctOwnership) {
    auto input = Tensor::ones({4}, Device::CUDA);
    const auto applied = input.apply([](const Tensor& tensor) { return tensor.add(1.0f); })
                             .apply([](const Tensor& tensor) { return tensor.mul(2.0f); })
                             .apply([](const Tensor& tensor) { return tensor.sub(1.0f); });

    EXPECT_EQ(input.cpu().to_vector(), (std::vector<float>(4, 1.0f)));
    EXPECT_EQ(applied.cpu().to_vector(), (std::vector<float>(4, 3.0f)));

    input.inplace([](Tensor& tensor) { tensor.add_(1.0f); })
        .inplace([](Tensor& tensor) { tensor.mul_(2.0f); })
        .inplace([](Tensor& tensor) { tensor.sub_(1.0f); });
    EXPECT_EQ(input.cpu().to_vector(), (std::vector<float>(4, 3.0f)));
}

TEST(TensorAdvancedTest, SpecialValuesAreDetectedAndClamped) {
    auto tensor = Tensor::from_vector(
        std::vector<float>{
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            0.0f},
        {4}, Device::CUDA);

    EXPECT_TRUE(tensor.has_nan());
    EXPECT_TRUE(tensor.has_inf());
    EXPECT_THROW(tensor.assert_finite(), TensorError);

    const auto clamped = tensor.clamp(-10.0f, 10.0f);
    EXPECT_TRUE(clamped.has_nan());
    EXPECT_FALSE(clamped.has_inf());
    const auto values = clamped.cpu().to_vector();
    EXPECT_FLOAT_EQ(values[1], 10.0f);
    EXPECT_FLOAT_EQ(values[2], -10.0f);
}

TEST(TensorAdvancedTest, DiagPlacesValuesOnlyOnDiagonal) {
    const auto diagonal = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, 3.0f}, {3}, Device::CUDA);
    const auto matrix = Tensor::diag(diagonal);

    EXPECT_EQ(matrix.shape(), TensorShape({3, 3}));
    EXPECT_EQ(matrix.cpu().to_vector(),
              (std::vector<float>{1.0f, 0.0f, 0.0f,
                                  0.0f, 2.0f, 0.0f,
                                  0.0f, 0.0f, 3.0f}));
}

TEST(TensorAdvancedTest, ProfilingWrapperPreservesResult) {
    struct ProfilingGuard {
        ProfilingGuard() { Tensor::enable_profiling(true); }
        ~ProfilingGuard() { Tensor::enable_profiling(false); }
    } guard;

    const auto input = Tensor::ones({4}, Device::CUDA);
    const auto result = input.timed("test_operation", [](const Tensor& tensor) {
        return tensor.add(1.0f).mul(2.0f).sub(1.0f);
    });

    EXPECT_EQ(result.cpu().to_vector(), (std::vector<float>{3.0f, 3.0f, 3.0f, 3.0f}));
}

TEST(TensorAdvancedTest, MetadataAssertionsRejectMismatches) {
    auto cuda_float = Tensor::ones({3, 4}, Device::CUDA, DataType::Float32);
    auto cpu_int = Tensor::zeros({3, 4}, Device::CPU, DataType::Int32);

    EXPECT_NO_THROW(cuda_float.assert_shape({3, 4}, "shape"));
    EXPECT_THROW(cuda_float.assert_shape({4, 3}, "shape"), TensorError);
    EXPECT_NO_THROW(cuda_float.assert_device(Device::CUDA));
    EXPECT_THROW(cuda_float.assert_device(Device::CPU), TensorError);
    EXPECT_NO_THROW(cpu_int.assert_dtype(DataType::Int32));
    EXPECT_THROW(cpu_int.assert_dtype(DataType::Float32), TensorError);
}
