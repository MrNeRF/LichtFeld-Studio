/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

using namespace lfs::core;

namespace {

    Tensor make_source(DataType dtype) {
        switch (dtype) {
        case DataType::Float32:
            return Tensor::from_vector(std::vector<float>{0.0f, 1.0f, 2.0f}, {3}, Device::CPU);
        case DataType::Float16:
            return Tensor::from_vector(std::vector<float>{0.0f, 1.0f, 2.0f}, {3}, Device::CPU)
                .to(DataType::Float16);
        case DataType::Int32:
            return Tensor::from_vector(std::vector<int>{0, 1, 2}, {3}, Device::CPU);
        case DataType::Int64:
            return Tensor::from_vector(std::vector<int>{0, 1, 2}, {3}, Device::CPU)
                .to(DataType::Int64);
        case DataType::UInt8: {
            std::array<uint8_t, 3> values = {0, 1, 2};
            return Tensor::from_blob(values.data(), {3}, Device::CPU, dtype).clone();
        }
        case DataType::Bool: {
            std::array<uint8_t, 3> values = {0, 1, 1};
            return Tensor::from_blob(values.data(), {3}, Device::CPU, dtype).clone();
        }
        default:
            throw std::runtime_error("unsupported conversion source dtype");
        }
    }

    std::vector<float> values_as_float(const Tensor& tensor) {
        return tensor.to(DataType::Float32).to_vector();
    }

    bool has_cuda_device() {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    DataType expected_promotion(const DataType lhs, const DataType rhs) {
        if (lhs == rhs) {
            return lhs;
        }
        for (const auto dtype : {DataType::Float32, DataType::Float16,
                                 DataType::Int64, DataType::Int32, DataType::UInt8}) {
            if (lhs == dtype || rhs == dtype) {
                return dtype;
            }
        }
        return DataType::Bool;
    }

} // namespace

TEST(CPUDtypeConversionTest, EverySupportedPairHasExactSemantics) {
    struct Conversion {
        DataType from;
        DataType to;
    };

    constexpr std::array conversions = {
        Conversion{DataType::Float32, DataType::Int32},
        Conversion{DataType::Float32, DataType::Int64},
        Conversion{DataType::Float32, DataType::UInt8},
        Conversion{DataType::Float32, DataType::Bool},
        Conversion{DataType::Int32, DataType::Float32},
        Conversion{DataType::Int32, DataType::Int64},
        Conversion{DataType::Int32, DataType::UInt8},
        Conversion{DataType::Int32, DataType::Bool},
        Conversion{DataType::Int64, DataType::Float32},
        Conversion{DataType::Int64, DataType::Int32},
        Conversion{DataType::Int64, DataType::UInt8},
        Conversion{DataType::Int64, DataType::Bool},
        Conversion{DataType::UInt8, DataType::Float32},
        Conversion{DataType::UInt8, DataType::Int32},
        Conversion{DataType::UInt8, DataType::Int64},
        Conversion{DataType::UInt8, DataType::Bool},
        Conversion{DataType::Bool, DataType::Float32},
        Conversion{DataType::Bool, DataType::Int32},
        Conversion{DataType::Bool, DataType::Int64},
        Conversion{DataType::Bool, DataType::UInt8},
    };

    for (const auto& conversion : conversions) {
        SCOPED_TRACE(std::string(dtype_name(conversion.from)) + " -> " +
                     std::string(dtype_name(conversion.to)));
        const auto source = make_source(conversion.from);
        const auto converted = source.to(conversion.to);

        EXPECT_EQ(converted.dtype(), conversion.to);
        EXPECT_EQ(converted.device(), Device::CPU);
        EXPECT_EQ(converted.shape(), source.shape());
        const std::vector<float> expected = conversion.from == DataType::Bool ||
                                                    conversion.to == DataType::Bool
                                                ? std::vector<float>{0.0f, 1.0f, 1.0f}
                                                : std::vector<float>{0.0f, 1.0f, 2.0f};
        EXPECT_EQ(values_as_float(converted), expected);
    }
}

TEST(CPUDtypeConversionTest, EveryDtypeRoundTripsThroughCuda) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    constexpr std::array dtypes = {
        DataType::Float32, DataType::Float16, DataType::Int32, DataType::Int64,
        DataType::UInt8, DataType::Bool};

    for (const auto dtype : dtypes) {
        SCOPED_TRACE(dtype_name(dtype));
        const auto source = make_source(dtype);
        const auto roundtrip = source.cuda().cpu();
        EXPECT_EQ(roundtrip.dtype(), dtype);
        EXPECT_EQ(roundtrip.shape(), source.shape());
        EXPECT_EQ(values_as_float(roundtrip), values_as_float(source));
    }
}

TEST(CPUDtypeConversionTest, EveryDtypePairHasExactArithmeticAndPromotion) {
    constexpr std::array dtypes = {
        DataType::Bool, DataType::UInt8, DataType::Int32,
        DataType::Int64, DataType::Float16, DataType::Float32};

    for (const auto lhs_dtype : dtypes) {
        for (const auto rhs_dtype : dtypes) {
            SCOPED_TRACE(std::string(dtype_name(lhs_dtype)) + " with " +
                         std::string(dtype_name(rhs_dtype)));
            const auto lhs = Tensor::from_vector(
                                 std::vector<float>{0.0f, 2.0f}, {2}, Device::CUDA)
                                 .to(lhs_dtype);
            const auto rhs = Tensor::from_vector(
                                 std::vector<float>{0.0f, 1.0f}, {2}, Device::CUDA)
                                 .to(rhs_dtype);

            if (lhs_dtype == DataType::Bool && rhs_dtype == DataType::Bool) {
                EXPECT_THROW(lhs + rhs, std::runtime_error);
                EXPECT_THROW(lhs - rhs, std::runtime_error);
                EXPECT_THROW(lhs * rhs, std::runtime_error);
                continue;
            }

            const auto expected_dtype = expected_promotion(lhs_dtype, rhs_dtype);
            const bool bool_lhs = lhs_dtype == DataType::Bool;
            const auto sum = lhs + rhs;
            const auto difference = lhs - rhs;
            const auto product = lhs * rhs;

            EXPECT_EQ(sum.dtype(), expected_dtype);
            EXPECT_EQ(difference.dtype(), expected_dtype);
            EXPECT_EQ(product.dtype(), expected_dtype);
            EXPECT_EQ(values_as_float(sum.cpu()),
                      (bool_lhs ? std::vector<float>{0.0f, 2.0f}
                                : std::vector<float>{0.0f, 3.0f}));
            EXPECT_EQ(values_as_float(difference.cpu()),
                      (bool_lhs ? std::vector<float>{0.0f, 0.0f}
                                : std::vector<float>{0.0f, 1.0f}));
            EXPECT_EQ(values_as_float(product.cpu()),
                      (bool_lhs ? std::vector<float>{0.0f, 1.0f}
                                : std::vector<float>{0.0f, 2.0f}));
        }
    }
}

TEST(CPUDtypeConversionTest, MixedDtypeBroadcastPromotesAndExpands) {
    const auto mask = Tensor::from_vector(
        std::vector<bool>{true}, {1}, Device::CUDA);
    const auto values = Tensor::from_vector(
                            std::vector<float>{1.0f, 2.0f, 3.0f}, {3}, Device::CUDA)
                            .to(DataType::Float16);

    const auto result = mask * values;

    EXPECT_EQ(result.dtype(), DataType::Float16);
    EXPECT_EQ(result.shape(), values.shape());
    EXPECT_EQ(values_as_float(result.cpu()), (std::vector<float>{1.0f, 2.0f, 3.0f}));
}

TEST(CPUDtypeConversionTest, Float16ToBoolHasExactValues) {
    const auto values = Tensor::from_vector(
                            std::vector<float>{0.0f, 1.0f, -2.0f}, {3}, Device::CUDA)
                            .to(DataType::Float16);

    const auto result = values.to(DataType::Bool);

    EXPECT_EQ(result.dtype(), DataType::Bool);
    EXPECT_EQ(values_as_float(result.cpu()), (std::vector<float>{0.0f, 1.0f, 1.0f}));
}

TEST(CPUDtypeConversionTest, ProductionMuralPointCloudWorkflow) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    constexpr size_t num_points = 4'042'850;
    const auto means_cpu = Tensor::zeros({num_points, 3}, Device::CPU);
    std::vector<uint8_t> color_data(num_points * 3);
    for (size_t i = 0; i < color_data.size(); ++i) {
        color_data[i] = static_cast<uint8_t>((i * 7 + 13) % 256);
    }
    const auto colors_cpu = Tensor::from_blob(
                                color_data.data(), {num_points, 3}, Device::CPU, DataType::UInt8)
                                .clone();

    const auto means_gpu = means_cpu.cuda();
    const auto colors_cpu_float = colors_cpu.to(DataType::Float32);
    const auto colors_gpu_float = colors_cpu.cuda().to(DataType::Float32);

    EXPECT_EQ(means_gpu.shape(), TensorShape({num_points, 3}));
    EXPECT_TRUE(colors_cpu_float.all_close(colors_gpu_float.cpu()));
    const auto normalized = (colors_gpu_float / 255.0f).cpu().to_vector();
    ASSERT_GE(normalized.size(), 2u);
    EXPECT_NEAR(normalized[0], 13.0f / 255.0f, 1e-6f);
    EXPECT_NEAR(normalized[1], 20.0f / 255.0f, 1e-6f);
}
