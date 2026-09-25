/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal backend conformance: every ported operation runs on Metal and on the
// CPU reference and must agree; operations not ported yet must say so.

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {
    using namespace lfs::core;

    class TensorMetal : public testing::Test {
    protected:
        void SetUp() override {
            if (!gpu_backend_available(GpuBackend::Metal))
                GTEST_SKIP() << "No Metal device";
        }

        void TearDown() override {
            EXPECT_TRUE(shutdown_gpu_backend(GpuBackend::Metal).has_value());
        }
    };

    Tensor to_metal(const Tensor& cpu) {
        GpuBackendScope scope(GpuBackend::Metal);
        return cpu.to(Device::GPU);
    }

    Tensor random_tensor(const size_t count, const float low, const float high, const unsigned seed) {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(low, high);
        std::vector<float> values(count);
        for (float& value : values)
            value = distribution(generator);
        return Tensor::from_vector(values, {count}, Device::CPU);
    }

    void expect_close(const Tensor& actual, const Tensor& expected,
                      const float rtol = 2.0e-5f, const float atol = 2.0e-6f) {
        ASSERT_EQ(actual.dtype(), expected.dtype());
        const auto actual_values = actual.cpu().to(DataType::Float32).to_vector();
        const auto expected_values = expected.cpu().to(DataType::Float32).to_vector();
        ASSERT_EQ(actual_values.size(), expected_values.size());
        for (size_t i = 0; i < actual_values.size(); ++i) {
            if (std::isnan(expected_values[i])) {
                EXPECT_TRUE(std::isnan(actual_values[i])) << "index=" << i;
            } else if (std::isinf(expected_values[i])) {
                EXPECT_EQ(actual_values[i], expected_values[i]) << "index=" << i;
            } else {
                EXPECT_NEAR(actual_values[i], expected_values[i], atol + rtol * std::abs(expected_values[i]))
                    << "index=" << i;
            }
        }
    }

    TEST_F(TensorMetal, UploadAndDownloadRoundTripEveryDtype) {
        for (const DataType dtype : {DataType::Float32, DataType::Float16, DataType::Int32,
                                     DataType::Int64, DataType::UInt8, DataType::Bool}) {
            SCOPED_TRACE(static_cast<int>(dtype));
            const Tensor cpu = random_tensor(1001, 0.0f, 100.0f, 1).to(dtype);
            const Tensor metal = to_metal(cpu);
            ASSERT_EQ(gpu_backend_of(metal), GpuBackend::Metal);
            expect_close(metal, cpu, 0.0f, 0.0f);
        }
    }

    TEST_F(TensorMetal, FactoriesMatchCpu) {
        GpuBackendScope scope(GpuBackend::Metal);
        expect_close(Tensor::zeros({37}, Device::GPU), Tensor::zeros({37}, Device::CPU), 0.0f, 0.0f);
        expect_close(Tensor::ones({1027}, Device::GPU), Tensor::ones({1027}, Device::CPU), 0.0f, 0.0f);
        expect_close(Tensor::full({1027}, 2.5f, Device::GPU), Tensor::full({1027}, 2.5f, Device::CPU), 0.0f, 0.0f);
        expect_close(Tensor::zeros({333}, Device::GPU, DataType::Int32),
                     Tensor::zeros({333}, Device::CPU, DataType::Int32), 0.0f, 0.0f);

        const Tensor range = Tensor::arange(0.5f, 100.0f, 0.25f);
        ASSERT_EQ(gpu_backend_of(range), GpuBackend::Metal);
        std::vector<float> expected(static_cast<size_t>(range.numel()));
        for (size_t i = 0; i < expected.size(); ++i)
            expected[i] = 0.5f + static_cast<float>(i) * 0.25f;
        expect_close(range, Tensor::from_vector(expected, {expected.size()}, Device::CPU), 0.0f, 0.0f);
    }

    TEST_F(TensorMetal, PointwiseMatchesCpu) {
        using Unary = std::function<Tensor(const Tensor&)>;
        using Binary = std::function<Tensor(const Tensor&, const Tensor&)>;
        const std::vector<std::pair<std::string, Unary>> unary{
            {"exp", [](const Tensor& x) { return x.exp(); }},
            {"log", [](const Tensor& x) { return x.log(); }},
            {"sqrt", [](const Tensor& x) { return x.sqrt(); }},
            {"neg", [](const Tensor& x) { return -x; }},
            {"abs", [](const Tensor& x) { return (x - 2.0f).abs(); }},
            {"sigmoid", [](const Tensor& x) { return x.sigmoid(); }},
            {"relu", [](const Tensor& x) { return (x - 2.0f).relu(); }},
            {"tanh", [](const Tensor& x) { return x.tanh(); }},
            {"floor", [](const Tensor& x) { return x.floor(); }},
            {"round", [](const Tensor& x) { return (x * 2.0f).round(); }},
            {"sin", [](const Tensor& x) { return x.sin(); }},
            {"cos", [](const Tensor& x) { return x.cos(); }},
            {"scalar_right", [](const Tensor& x) { return x / 7.0f + 1.5f; }},
        };
        const std::vector<std::pair<std::string, Binary>> binary{
            {"add", [](const Tensor& x, const Tensor& y) { return x + y; }},
            {"sub", [](const Tensor& x, const Tensor& y) { return x - y; }},
            {"mul", [](const Tensor& x, const Tensor& y) { return x * y; }},
            {"div", [](const Tensor& x, const Tensor& y) { return x / y; }},
            {"maximum", [](const Tensor& x, const Tensor& y) { return x.maximum(y); }},
            {"chain", [](const Tensor& x, const Tensor& y) { return ((x + y) * 2.0f - y).exp(); }},
        };
        // Eager tails, float4-vectorized sizes, and fused chains with and without a tail.
        for (const size_t count : {size_t{1}, size_t{255}, size_t{1024}, size_t{1027}, size_t{1} << 20}) {
            SCOPED_TRACE(count);
            const Tensor x_cpu = random_tensor(count, 0.1f, 4.0f, 2);
            const Tensor y_cpu = random_tensor(count, 0.5f, 3.0f, 3);
            const Tensor x = to_metal(x_cpu), y = to_metal(y_cpu);
            for (const auto& [name, op] : unary) {
                SCOPED_TRACE(name);
                expect_close(op(x), op(x_cpu));
            }
            for (const auto& [name, op] : binary) {
                SCOPED_TRACE(name);
                expect_close(op(x, y), op(x_cpu, y_cpu));
            }
        }
    }

    TEST_F(TensorMetal, ComparisonsAndIntegersMatchCpu) {
        const Tensor x_cpu = random_tensor(4099, -50.0f, 50.0f, 4);
        const Tensor y_cpu = random_tensor(4099, -50.0f, 50.0f, 5);
        const Tensor x = to_metal(x_cpu), y = to_metal(y_cpu);
        expect_close(x.lt(y), x_cpu.lt(y_cpu), 0.0f, 0.0f);
        expect_close(x > 0.0f, x_cpu > 0.0f, 0.0f, 0.0f);
        expect_close(x.eq(x), x_cpu.eq(x_cpu), 0.0f, 0.0f);

        const Tensor a_cpu = x_cpu.to(DataType::Int32), b_cpu = y_cpu.to(DataType::Int32);
        const Tensor a = to_metal(a_cpu), b = to_metal(b_cpu);
        expect_close(a + b, a_cpu + b_cpu, 0.0f, 0.0f);
        expect_close(a * b, a_cpu * b_cpu, 0.0f, 0.0f);
        expect_close(a - 7, a_cpu - 7, 0.0f, 0.0f);
        expect_close(a.lt(b), a_cpu.lt(b_cpu), 0.0f, 0.0f);
    }

    TEST_F(TensorMetal, ConversionsMatchCpu) {
        const Tensor x_cpu = random_tensor(2053, -300.0f, 300.0f, 6);
        const Tensor x = to_metal(x_cpu);
        for (const DataType dtype : {DataType::Int32, DataType::Int64, DataType::UInt8,
                                     DataType::Bool, DataType::Float16}) {
            SCOPED_TRACE(static_cast<int>(dtype));
            const Tensor converted = x.to(dtype);
            ASSERT_EQ(gpu_backend_of(converted), GpuBackend::Metal);
            expect_close(converted, x_cpu.to(dtype), 0.0f, 0.0f);
            expect_close(converted.to(DataType::Float32), x_cpu.to(dtype).to(DataType::Float32), 0.0f, 0.0f);
        }
    }

    TEST_F(TensorMetal, ScalarReductionsMatchCpu) {
        for (const size_t count : {size_t{1}, size_t{777}, size_t{3} << 20}) {
            SCOPED_TRACE(count);
            const Tensor x_cpu = random_tensor(count, -1.0f, 1.0f, 7);
            const auto values = x_cpu.to_vector();
            double sum = 0.0;
            for (const float value : values)
                sum += value;
            const Tensor x = to_metal(x_cpu);
            const double tolerance = 1.0e-5 * std::sqrt(static_cast<double>(count)) + 1.0e-6;
            EXPECT_NEAR(x.sum_scalar(), sum, tolerance);
            EXPECT_NEAR(x.mean_scalar(), sum / static_cast<double>(count), tolerance);
            EXPECT_FLOAT_EQ(x.max_scalar(), x_cpu.max_scalar());
            EXPECT_FLOAT_EQ(x.min_scalar(), x_cpu.min_scalar());
        }
        std::vector<float> with_nan(1000, 1.0f);
        with_nan[617] = std::numeric_limits<float>::quiet_NaN();
        EXPECT_TRUE(std::isnan(to_metal(Tensor::from_vector(with_nan, {with_nan.size()}, Device::CPU)).max_scalar()));
    }

    TEST_F(TensorMetal, OffsetViewsUseTheirOffset) {
        // Eager ops and fused chains, with offsets that do and do not allow four-wide access.
        for (const size_t count : {size_t{1000}, size_t{4096}}) {
            SCOPED_TRACE(count);
            const Tensor base_cpu = random_tensor(count + 4, -5.0f, 5.0f, 8);
            const Tensor base = to_metal(base_cpu);
            const auto view = [&](const Tensor& tensor, const int64_t offset) {
                return tensor.slice(0, offset, offset + static_cast<int64_t>(count));
            };
            const Tensor misaligned_cpu = view(base_cpu, 3), aligned_cpu = view(base_cpu, 4);
            const Tensor misaligned = view(base, 3), aligned = view(base, 4);
            expect_close(misaligned * 2.0f, misaligned_cpu * 2.0f);
            expect_close(misaligned + misaligned, misaligned_cpu + misaligned_cpu);
            expect_close(aligned + misaligned, aligned_cpu + misaligned_cpu);
            expect_close(aligned * aligned, aligned_cpu * aligned_cpu);
            EXPECT_NEAR(misaligned.sum_scalar(), misaligned_cpu.sum_scalar(), 5.0e-3f);
        }
    }

    TEST_F(TensorMetal, TransposedCopiesMatchCpu) {
        for (const auto& [rows, columns] : {std::pair{3, 5}, std::pair{64, 1000}, std::pair{33, 65}}) {
            for (const DataType dtype : {DataType::Float32, DataType::Float16, DataType::Int64, DataType::UInt8}) {
                SCOPED_TRACE(std::to_string(rows) + "x" + std::to_string(columns));
                SCOPED_TRACE(static_cast<int>(dtype));
                const Tensor cpu = random_tensor(static_cast<size_t>(rows * columns), 0.0f, 100.0f, 10)
                                       .to(dtype)
                                       .reshape({rows, columns});
                expect_close(to_metal(cpu).transpose(0, 1).contiguous(), cpu.transpose(0, 1).contiguous(), 0.0f, 0.0f);
            }
        }
    }

    TEST_F(TensorMetal, UnportedOperationsSaySo) {
        const Tensor x = to_metal(random_tensor(16, 0.0f, 1.0f, 9)).reshape({4, 4});
        try {
            (void)x.matmul(x).cpu();
            ADD_FAILURE() << "matmul should not be ported yet";
        } catch (const std::exception& error) {
            EXPECT_NE(std::string(error.what()).find("Metal backend:"), std::string::npos) << error.what();
        }
    }

} // namespace
