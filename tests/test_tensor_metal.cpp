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
#include <tuple>
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

    TEST_F(TensorMetal, MatrixProductsMatchCpu) {
        // Single elements, tile edges, and depths that are no multiple of a tile.
        for (const auto& [m, k, n] : {std::tuple{1, 1, 1}, std::tuple{4, 0, 3}, std::tuple{3, 7, 5}, std::tuple{65, 17, 33},
                                      std::tuple{128, 256, 64}, std::tuple{257, 1025, 129}}) {
            SCOPED_TRACE(std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n));
            const Tensor a = random_tensor(static_cast<size_t>(m * k), -1.0f, 1.0f, 11).reshape({m, k});
            const Tensor b = random_tensor(static_cast<size_t>(k * n), -1.0f, 1.0f, 12).reshape({k, n});
            expect_close(to_metal(a).mm(to_metal(b)), a.mm(b), 1.0e-4f, 1.0e-4f);
        }
        const Tensor a = random_tensor(3 * 20 * 30, -1.0f, 1.0f, 13).reshape({3, 20, 30});
        const Tensor b = random_tensor(3 * 30 * 10, -1.0f, 1.0f, 14).reshape({3, 30, 10});
        expect_close(to_metal(a).bmm(to_metal(b)), a.bmm(b), 1.0e-4f, 1.0e-4f);

        // linear is x @ weight^T, conv1x1 is weight @ x per image; both add the bias per output channel.
        const Tensor x = random_tensor(37 * 24, -1.0f, 1.0f, 15).reshape({37, 24});
        const Tensor weight = random_tensor(16 * 24, -1.0f, 1.0f, 16).reshape({16, 24});
        const Tensor bias = random_tensor(16, -1.0f, 1.0f, 17);
        expect_close(to_metal(x).linear(to_metal(weight)), x.linear(weight), 1.0e-4f, 1.0e-4f);
        expect_close(to_metal(x).linear(to_metal(weight), to_metal(bias)), x.linear(weight, bias), 1.0e-4f, 1.0e-4f);
        const Tensor image = random_tensor(24 * 9 * 11, -1.0f, 1.0f, 18).reshape({1, 24, 9, 11});
        expect_close(to_metal(image).conv1x1(to_metal(weight), to_metal(bias)), image.conv1x1(weight, bias),
                     1.0e-4f, 1.0e-4f);

        // The fused bias and ReLU epilogues, and ReLU on its own.
        Tensor linear_out = to_metal(Tensor::zeros({37, 16}, Device::CPU));
        to_metal(x).linear_bias_relu_out(to_metal(weight), to_metal(bias), linear_out);
        expect_close(linear_out, x.linear(weight, bias).relu(), 1.0e-4f, 1.0e-4f);
        Tensor conv_out = to_metal(Tensor::zeros({1, 16, 9, 11}, Device::CPU));
        to_metal(image).conv1x1_bias_relu_out(to_metal(weight), to_metal(bias), conv_out);
        expect_close(conv_out, image.conv1x1(weight, bias).relu(), 1.0e-4f, 1.0e-4f);
        Tensor relu_out = to_metal(Tensor::zeros({37, 24}, Device::CPU));
        to_metal(x).relu_out(relu_out);
        expect_close(relu_out, x.relu(), 0.0f, 0.0f);
    }

    TEST_F(TensorMetal, PoolingMatchesCpu) {
        const Tensor x_cpu = random_tensor(2 * 3 * 9 * 11, -2.0f, 2.0f, 19).reshape({2, 3, 9, 11});
        const Tensor x = to_metal(x_cpu);
        expect_close(x.max_pool2d(2), x_cpu.max_pool2d(2), 0.0f, 0.0f);
        expect_close(x.max_pool2d(3, 2, 1), x_cpu.max_pool2d(3, 2, 1), 0.0f, 0.0f);
        expect_close(x.adaptive_avg_pool2d(4, 5), x_cpu.adaptive_avg_pool2d(4, 5));
    }

    TEST_F(TensorMetal, MatrixHelpersMatchCpu) {
        {
            GpuBackendScope scope(GpuBackend::Metal);
            expect_close(Tensor::eye(5, 7, Device::GPU), Tensor::eye(5, 7, Device::CPU), 0.0f, 0.0f);
        }
        const Tensor diagonal = random_tensor(6, -1.0f, 1.0f, 20);
        expect_close(Tensor::diag(to_metal(diagonal)), Tensor::diag(diagonal), 0.0f, 0.0f);

        for (const size_t count : {size_t{1}, size_t{1000}, size_t{3} << 20}) {
            SCOPED_TRACE(count);
            const Tensor a = random_tensor(count, -1.0f, 1.0f, 21);
            const Tensor b = random_tensor(count, -1.0f, 1.0f, 22);
            const auto a_values = a.to_vector(), b_values = b.to_vector();
            double expected = 0.0;
            for (size_t i = 0; i < count; ++i)
                expected += static_cast<double>(a_values[i]) * b_values[i];
            EXPECT_NEAR(to_metal(a).dot(to_metal(b)).item(), expected,
                        1.0e-5 * std::sqrt(static_cast<double>(count)) + 1.0e-6);
        }

        const Tensor lhs = random_tensor(7 * 5, -1.0f, 1.0f, 23).reshape({7, 5});
        const Tensor rhs = random_tensor(9 * 5, -1.0f, 1.0f, 24).reshape({9, 5});
        for (const float p : {0.0f, 1.0f, 2.0f, 3.0f, std::numeric_limits<float>::infinity()}) {
            SCOPED_TRACE(p);
            expect_close(to_metal(lhs).cdist(to_metal(rhs), p), lhs.cdist(rhs, p));
        }
    }

    TEST_F(TensorMetal, AxisReductionsMatchCpu) {
        // Every axis combination: segmented, strided and permuted paths.
        const Tensor x_cpu = random_tensor(6 * 70 * 33, -2.0f, 2.0f, 25).reshape({6, 70, 33});
        const Tensor x = to_metal(x_cpu);
        for (const std::vector<int>& axes : std::vector<std::vector<int>>{{0}, {1}, {2}, {0, 1}, {1, 2}, {0, 2}}) {
            SCOPED_TRACE(testing::PrintToString(axes));
            expect_close(x.sum(axes), x_cpu.sum(axes), 1.0e-5f, 1.0e-5f);
            expect_close(x.mean(axes), x_cpu.mean(axes), 1.0e-5f, 1.0e-5f);
            expect_close(x.max(axes), x_cpu.max(axes), 0.0f, 0.0f);
            expect_close(x.min(axes), x_cpu.min(axes), 0.0f, 0.0f);
        }
        expect_close(x.sum(), x_cpu.sum(), 1.0e-5f, 1.0e-4f);
        expect_close(x.std(1), x_cpu.std(1), 1.0e-5f, 1.0e-5f);

        // Few outputs over a long extent split across grid rows; few long segments.
        const Tensor tall = random_tensor(5000 * 3, -1.0f, 1.0f, 26).reshape({5000, 3});
        expect_close(to_metal(tall).sum(0), tall.sum(0), 1.0e-5f, 1.0e-4f);
        expect_close(to_metal(tall).max(0), tall.max(0), 0.0f, 0.0f);
        const Tensor wide = tall.reshape({3, 5000});
        expect_close(to_metal(wide).mean(1), wide.mean(1), 1.0e-5f, 1.0e-5f);

        const Tensor factors = random_tensor(4 * 9, 0.5f, 1.5f, 27).reshape({4, 9});
        expect_close(to_metal(factors).prod(1), factors.prod(1), 1.0e-5f, 0.0f);
        const Tensor integers = (x_cpu * 10.0f).to(DataType::Int32);
        expect_close(to_metal(integers).sum(1), integers.sum(1), 0.0f, 0.0f);
        expect_close(to_metal(integers).max(2), integers.max(2), 0.0f, 0.0f);
        const Tensor mask = x_cpu > 1.5f;
        expect_close(to_metal(mask).any(1), mask.any(1), 0.0f, 0.0f);
        expect_close(to_metal(mask).all(2), mask.all(2), 0.0f, 0.0f);
    }

    TEST_F(TensorMetal, FusedReductionsMatchCpu) {
        const Tensor a_cpu = random_tensor(size_t{1} << 16, -1.0f, 1.0f, 28);
        const Tensor b_cpu = random_tensor(size_t{1} << 16, -1.0f, 1.0f, 29);
        const Tensor a = to_metal(a_cpu), b = to_metal(b_cpu);
        expect_close((a * b).sum(), (a_cpu * b_cpu).sum(), 1.0e-4f, 1.0e-4f);
        expect_close(((a + 1.0f) * 2.0f).mean(), ((a_cpu + 1.0f) * 2.0f).mean(), 1.0e-5f, 1.0e-5f);
        expect_close((a - b).abs().max(), (a_cpu - b_cpu).abs().max(), 0.0f, 0.0f);
        expect_close((a * b).reshape({256, 256}).sum(1), (a_cpu * b_cpu).reshape({256, 256}).sum(1), 1.0e-5f, 1.0e-5f);
    }

    TEST_F(TensorMetal, CountsAndScansMatchCpu) {
        const Tensor x_cpu = random_tensor(100000, -1.0f, 1.0f, 30);
        const Tensor x = to_metal(x_cpu);
        EXPECT_EQ((x > 0.5f).count_nonzero(), (x_cpu > 0.5f).count_nonzero());
        EXPECT_EQ(x.relu().count_nonzero(), x_cpu.relu().count_nonzero());
        EXPECT_FALSE(x.has_nan());
        EXPECT_FALSE(x.has_inf());
        std::vector<float> special(1000, 1.0f);
        special[500] = std::numeric_limits<float>::quiet_NaN();
        special[900] = -std::numeric_limits<float>::infinity();
        const Tensor with_special = to_metal(Tensor::from_vector(special, {special.size()}, Device::CPU));
        EXPECT_TRUE(with_special.has_nan());
        EXPECT_TRUE(with_special.has_inf());

        // Short lines, one block, and two levels of block totals.
        for (const size_t length : {size_t{1}, size_t{17}, size_t{256}, size_t{257}, size_t{70000}}) {
            SCOPED_TRACE(length);
            const Tensor line = random_tensor(length, 0.0f, 1.0f, 31);
            expect_close(to_metal(line).cumsum(0), line.cumsum(0), 2.0e-4f, 1.0e-3f);
            const Tensor integers = (line * 10.0f).to(DataType::Int32);
            expect_close(to_metal(integers).cumsum(0), integers.cumsum(0), 0.0f, 0.0f);
        }
        const Tensor volume = random_tensor(5 * 300 * 7, 0.0f, 1.0f, 32).reshape({5, 300, 7});
        for (const int dim : {0, 1, 2}) {
            SCOPED_TRACE(dim);
            expect_close(to_metal(volume).cumsum(dim), volume.cumsum(dim), 2.0e-4f, 1.0e-3f);
        }
    }

    TEST_F(TensorMetal, UnportedOperationsSaySo) {
        const Tensor x = to_metal(random_tensor(16, 0.0f, 1.0f, 9));
        try {
            (void)x.sort().first.cpu();
            ADD_FAILURE() << "sort should not be ported yet";
        } catch (const std::exception& error) {
            EXPECT_NE(std::string(error.what()).find("Metal backend:"), std::string::npos) << error.what();
        }
    }

} // namespace
