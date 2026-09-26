/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal backend conformance: every operation runs on Metal and on the CPU
// reference or the Vulkan backend, and they must agree.

#include "core/nn/ops.hpp"
#include "core/rad_dequant_math.hpp"
#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "core/tensor.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_environment.hpp"
#include "core/tensor_export.hpp"
#include "core/tensor_filters.hpp"
#include "core/tensor_fused.hpp"
#include "core/tensor_histogram.hpp"
#include "core/tensor_image.hpp"
#include "core/tensor_labels.hpp"
#include "core/tensor_ppisp.hpp"
#include "core/tensor_rad.hpp"
#include "core/tensor_sh.hpp"
#include "core/tensor_spatial.hpp"
#include "core/tensor_splat.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
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

    TEST_F(TensorMetal, BroadcastsMatchCpu) {
        const Tensor a_cpu = random_tensor(4 * 1 * 3, -2.0f, 2.0f, 33).reshape({4, 1, 3});
        const Tensor b_cpu = random_tensor(5 * 1, -2.0f, 2.0f, 34).reshape({5, 1});
        const Tensor a = to_metal(a_cpu), b = to_metal(b_cpu);
        expect_close(a + b, a_cpu + b_cpu);
        expect_close(a * b, a_cpu * b_cpu);
        expect_close(a.maximum(b), a_cpu.maximum(b_cpu), 0.0f, 0.0f);
        expect_close(a.lt(b), a_cpu.lt(b_cpu), 0.0f, 0.0f);
        const Tensor a_int = (a_cpu * 10.0f).to(DataType::Int32), b_int = (b_cpu * 10.0f).to(DataType::Int32);
        expect_close(to_metal(a_int) - to_metal(b_int), a_int - b_int, 0.0f, 0.0f);
        const Tensor matrix = random_tensor(300 * 7, 1.0f, 2.0f, 35).reshape({300, 7});
        const Tensor row = random_tensor(7, 1.0f, 2.0f, 36);
        expect_close(to_metal(matrix) / to_metal(row), matrix / row);
    }

    TEST_F(TensorMetal, ClampCatAndPadMatchCpu) {
        std::vector<float> values = random_tensor(1000, -3.0f, 3.0f, 37).to_vector();
        values[123] = std::numeric_limits<float>::quiet_NaN();
        const Tensor x_cpu = Tensor::from_vector(values, {values.size()}, Device::CPU);
        expect_close(to_metal(x_cpu).clamp(-1.0f, 2.0f), x_cpu.clamp(-1.0f, 2.0f), 0.0f, 0.0f);
        Tensor in_place = to_metal(x_cpu);
        in_place.clamp_(-0.5f, 0.5f);
        expect_close(in_place, x_cpu.clamp(-0.5f, 0.5f), 0.0f, 0.0f);
        const Tensor integers = (random_tensor(1000, -3.0f, 3.0f, 38) * 10.0f).to(DataType::Int32);
        Tensor integers_metal = to_metal(integers);
        integers_metal.clamp_(-5.0f, 7.0f);
        expect_close(integers_metal, integers.clamp(-5.0f, 7.0f), 0.0f, 0.0f);

        for (const DataType dtype : {DataType::Float32, DataType::Int64, DataType::UInt8, DataType::Float16}) {
            SCOPED_TRACE(static_cast<int>(dtype));
            const auto make = [&](const int rows, const int columns, const unsigned seed) {
                return random_tensor(static_cast<size_t>(rows * columns), 0.0f, 100.0f, seed).to(dtype).reshape({rows, columns});
            };
            const Tensor p = make(3, 4, 39), q = make(3, 2, 40), r = make(3, 5, 41);
            expect_close(Tensor::cat({to_metal(p), to_metal(q), to_metal(r)}, 1), Tensor::cat({p, q, r}, 1), 0.0f, 0.0f);
            const Tensor u = make(2, 12, 42).reshape({2, 3, 4}), v = make(2, 8, 43).reshape({2, 2, 4});
            expect_close(Tensor::cat({to_metal(u), to_metal(v)}, 1), Tensor::cat({u, v}, 1), 0.0f, 0.0f);
        }

        MovementArgs pad_args;
        pad_args.args = std::vector<std::pair<int, int>>{{1, 1}, {2, 1}};
        const Tensor base = random_tensor(3 * 4, -1.0f, 1.0f, 44).reshape({3, 4});
        expect_close(to_metal(base).movement(MovementOp::Pad, pad_args), base.movement(MovementOp::Pad, pad_args), 0.0f, 0.0f);
        const Tensor transposed = base.transpose(0, 1);
        expect_close(to_metal(base).transpose(0, 1).movement(MovementOp::Pad, pad_args),
                     transposed.movement(MovementOp::Pad, pad_args), 0.0f, 0.0f);
        // A strided view fills only its own elements.
        for (const DataType dtype : {DataType::Float32, DataType::Int32, DataType::Bool}) {
            SCOPED_TRACE(static_cast<int>(dtype));
            const Tensor source = random_tensor(24, 0.0f, 3.0f, 44).to(dtype).reshape({4, 6});
            Tensor filled = to_metal(source), expected = source.clone();
            filled.transpose(0, 1).slice(0, 1, 4).fill_(1.0f);
            expected.transpose(0, 1).slice(0, 1, 4).fill_(1.0f);
            expect_close(filled, expected, 0.0f, 0.0f);
        }
    }

    std::vector<int> pseudo_indices(const size_t count, const size_t extent, const unsigned seed) {
        std::mt19937 generator(seed);
        std::uniform_int_distribution<int> distribution(0, static_cast<int>(extent) - 1);
        std::vector<int> indices(count);
        for (int& index : indices)
            index = distribution(generator);
        return indices;
    }

    Tensor int_tensor(const std::vector<int>& values, const TensorShape& shape) {
        return Tensor::from_vector(values, shape, Device::CPU);
    }

    TEST_F(TensorMetal, IndexingMatchesCpu) {
        // One element, a partial threadgroup and several threadgroups; take
        // counts negative indices from the end and clamps.
        for (const size_t count : {size_t{1}, size_t{7}, size_t{4099}}) {
            SCOPED_TRACE(count);
            const Tensor source = random_tensor(count, -5.0f, 5.0f, 45);
            std::vector<int> picks = pseudo_indices(count + 3, count, 46);
            for (size_t i = 0; i < picks.size(); i += 5)
                picks[i] = -picks[i] - 1;
            const Tensor with_negatives = int_tensor(picks, {picks.size()});
            expect_close(to_metal(source).take(to_metal(with_negatives)), source.take(with_negatives), 0.0f, 0.0f);
            const Tensor valid = int_tensor(pseudo_indices(count + 3, count, 47), {count + 3});
            expect_close(to_metal(source).gather(0, to_metal(valid)), source.gather(0, valid), 0.0f, 0.0f);
            expect_close(to_metal(source).index_select(0, to_metal(valid)), source.index_select(0, valid), 0.0f, 0.0f);
        }

        const Tensor matrix = random_tensor(37 * 65, -5.0f, 5.0f, 48).reshape({37, 65});
        const Tensor columns = int_tensor(pseudo_indices(37 * 9, 65, 49), {37, 9});
        expect_close(to_metal(matrix).gather(1, to_metal(columns)), matrix.gather(1, columns), 0.0f, 0.0f);
        const Tensor rows = int_tensor(pseudo_indices(11, 37, 50), {11});
        const Tensor inner = int_tensor(pseudo_indices(5, 65, 51), {5});
        for (const DataType dtype : {DataType::Float32, DataType::Int32, DataType::UInt8}) {
            SCOPED_TRACE(static_cast<int>(dtype));
            const Tensor typed = (matrix + 5.0f).to(dtype);
            expect_close(to_metal(typed).index_select(0, to_metal(rows)), typed.index_select(0, rows), 0.0f, 0.0f);
            expect_close(to_metal(typed).index_select(1, to_metal(inner)), typed.index_select(1, inner), 0.0f, 0.0f);
        }

        // Clamp and wrap, then an asserted out-of-range index surfaces at the
        // next readback and is cleared.
        const Tensor line = random_tensor(50, -5.0f, 5.0f, 52);
        const Tensor wild = int_tensor({-3, 0, 49, 50, 77, -120, 12}, {7});
        for (const BoundaryMode mode : {BoundaryMode::Clamp, BoundaryMode::Wrap}) {
            expect_close(to_metal(line).index_select(0, to_metal(wild), mode), line.index_select(0, wild, mode), 0.0f, 0.0f);
            expect_close(to_metal(line).gather(0, to_metal(wild), mode), line.gather(0, wild, mode), 0.0f, 0.0f);
        }
        EXPECT_THROW((void)to_metal(line).index_select(0, to_metal(wild), BoundaryMode::Assert).cpu(), std::exception);
        // A fault that another operation's wait raises is raised once.
        const Tensor pending = to_metal(line).index_select(0, to_metal(wild), BoundaryMode::Assert);
        EXPECT_THROW((void)to_metal(line).count_nonzero(), std::exception);
        EXPECT_EQ(to_metal(line).count_nonzero(), line.count_nonzero());
        const Tensor fine = int_tensor({1, 2, 3}, {3});
        expect_close(to_metal(line).index_select(0, to_metal(fine)), line.index_select(0, fine), 0.0f, 0.0f);
    }

    TEST_F(TensorMetal, ScattersMatchCpu) {
        constexpr size_t count = 4099;
        const Tensor base = random_tensor(count, -5.0f, 5.0f, 53);
        const Tensor source = random_tensor(count, -5.0f, 5.0f, 54);
        const Tensor targets = int_tensor(pseudo_indices(count, count, 55), {count});

        // Duplicate targets: the last source position wins, exactly as on the CPU.
        Tensor scattered = to_metal(base);
        scattered.scatter_(0, to_metal(targets), to_metal(source));
        Tensor scattered_cpu = base.clone();
        scattered_cpu.scatter_(0, targets, source);
        expect_close(scattered, scattered_cpu, 0.0f, 0.0f);

        Tensor added = to_metal(base);
        added.index_add_(0, to_metal(targets), to_metal(source));
        Tensor added_cpu = base.clone();
        added_cpu.index_add_(0, targets, source);
        expect_close(added, added_cpu, 1.0e-5f, 1.0e-5f);
        const Tensor integers = (base * 10.0f).to(DataType::Int32);
        Tensor added_integers = to_metal(integers);
        added_integers.index_add_(0, to_metal(targets), to_metal((source * 10.0f).to(DataType::Int32)));
        Tensor added_integers_cpu = integers.clone();
        added_integers_cpu.index_add_(0, targets, (source * 10.0f).to(DataType::Int32));
        expect_close(added_integers, added_integers_cpu, 0.0f, 0.0f);

        const Tensor matrix = random_tensor(19 * 33, -5.0f, 5.0f, 56).reshape({19, 33});
        const Tensor row_targets = int_tensor({4, 0, 18, 7, 11}, {5});
        const Tensor row_source = random_tensor(5 * 33, -5.0f, 5.0f, 57).reshape({5, 33});
        Tensor copied = to_metal(matrix);
        copied.index_copy_(0, to_metal(row_targets), to_metal(row_source));
        Tensor copied_cpu = matrix.clone();
        copied_cpu.index_copy_(0, row_targets, row_source);
        expect_close(copied, copied_cpu, 0.0f, 0.0f);
        const Tensor column_targets = int_tensor({2, 30, 15}, {3});
        Tensor filled = to_metal(matrix);
        filled.index_fill_(1, to_metal(column_targets), -2.5f);
        Tensor filled_cpu = matrix.clone();
        filled_cpu.index_fill_(1, column_targets, -2.5f);
        expect_close(filled, filled_cpu, 0.0f, 0.0f);

        const Tensor flat_targets = int_tensor({3, -1, 7, 3}, {4});
        const Tensor flat_values = random_tensor(4, -5.0f, 5.0f, 58);
        Tensor put = to_metal(matrix.flatten());
        put.index_put_(to_metal(flat_targets.slice(0, 0, 3)), to_metal(flat_values.slice(0, 0, 3)));
        Tensor put_cpu = matrix.flatten().clone();
        put_cpu.index_put_(flat_targets.slice(0, 0, 3), flat_values.slice(0, 0, 3));
        expect_close(put, put_cpu, 0.0f, 0.0f);
    }

    TEST_F(TensorMetal, MasksMatchCpu) {
        for (const size_t count : {size_t{1}, size_t{7}, size_t{4099}, size_t{300000}}) {
            SCOPED_TRACE(count);
            std::vector<float> values = random_tensor(count, -1.0f, 3.0f, 59).to_vector();
            for (size_t i = 0; i < count; i += 3)
                values[i] = 0.0f;
            const Tensor x_cpu = Tensor::from_vector(values, {count}, Device::CPU);
            const Tensor x = to_metal(x_cpu);
            const Tensor mask = x > 0.0f, mask_cpu = x_cpu > 0.0f;
            expect_close(x.masked_select(mask), x_cpu.masked_select(mask_cpu), 0.0f, 0.0f);
            Tensor filled = x.clone(), filled_cpu = x_cpu.clone();
            filled.masked_fill_(mask, 9.0f);
            filled_cpu.masked_fill_(mask_cpu, 9.0f);
            expect_close(filled, filled_cpu, 0.0f, 0.0f);
            const size_t selected = mask_cpu.count_nonzero();
            if (selected > 0) {
                const Tensor replacement = random_tensor(selected, -5.0f, -1.0f, 60);
                Tensor scattered = x.clone(), scattered_cpu = x_cpu.clone();
                scattered[mask] = to_metal(replacement);
                scattered_cpu[mask_cpu] = replacement;
                expect_close(scattered, scattered_cpu, 0.0f, 0.0f);
            }
            expect_close(x.nonzero(), x_cpu.nonzero(), 0.0f, 0.0f);
            expect_close(mask.nonzero(), mask_cpu.nonzero(), 0.0f, 0.0f);
            Tensor live = x > 0.0f, live_cpu = x_cpu > 0.0f;
            live.and_live_(x < 2.0f);
            live_cpu.and_live_(x_cpu < 2.0f);
            expect_close(live, live_cpu, 0.0f, 0.0f);
        }
        const Tensor integers = (random_tensor(4099, -3.0f, 3.0f, 61)).to(DataType::Int32);
        Tensor integers_metal = to_metal(integers), integers_cpu = integers.clone();
        integers_metal.masked_fill_(to_metal(integers) > 0.0f, 7.0f);
        integers_cpu.masked_fill_(integers > 0.0f, 7.0f);
        expect_close(integers_metal, integers_cpu, 0.0f, 0.0f);
    }

    TEST_F(TensorMetal, SortsMatchCpu) {
        // Short lines sort in one threadgroup, longer ones through the radix sort.
        for (const size_t count : {size_t{1}, size_t{7}, size_t{2048}, size_t{2049}, size_t{100000}}) {
            SCOPED_TRACE(count);
            std::vector<float> values = random_tensor(count, -3.0f, 3.0f, 62).to_vector();
            for (size_t i = 0; i < count; i += 11)
                values[i] = i % 2 == 0 ? 0.0f : -0.0f;
            for (size_t i = 5; i < count; i += 97)
                values[i] = std::numeric_limits<float>::quiet_NaN();
            const Tensor x_cpu = Tensor::from_vector(values, {count}, Device::CPU);
            for (const bool descending : {false, true}) {
                SCOPED_TRACE(descending);
                const auto [sorted, indices] = to_metal(x_cpu).sort(0, descending);
                const auto [sorted_cpu, indices_cpu] = x_cpu.sort(0, descending);
                expect_close(sorted, sorted_cpu, 0.0f, 0.0f);
                expect_close(indices, indices_cpu, 0.0f, 0.0f);
            }
        }
        // Along an inner axis of a 3D tensor, short and long lines.
        for (const int length : {33, 3000}) {
            SCOPED_TRACE(length);
            const Tensor volume = random_tensor(static_cast<size_t>(4 * length * 3), -3.0f, 3.0f, 63).reshape({4, length, 3});
            const auto [sorted, indices] = to_metal(volume).sort(1);
            const auto [sorted_cpu, indices_cpu] = volume.sort(1);
            expect_close(sorted, sorted_cpu, 0.0f, 0.0f);
            expect_close(indices, indices_cpu, 0.0f, 0.0f);
        }
    }

    // Metal and Vulkan draw the same Philox blocks, so a seed gives both the
    // same numbers.
    TEST_F(TensorMetal, RandomDrawsMatchVulkan) {
        if (!gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP() << "No Vulkan device";
        const auto draw = [](const GpuBackend backend, const auto& make) {
            GpuBackendScope scope(backend);
            Tensor::manual_seed(1234);
            return make().cpu();
        };
        const auto compare = [&](const auto& make, const float tolerance) {
            const Tensor metal = draw(GpuBackend::Metal, make);
            const Tensor vulkan = draw(GpuBackend::Vulkan, make);
            expect_close(metal, vulkan, tolerance, tolerance);
        };
        compare([] { return Tensor::rand({10007}, Device::GPU); }, 0.0f);
        compare([] { return Tensor::randint({10007}, -7, 1000, Device::GPU); }, 0.0f);
        compare([] { return Tensor::bernoulli({10007}, 0.3f, Device::GPU); }, 0.0f);
        compare([] { return Tensor::randn({10007}, Device::GPU); }, 1.0e-5f);
        const Tensor weights = random_tensor(50, 0.0f, 2.0f, 64);
        compare([&] { return Tensor::multinomial(weights.to(Device::GPU), 200, true); }, 0.0f);
        compare([&] { return Tensor::multinomial(weights.to(Device::GPU), 20, false); }, 0.0f);

        const Tensor uniform = draw(GpuBackend::Metal, [] { return Tensor::rand({100000}, Device::GPU); });
        EXPECT_NEAR(uniform.mean().item(), 0.5f, 0.01f);
        EXPECT_GE(uniform.min().item(), 0.0f);
        EXPECT_LT(uniform.max().item(), 1.0f);
        const Tensor normal = draw(GpuBackend::Metal, [] { return Tensor::randn({100000}, Device::GPU); });
        EXPECT_NEAR(normal.mean().item(), 0.0f, 0.02f);
        EXPECT_NEAR(normal.std().item(), 1.0f, 0.02f);
    }

    // Fused kernels compile to MSL with the lowerings of the SPIR-V emitter,
    // so Metal and Vulkan compute the same values.
    TEST_F(TensorMetal, FusedKernelsMatchVulkan) {
        if (!gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP() << "No Vulkan device";
        const auto run = [](const GpuBackend backend, const fused::Kernel& kernel, const std::vector<size_t>& domain,
                            const std::vector<Tensor>& inputs) {
            GpuBackendScope scope(backend);
            std::vector<Tensor> bound;
            for (const Tensor& input : inputs)
                bound.push_back(input.numel() > 4 ? input.to(Device::GPU) : input);
            std::vector<Tensor> outputs = kernel(domain, bound);
            for (Tensor& output : outputs)
                output = output.cpu();
            return outputs;
        };
        const auto compare = [&](const fused::Kernel& kernel, const std::vector<size_t>& domain,
                                 const std::vector<Tensor>& inputs, const float tolerance) {
            const auto metal = run(GpuBackend::Metal, kernel, domain, inputs);
            const auto vulkan = run(GpuBackend::Vulkan, kernel, domain, inputs);
            ASSERT_EQ(metal.size(), vulkan.size());
            for (size_t i = 0; i < metal.size(); ++i) {
                SCOPED_TRACE(i);
                expect_close(metal[i], vulkan[i], tolerance, tolerance);
            }
        };

        const Tensor x = random_tensor(37 * 53, -3.0f, 3.0f, 65).reshape({37, 53});
        const Tensor y = random_tensor(53, 0.1f, 2.0f, 66);
        const Tensor gain = Tensor::from_vector(std::vector<float>{1.5f, -0.25f}, {2}, Device::CPU);
        {
            // Exact arithmetic, comparisons, integer and bit operations.
            fused::Builder b(2);
            const auto a = b.input(DataType::Float32, 2).load();
            const auto c = b.input(DataType::Float32, 1).load();
            const auto g = b.input(DataType::Float32, 1);
            b.output(fused::fma(a, g.at({0}), c) - a / c + fused::min(a, c) * fused::max(a, 0.5f), DataType::Float32);
            b.output(fused::where(a > c, fused::clamp(a, -1.0f, 1.0f), fused::floor(a * 3.0f) + fused::round(a)),
                     DataType::Float32);
            const auto row = b.iota(0), column = b.iota(1);
            b.output((row * 7 - column) / 3 + (row - column) % 5 + (column << 2) - (row >> 1), DataType::Int32);
            b.output((a.cast(DataType::Int32) / (column % 4)) ^ row, DataType::Int32);
            const fused::Kernel kernel(b);
            // Narrow outputs pack into words.
            fused::Builder narrow(2);
            const auto p = narrow.input(DataType::Float32, 2).load();
            const auto q = narrow.input(DataType::Float32, 1).load();
            narrow.output(fused::abs(p) + fused::sign(p) * fused::square(q) - fused::relu(-p), DataType::Float16);
            narrow.output(fused::isfinite(p / (p - p)) || (p > 1.0f && q < 1.0f), DataType::Bool);
            narrow.output((p * 10.0f + 128.0f).cast(DataType::Int32), DataType::UInt8);
            const fused::Kernel narrow_kernel(narrow);
            compare(narrow_kernel, {37, 53}, {x, y}, 0.0f);
            compare(narrow_kernel, {7, 53}, {x.slice(0, 0, 7), y}, 0.0f);
            compare(kernel, {37, 53}, {x, y, gain}, 0.0f);
            // A second shape reuses the compiled layout class.
            compare(kernel, {5, 53}, {x.slice(0, 0, 5), y, gain}, 0.0f);
        }
        {
            // Transcendentals, to within their last bits.
            fused::Builder b(2);
            const auto a = b.input(DataType::Float32, 2).load();
            const auto c = b.input(DataType::Float32, 1).load();
            b.output(fused::exp(a) + fused::log(c) + fused::sqrt(c) + fused::sigmoid(a) + fused::tanh(a), DataType::Float32);
            b.output(fused::sin(a) * fused::cos(a) + fused::atan2(a, c) + fused::pow(c, a) + fused::log1p(fused::abs(a)),
                     DataType::Float32);
            b.output(fused::asin(a / 3.0f) + fused::acos(a / 3.0f) + fused::gelu(a) + fused::swish(a) + fused::exp2(a) +
                         fused::log2(c) + fused::log10(c) + fused::rsqrt(c) + fused::sinh(a) - fused::cosh(a),
                     DataType::Float32);
            const fused::Kernel kernel(b);
            compare(kernel, {37, 53}, {x, y}, 2.0e-5f);
        }
        {
            // Gathers under every bound and folds of a fixed and a run-time length.
            fused::Builder b(2);
            const auto source = b.input(DataType::Float32, 2);
            const auto a = source.load();
            const auto shifted = source.gather({b.iota(0) - 3, b.iota(1) * 2}, fused::Bounds::Zero) +
                                 source.gather({b.iota(0) + 1, b.iota(1) - 5}, fused::Bounds::Clamp) +
                                 source.gather({b.iota(0) - 40, b.iota(1) + 60}, fused::Bounds::Wrap);
            b.output(shifted, DataType::Float32);
            const fused::Kernel gathers(b);
            compare(gathers, {37, 53}, {x}, 0.0f);
            for (const size_t length : {size_t{16}, size_t{300}}) {
                fused::Builder folds(2);
                const auto values = folds.input(DataType::Float32, 2).load();
                folds.output(folds.fold(fused::Fold::Sum, values, 1), DataType::Float32);
                folds.output(folds.fold(fused::Fold::Max, values, 1), DataType::Float32);
                folds.output(folds.fold(fused::Fold::Count, values > 0.0f, 1), DataType::Int32);
                const fused::Kernel kernel(folds);
                const Tensor wide = random_tensor(9 * length, -3.0f, 3.0f, 67).reshape({9, static_cast<int>(length)});
                compare(kernel, {9, length}, {wide}, 1.0e-5f);
            }
        }
    }

    // Domain kernels port the Vulkan shaders; both backends of the Mac must agree.
    template <class Run>
    void expect_same_on_both(const Run& run, const float rtol = 0.0f, const float atol = 0.0f) {
        if (!gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP() << "No Vulkan device";
        const auto result = [&](const GpuBackend backend) {
            GpuBackendScope scope(backend);
            return run().cpu();
        };
        expect_close(result(GpuBackend::Metal), result(GpuBackend::Vulkan), rtol, atol);
    }

    TEST_F(TensorMetal, SpatialSelectionMatchesVulkan) {
        const Tensor points = random_tensor(4000 * 3, -2.0f, 2.0f, 68).reshape({4000, 3});
        const Tensor references = random_tensor(4000, 0.0f, 1.0f, 69) > 0.9f;
        expect_same_on_both([&] { return radius_neighbors(points.to(Device::GPU), references.to(Device::GPU), 0.15f); });

        PointProjection projection;
        projection.rotation = {0.8f, 0.0f, -0.6f, 0.0f, 1.0f, 0.0f, 0.6f, 0.0f, 0.8f};
        projection.translation = {0.1f, -0.2f, 5.0f};
        projection.focal_x = projection.focal_y = 500.0f;
        projection.center_x = 320.0f;
        projection.center_y = 240.0f;
        projection.width = 640;
        projection.height = 480;
        const Tensor transforms = Tensor::eye(4, Device::CPU).reshape({1, 4, 4});
        for (const PointProjectionModel model : {PointProjectionModel::Pinhole, PointProjectionModel::Orthographic,
                                                 PointProjectionModel::Equirectangular}) {
            SCOPED_TRACE(static_cast<int>(model));
            projection.model = model;
            expect_same_on_both([&] { return project_points(points.to(Device::GPU), projection); });
            expect_same_on_both([&] {
                const Tensor gpu_transforms = transforms.to(Device::GPU);
                return project_points(points.to(Device::GPU), projection, &gpu_transforms);
            });
        }

        const Tensor flat = random_tensor(5000 * 2, 0.0f, 100.0f, 70).reshape({5000, 2});
        const Tensor polygon = Tensor::from_vector(std::vector<float>{10, 10, 80, 20, 60, 90, 20, 70}, {4, 2}, Device::CPU);
        for (const PointRegion2DKind kind : {PointRegion2DKind::Disk, PointRegion2DKind::Rectangle,
                                             PointRegion2DKind::Polygon, PointRegion2DKind::Disks}) {
            SCOPED_TRACE(static_cast<int>(kind));
            const PointRegion2D region{.kind = kind, .x0 = 20, .y0 = 30, .x1 = 70, .y1 = 60, .radius = 15};
            expect_same_on_both([&] {
                Tensor mask = Tensor::zeros_bool({5000}, Device::GPU);
                const Tensor geometry = polygon.to(Device::GPU);
                mark_points_2d(mask, flat.to(Device::GPU), region, &geometry);
                return mask;
            });
        }
    }

    TEST_F(TensorMetal, SelectionEditsMatchVulkan) {
        constexpr size_t count = 6000;
        const Tensor points = random_tensor(count * 3, -3.0f, 3.0f, 77).reshape({count, 3});
        const Tensor nodes = random_tensor(count, -2.0f, 4.0f, 78).to(DataType::Int32);
        const Tensor allowed = Tensor::from_vector(std::vector<float>{1, 0, 1}, {3}, Device::CPU) > 0.5f;
        const Tensor transforms =
            Tensor::from_vector(std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                                                   0.8f, 0, -0.6f, 0.5f, 0, 1, 0, -0.25f, 0.6f, 0, 0.8f, 0.1f, 0, 0, 0, 1},
                                {2, 4, 4}, Device::CPU);
        const Tensor frame = Tensor::from_vector(
            std::vector<float>{0.9f, 0.1f, 0, 0, -0.1f, 0.9f, 0, 0, 0, 0, 1.2f, 0, 0.2f, -0.3f, 0.1f, 1}, {16}, Device::CPU);
        const Tensor lower = Tensor::from_vector(std::vector<float>{-1.5f, -1.0f, -2.0f}, {3}, Device::CPU);
        const Tensor upper = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 1.5f}, {3}, Device::CPU);
        const Tensor radii = Tensor::from_vector(std::vector<float>{2.0f, 1.5f, 2.5f}, {3}, Device::CPU);
        const Tensor initial = random_tensor(count, 0.0f, 1.0f, 79) > 0.1f;
        enum : unsigned { Nodes = 1,
                          Transforms = 2,
                          Shapes = 4 };
        const auto filtered = [&](const unsigned use, const PointFilterWindow* window) {
            expect_same_on_both([&] {
                const Tensor p = points.to(Device::GPU), t = transforms.to(Device::GPU), n = nodes.to(Device::GPU),
                             a = allowed.to(Device::GPU), f = frame.to(Device::GPU), lo = lower.to(Device::GPU),
                             hi = upper.to(Device::GPU), r = radii.to(Device::GPU);
                PointFilter filter{.indices = &n, .ellipsoid_inverse = true, .window = window};
                if (use & Nodes)
                    filter.allowed = &a;
                if (use & Transforms)
                    filter.transforms = &t;
                if (use & Shapes) {
                    filter.box_transform = filter.ellipsoid_transform = &f;
                    filter.box_min = &lo;
                    filter.box_max = &hi;
                    filter.ellipsoid_radii = &r;
                }
                Tensor mask = initial.to(Device::GPU);
                filter_points(mask, &p, filter);
                return mask;
            });
        };
        filtered(Nodes, nullptr);
        filtered(Transforms | Shapes, nullptr);
        PointFilterWindow window{.near_depth = 0.5f, .far_depth = 7.0f, .scale_x = 0.6f, .scale_y = 0.8f, .offset_x = 0.3f, .offset_y = -0.2f};
        window.projection.rotation = {0.8f, 0.0f, -0.6f, 0.0f, 1.0f, 0.0f, 0.6f, 0.0f, 0.8f};
        window.projection.translation = {0.1f, -0.2f, 5.0f};
        window.projection.focal_x = window.projection.focal_y = 500.0f;
        window.projection.center_x = 320.0f;
        window.projection.center_y = 240.0f;
        window.projection.ortho_scale = 100.0f;
        window.projection.width = 640;
        window.projection.height = 480;
        for (const PointProjectionModel model : {PointProjectionModel::Pinhole, PointProjectionModel::Orthographic,
                                                 PointProjectionModel::Equirectangular}) {
            SCOPED_TRACE(static_cast<int>(model));
            window.projection.model = model;
            filtered(Nodes | Transforms, &window);
        }

        constexpr size_t labels = 3000;
        const Tensor selected = random_tensor(count, 0.0f, 1.0f, 80) > 0.5f;
        const Tensor targets = random_tensor(count, -2.0f, 3100.0f, 81).to(DataType::Int32);
        const Tensor categories = random_tensor(count, -2.0f, 3.0f, 82).to(DataType::Int32);
        std::vector<float> lock_flags(256, 0.0f);
        lock_flags[2] = lock_flags[5] = 1.0f;
        const Tensor locked = Tensor::from_vector(lock_flags, {256}, Device::CPU) > 0.5f;
        for (const LabelUpdateMode mode : {LabelUpdateMode::Add, LabelUpdateMode::Remove, LabelUpdateMode::Replace}) {
            for (const bool indexed : {false, true}) {
                SCOPED_TRACE(static_cast<int>(mode) * 2 + indexed);
                const Tensor existing = random_tensor(indexed ? labels : count, 0.0f, 5.99f, 83).to(DataType::UInt8);
                expect_same_on_both([&] {
                    const Tensor s = selected.to(Device::GPU), e = existing.to(Device::GPU), l = locked.to(Device::GPU),
                                 i = targets.to(Device::GPU), c = categories.to(Device::GPU), a = allowed.to(Device::GPU);
                    Tensor output = Tensor::zeros({existing.numel()}, Device::GPU, DataType::UInt8);
                    update_labels(output, s,
                                  {.label = 3, .mode = mode, .existing = &e, .locked = &l, .indices = indexed ? &i : nullptr, .categories = &c, .allowed = &a});
                    return output;
                });
            }
        }
    }

    TEST_F(TensorMetal, ImageOperationsMatchVulkan) {
        const Tensor bytes = random_tensor(100003, 0.0f, 255.99f, 71).to(DataType::UInt8);
        expect_same_on_both([&] {
            Tensor counts = Tensor::zeros({256}, Device::GPU, DataType::Int32);
            histogram_u8(bytes.to(Device::GPU), counts);
            return counts;
        });

        const Tensor rgb = random_tensor(3 * 40 * 64, 0.0f, 1.0f, 72).reshape({3, 40, 64});
        PpispParams isp;
        isp.exposure_factor = 1.3f;
        for (int c = 0; c < 3; ++c) {
            const float vignetting[5] = {0.02f * c, -0.01f * c, -0.3f, 0.1f, -0.05f};
            const float crf[5] = {1.2f + 0.1f * c, 0.8f, 1.1f, 0.45f, 0.5f};
            std::copy(std::begin(vignetting), std::end(vignetting), isp.vignetting + 5 * c);
            std::copy(std::begin(crf), std::end(crf), isp.crf + 5 * c);
        }
        const float color_matrix[9] = {0.9f, 0.05f, 0.02f, 0.03f, 0.95f, 0.01f, -0.02f, 0.01f, 1.0f};
        std::copy(std::begin(color_matrix), std::end(color_matrix), isp.color_matrix);
        isp.y_offset = 12;
        isp.full_height = 60;
        expect_same_on_both([&] { return ppisp_apply(rgb.to(Device::GPU), isp); }, 2.0e-5f, 2.0e-5f);

        const Tensor alpha = random_tensor(40 * 64, 0.0f, 1.0f, 73).reshape({40, 64});
        const Tensor environment = random_tensor(16 * 32 * 3, 0.0f, 4.0f, 74).reshape({16, 32, 3});
        const float c = std::cos(0.5f), s = std::sin(0.5f);
        EnvironmentCompositeParams composite{.rotation = {c, 0, -s, 0, 1, 0, s, 0, c},
                                             .full_width = 64,
                                             .full_height = 60,
                                             .band_width = 64,
                                             .band_height = 40,
                                             .y_offset = 12,
                                             .focal_x = 50.0f,
                                             .focal_y = 45.0f,
                                             .center_x = 32.0f,
                                             .center_y = 30.0f,
                                             .exposure_factor = 1.5f,
                                             .env_rotation_radians = 0.7f,
                                             .env_width = 32,
                                             .env_height = 16};
        for (const int panorama : {0, 1}) {
            SCOPED_TRACE(panorama);
            composite.equirect_view = panorama;
            // Rounding to bytes may flip where the transcendentals differ in their last bit.
            expect_same_on_both([&] {
                return environment_composite(rgb.to(Device::GPU), alpha.to(Device::GPU), environment.to(Device::GPU),
                                             composite);
            },
                                0.0f, 1.0f);
        }
    }

    TEST_F(TensorMetal, ImageResamplingMatchesVulkan) {
        const Tensor image = random_tensor(3 * 45 * 61, -0.2f, 1.0f, 84).reshape({3, 45, 61});
        UndistortParams camera{.src_fx = 60.0f,
                               .src_fy = 58.0f,
                               .src_cx = 30.5f,
                               .src_cy = 22.5f,
                               .dst_fx = 55.0f,
                               .dst_fy = 52.0f,
                               .dst_cx = 28.0f,
                               .dst_cy = 20.0f,
                               .src_width = 61,
                               .src_height = 45,
                               .dst_width = 57,
                               .dst_height = 40,
                               .model_type = CameraModelType::PINHOLE,
                               .distortion = {0.12f, -0.03f, 0.004f, 0.002f, -0.003f, 0.001f, 0.002f, -0.001f, 0.001f, -0.002f},
                               .num_distortion = 10};
        for (const CameraModelType model :
             {CameraModelType::PINHOLE, CameraModelType::FISHEYE, CameraModelType::THIN_PRISM_FISHEYE}) {
            SCOPED_TRACE(static_cast<int>(model));
            camera.model_type = model;
            expect_same_on_both([&] { return internal::undistort_image_tensor(image.to(Device::GPU), camera, false); },
                                1.0e-5f, 1.0e-5f);
            expect_same_on_both([&] { return internal::undistort_image_tensor(image.slice(0, 0, 1).squeeze(0).to(Device::GPU), camera, true); },
                                1.0e-5f, 1.0e-5f);
        }
        // Negative depths and short normals are invalid taps.
        const Tensor depth = random_tensor(45 * 61, -0.5f, 3.0f, 85).reshape({45, 61});
        const Tensor normals = random_tensor(3 * 45 * 61, -1.0f, 1.0f, 86).reshape({3, 45, 61});
        expect_same_on_both([&] { return internal::resize_image_prior_tensor(depth.to(Device::GPU), 23, 97, false); },
                            1.0e-6f, 1.0e-6f);
        expect_same_on_both([&] { return internal::resize_image_prior_tensor(normals.to(Device::GPU), 23, 97, true); },
                            1.0e-6f, 1.0e-6f);
    }

    TEST_F(TensorMetal, ShCodecMatchesVulkan) {
        constexpr size_t rows = 1000, picked = 300;
        constexpr uint32_t rest = 15;
        const Tensor canonical = random_tensor(rows * rest * 3, -1.75f, 1.25f, 87).reshape({rows, size_t{rest}, 3});
        std::vector<float> order(picked);
        for (size_t i = 0; i < picked; ++i)
            order[i] = static_cast<float>(i * 7 % rows);
        const Tensor picks = Tensor::from_vector(order, {picked}, Device::CPU).to(DataType::Int32);
        const auto resident = [](const ShFormat format) {
            const size_t count = format == ShFormat::Q16 ? sh_value_quant::sh_value_u16_count(rows, rest)
                                                         : sh_swizzled_float_count(rows, rest);
            return Tensor::zeros({count}, Device::GPU, format == ShFormat::Float32 ? DataType::Float32 : DataType::Float16);
        };
        for (const ShFormat format : {ShFormat::Float32, ShFormat::Float16, ShFormat::Q16}) {
            for (const DataType index_type : {DataType::Int32, DataType::Int64}) {
                SCOPED_TRACE(static_cast<int>(format) * 2 + (index_type == DataType::Int64));
                // Pack all rows, decode them back, and gather rows at a lower degree.
                expect_same_on_both([&] {
                    const bool q16 = format == ShFormat::Q16;
                    Tensor packed = resident(format);
                    Tensor bounds = Tensor::zeros({sh_value_quant::n_bounds_for_prims(rows) * 2}, Device::GPU);
                    const ShCodec all{.source_rows = rows, .destination_rows = rows, .count = rows, .source_rest = rest, .destination_rest = rest};
                    ShCodec pack = all, unpack = all;
                    pack.source_format = ShFormat::Canonical;
                    pack.destination_format = unpack.source_format = format;
                    unpack.destination_format = ShFormat::Canonical;
                    sh_codec(canonical.to(Device::GPU), packed, pack, nullptr, nullptr, q16 ? &bounds : nullptr);
                    Tensor decoded = Tensor::zeros({rows, size_t{rest}, 3}, Device::GPU);
                    sh_codec(packed, decoded, unpack, nullptr, q16 ? &bounds : nullptr);
                    const Tensor ids = picks.to(index_type).to(Device::GPU);
                    Tensor gathered = Tensor::zeros({picked, 8, 3}, Device::GPU);
                    ShCodec gather = unpack;
                    gather.destination_rows = gather.count = picked;
                    gather.destination_rest = 8;
                    sh_codec(packed, gathered, gather, &ids, q16 ? &bounds : nullptr);
                    return Tensor::cat({packed.to(DataType::Float32), decoded.flatten(), gathered.flatten(), bounds}, 0);
                });
            }
        }
        // Scatter canonical rows, then copy a row range that ends the destination.
        expect_same_on_both([&] {
            Tensor packed = resident(ShFormat::Float32);
            const Tensor ids = picks.to(Device::GPU);
            sh_codec(canonical.slice(0, 0, picked).to(Device::GPU), packed,
                     {.source_format = ShFormat::Canonical, .destination_format = ShFormat::Float32, .source_rows = picked, .destination_rows = rows, .count = picked, .source_rest = rest, .destination_rest = rest, .scatter = true},
                     &ids);
            Tensor copy = Tensor::full({packed.numel()}, 2.0f, Device::GPU);
            sh_codec(packed, copy,
                     {.source_rows = rows, .destination_rows = rows, .count = 500, .source_rest = rest, .destination_rest = rest, .source_offset = 100, .destination_offset = 500});
            return Tensor::cat({packed, copy}, 0);
        });
    }

    // Metal and Vulkan share the portable neural-network ops; only their
    // inference kernels and matrix products differ.
    TEST_F(TensorMetal, NeuralNetworkOpsMatchVulkan) {
        namespace nn = lfs::core::nn;
        const auto gpu = [](const Tensor& host) { return host.to(Device::GPU); };
        const auto shaped = [](const size_t count, const unsigned seed, const TensorShape& shape) {
            return random_tensor(count, -1.0f, 1.0f, seed).reshape(shape);
        };
        const Tensor a = shaped(2 * 33 * 40, 101, {2, 33, 40}), w = shaped(24 * 40, 102, {24, 40});
        const Tensor bias = shaped(24, 103, {24}), residual = shaped(2 * 33 * 24, 104, {2, 33, 24});
        const Tensor scale = shaped(24, 105, {24});
        constexpr float matrix_tolerance = 1.0e-4f;
        expect_same_on_both([&] {
            const Tensor gb = gpu(bias), gr = gpu(residual), gs = gpu(scale);
            return nn::gemm(gpu(a), gpu(w), false, true, &gb, nn::Activation::GeluTanh, &gr, &gs);
        },
                            matrix_tolerance, matrix_tolerance);
        expect_same_on_both([&] {
            return nn::linear(gpu(a).to(DataType::Float16), gpu(w).to(DataType::Float16), nullptr, nn::Activation::Relu)
                .to(DataType::Float32);
        },
                            1.0e-2f, 1.0e-2f);
        const Tensor rows = shaped(6 * 32, 106, {6, 32}), gamma = shaped(32, 107, {32}), beta = shaped(32, 108, {32});
        expect_same_on_both([&] { return nn::layer_norm(gpu(rows), gpu(gamma), gpu(beta)); }, 1.0e-5f, 1.0e-5f);
        expect_same_on_both([&] { return nn::rms_norm(gpu(rows), gpu(gamma)); }, 1.0e-5f, 1.0e-5f);
        const Tensor logits = shaped(4 * 9 * 17, 109, {4, 9, 17}), mask = shaped(4 * 9 * 17, 110, {4, 9, 17});
        expect_same_on_both([&] {
            const Tensor gm = gpu(mask);
            return nn::softmax(gpu(logits), &gm);
        },
                            1.0e-5f, 1.0e-5f);
        const Tensor q = shaped(2 * 3 * 20 * 16, 111, {2, 3, 20, 16}), k = shaped(2 * 3 * 28 * 16, 112, {2, 3, 28, 16});
        const Tensor v = shaped(2 * 3 * 28 * 16, 113, {2, 3, 28, 16});
        expect_same_on_both([&] { return nn::attention(gpu(q), gpu(k), gpu(v)); }, matrix_tolerance, matrix_tolerance);

        const Tensor image = shaped(6 * 19 * 23, 114, {1, 6, 19, 23}), kernel = shaped(8 * 3 * 9, 115, {8, 3, 3, 3});
        const Tensor conv_bias = shaped(8, 116, {8});
        for (const nn::Conv2dParams params : {nn::Conv2dParams{.stride_h = 2, .stride_w = 2, .pad_h = 1, .pad_w = 1, .groups = 2},
                                              nn::Conv2dParams{.pad_h = 2, .pad_w = 2, .dilation_h = 2, .dilation_w = 2, .groups = 2, .pad_mode = nn::ConvPadMode::Replicate, .activation = nn::Activation::Silu}}) {
            SCOPED_TRACE(params.dilation_h);
            expect_same_on_both([&] {
                const Tensor gb = gpu(conv_bias);
                return nn::conv2d(gpu(image), gpu(kernel), &gb, params);
            },
                                matrix_tolerance, matrix_tolerance);
        }
        const Tensor small = shaped(4 * 7 * 9, 117, {1, 4, 7, 9}), up = shaped(4 * 3 * 4, 118, {4, 3, 2, 2});
        expect_same_on_both([&] { return nn::conv_transpose2d(gpu(small), gpu(up), nullptr, {.stride_h = 2, .stride_w = 2}); },
                            matrix_tolerance, matrix_tolerance);
        const Tensor picture = shaped(3 * 11 * 13, 119, {1, 3, 11, 13});
        for (const auto mode : {nn::ResizeMode::Nearest, nn::ResizeMode::Bilinear, nn::ResizeMode::Cubic}) {
            for (const auto coord : {nn::CoordTransform::HalfPixel, nn::CoordTransform::Asymmetric, nn::CoordTransform::AlignCorners}) {
                SCOPED_TRACE(static_cast<int>(mode) * 3 + static_cast<int>(coord));
                expect_same_on_both([&] { return nn::resize2d(gpu(picture), 17, 7, mode, coord); }, 1.0e-6f, 1.0e-6f);
            }
        }
        expect_same_on_both([&] { return nn::max_pool2d(gpu(picture), 3, 3, 2, 2, 1, 1); });
        for (const bool include_pad : {false, true})
            expect_same_on_both([&] { return nn::avg_pool2d(gpu(picture), 3, 2, 2, 1, 1, 0, include_pad); }, 1.0e-6f, 1.0e-6f);
        for (const auto approx : {nn::GELUApprox::Erf, nn::GELUApprox::Tanh})
            expect_same_on_both([&] { return nn::gelu(gpu(a), approx); }, 1.0e-6f, 1.0e-6f);
        expect_same_on_both([&] {
            const Tensor x = gpu(a);
            return Tensor::cat({nn::silu(x), nn::relu(x), nn::sigmoid(x)}, 0);
        },
                            1.0e-6f, 1.0e-6f);

        const Tensor bhwc = shaped(2 * 10 * 12 * 8, 120, {2, 10, 12, 8});
        expect_same_on_both([&] {
            const auto windows = nn::window_partition_2d(gpu(bhwc), 4);
            return nn::window_unpartition_2d(windows.windows, 4, windows.pad_h, windows.pad_w, 10, 12).sub(gpu(bhwc));
        });
        expect_same_on_both([&] { return nn::max_pool2d_bhwc(gpu(bhwc)); });
        const Tensor qkv = shaped(2 * 20 * 3 * 4 * 8, 121, {2, 20, 3 * 4 * 8});
        expect_same_on_both([&] {
            const auto [sq, sk, sv] = nn::split_qkv(gpu(qkv), 4);
            return Tensor::cat({nn::merge_heads(sq), nn::merge_heads(sk), nn::merge_heads(sv)}, 0);
        });
        const Tensor heads = shaped(2 * 3 * 24 * 5, 122, {2, 3, 24, 5});
        expect_same_on_both([&] { return nn::max_pool_heads_2d(gpu(heads), 4, 6); });
        const Tensor coords = random_tensor(5 * 2, 0.0f, 1.0f, 123).reshape({5, 2}), gaussian = shaped(2 * 8, 124, {2, 8});
        expect_same_on_both([&] { return nn::fourier_pe(gpu(coords), gpu(gaussian)); }, 1.0e-5f, 1.0e-5f);
        expect_same_on_both([&] { return nn::uv_grid(12, 16, 1.5f, DataType::Float32, Device::GPU, nullptr); }, 1.0e-6f, 1.0e-6f);
        expect_same_on_both([&] { return nn::residual_scale(gpu(rows), gpu(rows), gpu(gamma)); }, 1.0e-6f, 1.0e-6f);
    }

    // Metal's dedicated linear, attention and norm kernels at partial tiles,
    // odd head dims, broadcast masks and half precision.
    TEST_F(TensorMetal, NeuralNetworkKernelEdgesMatchVulkan) {
        namespace nn = lfs::core::nn;
        const auto gpu = [](const Tensor& host) { return host.to(Device::GPU); };
        const auto half = [&](const Tensor& host) { return gpu(host).to(DataType::Float16); };
        const auto shaped = [](const TensorShape& shape, const unsigned seed) {
            return random_tensor(shape.elements(), -1.0f, 1.0f, seed).reshape(shape);
        };
        constexpr float tolerance = 1.0e-4f, half_tolerance = 2.0e-2f;

        // Batched and shared weights, stored [k][n] and [n][k], with k off
        // every tile multiple.
        const Tensor a = shaped({3, 65, 37}, 201), shared = shaped({37, 70}, 202), batched = shaped({3, 70, 37}, 203);
        const Tensor bias = shaped({70}, 204);
        for (const auto activation : {nn::Activation::None, nn::Activation::Relu, nn::Activation::GeluErf,
                                      nn::Activation::Silu}) {
            SCOPED_TRACE(static_cast<int>(activation));
            expect_same_on_both([&] {
                const Tensor gb = gpu(bias);
                return nn::gemm(gpu(a), gpu(shared), false, false, &gb, activation);
            },
                                tolerance, tolerance);
        }
        expect_same_on_both([&] { return nn::bmm(gpu(a), gpu(batched), false, true); }, tolerance, tolerance);
        expect_same_on_both([&] { return nn::bmm(half(a), half(batched), false, true).to(DataType::Float32); },
                            half_tolerance, half_tolerance);
        const Tensor single = shaped({1, 1, 5}, 205), tiny = shaped({3, 5}, 206);
        expect_same_on_both([&] { return nn::linear(gpu(single), gpu(tiny)); }, tolerance, tolerance);

        // Head dims of 5, 72 and 128; keys and queries off the 8, 16 and 32 blocks.
        for (const int dim : {5, 72, 128}) {
            SCOPED_TRACE(dim);
            const Tensor q = shaped({2, 2, 45, static_cast<size_t>(dim)}, 207);
            const Tensor k = shaped({2, 2, 77, static_cast<size_t>(dim)}, 208);
            const Tensor v = shaped({2, 2, 77, static_cast<size_t>(dim)}, 209);
            expect_same_on_both([&] { return nn::attention(gpu(q), gpu(k), gpu(v)); }, tolerance, tolerance);
            expect_same_on_both([&] { return nn::attention(half(q), half(k), half(v)).to(DataType::Float32); },
                                half_tolerance, half_tolerance);
        }

        // Masks broadcast over batch and heads, and over keys of rank-2 masks.
        const Tensor q = shaped({2, 3, 19, 16}, 210), k = shaped({2, 3, 40, 16}, 211), v = shaped({2, 3, 40, 16}, 212);
        const Tensor full_mask = shaped({1, 3, 19, 40}, 213), row_mask = shaped({19, 40}, 214);
        for (const Tensor* const mask : {static_cast<const Tensor*>(&full_mask), &row_mask}) {
            SCOPED_TRACE(mask->ndim());
            expect_same_on_both([&] {
                const Tensor gm = gpu(*mask);
                return nn::attention(gpu(q), gpu(k), gpu(v), &gm, 0.3f);
            },
                                tolerance, tolerance);
            expect_same_on_both([&] {
                const Tensor gm = half(*mask);
                return nn::attention(half(q), half(k), half(v), &gm, 0.3f).to(DataType::Float32);
            },
                                half_tolerance, half_tolerance);
        }

        // Few queries over many keys split the keys; with a mask, whole splits
        // of a row can be masked out.
        const Tensor few = shaped({2, 2, 5, 40}, 227), many_keys = shaped({2, 2, 1000, 40}, 228);
        const Tensor many_values = shaped({2, 2, 1000, 40}, 229);
        auto split_mask = shaped({1, 1, 5, 1000}, 230);
        split_mask.slice(2, 1, 2).slice(3, 0, 700).fill_(-std::numeric_limits<float>::infinity());
        expect_same_on_both([&] { return nn::attention(gpu(few), gpu(many_keys), gpu(many_values)); }, tolerance,
                            tolerance);
        expect_same_on_both([&] {
            const Tensor gm = half(split_mask);
            return nn::attention(half(few), half(many_keys), half(many_values), &gm).to(DataType::Float32);
        },
                            half_tolerance, half_tolerance);

        // A query whose keys are all masked out attends to nothing, as on CUDA.
        auto blocked = Tensor::zeros({1, 1, 19, 40}, Device::CPU);
        blocked.slice(2, 4, 5).fill_(-std::numeric_limits<float>::infinity());
        expect_same_on_both([&] {
            const Tensor gm = gpu(blocked);
            return nn::attention(gpu(q), gpu(k), gpu(v), &gm);
        },
                            tolerance, tolerance);
        auto blocked_split = Tensor::zeros({1, 1, 5, 1000}, Device::CPU);
        blocked_split.slice(2, 3, 4).fill_(-std::numeric_limits<float>::infinity());
        expect_same_on_both([&] {
            const Tensor gm = gpu(blocked_split);
            return nn::attention(gpu(few), gpu(many_keys), gpu(many_values), &gm);
        },
                            tolerance, tolerance);

        // Convolutions over batches: grouped 1x1, a 3x3 with more output
        // channels than one tile whose patches exceed one 64 MiB chunk, and
        // implicit-GEMM ones with fewer.
        const Tensor images = shaped({2, 8, 13, 11}, 218), pointwise = shaped({6, 4, 1, 1}, 219);
        const Tensor pointwise_bias = shaped({6}, 220);
        expect_same_on_both([&] {
            const Tensor gb = gpu(pointwise_bias);
            return nn::conv2d(gpu(images), gpu(pointwise), &gb, {.groups = 2, .activation = nn::Activation::Relu});
        },
                            tolerance, tolerance);
        const Tensor large = shaped({2, 32, 200, 200}, 221), kernel = shaped({72, 32, 3, 3}, 222);
        const Tensor kernel_bias = shaped({72}, 223);
        expect_same_on_both([&] {
            const Tensor gb = gpu(kernel_bias);
            return nn::conv2d(gpu(large), gpu(kernel), &gb, {.pad_h = 1, .pad_w = 1});
        },
                            tolerance, tolerance);
        expect_same_on_both([&] {
            const Tensor gb = half(kernel_bias.slice(0, 0, 40));
            return nn::conv2d(half(images), half(kernel.slice(0, 0, 40).slice(1, 0, 8)), &gb,
                              {.stride_h = 2, .stride_w = 2, .pad_h = 1, .pad_w = 1})
                .to(DataType::Float32);
        },
                            half_tolerance, half_tolerance);

        // Transposed convolutions: grouped, padded and dilated, and SAM2's
        // 2x2 stride-2 upscaling.
        const Tensor up_grouped = shaped({8, 3, 3, 3}, 224), up_bias = shaped({6}, 225);
        expect_same_on_both([&] {
            const Tensor gb = gpu(up_bias);
            return nn::conv_transpose2d(gpu(images), gpu(up_grouped), &gb,
                                        {.stride_h = 2, .stride_w = 2, .pad_h = 1, .pad_w = 1, .dilation_h = 2, .groups = 2, .output_pad_h = 1, .activation = nn::Activation::GeluTanh});
        },
                            tolerance, tolerance);
        const Tensor upscale = shaped({8, 5, 2, 2}, 226);
        expect_same_on_both([&] {
            return nn::conv_transpose2d(half(images), half(upscale), nullptr, {.stride_h = 2, .stride_w = 2})
                .to(DataType::Float32);
        },
                            half_tolerance, half_tolerance);

        // The portable inference kernels read and write Float16 on Metal.
        for (const auto mode : {nn::ResizeMode::Nearest, nn::ResizeMode::Bilinear, nn::ResizeMode::Cubic}) {
            SCOPED_TRACE(static_cast<int>(mode));
            expect_same_on_both([&] {
                return nn::resize2d(half(images), 29, 17, mode, nn::CoordTransform::HalfPixel).to(DataType::Float32);
            },
                                half_tolerance, half_tolerance);
        }
        expect_same_on_both([&] { return nn::max_pool2d(half(images), 3, 3, 2, 2, 1, 1).to(DataType::Float32); });
        expect_same_on_both([&] { return nn::gelu(half(images)).to(DataType::Float32); }, half_tolerance,
                            half_tolerance);

        // Rows past a threadgroup's eight, and widths off the SIMD width.
        const Tensor rows = shaped({13, 3, 77}, 215), gamma = shaped({77}, 216), beta = shaped({77}, 217);
        expect_same_on_both([&] { return nn::layer_norm(gpu(rows), gpu(gamma), gpu(beta)); }, 1.0e-5f, 1.0e-5f);
        expect_same_on_both([&] { return nn::rms_norm(gpu(rows), gpu(gamma), 1.0e-3f); }, 1.0e-5f, 1.0e-5f);
        expect_same_on_both([&] {
            return nn::layer_norm(half(rows), half(gamma), half(beta)).to(DataType::Float32);
        },
                            half_tolerance, half_tolerance);
    }

    TEST_F(TensorMetal, SplatTransformMatchesVulkan) {
        constexpr size_t count = 5000;
        const Tensor scales = random_tensor(count * 3, -6.0f, 1.0f, 75).reshape({count, 3});
        const Tensor rotations = random_tensor(count * 4, -1.0f, 1.0f, 76).reshape({count, 4});
        for (const splat_transform::LinearTransform linear : {
                 splat_transform::LinearTransform{{2.0f, 0.3f, 0.0f, -0.2f, 0.5f, 0.1f, 0.0f, 0.4f, 1.5f}},
                 splat_transform::LinearTransform{{-1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}},
                 splat_transform::LinearTransform{{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
             }) {
            SCOPED_TRACE(linear.rows[0]);
            // The null axis of a singular map is the log of rounding noise, so
            // scales compare linearly.
            expect_same_on_both([&] {
                Tensor out_scales = Tensor::empty({count, 3}, Device::GPU);
                Tensor out_rotations = Tensor::empty({count, 4}, Device::GPU);
                affine_splat_geometry(linear, scales.to(Device::GPU), rotations.to(Device::GPU), out_scales, out_rotations);
                return Tensor::cat({out_scales.exp(), out_rotations}, 1);
            },
                                2.0e-5f, 2.0e-5f);
        }
    }

    // Swizzled SH: 32-point blocks of float4 cell groups.
    size_t swizzled_sh_index(const size_t point, const size_t dim, const size_t dims) {
        const size_t slots = (dims + 3) / 4;
        return ((point / 32) * (slots * 32) + (dim / 4) * 32 + point % 32) * 4 + dim % 4;
    }

    TEST_F(TensorMetal, MortonOrderMatchesCpu) {
        constexpr size_t count = 20000;
        const Tensor unique = random_tensor(count * 3, -5.0f, 5.0f, 88).reshape({count, 3});
        // Repeated positions keep their source order.
        const Tensor points = Tensor::cat({unique, unique.slice(0, 0, 1000)}, 0);
        Tensor expected_keys;
        const Tensor expected = morton_sort_indices(points, &expected_keys);
        for (const GpuBackend backend : {GpuBackend::Metal, GpuBackend::Vulkan}) {
            if (!gpu_backend_available(backend))
                continue;
            SCOPED_TRACE(gpu_backend_name(backend));
            GpuBackendScope scope(backend);
            Tensor keys;
            const Tensor order = morton_sort_indices(points.to(Device::GPU), &keys);
            EXPECT_EQ(order.to_vector_int(), expected.to_vector_int());
            EXPECT_EQ(keys.to_vector_int64(), expected_keys.to_vector_int64());
        }
    }

    // The exact assignment is an ordered FP32 argmin that the CPU reproduces.
    TEST_F(TensorMetal, Sh3AssignmentIsTheFp32Argmin) {
        constexpr size_t points = 2000, palette = 300, dims = 45;
        const Tensor sh = random_tensor((points + 31) / 32 * 12 * 32 * 4, -1.0f, 1.0f, 89);
        const Tensor centroids = random_tensor(palette * dims, -1.0f, 1.0f, 90).reshape({palette, dims});
        const Tensor norms = random_tensor(palette, 10.0f, 20.0f, 91);
        const auto values = sh.to_vector(), centers = centroids.to_vector(), lengths = norms.to_vector();
        std::vector<int> expected(points);
        for (size_t i = 0; i < points; ++i) {
            float best = 1e30f;
            for (size_t c = 0; c < palette; ++c) {
                float dot = 0;
                for (size_t d = 0; d < dims; ++d)
                    dot = std::fma(values[swizzled_sh_index(i, d, dims)], centers[c * dims + d], dot);
                const float distance = std::fma(-2.0f, dot, lengths[c]);
                if (distance < best) {
                    best = distance;
                    expected[i] = static_cast<int>(c);
                }
            }
        }
        for (const bool fast : {false, true}) {
            SCOPED_TRACE(fast);
            Tensor labels = to_metal(Tensor::zeros({points}, Device::CPU, DataType::Int32));
            assign_sh3(to_metal(sh), to_metal(centroids), to_metal(norms), labels, fast);
            EXPECT_EQ(labels.to_vector_int(), expected);
        }
    }

    // Screening must keep the FP32 argmin's winners: near ties, centroids that
    // round to one half value or exceed half's range, and seeds that are not
    // labels. Each point sits on a centroid or halfway between two.
    TEST_F(TensorMetal, ScreenedSh3AssignmentKeepsTheFp32ArgminAtHalfLimits) {
        constexpr size_t points = 257, palette = 1025, dims = 45;
        std::mt19937 rng(1741);
        std::uniform_real_distribution<float> jitter(-1.0f, 1.0f);
        for (const float scale : {1e-20f, 1e-8f, 1e-4f, 0.01f, 1.0f, 100.0f, 100000.0f}) {
            SCOPED_TRACE(scale);
            std::vector<float> sh((points + 31) / 32 * 12 * 32 * 4), centers(palette * dims), lengths(palette);
            for (size_t c = 0; c < palette; ++c) {
                for (size_t d = 0; d < dims; ++d) {
                    const float v = scale * (0.75f + jitter(rng) * 0.0001f);
                    centers[c * dims + d] = v;
                    lengths[c] = std::fma(v, v, lengths[c]);
                }
            }
            for (size_t i = 0; i < points; ++i) {
                for (size_t d = 0; d < dims; ++d) {
                    const float a = centers[(i * 17 % palette) * dims + d];
                    const float b = centers[((i * 17 + 1) % palette) * dims + d];
                    sh[swizzled_sh_index(i, d, dims)] = i % 2 ? a : (a + b) * 0.5f;
                }
            }
            const Tensor values = to_metal(Tensor::from_vector(sh, {sh.size()}, Device::CPU));
            const Tensor centroids = to_metal(Tensor::from_vector(centers, {palette, dims}, Device::CPU));
            const Tensor norms = to_metal(Tensor::from_vector(lengths, {palette}, Device::CPU));
            Tensor reference = to_metal(Tensor::zeros({points}, Device::CPU, DataType::Int32));
            Tensor screened = to_metal(Tensor::zeros({points}, Device::CPU, DataType::Int32));
            assign_sh3(values, centroids, norms, reference, false);
            assign_sh3(values, centroids, norms, screened, true);
            const auto expected = reference.to_vector_int();
            EXPECT_EQ(screened.to_vector_int(), expected);
            std::vector<int> seeds(points);
            for (size_t i = 0; i < points; ++i)
                seeds[i] = i % 3 == 0 ? -1 : i % 3 == 1 ? int(palette + 1)
                                                        : int(i % palette);
            Tensor seeded = to_metal(Tensor::from_vector(seeds, {points}, Device::CPU));
            assign_sh3(values, centroids, norms, seeded, true, true);
            EXPECT_EQ(seeded.to_vector_int(), expected);
        }
    }

    // Seeds are random, but the last update leaves every used centroid at the
    // mean of the points labelled with it. Both backends run the same driver.
    TEST_F(TensorMetal, PaletteCentroidsAreTheMeansOfTheirPoints) {
        for (const auto [points, coefficients, palette] :
             {std::tuple{3000, 3, 64}, std::tuple{3000, 15, 64}, std::tuple{8192, 15, 4096}}) {
            const size_t dims = size_t(coefficients) * 3;
            const Tensor sh = random_tensor((size_t(points) + 31) / 32 * ((dims + 3) / 4) * 32 * 4, -1.0f, 1.0f, 92);
            const auto values = sh.to_vector();
            for (const GpuBackend backend : {GpuBackend::Metal, GpuBackend::Vulkan}) {
                if (!gpu_backend_available(backend))
                    continue;
                SCOPED_TRACE(testing::Message() << gpu_backend_name(backend) << " palette " << palette << " dims " << dims);
                GpuBackendScope scope(backend);
                const auto [centroids, labels] = kmeans_sh(sh.to(Device::GPU), points, coefficients, palette, 4);
                ASSERT_TRUE(centroids.is_valid() && labels.is_valid());
                ASSERT_EQ(centroids.numel(), size_t(palette) * dims);
                const auto means = centroids.to_vector();
                const auto ids = labels.to_vector_int();
                std::vector<double> sums(size_t(palette) * dims);
                std::vector<int> counts(palette);
                for (size_t i = 0; i < size_t(points); ++i) {
                    ASSERT_TRUE(ids[i] >= 0 && ids[i] < palette) << i;
                    ++counts[ids[i]];
                    for (size_t d = 0; d < dims; ++d)
                        sums[ids[i] * dims + d] += values[swizzled_sh_index(i, d, dims)];
                }
                for (size_t c = 0; c < size_t(palette); ++c) {
                    for (size_t d = 0; counts[c] > 0 && d < dims; ++d)
                        ASSERT_NEAR(means[c * dims + d], sums[c * dims + d] / counts[c], 1e-5) << c;
                }
            }
        }
    }

    TEST_F(TensorMetal, DecimationMatchesVulkan) {
        if (!gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP() << "No Vulkan device";
        constexpr size_t count = 3000;
        constexpr int rest = 3;
        const Tensor position = random_tensor(count * 3, -2.0f, 2.0f, 93).reshape({count, 3});
        const Tensor rotation = random_tensor(count * 4, -1.0f, 1.0f, 94).reshape({count, 4});
        const Tensor scale = random_tensor(count * 3, -5.0f, -2.0f, 95).reshape({count, 3});
        const Tensor opacity = random_tensor(count, -2.0f, 2.0f, 96).reshape({count, 1});
        const Tensor dc = random_tensor(count * 3, -1.0f, 1.0f, 97).reshape({count, 3});
        const Tensor sh = random_tensor(count * rest * 3, -0.5f, 0.5f, 98).reshape({count, size_t{rest}, 3});
        // Groups of two to four splats, each kept at its first member.
        std::vector<int> member_group(count, -1);
        std::vector<uint32_t> minimum, members, offsets{0};
        size_t removed = 0;
        for (uint32_t group = 0; group < 100; ++group) {
            const uint32_t size = 2 + group % 3;
            minimum.push_back(10 * group);
            for (uint32_t m = 0; m < size; ++m) {
                members.push_back(10 * group + 2 * m);
                member_group[10 * group + 2 * m] = static_cast<int>(group);
            }
            offsets.push_back(static_cast<uint32_t>(members.size()));
            removed += size - 1;
        }
        const auto run = [&](const GpuBackend backend) {
            GpuBackendScope scope(backend);
            const Tensor p = position.to(Device::GPU), r = rotation.to(Device::GPU), s = scale.to(Device::GPU),
                         o = opacity.to(Device::GPU), c = dc.to(Device::GPU), h = sh.to(Device::GPU);
            std::vector<uint32_t> idx;
            std::vector<float> cost;
            decimate_candidates(p, r, s, o, c, h, rest, idx, cost);
            const auto merged = decimate_merge(p, r, s, o, c, h, rest, member_group, minimum, members, offsets, removed);
            const Tensor rows = Tensor::cat({merged.position.flatten(), merged.rotation.flatten(), merged.scale.flatten(),
                                             merged.opacity.flatten(), merged.dc.flatten(), merged.sh.flatten()},
                                            0)
                                    .cpu();
            return std::tuple{idx, cost, rows};
        };
        const auto [metal_idx, metal_cost, metal_rows] = run(GpuBackend::Metal);
        const auto [vulkan_idx, vulkan_cost, vulkan_rows] = run(GpuBackend::Vulkan);
        EXPECT_EQ(metal_idx, vulkan_idx);
        expect_close(Tensor::from_vector(metal_cost, {metal_cost.size()}, Device::CPU),
                     Tensor::from_vector(vulkan_cost, {vulkan_cost.size()}, Device::CPU), 1.0e-5f, 1.0e-5f);
        expect_close(metal_rows, vulkan_rows, 1.0e-5f, 1.0e-5f);
    }

    // LOD pool pages: packed pages in every encoding, with metadata, both
    // opacity modes and a partial page, then resident pages with float, half,
    // Q16 and no SH. The pool stores half bits, so the backends must agree.
    TEST_F(TensorMetal, RadPagesMatchVulkan) {
        if (!gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP() << "No Vulkan device";
        constexpr uint32_t splats = 65536, slots = 12;
        std::mt19937 generator(101);
        const auto random_bytes = [&](const size_t count) {
            std::vector<uint8_t> bytes(count);
            for (auto& byte : bytes)
                byte = static_cast<uint8_t>(generator());
            return bytes;
        };
        const auto float_bytes = [&](const size_t count, const float low, const float high, const bool half) {
            std::uniform_real_distribution<float> distribution(low, high);
            std::vector<uint8_t> bytes;
            for (size_t i = 0; i < count; ++i) {
                const float value = distribution(generator);
                const uint16_t bits = radmath::floatToHalf(value);
                const auto* source = half ? reinterpret_cast<const uint8_t*>(&bits) : reinterpret_cast<const uint8_t*>(&value);
                bytes.insert(bytes.end(), source, source + (half ? 2 : 4));
            }
            return bytes;
        };
        // Byte-planar encodings keep byte k of every element in plane k.
        const auto planar = [](const std::vector<uint8_t>& bytes, const size_t width) {
            const size_t count = bytes.size() / width;
            std::vector<uint8_t> result(bytes.size());
            for (size_t e = 0; e < count; ++e)
                for (size_t k = 0; k < width; ++k)
                    result[k * count + e] = bytes[e * width + k];
            return result;
        };
        struct Packet {
            RadPagePackedDesc desc;
            std::vector<uint8_t> planes;
            void add(const RadPackedKind kind, const RadPackedEncoding encoding, const std::vector<uint8_t>& plane,
                     const float low = 0.0f, const float high = 1.0f, const float scale = 1.0f) {
                planes.resize((planes.size() + 15) & ~size_t{15});
                desc.props[desc.property_count++] = {static_cast<uint32_t>(kind), static_cast<uint32_t>(encoding),
                                                     static_cast<uint32_t>(planes.size()),
                                                     static_cast<uint32_t>(plane.size()), low, high, scale};
                planes.insert(planes.end(), plane.begin(), plane.end());
                desc.used_bytes = static_cast<uint32_t>(planes.size());
            }
            Tensor tensor() const {
                Tensor bytes = Tensor::empty({sizeof(desc) + planes.size()}, Device::CPU, DataType::UInt8);
                std::memcpy(bytes.data_ptr(), &desc, sizeof(desc));
                std::memcpy(static_cast<uint8_t*>(bytes.data_ptr()) + sizeof(desc), planes.data(), planes.size());
                return bytes;
            }
        };
        using Kind = RadPackedKind;
        using Encoding = RadPackedEncoding;
        std::vector<Packet> packets(3);
        {
            auto& p = packets[0];
            const uint32_t n = p.desc.count = splats;
            p.desc.sh_coeffs_rest = 15;
            p.desc.frame = {{-3, 4, 2}, {2, 5, 8}, -4, 9};
            p.add(Kind::Means, Encoding::F32, float_bytes(n * 3, -10, 10, false));
            p.add(Kind::Alpha, Encoding::R8, random_bytes(n));
            p.add(Kind::Sh0, Encoding::F16, float_bytes(n * 3, 0, 1, true));
            p.add(Kind::Scales, Encoding::LnF16, float_bytes(n * 3, -6, 1, true));
            // Every axis pair once, with the angles spread.
            std::vector<uint8_t> rotation(n * 3);
            for (uint32_t i = 0; i < n; ++i) {
                rotation[i * 3] = i & 255;
                rotation[i * 3 + 1] = i >> 8;
                rotation[i * 3 + 2] = (i * 37) & 255;
            }
            p.add(Kind::Rotation, Encoding::Oct88R8, rotation);
            p.add(Kind::Sh1, Encoding::S8, random_bytes(n * 9), -0.5f, 0.5f, 0.5f);
            p.add(Kind::Sh2, Encoding::R8, random_bytes(n * 15), -0.3f, 0.4f);
            p.add(Kind::Sh3, Encoding::F32, float_bytes(n * 21, -0.2f, 0.2f, false));
            p.planes.resize((p.planes.size() + 15) & ~size_t{15});
            p.desc.meta_node_count = n;
            p.desc.meta_bounds_offset = static_cast<uint32_t>(p.planes.size());
            p.desc.meta_links_offset = p.desc.meta_bounds_offset + n * 8;
            const auto metadata = random_bytes(n * 20);
            p.planes.insert(p.planes.end(), metadata.begin(), metadata.end());
            p.desc.used_bytes = static_cast<uint32_t>(p.planes.size());
        }
        {
            auto& p = packets[1];
            const uint32_t n = p.desc.count = 40000;
            p.desc.sh_coeffs_rest = 8;
            p.desc.lod_opacity = 1;
            p.add(Kind::Means, Encoding::F32LeBytes, planar(float_bytes(n * 3, -10, 10, false), 4));
            p.add(Kind::Alpha, Encoding::F16LeBytes, planar(float_bytes(n, -0.5f, 1.5f, true), 2));
            p.add(Kind::Sh0, Encoding::R8, random_bytes(n * 3));
            p.add(Kind::Scales, Encoding::Ln0R8, random_bytes(n * 3), -8.0f, 2.0f);
            p.add(Kind::Rotation, Encoding::F16, float_bytes(n * 3, -0.7f, 0.7f, true));
            p.add(Kind::Sh1, Encoding::F16, float_bytes(n * 9, -0.5f, 0.5f, true), -0.5f, 0.5f);
            p.add(Kind::Sh2, Encoding::S8, random_bytes(n * 15), -0.3f, 0.2f, 0.1f);
            p.add(Kind::Sh3, Encoding::F32LeBytes, planar(float_bytes(n * 21, -0.2f, 0.2f, false), 4));
        }
        {
            auto& p = packets[2];
            const uint32_t n = p.desc.count = 1025;
            p.desc.sh_coeffs_rest = 3;
            p.add(Kind::Means, Encoding::F16, float_bytes(n * 3, -10, 10, true));
            p.add(Kind::Alpha, Encoding::F32, float_bytes(n, -0.1f, 1.1f, false));
            p.add(Kind::Scales, Encoding::F32, float_bytes(n * 3, -0.01f, 2.0f, false));
            p.add(Kind::Rotation, Encoding::F32, float_bytes(n * 3, -0.7f, 0.7f, false));
            p.add(Kind::Sh1, Encoding::R8, random_bytes(n * 9), -0.4f, 0.4f);
        }
        struct Resident {
            RadPageSources source;
            Tensor sh, bounds;
        };
        std::vector<Resident> residents;
        struct ShLayout {
            uint32_t rest;
            bool half, q16;
        };
        for (const auto [rest, half, q16] :
             {ShLayout{15, false, false}, ShLayout{8, true, false}, ShLayout{15, true, true}, ShLayout{0, false, false}}) {
            constexpr uint32_t offset = 100, count = 5000, rows = offset + count, tiles = (rows + 31) / 32 * 32;
            Resident r;
            r.source = {.means = random_tensor(rows * 3, -10, 10, 102),
                        .sh0 = random_tensor(rows * 3, -2, 2, 103),
                        .rotation = random_tensor(rows * 4, -1, 1, 104),
                        .scaling = random_tensor(rows * 3, -8, 1, 105),
                        .opacity = random_tensor(rows, -4, 4, 106),
                        .offset = offset,
                        .count = count,
                        .sh_rest = rest,
                        .sh_q16 = q16};
            const size_t values = tiles * (q16 ? rest * 3 : (rest * 3 + 3) / 4 * 4);
            r.sh = Tensor::empty({values}, Device::CPU, half ? DataType::Float16 : DataType::Float32);
            const auto bytes = q16 ? random_bytes(values * 2) : float_bytes(values, -0.8f, 0.8f, half);
            if (!bytes.empty())
                std::memcpy(r.sh.data_ptr(), bytes.data(), bytes.size());
            if (q16) {
                std::vector<float> bounds;
                for (uint32_t group = 0; group < (rows + 255) / 256; ++group)
                    bounds.insert(bounds.end(), {-0.5f - 0.01f * float(group), 0.25f + 0.02f * float(group)});
                r.bounds = Tensor::from_vector(bounds, {bounds.size()}, Device::CPU);
            }
            residents.push_back(std::move(r));
        }
        const uint32_t pages = static_cast<uint32_t>(packets.size() + residents.size());
        const auto run = [&](const GpuBackend backend) {
            GpuBackendScope scope(backend);
            const size_t n = size_t{pages} * splats;
            const std::array<size_t, 9> sizes{n * 12, n * 8, n * slots * 4, n * 8, n * 8, n * 2,
                                              pages * radq::kPageFrameBytes, n * 8, n * 12};
            RadPagePool pool{.page_splats = splats, .sh_slots = slots};
            for (size_t i = 0; i < sizes.size(); ++i)
                pool.regions[i] = Tensor::zeros({sizes[i]}, Device::GPU, DataType::UInt8);
            std::vector<Tensor> inputs;
            for (uint32_t page = 0; page < packets.size(); ++page) {
                inputs.push_back(packets[page].tensor().to(Device::GPU));
                rad_page_dequant(inputs.back(), packets[page].desc, pool, page);
            }
            for (uint32_t k = 0; k < residents.size(); ++k) {
                auto source = residents[k].source;
                for (Tensor* tensor : {&source.means, &source.sh0, &source.rotation, &source.scaling, &source.opacity})
                    *tensor = tensor->to(Device::GPU);
                source.shN = residents[k].sh.to(Device::GPU);
                if (residents[k].bounds.is_valid())
                    source.shN_bounds = residents[k].bounds.to(Device::GPU);
                inputs.insert(inputs.end(), {source.means, source.sh0, source.rotation, source.scaling,
                                             source.opacity, source.shN, source.shN_bounds});
                rad_page_quantize(source, pool, static_cast<uint32_t>(packets.size()) + k);
            }
            std::vector<std::vector<uint8_t>> regions;
            for (const auto& region : pool.regions) {
                const Tensor cpu = region.cpu();
                const auto* data = static_cast<const uint8_t*>(cpu.data_ptr());
                regions.emplace_back(data, data + cpu.bytes());
            }
            return regions;
        };
        const auto metal = run(GpuBackend::Metal);
        const auto vulkan = run(GpuBackend::Vulkan);
        // MoltenVK builds the Vulkan kernel with fast math: a product can lose
        // the sign of zero, and the Q16 SH decode can contract into an fma that
        // moves a rare quantized SH byte by one. Metal keeps IEEE results, like
        // CUDA and the file codec; everything else must match exactly.
        const auto signed_magnitude = [](const uint16_t half) {
            return (half & 0x8000u) != 0 ? -int(half & 0x7fffu) : int(half & 0x7fffu);
        };
        size_t rounded = 0;
        for (size_t region = 0; region < metal.size(); ++region) {
            ASSERT_EQ(metal[region].size(), vulkan[region].size());
            const bool halves = region == 1 || (region >= 3 && region <= 5);
            size_t mismatches = 0;
            for (size_t i = 0; i < metal[region].size(); i += halves ? 2 : 1) {
                if (halves) {
                    uint16_t a = 0, b = 0;
                    std::memcpy(&a, &metal[region][i], 2);
                    std::memcpy(&b, &vulkan[region][i], 2);
                    if (signed_magnitude(a) == signed_magnitude(b))
                        continue;
                } else if (metal[region][i] == vulkan[region][i]) {
                    continue;
                } else if (region == 2 && std::abs(int(int8_t(metal[region][i])) - int(int8_t(vulkan[region][i]))) == 1) {
                    ++rounded;
                    continue;
                }
                if (mismatches++ < 4)
                    ADD_FAILURE() << "region=" << region << " byte=" << i << " metal=" << int(metal[region][i])
                                  << " vulkan=" << int(vulkan[region][i]);
            }
            EXPECT_EQ(mismatches, 0u) << "region " << region;
        }
        EXPECT_LT(rounded, 64u);
    }

} // namespace
