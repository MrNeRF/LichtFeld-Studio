/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nn/ops.hpp"
#include "core/tensor.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {
    using namespace lfs::core;

    Tensor upload_vulkan(const Tensor& cpu) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        return cpu.to(Device::GPU);
    }

    // Positive values in [0.05, 1] so accumulations never cancel and a relative
    // bound is meaningful; the salt decorrelates operands.
    std::vector<float> positive_pattern(const size_t count, const size_t salt) {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) {
            values[i] = 0.05f + 0.95f * static_cast<float>(((i + salt) * 7919) % 1000) / 1000.0f;
        }
        return values;
    }

    std::vector<float> signed_pattern(const size_t count, const size_t salt) {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) {
            values[i] = static_cast<float>(static_cast<int>(((i + salt) * 7919) % 41) - 20) / 20.0f;
        }
        return values;
    }

    double bound(const double expected, const size_t terms) {
        const double rtol = 1.0e-5 * std::max(1.0, std::log2(static_cast<double>(std::max<size_t>(2, terms))));
        return 1.0e-6 + rtol * std::abs(expected);
    }

    Tensor upload(const std::vector<float>& values, const std::vector<size_t>& shape) {
        return upload_vulkan(Tensor::from_vector(values, TensorShape(shape), Device::CPU));
    }

    void expect_matrix(const Tensor& actual, const std::vector<double>& expected,
                       const size_t terms, const std::string& label) {
        ASSERT_EQ(gpu_backend_of(actual), GpuBackend::Vulkan) << label;
        const std::vector<float> values = actual.cpu().to_vector();
        ASSERT_EQ(values.size(), expected.size()) << label;
        for (size_t i = 0; i < values.size(); ++i) {
            ASSERT_NEAR(values[i], expected[i], bound(expected[i], terms)) << label << " index=" << i;
        }
    }

    // C[m][n] = A[m][k] * B[k][n] with B optionally stored as [n][k].
    std::vector<double> matmul_reference(const std::vector<float>& a, const std::vector<float>& b,
                                         const size_t m, const size_t k, const size_t n,
                                         const bool b_transposed) {
        std::vector<double> c(m * n, 0.0);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t l = 0; l < k; ++l) {
                    const double rhs = b_transposed ? b[j * k + l] : b[l * n + j];
                    sum += static_cast<double>(a[i * k + l]) * rhs;
                }
                c[i * n + j] = sum;
            }
        }
        return c;
    }

    class TensorVulkanMatrixNn : public testing::Test {
    protected:
        void SetUp() override {
            ASSERT_TRUE(gpu_backend_available(GpuBackend::Vulkan));
        }

        void TearDown() override {
            const auto status = shutdown_gpu_backend(GpuBackend::Vulkan);
            EXPECT_TRUE(status.has_value());
            for (const std::string& message :
                 internal::vulkan_validation_messages_for_testing()) {
                ADD_FAILURE() << message;
            }
        }
    };

    TEST_F(TensorVulkanMatrixNn, MatmulCoversTileEdgesAndPartialTiles) {
        // Catches a tile loader that reads past the matrix edge, a stale shared
        // tile between k steps, and a store that skips partial tiles.
        struct Case {
            size_t m, k, n;
        };
        const std::array cases{
            Case{1, 1, 1}, Case{7, 3, 5}, Case{64, 16, 64}, Case{65, 17, 63},
            Case{130, 33, 200}, Case{300, 257, 129}, Case{16, 1024, 8}};
        for (const Case& test : cases) {
            const std::vector<float> a = positive_pattern(test.m * test.k, 1);
            const std::vector<float> b = positive_pattern(test.k * test.n, 2);
            const Tensor product = upload(a, {test.m, test.k}).mm(upload(b, {test.k, test.n}));
            expect_matrix(product, matmul_reference(a, b, test.m, test.k, test.n, false), test.k,
                          "mm " + std::to_string(test.m) + "x" + std::to_string(test.k) + "x" +
                              std::to_string(test.n));
        }
        const std::vector<float> a = signed_pattern(70 * 129, 3);
        const std::vector<float> b = signed_pattern(129 * 66, 4);
        const std::vector<float> values = upload(a, {70, 129}).mm(upload(b, {129, 66})).cpu().to_vector();
        const std::vector<double> expected = matmul_reference(a, b, 70, 129, 66, false);
        for (size_t i = 0; i < values.size(); ++i) {
            ASSERT_NEAR(values[i], expected[i], 1.0e-4) << "signed mm index=" << i;
        }
    }

    TEST_F(TensorVulkanMatrixNn, LinearUsesTheTransposedWeightLayoutAndAddsBiasPerColumn) {
        // Catches sgemm_tn indexing B as [k][n] instead of [n][k], and a bias add
        // that uses the wrong channel stride.
        constexpr size_t batch = 37;
        constexpr size_t in_features = 33;
        constexpr size_t out_features = 65;
        const std::vector<float> input = positive_pattern(batch * in_features, 5);
        const std::vector<float> weight = positive_pattern(out_features * in_features, 6);
        const std::vector<float> bias = signed_pattern(out_features, 7);
        const Tensor x = upload(input, {batch, in_features});
        const Tensor w = upload(weight, {out_features, in_features});
        const Tensor b = upload(bias, {out_features});
        const std::vector<double> plain =
            matmul_reference(input, weight, batch, in_features, out_features, true);
        expect_matrix(x.linear(w, Tensor{}), plain, in_features, "linear");
        std::vector<double> biased(plain);
        std::vector<double> rectified(plain);
        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < out_features; ++j) {
                biased[i * out_features + j] += bias[j];
                rectified[i * out_features + j] = std::max(0.0, biased[i * out_features + j]);
            }
        }
        expect_matrix(x.linear(w, b), biased, in_features, "linear+bias");
        Tensor output;
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            output = Tensor::zeros({batch, out_features}, Device::GPU);
        }
        x.linear_bias_relu_out(w, b, output);
        expect_matrix(output, rectified, in_features, "linear_bias_relu_out");
    }

    TEST_F(TensorVulkanMatrixNn, SmallDepthAndWidthCoverEveryGemmLayout) {
        // k and n of at most 16 take the element-per-thread path: catches a
        // transposed B read as [k][n], a bias on the wrong axis, and the edges
        // of its shared B table.
        struct Case {
            size_t m, k, n;
        };
        const std::array cases{Case{1000, 3, 3}, Case{257, 16, 16}, Case{33, 16, 1}, Case{5, 1, 16}};
        for (const Case& test : cases) {
            const std::string label = std::to_string(test.m) + "x" + std::to_string(test.k) + "x" + std::to_string(test.n);
            const std::vector<float> input = signed_pattern(test.m * test.k, 10);
            const std::vector<float> weight = positive_pattern(test.k * test.n, 11);
            const std::vector<float> bias = signed_pattern(test.n, 12);
            const Tensor x = upload(input, {test.m, test.k});
            expect_matrix(x.mm(upload(weight, {test.k, test.n})),
                          matmul_reference(input, weight, test.m, test.k, test.n, false), test.k, "mm " + label);
            const Tensor w = upload(weight, {test.n, test.k});
            const std::vector<double> plain = matmul_reference(input, weight, test.m, test.k, test.n, true);
            expect_matrix(x.linear(w, Tensor{}), plain, test.k, "linear " + label);
            std::vector<double> rectified(plain);
            for (size_t i = 0; i < test.m; ++i)
                for (size_t j = 0; j < test.n; ++j)
                    rectified[i * test.n + j] = std::max(0.0, plain[i * test.n + j] + bias[j]);
            Tensor output;
            {
                GpuBackendScope scope(GpuBackend::Vulkan);
                output = Tensor::zeros({test.m, test.n}, Device::GPU);
            }
            x.linear_bias_relu_out(w, upload(bias, {test.n}), output);
            expect_matrix(output, rectified, test.k, "linear_bias_relu_out " + label);
            // The fused epilogue adds one bias per output row.
            const std::vector<float> row_bias = signed_pattern(test.m, 13);
            const Tensor b = upload(weight, {test.k, test.n});
            const Tensor r = upload(row_bias, {test.m});
            std::vector<double> fused = matmul_reference(input, weight, test.m, test.k, test.n, false);
            for (size_t i = 0; i < test.m; ++i)
                for (size_t j = 0; j < test.n; ++j)
                    fused[i * test.n + j] = std::max(0.0, fused[i * test.n + j] + row_bias[i]);
            internal::backend_ops(GpuBackend::Vulkan)
                .sgemm_bias_relu(internal::storage_ref(x), internal::storage_ref(b), internal::storage_ref(r),
                                 internal::storage_ref(output),
                                 internal::GemmProgram{.m = test.m, .n = test.n, .k = test.k}, {});
            expect_matrix(output, fused, test.k, "sgemm_bias_relu " + label);
        }
    }

    TEST_F(TensorVulkanMatrixNn, BatchedMatmulStridesEveryBatch) {
        // Catches a batched GEMM that reads batch 0 for every batch or applies the
        // batch stride to the wrong operand.
        struct Case {
            size_t batch, m, k, n;
        };
        const std::array cases{Case{16, 3, 5, 7}, Case{3, 70, 33, 65}};
        for (const Case& test : cases) {
            const std::vector<float> a = positive_pattern(test.batch * test.m * test.k, 8);
            const std::vector<float> b = positive_pattern(test.batch * test.k * test.n, 9);
            std::vector<double> expected;
            for (size_t batch = 0; batch < test.batch; ++batch) {
                const std::vector<float> a_batch(a.begin() + batch * test.m * test.k,
                                                 a.begin() + (batch + 1) * test.m * test.k);
                const std::vector<float> b_batch(b.begin() + batch * test.k * test.n,
                                                 b.begin() + (batch + 1) * test.k * test.n);
                const std::vector<double> product =
                    matmul_reference(a_batch, b_batch, test.m, test.k, test.n, false);
                expected.insert(expected.end(), product.begin(), product.end());
            }
            const Tensor result =
                upload(a, {test.batch, test.m, test.k}).bmm(upload(b, {test.batch, test.k, test.n}));
            expect_matrix(result, expected, test.k, "bmm batch=" + std::to_string(test.batch));
        }
    }

    TEST_F(TensorVulkanMatrixNn, Conv1x1BiasReluTakesBothHostPaths) {
        // Catches the fused GEMM epilogue adding the bias per column instead of
        // per output channel, and the separate bias_relu kernel's channel stride.
        struct Case {
            size_t channels_in, channels_out, height, width;
        };
        const std::array cases{Case{4, 4, 800, 200}, Case{3, 5, 17, 29}};
        for (const Case& test : cases) {
            const size_t spatial = test.height * test.width;
            const std::vector<float> input = signed_pattern(test.channels_in * spatial, 10);
            const std::vector<float> weight = signed_pattern(test.channels_out * test.channels_in, 11);
            const std::vector<float> bias = signed_pattern(test.channels_out, 12);
            const Tensor x = upload(input, {1, test.channels_in, test.height, test.width});
            const Tensor w = upload(weight, {test.channels_out, test.channels_in});
            const Tensor b = upload(bias, {test.channels_out});
            Tensor output;
            {
                GpuBackendScope scope(GpuBackend::Vulkan);
                output = Tensor::zeros({1, test.channels_out, test.height, test.width}, Device::GPU);
            }
            x.conv1x1_bias_relu_out(w, b, output);
            std::vector<double> expected =
                matmul_reference(weight, input, test.channels_out, test.channels_in, spatial, false);
            for (size_t channel = 0; channel < test.channels_out; ++channel) {
                for (size_t s = 0; s < spatial; ++s) {
                    expected[channel * spatial + s] =
                        std::max(0.0, expected[channel * spatial + s] + bias[channel]);
                }
            }
            const std::vector<float> values = output.cpu().to_vector();
            ASSERT_EQ(values.size(), expected.size());
            for (size_t i = 0; i < values.size(); ++i) {
                ASSERT_NEAR(values[i], expected[i], 1.0e-5) << "spatial=" << spatial << " index=" << i;
            }
        }
    }

    TEST_F(TensorVulkanMatrixNn, DotProductCoversSingleGroupAndPartialPaths) {
        // Catches a dot product that drops the tail past the last full workgroup or
        // reads partials with the wrong count.
        constexpr std::array sizes{size_t{1}, size_t{7}, size_t{2048}, size_t{2049}, size_t{1000003}};
        for (const size_t count : sizes) {
            const std::vector<float> a = positive_pattern(count, 13);
            const std::vector<float> b = positive_pattern(count, 14);
            double expected = 0.0;
            for (size_t i = 0; i < count; ++i) {
                expected += static_cast<double>(a[i]) * b[i];
            }
            const Tensor result = upload(a, {count}).dot(upload(b, {count}));
            EXPECT_EQ(gpu_backend_of(result), GpuBackend::Vulkan);
            EXPECT_NEAR(result.item(), expected, bound(expected, count)) << "n=" << count;
        }
    }

    TEST_F(TensorVulkanMatrixNn, EyeWritesEveryElement) {
        // Catches kernels that leave off-diagonal elements untouched or index the
        // diagonal by column.
        struct EyeCase {
            size_t rows, columns;
        };
        for (const EyeCase& test : {EyeCase{5, 9}, EyeCase{9, 5}, EyeCase{1, 1}, EyeCase{300, 300}}) {
            Tensor identity;
            {
                GpuBackendScope scope(GpuBackend::Vulkan);
                identity = Tensor::eye(test.rows, test.columns, Device::GPU);
            }
            EXPECT_EQ(gpu_backend_of(identity), GpuBackend::Vulkan);
            const std::vector<float> values = identity.cpu().to_vector();
            ASSERT_EQ(values.size(), test.rows * test.columns);
            for (size_t i = 0; i < values.size(); ++i) {
                const bool diagonal = i / test.columns == i % test.columns;
                ASSERT_EQ(values[i], diagonal ? 1.0f : 0.0f) << "eye index=" << i;
            }
        }
    }
    TEST_F(TensorVulkanMatrixNn, PoolingMatchesTheCudaWindowRules) {
        // Catches wrong padding handling in max_pool2d, adaptive windows that
        // drop the last row or column, and a max that loses NaN.
        constexpr size_t batch = 2;
        constexpr size_t channels = 3;
        constexpr size_t height = 17;
        constexpr size_t width = 29;
        std::vector<float> input = signed_pattern(batch * channels * height * width, 18);
        input[(1 * channels + 2) * height * width + 4 * width + 7] = std::numeric_limits<float>::quiet_NaN();
        const Tensor x = upload(input, {batch, channels, height, width});
        constexpr int kernel = 3;
        constexpr int stride = 2;
        constexpr int padding = 1;
        const size_t out_height = (height + 2 * padding - kernel) / stride + 1;
        const size_t out_width = (width + 2 * padding - kernel) / stride + 1;
        const std::vector<float> pooled = x.max_pool2d(kernel, stride, padding).cpu().to_vector();
        ASSERT_EQ(pooled.size(), batch * channels * out_height * out_width);
        for (size_t n = 0; n < batch; ++n) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t oh = 0; oh < out_height; ++oh) {
                    for (size_t ow = 0; ow < out_width; ++ow) {
                        float best = -std::numeric_limits<float>::infinity();
                        bool nan = false;
                        for (int kh = 0; kh < kernel; ++kh) {
                            const int ih = static_cast<int>(oh) * stride - padding + kh;
                            if (ih < 0 || ih >= static_cast<int>(height))
                                continue;
                            for (int kw = 0; kw < kernel; ++kw) {
                                const int iw = static_cast<int>(ow) * stride - padding + kw;
                                if (iw < 0 || iw >= static_cast<int>(width))
                                    continue;
                                const float value = input[((n * channels + c) * height + ih) * width + iw];
                                nan = nan || std::isnan(value);
                                best = std::max(best, value);
                            }
                        }
                        const float actual = pooled[((n * channels + c) * out_height + oh) * out_width + ow];
                        if (nan) {
                            ASSERT_TRUE(std::isnan(actual)) << "n=" << n << " c=" << c << " oh=" << oh << " ow=" << ow;
                        } else {
                            ASSERT_EQ(actual, best) << "n=" << n << " c=" << c << " oh=" << oh << " ow=" << ow;
                        }
                    }
                }
            }
        }
        std::vector<float> clean = signed_pattern(batch * channels * height * width, 19);
        const Tensor y = upload(clean, {batch, channels, height, width});
        constexpr size_t adaptive_height = 3;
        constexpr size_t adaptive_width = 5;
        const std::vector<float> averaged = y.adaptive_avg_pool2d(adaptive_height, adaptive_width).cpu().to_vector();
        ASSERT_EQ(averaged.size(), batch * channels * adaptive_height * adaptive_width);
        for (size_t n = 0; n < batch; ++n) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t oh = 0; oh < adaptive_height; ++oh) {
                    for (size_t ow = 0; ow < adaptive_width; ++ow) {
                        const size_t h_start = (oh * height) / adaptive_height;
                        const size_t h_end = ((oh + 1) * height + adaptive_height - 1) / adaptive_height;
                        const size_t w_start = (ow * width) / adaptive_width;
                        const size_t w_end = ((ow + 1) * width + adaptive_width - 1) / adaptive_width;
                        double sum = 0.0;
                        size_t count = 0;
                        for (size_t h = h_start; h < h_end; ++h) {
                            for (size_t w = w_start; w < w_end; ++w) {
                                sum += clean[((n * channels + c) * height + h) * width + w];
                                ++count;
                            }
                        }
                        ASSERT_NEAR(averaged[((n * channels + c) * adaptive_height + oh) * adaptive_width + ow],
                                    sum / static_cast<double>(count), 1.0e-5)
                            << "n=" << n << " c=" << c << " oh=" << oh << " ow=" << ow;
                    }
                }
            }
        }
    }

    TEST_F(TensorVulkanMatrixNn, ReluOutZeroesNaNLikeFmaxf) {
        // Catches a relu that propagates NaN where the CUDA kernel's fmaxf returns 0.
        std::vector<float> values = signed_pattern(4099, 20);
        values[4098] = std::numeric_limits<float>::quiet_NaN();
        values[0] = -0.0f;
        const Tensor x = upload(values, {4099});
        Tensor output;
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            output = Tensor::zeros({4099}, Device::GPU);
        }
        x.relu_out(output);
        const std::vector<float> result = output.cpu().to_vector();
        ASSERT_EQ(result.size(), values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            const float expected = values[i] > 0.0f ? values[i] : 0.0f;
            ASSERT_EQ(result[i], expected) << "index=" << i;
        }
    }

    TEST_F(TensorVulkanMatrixNn, FullyMaskedRowsAttendToNothingLikeCuda) {
        // Rows masked out entirely come back zero, as from the CUDA kernels,
        // instead of NaN; the other rows stay normalized.
        constexpr float kMasked = -std::numeric_limits<float>::infinity();
        std::vector<float> mask(5 * 40, 0.0f);
        std::fill(mask.begin() + 2 * 40, mask.begin() + 3 * 40, kMasked);
        const Tensor gpu_mask = upload(mask, {1, 1, 5, 40});
        const Tensor q = upload(signed_pattern(5 * 16, 1), {1, 1, 5, 16});
        const Tensor k = upload(signed_pattern(40 * 16, 2), {1, 1, 40, 16});
        const Tensor v = upload(positive_pattern(40 * 16, 3), {1, 1, 40, 16});
        const std::vector<float> attended = lfs::core::nn::attention(q, k, v, &gpu_mask).cpu().to_vector();
        for (size_t i = 0; i < attended.size(); ++i) {
            if (i / 16 == 2)
                ASSERT_EQ(attended[i], 0.0f) << "index=" << i;
            else
                ASSERT_GT(attended[i], 0.0f) << "index=" << i;
        }
    }

} // namespace
