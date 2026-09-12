/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Integer pointwise arithmetic stays integer, counts stay exact, every max and
// min path shares one identity and one NaN and signed-zero policy, byte
// scatter-add is atomic, and normal draws advance the seed, on both backends.

#include "core/tensor.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor_backend.hpp"
#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    using namespace lfs::core;

    constexpr float kInf = std::numeric_limits<float>::infinity();
    constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

    std::vector<GpuBackend> backends_under_test() {
        std::vector<GpuBackend> backends{GpuBackend::CUDA};
        if (gpu_backend_available(GpuBackend::Vulkan)) {
            backends.push_back(GpuBackend::Vulkan);
        }
        return backends;
    }

    std::string label(const GpuBackend backend) {
        return backend == GpuBackend::CUDA ? "cuda" : "vulkan";
    }

    Tensor upload(const Tensor& cpu, const GpuBackend backend) {
        GpuBackendScope scope(backend);
        return cpu.to(Device::CUDA);
    }

    Tensor upload_int(const std::vector<int>& values, const GpuBackend backend) {
        return upload(Tensor::from_vector(values, TensorShape{values.size()}, Device::CPU), backend);
    }

    Tensor upload_float(const std::vector<float>& values, const TensorShape& shape,
                        const GpuBackend backend) {
        return upload(Tensor::from_vector(values, shape, Device::CPU), backend);
    }

    int reference_pow(const int base, const int exponent) {
        if (exponent < 0) {
            if (base == 1)
                return 1;
            if (base == -1)
                return (exponent % 2) != 0 ? -1 : 1;
            return 0;
        }
        uint64_t result = 1;
        uint64_t factor = static_cast<uint64_t>(static_cast<int64_t>(base));
        for (int i = 0; i < exponent; ++i)
            result *= factor;
        return static_cast<int>(static_cast<uint32_t>(result));
    }

    uint32_t bits(const float value) { return std::bit_cast<uint32_t>(value); }
} // namespace

TEST(TensorIntegerPolicy, AbsAndSquareAreExactAboveTwentyFourBits) {
    const std::vector<int> values{INT_MIN, INT_MIN + 1, -16777217, -16777216, -1,
                                  0, 1, 16777217, 46341, INT_MAX};
    std::vector<int> expected_abs;
    std::vector<int> expected_square;
    for (const int v : values) {
        expected_abs.push_back(v == INT_MIN ? INT_MIN : std::abs(v));
        expected_square.push_back(static_cast<int>(static_cast<uint32_t>(
            static_cast<uint64_t>(static_cast<int64_t>(v)) * static_cast<uint64_t>(static_cast<int64_t>(v)))));
    }
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor t = upload_int(values, backend);
        EXPECT_EQ(t.abs().to_vector_int(), expected_abs);
        EXPECT_EQ(t.square().to_vector_int(), expected_square);
    }
}

TEST(TensorIntegerPolicy, ClampIsExactAboveTwentyFourBits) {
    const std::vector<int> values{INT_MIN, -33554433, -33554432, -33554431,
                                  33554431, 33554432, 33554433, INT_MAX};
    const std::vector<int> expected{-33554432, -33554432, -33554432, -33554431,
                                    33554431, 33554432, 33554432, 33554432};
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor t = upload_int(values, backend);
        EXPECT_EQ(t.clamp(-33554432.0f, 33554432.0f).to_vector_int(), expected);
        Tensor inplace = t.clone();
        inplace.clamp_(-33554432.0f, 33554432.0f);
        EXPECT_EQ(inplace.to_vector_int(), expected);
    }
}

TEST(TensorIntegerPolicy, PowIsWrapAroundIntegerArithmetic) {
    const std::vector<int> bases{3, -2, 7, 16777217, 1, -1, 0, 2, 1291, 46341, -7, 5};
    const std::vector<int> exponents{19, 31, 11, 1, -5, -3, 0, 30, 3, 2, -1, -2};
    std::vector<int> expected_tensor;
    std::vector<int> expected_cubed;
    for (size_t i = 0; i < bases.size(); ++i) {
        expected_tensor.push_back(reference_pow(bases[i], exponents[i]));
        expected_cubed.push_back(reference_pow(bases[i], 3));
    }
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor base = upload_int(bases, backend);
        const Tensor exponent = upload_int(exponents, backend);
        EXPECT_EQ(base.pow(exponent).to_vector_int(), expected_tensor);
        EXPECT_EQ(base.pow(3).to_vector_int(), expected_cubed);
        EXPECT_EQ(base.pow(3.0f).dtype(), DataType::Float32);
    }
}

TEST(TensorCountPolicy, CountNonzeroIsExactAboveTwentyFourBits) {
    const size_t count = (size_t{1} << 24) + 1001;
    std::vector<float> floats(count, 1.0f);
    floats[7] = 0.0f;
    floats[count / 2] = 0.0f;
    floats[count - 1] = 0.0f;
    std::vector<bool> flags(count, true);
    flags[11] = false;
    flags[count - 2] = false;
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor f = upload_float(floats, TensorShape{count}, backend);
        EXPECT_EQ(f.count_nonzero(), count - 3);
        const Tensor b = upload(Tensor::from_vector(flags, TensorShape{count}, Device::CPU), backend);
        EXPECT_EQ(b.count_nonzero(), count - 2);
    }
}

TEST(TensorReducePolicy, ColumnAndStridedMaxMinKeepInfiniteAxes) {
    constexpr size_t rows = 4096;
    constexpr size_t cols = 64;
    std::vector<float> matrix(rows * cols);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            matrix[r * cols + c] = static_cast<float>((r * 7 + c * 13) % 101) - 50.0f;
        }
    }
    for (size_t r = 0; r < rows; ++r) {
        matrix[r * cols + 5] = -kInf;
        matrix[r * cols + 9] = kInf;
    }
    std::vector<float> expected_max(cols, -kInf);
    std::vector<float> expected_min(cols, kInf);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            expected_max[c] = std::max(expected_max[c], matrix[r * cols + c]);
            expected_min[c] = std::min(expected_min[c], matrix[r * cols + c]);
        }
    }

    constexpr size_t outer = 4;
    constexpr size_t inner = 8;
    std::vector<float> block(outer * rows * inner);
    for (size_t i = 0; i < block.size(); ++i) {
        block[i] = static_cast<float>((i * 31) % 97) - 48.0f;
    }
    for (size_t r = 0; r < rows; ++r) {
        block[(2 * rows + r) * inner + 3] = -kInf;
        block[(1 * rows + r) * inner + 6] = kInf;
    }
    std::vector<float> expected_block_max(outer * inner, -kInf);
    std::vector<float> expected_block_min(outer * inner, kInf);
    for (size_t o = 0; o < outer; ++o) {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t i = 0; i < inner; ++i) {
                const float v = block[(o * rows + r) * inner + i];
                expected_block_max[o * inner + i] = std::max(expected_block_max[o * inner + i], v);
                expected_block_min[o * inner + i] = std::min(expected_block_min[o * inner + i], v);
            }
        }
    }

    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor m = upload_float(matrix, TensorShape{rows, cols}, backend);
        EXPECT_EQ(m.max(0).to_vector(), expected_max);
        EXPECT_EQ(m.min(0).to_vector(), expected_min);
        const Tensor b = upload_float(block, TensorShape{outer, rows, inner}, backend);
        EXPECT_EQ(b.max(1).to_vector(), expected_block_max);
        EXPECT_EQ(b.min(1).to_vector(), expected_block_min);
        const Tensor all_negative_infinite = upload_float(std::vector<float>(300000, -kInf), TensorShape{300000}, backend);
        EXPECT_EQ(all_negative_infinite.max_scalar(), -kInf);
        EXPECT_EQ(all_negative_infinite.max().to_vector()[0], -kInf);
        EXPECT_EQ(all_negative_infinite.min_scalar(), -kInf);
    }
}

TEST(TensorReducePolicy, ScalarMaxAndMinShareTheReducePolicy) {
    constexpr size_t count = 300007;
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = static_cast<float>((i * 17) % 1009) * 0.125f - 60.0f;
    }
    std::vector<float> with_nan = values;
    with_nan[count / 3] = kNaN;
    const std::vector<float> zeros{-0.0f, 0.0f, -0.0f, 0.0f, -0.0f};
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor t = upload_float(with_nan, TensorShape{count}, backend);
        EXPECT_TRUE(std::isnan(t.max_scalar()));
        EXPECT_TRUE(std::isnan(t.min_scalar()));
        EXPECT_TRUE(std::isnan(t.max().to_vector()[0]));
        EXPECT_TRUE(std::isnan(t.min().to_vector()[0]));

        const Tensor z = upload_float(zeros, TensorShape{zeros.size()}, backend);
        EXPECT_EQ(bits(z.max_scalar()), 0x00000000u);
        EXPECT_EQ(bits(z.min_scalar()), 0x80000000u);
        EXPECT_EQ(bits(z.max().to_vector()[0]), 0x00000000u);
        EXPECT_EQ(bits(z.min().to_vector()[0]), 0x80000000u);

        const Tensor plain = upload_float(values, TensorShape{count}, backend);
        const float expected_max = *std::max_element(values.begin(), values.end());
        const float expected_min = *std::min_element(values.begin(), values.end());
        EXPECT_EQ(plain.max_scalar(), expected_max);
        EXPECT_EQ(plain.min_scalar(), expected_min);
    }
}

// Byte scatter-add is only reachable through the facade (the public scatter_
// routes additions to index_add_), so the entry is driven directly.
TEST(TensorScatterPolicy, ByteScatterAddWithDuplicateIndicesIsAtomic) {
    constexpr size_t count = 1000;
    std::vector<int> indices(count, 0);
    for (size_t i = 700; i < count; ++i)
        indices[i] = 1;
    const std::vector<uint8_t> expected{static_cast<uint8_t>(700 % 256), static_cast<uint8_t>(300 % 256), 0, 0};
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        Tensor out = upload_int(std::vector<int>(4, 0), backend).to(DataType::UInt8);
        const Tensor idx = upload_int(indices, backend);
        const Tensor src = upload_int(std::vector<int>(count, 1), backend).to(DataType::UInt8);
        internal::backend_ops_for(out).scatter(
            internal::storage_ref(out), internal::storage_ref(idx), internal::storage_ref(src),
            internal::strided_layout(out), internal::strided_layout(src),
            internal::IndexProgram{.dim = 0, .scatter_mode = 1, .total_elements = count},
            internal::ExecContext{out.stream()});
        EXPECT_EQ(out.to_vector_uint8(), expected);
    }
}

TEST(TensorMovementPolicy, Int32CopiesIntoFloatViewsOfEveryRank) {
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        {
            Tensor dst = upload_float(std::vector<float>(3 * 4, 0.0f), TensorShape{3, 4}, backend);
            Tensor view = dst.slice(1, 0, 2);
            std::vector<int> ints(3 * 2);
            std::iota(ints.begin(), ints.end(), 100);
            view.copy_from(upload(Tensor::from_vector(ints, TensorShape{3, 2}, Device::CPU), backend));
            const auto got = dst.to_vector();
            for (size_t r = 0; r < 3; ++r) {
                EXPECT_EQ(got[r * 4 + 0], static_cast<float>(100 + r * 2));
                EXPECT_EQ(got[r * 4 + 1], static_cast<float>(101 + r * 2));
                EXPECT_EQ(got[r * 4 + 2], 0.0f);
                EXPECT_EQ(got[r * 4 + 3], 0.0f);
            }
        }
        {
            Tensor dst = upload_float(std::vector<float>(2 * 3 * 4, 0.0f), TensorShape{2, 3, 4}, backend);
            Tensor view = dst.slice(2, 0, 2);
            std::vector<int> ints(2 * 3 * 2);
            std::iota(ints.begin(), ints.end(), -5);
            view.copy_from(upload(Tensor::from_vector(ints, TensorShape{2, 3, 2}, Device::CPU), backend));
            const auto got = dst.to_vector();
            size_t k = 0;
            for (size_t a = 0; a < 2; ++a) {
                for (size_t b = 0; b < 3; ++b) {
                    EXPECT_EQ(got[(a * 3 + b) * 4 + 0], static_cast<float>(ints[k++]));
                    EXPECT_EQ(got[(a * 3 + b) * 4 + 1], static_cast<float>(ints[k++]));
                    EXPECT_EQ(got[(a * 3 + b) * 4 + 2], 0.0f);
                    EXPECT_EQ(got[(a * 3 + b) * 4 + 3], 0.0f);
                }
            }
        }
    }
}

TEST(TensorRandomPolicy, NormalDrawsAdvanceBetweenCalls) {
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        for (const size_t count : {size_t{1024}, size_t{1023}}) {
            Tensor t = upload_float(std::vector<float>(count, 0.0f), TensorShape{count}, backend);
            t.normal_(0.0f, 1.0f);
            const auto first = t.to_vector();
            t.normal_(0.0f, 1.0f);
            const auto second = t.to_vector();
            size_t equal = 0;
            double sum = 0.0;
            double sum_squares = 0.0;
            for (size_t i = 0; i < count; ++i) {
                equal += first[i] == second[i] ? 1 : 0;
                sum += second[i];
                sum_squares += static_cast<double>(second[i]) * second[i];
            }
            EXPECT_LT(equal, count / 100);
            const double mean = sum / static_cast<double>(count);
            const double variance = sum_squares / static_cast<double>(count) - mean * mean;
            EXPECT_NEAR(mean, 0.0, 0.15);
            EXPECT_NEAR(variance, 1.0, 0.25);
        }
    }
}

TEST(TensorMaskPolicy, LeadingAxisMasksBroadcastInMaskedFillAndExpand) {
    constexpr size_t channels = 3, rows = 4, cols = 5;
    std::vector<bool> mask_values(rows * cols, false);
    mask_values[1] = true;
    mask_values[7] = true;
    mask_values[rows * cols - 1] = true;
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor mask = upload(Tensor::from_vector(mask_values, TensorShape{rows, cols}, Device::CPU), backend);
        const Tensor expanded = mask.unsqueeze(0).expand({static_cast<int>(channels), static_cast<int>(rows), static_cast<int>(cols)}).contiguous();
        const auto expanded_values = expanded.to_vector_bool();
        ASSERT_EQ(expanded_values.size(), channels * rows * cols);
        for (size_t c = 0; c < channels; ++c)
            for (size_t i = 0; i < rows * cols; ++i)
                EXPECT_EQ(expanded_values[c * rows * cols + i], mask_values[i]) << "channel=" << c << " index=" << i;

        const Tensor filled = upload_float(std::vector<float>(channels * rows * cols, 1.0f), TensorShape{channels, rows, cols}, backend)
                                  .masked_fill(mask.unsqueeze(0), 0.5f);
        const auto filled_values = filled.to_vector();
        for (size_t c = 0; c < channels; ++c)
            for (size_t i = 0; i < rows * cols; ++i)
                EXPECT_EQ(filled_values[c * rows * cols + i], mask_values[i] ? 0.5f : 1.0f) << "channel=" << c << " index=" << i;
    }
}

TEST(TensorMaskPolicy, DeferredComparisonMasksBroadcastInMaskedFill) {
    constexpr size_t channels = 3, rows = 4, cols = 5;
    std::vector<float> scores(rows * cols);
    for (size_t i = 0; i < scores.size(); ++i)
        scores[i] = static_cast<float>(i % 3) * 0.4f;
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor score = upload_float(scores, TensorShape{rows, cols}, backend);
        const Tensor mask = score.isfinite().logical_and(score > 0.5f).logical_not().unsqueeze(0);
        const Tensor filled = upload_float(std::vector<float>(channels * rows * cols, 1.0f), TensorShape{channels, rows, cols}, backend)
                                  .masked_fill(mask, 0.5f);
        const auto filled_values = filled.to_vector();
        for (size_t c = 0; c < channels; ++c)
            for (size_t i = 0; i < rows * cols; ++i)
                EXPECT_EQ(filled_values[c * rows * cols + i], scores[i] > 0.5f ? 1.0f : 0.5f) << "channel=" << c << " index=" << i;
    }
}

TEST(TensorBroadcastPolicy, LeadingAxisDivisionBroadcastsOverEveryChannel) {
    constexpr size_t channels = 3, rows = 4, cols = 5;
    std::vector<float> numerator(channels * rows * cols);
    for (size_t i = 0; i < numerator.size(); ++i)
        numerator[i] = static_cast<float>(i % 17) + 1.0f;
    std::vector<float> denominator(rows * cols);
    for (size_t i = 0; i < denominator.size(); ++i)
        denominator[i] = static_cast<float>(i % 5) + 2.0f;
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor n = upload_float(numerator, TensorShape{channels, rows, cols}, backend);
        const Tensor d = upload_float(denominator, TensorShape{rows, cols}, backend);
        const Tensor squared_length = (n * n).sum(0).sqrt();
        ASSERT_EQ(squared_length.shape(), TensorShape({rows, cols}));
        const auto quotient = n.div(d.unsqueeze(0)).to_vector();
        const auto scaled = n.div(d.unsqueeze(0)).mul(0.5f).add(0.5f).to_vector();
        for (size_t c = 0; c < channels; ++c) {
            for (size_t i = 0; i < rows * cols; ++i) {
                const float expected = numerator[c * rows * cols + i] / denominator[i];
                EXPECT_NEAR(quotient[c * rows * cols + i], expected, 2.0e-7f * expected) << "channel=" << c << " index=" << i;
                EXPECT_NEAR(scaled[c * rows * cols + i], expected * 0.5f + 0.5f, 2.0e-7f * expected) << "channel=" << c << " index=" << i;
            }
        }
    }
}

TEST(TensorBroadcastPolicy, ClampAndMaskedFillAfterABroadcastDivisionKeepEveryChannel) {
    constexpr size_t channels = 3, rows = 4, cols = 5;
    std::vector<float> values(channels * rows * cols);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.3f;
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor n = upload_float(values, TensorShape{channels, rows, cols}, backend);
        const Tensor length = (n * n).sum(0).sqrt();
        const Tensor degenerate = length.isfinite().logical_and(length > 1.0e-6f).logical_not().unsqueeze(0);
        const Tensor color = n.div(length.unsqueeze(0)).mul(0.5f).add(0.5f).clamp(0.0f, 1.0f).masked_fill(degenerate, 0.5f);
        const auto got = color.cpu().contiguous().to_vector();
        ASSERT_EQ(got.size(), values.size());
        for (size_t i = 0; i < rows * cols; ++i) {
            const float x = values[i], y = values[rows * cols + i], z = values[2 * rows * cols + i];
            const float len = std::sqrt(x * x + y * y + z * z);
            const std::array<float, 3> expected = len > 1.0e-6f
                                                      ? std::array<float, 3>{std::clamp(x / len * 0.5f + 0.5f, 0.0f, 1.0f), std::clamp(y / len * 0.5f + 0.5f, 0.0f, 1.0f), std::clamp(z / len * 0.5f + 0.5f, 0.0f, 1.0f)}
                                                      : std::array<float, 3>{0.5f, 0.5f, 0.5f};
            for (size_t c = 0; c < channels; ++c)
                EXPECT_NEAR(got[c * rows * cols + i], expected[c], 1.0e-6f) << "channel=" << c << " index=" << i;
        }
    }
}

TEST(TensorBroadcastPolicy, FullRangeLeadingSliceKeepsEveryChannelThroughTheDisplayChain) {
    constexpr size_t channels = 3, rows = 61, cols = 83;
    std::vector<float> values(channels * rows * cols);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.3f;
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        Tensor n = upload_float(values, TensorShape{channels, rows, cols}, backend).to(DataType::Float32);
        n = n.slice(0, 0, 3).contiguous();
        EXPECT_EQ(n.to_vector(), values);
        const Tensor length = (n * n).sum(0).sqrt();
        const Tensor degenerate = length.isfinite().logical_and(length > 1.0e-6f).logical_not().unsqueeze(0);
        const Tensor color = n.div(length.unsqueeze(0)).mul(0.5f).add(0.5f).clamp(0.0f, 1.0f).masked_fill(degenerate, 0.5f);
        const auto got = color.cpu().contiguous().to_vector();
        for (size_t i = 0; i < rows * cols; ++i) {
            const float x = values[i], y = values[rows * cols + i], z = values[2 * rows * cols + i];
            const float len = std::sqrt(x * x + y * y + z * z);
            for (size_t c = 0; c < channels; ++c) {
                const float component = c == 0 ? x : c == 1 ? y
                                                            : z;
                const float expected = len > 1.0e-6f ? std::clamp(component / len * 0.5f + 0.5f, 0.0f, 1.0f) : 0.5f;
                EXPECT_NEAR(got[c * rows * cols + i], expected, 1.0e-6f) << "channel=" << c << " index=" << i;
            }
        }
    }
}

TEST(TensorBroadcastPolicy, DeferredLeadingBroadcastMaskFillsEveryChannel) {
    constexpr size_t channels = 3, rows = 64, cols = 65;
    std::vector<float> scores(rows * cols);
    for (size_t i = 0; i < scores.size(); ++i)
        scores[i] = static_cast<float>(i % 3) * 0.4f;
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor score = upload_float(scores, TensorShape{rows, cols}, backend);
        const Tensor mask = score.isfinite().logical_and(score > 0.5f).logical_not().unsqueeze(0);
        const Tensor filled = upload_float(std::vector<float>(channels * rows * cols, 1.0f),
                                           TensorShape{channels, rows, cols}, backend)
                                  .masked_fill(mask, 0.5f);
        const auto filled_values = filled.to_vector();
        for (size_t c = 0; c < channels; ++c)
            for (size_t i = 0; i < rows * cols; ++i)
                EXPECT_EQ(filled_values[c * rows * cols + i], scores[i] > 0.5f ? 1.0f : 0.5f)
                    << "channel=" << c << " index=" << i;
    }
}

TEST(TensorBroadcastPolicy, DisplayChainStagesAtImageSize) {
    constexpr size_t channels = 3, rows = 61, cols = 83;
    const size_t pixels = rows * cols;
    std::vector<float> values(channels * pixels);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.3f + 0.05f;
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor n = upload_float(values, TensorShape{channels, rows, cols}, backend);
        const Tensor length = (n * n).sum(0).sqrt();
        const auto len = length.to_vector();
        const Tensor quotient = n.div(length.unsqueeze(0));
        const auto q = quotient.to_vector();
        const Tensor scaled = quotient.mul(0.5f).add(0.5f);
        const auto sc = scaled.to_vector();
        const Tensor clamped = scaled.clamp(0.0f, 1.0f);
        const auto cl = clamped.to_vector();
        const Tensor mask = length.isfinite().logical_and(length > 1.0e-6f).logical_not().unsqueeze(0);
        const auto filled = clamped.masked_fill(mask, 0.5f).to_vector();
        size_t bad_q = 0, bad_sc = 0, bad_cl = 0, bad_fill = 0;
        for (size_t c = 0; c < channels; ++c) {
            for (size_t i = 0; i < pixels; ++i) {
                const size_t k = c * pixels + i;
                const float expected_q = values[k] / len[i];
                bad_q += std::abs(q[k] - expected_q) > 1.0e-5f;
                bad_sc += std::abs(sc[k] - (expected_q * 0.5f + 0.5f)) > 1.0e-5f;
                bad_cl += std::abs(cl[k] - std::clamp(expected_q * 0.5f + 0.5f, 0.0f, 1.0f)) > 1.0e-5f;
                bad_fill += std::abs(filled[k] - std::clamp(expected_q * 0.5f + 0.5f, 0.0f, 1.0f)) > 1.0e-5f;
            }
        }
        EXPECT_EQ(bad_q, 0u) << "division";
        EXPECT_EQ(bad_sc, 0u) << "mul add";
        EXPECT_EQ(bad_cl, 0u) << "clamp";
        EXPECT_EQ(bad_fill, 0u) << "masked_fill";
    }
}

TEST(TensorViewAs, Float32ToUInt8AppendsPackedDimAndRoundTripsBytes) {
    const std::vector<float> values{1.0f, -2.0f, 3.5f, 0.0f};
    std::vector<uint8_t> expected(values.size() * 4);
    std::memcpy(expected.data(), values.data(), expected.size());
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor src = upload_float(values, TensorShape{values.size()}, backend);
        const Tensor viewed = src.view_as(DataType::UInt8);
        EXPECT_EQ(gpu_backend_of(viewed), gpu_backend_of(src));
        EXPECT_EQ(viewed.stream(), src.stream());
        EXPECT_EQ(viewed.data_ptr(), src.data_ptr());
        ASSERT_EQ(viewed.dtype(), DataType::UInt8);
        ASSERT_EQ(viewed.ndim(), 2u);
        EXPECT_EQ(viewed.size(0), values.size());
        EXPECT_EQ(viewed.size(1), 4u);
        EXPECT_EQ(viewed.to_vector_uint8(), expected);
        const Tensor back = viewed.view_as(DataType::Float32);
        EXPECT_EQ(back.data_ptr(), src.data_ptr());
        EXPECT_EQ(back.to_vector(), values);
    }
}

TEST(TensorViewAs, UInt8PairToFloat16DropsTrailingDimAndRoundTripsBytes) {
    const std::vector<uint8_t> bytes{0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        Tensor host = Tensor::empty(TensorShape{3, 2}, Device::CPU, DataType::UInt8);
        std::memcpy(host.data_ptr(), bytes.data(), bytes.size());
        const Tensor src = upload(host, backend);
        const Tensor viewed = src.view_as(DataType::Float16);
        EXPECT_EQ(gpu_backend_of(viewed), gpu_backend_of(src));
        EXPECT_EQ(viewed.stream(), src.stream());
        EXPECT_EQ(viewed.data_ptr(), src.data_ptr());
        ASSERT_EQ(viewed.dtype(), DataType::Float16);
        ASSERT_EQ(viewed.ndim(), 1u);
        EXPECT_EQ(viewed.size(0), 3u);
        const Tensor back = viewed.view_as(DataType::UInt8);
        EXPECT_EQ(back.to_vector_uint8(), bytes);
    }
}

TEST(TensorViewAs, RejectsNonContiguousAndUnalignedGrow) {
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor src = upload_float(std::vector<float>(16, 1.0f), TensorShape{4, 4}, backend);
        const Tensor sliced = src.slice(1, 0, 2);
        EXPECT_FALSE(sliced.is_contiguous());
        EXPECT_THROW(sliced.view_as(DataType::UInt8), std::runtime_error);

        Tensor host = Tensor::empty(TensorShape{8}, Device::CPU, DataType::UInt8);
        std::vector<uint8_t> bytes(8, 7);
        std::memcpy(host.data_ptr(), bytes.data(), bytes.size());
        const Tensor u8 = upload(host, backend);
        EXPECT_THROW(u8.view_as(DataType::Float16), std::runtime_error);
        EXPECT_THROW(u8.view_as(DataType::Float32), std::runtime_error);
    }
}
