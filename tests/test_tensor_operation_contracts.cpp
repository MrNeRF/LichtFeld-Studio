/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/nn/ops.hpp"
#include "tensor_backend_contract.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
    using namespace lfs::core;

    class TensorOperationContracts : public lfs::test::TensorBackendContract {
    protected:
        Tensor values(const std::vector<float>& data, const TensorShape& shape) const {
            return Tensor::from_vector(data, shape, device());
        }

        void expect_values(const Tensor& result, const std::vector<float>& expected, const float tolerance = 0.f) const {
            expect_placement(result);
            const auto actual = result.to_vector();
            ASSERT_EQ(actual.size(), expected.size());
            for (size_t i = 0; i < actual.size(); ++i) {
                ASSERT_TRUE(std::isfinite(actual[i])) << i;
                EXPECT_NEAR(actual[i], expected[i], tolerance) << i;
            }
        }
    };

    TEST_P(TensorOperationContracts, PointwiseBroadcastUsesAnalyticValuesAtOddSizes) {
        constexpr size_t rows = 33, columns = 127;
        std::vector<float> a(rows * columns), b(columns), expected(a.size());
        for (size_t i = 0; i < a.size(); ++i)
            a[i] = static_cast<float>(int(i % 31) - 15) / 8.f;
        for (size_t i = 0; i < b.size(); ++i)
            b[i] = static_cast<float>(i % 7) / 4.f;
        for (size_t i = 0; i < a.size(); ++i)
            expected[i] = a[i] * b[i % columns] - 0.25f;
        // Dyadic inputs and exact products avoid a backend-specific math oracle.
        expect_values(values(a, {rows, columns}).mul(values(b, {1, columns})).add(0.25f).sub(0.5f), expected);
    }

    TEST_P(TensorOperationContracts, MovementPreservesLogicalStridedValues) {
        const auto source = values({1, 2, 3, 4, 5, 6}, {2, 3});
        const auto transposed = source.transpose(0, 1);
        const auto result = Tensor::cat({transposed.contiguous(), transposed.contiguous()}, 1);
        EXPECT_EQ(result.shape(), TensorShape({3, 4}));
        expect_values(result, {1, 4, 1, 4, 2, 5, 2, 5, 3, 6, 3, 6});
        auto detached = transposed.clone();
        detached.fill_(9.f);
        expect_values(source, {1, 2, 3, 4, 5, 6});
        expect_values(detached, std::vector<float>(6, 9.f));
    }

    TEST_P(TensorOperationContracts, ReductionsAndScansMatchAnalyticIntegers) {
        const auto input = values({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4});
        expect_values(input.sum(0), {15, 18, 21, 24});
        expect_values(input.sum(1), {10, 26, 42});
        expect_values(input.cumsum(1), {1, 3, 6, 10, 5, 11, 18, 26, 9, 19, 30, 42});
        EXPECT_FLOAT_EQ(input.sum_scalar(), 78.f);
    }

    TEST_P(TensorOperationContracts, IndexSelectReadsStridedIndices) {
        const auto input = values({1, 2, 3, 4, 5, 6}, {3, 2});
        const auto indices = Tensor::from_vector(std::vector<int>{2, 99, 0, 99, 2, 99}, {3, 2}, device())
                                 .slice(1, 0, 1)
                                 .squeeze();
        expect_values(input.index_select(0, indices), {5, 6, 1, 2, 5, 6});
    }

    TEST_P(TensorOperationContracts, SortReturnsAValidPermutationForDuplicateKeys) {
        const std::vector<float> data{3, -1, 3, 0, -1, 2};
        const auto [sorted, indices] = values(data, {data.size()}).sort();
        expect_values(sorted, {-1, -1, 0, 2, 3, 3});
        expect_placement(indices);
        const auto permutation = indices.to_vector_int64();
        ASSERT_EQ(permutation.size(), data.size());
        std::vector<bool> seen(data.size(), false);
        const auto ordered = sorted.to_vector();
        for (size_t i = 0; i < permutation.size(); ++i) {
            ASSERT_GE(permutation[i], 0);
            ASSERT_LT(static_cast<size_t>(permutation[i]), data.size());
            EXPECT_FALSE(seen[permutation[i]]);
            seen[permutation[i]] = true;
            EXPECT_EQ(ordered[i], data[permutation[i]]);
        }
    }

    TEST_P(TensorOperationContracts, MatrixMultiplyMatchesAnalyticValuesForStridedInputs) {
        const auto a = values({1, 2, 3, 4, 5, 6}, {3, 2}).transpose(0, 1);
        const auto b = values({7, 8, 9, 10, 11, 12}, {2, 3}).transpose(0, 1);
        // [1 3 5; 2 4 6] times [7 10; 8 11; 9 12].
        expect_values(a.matmul(b), {76, 103, 100, 136});
    }

    TEST_P(TensorOperationContracts, SoftmaxMatchesAnIndependentStableReference) {
        const std::vector<float> input{1000, 1001, 999, -1000, -999, -1001};
        std::vector<float> expected(input.size());
        for (size_t row = 0; row < 2; ++row) {
            const double maximum = *std::max_element(input.begin() + row * 3, input.begin() + (row + 1) * 3);
            double denominator = 0.;
            for (size_t j = 0; j < 3; ++j)
                denominator += std::exp(double(input[row * 3 + j]) - maximum);
            for (size_t j = 0; j < 3; ++j)
                expected[row * 3 + j] = static_cast<float>(std::exp(double(input[row * 3 + j]) - maximum) / denominator);
        }
        // FP32 reductions and exp may round differently; no tolerance depends on the backend.
        expect_values(nn::softmax(values(input, {2, 3})), expected, 2e-6f);
    }

    TEST_P(TensorOperationContracts, DegenerateSamplingHasBackendIndependentOutcome) {
        const auto weights = values({0, 1, 0}, {3});
        const auto result = Tensor::multinomial(weights, 17, true, 42);
        expect_placement(result);
        EXPECT_EQ(result.to_vector_int64(), std::vector<int64_t>(17, 1));
    }

    INSTANTIATE_TEST_SUITE_P(Backends, TensorOperationContracts,
                             testing::ValuesIn(lfs::test::tensor_backend_targets()),
                             lfs::test::tensor_backend_parameter_name);
} // namespace
