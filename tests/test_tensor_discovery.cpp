/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <sstream>
#include <string_view>
#include <vector>

using namespace lfs::core;

namespace {

    std::vector<float> torch_float_values(const torch::Tensor& tensor) {
        const auto cpu = tensor.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
        const auto* data = cpu.data_ptr<float>();
        return {data, data + cpu.numel()};
    }

    std::vector<int64_t> torch_int64_values(const torch::Tensor& tensor) {
        const auto cpu = tensor.detach().to(torch::kCPU).to(torch::kInt64).contiguous();
        const auto* data = cpu.data_ptr<int64_t>();
        return {data, data + cpu.numel()};
    }

    std::vector<uint8_t> torch_uint8_values(const torch::Tensor& tensor) {
        const auto cpu = tensor.detach().to(torch::kCPU).to(torch::kUInt8).contiguous();
        const auto* data = cpu.data_ptr<uint8_t>();
        return {data, data + cpu.numel()};
    }

    std::vector<float> lfs_float_values(const Tensor& tensor) {
        return tensor.cpu().to(DataType::Float32).contiguous().to_vector();
    }

    void expect_shape(const Tensor& actual, const torch::Tensor& expected,
                      const std::string_view context) {
        ASSERT_EQ(actual.ndim(), static_cast<size_t>(expected.dim())) << context;
        for (size_t dim = 0; dim < actual.ndim(); ++dim) {
            EXPECT_EQ(actual.size(dim), static_cast<size_t>(expected.size(dim)))
                << context << " at dimension " << dim;
        }
    }

    void expect_float_tensor(const Tensor& actual, const torch::Tensor& expected,
                             const std::string_view context,
                             const float rtol = 1e-5f,
                             const float atol = 1e-6f) {
        expect_shape(actual, expected, context);
        const auto actual_values = lfs_float_values(actual);
        const auto expected_values = torch_float_values(expected);
        ASSERT_EQ(actual_values.size(), expected_values.size()) << context;
        for (size_t i = 0; i < actual_values.size(); ++i) {
            const float a = actual_values[i];
            const float e = expected_values[i];
            if (std::isnan(e)) {
                EXPECT_TRUE(std::isnan(a))
                    << context << " at flat index " << i << ": actual=" << a;
            } else if (std::isinf(e)) {
                EXPECT_EQ(a, e)
                    << context << " at flat index " << i;
            } else {
                EXPECT_LE(std::abs(a - e), atol + rtol * std::abs(e))
                    << context << " at flat index " << i
                    << ": actual=" << a << ", expected=" << e;
            }
        }
    }

    void expect_int64_tensor(const Tensor& actual, const torch::Tensor& expected,
                             const std::string_view context) {
        expect_shape(actual, expected, context);
        const auto actual_values = actual.cpu().contiguous().to_vector_int64();
        const auto expected_values = torch_int64_values(expected);
        EXPECT_EQ(actual_values, expected_values) << context;
    }

} // namespace

TEST(DiscoverySweep, PartialStdAndVarWithoutKeepdimMatchTorch) {
    const std::vector<float> data = {0.0f, 1.0f, 2.0f,
                                     3.0f, 4.0f, 5.0f};
    const auto input = Tensor::from_vector(data, {2, 3}, Device::CPU);
    const auto reference = torch::tensor(data, torch::kFloat32).reshape({2, 3});
    const auto expected_std = reference.std(c10::IntArrayRef{1}, false, false);
    const auto expected_var = reference.var(c10::IntArrayRef{1}, false, false);

    try {
        const auto actual_std = input.std(1, false, false);
        expect_float_tensor(actual_std, expected_std,
                            "std(dim=1, keepdim=false, unbiased=false)");
    } catch (const std::exception& error) {
        ADD_FAILURE() << "std unexpectedly threw: " << error.what();
    }

    try {
        const auto actual_var = input.var(1, false, false);
        expect_float_tensor(actual_var, expected_var,
                            "var(dim=1, keepdim=false, unbiased=false)");
    } catch (const std::exception& error) {
        ADD_FAILURE() << "var unexpectedly threw: " << error.what();
    }
}

TEST(DiscoverySweep, UnbiasedSingletonStdAndVarProduceNaN) {
    const auto input = Tensor::from_vector({5.0f}, {1}, Device::CPU);
    const auto reference = torch::tensor({5.0f}, torch::kFloat32);

    const float expected_std = reference.std(true).item<float>();
    const float expected_var = reference.var(true).item<float>();
    ASSERT_TRUE(std::isnan(expected_std));
    ASSERT_TRUE(std::isnan(expected_var));

    EXPECT_TRUE(std::isnan(input.std_scalar(true)))
        << "Torch's correction=1 estimate is undefined for one sample";
    EXPECT_TRUE(std::isnan(input.var_scalar(true)))
        << "Torch's correction=1 estimate is undefined for one sample";
}

TEST(DiscoverySweep, EmptyReductionSemanticsMatchTorch) {
    const auto input = Tensor::empty({0}, Device::CPU, DataType::Float32);
    const auto reference = torch::empty({0}, torch::kFloat32);
    ASSERT_TRUE(std::isnan(reference.mean().item<float>()));

    EXPECT_TRUE(std::isnan(input.mean().item()))
        << "mean(empty) must be NaN, not an additive identity";
    EXPECT_THROW({
        const auto ignored = input.max();
        (void)ignored;
    },
                 std::runtime_error);
    EXPECT_THROW({
        const auto ignored = input.min();
        (void)ignored;
    },
                 std::runtime_error);
}

TEST(DiscoverySweep, BoolReductionDtypesAndDomainsMatchTorch) {
    const auto input = Tensor::from_vector(
        std::vector<bool>{true, false, true}, {3}, Device::CPU);
    const auto reference = torch::tensor(
                               std::vector<int>{1, 0, 1}, torch::kInt32)
                               .to(torch::kBool);

    EXPECT_THROW({
        const auto ignored = reference.mean();
        (void)ignored;
    },
                 c10::Error);
    EXPECT_THROW({
        const auto ignored = input.mean();
        (void)ignored;
    },
                 std::runtime_error);

    const auto expected_max = reference.max();
    const auto expected_min = reference.min();
    ASSERT_EQ(expected_max.scalar_type(), torch::kBool);
    ASSERT_EQ(expected_min.scalar_type(), torch::kBool);
    EXPECT_EQ(input.max().dtype(), DataType::Bool);
    EXPECT_EQ(input.min().dtype(), DataType::Bool);
}

TEST(DiscoverySweep, AccessorRejectsDtypeMismatchLikeTorch) {
    auto input = Tensor::from_vector(
        std::vector<int>{1, 2, 3}, {3}, Device::CPU);
    auto reference = torch::tensor(
        std::vector<int>{1, 2, 3}, torch::kInt32);

    EXPECT_THROW((reference.accessor<float, 1>()), c10::Error);
    EXPECT_THROW((input.accessor<float, 1>()), std::runtime_error)
        << "accessor<T> must not reinterpret storage of a different dtype";
}

TEST(DiscoverySweep, ScalarNormZeroAndNegativeInfinityMatchTorch) {
    const std::vector<float> data = {0.0f, 2.0f, -3.0f};
    const auto input = Tensor::from_vector(data, {3}, Device::CPU);
    const auto reference = torch::tensor(data, torch::kFloat32);

    const float expected_zero = reference.norm(0).item<float>();
    const float expected_negative_infinity =
        reference.norm(-std::numeric_limits<double>::infinity()).item<float>();

    EXPECT_FLOAT_EQ(input.norm(0.0f), expected_zero);
    EXPECT_FLOAT_EQ(input.norm(-std::numeric_limits<float>::infinity()),
                    expected_negative_infinity);
}

TEST(DiscoverySweep, SortOrdersNaNsLikeTorch) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> data = {nan, -2.0f};
    const auto input = Tensor::from_vector(data, {2}, Device::CPU);
    const auto reference = torch::tensor(data, torch::kFloat32);

    const auto [actual_values, actual_indices] = input.sort(0, false);
    const auto [expected_values, expected_indices] = reference.sort(0, false);
    expect_float_tensor(actual_values, expected_values, "ascending sort values with NaN");
    expect_int64_tensor(actual_indices, expected_indices, "ascending sort indices with NaN");
}

TEST(DiscoverySweep, RoundUsesTiesToEvenLikeTorch) {
    const std::vector<float> data = {0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f};
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto input = Tensor::from_vector(data, {data.size()}, device);
        auto reference = torch::tensor(data, torch::kFloat32);
        if (device == Device::CUDA)
            reference = reference.cuda();
        expect_float_tensor(input.round(), reference.round(),
                            "round must use IEEE ties-to-even");
    }
}

TEST(DiscoverySweep, NumericCastsToUInt8MatchTorch) {
    const std::vector<float> data = {-1.2f, 0.9f, 1.9f, 255.9f, 256.0f};
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto input = Tensor::from_vector(data, {data.size()}, device);
        auto reference = torch::tensor(data, torch::kFloat32);
        if (device == Device::CUDA)
            reference = reference.cuda();

        const auto actual = input.to(DataType::UInt8).cpu().to_vector_uint8();
        const auto expected = torch_uint8_values(reference);
        EXPECT_EQ(actual, expected)
            << "numeric to(UInt8) must use cast truncation/wrapping semantics";
    }
}

TEST(DiscoverySweep, SortAcceptsAnEmptyDimension) {
    const auto input = Tensor::empty({0}, Device::CPU, DataType::Float32);
    const auto reference = torch::empty({0}, torch::kFloat32);

    std::optional<std::pair<Tensor, Tensor>> actual;
    ASSERT_NO_THROW(actual.emplace(input.sort(0, false)));
    const auto [expected_values, expected_indices] = reference.sort(0, false);
    expect_float_tensor(actual->first, expected_values, "empty sort values");
    expect_int64_tensor(actual->second, expected_indices, "empty sort indices");
}

TEST(DiscoverySweep, MinMaxWithIndicesAcceptEmptyOuterDimensions) {
    const auto input = Tensor::empty({0, 3}, Device::CPU, DataType::Float32);
    const auto reference = torch::empty({0, 3}, torch::kFloat32);

    const auto [expected_min, expected_min_indices] = reference.min(1, false);
    try {
        const auto actual_min = input.min_with_indices(1, false);
        expect_float_tensor(actual_min.first, expected_min, "empty outer min values");
        expect_int64_tensor(actual_min.second, expected_min_indices, "empty outer min indices");
    } catch (const std::exception& error) {
        ADD_FAILURE() << "min_with_indices unexpectedly threw: " << error.what();
    }

    const auto [expected_max, expected_max_indices] = reference.max(1, false);
    try {
        const auto actual_max = input.max_with_indices(1, false);
        expect_float_tensor(actual_max.first, expected_max, "empty outer max values");
        expect_int64_tensor(actual_max.second, expected_max_indices, "empty outer max indices");
    } catch (const std::exception& error) {
        ADD_FAILURE() << "max_with_indices unexpectedly threw: " << error.what();
    }
}

TEST(DiscoverySweep, SortUsesLogicalViewValues) {
    const std::vector<float> data = {4.0f, 1.0f, 3.0f,
                                     2.0f, 6.0f, 5.0f};
    const auto input = Tensor::from_vector(data, {2, 3}, Device::CPU).transpose(0, 1);
    const auto reference = torch::tensor(data, torch::kFloat32)
                               .reshape({2, 3})
                               .transpose(0, 1);

    const auto [actual_values, actual_indices] = input.sort(1, false);
    const auto [expected_values, expected_indices] = reference.sort(1, false);
    expect_float_tensor(actual_values, expected_values, "sort values from a transposed view");
    expect_int64_tensor(actual_indices, expected_indices,
                        "sort indices from a transposed view");
}

TEST(DiscoverySweep, NonzeroSplitReturnsOneCoordinateTensorPerAxis) {
    const std::vector<float> data = {1.0f, 0.0f,
                                     0.0f, 2.0f};
    const auto input = Tensor::from_vector(data, {2, 2}, Device::CPU);
    const auto reference = torch::tensor(data, torch::kFloat32).reshape({2, 2});

    const auto actual = input.nonzero_split();
    const auto expected = reference.nonzero().unbind(1);
    ASSERT_EQ(actual.size(), expected.size())
        << "nonzero_split must mirror nonzero(as_tuple=true)";
    for (size_t axis = 0; axis < actual.size(); ++axis) {
        expect_int64_tensor(actual[axis], expected[axis], "nonzero_split coordinate axis");
    }
}

TEST(DiscoverySweep, NoOpViewTransformsPreserveAliasing) {
    {
        const std::vector<float> data = {1.0f, 2.0f,
                                         3.0f, 4.0f};
        auto actual_base = Tensor::from_vector(data, {2, 2}, Device::CPU);
        auto expected_base = torch::tensor(data, torch::kFloat32).reshape({2, 2});

        auto actual_view = actual_base.squeeze(0); // dimension 0 is not singleton
        auto expected_view = expected_base.squeeze(0);
        actual_view.add_(10.0f);
        expected_view.add_(10.0f);
        expect_float_tensor(actual_base, expected_base,
                            "squeeze(non-singleton) must remain an alias");
    }

    {
        const std::vector<float> data = {1.0f, 2.0f, 3.0f};
        auto actual_base = Tensor::from_vector(data, {3}, Device::CPU);
        auto expected_base = torch::tensor(data, torch::kFloat32);

        auto actual_view = actual_base.t();
        auto expected_view = expected_base.t();
        actual_view.mul_(2.0f);
        expected_view.mul_(2.0f);
        expect_float_tensor(actual_base, expected_base,
                            "t() on a vector must remain an alias");
    }
}

TEST(DiscoverySweep, RowProxyTensorConversionPreservesAliasing) {
    const std::vector<float> data = {1.0f, 2.0f,
                                     3.0f, 4.0f};
    auto actual_base = Tensor::from_vector(data, {2, 2}, Device::CPU);
    auto expected_base = torch::tensor(data, torch::kFloat32).reshape({2, 2});

    Tensor actual_row = actual_base[0];
    auto expected_row = expected_base[0];
    actual_row.add_(10.0f);
    expected_row.add_(10.0f);

    expect_float_tensor(actual_base, expected_base,
                        "Tensor converted from tensor[row] must alias the parent row");
}

TEST(DiscoverySweep, CdistMaterializesTheLeftView) {
    const std::vector<float> lhs_data = {1.0f, 2.0f, 3.0f,
                                         4.0f, 5.0f, 6.0f};
    const std::vector<float> rhs_data = {0.0f, 0.0f};
    const auto lhs = Tensor::from_vector(lhs_data, {2, 3}, Device::CPU).transpose(0, 1);
    const auto rhs = Tensor::from_vector(rhs_data, {1, 2}, Device::CPU);
    const auto reference_lhs = torch::tensor(lhs_data, torch::kFloat32)
                                   .reshape({2, 3})
                                   .transpose(0, 1);
    const auto reference_rhs = torch::tensor(rhs_data, torch::kFloat32).reshape({1, 2});
    const auto expected = (reference_lhs.unsqueeze(1) - reference_rhs.unsqueeze(0))
                              .square()
                              .sum(2)
                              .sqrt();

    expect_float_tensor(lhs.cdist(rhs, 2.0f), expected,
                        "cdist with a non-contiguous left input");
}

TEST(DiscoverySweep, CdistSupportsZeroAndInfinityNorms) {
    const std::vector<float> lhs_data = {0.0f, 2.0f, 3.0f};
    const std::vector<float> rhs_data = {0.0f, 5.0f, 1.0f};
    const auto lhs = Tensor::from_vector(lhs_data, {1, 3}, Device::CPU);
    const auto rhs = Tensor::from_vector(rhs_data, {1, 3}, Device::CPU);
    const auto reference_lhs = torch::tensor(lhs_data, torch::kFloat32).reshape({1, 3});
    const auto reference_rhs = torch::tensor(rhs_data, torch::kFloat32).reshape({1, 3});

    for (const float p : {0.0f, std::numeric_limits<float>::infinity()}) {
        SCOPED_TRACE(::testing::Message() << "p=" << p);
        const auto expected = torch::cdist(reference_lhs, reference_rhs, p);
        std::optional<Tensor> actual;
        ASSERT_NO_THROW(actual.emplace(lhs.cdist(rhs, p)));
        expect_float_tensor(*actual, expected, "cdist norm domain");
    }
}

TEST(DiscoverySweep, MaxPool2dPreservesNonFiniteMaxima) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> data = {1.0f, nan, 2.0f, 3.0f};
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto input = Tensor::from_vector(data, {1, 1, 2, 2}, device);
        auto reference = torch::tensor(data, torch::kFloat32).reshape({1, 1, 2, 2});
        if (device == Device::CUDA) {
            reference = reference.cuda();
        }
        expect_float_tensor(input.max_pool2d(2, 2),
                            torch::max_pool2d(reference, {2, 2}, {2, 2}),
                            "max_pool2d NaN propagation");

        const std::vector<float> negative_infinity = {
            -std::numeric_limits<float>::infinity()};
        const auto inf_input = Tensor::from_vector(negative_infinity, {1, 1, 1, 1}, device);
        auto inf_reference = torch::tensor(negative_infinity, torch::kFloat32)
                                 .reshape({1, 1, 1, 1});
        if (device == Device::CUDA)
            inf_reference = inf_reference.cuda();
        expect_float_tensor(inf_input.max_pool2d(1, 1),
                            torch::max_pool2d(inf_reference, {1, 1}, {1, 1}),
                            "max_pool2d negative-infinity propagation");
    }
}

TEST(DiscoverySweep, MaxPool2dRejectsZeroSizedOutput) {
    const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto input = Tensor::from_vector(data, {1, 1, 2, 2}, device);
        auto reference = torch::tensor(data, torch::kFloat32).reshape({1, 1, 2, 2});
        if (device == Device::CUDA) {
            reference = reference.cuda();
        }

        EXPECT_THROW(torch::max_pool2d(reference, {3, 3}, {2, 2}), std::exception);
        EXPECT_THROW(input.max_pool2d(3, 2), std::exception);
    }
}

TEST(DiscoverySweep, ClampMaterializesItsInputView) {
    const std::vector<float> data = {1.0f, 2.0f, 3.0f,
                                     4.0f, 5.0f, 6.0f};
    const auto input = Tensor::from_vector(data, {2, 3}, Device::CPU).transpose(0, 1);
    const auto reference = torch::tensor(data, torch::kFloat32)
                               .reshape({2, 3})
                               .transpose(0, 1);

    expect_float_tensor(input.clamp(2.5f, 4.5f), reference.clamp(2.5f, 4.5f),
                        "clamp on a transposed view");
}

TEST(DiscoverySweep, LazyPointwiseOnOffsetSliceMatchesTorch) {
    const std::vector<float> data = {0.0f, 0.0f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 0.0f, 0.0f,
                                     0.0f, 3.0f, 0.0f, 0.0f};
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto input = Tensor::from_vector(data, {1, 3, 4}, device).slice(2, 1, 2);
        auto reference = torch::tensor(data, torch::kFloat32)
                             .reshape({1, 3, 4})
                             .slice(2, 1, 2);
        if (device == Device::CUDA)
            reference = reference.cuda();

        const auto actual_pointwise = input.mul(3.0f);
        const auto expected_pointwise = reference.mul(3.0f);
        const auto actual_fused_reduce = input.mul(3.0f).mean();
        const auto expected_fused_reduce = reference.mul(3.0f).mean();
        expect_float_tensor(actual_fused_reduce, expected_fused_reduce,
                            "reduction directly consuming deferred scalar multiplication");
        expect_float_tensor(actual_pointwise, expected_pointwise,
                            "deferred scalar multiplication on an offset slice");
        expect_float_tensor(actual_pointwise.mean(), expected_pointwise.mean(),
                            "reduction after deferred scalar multiplication on an offset slice");
    }
}

TEST(DiscoverySweep, InPlaceMatchesOutOfPlaceOnContiguousInputs) {
    const std::vector<float> data = {-2.0f, 0.0f, 3.0f, 5.0f};
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        auto actual = Tensor::from_vector(data, {2, 2}, device);
        actual.add_(2.0f);
        auto reference = torch::tensor(data, torch::kFloat32).reshape({2, 2});
        if (device == Device::CUDA)
            reference = reference.cuda();
        expect_float_tensor(actual, reference.add(2.0f),
                            "in-place add equals out-of-place add");
    }
}

TEST(DiscoverySweep, NonzeroUsesLogicalViewOrder) {
    const std::vector<float> data = {0.0f, 1.0f, 0.0f,
                                     0.0f, 0.0f, 2.0f};
    const auto input = Tensor::from_vector(data, {2, 3}, Device::CPU).transpose(0, 1);
    const auto reference = torch::tensor(data, torch::kFloat32)
                               .reshape({2, 3})
                               .transpose(0, 1);

    expect_int64_tensor(input.nonzero(), reference.nonzero(),
                        "nonzero on a transposed view");
}

TEST(DiscoverySweep, MinMaxWithIndicesPropagateNaNs) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> data = {1.0f, nan, -1.0f};
    const auto input = Tensor::from_vector(data, {1, 3}, Device::CPU);
    const auto reference = torch::tensor(data, torch::kFloat32).reshape({1, 3});

    const auto [actual_min, actual_min_index] = input.min_with_indices(1, false);
    const auto [expected_min, expected_min_index] = reference.min(1, false);
    expect_float_tensor(actual_min, expected_min, "min_with_indices NaN value");
    expect_int64_tensor(actual_min_index, expected_min_index, "min_with_indices NaN index");

    const auto [actual_max, actual_max_index] = input.max_with_indices(1, false);
    const auto [expected_max, expected_max_index] = reference.max(1, false);
    expect_float_tensor(actual_max, expected_max, "max_with_indices NaN value");
    expect_int64_tensor(actual_max_index, expected_max_index, "max_with_indices NaN index");
}

TEST(DiscoverySweep, ReassigningLazyPointwiseOnExpandedSliceMatchesTorch) {
    const std::vector<float> data = {0.0f, 5.0f};
    const auto input = Tensor::from_vector(data, {2}, Device::CPU)
                           .unsqueeze(1)
                           .expand({2, 3})
                           .slice(1, 0, 1);
    const auto reference = torch::tensor(data, torch::kFloat32)
                               .unsqueeze(1)
                               .expand({2, 3})
                               .slice(1, 0, 1);

    auto actual = input;
    actual = actual.mul(2.0f);
    actual = actual.mul(-2.0f);
    auto expected = reference;
    expected = expected.mul(2.0f);
    expected = expected.mul(-2.0f);

    expect_float_tensor(actual.sum(), expected.sum(),
                        "sum consuming a pointwise chain on an expanded slice");
    expect_float_tensor(actual, expected, "reassigned pointwise result on an expanded slice");
}

TEST(DiscoverySweep, ArgMinAndArgMaxPublicApisMatchTorch) {
    const std::vector<float> data = {3.0f, -2.0f, 7.0f, 1.0f};
    const auto input = Tensor::from_vector(data, {4}, Device::CPU);
    const auto reference = torch::tensor(data, torch::kFloat32);

    try {
        expect_int64_tensor(input.argmax(), reference.argmax(), "argmax public API");
    } catch (const std::exception& error) {
        ADD_FAILURE() << "argmax unexpectedly threw: " << error.what();
    }
    try {
        expect_int64_tensor(input.argmin(), reference.argmin(), "argmin public API");
    } catch (const std::exception& error) {
        ADD_FAILURE() << "argmin unexpectedly threw: " << error.what();
    }
}

TEST(DiscoverySweep, IntegerClampBoundsMatchTorch) {
    const std::vector<int> data = {-3, -1, 0, 1, 3};
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        auto input = Tensor::from_vector(data, {data.size()}, Device::CPU);
        auto reference = torch::tensor(data, torch::kInt32);
        if (device == Device::CUDA) {
            input = input.cuda();
            reference = reference.cuda();
        }

        const auto actual_unbounded = input.clamp(
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity());
        const auto expected_unbounded = reference.clamp(
            -std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity());
        expect_float_tensor(actual_unbounded, expected_unbounded,
                            "Int32 clamp with infinite bounds");

        const auto actual_fractional = input.clamp(-0.5f, 1.5f);
        const auto expected_fractional = reference.clamp(-0.5, 1.5);
        expect_float_tensor(actual_fractional, expected_fractional,
                            "Int32 clamp with fractional bounds");
        EXPECT_EQ(actual_fractional.dtype(),
                  expected_fractional.scalar_type() == torch::kFloat32
                      ? DataType::Float32
                      : DataType::Int32);
    }
}

TEST(DiscoverySweep, FullShapeBooleanIndexingMatchesTorch) {
    const std::vector<float> data = {1.0f, 2.0f, 3.0f,
                                     4.0f, 5.0f, 6.0f};
    const std::vector<bool> mask_data = {true, false, true,
                                         false, true, false};
    const auto reference = torch::tensor(data, torch::kFloat32).reshape({2, 3});
    const auto reference_mask = torch::tensor(
                                    std::vector<int>{1, 0, 1, 0, 1, 0}, torch::kInt32)
                                    .reshape({2, 3})
                                    .to(torch::kBool);
    const auto expected = reference.index({reference_mask});

    auto mutable_input = Tensor::from_vector(data, {2, 3}, Device::CPU);
    const auto mutable_mask = Tensor::from_vector(mask_data, {2, 3}, Device::CPU);
    try {
        const Tensor actual = mutable_input[mutable_mask];
        expect_float_tensor(actual, expected, "mutable full-shape Bool operator[]");
    } catch (const std::exception& error) {
        ADD_FAILURE() << "mutable full-shape Bool indexing unexpectedly threw: "
                      << error.what();
    }

    const auto const_input = Tensor::from_vector(data, {2, 3}, Device::CPU);
    const auto const_mask = Tensor::from_vector(mask_data, {2, 3}, Device::CPU);
    try {
        const Tensor actual = const_input[const_mask];
        expect_float_tensor(actual, expected, "const full-shape Bool operator[]");
    } catch (const std::exception& error) {
        ADD_FAILURE() << "const full-shape Bool indexing unexpectedly threw: "
                      << error.what();
    }
}

TEST(DiscoverySweep, MaskedOpsBroadcastMasksLikeTorch) {
    const std::vector<float> data = {1.0f, 2.0f, 3.0f,
                                     4.0f, 5.0f, 6.0f};
    const std::vector<bool> mask_data = {true, false, true};
    const auto reference = torch::tensor(data, torch::kFloat32).reshape({2, 3});
    const auto reference_mask = torch::tensor(
                                    std::vector<int>{1, 0, 1}, torch::kInt32)
                                    .reshape({1, 3})
                                    .to(torch::kBool);

    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        auto input = Tensor::from_vector(data, {2, 3}, Device::CPU);
        auto mask = Tensor::from_vector(mask_data, {1, 3}, Device::CPU);
        auto torch_input = reference;
        auto torch_mask = reference_mask;
        if (device == Device::CUDA) {
            input = input.cuda();
            mask = mask.cuda();
            torch_input = torch_input.cuda();
            torch_mask = torch_mask.cuda();
        }

        try {
            const auto actual = input.masked_select(mask);
            expect_float_tensor(actual, torch_input.masked_select(torch_mask),
                                "masked_select with broadcast mask");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "masked_select unexpectedly rejected a broadcast mask: "
                          << error.what();
        }

        try {
            const auto actual = input.masked_fill(mask, -9.0f);
            expect_float_tensor(actual, torch_input.masked_fill(torch_mask, -9.0),
                                "masked_fill with broadcast mask");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "masked_fill unexpectedly rejected a broadcast mask: "
                          << error.what();
        }
    }
}

TEST(DiscoverySweep, MaskedFillAcceptsNonFiniteFloatValues) {
    const std::vector<float> data = {1.0f, 2.0f, 3.0f};
    const std::vector<bool> mask_data = {false, true, false};
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        auto input = Tensor::from_vector(data, {3}, Device::CPU);
        auto mask = Tensor::from_vector(mask_data, {3}, Device::CPU);
        auto reference = torch::tensor(data, torch::kFloat32);
        auto reference_mask = torch::tensor(
                                  std::vector<int>{0, 1, 0}, torch::kInt32)
                                  .to(torch::kBool);
        if (device == Device::CUDA) {
            input = input.cuda();
            mask = mask.cuda();
            reference = reference.cuda();
            reference_mask = reference_mask.cuda();
        }

        for (const float value : {
                 -std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::quiet_NaN()}) {
            SCOPED_TRACE(std::isnan(value) ? "NaN" : "-Inf");
            try {
                const auto actual = input.masked_fill(mask, value);
                expect_float_tensor(actual, reference.masked_fill(reference_mask, value),
                                    "masked_fill non-finite Float32 scalar");
            } catch (const std::exception& error) {
                ADD_FAILURE() << "masked_fill unexpectedly rejected a non-finite "
                                 "Float32 scalar: "
                              << error.what();
            }
        }
    }
}

TEST(DiscoverySweep, BoolMaskedFillUsesScalarTruthiness) {
    const std::vector<bool> mask_data = {true, false, true};
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto mask = Tensor::from_vector(mask_data, {3}, device);
        const auto input = Tensor::zeros_bool({3}, device);
        auto reference_mask = torch::tensor(
                                  std::vector<int>{1, 0, 1}, torch::kInt32)
                                  .to(torch::kBool);
        auto reference = torch::zeros({3}, torch::kBool);
        if (device == Device::CUDA) {
            reference_mask = reference_mask.cuda();
            reference = reference.cuda();
        }
        const auto expected = reference.masked_fill(reference_mask, -1.0);

        try {
            const auto actual = input.masked_fill(mask, -1.0f);
            expect_shape(actual, expected, "Bool masked_fill scalar truthiness");
            EXPECT_EQ(actual.cpu().to(DataType::UInt8).to_vector_uint8(),
                      torch_uint8_values(expected));
        } catch (const std::exception& error) {
            ADD_FAILURE() << "Bool masked_fill unexpectedly rejected a truthy scalar: "
                          << error.what();
        }
    }
}

TEST(DiscoverySweep, NormalAllowsZeroStandardDeviation) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        if (device == Device::CUDA)
            options = options.device(torch::kCUDA);
        auto expected = torch::empty({4}, options);
        ASSERT_NO_THROW(expected.normal_(2.5, 0.0));

        try {
            const auto actual = Tensor::normal({4}, 2.5f, 0.0f, device);
            expect_float_tensor(actual, expected, "normal with std=0");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "normal unexpectedly rejected std=0: " << error.what();
        }

        auto actual_in_place = Tensor::empty({4}, device, DataType::Float32);
        try {
            actual_in_place.normal_(2.5f, 0.0f);
            expect_float_tensor(actual_in_place, expected, "normal_ with std=0");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "normal_ unexpectedly rejected std=0: " << error.what();
        }
    }
}

TEST(DiscoverySweep, MixedNormalApisAdvanceCudaGenerator) {
    constexpr uint64_t seed = 0x12345;
    constexpr size_t count = 8;

    torch::manual_seed(seed);
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto torch_first = torch::randn({static_cast<int64_t>(count)}, options);
    auto torch_second = torch::empty({static_cast<int64_t>(count)}, options);
    torch_second.normal_();
    EXPECT_FALSE(torch::equal(torch_first, torch_second))
        << "Torch's sequential normal APIs must advance their shared generator";

    Tensor::manual_seed(seed);
    const auto first = Tensor::normal({count}, 0.0f, 1.0f, Device::CUDA);
    auto second = Tensor::empty({count}, Device::CUDA, DataType::Float32);
    second.normal_(0.0f, 1.0f);
    EXPECT_NE(lfs_float_values(first), lfs_float_values(second))
        << "static normal followed by normal_ reused the same CUDA subsequence";
}

TEST(DiscoverySweep, BatchedMatmulBroadcastsSingletonBatch) {
    const std::vector<float> left_data = {1.0f, 2.0f, 3.0f,
                                          4.0f, 5.0f, 6.0f};
    std::vector<float> right_data(4 * 3 * 2);
    for (size_t i = 0; i < right_data.size(); ++i) {
        right_data[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
    }

    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        auto left = Tensor::from_vector(left_data, {1, 2, 3}, Device::CPU);
        auto right = Tensor::from_vector(right_data, {4, 3, 2}, Device::CPU);
        auto torch_left = torch::tensor(left_data, torch::kFloat32).reshape({1, 2, 3});
        auto torch_right = torch::tensor(right_data, torch::kFloat32).reshape({4, 3, 2});
        if (device == Device::CUDA) {
            left = left.cuda();
            right = right.cuda();
            torch_left = torch_left.cuda();
            torch_right = torch_right.cuda();
        }

        try {
            const auto actual = left.matmul(right);
            expect_float_tensor(actual, torch::matmul(torch_left, torch_right),
                                "rank-3 matmul singleton batch broadcast");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "matmul unexpectedly rejected broadcastable batch dimensions: "
                          << error.what();
        }
    }
}

TEST(DiscoverySweep, FullMinMaxReductionsPropagateNaNs) {
    const std::vector<float> data = {1.0f, std::numeric_limits<float>::quiet_NaN(), -2.0f};

    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        auto input = Tensor::from_vector(data, {3}, Device::CPU);
        auto reference = torch::tensor(data, torch::kFloat32);
        if (device == Device::CUDA) {
            input = input.cuda();
            reference = reference.cuda();
        }

        expect_float_tensor(input.max(), reference.max(), "full max with a later NaN");
        expect_float_tensor(input.min(), reference.min(), "full min with a later NaN");
    }
}

TEST(DiscoverySweep, UniformFactoryAllowsDegenerateInterval) {
    constexpr float value = 2.5f;

    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        auto expected = torch::empty({4}, torch::TensorOptions().dtype(torch::kFloat32));
        if (device == Device::CUDA) {
            expected = expected.cuda();
        }
        expected.uniform_(value, value);

        auto inplace = Tensor::empty({4}, device, DataType::Float32);
        inplace.uniform_(value, value);
        expect_float_tensor(inplace, expected, "uniform_ with equal bounds");

        try {
            const auto factory = Tensor::uniform({4}, value, value, device, DataType::Float32);
            expect_float_tensor(factory, expected, "uniform factory with equal bounds");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "uniform factory unexpectedly rejected equal bounds: " << error.what();
        }
    }
}

TEST(DiscoverySweep, ArangeDoesNotIncludeFloatingEndpoint) {
    const auto actual = Tensor::arange(0.0f, 0.3f, 0.1f);
    const auto expected = torch::arange(
        0.0f, 0.3f, 0.1f,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    expect_float_tensor(actual, expected, "floating-point arange endpoint exclusion");
}

TEST(DiscoverySweep, SplitBatchPreservesEmptyInput) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto input = Tensor::empty({0, 3}, device, DataType::Float32);
        const auto actual = Tensor::split_batch(input, 2);
        const auto reference = torch::empty(
            {0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device == Device::CUDA ? torch::kCUDA : torch::kCPU));
        const auto expected = reference.split(2, 0);

        EXPECT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < std::min(actual.size(), expected.size()); ++i) {
            expect_shape(actual[i], expected[i], "empty split_batch chunk");
        }
    }
}

TEST(DiscoverySweep, AdaptiveAvgPoolRejectsEmptySpatialInput) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        auto input = Tensor::empty({1, 1, 0, 3}, device, DataType::Float32);
        auto reference = torch::empty({1, 1, 0, 3}, torch::TensorOptions().dtype(torch::kFloat32));
        if (device == Device::CUDA) {
            reference = reference.cuda();
        }

        EXPECT_THROW(torch::adaptive_avg_pool2d(reference, {2, 2}), c10::Error);
        EXPECT_THROW(static_cast<void>(input.adaptive_avg_pool2d(2, 2)), std::exception);
    }
}

TEST(DiscoverySweep, FullUInt8ScalarDomainMatchesTorch) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        for (const float value : {-1.0f, -0.5f, 255.5f, 256.0f}) {
            SCOPED_TRACE(value);
            bool torch_threw = false;
            std::vector<uint8_t> expected;
            try {
                auto reference = torch::full(
                    {1}, value,
                    torch::TensorOptions().dtype(torch::kUInt8).device(device == Device::CUDA ? torch::kCUDA : torch::kCPU));
                expected = torch_uint8_values(reference);
            } catch (const std::exception&) {
                torch_threw = true;
            }

            bool lfs_threw = false;
            std::vector<uint8_t> actual;
            try {
                actual = Tensor::full({1}, value, device, DataType::UInt8)
                             .cpu()
                             .to_vector_uint8();
            } catch (const std::exception&) {
                lfs_threw = true;
            }

            EXPECT_EQ(lfs_threw, torch_threw);
            if (!torch_threw && !lfs_threw) {
                EXPECT_EQ(actual, expected);
            }
        }
    }
}

TEST(DiscoverySweep, MaximumMinimumPreserveSignedZero) {
    const std::vector<float> left_data = {-0.0f, 0.0f};
    const std::vector<float> right_data = {0.0f, -0.0f};

    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        auto left = Tensor::from_vector(left_data, {2}, device);
        auto right = Tensor::from_vector(right_data, {2}, device);
        auto torch_left = torch::tensor(left_data, torch::kFloat32);
        auto torch_right = torch::tensor(right_data, torch::kFloat32);
        if (device == Device::CUDA) {
            torch_left = torch_left.cuda();
            torch_right = torch_right.cuda();
        }

        const auto actual_max = lfs_float_values(left.maximum(right));
        const auto actual_min = lfs_float_values(left.minimum(right));
        const auto expected_max = torch_float_values(torch::maximum(torch_left, torch_right));
        const auto expected_min = torch_float_values(torch::minimum(torch_left, torch_right));
        ASSERT_EQ(actual_max.size(), expected_max.size());
        ASSERT_EQ(actual_min.size(), expected_min.size());
        for (size_t i = 0; i < actual_max.size(); ++i) {
            EXPECT_EQ(std::signbit(actual_max[i]), std::signbit(expected_max[i]))
                << "maximum signed zero at index " << i;
            EXPECT_EQ(std::signbit(actual_min[i]), std::signbit(expected_min[i]))
                << "minimum signed zero at index " << i;
        }

        const auto reduced_max = lfs_float_values(left.max());
        const auto reduced_min = lfs_float_values(right.min());
        const auto expected_reduced_max = torch_float_values(torch_left.max());
        const auto expected_reduced_min = torch_float_values(torch_right.min());
        EXPECT_EQ(std::signbit(reduced_max.front()),
                  std::signbit(expected_reduced_max.front()))
            << "full maximum signed zero";
        EXPECT_EQ(std::signbit(reduced_min.front()),
                  std::signbit(expected_reduced_min.front()))
            << "full minimum signed zero";
    }
}

TEST(DiscoverySweep, ElementwiseMaximumMinimumPropagateNaNs) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> left_data = {1.0f, nan};
    const std::vector<float> right_data = {nan, 1.0f};

    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto left = Tensor::from_vector(left_data, {2}, device);
        const auto right = Tensor::from_vector(right_data, {2}, device);
        auto torch_left = torch::tensor(left_data, torch::kFloat32);
        auto torch_right = torch::tensor(right_data, torch::kFloat32);
        if (device == Device::CUDA) {
            torch_left = torch_left.cuda();
            torch_right = torch_right.cuda();
        }

        expect_float_tensor(left.maximum(right),
                            torch::maximum(torch_left, torch_right),
                            "elementwise maximum NaN propagation");
        expect_float_tensor(left.minimum(right),
                            torch::minimum(torch_left, torch_right),
                            "elementwise minimum NaN propagation");
    }
}

TEST(DiscoverySweep, RowProxyAssignmentAcceptsNonFiniteFloatValues) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        for (const float value : {std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::quiet_NaN()}) {
            auto actual = Tensor::zeros({1}, device, DataType::Float32);
            auto expected = torch::full(
                {1}, value,
                torch::TensorOptions().dtype(torch::kFloat32).device(device == Device::CUDA ? torch::kCUDA : torch::kCPU));
            try {
                actual[0] = value;
                expect_float_tensor(actual, expected, "row-proxy non-finite scalar assignment");
            } catch (const std::exception& error) {
                ADD_FAILURE() << "row proxy unexpectedly rejected a representable Float32 value: "
                              << error.what();
            }
        }
    }
}

TEST(DiscoverySweep, FillAcceptsNonFiniteFloatValues) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        for (const float value : {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::quiet_NaN()}) {
            auto actual = Tensor::zeros({3}, device, DataType::Float32);
            auto expected = torch::zeros(
                {3}, torch::TensorOptions().dtype(torch::kFloat32).device(device == Device::CUDA ? torch::kCUDA : torch::kCPU));
            expected.fill_(value);
            try {
                actual.fill_(value);
                expect_float_tensor(actual, expected, "fill_ with non-finite Float32 value");
            } catch (const std::exception& error) {
                ADD_FAILURE() << "fill_ unexpectedly rejected a representable Float32 value: "
                              << error.what();
            }
        }
    }
}

TEST(DiscoverySweep, IntegerFillRejectsOutOfRangeValuesLikeTorch) {
    constexpr float out_of_range = std::numeric_limits<float>::max();
    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        auto actual = Tensor::zeros({2}, device, DataType::Int32);
        auto expected = torch::zeros(
            {2}, torch::TensorOptions().dtype(torch::kInt32).device(device == Device::CUDA ? torch::kCUDA : torch::kCPU));

        EXPECT_THROW(expected.fill_(out_of_range), std::exception);
        EXPECT_THROW(actual.fill_(out_of_range), std::exception);
    }
}

TEST(DiscoverySweep, ScalarMathAcceptsNonFiniteOperands) {
    const std::vector<float> data = {0.0f, 1.0f};
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    for (const Device device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        auto input = Tensor::from_vector(data, {2}, device);
        auto reference = torch::tensor(data, torch::kFloat32);
        if (device == Device::CUDA) {
            reference = reference.cuda();
        }

        try {
            expect_float_tensor(input.add(infinity), reference.add(infinity),
                                "scalar addition with infinity");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "scalar arithmetic unexpectedly rejected infinity: " << error.what();
        }

        try {
            const auto actual = input.eq(nan);
            const auto expected = reference.eq(nan);
            expect_shape(actual, expected, "scalar equality with NaN");
            EXPECT_EQ(actual.cpu().to(DataType::UInt8).to_vector_uint8(),
                      torch_uint8_values(expected))
                << "scalar equality with NaN";
        } catch (const std::exception& error) {
            ADD_FAILURE() << "scalar comparison unexpectedly rejected NaN: " << error.what();
        }
    }
}

TEST(DiscoverySweep, ScalarWhereSupportsZeroDimensionalInputs) {
    for (const auto device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");

        const auto condition = Tensor::full_bool({}, true, device);
        const auto x = Tensor::full({}, 2.0f, device);
        const auto y = Tensor::full({}, -3.0f, device);

        const auto torch_device = device == Device::CPU ? torch::kCPU : torch::kCUDA;
        const auto expected = torch::where(
            torch::scalar_tensor(true, torch::TensorOptions().dtype(torch::kBool).device(torch_device)),
            torch::scalar_tensor(2.0f, torch::TensorOptions().dtype(torch::kFloat32).device(torch_device)),
            torch::scalar_tensor(-3.0f, torch::TensorOptions().dtype(torch::kFloat32).device(torch_device)));

        try {
            const auto actual = Tensor::where(condition, x, y);
            expect_float_tensor(actual, expected, "scalar where");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "scalar where unexpectedly threw: " << error.what();
        }
    }
}

TEST(DiscoverySweep, CatAndStackPromoteMixedDtypesLikeTorch) {
    for (const auto device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");

        const auto ints = Tensor::from_vector(
            std::vector<int>{1, 2}, {2}, device);
        const auto floats = Tensor::from_vector(
            std::vector<float>{0.5f, 1.5f}, {2}, device);

        const auto torch_device = device == Device::CPU ? torch::kCPU : torch::kCUDA;
        const auto torch_ints = torch::tensor(
            {1, 2}, torch::TensorOptions().dtype(torch::kInt32).device(torch_device));
        const auto torch_floats = torch::tensor(
            {0.5f, 1.5f}, torch::TensorOptions().dtype(torch::kFloat32).device(torch_device));
        const auto expected_cat = torch::cat({torch_ints, torch_floats});
        const auto expected_stack = torch::stack({torch_ints, torch_floats});
        ASSERT_EQ(expected_cat.scalar_type(), torch::kFloat32);
        ASSERT_EQ(expected_stack.scalar_type(), torch::kFloat32);

        try {
            const auto actual = Tensor::cat({ints, floats});
            EXPECT_EQ(actual.dtype(), DataType::Float32);
            expect_float_tensor(actual, expected_cat, "mixed-dtype cat");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "mixed-dtype cat unexpectedly threw: " << error.what();
        }

        try {
            const auto actual = Tensor::stack({ints, floats});
            EXPECT_EQ(actual.dtype(), DataType::Float32);
            expect_float_tensor(actual, expected_stack, "mixed-dtype stack");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "mixed-dtype stack unexpectedly threw: " << error.what();
        }
    }
}

TEST(DiscoverySweep, ScalarReductionsAcceptDimZeroLikeTorch) {
    for (const auto device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");

        const auto input = Tensor::full({}, 6.0f, device);
        const auto torch_device = device == Device::CPU ? torch::kCPU : torch::kCUDA;
        const auto reference = torch::scalar_tensor(
            6.0f, torch::TensorOptions().dtype(torch::kFloat32).device(torch_device));

        const auto expected_sum = reference.sum(/*dim=*/0, /*keepdim=*/false);
        const auto expected_mean = reference.mean(/*dim=*/0, /*keepdim=*/false);
        const auto expected_prod = reference.prod(/*dim=*/0, /*keepdim=*/false);

        try {
            expect_float_tensor(input.sum(0), expected_sum, "scalar sum(dim=0)");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "scalar sum(dim=0) unexpectedly threw: " << error.what();
        }
        try {
            expect_float_tensor(input.mean(0), expected_mean, "scalar mean(dim=0)");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "scalar mean(dim=0) unexpectedly threw: " << error.what();
        }
        try {
            expect_float_tensor(input.prod(0), expected_prod, "scalar prod(dim=0)");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "scalar prod(dim=0) unexpectedly threw: " << error.what();
        }
    }
}

TEST(DiscoverySweep, IntegerPowScalarOverloadRejectsNegativeIntegerExponent) {
    for (const auto device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");

        const auto input = Tensor::from_vector(
            std::vector<int>{2, -2}, {2}, device);
        const auto torch_device = device == Device::CPU ? torch::kCPU : torch::kCUDA;
        const auto torch_input = torch::tensor(
            {2, -2}, torch::TensorOptions().dtype(torch::kInt32).device(torch_device));

        EXPECT_THROW((void)torch::pow(torch_input, -1), c10::Error);

        EXPECT_THROW({
            const auto result = input.pow(-1);
            (void)result.cpu().to_vector_int();
        },
                     std::runtime_error);
    }
}

TEST(DiscoverySweep, IntegerModuloByZeroThrowsInsteadOfCrashing) {
    const auto reference = torch::tensor({7}, torch::kInt32);
    const auto zero = torch::tensor({0}, torch::kInt32);
    EXPECT_THROW(static_cast<void>(torch::remainder(reference, zero)), c10::Error);

    EXPECT_EXIT(
        {
            try {
                const auto input = Tensor::from_vector(
                    std::vector<int>{7}, {1}, Device::CPU);
                const auto divisor = Tensor::from_vector(
                    std::vector<int>{0}, {1}, Device::CPU);
                const auto values = input.mod(divisor).to_vector_int();
                (void)values;
                std::_Exit(2);
            } catch (const std::exception&) {
                std::_Exit(0);
            }
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST(DiscoverySweep, LargeNormalInplaceCallsDoNotReuseSubsequence) {
    constexpr size_t offset_step = 1'000'000;
    constexpr size_t tail_size = 8;
    constexpr uint64_t seed = 0x12345678ULL;

    Tensor::manual_seed(seed);
    auto first = Tensor::empty(
        {offset_step + tail_size}, Device::CUDA, DataType::Float32);
    auto second = Tensor::empty(
        {offset_step + tail_size}, Device::CUDA, DataType::Float32);
    first.normal_();
    second.normal_();

    const auto first_tail = lfs_float_values(
        first.slice(0, offset_step, offset_step + tail_size));
    const auto second_head = lfs_float_values(second.slice(0, 0, tail_size));

    torch::manual_seed(seed);
    auto torch_first = torch::empty(
        {static_cast<int64_t>(offset_step + tail_size)},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto torch_second = torch::empty_like(torch_first);
    torch_first.normal_();
    torch_second.normal_();
    ASSERT_FALSE(torch::equal(
        torch_first.slice(0, offset_step, offset_step + tail_size),
        torch_second.slice(0, 0, tail_size)))
        << "Torch consumes disjoint generator state across consecutive calls";

    EXPECT_NE(first_tail, second_head)
        << "consecutive normal_ calls reused the first call's millionth-value subsequence";
}

TEST(DiscoverySweep, ScatterMinMaxPropagateNaNsLikeTorch) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const auto indices = Tensor::from_vector(
        std::vector<int>{0}, {1}, Device::CPU);
    const auto source = Tensor::from_vector(
        std::vector<float>{nan}, {1}, Device::CPU);
    const auto torch_indices = torch::tensor({0}, torch::kInt64);
    const auto torch_source = torch::tensor({nan}, torch::kFloat32);

    for (const auto [mode, reduce] : {
             std::pair{ScatterMode::Max, std::string_view{"amax"}},
             std::pair{ScatterMode::Min, std::string_view{"amin"}}}) {
        auto actual = Tensor::from_vector(
            std::vector<float>{1.0f}, {1}, Device::CPU);
        const auto expected = torch::tensor({1.0f}, torch::kFloat32)
                                  .scatter_reduce(0, torch_indices, torch_source,
                                                  reduce, /*include_self=*/true);
        ASSERT_TRUE(std::isnan(expected.item<float>()));

        actual.scatter_(0, indices, source, mode);
        EXPECT_TRUE(std::isnan(actual.item<float>()))
            << reduce << " scatter reduction suppressed the source NaN";
    }
}

TEST(DiscoverySweep, MaxPool2dRejectsInvalidStrideAndPaddingLikeTorch) {
    const std::vector<float> values = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f};

    for (const auto device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto input = Tensor::from_vector(values, {1, 1, 4, 4}, device);
        const auto torch_device = device == Device::CPU ? torch::kCPU : torch::kCUDA;
        const auto reference = torch::tensor(
                                   values, torch::TensorOptions().dtype(torch::kFloat32).device(torch_device))
                                   .reshape({1, 1, 4, 4});

        EXPECT_THROW(
            (void)torch::max_pool2d(reference, {2, 2}, {1, 1}, {-1, -1}),
            c10::Error);
        EXPECT_THROW((void)input.max_pool2d(2, 1, -1), std::runtime_error)
            << "negative padding must not return a cropped-looking value";

        EXPECT_THROW(
            (void)torch::max_pool2d(reference, {2, 2}, {0, 0}),
            c10::Error);
        EXPECT_THROW((void)input.max_pool2d(2, 0), std::runtime_error)
            << "an explicit zero stride must not silently become kernel_size";
    }
}

TEST(DiscoverySweep, SqueezeProducesZeroDimensionalScalarsLikeTorch) {
    for (const auto device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto input = Tensor::full({1, 1}, 7.0f, device);
        const auto torch_device = device == Device::CPU ? torch::kCPU : torch::kCUDA;
        const auto reference = torch::full(
            {1, 1}, 7.0f,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch_device));

        const auto expected_all = reference.squeeze();
        const auto expected_dim = reference.squeeze(0).squeeze(0);
        ASSERT_EQ(expected_all.dim(), 0);
        ASSERT_EQ(expected_dim.dim(), 0);

        expect_float_tensor(input.squeeze(), expected_all,
                            "squeeze all singleton dimensions");
        expect_float_tensor(input.squeeze(0).squeeze(0), expected_dim,
                            "squeeze singleton dimensions explicitly");

        const auto scalar = Tensor::full({}, 7.0f, device);
        const auto torch_scalar = torch::scalar_tensor(
            7.0f, torch::TensorOptions().dtype(torch::kFloat32).device(torch_device));
        try {
            expect_float_tensor(scalar.squeeze(0), torch_scalar.squeeze(0),
                                "squeeze(dim=0) on a scalar");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "scalar squeeze(dim=0) unexpectedly threw: " << error.what();
        }
    }
}

TEST(DiscoverySweep, FlattenAcceptsZeroDimensionalScalarsLikeTorch) {
    for (const auto device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto input = Tensor::full({}, 7.0f, device);
        const auto torch_device = device == Device::CPU ? torch::kCPU : torch::kCUDA;
        const auto reference = torch::scalar_tensor(
            7.0f, torch::TensorOptions().dtype(torch::kFloat32).device(torch_device));
        const auto expected = reference.flatten();
        ASSERT_EQ(expected.dim(), 1);
        ASSERT_EQ(expected.size(0), 1);

        try {
            expect_float_tensor(input.flatten(), expected,
                                "flatten() on a scalar");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "scalar flatten unexpectedly threw: " << error.what();
        }
    }
}

TEST(DiscoverySweep, IntegerListReshapeCanProduceScalarsLikeTorch) {
    for (const auto device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto input = Tensor::full({1}, 7.0f, device);
        const auto torch_device = device == Device::CPU ? torch::kCPU : torch::kCUDA;
        const auto reference = torch::full(
            {1}, 7.0f,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch_device));
        const auto expected = reference.reshape(c10::IntArrayRef{});
        ASSERT_EQ(expected.dim(), 0);

        try {
            const auto actual = input.reshape(std::initializer_list<int>{});
            expect_float_tensor(actual, expected,
                                "reshape({}) through the integer-list overload");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "reshape({}) unexpectedly threw: " << error.what();
        }
    }
}

TEST(DiscoverySweep, TransposeAcceptsScalarDimensionAliasesLikeTorch) {
    for (const auto device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto input = Tensor::full({}, 7.0f, device);
        const auto torch_device = device == Device::CPU ? torch::kCPU : torch::kCUDA;
        const auto reference = torch::scalar_tensor(
            7.0f, torch::TensorOptions().dtype(torch::kFloat32).device(torch_device));
        const auto expected = reference.transpose(0, 0);
        ASSERT_EQ(expected.dim(), 0);

        try {
            expect_float_tensor(input.transpose(0, 0), expected,
                                "transpose(0, 0) on a scalar");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "scalar transpose unexpectedly threw: " << error.what();
        }
    }
}

TEST(DiscoverySweep, SerializationRoundTripsSupportedHighRankTensor) {
    const auto input = Tensor::full(
        {1, 1, 1, 1, 1, 1, 1, 1}, 3.0f,
        Device::CPU, DataType::Float32);
    ASSERT_EQ(input.ndim(), MAX_TENSOR_RANK);

    std::stringstream stream;
    ASSERT_NO_THROW(stream << input);

    Tensor loaded;
    try {
        stream >> loaded;
        expect_float_tensor(
            loaded,
            torch::full({1, 1, 1, 1, 1, 1, 1, 1}, 3.0f, torch::kFloat32),
            "maximum supported rank serialization round trip");
    } catch (const std::exception& error) {
        ADD_FAILURE() << "the serializer emitted a tensor its reader rejected: "
                      << error.what();
    }

    // Round 3 deliberately made MAX_TENSOR_RANK a fail-loud library contract.
    // Serialization therefore tests the maximum supported rank, while rank 9
    // must be rejected before a writer can emit an unreadable tensor.
    EXPECT_ANY_THROW((void)Tensor::full(
        {1, 1, 1, 1, 1, 1, 1, 1, 1}, 3.0f,
        Device::CPU, DataType::Float32));
}

TEST(DiscoverySweep, UnaryOnViewMatchesMaterializedLogicalValues) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> values = {0.0f, 0.0f, 0.0f, 0.0f,
                                       0.0f, 0.0f, nan, 0.0f};
    for (const auto device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto input = Tensor::from_vector(values, {4, 2}, device).transpose(0, 1);
        auto reference = torch::tensor(values, torch::kFloat32).reshape({4, 2}).transpose(0, 1);
        if (device == Device::CUDA) {
            reference = reference.cuda();
        }
        expect_float_tensor(input.neg(), reference.neg(),
                            "neg on a transposed view");
    }
}

TEST(DiscoverySweep, CatAcceptsZeroElementCudaInputsLikeTorch) {
    const auto input = Tensor::empty(
        {0, 1, 1}, Device::CUDA, DataType::Float32);
    const auto reference = torch::empty(
        {0, 1, 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    const auto expected = torch::cat({reference, reference}, 2);

    try {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        const auto actual = Tensor::cat({input, input}, 2);
        const auto status = cudaGetLastError();
        ASSERT_EQ(status, cudaSuccess)
            << "empty cat left a CUDA launch error: "
            << cudaGetErrorString(status);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        expect_float_tensor(actual, expected,
                            "CUDA cat with zero outer rows");
    } catch (const std::exception& error) {
        ADD_FAILURE() << "valid empty CUDA cat unexpectedly failed: " << error.what();
    }
}

TEST(DiscoverySweep, SquareThenCumsumOnSliceMatchesMaterializedValues) {
    const std::vector<float> values = {
        4.0f, 1.0f,
        -3.0f, -3.0f,
        5.0f, 0.0f,
        -1.0f, 4.0f};
    for (const auto device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto input = Tensor::from_vector(values, {4, 2}, device).slice(1, 1, 2);
        auto reference = torch::tensor(values, torch::kFloat32)
                             .reshape({4, 2})
                             .slice(1, 1, 2);
        if (device == Device::CUDA) {
            reference = reference.cuda();
        }

        const auto squared = input.square();
        const auto expected_squared = reference.square();
        expect_float_tensor(squared, expected_squared,
                            "square on a sliced column");
        expect_float_tensor(squared.cumsum(-1), expected_squared.cumsum(-1),
                            "cumsum after square on a sliced column");
    }
}

TEST(DiscoverySweep, CudaColumnReductionsAcceptZeroWidthOutputsLikeTorch) {
    const auto input = Tensor::empty({1, 0}, Device::CUDA, DataType::Float32);
    const auto reference = torch::empty(
        {1, 0}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    const auto check = [&](const char* name, const auto& lfs_reduce,
                           const auto& torch_reduce) {
        SCOPED_TRACE(name);
        static_cast<void>(cudaGetLastError());
        const auto actual = lfs_reduce();
        const auto expected = torch_reduce();
        expect_shape(actual, expected, name);
        const cudaError_t status = cudaGetLastError();
        EXPECT_EQ(status, cudaSuccess)
            << name << " left CUDA error " << static_cast<int>(status)
            << " (" << cudaGetErrorString(status) << ")";
    };

    check(
        "sum(dim=0)", [&] { return input.sum(0); },
        [&] { return reference.sum(0); });
    check(
        "mean(dim=0)", [&] { return input.mean(0); },
        [&] { return reference.mean(0); });
    check(
        "max(dim=0)", [&] { return input.max(0); },
        [&] { return std::get<0>(reference.max(0)); });
    check(
        "min(dim=0)", [&] { return input.min(0); },
        [&] { return std::get<0>(reference.min(0)); });
}

TEST(DiscoverySweep, LinspacePreservesSubnormalEndpointLikeTorch) {
    constexpr float low = 0.0f;
    const float high = std::numeric_limits<float>::denorm_min();

    for (const auto device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto actual = Tensor::linspace(low, high, 3, device);
        const auto torch_device = device == Device::CPU ? torch::kCPU : torch::kCUDA;
        const auto expected = torch::linspace(
            low, high, 3,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch_device));
        expect_float_tensor(actual, expected,
                            "linspace ending at the smallest Float32 subnormal");
        const auto actual_values = lfs_float_values(actual);
        const auto expected_values = torch_float_values(expected);
        ASSERT_EQ(expected_values.back(), high);
        EXPECT_EQ(actual_values.back(), expected_values.back())
            << "linspace must preserve its explicitly requested endpoint exactly";
    }
}

TEST(DiscoverySweep, ScalarCumsumAcceptsDimZeroLikeTorch) {
    for (const auto device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto input = Tensor::full({}, 7.0f, device);
        const auto torch_device = device == Device::CPU ? torch::kCPU : torch::kCUDA;
        const auto reference = torch::scalar_tensor(
            7.0f, torch::TensorOptions().dtype(torch::kFloat32).device(torch_device));

        try {
            expect_float_tensor(input.cumsum(0), reference.cumsum(0),
                                "scalar cumsum(dim=0)");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "scalar cumsum(dim=0) unexpectedly threw: " << error.what();
        }
    }
}

TEST(DiscoverySweep, ScalarAdvancedOpsAcceptDimZeroLikeTorch) {
    for (const auto device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto input = Tensor::full({}, 7.0f, device);
        const auto torch_device = device == Device::CPU ? torch::kCPU : torch::kCUDA;
        const auto reference = torch::scalar_tensor(
            7.0f, torch::TensorOptions().dtype(torch::kFloat32).device(torch_device));

        try {
            const auto [actual_values, actual_indices] = input.sort(0);
            const auto [expected_values, expected_indices] = reference.sort(0);
            expect_float_tensor(actual_values, expected_values, "scalar sort values");
            expect_int64_tensor(actual_indices, expected_indices, "scalar sort indices");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "scalar sort(dim=0) unexpectedly threw: " << error.what();
        }

        try {
            const auto [actual_values, actual_indices] = input.min_with_indices(0);
            const auto [expected_values, expected_indices] = reference.min(0);
            expect_float_tensor(actual_values, expected_values,
                                "scalar min_with_indices values");
            expect_int64_tensor(actual_indices, expected_indices,
                                "scalar min_with_indices indices");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "scalar min_with_indices(dim=0) unexpectedly threw: "
                          << error.what();
        }

        try {
            const auto [actual_values, actual_indices] = input.max_with_indices(0);
            const auto [expected_values, expected_indices] = reference.max(0);
            expect_float_tensor(actual_values, expected_values,
                                "scalar max_with_indices values");
            expect_int64_tensor(actual_indices, expected_indices,
                                "scalar max_with_indices indices");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "scalar max_with_indices(dim=0) unexpectedly threw: "
                          << error.what();
        }
    }
}

TEST(DiscoverySweep, ScalarDimensionalNormAcceptsDimZeroLikeTorch) {
    for (const auto device : {Device::CPU, Device::CUDA}) {
        SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
        const auto input = Tensor::full({}, -7.0f, device);
        const auto torch_device = device == Device::CPU ? torch::kCPU : torch::kCUDA;
        const auto reference = torch::scalar_tensor(
            -7.0f, torch::TensorOptions().dtype(torch::kFloat32).device(torch_device));
        const std::array<int64_t, 1> axes = {0};
        const auto expected = at::linalg_vector_norm(
            reference, 2.0, c10::IntArrayRef(axes), false);

        try {
            expect_float_tensor(input.norm(2.0f, 0), expected,
                                "scalar norm(p=2, dim=0)");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "scalar norm(p=2, dim=0) unexpectedly threw: "
                          << error.what();
        }
    }
}

TEST(DiscoverySweep, StackWritesContiguousInputsAlongInnerDimensions) {
    const auto input = Tensor::from_vector(
        std::vector<float>{4.0f}, {1, 1}, Device::CPU);
    const auto reference = torch::tensor(
                               std::vector<float>{4.0f}, torch::kFloat32)
                               .reshape({1, 1});

    expect_float_tensor(Tensor::stack({input, input}, 2),
                        torch::stack({reference, reference}, 2),
                        "CPU stack along an inserted inner dimension");
}
