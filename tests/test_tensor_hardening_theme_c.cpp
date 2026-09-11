/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tensor_hardening_test_utils.hpp"

#include <limits>
#include <numeric>

using namespace lfs::core;
using namespace tensor_hardening;

TEST_F(CudaTest, C1_NonSuffixMultiAxisSumUsesLogicalReductionAxes) {
    std::vector<float> values(8);
    std::iota(values.begin(), values.end(), 1.0f);
    const auto ours = lfs_float_tensor(values, {2, 2, 2}, Device::CUDA).sum({0, 1});
    const std::vector<int64_t> axes = {0, 1};
    const auto theirs = torch::tensor(values, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 2, 2})
                            .sum(axes);
    expect_float_values_match(ours, theirs, "C1 non-suffix multi-axis sum");
}

TEST_F(CudaTest, C2_SuffixMultiAxisMeanNormalizesExactlyOnce) {
    std::vector<float> values(24);
    std::iota(values.begin(), values.end(), 1.0f);
    const auto ours = lfs_float_tensor(values, {2, 3, 4}, Device::CUDA).mean({1, 2});
    const std::vector<int64_t> axes = {1, 2};
    const auto theirs = torch::tensor(values, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 3, 4})
                            .mean(axes);
    expect_float_values_match(ours, theirs, "C2 suffix multi-axis mean");
}

TEST_F(CudaTest, C3_RGBAlphaCatReadsProvidedAlphaTensor) {
    const auto rgb = lfs_float_tensor({0.1f, 0.2f, 0.3f,
                                       0.4f, 0.5f, 0.6f},
                                      {2, 3}, Device::CUDA);
    const auto alpha = lfs_float_tensor({0.2f, 0.7f}, {2, 1}, Device::CUDA);
    const auto torch_rgb = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f},
                                         torch::TensorOptions().device(torch::kCUDA))
                               .reshape({2, 3});
    const auto torch_alpha = torch::tensor({0.2f, 0.7f},
                                           torch::TensorOptions().device(torch::kCUDA))
                                 .reshape({2, 1});
    expect_float_values_match(Tensor::cat({rgb, alpha}, -1),
                              torch::cat({torch_rgb, torch_alpha}, -1),
                              "C3 RGB plus alpha cat");
}

TEST_F(CudaTest, C4_Int32FullSumPromotesAndDoesNotOverflow_CPUAndCUDA) {
    const std::vector<int> values = {std::numeric_limits<int>::max(), 1};
    for (const Device device : {Device::CPU, Device::CUDA}) {
        const auto ours = lfs_int_tensor(values, {2}, device).sum();
        auto theirs = torch::tensor(values, torch::TensorOptions().dtype(torch::kInt32));
        if (device == Device::CUDA) {
            theirs = theirs.cuda();
        }
        const auto torch_sum = theirs.sum();
        EXPECT_EQ(ours.dtype(), DataType::Int64)
            << (device == Device::CPU ? "C4 CPU result dtype" : "C4 CUDA result dtype");
        expect_int_values_match(ours, torch_sum,
                                device == Device::CPU ? "C4 CPU sum" : "C4 CUDA sum");
    }
}

TEST_F(CudaTest, C5_IntegerMeanRejectsLikeTorch_CPUAndCUDA) {
    const std::vector<int> values = {1, 2, 3, 4};
    for (const Device device : {Device::CPU, Device::CUDA}) {
        auto theirs = torch::tensor(values, torch::TensorOptions().dtype(torch::kInt32));
        if (device == Device::CUDA) {
            theirs = theirs.cuda();
        }
        EXPECT_THROW(theirs.mean(), c10::Error);

        const auto ours = lfs_int_tensor(values, {2, 2}, device);
        EXPECT_THROW(static_cast<void>(ours.mean()), std::exception)
            << (device == Device::CPU ? "C5 CPU integer mean" : "C5 CUDA integer mean");
    }
}

TEST_F(CudaTest, C6_TransposeBeforeReduceRestoresKeepdimAxisOrder) {
    std::vector<float> values(2 * 3 * 256);
    std::iota(values.begin(), values.end(), 1.0f);
    const auto ours = lfs_float_tensor(values, {2, 3, 256}, Device::CUDA).sum({1}, true);
    const std::vector<int64_t> axis = {1};
    const auto theirs = torch::tensor(values, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 3, 256})
                            .sum(axis, true);
    expect_float_values_match(ours, theirs, "C6 keepdim axis restoration");
}

TEST_F(CudaTest, C7_EmptyDimensionalNormWritesIdentityValues_CPUAndCUDA) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        const auto ours = Tensor::empty({0, 3}, device).norm(2.0f, {0});
        auto theirs = torch::empty({0, 3});
        if (device == Device::CUDA) {
            theirs = theirs.cuda();
        }
        const auto torch_norm = theirs.norm(2, 0);
        expect_float_values_match(ours, torch_norm,
                                  device == Device::CPU ? "C7 CPU empty norm" : "C7 CUDA empty norm");
    }
}
