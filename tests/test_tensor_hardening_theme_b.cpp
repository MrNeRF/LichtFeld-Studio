/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tensor_hardening_test_utils.hpp"

#include <numeric>

using namespace lfs::core;
using namespace tensor_hardening;

TEST_F(CudaTest, B1_Float16TransposeContiguousDispatchesAKernel) {
    const std::vector<float> values = {1, 2, 3, 4, 5, 6};
    const auto ours = lfs_float_tensor(values, {2, 3}, Device::CUDA)
                          .to(DataType::Float16)
                          .transpose(0, 1)
                          .contiguous();
    const auto theirs = torch::tensor(values, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 3})
                            .to(torch::kFloat16)
                            .transpose(0, 1)
                            .contiguous();
    expect_float_values_match(ours, theirs, "B1 CUDA half contiguous");
}

TEST_F(CudaTest, B1_Float16StridedCPUUploadDispatchesAKernel) {
    const std::vector<float> values = {1, 2, 3, 4, 5, 6};
    const auto ours = lfs_float_tensor(values, {2, 3}, Device::CPU)
                          .to(DataType::Float16)
                          .transpose(0, 1)
                          .cuda();
    const auto theirs = torch::tensor(values)
                            .reshape({2, 3})
                            .to(torch::kFloat16)
                            .transpose(0, 1)
                            .cuda();
    expect_float_values_match(ours, theirs, "B1 CPU half strided upload");
}

TEST_F(CudaTest, B2_HWCToCHWInt32UploadFallsBackToGenericDispatch) {
    std::vector<int> values(12);
    std::iota(values.begin(), values.end(), 1);
    const auto ours = lfs_int_tensor(values, {2, 2, 3}, Device::CPU)
                          .permute({2, 0, 1})
                          .cuda();
    const auto theirs = torch::tensor(values, torch::TensorOptions().dtype(torch::kInt32))
                            .reshape({2, 2, 3})
                            .permute({2, 0, 1})
                            .cuda();
    expect_int_values_match(ours, theirs, "B2 HWC Int32 upload");
}

TEST_F(CudaTest, B2_HWCToCHWBoolUploadFallsBackToGenericDispatch) {
    const std::vector<bool> values = {
        true, false, true, false, true, false,
        false, true, false, true, false, true};
    const auto ours = lfs_bool_tensor(values, {2, 2, 3}, Device::CPU)
                          .permute({2, 0, 1})
                          .cuda();
    const auto theirs = torch::tensor(
                            {true, false, true, false, true, false,
                             false, true, false, true, false, true},
                            torch::TensorOptions().dtype(torch::kBool))
                            .reshape({2, 2, 3})
                            .permute({2, 0, 1})
                            .cuda();
    expect_bool_values_match(ours, theirs, "B2 HWC Bool upload");
}

TEST_F(CudaTest, B2_HWCToCHWInt64UploadFallsBackToGenericDispatch) {
    std::vector<int> values(12);
    std::iota(values.begin(), values.end(), 1);
    const auto ours = lfs_int_tensor(values, {2, 2, 3}, Device::CPU)
                          .to(DataType::Int64)
                          .permute({2, 0, 1})
                          .cuda();
    const auto theirs = torch::tensor(values, torch::TensorOptions().dtype(torch::kInt64))
                            .reshape({2, 2, 3})
                            .permute({2, 0, 1})
                            .cuda();
    expect_int_values_match(ours, theirs, "B2 HWC Int64 upload");
}

TEST_F(CudaTest, B3_MultiAxisProdWritesEveryOutputElement) {
    const auto ours = Tensor::full({2, 2, 2}, 2.0f, Device::CUDA).prod({0, 1});
    const auto theirs = torch::full({2, 2, 2}, 2.0f,
                                    torch::TensorOptions().device(torch::kCUDA))
                            .prod(0)
                            .prod(0);
    expect_float_values_match(ours, theirs, "B3 multi-axis prod dispatch");
}

TEST(HardeningThemeB_DtypeDispatch, B4_CPUInt32PartialReductionWritesAndPromotes) {
    const auto ours = lfs_int_tensor({1, 2, 3, 4}, {2, 2}, Device::CPU).sum({0});
    const auto theirs = torch::tensor({1, 2, 3, 4}, torch::TensorOptions().dtype(torch::kInt32))
                            .reshape({2, 2})
                            .sum(0);
    EXPECT_EQ(ours.dtype(), DataType::Int64)
        << "B4 torch promotes Int32 sum to Int64";
    expect_int_values_match(ours, theirs, "B4 CPU Int32 partial sum");
}
