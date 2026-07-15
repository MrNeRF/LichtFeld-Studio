/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tensor_hardening_test_utils.hpp"

#include <algorithm>
#include <array>
#include <numeric>

using namespace lfs::core;
using namespace tensor_hardening;

namespace {

    template <typename CustomMutation, typename TorchMutation>
    void check_view_mutation(const Device device, CustomMutation&& custom_mutation,
                             TorchMutation&& torch_mutation, const std::string& context) {
        const std::vector<float> values = {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f};
        auto ours = lfs_float_tensor(values, {4, 4}, device);
        auto theirs = torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32))
                          .reshape({4, 4});
        if (device == Device::CUDA) {
            theirs = theirs.cuda();
        }

        custom_mutation(ours);
        torch_mutation(theirs);
        expect_float_values_match(ours, theirs, context);
    }

    std::vector<bool> changed_from_zero(const Tensor& tensor) {
        const auto values = lfs_values_as_float(tensor);
        std::vector<bool> changed;
        changed.reserve(values.size());
        for (const float value : values) {
            changed.push_back(value != 0.0f);
        }
        return changed;
    }

    std::vector<bool> changed_from_zero(const torch::Tensor& tensor) {
        const auto values = torch_values_as_float(tensor);
        std::vector<bool> changed;
        changed.reserve(values.size());
        for (const float value : values) {
            changed.push_back(value != 0.0f);
        }
        return changed;
    }

} // namespace

TEST_F(CudaTest, A1_NonContiguousZeroFollowsViewStrides_CPUAndCUDA) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        check_view_mutation(
            device,
            [](Tensor& base) { base.slice(0, 0, 2).slice(1, 0, 2).zero_(); },
            [](torch::Tensor& base) { base.slice(0, 0, 2).slice(1, 0, 2).zero_(); },
            device == Device::CPU ? "A1 zero CPU" : "A1 zero CUDA");
    }
}

TEST_F(CudaTest, A1_NonContiguousScalarInPlaceFollowsViewStrides_CPUAndCUDA) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        check_view_mutation(
            device,
            [](Tensor& base) { base.slice(1, 1, 2).squeeze(1).add_(10.0f); },
            [](torch::Tensor& base) { base.slice(1, 1, 2).squeeze(1).add_(10.0f); },
            device == Device::CPU ? "A1 scalar add CPU" : "A1 scalar add CUDA");
    }
}

TEST_F(CudaTest, A1_NonContiguousBinaryInPlaceFollowsViewStrides_CPUAndCUDA) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        check_view_mutation(
            device,
            [device](Tensor& base) {
                auto view = base.slice(1, 1, 2).squeeze(1);
                view.add_(Tensor::full({4}, 10.0f, device));
            },
            [](torch::Tensor& base) {
                auto view = base.slice(1, 1, 2).squeeze(1);
                view.add_(torch::full({4}, 10.0f, base.options()));
            },
            device == Device::CPU ? "A1 binary add CPU" : "A1 binary add CUDA");
    }
}

TEST_F(CudaTest, A1_NonContiguousClampFollowsViewStrides_CPUAndCUDA) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        check_view_mutation(
            device,
            [](Tensor& base) { base.slice(1, 1, 2).squeeze(1).clamp_(6.0f, 10.0f); },
            [](torch::Tensor& base) { base.slice(1, 1, 2).squeeze(1).clamp_(6.0f, 10.0f); },
            device == Device::CPU ? "A1 clamp CPU" : "A1 clamp CUDA");
    }
}

TEST(HardeningThemeA_Strided, A2_CPUCopyFromWritesLogicalDestinationCells) {
    auto base = Tensor::zeros({2, 2}, Device::CPU);
    auto destination = base.transpose(0, 1);
    auto source = lfs_float_tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, Device::CPU);
    destination.copy_from(source);

    auto torch_base = torch::zeros({2, 2});
    auto torch_destination = torch_base.transpose(0, 1);
    torch_destination.copy_(torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}));
    expect_float_values_match(base, torch_base, "A2 CPU copy_from");
}

TEST(HardeningThemeA_Strided, A3_ViewCopyAssignmentUsesBothTensorStrides) {
    auto base = Tensor::zeros({3, 3}, Device::CPU);
    auto destination = base.slice(1, 0, 1);
    const auto source = Tensor::full({3, 1}, 9.0f, Device::CPU);
    destination = source;

    auto torch_base = torch::zeros({3, 3});
    torch_base.slice(1, 0, 1).copy_(torch::full({3, 1}, 9.0f));
    expect_float_values_match(base, torch_base, "A3 copy assignment");
}

TEST(HardeningThemeA_Strided, A3_ViewMoveAssignmentUsesBothTensorStrides) {
    auto base = Tensor::zeros({3, 3}, Device::CPU);
    auto destination = base.slice(1, 0, 1);
    destination = Tensor::full({3, 1}, 7.0f, Device::CPU);

    auto torch_base = torch::zeros({3, 3});
    torch_base.slice(1, 0, 1).copy_(torch::full({3, 1}, 7.0f));
    expect_float_values_match(base, torch_base, "A3 move assignment");
}

TEST(HardeningThemeA_Strided, A4_RangeSlicePreservesExistingStorageOffset) {
    auto ours = Tensor::arange(10.0f).to(Device::CPU);
    auto ours_result = ours.slice(0, 2, 8).slice({{1, 3}});

    auto theirs = torch::arange(10.0f);
    auto torch_result = theirs.slice(0, 2, 8).slice(0, 1, 3);
    expect_float_values_match(ours_result, torch_result, "A4 chained range slice");
}

TEST(HardeningThemeA_Strided, A4_RangeSlicePreservesSourceStrides) {
    const std::vector<float> values = {1, 2, 3, 4, 5, 6};
    auto ours = lfs_float_tensor(values, {2, 3}, Device::CPU).transpose(0, 1);
    auto ours_result = ours.slice({{1, 3}, {0, 2}});

    auto theirs = torch::tensor(values).reshape({2, 3}).transpose(0, 1);
    auto torch_result = theirs.slice(0, 1, 3).slice(1, 0, 2);
    expect_float_values_match(ours_result, torch_result, "A4 range slice of transpose");
}

TEST(HardeningThemeA_Strided, A5_Int32VectorExportUsesLogicalOrder) {
    auto ours = lfs_int_tensor({1, 2, 3, 4}, {2, 2}, Device::CPU).transpose(0, 1);
    const std::vector<int> expected = {1, 3, 2, 4};
    EXPECT_EQ(ours.to_vector_int(), expected);
}

TEST(HardeningThemeA_Strided, A5_Int64VectorExportUsesLogicalOrder) {
    auto ours = lfs_int_tensor({1, 2, 3, 4}, {2, 2}, Device::CPU)
                    .to(DataType::Int64)
                    .transpose(0, 1);
    const std::vector<int64_t> expected = {1, 3, 2, 4};
    EXPECT_EQ(ours.to_vector_int64(), expected);
}

TEST(HardeningThemeA_Strided, A5_BoolAndUInt8VectorExportsUseLogicalOrder) {
    auto bool_view = lfs_bool_tensor({true, false, true, true}, {2, 2}, Device::CPU)
                         .transpose(0, 1);
    const std::vector<bool> expected_bool = {true, true, false, true};
    EXPECT_EQ(bool_view.to_vector_bool(), expected_bool);

    auto byte_view = lfs_int_tensor({1, 2, 3, 4}, {2, 2}, Device::CPU)
                         .to(DataType::UInt8)
                         .transpose(0, 1);
    const std::vector<uint8_t> expected_bytes = {1, 3, 2, 4};
    EXPECT_EQ(byte_view.to_vector_uint8(), expected_bytes);
}

TEST(HardeningThemeA_Strided, A6_CPUAllCloseComparesLogicalOrder) {
    auto transposed = lfs_float_tensor({1, 2, 3, 4}, {2, 2}, Device::CPU).transpose(0, 1);
    auto dense_physical_order = lfs_float_tensor({1, 2, 3, 4}, {2, 2}, Device::CPU);

    const auto torch_transposed = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}).transpose(0, 1);
    const auto torch_dense = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    EXPECT_EQ(transposed.all_close(dense_physical_order),
              torch::allclose(torch_transposed, torch_dense));
}

TEST_F(CudaTest, A7_CUDASpecialValueScansFollowViewStrides) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    auto base = lfs_float_tensor({0, 0, 0, 0, 0, 0, nan, 0, 0}, {3, 3}, Device::CUDA);
    auto view = base.slice(1, 0, 1).squeeze(1);
    auto torch_base = torch::tensor({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, nan, 0.0f, 0.0f},
                                    torch::TensorOptions().device(torch::kCUDA))
                          .reshape({3, 3});
    auto torch_view = torch_base.slice(1, 0, 1).squeeze(1);
    EXPECT_EQ(view.has_nan(), torch_view.isnan().any().item<bool>());

    auto inf_base = lfs_float_tensor(
        {0, 0, 0, 0, 0, 0, std::numeric_limits<float>::infinity(), 0, 0},
        {3, 3}, Device::CUDA);
    auto inf_view = inf_base.slice(1, 0, 1).squeeze(1);
    auto torch_inf = torch::tensor(
                         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          std::numeric_limits<float>::infinity(), 0.0f, 0.0f},
                         torch::TensorOptions().device(torch::kCUDA))
                         .reshape({3, 3})
                         .slice(1, 0, 1)
                         .squeeze(1);
    EXPECT_EQ(inf_view.has_inf(), torch_inf.isinf().any().item<bool>());
}

TEST(HardeningThemeA_Strided, A8_ReserveOnOffsetViewPreservesLogicalValues) {
    auto base = lfs_float_tensor({1, 2, 3, 4, 5, 6}, {3, 2}, Device::CPU);
    auto view = base.slice(0, 1, 3);
    const auto torch_reference = torch::tensor({{3.0f, 4.0f}, {5.0f, 6.0f}});
    view.reserve(4);
    expect_float_values_match(view, torch_reference, "A8 reserve view");
}

TEST_F(CudaTest, A9_CatReadsNonContiguousOperands_CPUAndCUDA) {
    const std::vector<float> values = {1, 2, 3, 4, 5, 6};
    for (const Device device : {Device::CPU, Device::CUDA}) {
        const auto ours = lfs_float_tensor(values, {2, 3}, device).transpose(0, 1);
        auto theirs = torch::tensor(values).reshape({2, 3}).transpose(0, 1);
        if (device == Device::CUDA) {
            theirs = theirs.cuda();
        }
        expect_float_values_match(Tensor::cat({ours, ours}, 0),
                                  torch::cat({theirs, theirs}, 0),
                                  device == Device::CPU ? "A9 cat CPU" : "A9 cat CUDA");
    }
}

TEST_F(CudaTest, A9_StackReadsNonContiguousOperands_CPUAndCUDA) {
    const std::vector<float> values = {1, 2, 3, 4, 5, 6};
    for (const Device device : {Device::CPU, Device::CUDA}) {
        const auto ours = lfs_float_tensor(values, {2, 3}, device).transpose(0, 1);
        auto theirs = torch::tensor(values).reshape({2, 3}).transpose(0, 1);
        if (device == Device::CUDA) {
            theirs = theirs.cuda();
        }
        expect_float_values_match(Tensor::stack({ours, ours}, 0),
                                  torch::stack({theirs, theirs}, 0),
                                  device == Device::CPU ? "A9 stack CPU" : "A9 stack CUDA");
    }
}

TEST_F(CudaTest, A10_MaskedSelectReadsLogicalViewOrder_CPUAndCUDA) {
    const std::vector<float> values = {1, 2, 3, 4, 5, 6};
    const std::vector<bool> mask_values = {false, false, false, true, false, false};
    for (const Device device : {Device::CPU, Device::CUDA}) {
        const auto ours = lfs_float_tensor(values, {2, 3}, device).transpose(0, 1);
        const auto mask = lfs_bool_tensor(mask_values, {3, 2}, device);
        auto theirs = torch::tensor(values).reshape({2, 3}).transpose(0, 1);
        auto torch_mask = torch::tensor(
                              {false, false, false, true, false, false},
                              torch::TensorOptions().dtype(torch::kBool))
                              .reshape({3, 2});
        if (device == Device::CUDA) {
            theirs = theirs.cuda();
            torch_mask = torch_mask.cuda();
        }
        expect_float_values_match(ours.masked_select(mask), theirs.masked_select(torch_mask),
                                  device == Device::CPU ? "A10 select CPU" : "A10 select CUDA");
    }
}

TEST_F(CudaTest, A10_MaskedFillWritesLogicalViewCells_CPUAndCUDA) {
    const std::vector<float> values = {1, 2, 3, 4, 5, 6};
    const std::vector<bool> mask_values = {false, false, false, true, false, false};
    for (const Device device : {Device::CPU, Device::CUDA}) {
        auto ours_base = lfs_float_tensor(values, {2, 3}, device);
        auto ours_view = ours_base.transpose(0, 1);
        const auto mask = lfs_bool_tensor(mask_values, {3, 2}, device);
        ours_view.masked_fill_(mask, 99.0f);

        auto torch_base = torch::tensor(values).reshape({2, 3});
        auto torch_mask = torch::tensor(
                              {false, false, false, true, false, false},
                              torch::TensorOptions().dtype(torch::kBool))
                              .reshape({3, 2});
        if (device == Device::CUDA) {
            torch_base = torch_base.cuda();
            torch_mask = torch_mask.cuda();
        }
        torch_base.transpose(0, 1).masked_fill_(torch_mask, 99.0f);
        expect_float_values_match(ours_base, torch_base,
                                  device == Device::CPU ? "A10 fill CPU" : "A10 fill CUDA");
    }
}

TEST_F(CudaTest, A11_IndexPutOnOffsetViewWritesLogicalCells_CPUAndCUDA) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        auto ours_base = Tensor::zeros({6}, device);
        auto ours_view = ours_base.slice(0, 2, 5);
        const auto index = lfs_int_tensor({1}, {1}, device);
        const auto value = lfs_float_tensor({9.0f}, {1}, device);
        ours_view.index_put_(index, value);

        auto torch_base = torch::zeros({6}, torch::TensorOptions().device(
                                                device == Device::CUDA ? torch::kCUDA : torch::kCPU));
        auto torch_view = torch_base.slice(0, 2, 5);
        auto torch_index = torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt64).device(torch_base.device()));
        torch_view.index_put_({torch_index}, torch::tensor({9.0f}, torch_base.options()));
        expect_float_values_match(ours_base, torch_base,
                                  device == Device::CPU ? "A11 index_put CPU" : "A11 index_put CUDA");
    }
}

TEST_F(CudaTest, A12_LinearReadsStridedBias) {
    const auto input = lfs_float_tensor({1, 2, 3, 4}, {2, 2}, Device::CUDA);
    const auto weight = lfs_float_tensor({1, 0, 0, 1}, {2, 2}, Device::CUDA);
    const auto bias_base = lfs_float_tensor({10, 99, 20, 99}, {2, 2}, Device::CUDA);
    const auto bias = bias_base.transpose(0, 1).slice(0, 0, 1).squeeze(0);

    const auto torch_input = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}, torch::kCUDA);
    const auto torch_weight = torch::eye(2, torch::TensorOptions().device(torch::kCUDA));
    const auto torch_bias_base = torch::tensor({{10.0f, 99.0f}, {20.0f, 99.0f}}, torch::kCUDA);
    const auto torch_bias = torch_bias_base.transpose(0, 1).slice(0, 0, 1).squeeze(0);
    expect_float_values_match(input.linear(weight, bias),
                              torch::linear(torch_input, torch_weight, torch_bias),
                              "A12 linear strided bias");
}

TEST_F(CudaTest, A13_LinearOutWritesLogicalOutputLayout) {
    const auto input = lfs_float_tensor({1, 2, 3, 4}, {2, 2}, Device::CUDA);
    const auto weight = lfs_float_tensor({1, 0, 0, 1}, {2, 2}, Device::CUDA);
    Tensor no_bias;
    auto output_base = Tensor::zeros({2, 2}, Device::CUDA);
    auto output_view = output_base.transpose(0, 1);
    input.linear_out(weight, no_bias, output_view);

    const auto torch_input = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}, torch::kCUDA);
    const auto torch_expected = torch_input.clone();
    expect_float_values_match(output_view, torch_expected, "A13 linear_out view");
}

TEST_F(CudaTest, A14_GatherReadsStridedInt32Indices) {
    const auto input = lfs_float_tensor({10, 20, 30}, {3}, Device::CUDA);
    const auto index_base = lfs_int_tensor({2, 0, 1, 0, 0, 0}, {3, 2}, Device::CUDA);
    const auto index = index_base.slice(1, 0, 1).squeeze(1);

    const auto torch_input = torch::tensor({10.0f, 20.0f, 30.0f}, torch::kCUDA);
    const auto torch_index_base = torch::tensor({{2, 0}, {1, 0}, {0, 0}},
                                                torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    const auto torch_index = torch_index_base.slice(1, 0, 1).squeeze(1);
    expect_float_values_match(input.gather(0, index), torch_input.gather(0, torch_index),
                              "A14 strided gather index");
}

TEST_F(CudaTest, A15_IndexSelectReadsLogicalSourceAndIndices) {
    const auto input = lfs_float_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, Device::CUDA)
                           .transpose(0, 1);
    const auto index_base = lfs_int_tensor({1, 0, 2, 0}, {2, 2}, Device::CUDA);
    const auto index = index_base.slice(1, 0, 1).squeeze(1);

    const auto torch_input = torch::tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, torch::kCUDA)
                                 .transpose(0, 1);
    const auto torch_index = torch::tensor({1, 2},
                                           torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    expect_float_values_match(input.index_select(0, index),
                              torch_input.index_select(0, torch_index),
                              "A15 index_select source/index");
}

TEST_F(CudaTest, A15_TakeFlattensLogicalOrder) {
    const auto input = lfs_float_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, Device::CUDA)
                           .transpose(0, 1);
    const auto index = lfs_int_tensor({1, 4}, {2}, Device::CUDA);
    const auto torch_input = torch::tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, torch::kCUDA)
                                 .transpose(0, 1);
    const auto torch_index = torch::tensor({1, 4},
                                           torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    expect_float_values_match(input.take(index), torch::take(torch_input, torch_index),
                              "A15 take logical flatten");
}

TEST_F(CudaTest, A15_IndexSelectIntoWritesLogicalOutputLayout) {
    const auto input = lfs_float_tensor({1, 2, 3, 4}, {2, 2}, Device::CUDA);
    const auto index = lfs_int_tensor({1, 0}, {2}, Device::CUDA);
    auto output_base = Tensor::zeros({2, 2}, Device::CUDA);
    auto output_view = output_base.transpose(0, 1);
    input.index_select_into(output_view, 0, index, BoundaryMode::Assert);

    const auto torch_input = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}, torch::kCUDA);
    const auto torch_index = torch::tensor({1, 0},
                                           torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    expect_float_values_match(output_view, torch_input.index_select(0, torch_index),
                              "A15 index_select_into output");
}

TEST_F(CudaTest, A16_ScatterWritesStridedDestination) {
    auto ours_base = Tensor::zeros({3, 2}, Device::CUDA);
    auto ours_view = ours_base.slice(1, 0, 1).squeeze(1);
    const auto index = lfs_int_tensor({2, 0, 1}, {3}, Device::CUDA);
    const auto source = lfs_float_tensor({7, 8, 9}, {3}, Device::CUDA);
    ours_view.scatter_(0, index, source);

    auto torch_base = torch::zeros({3, 2}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_view = torch_base.slice(1, 0, 1).squeeze(1);
    const auto torch_index = torch::tensor({2, 0, 1},
                                           torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    torch_view.scatter_(0, torch_index, torch::tensor({7.0f, 8.0f, 9.0f}, torch::kCUDA));
    expect_float_values_match(ours_base, torch_base, "A16 scatter destination");
}

TEST_F(CudaTest, A16_IndexCopyReadsAndWritesStridedOperands) {
    auto ours_base = Tensor::zeros({3, 2}, Device::CUDA);
    auto ours_view = ours_base.slice(1, 0, 1).squeeze(1);
    const auto index = lfs_int_tensor({2, 0, 1}, {3}, Device::CUDA);
    const auto source_base = lfs_float_tensor({7, -1, 8, -1, 9, -1}, {3, 2}, Device::CUDA);
    const auto source = source_base.slice(1, 0, 1).squeeze(1);
    ours_view.index_copy_(0, index, source);

    auto torch_base = torch::zeros({3, 2}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_view = torch_base.slice(1, 0, 1).squeeze(1);
    const auto torch_index = torch::tensor({2, 0, 1},
                                           torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    const auto torch_source = torch::tensor({{7.0f, -1.0f}, {8.0f, -1.0f}, {9.0f, -1.0f}}, torch::kCUDA)
                                  .slice(1, 0, 1)
                                  .squeeze(1);
    torch_view.index_copy_(0, torch_index, torch_source);
    expect_float_values_match(ours_base, torch_base, "A16 index_copy operands");
}

TEST_F(CudaTest, A16_IndexAddReadsAndWritesStridedOperands) {
    auto ours_base = Tensor::zeros({3, 2}, Device::CUDA);
    auto ours_view = ours_base.slice(1, 0, 1).squeeze(1);
    const auto index = lfs_int_tensor({2, 0, 1}, {3}, Device::CUDA);
    const auto source_base = lfs_float_tensor({7, -1, 8, -1, 9, -1}, {3, 2}, Device::CUDA);
    const auto source = source_base.slice(1, 0, 1).squeeze(1);
    ours_view.index_add_(0, index, source);

    auto torch_base = torch::zeros({3, 2}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_view = torch_base.slice(1, 0, 1).squeeze(1);
    const auto torch_index = torch::tensor({2, 0, 1},
                                           torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    const auto torch_source = torch::tensor({{7.0f, -1.0f}, {8.0f, -1.0f}, {9.0f, -1.0f}}, torch::kCUDA)
                                  .slice(1, 0, 1)
                                  .squeeze(1);
    torch_view.index_add_(0, torch_index, torch_source);
    expect_float_values_match(ours_base, torch_base, "A16 index_add operands");
}

TEST_F(CudaTest, A17_UniformTouchesOnlyLogicalViewCells_CPUAndCUDA) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        auto ours_base = Tensor::zeros({3, 4}, device);
        auto ours_view = ours_base.slice(1, 1, 3);
        ours_view.uniform_(2.0f, 4.0f);

        auto torch_base = torch::zeros({3, 4}, torch::TensorOptions().device(
                                                   device == Device::CUDA ? torch::kCUDA : torch::kCPU));
        torch_base.slice(1, 1, 3).uniform_(2.0, 4.0);
        EXPECT_EQ(changed_from_zero(ours_base), changed_from_zero(torch_base))
            << (device == Device::CPU ? "A17 uniform CPU" : "A17 uniform CUDA");
    }
}

TEST_F(CudaTest, A17_NormalTouchesOnlyLogicalViewCells_CPUAndCUDA) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        auto ours_base = Tensor::zeros({3, 4}, device);
        auto ours_view = ours_base.slice(1, 1, 3);
        ours_view.normal_(10.0f, 0.01f);

        auto torch_base = torch::zeros({3, 4}, torch::TensorOptions().device(
                                                   device == Device::CUDA ? torch::kCUDA : torch::kCPU));
        torch_base.slice(1, 1, 3).normal_(10.0, 0.01);
        EXPECT_EQ(changed_from_zero(ours_base), changed_from_zero(torch_base))
            << (device == Device::CPU ? "A17 normal CPU" : "A17 normal CUDA");
    }
}

TEST_F(CudaTest, A18_RankOneGatherBroadcastsAcrossOuterDimensions) {
    const auto ours = lfs_float_tensor({10, 11, 12, 20, 21, 22}, {2, 3}, Device::CUDA);
    const auto index = lfs_int_tensor({2, 0}, {2}, Device::CUDA);
    const auto theirs = torch::tensor({{10.0f, 11.0f, 12.0f}, {20.0f, 21.0f, 22.0f}}, torch::kCUDA);
    const auto torch_index = torch::tensor({2, 0},
                                           torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    expect_float_values_match(ours.gather(1, index), theirs.index_select(1, torch_index),
                              "A18 rank-one gather");
}

TEST_F(CudaTest, A19_MaskedScatterTreatsUInt8MaskAsTruthValues) {
    auto ours = Tensor::zeros({2}, Device::CUDA);
    const auto mask = lfs_int_tensor({2, 1}, {2}, Device::CUDA).to(DataType::UInt8);
    const auto source = lfs_float_tensor({10, 20}, {2}, Device::CUDA);
    ours[mask] = source;

    auto theirs = torch::zeros({2}, torch::TensorOptions().device(torch::kCUDA));
    const auto torch_mask = torch::tensor({true, true},
                                          torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    theirs.masked_scatter_(torch_mask, torch::tensor({10.0f, 20.0f}, torch::kCUDA));
    expect_float_values_match(ours, theirs, "A19 UInt8 masked scatter");
}

TEST_F(CudaTest, A20_MultinomialSamplesLogicalStridedWeights) {
    const auto base = lfs_float_tensor({1, 100, 0, 0}, {2, 2}, Device::CUDA);
    const auto weights = base.transpose(0, 1).slice(0, 0, 1).squeeze(0);
    const auto torch_base = torch::tensor({{1.0f, 100.0f}, {0.0f, 0.0f}}, torch::kCUDA);
    const auto torch_weights = torch_base.transpose(0, 1).slice(0, 0, 1).squeeze(0);

    const auto ours = Tensor::multinomial(weights, 512, true);
    const auto theirs = torch::multinomial(torch_weights, 512, true);
    const auto ours_values = ours.cpu().to_vector_int64();
    const auto torch_cpu = theirs.cpu().contiguous();
    const auto* torch_values = torch_cpu.data_ptr<int64_t>();
    EXPECT_EQ(std::count(ours_values.begin(), ours_values.end(), 0),
              std::count(torch_values, torch_values + torch_cpu.numel(), int64_t{0}));
}

TEST_F(CudaTest, A21_DiagReadsRankOneStride_CPUAndCUDA) {
    const std::vector<float> values = {1, 2, 3, 4};
    for (const Device device : {Device::CPU, Device::CUDA}) {
        const auto base = lfs_float_tensor(values, {2, 2}, device);
        const auto diagonal = base.transpose(0, 1).slice(0, 0, 1).squeeze(0);
        auto torch_base = torch::tensor(values).reshape({2, 2});
        if (device == Device::CUDA) {
            torch_base = torch_base.cuda();
        }
        const auto torch_diagonal = torch_base.transpose(0, 1).slice(0, 0, 1).squeeze(0);
        expect_float_values_match(Tensor::diag(diagonal), torch::diag(torch_diagonal),
                                  device == Device::CPU ? "A21 diag CPU" : "A21 diag CUDA");
    }
}
