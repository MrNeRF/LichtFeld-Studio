/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "rendering/rasterizer/rasterization/include/rasterization_api_tensor.h"

#include <algorithm>
#include <array>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;

namespace {

    constexpr size_t LOCKED_GROUPS_SIZE = 8;

    Tensor make_bool_mask(const std::vector<uint8_t>& values) {
        return Tensor::from_vector(std::vector<bool>(values.begin(), values.end()), {values.size()}, Device::CUDA);
    }

    Tensor make_uint8_mask(const std::vector<uint8_t>& values) {
        auto tensor = Tensor::empty({values.size()}, Device::CPU, DataType::UInt8);
        std::copy(values.begin(), values.end(), tensor.ptr<uint8_t>());
        return tensor.cuda();
    }

    Tensor make_int32_values(const std::vector<int>& values) {
        return Tensor::from_vector(values, {values.size()}, Device::CUDA).to(DataType::Int32);
    }

    Tensor make_float32_values(const std::vector<float>& values, const lfs::core::TensorShape& shape) {
        return Tensor::from_vector(values, shape, Device::CUDA).to(DataType::Float32);
    }

    std::vector<uint8_t> run_group_apply(const std::vector<uint8_t>& selection_values,
                                         const std::vector<uint8_t>& existing_values,
                                         const uint8_t group_id,
                                         const std::vector<uint8_t>& locked_groups = {},
                                         const bool add_mode = true,
                                         const bool replace_mode = false,
                                         const std::vector<int>* transform_indices = nullptr,
                                         const std::vector<bool>& valid_nodes = {}) {
        if (selection_values.size() != existing_values.size()) {
            ADD_FAILURE() << "Selection and existing mask sizes differ";
            return {};
        }

        const auto selection = make_bool_mask(selection_values);
        const auto existing = make_uint8_mask(existing_values);
        auto output = Tensor::empty({selection_values.size()}, Device::CUDA, DataType::UInt8);

        std::array<uint32_t, LOCKED_GROUPS_SIZE> locked_bitmask{};
        for (const auto locked_group : locked_groups) {
            locked_bitmask[locked_group / 32] |= (1u << (locked_group % 32));
        }

        uint32_t* d_locked = nullptr;
        if (cudaMalloc(&d_locked, sizeof(locked_bitmask)) != cudaSuccess) {
            ADD_FAILURE() << "cudaMalloc failed for locked group mask";
            return {};
        }
        if (cudaMemcpy(d_locked, locked_bitmask.data(), sizeof(locked_bitmask), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_locked);
            ADD_FAILURE() << "cudaMemcpy failed for locked group mask";
            return {};
        }

        Tensor transform_indices_tensor;
        const Tensor* transform_indices_ptr = nullptr;
        if (transform_indices) {
            transform_indices_tensor = make_int32_values(*transform_indices);
            transform_indices_ptr = &transform_indices_tensor;
        }

        lfs::rendering::apply_selection_group_tensor_mask(
            selection,
            existing,
            output,
            group_id,
            d_locked,
            add_mode,
            transform_indices_ptr,
            valid_nodes,
            replace_mode);
        if (cudaFree(d_locked) != cudaSuccess) {
            ADD_FAILURE() << "cudaFree failed for locked group mask";
            return {};
        }

        return output.cpu().to_vector_uint8();
    }

} // namespace

class SelectionRasterizationOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(Tensor::zeros({1}, Device::CUDA).is_valid());
    }
};

TEST_F(SelectionRasterizationOpsTest, ReplaceClearsActiveGroupAndPreservesLockedGroups) {
    const auto result = run_group_apply(
        {1, 0, 0, 1, 1},
        {1, 1, 2, 2, 0},
        1,
        {2},
        true,
        true);

    EXPECT_EQ(result, (std::vector<uint8_t>{1, 0, 2, 2, 1}));
}

TEST_F(SelectionRasterizationOpsTest, AddRespectsNodeMask) {
    const std::vector<int> transform_indices = {0, 1, 0, 1};
    const std::vector<bool> valid_nodes = {true, false};

    const auto result = run_group_apply(
        {1, 1, 1, 1},
        {0, 1, 1, 0},
        1,
        {},
        true,
        false,
        &transform_indices,
        valid_nodes);

    EXPECT_EQ(result, (std::vector<uint8_t>{1, 1, 1, 0}));
}

TEST_F(SelectionRasterizationOpsTest, RemoveRespectsNodeMask) {
    const std::vector<int> transform_indices = {0, 1, 0, 1};
    const std::vector<bool> valid_nodes = {true, false};

    const auto result = run_group_apply(
        {1, 1, 1, 1},
        {1, 1, 1, 1},
        1,
        {},
        false,
        false,
        &transform_indices,
        valid_nodes);

    EXPECT_EQ(result, (std::vector<uint8_t>{0, 1, 0, 1}));
}

TEST_F(SelectionRasterizationOpsTest, BrushSelectMarksPointsInsideRadius) {
    const auto screen_positions = make_float32_values(
        {
            0.0f,
            0.0f,
            0.75f,
            0.0f,
            1.5f,
            0.0f,
            0.0f,
            2.0f,
        },
        {4, 2});
    auto selection = Tensor::zeros({4}, Device::CUDA, DataType::UInt8);

    lfs::rendering::brush_select_tensor(screen_positions, 0.0f, 0.0f, 1.0f, selection);

    EXPECT_EQ(selection.cpu().to_vector_uint8(), (std::vector<uint8_t>{1, 1, 0, 0}));
}

TEST_F(SelectionRasterizationOpsTest, RectSelectMarksPointsInsideBox) {
    const auto screen_positions = make_float32_values(
        {
            0.0f,
            0.0f,
            2.0f,
            2.0f,
            -0.1f,
            1.0f,
            1.0f,
            2.1f,
            3.0f,
            3.0f,
        },
        {5, 2});
    auto selection = Tensor::zeros({5}, Device::CUDA, DataType::Bool);

    lfs::rendering::rect_select_tensor(screen_positions, 0.0f, 0.0f, 2.0f, 2.0f, selection);

    EXPECT_EQ(selection.cpu().to_vector_bool(), (std::vector<bool>{true, true, false, false, false}));
}

TEST_F(SelectionRasterizationOpsTest, PolygonSelectMarksPointsInsideTriangle) {
    const auto screen_positions = make_float32_values(
        {
            0.5f,
            0.5f,
            1.0f,
            1.0f,
            3.0f,
            1.5f,
            -0.1f,
            0.5f,
            0.5f,
            3.0f,
        },
        {5, 2});
    const auto polygon = make_float32_values(
        {
            0.0f,
            0.0f,
            4.0f,
            0.0f,
            0.0f,
            4.0f,
        },
        {3, 2});
    auto selection = Tensor::zeros({5}, Device::CUDA, DataType::Bool);

    lfs::rendering::polygon_select_tensor(screen_positions, polygon, selection);

    EXPECT_EQ(selection.cpu().to_vector_bool(), (std::vector<bool>{true, true, false, false, true}));
}

TEST_F(SelectionRasterizationOpsTest, CropFilterKeepsOnlyPointsInsideCropBox) {
    auto selection = make_bool_mask({1, 1, 1, 1});
    const auto means = make_float32_values(
        {
            0.0f,
            0.0f,
            0.0f,
            2.0f,
            0.0f,
            0.0f,
            0.5f,
            0.5f,
            0.5f,
            -0.25f,
            0.0f,
            0.0f,
        },
        {4, 3});

    const auto crop_transform = make_float32_values(
        {
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
        },
        {4, 4});
    const auto crop_min = make_float32_values({-0.5f, -0.5f, -0.5f}, {3});
    const auto crop_max = make_float32_values({1.0f, 1.0f, 1.0f}, {3});

    lfs::rendering::filter_selection_by_crop(
        selection,
        means,
        &crop_transform,
        &crop_min,
        &crop_max,
        false,
        nullptr,
        nullptr,
        false);

    EXPECT_EQ(selection.cpu().to_vector_bool(), (std::vector<bool>{true, false, true, true}));
}

TEST_F(SelectionRasterizationOpsTest, DepthFilterKeepsOnlyPointsInsideCameraSpaceRange) {
    auto selection = make_bool_mask({1, 1, 1, 1, 1});
    const auto means = make_float32_values(
        {
            0.0f,
            0.0f,
            0.5f,
            0.0f,
            0.0f,
            2.0f,
            0.6f,
            0.0f,
            2.0f,
            0.0f,
            0.0f,
            5.0f,
            -0.25f,
            0.0f,
            3.0f,
        },
        {5, 3});

    const auto depth_transform = make_float32_values(
        {
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
        },
        {4, 4});
    const auto depth_min = make_float32_values({-0.5f, -10000.0f, 1.0f}, {3});
    const auto depth_max = make_float32_values({0.5f, 10000.0f, 4.0f}, {3});

    lfs::rendering::filter_selection_by_crop(
        selection,
        means,
        &depth_transform,
        &depth_min,
        &depth_max,
        false,
        nullptr,
        nullptr,
        false);

    EXPECT_EQ(selection.cpu().to_vector_bool(), (std::vector<bool>{false, true, false, false, true}));
}
