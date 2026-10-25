/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "rendering/selection_ops.hpp"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;

namespace {

    Tensor make_bool_mask(const std::vector<uint8_t>& values) {
        return Tensor::from_vector(std::vector<bool>(values.begin(), values.end()), {values.size()}, Device::GPU);
    }

    Tensor make_uint8_mask(const std::vector<uint8_t>& values) {
        auto tensor = Tensor::empty({values.size()}, Device::CPU, DataType::UInt8);
        std::copy(values.begin(), values.end(), tensor.ptr<uint8_t>());
        return tensor.gpu();
    }

    Tensor make_int32_values(const std::vector<int>& values) {
        return Tensor::from_vector(values, {values.size()}, Device::GPU).to(DataType::Int32);
    }

    Tensor make_float32_values(const std::vector<float>& values, const lfs::core::TensorShape& shape) {
        return Tensor::from_vector(values, shape, Device::GPU).to(DataType::Float32);
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
        auto output = Tensor::empty({selection_values.size()}, Device::GPU, DataType::UInt8);

        std::vector<bool> flags(256, false);
        for (const auto group : locked_groups)
            flags[group] = true;
        const Tensor locked_mask = Tensor::from_vector(flags, {flags.size()}, Device::GPU);

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
            locked_mask,
            add_mode,
            transform_indices_ptr,
            valid_nodes,
            replace_mode);

        return output.cpu().to_vector_uint8();
    }

    std::vector<uint8_t> run_indexed_group_apply(const std::vector<uint8_t>& selection_values,
                                                 const std::vector<int>& visible_indices,
                                                 const std::vector<uint8_t>& existing_values,
                                                 const uint8_t group_id,
                                                 const std::vector<uint8_t>& locked_groups = {},
                                                 const bool add_mode = true,
                                                 const bool replace_mode = false,
                                                 const std::vector<int>* transform_indices = nullptr,
                                                 const std::vector<bool>& valid_nodes = {}) {
        if (selection_values.size() != visible_indices.size()) {
            ADD_FAILURE() << "Selection and visible index sizes differ";
            return {};
        }

        const auto selection = make_bool_mask(selection_values);
        const auto indices = make_int32_values(visible_indices);
        const auto existing = make_uint8_mask(existing_values);
        auto output = Tensor::empty({existing_values.size()}, Device::GPU, DataType::UInt8);

        std::vector<bool> flags(256, false);
        for (const auto group : locked_groups)
            flags[group] = true;
        const Tensor locked_mask = Tensor::from_vector(flags, {flags.size()}, Device::GPU);

        Tensor transform_indices_tensor;
        const Tensor* transform_indices_ptr = nullptr;
        if (transform_indices) {
            transform_indices_tensor = make_int32_values(*transform_indices);
            transform_indices_ptr = &transform_indices_tensor;
        }

        lfs::rendering::apply_selection_group_indexed_tensor_mask(
            selection,
            indices,
            existing,
            output,
            group_id,
            locked_mask,
            add_mode,
            transform_indices_ptr,
            valid_nodes,
            replace_mode);

        return output.cpu().to_vector_uint8();
    }

} // namespace

class SelectionRasterizationOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(Tensor::zeros({1}, Device::GPU).is_valid());
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

TEST_F(SelectionRasterizationOpsTest, IndexedReplacePreservesNonVisibleActiveGroup) {
    const auto result = run_indexed_group_apply(
        {0, 1, 0},
        {1, 3, 5},
        {1, 1, 2, 0, 1, 1},
        1,
        {},
        true,
        true);

    EXPECT_EQ(result, (std::vector<uint8_t>{1, 0, 2, 1, 1, 0}));
}

TEST_F(SelectionRasterizationOpsTest, IndexedReplaceRespectsNodeMask) {
    const std::vector<int> transform_indices = {0, 1, 0, 1};
    const std::vector<bool> valid_nodes = {true, false};

    const auto result = run_indexed_group_apply(
        {0, 0, 1, 0},
        {0, 1, 2, 3},
        {1, 1, 1, 2},
        1,
        {},
        true,
        true,
        &transform_indices,
        valid_nodes);

    EXPECT_EQ(result, (std::vector<uint8_t>{0, 1, 1, 2}));
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
        false,
        nullptr,
        nullptr);

    EXPECT_EQ(selection.cpu().to_vector_bool(), (std::vector<bool>{true, false, true, true}));
}

TEST_F(SelectionRasterizationOpsTest, ScopedCropLeavesOtherNodesAndExistingExclusionsUnchanged) {
    const auto means = make_float32_values({0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0}, {4, 3});
    const auto indices = make_int32_values({0, 0, 1, 1});
    const auto transform = make_float32_values(
        {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, {4, 4});
    const auto min = make_float32_values({-1, -1, -1}, {3});
    const auto max = make_float32_values({1, 1, 1}, {3});
    for (const bool inverse : {false, true}) {
        auto selection = make_bool_mask({1, 1, 1, 0});
        lfs::rendering::filter_selection_by_crop(
            selection, means, &transform, &min, &max, inverse,
            nullptr, nullptr, false, nullptr, &indices, 0);
        EXPECT_EQ(selection.cpu().to_vector_bool(), (std::vector<bool>{!inverse, inverse, true, false}));
    }
}

TEST_F(SelectionRasterizationOpsTest, ScopedEllipsoidIncludesBoundaryAndSupportsInverse) {
    const auto means = make_float32_values({1, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0}, {4, 3});
    const auto indices = make_int32_values({1, 1, 0, 1});
    const auto transform = make_float32_values(
        {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, {4, 4});
    const auto radii = make_float32_values({1, 1, 1}, {3});
    for (const bool inverse : {false, true}) {
        auto selection = make_bool_mask({1, 1, 1, 0});
        lfs::rendering::filter_selection_by_crop(
            selection, means, nullptr, nullptr, nullptr, false,
            &transform, &radii, inverse, nullptr, &indices, 1);
        EXPECT_EQ(selection.cpu().to_vector_bool(), (std::vector<bool>{!inverse, inverse, true, false}));
    }
}

TEST_F(SelectionRasterizationOpsTest, ScopedCropPolygonUsesEvenOddRuleAndDepthAndSupportsInverse) {
    // Concave L in a frame moved by +10 on x: the notch (1..2, 1..2) is outside.
    const auto vertices = make_float32_values({-2, -2, 2, -2, 2, 1, 1, 1, 1, 2, -2, 2}, {6, 2});
    const auto world_to_local = make_float32_values(
        {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -10, 0, 0, 1}, {4, 4});
    // Inside, in the notch, outside the outline, beyond a cap, another node, already unselected.
    const auto means = make_float32_values(
        {10, 0, 0, 11.5f, 1.5f, 0, 13, 0, 0, 10, 0, 1.5f, 13, 0, 0, 10, 0, 0}, {6, 3});
    const auto indices = make_int32_values({0, 0, 0, 0, 1, 0});
    for (const bool inverse : {false, true}) {
        const lfs::rendering::SelectionCropPolygon polygon{
            .transform = &world_to_local,
            .vertices = &vertices,
            .half_depth = 1.0f,
            .inverse = inverse};
        auto selection = make_bool_mask({1, 1, 1, 1, 1, 0});
        lfs::rendering::filter_selection_by_crop(
            selection, means, nullptr, nullptr, nullptr, false,
            nullptr, nullptr, false, nullptr, &indices, 0, &polygon);
        EXPECT_EQ(selection.cpu().to_vector_bool(),
                  (std::vector<bool>{!inverse, inverse, inverse, inverse, true, false}));
    }
}

TEST_F(SelectionRasterizationOpsTest, ScopedCropDefaultsToFirstNodeWithoutIndexStorage) {
    const auto means = make_float32_values({0, 0, 0, 2, 0, 0}, {2, 3});
    const auto transform = make_float32_values(
        {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, {4, 4});
    const auto min = make_float32_values({-1, -1, -1}, {3});
    const auto max = make_float32_values({1, 1, 1}, {3});
    auto selection = make_bool_mask({1, 1});
    lfs::rendering::filter_selection_by_crop(
        selection, means, &transform, &min, &max, false,
        nullptr, nullptr, false, nullptr, nullptr, 0);
    EXPECT_EQ(selection.cpu().to_vector_bool(), (std::vector<bool>{true, false}));
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
        false,
        nullptr,
        nullptr);

    EXPECT_EQ(selection.cpu().to_vector_bool(), (std::vector<bool>{false, true, false, false, true}));
}

TEST_F(SelectionRasterizationOpsTest, CropFilterAppliesModelTransformsBeforeWorldSpaceBounds) {
    auto selection = make_bool_mask({1, 1});
    const auto means = make_float32_values(
        {
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
        },
        {2, 3});
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
    const auto crop_min = make_float32_values({0.5f, -0.5f, -0.5f}, {3});
    const auto crop_max = make_float32_values({1.5f, 0.5f, 0.5f}, {3});
    const auto model_transforms = make_float32_values(
        {
            1.0f,
            0.0f,
            0.0f,
            1.0f,
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

            1.0f,
            0.0f,
            0.0f,
            -1.0f,
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
        {2, 4, 4});
    const auto transform_indices = make_int32_values({0, 1});

    lfs::rendering::filter_selection_by_crop(
        selection,
        means,
        &crop_transform,
        &crop_min,
        &crop_max,
        false,
        nullptr,
        nullptr,
        false,
        &model_transforms,
        &transform_indices);

    EXPECT_EQ(selection.cpu().to_vector_bool(), (std::vector<bool>{true, false}));
}

TEST_F(SelectionRasterizationOpsTest, ProjectsSplatsPastTheCappedLaunch) {
    // The projection launch on master was capped at 4096 blocks of 256 threads.
    // Rectangle, polygon (lasso), and brush selection all read this buffer, so a
    // splat past that prefix still needs a screen position. Dev projects and
    // selects through the tensor backends.
    constexpr size_t kPrefix = 4096u * 256u;
    constexpr size_t n = kPrefix + 2u;
    constexpr std::array<float, 9> kRotation{1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
    constexpr std::array<float, 3> kTranslation{0.0f, 0.0f, 5.0f};

    int ran = 0;
    for (const auto backend : {lfs::core::GpuBackend::CUDA, lfs::core::GpuBackend::Vulkan}) {
        if (!lfs::core::gpu_backend_available(backend))
            continue;
        ++ran;
        SCOPED_TRACE(backend == lfs::core::GpuBackend::CUDA ? "cuda" : "vulkan");
        const lfs::core::GpuBackendScope scope(backend);

        auto means = Tensor::zeros({n, size_t{3}}, Device::GPU);
        means.slice(0, n - 1, n)
            .copy_from(Tensor::from_vector(
                std::vector<float>{1.0f, 0.0f, 0.0f}, {size_t{1}, size_t{3}}, Device::GPU));

        const auto positions = lfs::rendering::project_screen_positions_tensor(
            means, 100, 80, kRotation, kTranslation, 100.0f, 100.0f, 50.0f, 40.0f,
            lfs::core::PointProjectionModel::Pinhole, 1.0f, nullptr, nullptr, {});
        ASSERT_TRUE(positions.is_valid());
        ASSERT_EQ(positions.size(0), n);

        const auto at = [&](const size_t index) {
            const auto row = positions.slice(0, index, index + 1).cpu().contiguous();
            return std::array<float, 2>{row.ptr<float>()[0], row.ptr<float>()[1]};
        };
        EXPECT_FLOAT_EQ(at(0)[0], 50.0f);
        EXPECT_FLOAT_EQ(at(0)[1], 40.0f);
        EXPECT_FLOAT_EQ(at(kPrefix)[0], 50.0f);
        EXPECT_FLOAT_EQ(at(kPrefix)[1], 40.0f);
        EXPECT_FLOAT_EQ(at(n - 1)[0], 70.0f);
        EXPECT_FLOAT_EQ(at(n - 1)[1], 40.0f);

        auto rect = Tensor::zeros({n}, Device::GPU, DataType::Bool);
        auto polygon = Tensor::zeros({n}, Device::GPU, DataType::Bool);
        auto brush = Tensor::zeros({n}, Device::GPU, DataType::Bool);
        lfs::rendering::rect_select_tensor(positions, 40.0f, 30.0f, 80.0f, 50.0f, rect);
        lfs::rendering::polygon_select_tensor(
            positions,
            Tensor::from_vector(std::vector<float>{40, 30, 80, 30, 80, 50, 40, 50}, {size_t{4}, size_t{2}}, Device::GPU),
            polygon);
        lfs::rendering::brush_select_tensor(positions, 70.0f, 40.0f, 2.0f, brush);

        EXPECT_TRUE(rect.get_bool(std::array<size_t, 1>{0}));
        EXPECT_TRUE(rect.get_bool(std::array<size_t, 1>{n - 1}));
        EXPECT_TRUE(polygon.get_bool(std::array<size_t, 1>{0}));
        EXPECT_TRUE(polygon.get_bool(std::array<size_t, 1>{n - 1}));
        EXPECT_FALSE(brush.get_bool(std::array<size_t, 1>{0}));
        EXPECT_TRUE(brush.get_bool(std::array<size_t, 1>{n - 1}));
    }
    if (ran == 0)
        GTEST_SKIP() << "No GPU backend available";
}
