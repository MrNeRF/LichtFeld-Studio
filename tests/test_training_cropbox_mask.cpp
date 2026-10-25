/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/scene_chapter_adapter.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/training_setup.hpp"
#include "training_cropbox_mask.hpp"
#include "visualizer/gui_capabilities.hpp"
#include "visualizer/scene_coordinate_utils.hpp"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <vector>

namespace {

    constexpr size_t kCropTestCount = 5;

    std::unique_ptr<lfs::core::SplatData> make_test_splat(
        std::vector<float> means,
        const lfs::core::Device device = lfs::core::Device::CPU) {
        const size_t count = means.size() / 3;
        std::vector<float> rotations(count * 4, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            rotations[i * 4] = 1.0f;
        }

        return std::make_unique<lfs::core::SplatData>(
            0,
            lfs::core::Tensor::from_vector(means, {count, size_t{3}}, device),
            lfs::core::Tensor::zeros({count, size_t{1}, size_t{3}},
                                     device,
                                     lfs::core::DataType::Float32),
            lfs::core::Tensor::zeros({count, lfs::core::sh_rest_coefficients_for_degree(0), size_t{3}},
                                     device,
                                     lfs::core::DataType::Float32),
            lfs::core::Tensor::zeros({count, size_t{3}},
                                     device,
                                     lfs::core::DataType::Float32),
            lfs::core::Tensor::from_vector(rotations, {count, size_t{4}}, device),
            lfs::core::Tensor::zeros({count, size_t{1}},
                                     device,
                                     lfs::core::DataType::Float32),
            1.0f);
    }

    std::unique_ptr<lfs::core::SplatData> make_crop_test_splat(
        const lfs::core::Device device = lfs::core::Device::CPU) {
        return make_test_splat({10.0f, 0.0f, 0.0f,
                                10.75f, 0.0f, 0.0f,
                                9.25f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f,
                                12.0f, 0.0f, 0.0f},
                               device);
    }

    // The unit box around x = 10 in the model frame, as add_training_model_with_translated_cropbox places it.
    lfs::training::TrainingCropBoxGeometry translated_cropbox_geometry(const bool inverse) {
        const glm::mat4 model_to_cropbox = glm::translate(glm::mat4(1.0f), glm::vec3(-10.0f, 0.0f, 0.0f));
        return {.min = glm::vec3(-1.0f),
                .max = glm::vec3(1.0f),
                .world_to_cropbox = model_to_cropbox,
                .model_to_cropbox = model_to_cropbox,
                .inverse = inverse};
    }

    lfs::core::SplatData* add_training_model_with_translated_cropbox(
        lfs::core::Scene& scene,
        const bool inverse) {
        auto model = make_crop_test_splat();
        auto* const model_ptr = model.get();
        const auto model_id = scene.addSplat("Model", std::move(model));
        scene.setTrainingModelNode("Model");

        const auto cropbox_id = scene.addCropBox("Model_cropbox", model_id);
        lfs::core::CropBoxData cropbox;
        cropbox.min = {-1.0f, -1.0f, -1.0f};
        cropbox.max = {1.0f, 1.0f, 1.0f};
        cropbox.enabled = true;
        cropbox.inverse = inverse;
        scene.setCropBoxData(cropbox_id, cropbox);

        const auto* cropbox_node = scene.getNodeById(cropbox_id);
        scene.setNodeTransform(cropbox_node->name, glm::translate(glm::mat4(1.0f), glm::vec3(10.0f, 0.0f, 0.0f)));

        return model_ptr;
    }

    lfs::core::NodeId cropbox_id_for_model(const lfs::core::Scene& scene) {
        const auto* const model_node = scene.getNode("Model");
        return model_node ? scene.getCropBoxForSplat(model_node->id) : lfs::core::NULL_NODE;
    }

    struct CropBoxSceneSnapshot {
        size_t node_count = 0;
        size_t model_size = 0;
        std::vector<float> means;
        std::optional<std::vector<bool>> deleted;
        lfs::core::CropBoxData cropbox;
        glm::mat4 cropbox_world_transform{1.0f};
    };

    void expect_vec3_eq(const glm::vec3& actual, const glm::vec3& expected) {
        EXPECT_FLOAT_EQ(actual.x, expected.x);
        EXPECT_FLOAT_EQ(actual.y, expected.y);
        EXPECT_FLOAT_EQ(actual.z, expected.z);
    }

    void expect_mat4_eq(const glm::mat4& actual, const glm::mat4& expected) {
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 4; ++r) {
                EXPECT_FLOAT_EQ(actual[c][r], expected[c][r]) << "at column " << c << ", row " << r;
            }
        }
    }

    // A training model with the crop box that --cropbox and --cropbox-rotation give it when training starts.
    lfs::core::SplatData* add_training_model_with_cli_cropbox(
        lfs::core::Scene& scene,
        std::vector<float> means,
        const std::array<float, 6>& box,
        const std::optional<std::array<float, 3>>& rotation) {
        auto model = make_test_splat(std::move(means));
        auto* const model_ptr = model.get();
        scene.addSplat("Model", std::move(model));
        scene.setTrainingModelNode("Model");
        lfs::core::param::TrainingParameters params;
        params.cropbox = box;
        params.cropbox_rotation = rotation;
        const auto applied = lfs::training::applyTrainingCropBox(params, scene);
        EXPECT_TRUE(applied) << (applied ? std::string_view{} : applied.error().user_message());
        return model_ptr;
    }

    CropBoxSceneSnapshot snapshot_cropbox_scene(const lfs::core::Scene& scene,
                                                const lfs::core::SplatData& model) {
        const lfs::core::NodeId cropbox_id = cropbox_id_for_model(scene);
        const auto* const cropbox = scene.getCropBoxData(cropbox_id);
        EXPECT_NE(cropbox, nullptr);

        CropBoxSceneSnapshot snapshot;
        snapshot.node_count = scene.getNodes().size();
        snapshot.model_size = static_cast<size_t>(model.size());
        snapshot.means = model.means().to(lfs::core::Device::CPU).to_vector();
        if (model.has_deleted_mask()) {
            snapshot.deleted = model.deleted().to(lfs::core::Device::CPU).to_vector_bool();
        }
        if (cropbox) {
            snapshot.cropbox = *cropbox;
        }
        snapshot.cropbox_world_transform = scene.getWorldTransform(cropbox_id);
        return snapshot;
    }

    void expect_cropbox_scene_unchanged(const lfs::core::Scene& scene,
                                        const lfs::core::SplatData& model,
                                        const CropBoxSceneSnapshot& before) {
        EXPECT_EQ(scene.getNodes().size(), before.node_count);
        EXPECT_EQ(static_cast<size_t>(model.size()), before.model_size);
        EXPECT_EQ(model.means().to(lfs::core::Device::CPU).to_vector(), before.means);

        if (before.deleted.has_value()) {
            ASSERT_TRUE(model.has_deleted_mask());
            EXPECT_EQ(model.deleted().to(lfs::core::Device::CPU).to_vector_bool(), *before.deleted);
        } else {
            EXPECT_FALSE(model.has_deleted_mask());
        }

        const lfs::core::NodeId cropbox_id = cropbox_id_for_model(scene);
        const auto* const cropbox = scene.getCropBoxData(cropbox_id);
        ASSERT_NE(cropbox, nullptr);
        expect_vec3_eq(cropbox->min, before.cropbox.min);
        expect_vec3_eq(cropbox->max, before.cropbox.max);
        EXPECT_EQ(cropbox->inverse, before.cropbox.inverse);
        EXPECT_EQ(cropbox->enabled, before.cropbox.enabled);
        expect_vec3_eq(cropbox->color, before.cropbox.color);
        EXPECT_FLOAT_EQ(cropbox->line_width, before.cropbox.line_width);
        EXPECT_FLOAT_EQ(cropbox->flash_intensity, before.cropbox.flash_intensity);
        expect_mat4_eq(scene.getWorldTransform(cropbox_id), before.cropbox_world_transform);
    }

} // namespace

TEST(TrainingCropBoxMask, TranslatedCropBoxRemovesOnlyPointsOutsideBox) {
    lfs::core::Scene scene;
    auto* const model = add_training_model_with_translated_cropbox(scene, false);
    const auto before = snapshot_cropbox_scene(scene, *model);

    const auto remove_mask = lfs::training::compute_training_cropbox_remove_mask(scene, *model);
    ASSERT_TRUE(remove_mask.has_value());

    EXPECT_EQ(remove_mask->to(lfs::core::Device::CPU).to_vector_bool(),
              (std::vector<bool>{false, false, false, true, true}));
    expect_cropbox_scene_unchanged(scene, *model, before);
}

TEST(TrainingCropBoxMask, InverseTranslatedCropBoxRemovesOnlyPointsInsideBox) {
    lfs::core::Scene scene;
    auto* const model = add_training_model_with_translated_cropbox(scene, true);
    const auto before = snapshot_cropbox_scene(scene, *model);

    const auto remove_mask = lfs::training::compute_training_cropbox_remove_mask(scene, *model);
    ASSERT_TRUE(remove_mask.has_value());

    EXPECT_EQ(remove_mask->to(lfs::core::Device::CPU).to_vector_bool(),
              (std::vector<bool>{true, true, true, false, false}));
    expect_cropbox_scene_unchanged(scene, *model, before);
}

TEST(TrainingCropBoxMask, SoftDeletedSplatsAreNotReturnedForRemovalAgain) {
    lfs::core::Scene scene;
    auto* const model = add_training_model_with_translated_cropbox(scene, false);
    model->deleted() = lfs::core::Tensor::from_vector(
        std::vector<bool>{false, false, false, true, false},
        {kCropTestCount},
        lfs::core::Device::CPU);
    const auto before = snapshot_cropbox_scene(scene, *model);

    const auto remove_mask = lfs::training::compute_training_cropbox_remove_mask(scene, *model);
    ASSERT_TRUE(remove_mask.has_value());

    EXPECT_EQ(remove_mask->to(lfs::core::Device::CPU).to_vector_bool(),
              (std::vector<bool>{false, false, false, false, true}));
    expect_cropbox_scene_unchanged(scene, *model, before);
}

TEST(TrainingCropBoxMask, MissingTrainingModelReturnsNulloptWithoutSceneMutation) {
    lfs::core::Scene scene;
    auto model = make_crop_test_splat();
    const size_t node_count_before = scene.getNodes().size();
    const size_t model_size_before = static_cast<size_t>(model->size());
    const auto means_before = model->means().to(lfs::core::Device::CPU).to_vector();

    const auto remove_mask = lfs::training::compute_training_cropbox_remove_mask(scene, *model);

    EXPECT_FALSE(remove_mask.has_value());
    EXPECT_EQ(scene.getNodes().size(), node_count_before);
    EXPECT_EQ(static_cast<size_t>(model->size()), model_size_before);
    EXPECT_EQ(model->means().to(lfs::core::Device::CPU).to_vector(), means_before);
}

TEST(TrainingCropBoxMask, MissingCropBoxReturnsNulloptWithoutCreatingCropBox) {
    lfs::core::Scene scene;
    auto model = make_crop_test_splat();
    auto* const model_ptr = model.get();
    const auto model_id = scene.addSplat("Model", std::move(model));
    scene.setTrainingModelNode("Model");
    const size_t node_count_before = scene.getNodes().size();
    const size_t model_size_before = static_cast<size_t>(model_ptr->size());
    const auto means_before = model_ptr->means().to(lfs::core::Device::CPU).to_vector();

    const auto remove_mask = lfs::training::compute_training_cropbox_remove_mask(scene, *model_ptr);

    EXPECT_FALSE(remove_mask.has_value());
    EXPECT_EQ(scene.getCropBoxForSplat(model_id), lfs::core::NULL_NODE);
    EXPECT_EQ(scene.getNodes().size(), node_count_before);
    EXPECT_EQ(static_cast<size_t>(model_ptr->size()), model_size_before);
    EXPECT_EQ(model_ptr->means().to(lfs::core::Device::CPU).to_vector(), means_before);
}

TEST(TrainingCropBoxMask, DisabledCropBoxReturnsNulloptWithoutMutation) {
    lfs::core::Scene scene;
    auto* const model = add_training_model_with_translated_cropbox(scene, false);
    const lfs::core::NodeId cropbox_id = cropbox_id_for_model(scene);
    auto cropbox = *scene.getCropBoxData(cropbox_id);
    cropbox.enabled = false;
    scene.setCropBoxData(cropbox_id, cropbox);
    const auto before = snapshot_cropbox_scene(scene, *model);

    const auto remove_mask = lfs::training::compute_training_cropbox_remove_mask(scene, *model);

    EXPECT_FALSE(remove_mask.has_value());
    expect_cropbox_scene_unchanged(scene, *model, before);
}

TEST(TrainingCropBoxMask, LossGeometryIsInactiveWithoutCropOrAtUnitWeight) {
    lfs::core::Scene no_crop_scene;
    EXPECT_FALSE(
        lfs::training::resolve_training_cropbox_loss_geom(no_crop_scene, 0.1f)
            .has_value());

    lfs::core::Scene active_scene;
    add_training_model_with_translated_cropbox(active_scene, false);
    EXPECT_TRUE(
        lfs::training::resolve_training_cropbox_loss_geom(active_scene, 0.1f)
            .has_value());
    EXPECT_FALSE(
        lfs::training::resolve_training_cropbox_loss_geom(active_scene, 1.0f)
            .has_value());
}

TEST(TrainingCropBoxMask, WiderMeansUseFirstThreeColumnsForCropMask) {
    const auto means = lfs::core::Tensor::from_vector(
        std::vector<float>{
            0.0f, 0.0f, 0.0f, 1000.0f,
            2.0f, 0.0f, 0.0f, -1000.0f,
            0.0f, -1.0f, 1.0f, 500.0f},
        {size_t{3}, size_t{4}},
        lfs::core::Device::CPU);
    const auto means_before = means.to(lfs::core::Device::CPU).to_vector();

    const auto remove_mask = lfs::training::compute_cropbox_remove_mask(
        means,
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        glm::mat4(1.0f),
        false);

    ASSERT_TRUE(remove_mask.has_value());
    EXPECT_EQ(remove_mask->to(lfs::core::Device::CPU).to_vector_bool(),
              (std::vector<bool>{false, true, false}));
    EXPECT_EQ(means.to(lfs::core::Device::CPU).to_vector(), means_before);
}

TEST(TrainingCropBoxMask, RotatedCropBoxUsesPointToBoxTransform) {
    const auto means = lfs::core::Tensor::from_vector(
        std::vector<float>{
            0.0f, 1.5f, 0.0f,
            1.0f, 0.0f, 0.0f,
            0.0f, -1.75f, 0.0f,
            0.75f, 2.5f, 0.0f},
        {size_t{4}, size_t{3}},
        lfs::core::Device::CPU);
    const glm::mat4 points_to_cropbox =
        glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    const auto remove_mask = lfs::training::compute_cropbox_remove_mask(
        means,
        {-2.0f, -0.5f, -1.0f},
        {2.0f, 0.5f, 1.0f},
        points_to_cropbox,
        false);

    ASSERT_TRUE(remove_mask.has_value());
    EXPECT_EQ(remove_mask->to(lfs::core::Device::CPU).to_vector_bool(),
              (std::vector<bool>{false, true, false, true}));
}

TEST(TrainingCropBoxMask, NonUniformlyScaledCropBoxUsesFullAffineTransform) {
    const auto means = lfs::core::Tensor::from_vector(
        std::vector<float>{
            1.5f, 0.0f, 0.0f,
            2.1f, 0.0f, 0.0f,
            0.0f, 0.4f, 0.0f,
            0.0f, 0.6f, 0.0f,
            0.0f, 0.0f, 0.8f},
        {size_t{5}, size_t{3}},
        lfs::core::Device::CPU);
    const glm::mat4 cropbox_to_points =
        glm::scale(glm::mat4(1.0f), glm::vec3(2.0f, 0.5f, 1.0f));

    const auto remove_mask = lfs::training::compute_cropbox_remove_mask(
        means,
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        glm::inverse(cropbox_to_points),
        false);

    ASSERT_TRUE(remove_mask.has_value());
    EXPECT_EQ(remove_mask->to(lfs::core::Device::CPU).to_vector_bool(),
              (std::vector<bool>{false, true, false, true, false}));
}

TEST(TrainingCropBoxMask, TrainingModelTransformIsComposedBeforeChildCropBox) {
    lfs::core::Scene scene;
    auto model = make_test_splat({2.0f, 0.0f, 0.0f,
                                  2.75f, 0.25f, 0.0f,
                                  0.0f, 0.0f, 0.0f,
                                  2.0f, 1.5f, 0.0f});
    auto* const model_ptr = model.get();
    const auto model_id = scene.addSplat("Model", std::move(model));
    scene.setTrainingModelNode("Model");

    const glm::mat4 model_transform = glm::rotate(
        glm::translate(glm::mat4(1.0f), glm::vec3(10.0f, -4.0f, 2.0f)),
        glm::radians(90.0f),
        glm::vec3(0.0f, 0.0f, 1.0f));
    scene.setNodeTransform("Model", model_transform);

    const auto cropbox_id = scene.addCropBox("Model_cropbox", model_id);
    lfs::core::CropBoxData cropbox;
    cropbox.min = {-1.0f, -1.0f, -1.0f};
    cropbox.max = {1.0f, 1.0f, 1.0f};
    cropbox.enabled = true;
    scene.setCropBoxData(cropbox_id, cropbox);
    scene.setNodeTransform(
        "Model_cropbox",
        glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f)));

    const auto before = snapshot_cropbox_scene(scene, *model_ptr);
    const auto remove_mask =
        lfs::training::compute_training_cropbox_remove_mask(scene, *model_ptr);

    ASSERT_TRUE(remove_mask.has_value());
    EXPECT_EQ(remove_mask->to(lfs::core::Device::CPU).to_vector_bool(),
              (std::vector<bool>{false, false, true, true}));
    expect_cropbox_scene_unchanged(scene, *model_ptr, before);
}

TEST(TrainingCropBoxMask, PureMaskMatchesOnCpuAndCuda) {
    const auto cpu_means = lfs::core::Tensor::from_vector(
        std::vector<float>{
            1.0f, 0.0f, 0.0f,
            2.5f, 0.0f, 0.0f,
            4.0f, 0.0f, 0.0f,
            1.0f, -1.0f, 0.0f},
        {size_t{4}, size_t{3}},
        lfs::core::Device::CPU);
    const auto cpu_deleted = lfs::core::Tensor::from_vector(
        std::vector<bool>{false, false, true, false},
        {size_t{4}},
        lfs::core::Device::CPU);
    const glm::mat4 points_to_cropbox =
        glm::translate(glm::mat4(1.0f), glm::vec3(-1.0f, 0.0f, 0.0f));

    const auto cpu_mask = lfs::training::compute_cropbox_remove_mask(
        cpu_means,
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        points_to_cropbox,
        false,
        &cpu_deleted);

    const auto cuda_means = cpu_means.to(lfs::core::Device::CUDA);
    const auto cuda_deleted = cpu_deleted.to(lfs::core::Device::CUDA);
    const auto cuda_mask = lfs::training::compute_cropbox_remove_mask(
        cuda_means,
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        points_to_cropbox,
        false,
        &cuda_deleted);

    ASSERT_TRUE(cpu_mask.has_value());
    ASSERT_TRUE(cuda_mask.has_value());
    const std::vector<bool> expected{false, true, false, false};
    EXPECT_EQ(cpu_mask->to(lfs::core::Device::CPU).to_vector_bool(), expected);
    EXPECT_EQ(cuda_mask->to(lfs::core::Device::CPU).to_vector_bool(), expected);
}

TEST(TrainingCropBoxMask, DampedOptimizerStepUsesComputedRejectedMask) {
    auto model = make_test_splat(
        {0.0f, 0.0f, 0.0f,
         2.0f, 0.0f, 0.0f},
        lfs::core::Device::CUDA);
    const auto remove_mask = lfs::training::compute_cropbox_remove_mask(
        model->means(),
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        glm::mat4(1.0f),
        false);
    ASSERT_TRUE(remove_mask.has_value());
    EXPECT_EQ(remove_mask->to(lfs::core::Device::CPU).to_vector_bool(),
              (std::vector<bool>{false, true}));

    lfs::training::AdamConfig config{
        .lr = 0.01f,
        .beta1 = 0.0,
        .beta2 = 0.0,
        .eps = 1e-15};
    lfs::training::AdamOptimizer optimizer(*model, config);
    optimizer.allocate_gradients();
    optimizer.zero_grad(0);
    optimizer.get_grad(lfs::training::ParamType::Means).fill_(1.0f);
    optimizer.set_crop_damping_mask(std::move(*remove_mask));
    optimizer.set_cropbox_lr_scale(0.1f);

    const auto before = model->means().clone();
    optimizer.step(1);
    const auto delta = (model->means() - before).abs().to(lfs::core::Device::CPU).to_vector();
    ASSERT_EQ(delta.size(), 6u);
    EXPECT_GT(delta[0], 0.0f);
    EXPECT_NEAR(delta[3], delta[0] * 0.1f, 1e-7f);
}

TEST(TrainingCropBoxMask, EmptyMeansReturnNulloptWithoutMutation) {
    const auto means = lfs::core::Tensor::zeros({size_t{0}, size_t{3}},
                                                lfs::core::Device::CPU,
                                                lfs::core::DataType::Float32);
    const auto means_before = means.to(lfs::core::Device::CPU).to_vector();

    const auto remove_mask = lfs::training::compute_cropbox_remove_mask(
        means,
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        glm::mat4(1.0f),
        false);

    EXPECT_FALSE(remove_mask.has_value());
    EXPECT_EQ(means.to(lfs::core::Device::CPU).to_vector(), means_before);
}

// Catches using model_to_cropbox instead of its inverse or a swapped corner bit order.
TEST(TrainingCropBoxMask, ModelCornersUndoTheCropBoxTransform) {
    const glm::mat4 cropbox_to_model =
        glm::translate(glm::mat4(1.0f), glm::vec3(5.0f, 0.0f, 0.0f)) *
        glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    const lfs::training::TrainingCropBoxGeometry geometry{
        .min = {-1.0f, -2.0f, -3.0f},
        .max = {1.0f, 2.0f, 3.0f},
        .world_to_cropbox = glm::inverse(cropbox_to_model),
        .model_to_cropbox = glm::inverse(cropbox_to_model)};
    const auto corners = lfs::training::training_cropbox_model_corners(geometry);
    const std::array<std::array<float, 3>, 3> expected{{
        {7.0f, -1.0f, -3.0f}, // corner 0: (-1, -2, -3) rotated to (2, -1), shifted by 5
        {7.0f, 1.0f, -3.0f},  // corner 1: (1, -2, -3) -> (2, 1)
        {3.0f, -1.0f, 3.0f},  // corner 6: (-1, 2, 3) -> (-2, -1)
    }};
    for (const auto& [index, point] : {std::pair{0, expected[0]}, std::pair{1, expected[1]}, std::pair{6, expected[2]}}) {
        for (size_t axis = 0; axis < 3; ++axis)
            EXPECT_NEAR(corners[index][axis], point[axis], 1e-5f) << "corner " << index << " axis " << axis;
    }
}

// Catches an export crop that keeps the rows it should remove or changes the model it copies.
TEST(TrainingCropBoxMask, ExportCopyHoldsOnlyTheCroppedRowsAndLeavesTheModelWhole) {
    for (const bool inverse : {false, true}) {
        SCOPED_TRACE(inverse ? "inverse" : "normal");
        auto model = make_crop_test_splat(lfs::core::Device::CUDA);
        const auto means_before = model->means().to(lfs::core::Device::CPU).to_vector();

        const auto copy = lfs::training::crop_copy_to_training_cropbox(*model, translated_cropbox_geometry(inverse));

        ASSERT_TRUE(copy.has_value());
        EXPECT_EQ(copy->means().to(lfs::core::Device::CPU).to_vector(),
                  inverse ? (std::vector<float>{0.0f, 0.0f, 0.0f, 12.0f, 0.0f, 0.0f})
                          : (std::vector<float>{10.0f, 0.0f, 0.0f, 10.75f, 0.0f, 0.0f, 9.25f, 0.0f, 0.0f}));
        EXPECT_EQ(static_cast<size_t>(model->size()), kCropTestCount);
        EXPECT_EQ(model->means().to(lfs::core::Device::CPU).to_vector(), means_before);
    }
}

// Catches soft-deleted rows coming back in a cropped export.
TEST(TrainingCropBoxMask, ExportCopyLeavesOutSoftDeletedRows) {
    auto model = make_crop_test_splat(lfs::core::Device::CUDA);
    model->deleted() = lfs::core::Tensor::from_vector(
        std::vector<bool>{false, true, false, false, false},
        {kCropTestCount},
        lfs::core::Device::CUDA);

    const auto copy = lfs::training::crop_copy_to_training_cropbox(*model, translated_cropbox_geometry(false));

    ASSERT_TRUE(copy.has_value());
    EXPECT_EQ(copy->means().to(lfs::core::Device::CPU).to_vector(),
              (std::vector<float>{10.0f, 0.0f, 0.0f, 9.25f, 0.0f, 0.0f}));
}

// Catches an export copy made although no live row lies outside the box, as after a completed run.
TEST(TrainingCropBoxMask, ExportCopyIsSkippedWhenNoLiveRowLiesOutsideTheBox) {
    auto inside = make_test_splat({10.0f, 0.0f, 0.0f, 10.5f, 0.5f, -0.5f}, lfs::core::Device::CUDA);
    EXPECT_FALSE(lfs::training::crop_copy_to_training_cropbox(*inside, translated_cropbox_geometry(false)));

    // MCMC crops by soft-deleting the rows outside the box.
    auto soft_cropped = make_crop_test_splat(lfs::core::Device::CUDA);
    soft_cropped->deleted() = lfs::core::Tensor::from_vector(
        std::vector<bool>{false, false, false, true, true},
        {kCropTestCount},
        lfs::core::Device::CUDA);
    EXPECT_FALSE(lfs::training::crop_copy_to_training_cropbox(*soft_cropped, translated_cropbox_geometry(false)));
}

// Catches --cropbox without --cropbox-rotation storing anything but the corners on an unrotated box.
TEST(TrainingCropBoxMask, CliCropBoxWithoutRotationKeepsTheCornersUnrotated) {
    lfs::core::Scene scene;
    add_training_model_with_cli_cropbox(scene, {0.0f, 0.0f, 0.0f}, {-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f},
                                        std::nullopt);

    const lfs::core::NodeId cropbox_id = cropbox_id_for_model(scene);
    const auto* const cropbox = scene.getCropBoxData(cropbox_id);
    ASSERT_NE(cropbox, nullptr);
    expect_vec3_eq(cropbox->min, {-1.0f, -2.0f, -3.0f});
    expect_vec3_eq(cropbox->max, {4.0f, 5.0f, 6.0f});
    EXPECT_TRUE(cropbox->enabled);
    EXPECT_FALSE(cropbox->inverse);
    expect_mat4_eq(scene.getNodeTransform(cropbox_id), glm::mat4(1.0f));
}

// Catches the rotation being ignored, turning the box about a corner or the origin instead of its centre,
// turning it the other way, or about the viewer's axes: 30 degrees about the dataset's Z turns the long X edge of
// the box toward the dataset's (cos 30, sin 30, 0).
TEST(TrainingCropBoxMask, CliCropBoxRotationTurnsTheBoxAboutItsCentre) {
    const float c = std::cos(glm::radians(30.0f));
    const float s = std::sin(glm::radians(30.0f));
    lfs::core::Scene scene;
    auto* const model = add_training_model_with_cli_cropbox(
        scene,
        {10.0f + 2.9f * c, 2.9f * s, 0.0f,  // along the turned long edge, outside the unturned box
         12.9f, 0.45f, 0.45f,               // near a corner of the unturned box
         10.0f + 2.9f * c, -2.9f * s, 0.0f, // turned the other way
         10.0f, 0.0f, 0.0f},                // centre
        {7.0f, -0.5f, -0.5f, 13.0f, 0.5f, 0.5f},
        std::array<float, 3>{0.0f, 0.0f, 30.0f});

    const auto remove_mask = lfs::training::compute_training_cropbox_remove_mask(scene, *model);
    ASSERT_TRUE(remove_mask.has_value());
    EXPECT_EQ(remove_mask->to(lfs::core::Device::CPU).to_vector_bool(),
              (std::vector<bool>{false, true, true, false}));

    const auto geometry = lfs::training::resolve_training_cropbox_geom(scene);
    ASSERT_TRUE(geometry.has_value());
    const auto means = model->means().to(lfs::core::Device::CPU).to_vector();
    std::vector<bool> inside;
    for (size_t i = 0; i < means.size(); i += 3) {
        const glm::vec3 p(geometry->world_to_cropbox * glm::vec4(means[i], means[i + 1], means[i + 2], 1.0f));
        inside.push_back(glm::all(glm::greaterThanEqual(p, geometry->min)) &&
                         glm::all(glm::lessThanEqual(p, geometry->max)));
    }
    EXPECT_EQ(inside, (std::vector<bool>{true, false, false, true}));

    // The evaluation mask box spans the turned box around the same centre.
    glm::vec3 reach{0.0f};
    for (const auto& corner : lfs::training::training_cropbox_model_corners(*geometry))
        reach = glm::max(reach, glm::abs(glm::vec3(corner[0], corner[1], corner[2]) - glm::vec3(10.0f, 0.0f, 0.0f)));
    EXPECT_NEAR(reach.x, 3.0f * c + 0.5f * s, 1e-5f);
    EXPECT_NEAR(reach.y, 3.0f * s + 0.5f * c, 1e-5f);
    EXPECT_NEAR(reach.z, 0.5f, 1e-5f);
}

// Catches a saved project that loses or alters the turned crop box, and a node the transform panel would not show
// as the same turn: the panel flips Y and Z into the viewer's axes, so it shows rx+180, ry, rz.
TEST(TrainingCropBoxMask, CliCropBoxRotationRoundTripsThroughTheProject) {
    using lfs::io::project::PayloadBinding;
    lfs::core::Scene scene;
    add_training_model_with_cli_cropbox(scene, {10.0f, 1.5f, 2.5f}, {7.0f, 1.0f, 2.0f, 13.0f, 2.0f, 3.0f},
                                        std::array<float, 3>{25.0f, -40.0f, 70.0f});
    const lfs::core::NodeId cropbox_id = cropbox_id_for_model(scene);
    const glm::mat4 transform = scene.getNodeTransform(cropbox_id);
    const glm::mat4 expected =
        glm::translate(glm::mat4(1.0f), glm::vec3(10.0f, 1.5f, 2.5f)) *
        glm::rotate(glm::mat4(1.0f), glm::radians(25.0f), glm::vec3(1.0f, 0.0f, 0.0f)) *
        glm::rotate(glm::mat4(1.0f), glm::radians(-40.0f), glm::vec3(0.0f, 1.0f, 0.0f)) *
        glm::rotate(glm::mat4(1.0f), glm::radians(70.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row)
            EXPECT_NEAR(transform[col][row], expected[col][row], 1e-6f) << "at column " << col << ", row " << row;
    }

    const auto* const model_node = scene.getNode("Model");
    ASSERT_NE(model_node, nullptr);
    const lfs::io::project::ScenePayloadBindings bindings{
        {model_node->uuid, PayloadBinding{.fourcc = "CKPT",
                                          .instance_uuid = *lfs::core::Uuid::from_string(
                                              "6b1f0c52-2d3e-4f61-9a7b-0c1d2e3f4a5b"),
                                          .source_kind = "checkpoint"}}};
    const auto captured = lfs::io::project::capture_scene_graph(scene, bindings);
    ASSERT_TRUE(captured) << lfs::format_for_developer(captured.error());
    const auto bytes = captured->to_bytes();
    const auto decoded = lfs::io::project::SceneGraphChapter::from_bytes(bytes);
    ASSERT_TRUE(decoded) << lfs::format_for_developer(decoded.error());
    lfs::core::Scene restored;
    const lfs::io::project::ScenePayloadResolver resolver{
        .splat = [](const PayloadBinding&) -> lfs::Result<std::unique_ptr<lfs::core::SplatData>> {
            return make_test_splat({10.0f, 1.5f, 2.5f});
        }};
    const auto hydrated = lfs::io::project::hydrate_scene_graph(*decoded, restored, resolver);
    ASSERT_TRUE(hydrated) << lfs::format_for_developer(hydrated.error());

    const lfs::core::NodeId restored_id = cropbox_id_for_model(restored);
    const auto* const restored_cropbox = restored.getCropBoxData(restored_id);
    ASSERT_NE(restored_cropbox, nullptr);
    const glm::mat4 restored_transform = restored.getNodeTransform(restored_id);
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row)
            EXPECT_EQ(restored_transform[col][row], transform[col][row]) << "at column " << col << ", row " << row;
    }
    expect_vec3_eq(restored_cropbox->min, {-3.0f, -0.5f, -0.5f});
    expect_vec3_eq(restored_cropbox->max, {3.0f, 0.5f, 0.5f});
    EXPECT_TRUE(restored_cropbox->enabled);
    EXPECT_FALSE(restored_cropbox->inverse);

    const glm::mat4 local = lfs::rendering::dataWorldTransformToVisualizerWorld(transform);
    const glm::mat4 world = lfs::vis::scene_coords::nodeVisualizerWorldTransform(restored, restored_id);
    for (const glm::mat4& panel : {local, world}) {
        const auto components = lfs::vis::cap::decomposeTransform(panel);
        const glm::mat3 shown = glm::mat3(
            glm::rotate(glm::mat4(1.0f), components.rotation.x, glm::vec3(1.0f, 0.0f, 0.0f)) *
            glm::rotate(glm::mat4(1.0f), components.rotation.y, glm::vec3(0.0f, 1.0f, 0.0f)) *
            glm::rotate(glm::mat4(1.0f), components.rotation.z, glm::vec3(0.0f, 0.0f, 1.0f)));
        const glm::mat3 flipped = lfs::rendering::DATA_TO_VISUALIZER_WORLD_AXES * glm::mat3(expected);
        for (int col = 0; col < 3; ++col) {
            for (int row = 0; row < 3; ++row)
                EXPECT_NEAR(shown[col][row], flipped[col][row], 1e-5f) << "at column " << col << ", row " << row;
        }
        const glm::vec3 degrees = glm::degrees(components.rotation);
        EXPECT_NEAR(degrees.x, -155.0f, 1e-3f);
        EXPECT_NEAR(degrees.y, -40.0f, 1e-3f);
        EXPECT_NEAR(degrees.z, 70.0f, 1e-3f);
        EXPECT_NEAR(components.translation.x, 10.0f, 1e-5f);
        EXPECT_NEAR(components.translation.y, -1.5f, 1e-5f);
        EXPECT_NEAR(components.translation.z, -2.5f, 1e-5f);
    }
}

namespace {

    // An L outline: the square [-1, 1]^2 without its (0, 1) x (0, 1) quarter, the notch.
    const std::vector<glm::vec2> kLOutline{{-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 1.0f}};

    // A training model with an L crop polygon over |z| <= 0.5, moved to x = 10 by its node transform.
    lfs::core::SplatData* add_training_model_with_l_crop_polygon(lfs::core::Scene& scene, const bool inverse) {
        auto model = make_test_splat({10.5f, -0.5f, 0.0f, // bottom arm
                                      10.5f, 0.5f, 0.0f,  // the notch
                                      9.5f, 0.5f, 0.25f,  // upper arm
                                      9.5f, -0.5f, 0.75f, // above the slab
                                      12.0f, 0.0f, 0.0f}, // beside the outline
                                     lfs::core::Device::CUDA);
        auto* const model_ptr = model.get();
        const auto model_id = scene.addSplat("Model", std::move(model));
        scene.setTrainingModelNode("Model");
        EXPECT_NE(scene.addCropPolygon("Model_crop_polygon", model_id,
                                       lfs::core::CropPolygonData{.vertices = kLOutline, .half_depth = 0.5f, .inverse = inverse},
                                       glm::translate(glm::mat4(1.0f), glm::vec3(10.0f, 0.0f, 0.0f))),
                  lfs::core::NULL_NODE);
        return model_ptr;
    }

} // namespace

// Catches a crop polygon on the training model that is ignored, or cropped by its bounding box or without its slab.
TEST(TrainingCropBoxMask, CropPolygonRemovesRowsOutsideTheConcaveOutlineAndSlab) {
    for (const bool inverse : {false, true}) {
        SCOPED_TRACE(inverse ? "inverse" : "normal");
        lfs::core::Scene scene;
        auto* const model = add_training_model_with_l_crop_polygon(scene, inverse);

        const auto geometry = lfs::training::resolve_training_cropbox_geom(scene);
        ASSERT_TRUE(geometry.has_value());
        EXPECT_TRUE(geometry->is_polygon());
        expect_vec3_eq(geometry->min, {-1.0f, -1.0f, -0.5f});
        expect_vec3_eq(geometry->max, {1.0f, 1.0f, 0.5f});
        EXPECT_FALSE(lfs::training::training_crop_region_contains(*geometry, {0.5f, 0.5f, 0.0f}));
        EXPECT_TRUE(lfs::training::training_crop_region_contains(*geometry, {-0.5f, 0.5f, 0.0f}));

        const auto remove_mask = lfs::training::compute_training_cropbox_remove_mask(scene, *model);
        ASSERT_TRUE(remove_mask.has_value());
        const std::vector<bool> outside{false, true, false, true, true};
        std::vector<bool> expected = outside;
        if (inverse)
            expected.flip();
        EXPECT_EQ(remove_mask->to(lfs::core::Device::CPU).to_vector_bool(), expected);
    }

    lfs::core::Scene disabled;
    add_training_model_with_l_crop_polygon(disabled, false);
    const auto* const model_node = disabled.getNode("Model");
    auto data = *disabled.getCropPolygonData(disabled.getCropPolygonForSplat(model_node->id));
    data.enabled = false;
    disabled.setCropPolygonData(disabled.getCropPolygonForSplat(model_node->id), data);
    EXPECT_FALSE(lfs::training::resolve_training_cropbox_geom(disabled).has_value());
}

// Catches an export or crop at end that keeps the notch of a concave crop polygon or changes the model it copies.
TEST(TrainingCropBoxMask, ExportCopyAndCropMaskFollowTheCropPolygon) {
    lfs::core::Scene scene;
    auto* const model = add_training_model_with_l_crop_polygon(scene, false);
    const auto geometry = lfs::training::resolve_training_cropbox_geom(scene);
    ASSERT_TRUE(geometry.has_value());
    const auto means_before = model->means().to(lfs::core::Device::CPU).to_vector();

    const auto crop = lfs::training::training_cropbox_crop_mask(*model, *geometry);
    ASSERT_TRUE(crop.has_value());
    EXPECT_EQ(crop->to(lfs::core::Device::CPU).to_vector_bool(), (std::vector<bool>{false, true, false, true, true}));

    const auto copy = lfs::training::crop_copy_to_training_cropbox(*model, *geometry);
    ASSERT_TRUE(copy.has_value());
    EXPECT_EQ(copy->means().to(lfs::core::Device::CPU).to_vector(),
              (std::vector<float>{10.5f, -0.5f, 0.0f, 9.5f, 0.5f, 0.25f}));
    EXPECT_EQ(model->means().to(lfs::core::Device::CPU).to_vector(), means_before);
}
