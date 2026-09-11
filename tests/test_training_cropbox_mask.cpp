/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training_cropbox_mask.hpp"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

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

    std::unique_ptr<lfs::core::SplatData> make_crop_test_splat() {
        return make_test_splat({10.0f, 0.0f, 0.0f,
                                10.75f, 0.0f, 0.0f,
                                9.25f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f,
                                12.0f, 0.0f, 0.0f});
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
