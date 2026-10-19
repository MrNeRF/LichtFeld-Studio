// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/scene.hpp"
#include "core/splat_data_transform.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

namespace {
    using namespace lfs::core;
    int content_invalidations = 0;
    int transform_invalidations = 0;

    TEST(SceneTransformInvalidationTest, MatrixChangesPreserveContentButStillPublishFrames) {
        Scene scene;
        const auto parent = scene.addGroup("parent");
        scene.addGroup("child", parent);
        content_invalidations = transform_invalidations = 0;
        scene.setRenderInvalidationCallback([] { ++content_invalidations; });
        scene.setTransformInvalidationCallback([] { ++transform_invalidations; });
        const auto generation = scene.renderGeneration();
        scene.setNodeTransform("parent", glm::translate(glm::mat4(1), glm::vec3(1, 2, 3)));
        EXPECT_GT(scene.renderGeneration(), generation);
        EXPECT_GT(transform_invalidations, 0);
        EXPECT_EQ(content_invalidations, 0);
        scene.notifyMutation(Scene::MutationType::MODEL_CHANGED);
        EXPECT_GT(content_invalidations, 0);
    }

    TEST(SceneTransformInvalidationTest, ExistingCallbackStillReceivesEveryInvalidation) {
        Scene scene;
        content_invalidations = 0;
        scene.setRenderInvalidationCallback([] { ++content_invalidations; });
        scene.invalidateTransformCache();
        EXPECT_EQ(content_invalidations, 1);
        scene.invalidateCache();
        EXPECT_EQ(content_invalidations, 2);
    }

    std::unique_ptr<SplatData> boundsModel(size_t n) {
        std::vector<float> means(n * 3);
        for (size_t i = 0; i < means.size(); ++i)
            means[i] = float(i % 997) / 100.0f;
        return std::make_unique<SplatData>(0,
                                           Tensor::from_vector(means, {n, 3}, Device::CPU),
                                           Tensor::zeros({n, 1, 3}, Device::CPU), Tensor::zeros({n, 0, 3}, Device::CPU),
                                           Tensor::zeros({n, 3}, Device::CPU), Tensor::zeros({n, 4}, Device::CPU),
                                           Tensor::zeros({n, 1}, Device::CPU), 1.0f);
    }
    void expectReferenceBounds(Scene& scene, NodeId id) {
        glm::vec3 actual_min, actual_max, expected_min, expected_max;
        const bool expected = compute_bounds(*scene.getNodeById(id)->model, expected_min, expected_max);
        ASSERT_EQ(scene.getNodeBounds(id, actual_min, actual_max), expected);
        if (expected) {
            EXPECT_EQ(actual_min, expected_min);
            EXPECT_EQ(actual_max, expected_max);
        }
    }
    TEST(SceneBoundsCacheTest, ContentNotificationRefreshesInPlaceMeans) {
        Scene scene;
        const auto id = scene.addSplat("model", boundsModel(20));
        expectReferenceBounds(scene, id);
        scene.getNodeById(id)->model->means_raw().add_(10.0f);
        scene.notifyMutation(Scene::MutationType::MODEL_CHANGED);
        expectReferenceBounds(scene, id);
    }
    TEST(SceneBoundsCacheTest, DeletedMaskRevisionRefreshesBoundsIncludingEmptyResult) {
        Scene scene;
        const auto id = scene.addSplat("model", boundsModel(20));
        expectReferenceBounds(scene, id);
        auto& model = *scene.getNodeById(id)->model;
        auto mask = Tensor::zeros_bool({20}, Device::CPU);
        mask.slice(0, 10, 20).copy_from(Tensor::ones_bool({10}, Device::CPU));
        model.soft_delete(mask);
        expectReferenceBounds(scene, id);
        model.soft_delete(Tensor::ones_bool({20}, Device::CPU));
        expectReferenceBounds(scene, id);
        model.clear_deleted();
        expectReferenceBounds(scene, id);
    }
    TEST(SceneBoundsCacheTest, GeometryPublicationRefreshesTrainingBounds) {
        Scene scene;
        const auto id = scene.addSplat("model", boundsModel(20));
        expectReferenceBounds(scene, id);
        scene.getNodeById(id)->model->means_raw().add_(2.0f);
        scene.invalidateBounds();
        expectReferenceBounds(scene, id);
    }
    TEST(SceneBoundsCacheTest, ReplacingModelRefreshesLocalBounds) {
        Scene scene;
        const auto id = scene.addSplat("model", boundsModel(20));
        expectReferenceBounds(scene, id);
        auto replacement = boundsModel(20);
        replacement->means_raw().add_(20.0f);
        scene.replaceNodeModel("model", std::move(replacement));
        expectReferenceBounds(scene, id);
    }
    TEST(SceneBoundsCacheTest, ParentBoundsFollowRotatedScaledChildren) {
        Scene scene;
        const auto parent = scene.addGroup("parent");
        const auto nested = scene.addGroup("nested", parent);
        const auto id = scene.addSplat("model", boundsModel(20), nested);
        const auto local = glm::translate(glm::mat4(1), glm::vec3(2, -3, 5)) *
                           glm::rotate(glm::mat4(1), 0.7f, glm::vec3(0, 0, 1)) *
                           glm::scale(glm::mat4(1), glm::vec3(2, .5f, 3));
        scene.setNodeTransform("model", local);
        const auto upper = glm::rotate(glm::mat4(1), -.3f, glm::vec3(1, 0, 0)) *
                           glm::scale(glm::mat4(1), glm::vec3(.7f, 2, 1));
        scene.setNodeTransform("nested", upper);
        glm::vec3 lo, hi;
        ASSERT_TRUE(scene.getNodeBounds(id, lo, hi));
        const glm::vec3 center = (lo + hi) * .5f;
        EXPECT_LT(glm::length(scene.getNodeBoundsCenter(parent) - glm::vec3(upper * local * glm::vec4(center, 1))), 1e-5f);
        scene.setNodeTransform("model", glm::translate(local, glm::vec3(1, 2, 3)));
        const auto moved = scene.getNodeTransform("model");
        EXPECT_LT(glm::length(scene.getNodeBoundsCenter(parent) - glm::vec3(upper * moved * glm::vec4(center, 1))), 1e-5f);
        expectReferenceBounds(scene, id);
    }
    class SceneBoundsCacheTimingTest : public ::testing::TestWithParam<int> {};
    TEST_P(SceneBoundsCacheTimingTest, NodeAndGroupTransformsReuseLocalGeometryBounds) {
        Scene scene;
        const auto parent = scene.addGroup("parent");
        const auto group = scene.addGroup("group", parent);
        std::vector<NodeId> ids;
        for (int i = 0; i < GetParam(); ++i)
            ids.push_back(scene.addSplat("node" + std::to_string(i), boundsModel(200000 / GetParam()), group));
        glm::vec3 lo, hi;
        ASSERT_TRUE(scene.getNodeBounds(parent, lo, hi));
        std::array<double, 9> queries{}, reference{};
        for (size_t repetition = 0; repetition < queries.size(); ++repetition) {
            const auto start = std::chrono::steady_clock::now();
            scene.setNodeTransform("group", glm::translate(glm::mat4(1), glm::vec3(float(repetition), 0, 0)));
            ASSERT_TRUE(scene.getNodeBounds(parent, lo, hi));
            const auto middle = std::chrono::steady_clock::now();
            for (auto id : ids)
                ASSERT_TRUE(compute_bounds(*scene.getNodeById(id)->model, lo, hi));
            const auto end = std::chrono::steady_clock::now();
            queries[repetition] = std::chrono::duration<double>(middle - start).count();
            reference[repetition] = std::chrono::duration<double>(end - middle).count();
        }
        std::sort(queries.begin(), queries.end());
        std::sort(reference.begin(), reference.end());
        EXPECT_LT(queries[4], reference[4] * .25);
    }
    INSTANTIATE_TEST_SUITE_P(NodeCounts, SceneBoundsCacheTimingTest, ::testing::Values(1, 2, 10, 50));
} // namespace
