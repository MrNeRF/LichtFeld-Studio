/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/apriltag_scene_scale.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/point_cloud.hpp"
#include "core/services.hpp"
#include "operation/undo_history.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/gui_capabilities.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

namespace {
    using namespace lfs;

    std::shared_ptr<core::Camera> sourceCamera() {
        auto camera = std::make_shared<core::Camera>();
        core::UndistortParams params{};
        params.src_fx = params.src_fy = 1000;
        params.dst_fx = params.dst_fy = 800;
        params.src_cx = params.dst_cx = 640;
        params.src_cy = params.dst_cy = 480;
        params.src_width = params.dst_width = 1280;
        params.src_height = params.dst_height = 960;
        params.model_type = core::CameraModelType::PINHOLE;
        params.distortion[0] = 0.1f;
        params.num_distortion = 1;
        camera->adopt_undistortion(params);
        return camera;
    }

    TEST(AprilTagSceneScale, SourceCalibrationSurvivesFocalChangeFrom1000To800) {
        auto camera = sourceCamera();
        const auto before = core::buildCameraObservationView(*camera);
        camera->prepare_undistortion();
        ASSERT_FLOAT_EQ(camera->focal_x(), 800);
        const auto after = core::buildCameraObservationView(*camera);
        EXPECT_DOUBLE_EQ(after.intrinsics.fx, 1000);
        EXPECT_EQ(after.intrinsics.distortion, before.intrinsics.distortion);
        const glm::dvec3 point(1.3, 0, 5);
        const auto expected = geometry::apriltag::projectWorldPointToDetectedPixel(before, point);
        const auto actual = geometry::apriltag::projectWorldPointToDetectedPixel(after, point);
        ASSERT_TRUE(expected && actual);
        EXPECT_NEAR(actual->x, expected->x, 1e-9);
        auto incorrect = after;
        incorrect.intrinsics.fx = 800;
        EXPECT_GT(glm::length(*geometry::apriltag::projectWorldPointToDetectedPixel(incorrect, point) - *actual), 52.0);
    }

    TEST(AprilTagSceneScale, SnapshotOwnsCalibrationAndRejectsChangedScene) {
        core::Scene scene;
        const auto dataset = scene.addDataset("dataset");
        auto camera = sourceCamera();
        scene.addCamera("camera", dataset, camera);
        const auto snapshot = core::snapshotAprilTagScene(scene);
        ASSERT_EQ(snapshot.cameras.size(), 1u);
        EXPECT_TRUE(snapshot.matches(scene));
        camera->prepare_undistortion();
        EXPECT_DOUBLE_EQ(snapshot.cameras[0].view.intrinsics.fx, 1000);
        scene.setNodeTransform(dataset, glm::scale(glm::mat4(1), glm::vec3(2)));
        EXPECT_FALSE(snapshot.matches(scene));
        scene.clear();
        scene.addDataset("dataset");
        EXPECT_FALSE(snapshot.matches(scene));
        // Destroying the source scene does not invalidate worker inputs.
        EXPECT_DOUBLE_EQ(snapshot.cameras[0].view.intrinsics.fx, 1000);
    }

    TEST(AprilTagSceneScale, CancellationStopsBeforeImageRead) {
        core::AprilTagCameraInput input;
        input.image_path = "unused-image.png";
        int calls = 0;
        const auto result = core::detectAprilTagsInSnapshot(std::span(&input, 1), {}, 0,
                                                            [&](std::size_t done, std::size_t total, const std::string&) {
                                                                EXPECT_EQ(done, 0u);
                                                                EXPECT_EQ(total, 1u);
                                                                ++calls;
                                                                return false;
                                                            });
        EXPECT_EQ(calls, 1);
        EXPECT_TRUE(result.empty());
    }

    TEST(AprilTagSceneScale, ParallelDetectionProgressIsMonotonic) {
        const std::vector<core::AprilTagCameraInput> inputs(64);
        std::size_t last = 0;
        const auto views = core::detectAprilTagsInSnapshot(inputs, {}, 0,
                                                           [&](const std::size_t done, const std::size_t total, const std::string&) {
                                                               EXPECT_GE(done, last);
                                                               EXPECT_EQ(total, inputs.size());
                                                               last = done;
                                                               return true;
                                                           });
        EXPECT_EQ(last, inputs.size());
        EXPECT_EQ(views.size(), inputs.size());
    }

    class AprilTagSceneApply : public ::testing::Test {
    protected:
        void SetUp() override {
            vis::services().clear();
            vis::op::undoHistory().clear();
            manager = std::make_unique<vis::SceneManager>();
            vis::services().set(manager.get());
            auto& scene = manager->getScene();
            dataset = scene.addDataset("dataset");
            scene.addSplatPlaceholder("splats", dataset);
            const auto cameras = scene.addCameraGroup("cameras", dataset, 1);
            scene.addCamera("camera", cameras, sourceCamera());
            scene.addPointCloud("points", std::make_shared<core::PointCloud>(), dataset);
            manager->selectNode("splats");
        }
        void TearDown() override {
            vis::op::undoHistory().clear();
            vis::services().clear();
            manager.reset();
            event::EventBridge::instance().clear_all();
        }
        std::unique_ptr<vis::SceneManager> manager;
        core::NodeId dataset = core::NULL_NODE;
    };

    TEST_F(AprilTagSceneApply, DefaultScalesCompleteDatasetAndUndoRedoRestoresIt) {
        auto& scene = manager->getScene();
        const auto result = vis::cap::applyUniformScaleAboutDataPivot(*manager, {}, {0, 0, 0}, 2);
        ASSERT_TRUE(result.applied) << result.error_message;
        EXPECT_EQ(result.transformed_nodes, std::vector<std::string>{"dataset"});
        EXPECT_FLOAT_EQ(scene.getWorldTransform(dataset)[0][0], 2);
        for (const auto* name : {"splats", "cameras", "camera", "points"}) {
            EXPECT_FLOAT_EQ(scene.getWorldTransform(scene.getNode(name)->id)[0][0], 2);
        }
        ASSERT_TRUE(vis::op::undoHistory().undo().success);
        EXPECT_EQ(scene.getNodeTransform(dataset), glm::mat4(1));
        ASSERT_TRUE(vis::op::undoHistory().redo().success);
        EXPECT_FLOAT_EQ(scene.getWorldTransform(dataset)[0][0], 2);
    }

    TEST_F(AprilTagSceneApply, ExplicitTargetsAllowPartialScaling) {
        const auto result = vis::cap::applyUniformScaleAboutDataPivot(*manager, {"splats"}, {0, 0, 0}, 2);
        ASSERT_TRUE(result.applied) << result.error_message;
        EXPECT_EQ(result.transformed_nodes, std::vector<std::string>{"splats"});
        EXPECT_EQ(manager->getScene().getNodeTransform(dataset), glm::mat4(1));
        EXPECT_EQ(manager->getScene().getNodeTransform("cameras"), glm::mat4(1));
    }

    TEST_F(AprilTagSceneApply, LockedRootDoesNotProducePartialSuccess) {
        auto& scene = manager->getScene();
        scene.addGroup("locked-root");
        scene.setNodeLocked("locked-root", true);
        const auto result = vis::cap::applyUniformScaleAboutDataPivot(*manager, {}, {0, 0, 0}, 2);
        EXPECT_FALSE(result.applied);
        EXPECT_EQ(scene.getNodeTransform(dataset), glm::mat4(1));
    }

    TEST_F(AprilTagSceneApply, StaleReportCannotScaleReplacementScene) {
        auto& scene = manager->getScene();
        const auto snapshot = core::snapshotAprilTagScene(scene);
        core::AprilTagSceneScaleReport report;
        report.success = true;
        report.scale_result.scale_factor = 2;
        scene.clear();
        const auto replacement = scene.addDataset("dataset");
        const auto result = vis::cap::applyAprilTagScaleReport(*manager, snapshot, std::move(report));
        EXPECT_FALSE(result.applied);
        EXPECT_FALSE(result.error_message.empty());
        EXPECT_EQ(scene.getNodeTransform(replacement), glm::mat4(1));
    }
} // namespace
