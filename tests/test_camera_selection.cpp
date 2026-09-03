/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/point_cloud.hpp"
#include "core/selection_domain.hpp"
#include "core/services.hpp"
#include "core/tensor.hpp"
#include "operation/ops/edit_ops.hpp"
#include "operation/ops/select_ops.hpp"
#include "operation/undo_history.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "selection/selection_service.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace {

    bool hasCudaDevice() {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }

    std::shared_ptr<lfs::core::Camera> make_camera(const std::string& name,
                                                   const int uid,
                                                   const glm::vec3& position) {
        auto rotation = lfs::core::Tensor::from_vector(
            {1.0f, 0.0f, 0.0f,
             0.0f, 1.0f, 0.0f,
             0.0f, 0.0f, 1.0f},
            {size_t{3}, size_t{3}},
            lfs::core::Device::CPU);
        auto translation = lfs::core::Tensor::from_vector(
            {-position.x, -position.y, -position.z},
            {size_t{3}},
            lfs::core::Device::CPU);
        return std::make_shared<lfs::core::Camera>(
            rotation,
            translation,
            50.0f,
            50.0f,
            50.0f,
            50.0f,
            lfs::core::Tensor{},
            lfs::core::Tensor{},
            lfs::core::CameraModelType::PINHOLE,
            name,
            std::filesystem::path{},
            std::filesystem::path{},
            100,
            100,
            uid);
    }

    bool contains_name(std::vector<std::string> names, const std::string& name) {
        return std::ranges::find(names, name) != names.end();
    }

    std::shared_ptr<lfs::core::PointCloud> make_point_cloud() {
        auto means = lfs::core::Tensor::from_vector(
            {0.0f, 0.0f, 0.0f,
             1.0f, 0.0f, 0.0f},
            {size_t{2}, size_t{3}},
            lfs::core::Device::CPU);
        auto colors = lfs::core::Tensor::from_vector(
            {1.0f, 1.0f, 1.0f,
             1.0f, 1.0f, 1.0f},
            {size_t{2}, size_t{3}},
            lfs::core::Device::CPU);
        auto point_cloud = std::make_shared<lfs::core::PointCloud>(std::move(means), std::move(colors));
        point_cloud->selection = std::make_shared<lfs::core::Tensor>(
            lfs::core::Tensor::from_vector(
                std::vector<bool>{true, false},
                {size_t{2}},
                lfs::core::Device::CPU));
        return point_cloud;
    }

} // namespace

class CameraSelectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        lfs::vis::op::undoHistory().clear();

        scene_manager_ = std::make_unique<lfs::vis::SceneManager>();
        rendering_manager_ = std::make_unique<lfs::vis::RenderingManager>();
        lfs::vis::services().set(scene_manager_.get());
        lfs::vis::services().set(rendering_manager_.get());

        auto settings = rendering_manager_->getSettings();
        settings.show_camera_frustums = true;
        rendering_manager_->updateSettings(settings);

        scene_manager_->initSelectionService();
        service_ = scene_manager_->getSelectionService();
        ASSERT_NE(service_, nullptr);
        service_->setTestingViewport({
            .x = 0.0f,
            .y = 0.0f,
            .width = 100.0f,
            .height = 100.0f,
            .render_width = 100,
            .render_height = 100,
        });

        camera_group_id_ = scene_manager_->getScene().addCameraGroup("cameras", lfs::core::NULL_NODE, 3);
        ASSERT_NE(camera_group_id_, lfs::core::NULL_NODE);
        addCamera("cam_center", 1, {0.0f, 0.0f, 0.0f});
        addCamera("cam_right", 2, {2.0f, 0.0f, 0.0f});
        addCamera("cam_behind", 3, {-20.0f, 10.0f, -20.0f});
        scene_manager_->selectNode("cameras");
        ASSERT_EQ(lfs::vis::resolveSelectionDomain(*scene_manager_), lfs::vis::SelectionDomain::Cameras);
    }

    void TearDown() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        service_ = nullptr;
        rendering_manager_.reset();
        scene_manager_.reset();
        lfs::vis::op::undoHistory().clear();
    }

    void addCamera(const std::string& name, const int uid, const glm::vec3& position) {
        const auto camera = hasCudaDevice()
                                ? make_camera(name, uid, position)
                                : std::make_shared<lfs::core::Camera>();
        const auto id = scene_manager_->getScene().addCamera(
            name, camera_group_id_, camera);
        ASSERT_NE(id, lfs::core::NULL_NODE);
    }

    const lfs::core::SceneNode* node(const std::string& name) const {
        return scene_manager_->getScene().getNode(name);
    }

    std::unique_ptr<lfs::vis::SceneManager> scene_manager_;
    std::unique_ptr<lfs::vis::RenderingManager> rendering_manager_;
    lfs::vis::SelectionService* service_ = nullptr;
    lfs::core::NodeId camera_group_id_ = lfs::core::NULL_NODE;
};

TEST_F(CameraSelectionTest, BrushSelectionIsDisabledForCameraDomain) {
    auto selected_before = scene_manager_->getSelectedNodeNames();

    auto brush = service_->selectBrush(50.0f, 50.0f, 10.0f, lfs::vis::SelectionMode::Replace, -1);

    EXPECT_FALSE(brush.success);
    EXPECT_EQ(scene_manager_->getSelectedNodeNames(), selected_before);
}

TEST_F(CameraSelectionTest, HitTestCamerasInShapeSelectsProjectedCameraNodes) {
    if (!hasCudaDevice()) {
        GTEST_SKIP() << "CUDA device required to construct projected camera fixtures";
    }

    auto rect = service_->selectRect(0.0f, 0.0f, 100.0f, 100.0f, lfs::vis::SelectionMode::Replace, -1);
    ASSERT_TRUE(rect.success) << rect.error;
    auto selected = scene_manager_->getSelectedNodeNames();
    EXPECT_TRUE(contains_name(selected, "cam_center"));
    EXPECT_TRUE(contains_name(selected, "cam_right"));
    EXPECT_FALSE(contains_name(selected, "cam_behind"));

    scene_manager_->selectNode("cameras");
    auto polygon = service_->selectPolygon(
        {{0.0f, 0.0f}, {100.0f, 0.0f}, {100.0f, 100.0f}, {0.0f, 100.0f}},
        lfs::vis::SelectionMode::Replace,
        -1);
    ASSERT_TRUE(polygon.success) << polygon.error;
    selected = scene_manager_->getSelectedNodeNames();
    EXPECT_TRUE(contains_name(selected, "cam_center"));
    EXPECT_TRUE(contains_name(selected, "cam_right"));
    EXPECT_FALSE(contains_name(selected, "cam_behind"));
}

TEST_F(CameraSelectionTest, SetCamerasTrainingEnabledWithHistoryUndoRedoAndNoop) {
    ASSERT_TRUE(node("cam_center")->training_enabled);
    ASSERT_TRUE(node("cam_right")->training_enabled);

    auto disabled = scene_manager_->setCamerasTrainingEnabledWithHistory(
        {"cam_center", "cam_right"}, false);
    ASSERT_TRUE(disabled.has_value()) << disabled.error();
    EXPECT_EQ(*disabled, 2u);
    EXPECT_FALSE(node("cam_center")->training_enabled);
    EXPECT_FALSE(node("cam_right")->training_enabled);
    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);

    ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
    EXPECT_TRUE(node("cam_center")->training_enabled);
    EXPECT_TRUE(node("cam_right")->training_enabled);

    ASSERT_TRUE(lfs::vis::op::undoHistory().redo().success);
    EXPECT_FALSE(node("cam_center")->training_enabled);
    EXPECT_FALSE(node("cam_right")->training_enabled);

    auto noop = scene_manager_->setCamerasTrainingEnabledWithHistory({"cam_center"}, false);
    ASSERT_TRUE(noop.has_value()) << noop.error();
    EXPECT_EQ(*noop, 0u);
    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
}

TEST_F(CameraSelectionTest, EditDeleteDisablesSelectedCamerasWithoutRemovingNodes) {
    scene_manager_->selectNodes({"cam_center", "cam_right"});
    lfs::vis::op::EditDelete op;
    ASSERT_TRUE(op.poll(*scene_manager_));

    const auto result = op.execute(*scene_manager_, lfs::vis::op::OperatorProperties{}, {});
    ASSERT_TRUE(result.ok()) << result.error;
    EXPECT_FALSE(node("cam_center")->training_enabled);
    EXPECT_FALSE(node("cam_right")->training_enabled);
    EXPECT_NE(node("cam_center"), nullptr);
    EXPECT_NE(node("cam_right"), nullptr);
}

TEST_F(CameraSelectionTest, SelectAllNoneInvertRoundTripUsesNodeSelection) {
    lfs::vis::op::SelectAll select_all;
    lfs::vis::op::SelectNone select_none;
    lfs::vis::op::SelectInvert select_invert;

    ASSERT_TRUE(select_all.poll(*scene_manager_));
    ASSERT_TRUE(select_all.execute(*scene_manager_, lfs::vis::op::OperatorProperties{}, {}).ok());
    auto selected = scene_manager_->getSelectedNodeNames();
    EXPECT_TRUE(contains_name(selected, "cam_center"));
    EXPECT_TRUE(contains_name(selected, "cam_right"));
    EXPECT_TRUE(contains_name(selected, "cam_behind"));

    ASSERT_TRUE(select_none.poll(*scene_manager_));
    ASSERT_TRUE(select_none.execute(*scene_manager_, lfs::vis::op::OperatorProperties{}, {}).ok());
    EXPECT_TRUE(scene_manager_->getSelectedNodeNames().empty());

    scene_manager_->selectNode("cameras");
    ASSERT_TRUE(select_invert.poll(*scene_manager_));
    ASSERT_TRUE(select_invert.execute(*scene_manager_, lfs::vis::op::OperatorProperties{}, {}).ok());
    selected = scene_manager_->getSelectedNodeNames();
    EXPECT_TRUE(contains_name(selected, "cam_center"));
    EXPECT_TRUE(contains_name(selected, "cam_right"));
    EXPECT_TRUE(contains_name(selected, "cam_behind"));
}

TEST_F(CameraSelectionTest, DeleteSelectedCommandRoutesByDomain) {
    scene_manager_->selectNodes({"cam_center", "cam_right"});
    lfs::core::events::cmd::DeleteSelected{}.emit();
    ASSERT_NE(node("cam_center"), nullptr);
    ASSERT_NE(node("cam_right"), nullptr);
    EXPECT_FALSE(node("cam_center")->training_enabled);
    EXPECT_FALSE(node("cam_right")->training_enabled);

    auto point_cloud = make_point_cloud();
    ASSERT_NE(scene_manager_->getScene().addPointCloud("points", point_cloud), lfs::core::NULL_NODE);
    scene_manager_->selectNode("points");
    ASSERT_EQ(lfs::vis::resolveSelectionDomain(*scene_manager_), lfs::vis::SelectionDomain::PointCloud);

    lfs::core::events::cmd::DeleteSelected{}.emit();
    const auto* points_node = node("points");
    ASSERT_NE(points_node, nullptr);
    ASSERT_NE(points_node->point_cloud, nullptr);
    ASSERT_TRUE(points_node->point_cloud->has_deleted());
    EXPECT_EQ(points_node->point_cloud->deleted->to_vector_bool(), (std::vector<bool>{true, false}));
}

TEST_F(CameraSelectionTest, PointDomainEditDeleteStillSoftDeletesPoints) {
    scene_manager_->clearSelection();
    auto point_cloud = make_point_cloud();
    const auto point_id = scene_manager_->getScene().addPointCloud("points", point_cloud);
    ASSERT_NE(point_id, lfs::core::NULL_NODE);
    scene_manager_->selectNode("points");
    ASSERT_EQ(lfs::vis::resolveSelectionDomain(*scene_manager_), lfs::vis::SelectionDomain::PointCloud);

    lfs::vis::op::EditDelete op;
    ASSERT_TRUE(op.poll(*scene_manager_));
    const auto result = op.execute(*scene_manager_, lfs::vis::op::OperatorProperties{}, {});
    ASSERT_TRUE(result.ok()) << result.error;

    const auto* points_node = scene_manager_->getScene().getNode("points");
    ASSERT_NE(points_node, nullptr);
    ASSERT_NE(points_node->point_cloud, nullptr);
    EXPECT_FALSE(points_node->point_cloud->has_selection());
    ASSERT_TRUE(points_node->point_cloud->has_deleted());
    EXPECT_EQ(points_node->point_cloud->deleted->to_vector_bool(), (std::vector<bool>{true, false}));
}
