/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/point_cloud.hpp"
#include "core/selection_domain.hpp"
#include "core/services.hpp"
#include "core/tensor.hpp"
#include "operation/undo_history.hpp"
#include "scene/scene_manager.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace {

    std::shared_ptr<lfs::core::PointCloud> make_point_cloud(const size_t count) {
        std::vector<float> xyz(count * 3, 0.0f);
        std::vector<float> rgb(count * 3, 1.0f);
        auto means = lfs::core::Tensor::from_vector(
            xyz,
            {count, size_t{3}},
            lfs::core::Device::CPU);
        auto colors = lfs::core::Tensor::from_vector(
            rgb,
            {count, size_t{3}},
            lfs::core::Device::CPU);
        return std::make_shared<lfs::core::PointCloud>(std::move(means), std::move(colors));
    }

} // namespace

class SelectionDomainTest : public ::testing::Test {
protected:
    void SetUp() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        lfs::vis::op::undoHistory().clear();
        scene_manager_ = std::make_unique<lfs::vis::SceneManager>();
    }

    void TearDown() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        scene_manager_.reset();
        lfs::vis::op::undoHistory().clear();
    }

    lfs::vis::SceneManager& scene_manager() { return *scene_manager_; }

private:
    std::unique_ptr<lfs::vis::SceneManager> scene_manager_;
};

TEST_F(SelectionDomainTest, SelectedPointCloudResolvesPointCloudDomain) {
    const auto id = scene_manager().getScene().addPointCloud("points", make_point_cloud(2));
    ASSERT_NE(id, lfs::core::NULL_NODE);

    scene_manager().selectNode("points");

    EXPECT_EQ(lfs::vis::resolveSelectionDomain(scene_manager()), lfs::vis::SelectionDomain::PointCloud);
    EXPECT_EQ(lfs::vis::resolveActivePointCloudNode(scene_manager())->id, id);
}

TEST_F(SelectionDomainTest, SelectedPointCloudCropBoxResolvesParentPointCloud) {
    const auto point_cloud_id = scene_manager().getScene().addPointCloud("points", make_point_cloud(2));
    ASSERT_NE(point_cloud_id, lfs::core::NULL_NODE);
    const auto cropbox_id = scene_manager().getScene().addCropBox("crop", point_cloud_id);
    ASSERT_NE(cropbox_id, lfs::core::NULL_NODE);

    scene_manager().selectNode("crop");

    EXPECT_EQ(lfs::vis::resolveSelectionDomain(scene_manager()), lfs::vis::SelectionDomain::PointCloud);
    EXPECT_EQ(lfs::vis::resolveActivePointCloudNode(scene_manager())->id, point_cloud_id);
}

TEST_F(SelectionDomainTest, CameraOnlySelectionResolvesCameraDomain) {
    const auto cameras_id = scene_manager().getScene().addCameraGroup("cameras", lfs::core::NULL_NODE, 0);
    ASSERT_NE(cameras_id, lfs::core::NULL_NODE);

    scene_manager().selectNode("cameras");

    EXPECT_EQ(lfs::vis::resolveSelectionDomain(scene_manager()), lfs::vis::SelectionDomain::Cameras);
    EXPECT_EQ(lfs::vis::resolveActivePointCloudNode(scene_manager()), nullptr);
}

TEST_F(SelectionDomainTest, EmptySelectionAutoTargetsSingleVisiblePointCloudWithPoints) {
    const auto point_cloud_id = scene_manager().getScene().addPointCloud("points", make_point_cloud(2));
    ASSERT_NE(point_cloud_id, lfs::core::NULL_NODE);

    EXPECT_EQ(lfs::vis::resolveSelectionDomain(scene_manager()), lfs::vis::SelectionDomain::PointCloud);
    EXPECT_EQ(lfs::vis::resolveActivePointCloudNode(scene_manager())->id, point_cloud_id);
}

TEST_F(SelectionDomainTest, EmptySelectionFallsBackToGaussiansWhenPointCloudTargetIsAmbiguous) {
    ASSERT_NE(scene_manager().getScene().addPointCloud("points_a", make_point_cloud(2)), lfs::core::NULL_NODE);
    ASSERT_NE(scene_manager().getScene().addPointCloud("points_b", make_point_cloud(2)), lfs::core::NULL_NODE);

    EXPECT_EQ(lfs::vis::resolveSelectionDomain(scene_manager()), lfs::vis::SelectionDomain::Gaussians);
    EXPECT_EQ(lfs::vis::resolveActivePointCloudNode(scene_manager()), nullptr);
}

TEST_F(SelectionDomainTest, EmptySelectionAutoTargetsCamerasWhenOnlyCamerasVisible) {
    const auto cameras_id = scene_manager().getScene().addCameraGroup("cameras", lfs::core::NULL_NODE, 1);
    ASSERT_NE(cameras_id, lfs::core::NULL_NODE);
    ASSERT_NE(scene_manager().getScene().addCamera("cam", cameras_id, std::make_shared<lfs::core::Camera>()),
              lfs::core::NULL_NODE);

    EXPECT_EQ(lfs::vis::resolveSelectionDomain(scene_manager()), lfs::vis::SelectionDomain::Cameras);

    ASSERT_NE(scene_manager().getScene().addPointCloud("points", make_point_cloud(2)), lfs::core::NULL_NODE);
    EXPECT_EQ(lfs::vis::resolveSelectionDomain(scene_manager()), lfs::vis::SelectionDomain::PointCloud);
}

TEST_F(SelectionDomainTest, CameraDomainSticksAfterSelectionCleared) {
    const auto cameras_id = scene_manager().getScene().addCameraGroup("cameras", lfs::core::NULL_NODE, 1);
    ASSERT_NE(cameras_id, lfs::core::NULL_NODE);
    ASSERT_NE(scene_manager().getScene().addCamera("cam", cameras_id, std::make_shared<lfs::core::Camera>()),
              lfs::core::NULL_NODE);
    ASSERT_NE(scene_manager().getScene().addPointCloud("points", make_point_cloud(2)), lfs::core::NULL_NODE);

    scene_manager().selectNode("cam");
    ASSERT_EQ(lfs::vis::resolveSelectionDomain(scene_manager()), lfs::vis::SelectionDomain::Cameras);

    scene_manager().clearSelection();
    EXPECT_EQ(lfs::vis::resolveSelectionDomain(scene_manager()), lfs::vis::SelectionDomain::Cameras);

    scene_manager().selectNode("points");
    EXPECT_EQ(lfs::vis::resolveSelectionDomain(scene_manager()), lfs::vis::SelectionDomain::PointCloud);

    scene_manager().clearSelection();
    EXPECT_EQ(lfs::vis::resolveSelectionDomain(scene_manager()), lfs::vis::SelectionDomain::PointCloud);
}
