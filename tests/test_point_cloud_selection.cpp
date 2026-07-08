/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/point_cloud.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "geometry/bounding_box.hpp"
#include "operation/ops/edit_ops.hpp"
#include "operation/undo_history.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "selection/selection_service.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::PointCloud;
using lfs::core::Tensor;

namespace {

    bool hasCudaDevice() {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }

    std::shared_ptr<PointCloud> make_point_cloud(const size_t count = 5) {
        std::vector<float> xyz(count * 3, 0.0f);
        std::vector<float> rgb(count * 3, 1.0f);
        return std::make_shared<PointCloud>(
            Tensor::from_vector(xyz, {count, size_t{3}}, Device::CPU),
            Tensor::from_vector(rgb, {count, size_t{3}}, Device::CPU));
    }

    Tensor make_means(const std::vector<float>& values, const Device device = Device::CUDA) {
        auto means = Tensor::from_vector(values, {values.size() / 3, size_t{3}}, Device::CPU).to(DataType::Float32);
        return device == Device::CUDA ? means.cuda() : means;
    }

    Tensor make_colors_u8(const std::vector<uint8_t>& values, const Device device = Device::CPU) {
        auto colors = Tensor::empty({values.size() / 3, size_t{3}}, Device::CPU, DataType::UInt8);
        std::copy(values.begin(), values.end(), colors.ptr<uint8_t>());
        return device == Device::CUDA ? colors.cuda() : colors;
    }

    std::shared_ptr<Tensor> make_screen_positions() {
        return std::make_shared<Tensor>(
            Tensor::from_vector(
                {
                    10.0f,
                    10.0f,
                    30.0f,
                    30.0f,
                    60.0f,
                    60.0f,
                    80.0f,
                    20.0f,
                    45.0f,
                    75.0f,
                },
                {size_t{5}, size_t{2}},
                Device::CUDA)
                .to(DataType::Float32));
    }

    Tensor make_u8_mask(const std::vector<uint8_t>& values) {
        auto tensor = Tensor::empty({values.size()}, Device::CPU, DataType::UInt8);
        std::copy(values.begin(), values.end(), tensor.ptr<uint8_t>());
        return tensor;
    }

    std::shared_ptr<Tensor> make_deleted_mask(const std::vector<bool>& values) {
        return std::make_shared<Tensor>(Tensor::from_vector(values, {values.size()}, Device::CPU));
    }

    std::vector<uint8_t> point_selection_values(const PointCloud& pc) {
        if (!pc.selection || !pc.selection->is_valid()) {
            return {};
        }
        return pc.selection->cpu().to_vector_uint8();
    }

    std::vector<bool> point_deleted_values(const PointCloud& pc) {
        if (!pc.deleted || !pc.deleted->is_valid()) {
            return {};
        }
        return pc.deleted->cpu().to_vector_bool();
    }

    std::unique_ptr<lfs::core::SplatData> make_test_splat(const size_t count) {
        auto means = Tensor::zeros({count, size_t{3}}, Device::CUDA, DataType::Float32);
        auto sh0 = Tensor::zeros({count, size_t{1}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto shN = Tensor::zeros({count, size_t{3}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto scaling = Tensor::zeros({count, size_t{3}}, Device::CUDA, DataType::Float32);
        auto rotation = Tensor::zeros({count, size_t{4}}, Device::CUDA, DataType::Float32);
        auto opacity = Tensor::zeros({count, size_t{1}}, Device::CUDA, DataType::Float32);
        return std::make_unique<lfs::core::SplatData>(
            1, std::move(means), std::move(sh0), std::move(shN), std::move(scaling),
            std::move(rotation), std::move(opacity), 1.0f);
    }

} // namespace

class PointCloudSelectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!hasCudaDevice()) {
            GTEST_SKIP() << "CUDA device required for selection kernels";
        }
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        lfs::vis::op::undoHistory().clear();

        scene_manager_ = std::make_unique<lfs::vis::SceneManager>();
        rendering_manager_ = std::make_unique<lfs::vis::RenderingManager>();
        lfs::vis::services().set(scene_manager_.get());
        lfs::vis::services().set(rendering_manager_.get());
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

    PointCloud& addPointCloud() {
        point_cloud_ = make_point_cloud();
        const auto id = scene_manager_->getScene().addPointCloud("points", point_cloud_);
        EXPECT_NE(id, lfs::core::NULL_NODE);
        scene_manager_->selectNode("points");
        service_->setTestingScreenPositions(make_screen_positions());
        return *point_cloud_;
    }

    void setPointSelection(const std::vector<uint8_t>& values) {
        point_cloud_->selection = std::make_shared<Tensor>(make_u8_mask(values));
    }

    void setLinePointCloudMeans() {
        point_cloud_->means = make_means({
            -2.0f,
            0.0f,
            0.0f,
            -0.5f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.5f,
            0.0f,
            0.0f,
            2.0f,
            0.0f,
            0.0f,
        });
    }

    std::unique_ptr<lfs::vis::SceneManager> scene_manager_;
    std::unique_ptr<lfs::vis::RenderingManager> rendering_manager_;
    lfs::vis::SelectionService* service_ = nullptr;
    std::shared_ptr<PointCloud> point_cloud_;
};

TEST_F(PointCloudSelectionTest, InteractiveRectSelectsExpectedPoints) {
    auto& pc = addPointCloud();

    ASSERT_TRUE(service_->beginInteractiveSelection(
        lfs::vis::SelectionShape::Rectangle, lfs::vis::SelectionMode::Replace, {0.0f, 0.0f}, 0.0f));
    service_->updateInteractiveSelection({50.0f, 50.0f});
    const auto result = service_->finishInteractiveSelection();

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{1, 1, 0, 0, 0}));
}

TEST_F(PointCloudSelectionTest, BrushAndLassoSelectExpectedPoints) {
    auto& pc = addPointCloud();

    auto brush = service_->selectBrush(80.0f, 20.0f, 6.0f, lfs::vis::SelectionMode::Replace, -1);
    ASSERT_TRUE(brush.success) << brush.error;
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{0, 0, 0, 1, 0}));

    auto lasso = service_->selectLasso(
        {{0.0f, 0.0f}, {50.0f, 0.0f}, {50.0f, 50.0f}, {0.0f, 50.0f}},
        lfs::vis::SelectionMode::Replace,
        -1);
    ASSERT_TRUE(lasso.success) << lasso.error;
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{1, 1, 0, 0, 0}));
}

TEST_F(PointCloudSelectionTest, PolygonSelectsExpectedPoints) {
    auto& pc = addPointCloud();

    auto polygon = service_->selectPolygon(
        {{0.0f, 0.0f}, {50.0f, 0.0f}, {50.0f, 50.0f}, {0.0f, 50.0f}},
        lfs::vis::SelectionMode::Replace,
        -1);

    ASSERT_TRUE(polygon.success) << polygon.error;
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{1, 1, 0, 0, 0}));
}

TEST_F(PointCloudSelectionTest, BoxAndSphereVolumeSelectPointMeans) {
    auto& pc = addPointCloud();
    setLinePointCloudMeans();

    rendering_manager_->setCropboxGizmoState(
        true, glm::vec3(-1.0f), glm::vec3(1.0f), glm::mat4(1.0f), false);
    auto box = service_->selectBoxVolume(lfs::vis::SelectionMode::Replace);
    ASSERT_TRUE(box.success) << box.error;
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{0, 1, 1, 1, 0}));

    rendering_manager_->setEllipsoidGizmoState(
        true, glm::vec3(0.75f), glm::mat4(1.0f), false);
    auto sphere = service_->selectSphereVolume(lfs::vis::SelectionMode::Replace);
    ASSERT_TRUE(sphere.success) << sphere.error;
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{0, 1, 1, 1, 0}));
}

TEST_F(PointCloudSelectionTest, ColorSelectsMatchingPointCluster) {
    auto& pc = addPointCloud();
    pc.colors = make_colors_u8({
        255,
        0,
        0,
        250,
        5,
        0,
        0,
        255,
        0,
        0,
        250,
        5,
        0,
        0,
        255,
    });

    auto color = service_->selectByColorAt(10.0f, 10.0f, lfs::vis::SelectionMode::Replace);

    ASSERT_TRUE(color.success) << color.error;
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{1, 1, 0, 0, 0}));
}

TEST_F(PointCloudSelectionTest, ModesCombineWithExistingSelection) {
    auto& pc = addPointCloud();

    setPointSelection({1, 0, 1, 0, 0});
    auto add = service_->selectRect(0.0f, 0.0f, 50.0f, 50.0f, lfs::vis::SelectionMode::Add, -1);
    ASSERT_TRUE(add.success) << add.error;
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{1, 1, 1, 0, 0}));

    setPointSelection({1, 1, 1, 0, 0});
    auto remove = service_->selectRect(0.0f, 0.0f, 50.0f, 50.0f, lfs::vis::SelectionMode::Remove, -1);
    ASSERT_TRUE(remove.success) << remove.error;
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{0, 0, 1, 0, 0}));

    setPointSelection({1, 0, 1, 1, 0});
    auto intersect = service_->selectRect(0.0f, 0.0f, 50.0f, 50.0f, lfs::vis::SelectionMode::Intersect, -1);
    ASSERT_TRUE(intersect.success) << intersect.error;
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{1, 0, 0, 0, 0}));
}

TEST_F(PointCloudSelectionTest, DeletedPointsAreExcludedFromSelectionAllAndInvert) {
    auto& pc = addPointCloud();
    pc.deleted = make_deleted_mask({false, true, false, true, false});

    auto rect = service_->selectRect(0.0f, 0.0f, 100.0f, 100.0f, lfs::vis::SelectionMode::Replace, -1);
    ASSERT_TRUE(rect.success) << rect.error;
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{1, 0, 1, 0, 1}));

    auto clear = scene_manager_->clearPointSelection();
    ASSERT_TRUE(clear) << clear.error();
    auto select_all = scene_manager_->selectAllPoints();
    ASSERT_TRUE(select_all) << select_all.error();
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{1, 0, 1, 0, 1}));

    setPointSelection({1, 0, 0, 0, 0});
    auto invert = scene_manager_->invertPointSelection();
    ASSERT_TRUE(invert) << invert.error();
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{0, 0, 1, 0, 1}));
}

TEST_F(PointCloudSelectionTest, CropFilterInPointCommitExcludesOutsidePoints) {
    auto& pc = addPointCloud();
    setLinePointCloudMeans();

    const auto* point_node = scene_manager_->getScene().getNode("points");
    ASSERT_NE(point_node, nullptr);
    const auto cropbox_id = scene_manager_->getScene().addCropBox("points_crop", point_node->id);
    ASSERT_NE(cropbox_id, lfs::core::NULL_NODE);
    lfs::core::CropBoxData cropbox;
    cropbox.min = glm::vec3(-1.0f);
    cropbox.max = glm::vec3(1.0f);
    cropbox.enabled = true;
    scene_manager_->getScene().setCropBoxData(cropbox_id, cropbox);

    auto settings = rendering_manager_->getSettings();
    settings.crop_filter_for_selection = true;
    rendering_manager_->updateSettings(settings);

    auto rect = service_->selectRect(0.0f, 0.0f, 100.0f, 100.0f, lfs::vis::SelectionMode::Replace, -1);
    ASSERT_TRUE(rect.success) << rect.error;
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{0, 1, 1, 1, 0}));
}

TEST_F(PointCloudSelectionTest, SelectAllInvertClearRoundTrip) {
    auto& pc = addPointCloud();

    auto select_all = scene_manager_->selectAllPoints();
    ASSERT_TRUE(select_all) << select_all.error();
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{1, 1, 1, 1, 1}));
    EXPECT_EQ(scene_manager_->getActivePointSelectionCount(), 5u);

    auto invert = scene_manager_->invertPointSelection();
    ASSERT_TRUE(invert) << invert.error();
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{0, 0, 0, 0, 0}));

    auto clear = scene_manager_->clearPointSelection();
    ASSERT_TRUE(clear) << clear.error();
    EXPECT_EQ(pc.selection, nullptr);
    EXPECT_EQ(scene_manager_->getActivePointSelectionCount(), 0u);
}

TEST_F(PointCloudSelectionTest, UndoRedoRestoresPointSelectionMaskPointerAndContents) {
    auto& pc = addPointCloud();
    setPointSelection({1, 0, 0, 0, 0});
    const auto before = pc.selection;

    auto rect = service_->selectRect(0.0f, 0.0f, 50.0f, 50.0f, lfs::vis::SelectionMode::Replace, -1);
    ASSERT_TRUE(rect.success) << rect.error;
    const auto after = pc.selection;
    ASSERT_NE(before, after);
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{1, 1, 0, 0, 0}));

    auto undo = lfs::vis::op::undoHistory().undo();
    ASSERT_TRUE(undo.success) << undo.error;
    EXPECT_EQ(pc.selection, before);
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{1, 0, 0, 0, 0}));

    auto redo = lfs::vis::op::undoHistory().redo();
    ASSERT_TRUE(redo.success) << redo.error;
    EXPECT_EQ(pc.selection, after);
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{1, 1, 0, 0, 0}));
}

TEST_F(PointCloudSelectionTest, DeleteSelectedPointsSoftDeletesClearsSelectionAndUndoRedoRestoresMasks) {
    auto& pc = addPointCloud();
    setPointSelection({1, 0, 1, 0, 0});
    pc.deleted = make_deleted_mask({false, true, false, false, false});
    const auto selection_before = pc.selection;
    const auto deleted_before = pc.deleted;

    auto result = scene_manager_->deleteSelectedPointsWithHistory();
    ASSERT_TRUE(result) << result.error();

    EXPECT_EQ(pc.selection, nullptr);
    EXPECT_EQ(point_deleted_values(pc), (std::vector<bool>{true, true, true, false, false}));
    EXPECT_EQ(scene_manager_->getActivePointSelectionCount(), 0u);
    EXPECT_TRUE(scene_manager_->getScene().isPointCloudModified());

    auto undo = lfs::vis::op::undoHistory().undo();
    ASSERT_TRUE(undo.success) << undo.error;
    EXPECT_EQ(pc.selection, selection_before);
    EXPECT_EQ(pc.deleted, deleted_before);
    EXPECT_EQ(point_selection_values(pc), (std::vector<uint8_t>{1, 0, 1, 0, 0}));
    EXPECT_EQ(point_deleted_values(pc), (std::vector<bool>{false, true, false, false, false}));
    EXPECT_EQ(scene_manager_->getActivePointSelectionCount(), 2u);

    auto redo = lfs::vis::op::undoHistory().redo();
    ASSERT_TRUE(redo.success) << redo.error;
    EXPECT_EQ(pc.selection, nullptr);
    EXPECT_EQ(point_deleted_values(pc), (std::vector<bool>{true, true, true, false, false}));
    EXPECT_EQ(scene_manager_->getActivePointSelectionCount(), 0u);
}

TEST_F(PointCloudSelectionTest, EditDeleteDispatchesPointDomainToSoftDelete) {
    auto& pc = addPointCloud();
    setPointSelection({0, 1, 0, 1, 0});

    lfs::vis::op::EditDelete op;
    ASSERT_TRUE(op.poll(*scene_manager_));
    auto result = op.execute(*scene_manager_, {}, {});

    ASSERT_TRUE(result.ok()) << result.error;
    EXPECT_EQ(pc.selection, nullptr);
    EXPECT_EQ(point_deleted_values(pc), (std::vector<bool>{false, true, false, true, false}));
}

TEST_F(PointCloudSelectionTest, CropApplyDoesNotResurrectSoftDeletedPointCloudPoints) {
    auto& pc = addPointCloud();
    pc.means = Tensor::from_vector(
                   {
                       -2.0f,
                       0.0f,
                       0.0f,
                       -0.5f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.5f,
                       0.0f,
                       0.0f,
                       2.0f,
                       0.0f,
                       0.0f,
                   },
                   {size_t{5}, size_t{3}},
                   Device::CUDA)
                   .to(DataType::Float32);
    pc.colors = Tensor::zeros({size_t{5}, size_t{3}}, Device::CUDA, DataType::UInt8);
    pc.deleted = make_deleted_mask({false, false, true, false, false});

    lfs::geometry::BoundingBox crop_box;
    crop_box.setBounds(glm::vec3(-1.0f), glm::vec3(1.0f));
    lfs::core::events::cmd::CropPLY{.crop_box = crop_box, .inverse = false}.emit();

    const auto* node = scene_manager_->getScene().getNode("points");
    ASSERT_NE(node, nullptr);
    ASSERT_TRUE(node->point_cloud);
    ASSERT_EQ(node->point_cloud->size(), 2);
    EXPECT_EQ(node->point_cloud->means.cpu().to_vector(),
              (std::vector<float>{-0.5f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f}));
    EXPECT_FALSE(node->point_cloud->has_deleted());
    EXPECT_FALSE(node->point_cloud->has_selection());
}

TEST_F(PointCloudSelectionTest, UnsupportedPointCloudShapesFailGracefully) {
    addPointCloud();

    EXPECT_FALSE(service_->selectRing(10.0f, 10.0f, lfs::vis::SelectionMode::Replace, -1).success);
}

TEST_F(PointCloudSelectionTest, MergedVisiblePointCloudMasksAreExposedWithOffsets) {
    auto first = make_point_cloud(2);
    first->selection = std::make_shared<Tensor>(make_u8_mask({1, 0}));
    auto second = make_point_cloud(3);
    second->deleted = make_deleted_mask({false, true, false});

    scene_manager_->getScene().addPointCloud("first", first);
    scene_manager_->getScene().addPointCloud("second", second);

    const auto state = scene_manager_->buildRenderState();

    ASSERT_NE(state.point_cloud, nullptr);
    ASSERT_TRUE(state.point_cloud_selection_mask && state.point_cloud_selection_mask->is_valid());
    ASSERT_TRUE(state.point_cloud_deleted_mask && state.point_cloud_deleted_mask->is_valid());
    EXPECT_EQ(state.point_cloud_selection_mask->cpu().to_vector_uint8(),
              (std::vector<uint8_t>{1, 0, 0, 0, 0}));
    EXPECT_EQ(state.point_cloud_deleted_mask->cpu().to_vector_bool(),
              (std::vector<bool>{false, false, false, true, false}));
}

TEST_F(PointCloudSelectionTest, GaussianSceneStillCommitsSceneSelectionMask) {
    scene_manager_->getScene().addSplat("splat", make_test_splat(5));
    service_->setTestingScreenPositions(make_screen_positions());

    auto result = service_->selectRect(0.0f, 0.0f, 50.0f, 50.0f, lfs::vis::SelectionMode::Replace, -1);
    ASSERT_TRUE(result.success) << result.error;

    const auto mask = scene_manager_->getScene().getSelectionMask();
    ASSERT_TRUE(mask && mask->is_valid());
    EXPECT_EQ(mask->cpu().to_vector_uint8(), (std::vector<uint8_t>{1, 1, 0, 0, 0}));
}
