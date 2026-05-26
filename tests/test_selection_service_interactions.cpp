/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "operation/undo_history.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "selection/selection_service.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;

namespace {

    Tensor make_uint8_mask(const std::vector<uint8_t>& values) {
        auto tensor = Tensor::empty({values.size()}, Device::CPU, DataType::UInt8);
        std::copy(values.begin(), values.end(), tensor.ptr<uint8_t>());
        return tensor.cuda();
    }

    std::shared_ptr<Tensor> make_screen_positions(const std::vector<float>& xy) {
        return std::make_shared<Tensor>(
            Tensor::from_vector(xy, {xy.size() / 2, size_t{2}}, Device::CUDA).to(DataType::Float32));
    }

    std::unique_ptr<lfs::core::SplatData> make_test_splat(const std::vector<float>& xyz) {
        const size_t count = xyz.size() / 3;
        auto means = Tensor::from_vector(xyz, {count, size_t{3}}, Device::CUDA).to(DataType::Float32);
        auto sh0 = Tensor::zeros({count, size_t{1}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto shN = Tensor::zeros({count, size_t{3}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto scaling = Tensor::zeros({count, size_t{3}}, Device::CUDA, DataType::Float32);

        std::vector<float> rotation_data(count * 4, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            rotation_data[i * 4] = 1.0f;
        }
        auto rotation = Tensor::from_vector(rotation_data, {count, size_t{4}}, Device::CUDA).to(DataType::Float32);
        auto opacity = Tensor::zeros({count, size_t{1}}, Device::CUDA, DataType::Float32);

        return std::make_unique<lfs::core::SplatData>(
            1,
            std::move(means),
            std::move(sh0),
            std::move(shN),
            std::move(scaling),
            std::move(rotation),
            std::move(opacity),
            1.0f);
    }

    std::vector<uint8_t> selection_values(const lfs::vis::SceneManager& scene_manager) {
        const auto mask = scene_manager.getScene().getSelectionMask();
        if (!mask || !mask->is_valid()) {
            return {};
        }
        return mask->cpu().to_vector_uint8();
    }

} // namespace

class SelectionServiceInteractionsTest : public ::testing::Test {
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

        scene_manager_->getScene().addNode(
            "test",
            make_test_splat({
                0.0f,
                0.0f,
                0.0f,
                1.0f,
                0.0f,
                0.0f,
            }));

        service_ = std::make_unique<lfs::vis::SelectionService>(scene_manager_.get(), rendering_manager_.get());
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
        service_.reset();
        rendering_manager_.reset();
        scene_manager_.reset();
        lfs::vis::op::undoHistory().clear();
    }

    void set_initial_selection(const std::vector<uint8_t>& values) {
        scene_manager_->getScene().setSelectionMask(std::make_shared<Tensor>(make_uint8_mask(values)));
    }

    std::unique_ptr<lfs::vis::SceneManager> scene_manager_;
    std::unique_ptr<lfs::vis::RenderingManager> rendering_manager_;
    std::unique_ptr<lfs::vis::SelectionService> service_;
};

TEST_F(SelectionServiceInteractionsTest, ClosedPolygonDragUpdatesVertexPosition) {
    ASSERT_TRUE(service_->beginInteractiveSelection(
        lfs::vis::SelectionShape::Polygon,
        lfs::vis::SelectionMode::Replace,
        {0.0f, 0.0f},
        0.0f));
    ASSERT_TRUE(service_->appendInteractivePolygonVertex({30.0f, 0.0f}));
    ASSERT_TRUE(service_->appendInteractivePolygonVertex({0.0f, 30.0f}));
    ASSERT_TRUE(service_->appendInteractivePolygonVertex({0.0f, 0.0f}));
    ASSERT_TRUE(service_->isInteractiveSelectionClosed());

    ASSERT_TRUE(service_->beginInteractivePolygonVertexDrag({30.0f, 0.0f}));
    service_->updateInteractiveSelection({40.0f, 0.0f});
    service_->endInteractivePolygonVertexDrag();
    service_->refreshInteractivePreview();

    ASSERT_TRUE(rendering_manager_->isPolygonPreviewActive());
    ASSERT_FALSE(rendering_manager_->isPolygonPreviewWorldSpace());
    const auto& points = rendering_manager_->getPolygonPoints();
    ASSERT_EQ(points.size(), 3u);
    EXPECT_FLOAT_EQ(points[0].first, 0.0f);
    EXPECT_FLOAT_EQ(points[0].second, 0.0f);
    EXPECT_FLOAT_EQ(points[1].first, 40.0f);
    EXPECT_FLOAT_EQ(points[1].second, 0.0f);
    EXPECT_FLOAT_EQ(points[2].first, 0.0f);
    EXPECT_FLOAT_EQ(points[2].second, 30.0f);
}

TEST_F(SelectionServiceInteractionsTest, ClosedPolygonInsertAndRemoveVertexUpdatePreview) {
    ASSERT_TRUE(service_->beginInteractiveSelection(
        lfs::vis::SelectionShape::Polygon,
        lfs::vis::SelectionMode::Replace,
        {0.0f, 0.0f},
        0.0f));
    ASSERT_TRUE(service_->appendInteractivePolygonVertex({30.0f, 0.0f}));
    ASSERT_TRUE(service_->appendInteractivePolygonVertex({0.0f, 30.0f}));
    ASSERT_TRUE(service_->appendInteractivePolygonVertex({0.0f, 0.0f}));
    ASSERT_TRUE(service_->isInteractiveSelectionClosed());

    ASSERT_TRUE(service_->insertInteractivePolygonVertex({15.0f, 15.0f}));
    service_->endInteractivePolygonVertexDrag();
    service_->refreshInteractivePreview();

    ASSERT_TRUE(rendering_manager_->isPolygonPreviewActive());
    const auto& inserted_points = rendering_manager_->getPolygonPoints();
    ASSERT_EQ(inserted_points.size(), 4u);
    EXPECT_FLOAT_EQ(inserted_points[0].first, 0.0f);
    EXPECT_FLOAT_EQ(inserted_points[0].second, 0.0f);
    EXPECT_FLOAT_EQ(inserted_points[1].first, 30.0f);
    EXPECT_FLOAT_EQ(inserted_points[1].second, 0.0f);
    EXPECT_FLOAT_EQ(inserted_points[2].first, 15.0f);
    EXPECT_FLOAT_EQ(inserted_points[2].second, 15.0f);
    EXPECT_FLOAT_EQ(inserted_points[3].first, 0.0f);
    EXPECT_FLOAT_EQ(inserted_points[3].second, 30.0f);

    ASSERT_TRUE(service_->removeInteractivePolygonVertex({15.0f, 15.0f}));
    service_->refreshInteractivePreview();

    const auto& reduced_points = rendering_manager_->getPolygonPoints();
    ASSERT_EQ(reduced_points.size(), 3u);
    EXPECT_FLOAT_EQ(reduced_points[0].first, 0.0f);
    EXPECT_FLOAT_EQ(reduced_points[0].second, 0.0f);
    EXPECT_FLOAT_EQ(reduced_points[1].first, 30.0f);
    EXPECT_FLOAT_EQ(reduced_points[1].second, 0.0f);
    EXPECT_FLOAT_EQ(reduced_points[2].first, 0.0f);
    EXPECT_FLOAT_EQ(reduced_points[2].second, 30.0f);
}

TEST_F(SelectionServiceInteractionsTest, InteractiveRectAndLassoPreviewTrackFrameCursor) {
    rendering_manager_->setRectPreview(0.0f, 0.0f, 10.0f, 10.0f);
    rendering_manager_->setLassoPreview({{0.0f, 0.0f}, {10.0f, 10.0f}});
    EXPECT_FALSE(rendering_manager_->rectPreviewTracksCursor());
    EXPECT_FALSE(rendering_manager_->lassoPreviewTracksCursor());
    rendering_manager_->clearSelectionPreviews();

    ASSERT_TRUE(service_->beginInteractiveSelection(
        lfs::vis::SelectionShape::Rectangle,
        lfs::vis::SelectionMode::Replace,
        {0.0f, 0.0f},
        0.0f));
    service_->updateInteractiveSelection({40.0f, 50.0f});
    service_->refreshInteractivePreview();
    EXPECT_TRUE(rendering_manager_->isRectPreviewActive());
    EXPECT_TRUE(rendering_manager_->rectPreviewTracksCursor());
    service_->cancelInteractiveSelection();

    ASSERT_TRUE(service_->beginInteractiveSelection(
        lfs::vis::SelectionShape::Lasso,
        lfs::vis::SelectionMode::Replace,
        {0.0f, 0.0f},
        0.0f));
    service_->updateInteractiveSelection({40.0f, 50.0f});
    service_->refreshInteractivePreview();
    EXPECT_TRUE(rendering_manager_->isLassoPreviewActive());
    EXPECT_TRUE(rendering_manager_->lassoPreviewTracksCursor());
}

TEST_F(SelectionServiceInteractionsTest, CancelInteractiveSelectionLeavesSelectionAndUndoUntouched) {
    set_initial_selection({1, 0});
    service_->setTestingScreenPositions(make_screen_positions({
        80.0f,
        80.0f,
        10.0f,
        10.0f,
    }));

    ASSERT_TRUE(service_->beginInteractiveSelection(
        lfs::vis::SelectionShape::Polygon,
        lfs::vis::SelectionMode::Replace,
        {0.0f, 0.0f},
        0.0f));
    ASSERT_TRUE(service_->appendInteractivePolygonVertex({30.0f, 0.0f}));
    ASSERT_TRUE(service_->appendInteractivePolygonVertex({0.0f, 30.0f}));

    service_->cancelInteractiveSelection();

    EXPECT_FALSE(service_->isInteractiveSelectionActive());
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
    EXPECT_EQ(lfs::vis::op::undoHistory().redoCount(), 0u);
}

TEST_F(SelectionServiceInteractionsTest, PointCloudModeBrushUsesScreenPositionFallback) {
    auto settings = rendering_manager_->getSettings();
    settings.point_cloud_mode = true;
    rendering_manager_->updateSettings(settings);

    service_->setTestingScreenPositions(make_screen_positions({
        10.0f,
        10.0f,
        80.0f,
        80.0f,
    }));

    ASSERT_TRUE(service_->beginInteractiveSelection(
        lfs::vis::SelectionShape::Brush,
        lfs::vis::SelectionMode::Replace,
        {10.0f, 10.0f},
        5.0f));

    const auto result = service_->finishInteractiveSelection();
    ASSERT_TRUE(result.success);
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionServiceInteractionsTest, PointCloudModeRectangleUsesScreenPositionFallback) {
    auto settings = rendering_manager_->getSettings();
    settings.point_cloud_mode = true;
    rendering_manager_->updateSettings(settings);

    service_->setTestingScreenPositions(make_screen_positions({
        10.0f,
        10.0f,
        80.0f,
        80.0f,
    }));

    ASSERT_TRUE(service_->beginInteractiveSelection(
        lfs::vis::SelectionShape::Rectangle,
        lfs::vis::SelectionMode::Replace,
        {0.0f, 0.0f},
        0.0f));
    service_->updateInteractiveSelection({50.0f, 50.0f});

    const auto result = service_->finishInteractiveSelection();
    ASSERT_TRUE(result.success);
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionServiceInteractionsTest, PointCloudModePolygonUsesScreenPositionFallback) {
    auto settings = rendering_manager_->getSettings();
    settings.point_cloud_mode = true;
    rendering_manager_->updateSettings(settings);

    service_->setTestingScreenPositions(make_screen_positions({
        10.0f,
        10.0f,
        80.0f,
        80.0f,
    }));

    ASSERT_TRUE(service_->beginInteractiveSelection(
        lfs::vis::SelectionShape::Polygon,
        lfs::vis::SelectionMode::Replace,
        {0.0f, 0.0f},
        0.0f));
    ASSERT_TRUE(service_->appendInteractivePolygonVertex({50.0f, 0.0f}));
    ASSERT_TRUE(service_->appendInteractivePolygonVertex({0.0f, 50.0f}));

    const auto result = service_->finishInteractiveSelection();
    ASSERT_TRUE(result.success);
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionServiceInteractionsTest, PointCloudModeSelectionFallbackAppliesDepthFilter) {
    auto settings = rendering_manager_->getSettings();
    settings.point_cloud_mode = true;
    settings.depth_filter_enabled = true;
    settings.depth_filter_min = {-0.25f, -0.25f, -0.25f};
    settings.depth_filter_max = {0.25f, 0.25f, 0.25f};
    rendering_manager_->updateSettings(settings);

    service_->setTestingScreenPositions(make_screen_positions({
        10.0f,
        10.0f,
        20.0f,
        20.0f,
    }));

    lfs::vis::SelectionFilterState filters;
    filters.depth_filter = true;

    ASSERT_TRUE(service_->beginInteractiveSelection(
        lfs::vis::SelectionShape::Rectangle,
        lfs::vis::SelectionMode::Replace,
        {0.0f, 0.0f},
        0.0f,
        filters));
    service_->updateInteractiveSelection({50.0f, 50.0f});

    const auto result = service_->finishInteractiveSelection();
    ASSERT_TRUE(result.success);
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionServiceInteractionsTest, PointCloudModeCommandSelectionUsesScreenPositionFallback) {
    auto settings = rendering_manager_->getSettings();
    settings.point_cloud_mode = true;
    rendering_manager_->updateSettings(settings);

    service_->setTestingScreenPositions(make_screen_positions({
        10.0f,
        10.0f,
        80.0f,
        80.0f,
    }));

    const auto result = service_->selectRect(
        0.0f, 0.0f, 50.0f, 50.0f, lfs::vis::SelectionMode::Replace, 0);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionServiceInteractionsTest, RingsCommitUsesHoveredGaussian) {
    service_->setTestingHoveredGaussianId(1);

    ASSERT_TRUE(service_->beginInteractiveSelection(
        lfs::vis::SelectionShape::Rings,
        lfs::vis::SelectionMode::Replace,
        {50.0f, 50.0f},
        0.0f));

    const auto result = service_->finishInteractiveSelection();
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.affected_count, 1u);
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 1}));
    EXPECT_FALSE(service_->isInteractiveSelectionActive());
}

TEST_F(SelectionServiceInteractionsTest, RingsCommitAppliesDepthFilter) {
    set_initial_selection({0, 0});

    auto settings = rendering_manager_->getSettings();
    settings.depth_filter_enabled = true;
    settings.depth_filter_min = {-0.25f, -0.25f, -0.25f};
    settings.depth_filter_max = {0.25f, 0.25f, 0.25f};
    rendering_manager_->updateSettings(settings);
    service_->setTestingHoveredGaussianId(1);

    lfs::vis::SelectionFilterState filters;
    filters.depth_filter = true;

    ASSERT_TRUE(service_->beginInteractiveSelection(
        lfs::vis::SelectionShape::Rings,
        lfs::vis::SelectionMode::Replace,
        {50.0f, 50.0f},
        0.0f,
        filters));

    const auto result = service_->finishInteractiveSelection();
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.affected_count, 0u);
    EXPECT_TRUE(selection_values(*scene_manager_).empty());
    EXPECT_FALSE(service_->isInteractiveSelectionActive());
}

TEST_F(SelectionServiceInteractionsTest, CommandRingSelectionUsesHoveredGaussianOverride) {
    service_->setTestingHoveredGaussianId(1);

    const auto result = service_->selectRing(50.0f, 50.0f, lfs::vis::SelectionMode::Replace, 0);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.affected_count, 1u);
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 1}));
}
