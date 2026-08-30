/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/image_io.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "operation/undo_history.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "selection/selection_service.hpp"
#include <filesystem>

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

    // A commit that selects nothing may either store an all-zero mask or clear the
    // mask entirely; both mean "no splat selected".
    bool nothing_selected(const lfs::vis::SceneManager& scene_manager) {
        const auto values = selection_values(scene_manager);
        return std::all_of(values.begin(), values.end(), [](const uint8_t v) { return v == 0; });
    }

    std::vector<bool> deleted_values(const lfs::core::SplatData& splat) {
        if (!splat.has_deleted_mask()) {
            return {};
        }
        return splat.deleted().cpu().to_vector_bool();
    }

    void arm_viewer_camera_depth_band(lfs::vis::RenderingManager& rendering_manager) {
        auto settings = rendering_manager.getSettings();
        settings.depth_filter_enabled = true;
        // Camera-depth band [8.0, 8.875]: keeps splat0 (8.5442), rejects splat1 (9.2063).
        settings.depth_filter_min = {-0.5f, -0.5f, -8.875f};
        settings.depth_filter_max = {0.5f, 0.5f, -8.0f};
        // Full-viewport window so only depth can decide.
        settings.depth_filter_scale = 1.0f;
        settings.depth_filter_offset_x = 0.0f;
        settings.depth_filter_offset_y = 0.0f;
        rendering_manager.updateSettings(settings);
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

        scene_manager_->getScene().addSplat(
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

TEST_F(SelectionServiceInteractionsTest, SelectionAfterVisibilityChangeUsesRefreshedSelectedNodeMask) {
    const auto copy_id = scene_manager_->getScene().addSplat(
        "copy",
        make_test_splat({
            2.0f,
            0.0f,
            0.0f,
            3.0f,
            0.0f,
            0.0f,
        }));
    ASSERT_NE(copy_id, lfs::core::NULL_NODE);

    scene_manager_->selectNodes({"copy"});
    EXPECT_EQ(scene_manager_->getSelectedNodeMask(), (std::vector<bool>{false, true}));

    const auto original_id = scene_manager_->getScene().getNodeIdByName("test");
    ASSERT_NE(original_id, lfs::core::NULL_NODE);
    scene_manager_->setNodeVisibility(original_id, false);

    const auto result = service_->applyMask(std::vector<uint8_t>{1, 0}, lfs::vis::SelectionMode::Replace);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.affected_count, 1u);
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 0, 1, 0}));
}

TEST_F(SelectionServiceInteractionsTest, DeleteSelectedGaussiansMapsFullSelectionMaskAcrossHiddenNodes) {
    const auto copy_id = scene_manager_->getScene().addSplat(
        "copy",
        make_test_splat({
            2.0f,
            0.0f,
            0.0f,
            3.0f,
            0.0f,
            0.0f,
        }));
    ASSERT_NE(copy_id, lfs::core::NULL_NODE);

    const auto original_id = scene_manager_->getScene().getNodeIdByName("test");
    ASSERT_NE(original_id, lfs::core::NULL_NODE);
    scene_manager_->setNodeVisibility(original_id, false);
    scene_manager_->getScene().setSelectionMask(
        std::make_shared<Tensor>(make_uint8_mask({0, 0, 1, 0})));

    const auto result = scene_manager_->softDeleteSelectedGaussians();

    ASSERT_TRUE(result) << result.error();
    const auto* original = scene_manager_->getScene().getNodeById(original_id);
    const auto* copy = scene_manager_->getScene().getNodeById(copy_id);
    ASSERT_NE(original, nullptr);
    ASSERT_NE(copy, nullptr);
    ASSERT_NE(original->model, nullptr);
    ASSERT_NE(copy->model, nullptr);
    EXPECT_TRUE(deleted_values(*original->model).empty());
    EXPECT_EQ(deleted_values(*copy->model), (std::vector<bool>{true, false}));
}

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

TEST_F(SelectionServiceInteractionsTest, TestingScreenPositionsBrushFallbackWorksInPointCloudMode) {
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
    ASSERT_TRUE(result.success) << "error: " << result.error;
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionServiceInteractionsTest, TestingScreenPositionsRectangleFallbackWorksInPointCloudMode) {
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
    ASSERT_TRUE(result.success) << "error: " << result.error;
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionServiceInteractionsTest, TestingScreenPositionsPolygonFallbackWorksInPointCloudMode) {
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
    ASSERT_TRUE(result.success) << "error: " << result.error;
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionServiceInteractionsTest, TestingScreenPositionsFallbackAppliesDepthFilterInPointCloudMode) {
    auto settings = rendering_manager_->getSettings();
    settings.point_cloud_mode = true;
    rendering_manager_->updateSettings(settings);
    arm_viewer_camera_depth_band(*rendering_manager_);

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
    ASSERT_TRUE(result.success) << "error: " << result.error;
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionServiceInteractionsTest, TestingScreenPositionsCommandFallbackWorksInPointCloudMode) {
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
        0.0f, 0.0f, 50.0f, 50.0f, lfs::vis::SelectionMode::Replace, -1);
    ASSERT_TRUE(result.success) << "error: " << result.error;
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
    ASSERT_TRUE(result.success) << "error: " << result.error;
    EXPECT_EQ(result.affected_count, 1u);
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 1}));
    EXPECT_FALSE(service_->isInteractiveSelectionActive());
}

TEST_F(SelectionServiceInteractionsTest, RingsCommitAppliesDepthFilter) {
    set_initial_selection({0, 0});

    arm_viewer_camera_depth_band(*rendering_manager_);
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
    ASSERT_TRUE(result.success) << "error: " << result.error;
    EXPECT_EQ(result.affected_count, 0u);
    EXPECT_TRUE(selection_values(*scene_manager_).empty());
    EXPECT_FALSE(service_->isInteractiveSelectionActive());
}
// If the filter never ran, the negative would yield {0,1} and fail; if the filter
// rejected everything, the positive would yield {} and fail. Neither test alone
// excludes both wrong modes.
TEST_F(SelectionServiceInteractionsTest, RingsCommitKeepsGaussianInsideDepthBand) {
    set_initial_selection({0, 0});

    arm_viewer_camera_depth_band(*rendering_manager_);
    service_->setTestingHoveredGaussianId(0);

    lfs::vis::SelectionFilterState filters;
    filters.depth_filter = true;

    ASSERT_TRUE(service_->beginInteractiveSelection(
        lfs::vis::SelectionShape::Rings,
        lfs::vis::SelectionMode::Replace,
        {50.0f, 50.0f},
        0.0f,
        filters));

    const auto result = service_->finishInteractiveSelection();
    ASSERT_TRUE(result.success) << "error: " << result.error;
    EXPECT_EQ(result.affected_count, 1u);
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
    EXPECT_FALSE(service_->isInteractiveSelectionActive());
}

TEST_F(SelectionServiceInteractionsTest, CommandRingSelectionUsesHoveredGaussianOverride) {
    service_->setTestingHoveredGaussianId(1);

    const auto result = service_->selectRing(50.0f, 50.0f, lfs::vis::SelectionMode::Replace, -1);
    ASSERT_TRUE(result.success) << "error: " << result.error;
    EXPECT_EQ(result.affected_count, 1u);
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 1}));
}

// --- Command-camera mask/filter projection identity -----------------
//
// These two tests pin that a command selection builds its mask from the camera
// named by camera_index, while applyDepthFilter must use the same command
// camera — not the interactive-or-focused viewer viewport. Mask construction
// and depth filtering must therefore run against the SAME camera.
//
// Fixture geometry (both derived from source, not assumed):
//   * Viewer camera: the file-static testing viewport, t = (-5.657, 3, -5.657)
//     looking at the origin. Camera depths: splat0 = 8.5442, splat1 = 9.2063.
//   * Dataset camera below: world->camera R is the cyclic permutation mapping
//     world +X onto camera +Z, with T = (0, 0, 10). Camera depths:
//     splat0 (0,0,0) -> 10.0, splat1 (1,0,0) -> 11.0. Both project exactly to
//     the principal point, so the screen-window rect never co-decides.
//
// The depth band is encoded near = -max.z, far = -min.z. Band [9.5, 10.5]
// therefore keeps ONLY splat0 under the dataset camera, and rejects BOTH splats
// under the viewer camera. The three distinguishable outcomes are:
//   {1, 0} filter ran against the command camera (correct)
//   {}     filter ran against the wrong camera
//   {1, 1} filter did not run
// Neither test can pass vacuously.
namespace {

    std::shared_ptr<lfs::core::Camera> make_x_axis_dataset_camera(
        lfs::core::CameraModelType model = lfs::core::CameraModelType::PINHOLE,
        const std::string& image_name = "dataset.png") {
        // world->camera rotation: R * (1,0,0) = (0,0,1), det = +1.
        auto rotation = Tensor::from_vector(
            std::vector<float>{
                0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 1.0f,
                1.0f, 0.0f, 0.0f},
            {size_t{3}, size_t{3}}, Device::CPU);
        auto translation = Tensor::from_vector(std::vector<float>{0.0f, 0.0f, 10.0f}, {size_t{3}}, Device::CPU);
        return std::make_shared<lfs::core::Camera>(
            std::move(rotation),
            std::move(translation),
            100.0f, 100.0f, 50.0f, 50.0f,
            Tensor{}, Tensor{},
            model,
            image_name, std::filesystem::path{}, std::filesystem::path{},
            100, 100, 0);
    }

    void arm_dataset_camera_depth_band(lfs::vis::RenderingManager& rendering_manager) {
        auto settings = rendering_manager.getSettings();
        settings.depth_filter_enabled = true;
        // near = -max.z = 9.5, far = -min.z = 10.5
        settings.depth_filter_min = {-0.5f, -0.5f, -10.5f};
        settings.depth_filter_max = {0.5f, 0.5f, -9.5f};
        // Full-viewport window so only depth can decide.
        settings.depth_filter_scale = 1.0f;
        settings.depth_filter_offset_x = 0.0f;
        settings.depth_filter_offset_y = 0.0f;
        rendering_manager.updateSettings(settings);
    }

    std::shared_ptr<lfs::core::Camera> make_dataset_camera_with_center(const float center_x, const float center_y) {
        auto rotation = Tensor::from_vector(
            std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f},
            {size_t{3}, size_t{3}}, Device::CPU);
        auto translation = Tensor::from_vector(std::vector<float>{0.0f, 0.0f, 10.0f}, {size_t{3}}, Device::CPU);
        return std::make_shared<lfs::core::Camera>(
            std::move(rotation), std::move(translation),
            100.0f, 100.0f, center_x, center_y,
            Tensor{}, Tensor{}, lfs::core::CameraModelType::PINHOLE,
            "dataset.png", std::filesystem::path{}, std::filesystem::path{},
            100, 100, 0);
    }

    void arm_off_centre_depth_window(lfs::vis::RenderingManager& rendering_manager) {
        auto settings = rendering_manager.getSettings();
        settings.depth_filter_enabled = true;
        settings.depth_filter_min = {-0.5f, -0.5f, -10.5f};
        settings.depth_filter_max = {0.5f, 0.5f, -9.5f};
        settings.depth_filter_scale = 0.3f;
        settings.depth_filter_offset_x = 1.0f;
        settings.depth_filter_offset_y = 0.0f;
        rendering_manager.updateSettings(settings);
    }

    std::shared_ptr<lfs::core::Camera> make_asymmetric_dataset_camera() {
        auto rotation = Tensor::from_vector(
            std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f},
            {size_t{3}, size_t{3}}, Device::CPU);
        auto translation = Tensor::from_vector(std::vector<float>{0.0f, 0.0f, 10.0f}, {size_t{3}}, Device::CPU);
        return std::make_shared<lfs::core::Camera>(
            std::move(rotation), std::move(translation),
            500.0f, 600.0f, 320.0f, 240.0f,
            Tensor{}, Tensor{}, lfs::core::CameraModelType::PINHOLE,
            "asym.png", std::filesystem::path{}, std::filesystem::path{},
            640, 480, 0);
    }

    void arm_asymmetric_depth_window(lfs::vis::RenderingManager& rendering_manager) {
        auto settings = rendering_manager.getSettings();
        settings.depth_filter_enabled = true;
        settings.depth_filter_min = {-0.5f, -0.5f, -10.5f};
        settings.depth_filter_max = {0.5f, 0.5f, -9.5f};
        settings.depth_filter_scale = 0.35f;
        settings.depth_filter_offset_x = 0.35f;
        settings.depth_filter_offset_y = 0.0f;
        rendering_manager.updateSettings(settings);
    }

    // x-axis dataset camera (T=(0,0,10), cam_pos=(-10,0,0)): depth = p_x + 10.
    // On the optical axis with p_y > 0: view_x = view_y = 0, px = center_x, py = center_y.
    void install_m44_axis_depth_pair_fixture(lfs::vis::SceneManager& scene_manager) {
        // A bare SceneManager stays ContentType::Empty, and buildRenderState() only
        // assembles the combined model for SplatFiles.
        scene_manager.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
        scene_manager.getScene().removeNode("test");
        // Convention for the x-axis dataset cameras in this file (R rows {0,1,0;0,0,1;1,0,0},
        // T=(0,0,10)), verified empirically against the real projector + depth filter:
        //   depth = p_x + 10;  px = center_x + f*(p_y/depth);  py = center_y - f*(p_z/depth)
        // splat0 = origin: depth 10 (inside band [9.5,10.5]), on-axis (px = center_x).
        // splat1 = (1.5,0,0): depth 11.5 — band-rejected.
        scene_manager.getScene().addSplat("m44_axis_depth", make_test_splat({0.0f, 0.0f, 0.0f,
                                                                             1.5f, 0.0f, 0.0f}));
    }

    void install_m44_resized_viewport_fixture(lfs::vis::SceneManager& scene_manager) {
        // A bare SceneManager stays ContentType::Empty, and buildRenderState() only
        // assembles the combined model for SplatFiles.
        scene_manager.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
        scene_manager.getScene().removeNode("test");
        // Convention for the x-axis dataset cameras in this file (R rows {0,1,0;0,0,1;1,0,0},
        // T=(0,0,10)), verified empirically against the real projector + depth filter:
        //   depth = p_x + 10;  px = center_x + f*(p_y/depth);  py = center_y - f*(p_z/depth)
        // Camera is CENTRED (cx=50) at viewport 100; correctly scaled intrinsics give
        // splat0 (0,3,0): depth 10, px = 50+100*0.3 = 80, py = 50 — inside the armed
        // window x-range [70,100] and the [70,100]x[35,65] rect. Wrongly image-scaled
        // intrinsics (cx=2, f=4) would put it at px = 2+4*0.3 = 3.2 — inside the [0,10]
        // control rect, which must select nothing.
        // splat1 (1.5,3,0): depth 11.5 — band-rejected.
        scene_manager.getScene().addSplat("m44_resized", make_test_splat({0.0f, 3.0f, 0.0f,
                                                                          1.5f, 3.0f, 0.0f}));
    }

    void install_m44_asymmetric_fixture(lfs::vis::SceneManager& scene_manager) {
        // A bare SceneManager stays ContentType::Empty, and buildRenderState() only
        // assembles the combined model for SplatFiles.
        scene_manager.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
        scene_manager.getScene().removeNode("test");
        // Convention for the x-axis dataset cameras in this file (R rows {0,1,0;0,0,1;1,0,0},
        // T=(0,0,10)), verified empirically against the real projector + depth filter:
        //   depth = p_x + 10;  px = center_x + f*(p_y/depth);  py = center_y - f*(p_z/depth)
        // Band here is [10.5,11.5]. splat0 (2.5,0,0): depth 12.5 — band-rejected.
        // splat1 (1,3.96,0): depth 11, u = 3.96/11 = 0.36 -> px = 320+500*0.36 = 500,
        // inside the window x-range [280.8,504.8] with the CARRIED fx=500; an
        // aspect-reconstructed fx=600 would give px = 536 — outside. py = 240.
        scene_manager.getScene().addSplat("m44_asym", make_test_splat({2.5f, 0.0f, 0.0f,
                                                                       1.0f, 3.96f, 0.0f}));
    }

    // Viewer orbit camera: the world origin projects exactly onto the containment
    // principal point (verified empirically), so px equals the supplied center_x.
    void install_m44_viewer_containment_fixture(lfs::vis::SceneManager& scene_manager) {
        // A bare SceneManager stays ContentType::Empty, and buildRenderState() only
        // assembles the combined model for SplatFiles — without this the real screen-
        // position projector sees no renderable gaussians.
        scene_manager.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
        scene_manager.getScene().removeNode("test");
        scene_manager.getScene().addSplat(
            "m44_viewer_containment",
            make_test_splat({0.0f, 0.0f, 0.0f,
                             0.0f, 5.0f, 0.0f}));
    }

    // Same x-axis camera as install_m44_axis_depth_pair_fixture, but splat1 is
    // off-axis so a hover pick at the optical-axis principal point cannot
    // resolve it. splat0 origin: depth 10, px = 50, py = 50. splat1 (1.5, 5, 0):
    // depth 11.5 (band-rejected), px = 50 + 100*(5/11.5) ≈ 93.5 — well outside
    // HOVER_PICK_RADIUS_PX of (50, 50).
    void install_m44_color_seed_fixture(lfs::vis::SceneManager& scene_manager) {
        scene_manager.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
        scene_manager.getScene().removeNode("test");
        scene_manager.getScene().addSplat(
            "m44_color_seed",
            make_test_splat({0.0f, 0.0f, 0.0f,
                             1.5f, 5.0f, 0.0f}));
    }

    // x-axis camera at cam_pos=(-10,0,0) looking +X. splat0 (-20,0,0) sits
    // directly behind the camera (visualizer view_z = +10): pinhole rejects
    // (vksplat depth = -view_z < 0), equirect maps it to the image edge with
    // depth = 10. splat1 (1000,0,0) is in front at depth 1010, outside a
    // (0, 100] band under both models.
    void install_equirect_behind_camera_fixture(lfs::vis::SceneManager& scene_manager) {
        scene_manager.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
        scene_manager.getScene().removeNode("test");
        scene_manager.getScene().addSplat(
            "equirect_behind",
            make_test_splat({-20.0f, 0.0f, 0.0f,
                             1000.0f, 0.0f, 0.0f}));
    }

    void arm_wide_full_frame_depth_window(lfs::vis::RenderingManager& rendering_manager) {
        auto settings = rendering_manager.getSettings();
        settings.depth_filter_enabled = true;
        settings.depth_filter_min = {-0.5f, -0.5f, -100.0f};
        settings.depth_filter_max = {0.5f, 0.5f, 0.0f};
        settings.depth_filter_scale = 1.0f;
        settings.depth_filter_offset_x = 0.0f;
        settings.depth_filter_offset_y = 0.0f;
        rendering_manager.updateSettings(settings);
    }

    void starve_testing_viewport(lfs::vis::SelectionService& service) {
        service.setTestingViewport({
            .x = 0.0f,
            .y = 0.0f,
            .width = 0.0f,
            .height = 0.0f,
            .render_width = 0,
            .render_height = 0,
        });
    }

} // namespace

TEST_F(SelectionServiceInteractionsTest, ExplicitCameraRectFiltersWithTheCommandCameraNotTheViewer) {
    install_m44_axis_depth_pair_fixture(*scene_manager_);
    const auto cameras_group = scene_manager_->getScene().addGroup("Cameras");
    scene_manager_->getScene().addCamera("dataset.png", cameras_group, make_x_axis_dataset_camera());
    ASSERT_EQ(scene_manager_->getScene().getAllCameras().size(), 1u);

    arm_dataset_camera_depth_band(*rendering_manager_);

    // Real projector, no screen-position stub. Both splats sit on the optical
    // axis so they share px = center_x = 50, py = 50 and fall inside this rect;
    // only the depth band can discriminate. splat0 origin: depth 10 (inside
    // [9.5, 10.5]). splat1 (1.5, 0, 0): depth 11.5 — band-rejected.
    const auto result = service_->selectRect(40.0f, 35.0f, 60.0f, 65.0f, lfs::vis::SelectionMode::Replace, 0);
    ASSERT_TRUE(result.success) << "error: " << result.error;

    // Depth filtering must use camera 0 (band keeps splat0 only), NOT the
    // viewer camera (which rejects both and yields {}).
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionServiceInteractionsTest, ExplicitCameraColorSeedFiltersWithTheCommandCamera) {
    install_m44_color_seed_fixture(*scene_manager_);
    const auto cameras_group = scene_manager_->getScene().addGroup("Cameras");
    scene_manager_->getScene().addCamera("dataset.png", cameras_group, make_x_axis_dataset_camera());
    ASSERT_EQ(scene_manager_->getScene().getAllCameras().size(), 1u);

    arm_dataset_camera_depth_band(*rendering_manager_);

    // Deliberately NOT setTestingHoveredGaussianId: the override short-circuits
    // resolveCommandHoveredGaussianId and would bypass the seed-camera identity
    // this test exists to prove. No command-camera screen-position stub either:
    // the seed is picked from the real projector. splat0 origin projects to the
    // principal point (50, 50); splat1 is off-axis so it cannot steal the pick.
    // camera_index >= 0 never consults the viewer-side testing_screen_positions_
    // hook, so that stub is omitted too.

    // CONTROL: with filtering off, the seed resolves and the colour selection
    // succeeds. This is what makes the assertion below non-vacuous - it proves
    // the fixture can seed, so a later failure is the FILTER's doing.
    {
        lfs::vis::SelectionFilterState unfiltered;
        const auto control =
            service_->selectByColorAt(50.0f, 50.0f, lfs::vis::SelectionMode::Replace, unfiltered, 0);
        ASSERT_TRUE(control.success) << "fixture cannot seed at all: " << control.error;
        ASSERT_FALSE(selection_values(*scene_manager_).empty());
    }

    lfs::vis::SelectionFilterState filters;
    filters.depth_filter = true;

    // Colour selection applies filters during SEED PICKING (applyFilters inside
    // pickHoveredGaussianIdFromScreenPositions), BEFORE commitSelection — which
    // is why a rect-only repair would leave this path wrong.
    // Under the command camera the band accepts splat0, so the seed must survive.
    const auto result = service_->selectByColorAt(50.0f, 50.0f, lfs::vis::SelectionMode::Replace, filters, 0);
    ASSERT_TRUE(result.success) << "seed rejected by a filter using the wrong camera: " << result.error;

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionServiceInteractionsTest, ExplicitCameraIndexOutOfRangeHardFailsWithoutOverrides) {
    set_initial_selection({1, 0});

    const auto result = service_->selectRect(
        0.0f, 0.0f, 50.0f, 50.0f, lfs::vis::SelectionMode::Replace, 99);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.affected_count, 0u);
    EXPECT_EQ(result.error, "Camera index out of range");
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionServiceInteractionsTest, ArmedOverrideWithInvalidCameraFailsInsteadOfFallingBackToViewer) {
    // An armed test switch must not silently move projection onto the viewer camera.
    auto rotation = Tensor::from_vector(
        std::vector<float>{
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f},
        {size_t{3}, size_t{3}}, Device::CPU);
    auto translation = Tensor::from_vector(std::vector<float>{0.0f, 0.0f, 10.0f}, {size_t{3}}, Device::CPU);
    const auto zero_dimension_camera = std::make_shared<lfs::core::Camera>(
        std::move(rotation),
        std::move(translation),
        100.0f, 100.0f, 50.0f, 50.0f,
        Tensor{}, Tensor{},
        lfs::core::CameraModelType::PINHOLE,
        "zero_dim.png", std::filesystem::path{}, std::filesystem::path{},
        0, 0, 0);

    const auto cameras_group = scene_manager_->getScene().addGroup("Cameras");
    scene_manager_->getScene().addCamera("zero_dim.png", cameras_group, zero_dimension_camera);
    ASSERT_EQ(scene_manager_->getScene().getAllCameras().size(), 1u);

    set_initial_selection({1, 0});
    service_->setTestingHoveredGaussianId(1);

    const auto result = service_->selectRing(50.0f, 50.0f, lfs::vis::SelectionMode::Replace, 0);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.error, "Camera projection unavailable");
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionServiceInteractionsTest, ExplicitCameraOffCentrePrincipalPointSurvivesTheDepthFilter) {
    install_m44_axis_depth_pair_fixture(*scene_manager_);
    const auto cameras_group = scene_manager_->getScene().addGroup("Cameras");
    scene_manager_->getScene().addCamera("dataset.png", cameras_group,
                                         make_dataset_camera_with_center(80.0f, 50.0f));
    arm_off_centre_depth_window(*rendering_manager_);

    // splat0 = origin: on the optical axis, so px = center_x = 80 - inside BOTH the
    // discriminating rectangle [70,100]x[35,65] (which a hard-centred shape projector
    // would miss at px = 50) and the armed window x-range [70,100]. splat1: depth 11.5,
    // band-rejected. The rectangle, not just the window, must discriminate so a broken
    // shape lane cannot pass on the depth filter alone.
    const auto selected = service_->selectRect(70.0f, 35.0f, 100.0f, 65.0f, lfs::vis::SelectionMode::Replace, 0);
    ASSERT_TRUE(selected.success) << "error: " << selected.error;
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));

    // Control: image-centred camera puts the same splat at px = 50 - outside [70,100].
    scene_manager_->getScene().addCamera("dataset_centre.png", cameras_group,
                                         make_dataset_camera_with_center(50.0f, 50.0f));
    const auto control = service_->selectRect(70.0f, 35.0f, 100.0f, 65.0f, lfs::vis::SelectionMode::Replace, 1);
    ASSERT_TRUE(control.success) << "error: " << control.error;
    EXPECT_TRUE(nothing_selected(*scene_manager_));
}

TEST_F(SelectionServiceInteractionsTest, ExplicitCameraResizedDatasetScalesIntrinsicsToTheSelectionViewport) {
    install_m44_resized_viewport_fixture(*scene_manager_);
    auto rotation = Tensor::from_vector(
        std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f},
        {size_t{3}, size_t{3}}, Device::CPU);
    auto translation = Tensor::from_vector(std::vector<float>{0.0f, 0.0f, 10.0f}, {size_t{3}}, Device::CPU);
    auto camera = std::make_shared<lfs::core::Camera>(
        std::move(rotation), std::move(translation),
        100.0f, 100.0f, 50.0f, 50.0f,
        Tensor{}, Tensor{}, lfs::core::CameraModelType::PINHOLE,
        "resize.png", std::filesystem::path{}, std::filesystem::path{},
        100, 100, 0);
    camera->set_image_dimensions(4, 4);

    const auto cameras_group = scene_manager_->getScene().addGroup("Cameras");
    scene_manager_->getScene().addCamera("resize.png", cameras_group, camera);
    arm_off_centre_depth_window(*rendering_manager_);

    // Correctly scaled intrinsics (to max(image_*, camera_*) = 100): splat0 (0,3,0)
    // projects to px = 50+100*0.3 = 80, py = 50 - inside [70,100]x[35,65] and the window.
    const auto scaled = service_->selectRect(70.0f, 35.0f, 100.0f, 65.0f, lfs::vis::SelectionMode::Replace, 0);
    ASSERT_TRUE(scaled.success) << "error: " << scaled.error;
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));

    // Wrongly image-scaled intrinsics (cx=2, f=4) would land it at px = 3.2 - inside
    // this control rect, which must therefore select nothing.
    const auto halved = service_->selectRect(0.0f, 0.0f, 10.0f, 10.0f, lfs::vis::SelectionMode::Replace, 0);
    ASSERT_TRUE(halved.success) << "error: " << halved.error;
    EXPECT_TRUE(nothing_selected(*scene_manager_));
}

TEST_F(SelectionServiceInteractionsTest, ExplicitCameraAsymmetricFocalsFilterThroughApplyDepthFilter) {
    install_m44_asymmetric_fixture(*scene_manager_);
    const auto cameras_group = scene_manager_->getScene().addGroup("Cameras");
    scene_manager_->getScene().addCamera("asym.png", cameras_group, make_asymmetric_dataset_camera());
    auto settings = rendering_manager_->getSettings();
    settings.depth_filter_enabled = true;
    settings.depth_filter_min = {-0.5f, -0.5f, -11.5f};
    settings.depth_filter_max = {0.5f, 0.5f, -10.5f};
    settings.depth_filter_scale = 0.35f;
    settings.depth_filter_offset_x = 0.35f;
    settings.depth_filter_offset_y = 0.0f;
    rendering_manager_->updateSettings(settings);

    // splat1 (1,3.96,0): depth 11, px = 320+500*0.36 = 500 with the CARRIED fx=500 -
    // inside the window x-range [280.8,504.8]; an aspect-reconstructed fx=600 would give
    // px = 536, outside. splat0: depth 12.5, band-rejected.
    const auto result = service_->selectRect(0.0f, 0.0f, 640.0f, 480.0f, lfs::vis::SelectionMode::Replace, 0);
    ASSERT_TRUE(result.success) << "error: " << result.error;
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 1}));
}

TEST_F(SelectionServiceInteractionsTest, ScreenPositionCacheIsNotSharedAcrossContainmentIntrinsics) {
    install_m44_viewer_containment_fixture(*scene_manager_);
    auto settings = rendering_manager_->getSettings();
    settings.depth_filter_enabled = false;
    rendering_manager_->updateSettings(settings);

    const lfs::rendering::CameraIntrinsics intrinsics_a{
        .focal_x = 100.0f,
        .focal_y = 100.0f,
        .center_x = 40.0f,
        .center_y = 50.0f};
    const lfs::rendering::CameraIntrinsics intrinsics_b{
        .focal_x = 100.0f,
        .focal_y = 100.0f,
        .center_x = 70.0f,
        .center_y = 50.0f};

    // The default viewer camera projects the world origin exactly onto the
    // containment principal point (verified empirically), so the rect around
    // px=40 catches splat0 under A (cx=40) and misses it under B (cx=70). A
    // stale screen-position cache from A would keep selecting it under B.
    service_->setTestingContainmentIntrinsics(intrinsics_a);
    const auto first = service_->selectRect(35.0f, 40.0f, 45.0f, 60.0f, lfs::vis::SelectionMode::Replace);
    ASSERT_TRUE(first.success) << "error: " << first.error;
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));

    service_->setTestingContainmentIntrinsics(intrinsics_b);
    const auto second = service_->selectRect(35.0f, 40.0f, 45.0f, 60.0f, lfs::vis::SelectionMode::Replace);
    ASSERT_TRUE(second.success) << "error: " << second.error;
    EXPECT_TRUE(nothing_selected(*scene_manager_));
    service_->setTestingContainmentIntrinsics(std::nullopt);
}

TEST_F(SelectionServiceInteractionsTest, ExplicitCameraEquirectangularKeepsSplatBehindTheCamera) {
    install_equirect_behind_camera_fixture(*scene_manager_);
    const auto cameras_group = scene_manager_->getScene().addGroup("Cameras");
    scene_manager_->getScene().addCamera(
        "equirect.png", cameras_group,
        make_x_axis_dataset_camera(lfs::core::CameraModelType::EQUIRECTANGULAR, "equirect.png"));
    ASSERT_EQ(scene_manager_->getScene().getAllCameras().size(), 1u);

    // Viewer stays pinhole: leaking settings.equirectangular into the explicit-
    // camera snapshot would classify this as pinhole and drop splat0.
    ASSERT_FALSE(rendering_manager_->getSettings().equirectangular);
    arm_wide_full_frame_depth_window(*rendering_manager_);

    // Real projector, no screen-position stub. Splat0 is directly behind the
    // camera: vis = (0, 0, +10), vk = (0, 0, -10),
    // px = (atan2(0, -1)/(2π) + 0.5)*100 = 100, py = 50 — the image edge.
    // Splat1 is in front on-axis at px = 50, py = 50, depth 1010. A full-frame
    // rect includes both so only the depth band can discriminate:
    // {1, 0} equirect kept the behind-camera splat and band-rejected the far one
    // {}     leaked the viewer's pinhole (behind-camera z-reject)
    // {1, 1} depth filter did not run
    const auto result = service_->selectRect(0.0f, 0.0f, 100.0f, 100.0f, lfs::vis::SelectionMode::Replace, 0);
    ASSERT_TRUE(result.success) << "error: " << result.error;
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionServiceInteractionsTest, FilteredSelectAllAndInvertRequireProjectionOnlyWhenDepthFilterOn) {
    starve_testing_viewport(*service_);

    ASSERT_FALSE(rendering_manager_->getSettings().depth_filter_enabled);

    const auto copy_id = scene_manager_->getScene().addSplat(
        "copy",
        make_test_splat({
            2.0f,
            0.0f,
            0.0f,
            3.0f,
            0.0f,
            0.0f,
        }));
    ASSERT_NE(copy_id, lfs::core::NULL_NODE);
    scene_manager_->selectNodes({"copy"});
    EXPECT_EQ(scene_manager_->getSelectedNodeMask(), (std::vector<bool>{false, true}));

    const auto all = service_->selectAllFiltered();
    ASSERT_TRUE(all.success) << "error: " << all.error;
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 0, 1, 1}));

    set_initial_selection({0, 0, 1, 0});
    const auto inverted = service_->invertFiltered();
    ASSERT_TRUE(inverted.success) << "error: " << inverted.error;
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 0, 0, 1}));

    auto settings = rendering_manager_->getSettings();
    settings.depth_filter_enabled = true;
    rendering_manager_->updateSettings(settings);

    const auto all_fail = service_->selectAllFiltered();
    EXPECT_FALSE(all_fail.success);
    EXPECT_EQ(all_fail.error, "Invalid projection context");

    const auto invert_fail = service_->invertFiltered();
    EXPECT_FALSE(invert_fail.success);
    EXPECT_EQ(invert_fail.error, "Invalid projection context");
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 0, 0, 1}));
}
