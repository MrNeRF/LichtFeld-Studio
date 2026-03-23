/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "operation/undo_history.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;
using lfs::core::events::cmd::ApplySelectionMask;
using lfs::core::events::cmd::SelectBrush;
using lfs::core::events::cmd::SelectLasso;
using lfs::core::events::cmd::SelectPolygon;
using lfs::core::events::cmd::SelectRect;
using lfs::core::events::cmd::SelectRing;

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

class SelectionCommandDispatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        lfs::vis::op::undoHistory().clear();

        rendering_manager_ = std::make_unique<lfs::vis::RenderingManager>();
        scene_manager_ = std::make_unique<lfs::vis::SceneManager>();
        lfs::vis::services().set(rendering_manager_.get());
        lfs::vis::services().set(scene_manager_.get());

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
        scene_manager_->initSelectionService();

        auto* const service = scene_manager_->getSelectionService();
        ASSERT_NE(service, nullptr);
        service->setTestingViewport({
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
        scene_manager_.reset();
        rendering_manager_.reset();
        lfs::vis::op::undoHistory().clear();
    }

    void set_initial_selection(const std::vector<uint8_t>& values) {
        scene_manager_->getScene().setSelectionMask(std::make_shared<Tensor>(make_uint8_mask(values)));
    }

    lfs::vis::SelectionService& service() {
        return *scene_manager_->getSelectionService();
    }

    std::unique_ptr<lfs::vis::RenderingManager> rendering_manager_;
    std::unique_ptr<lfs::vis::SceneManager> scene_manager_;
};

TEST_F(SelectionCommandDispatchTest, SelectRectCommandUsesCameraSpecificProjection) {
    service().setTestingScreenPositions(make_screen_positions({
        10.0f,
        10.0f,
        80.0f,
        80.0f,
    }));
    service().setTestingScreenPositionsForCamera(7, make_screen_positions({
                                                        80.0f,
                                                        80.0f,
                                                        10.0f,
                                                        10.0f,
                                                    }));

    SelectRect{
        .x0 = 0.0f,
        .y0 = 0.0f,
        .x1 = 30.0f,
        .y1 = 30.0f,
        .camera_index = 7,
        .mode = "replace",
    }
        .emit();

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 1}));
}

TEST_F(SelectionCommandDispatchTest, SelectBrushCommandsRespectReplaceAddAndRemoveModes) {
    service().setTestingScreenPositions(make_screen_positions({
        10.0f,
        10.0f,
        80.0f,
        80.0f,
    }));

    SelectBrush{.x = 10.0f, .y = 10.0f, .radius = 8.0f, .camera_index = 0, .mode = "replace"}.emit();
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));

    SelectBrush{.x = 80.0f, .y = 80.0f, .radius = 8.0f, .camera_index = 0, .mode = "add"}.emit();
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 1}));

    SelectBrush{.x = 10.0f, .y = 10.0f, .radius = 8.0f, .camera_index = 0, .mode = "remove"}.emit();
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 1}));
}

TEST_F(SelectionCommandDispatchTest, SelectPolygonCommandRoutesThroughSceneManager) {
    service().setTestingScreenPositions(make_screen_positions({
        80.0f,
        80.0f,
        10.0f,
        10.0f,
    }));

    SelectPolygon{
        .points = {0.0f, 0.0f, 30.0f, 0.0f, 0.0f, 30.0f},
        .camera_index = 0,
        .mode = "replace",
    }
        .emit();

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 1}));
}

TEST_F(SelectionCommandDispatchTest, SelectLassoCommandRoutesThroughSceneManager) {
    service().setTestingScreenPositions(make_screen_positions({
        80.0f,
        80.0f,
        10.0f,
        10.0f,
    }));

    SelectLasso{
        .points = {0.0f, 0.0f, 30.0f, 0.0f, 0.0f, 30.0f},
        .camera_index = 0,
        .mode = "replace",
    }
        .emit();

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 1}));
}

TEST_F(SelectionCommandDispatchTest, SelectRingCommandRoutesThroughSceneManager) {
    service().setTestingHoveredGaussianId(1);

    SelectRing{
        .x = 50.0f,
        .y = 50.0f,
        .camera_index = 0,
        .mode = "replace",
    }
        .emit();

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 1}));
}

TEST_F(SelectionCommandDispatchTest, SelectRectCommandRespectsDepthFilterSettings) {
    service().setTestingScreenPositions(make_screen_positions({
        10.0f,
        10.0f,
        20.0f,
        20.0f,
    }));

    auto settings = rendering_manager_->getSettings();
    settings.depth_filter_enabled = true;
    settings.depth_filter_min = {-0.5f, -10000.0f, -1.0f};
    settings.depth_filter_max = {0.5f, 10000.0f, 1.0f};
    rendering_manager_->updateSettings(settings);

    SelectRect{
        .x0 = 0.0f,
        .y0 = 0.0f,
        .x1 = 30.0f,
        .y1 = 30.0f,
        .camera_index = 0,
        .mode = "replace",
    }
        .emit();

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionCommandDispatchTest, SelectRectCommandRespectsCropFilterSettings) {
    service().setTestingScreenPositions(make_screen_positions({
        10.0f,
        10.0f,
        20.0f,
        20.0f,
    }));

    const auto splat_id = scene_manager_->getScene().getNodeIdByName("test");
    ASSERT_NE(splat_id, lfs::core::NULL_NODE);

    const auto crop_id = scene_manager_->getScene().addCropBox("test.crop", splat_id);
    lfs::core::CropBoxData crop_data;
    crop_data.min = {-0.5f, -0.5f, -0.5f};
    crop_data.max = {0.5f, 0.5f, 0.5f};
    crop_data.enabled = true;
    scene_manager_->getScene().setCropBoxData(crop_id, crop_data);

    auto settings = rendering_manager_->getSettings();
    settings.crop_filter_for_selection = true;
    rendering_manager_->updateSettings(settings);

    SelectRect{
        .x0 = 0.0f,
        .y0 = 0.0f,
        .x1 = 30.0f,
        .y1 = 30.0f,
        .camera_index = 0,
        .mode = "replace",
    }
        .emit();

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionCommandDispatchTest, ApplySelectionMaskCommandReplacesSelection) {
    set_initial_selection({1, 0});

    ApplySelectionMask{.mask = {0, 1}}.emit();

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 1}));
}
