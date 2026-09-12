/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "input/input_controller.hpp"
#include "input/key_codes.hpp"
#include "internal/viewport.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_context.hpp"
#include "operator/operator_properties.hpp"
#include "operator/operator_registry.hpp"
#include "operator/ops/depth_window_ops.hpp"
#include "operator/ops/selection_ops.hpp"
#include "rendering/render_constants.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "selection/depth_window_geometry.hpp"
#include "selection/selection_service.hpp"
#include "tools/selection_tool.hpp"
#include "tools/tool_base.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer_impl.hpp"

#include <algorithm>
#include <glm/glm.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;
using lfs::vis::KeyEvent;
using lfs::vis::MouseButtonEvent;
using lfs::vis::MouseMoveEvent;
using lfs::vis::MouseScrollEvent;
using lfs::vis::op::ActionEvent;
using lfs::vis::op::ModalEvent;
using lfs::vis::op::OperatorContext;
using lfs::vis::op::OperatorProperties;
using lfs::vis::op::OperatorResult;
using lfs::vis::op::SelectionStrokeOperator;

namespace {

    Tensor make_uint8_mask(const std::vector<uint8_t>& values) {
        auto tensor = Tensor::empty({values.size()}, Device::CPU, DataType::UInt8);
        std::copy(values.begin(), values.end(), tensor.ptr<uint8_t>());
        return tensor.gpu();
    }

    std::shared_ptr<Tensor> make_screen_positions(const std::vector<float>& xy) {
        return std::make_shared<Tensor>(
            Tensor::from_vector(xy, {xy.size() / 2, size_t{2}}, Device::GPU).to(DataType::Float32));
    }

    std::unique_ptr<lfs::core::SplatData> make_test_splat(const std::vector<float>& xyz) {
        const size_t count = xyz.size() / 3;
        auto means = Tensor::from_vector(xyz, {count, size_t{3}}, Device::GPU).to(DataType::Float32);
        auto sh0 = Tensor::zeros({count, size_t{1}, size_t{3}}, Device::GPU, DataType::Float32);
        auto shN = Tensor::zeros({count, size_t{3}, size_t{3}}, Device::GPU, DataType::Float32);
        auto scaling = Tensor::zeros({count, size_t{3}}, Device::GPU, DataType::Float32);

        std::vector<float> rotation_data(count * 4, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            rotation_data[i * 4] = 1.0f;
        }
        auto rotation = Tensor::from_vector(rotation_data, {count, size_t{4}}, Device::GPU).to(DataType::Float32);
        auto opacity = Tensor::zeros({count, size_t{1}}, Device::GPU, DataType::Float32);

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

    void arm_viewer_camera_depth_band(lfs::vis::RenderingManager& rendering_manager) {
        auto settings = rendering_manager.getSettings();
        settings.depth_filter_enabled = true;
        // Camera-depth band [8.0, 8.875]: keeps splat0 (8.5442), rejects splat1 (9.2063).
        settings.depth_filter_min = {-0.5f, -0.5f, -8.875f};
        settings.depth_filter_max = {0.5f, 0.5f, -8.0f};
        // Full-viewport window so only depth can decide.
        settings.depth_filter_scale_x = 1.0f;
        settings.depth_filter_scale_y = 1.0f;
        settings.depth_filter_offset_x = 0.0f;
        settings.depth_filter_offset_y = 0.0f;
        rendering_manager.updateSettings(settings);
    }

    ModalEvent mouse_move(const double x, const double y, const double dx = 0.0, const double dy = 0.0) {
        return ModalEvent{
            .type = ModalEvent::Type::MOUSE_MOVE,
            .data = MouseMoveEvent{
                .position = {x, y},
                .delta = {dx, dy},
            },
        };
    }

    ModalEvent mouse_button(const int button, const int action, const double x, const double y, const int mods = 0) {
        return ModalEvent{
            .type = ModalEvent::Type::MOUSE_BUTTON,
            .data = MouseButtonEvent{
                .button = button,
                .action = action,
                .mods = mods,
                .position = {x, y},
            },
        };
    }

    ModalEvent mouse_scroll(const double xoffset = 0.0, const double yoffset = 1.0) {
        return ModalEvent{
            .type = ModalEvent::Type::MOUSE_SCROLL,
            .data = MouseScrollEvent{
                .xoffset = xoffset,
                .yoffset = yoffset,
            },
        };
    }

    ModalEvent key_press(const int key, const int mods = 0) {
        return ModalEvent{
            .type = ModalEvent::Type::KEY,
            .data = KeyEvent{
                .key = key,
                .scancode = 0,
                .action = lfs::vis::input::ACTION_PRESS,
                .mods = mods,
            },
        };
    }

    ModalEvent modal_action(const lfs::vis::input::Action action, const int mods = 0,
                            const double x = 0.0, const double y = 0.0) {
        return ModalEvent{
            .type = ModalEvent::Type::ACTION,
            .data = ActionEvent{
                .action = action,
                .mods = mods,
                .position = {x, y},
            },
        };
    }

} // namespace

class SelectionOperatorModalTest : public ::testing::Test {
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

        context_ = std::make_unique<OperatorContext>(*scene_manager_);
    }

    void TearDown() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        context_.reset();
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

    OperatorResult dispatch(SelectionStrokeOperator& op, const ModalEvent& event, OperatorProperties& props) {
        context_->setModalEvent(event);
        return op.modal(*context_, props);
    }

    std::unique_ptr<lfs::vis::RenderingManager> rendering_manager_;
    std::unique_ptr<lfs::vis::SceneManager> scene_manager_;
    std::unique_ptr<OperatorContext> context_;
};

TEST_F(SelectionOperatorModalTest, PolygonOperatorUndoesOnlyFromBoundActionAndUnboundKeysPassThrough) {
    set_initial_selection({1, 0});
    service().setTestingScreenPositions(make_screen_positions({
        80.0f,
        80.0f,
        10.0f,
        10.0f,
    }));

    SelectionStrokeOperator op;
    OperatorProperties props;
    props.set("mode", 2);
    props.set("op", 0);
    props.set("x", 0.0);
    props.set("y", 0.0);

    EXPECT_EQ(op.invoke(*context_, props), OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                                    lfs::vis::input::ACTION_PRESS, 30.0, 0.0),
                       props),
              OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::RIGHT),
                                    lfs::vis::input::ACTION_PRESS, 30.0, 0.0),
                       props),
              OperatorResult::PASS_THROUGH);
    EXPECT_EQ(dispatch(op,
                       modal_action(lfs::vis::input::Action::UNDO_POLYGON_VERTEX),
                       props),
              OperatorResult::RUNNING_MODAL);
    EXPECT_TRUE(service().isInteractiveSelectionActive());

    EXPECT_EQ(dispatch(op, key_press(lfs::vis::input::KEY_ESCAPE), props), OperatorResult::PASS_THROUGH);
    op.cancel(*context_);

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
    EXPECT_FALSE(service().isInteractiveSelectionActive());
}

TEST_F(SelectionOperatorModalTest, PolygonOperatorUsesConfiguredRightButtonForVertices) {
    set_initial_selection({1, 0});
    service().setTestingScreenPositions(make_screen_positions({
        80.0f,
        80.0f,
        10.0f,
        10.0f,
    }));

    SelectionStrokeOperator op;
    OperatorProperties props;
    props.set("mode", 2);
    props.set("op", 0);
    props.set("x", 0.0);
    props.set("y", 0.0);
    props.set("button", static_cast<int>(lfs::vis::input::AppMouseButton::RIGHT));

    EXPECT_EQ(op.invoke(*context_, props), OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::RIGHT),
                                    lfs::vis::input::ACTION_PRESS, 30.0, 0.0),
                       props),
              OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::RIGHT),
                                    lfs::vis::input::ACTION_PRESS, 0.0, 30.0),
                       props),
              OperatorResult::RUNNING_MODAL);
    EXPECT_TRUE(service().isInteractiveSelectionActive());
    EXPECT_FALSE(selection_values(*scene_manager_).empty());

    op.cancel(*context_);
    EXPECT_FALSE(service().isInteractiveSelectionActive());
}

TEST_F(SelectionOperatorModalTest, ColorSelectionAppliesDepthFilterToSimilarityMask) {
    arm_viewer_camera_depth_band(*rendering_manager_);

    service().setTestingHoveredGaussianId(0);

    SelectionStrokeOperator op;
    OperatorProperties props;
    props.set("mode", 5);
    props.set("op", 0);
    props.set("x", 0.0);
    props.set("y", 0.0);
    props.set("use_depth_filter", true);

    EXPECT_EQ(op.invoke(*context_, props), OperatorResult::FINISHED);
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionOperatorModalTest, DepthFilterDoesNotOverrideGaussianRenderMode) {
    auto settings = rendering_manager_->getSettings();
    settings.point_cloud_mode = true;
    settings.show_rings = true;
    settings.show_center_markers = false;
    rendering_manager_->updateSettings(settings);

    Viewport viewport(100, 100);
    lfs::vis::ToolContext tool_context(rendering_manager_.get(), scene_manager_.get(), &viewport, nullptr);
    tool_context.updateViewportBounds(0.0f, 0.0f, 100.0f, 100.0f);

    lfs::vis::tools::SelectionTool tool;
    ASSERT_TRUE(tool.initialize(tool_context));
    tool.setEnabled(true);
    tool.setDepthFilterRange(true, 0.0f, 15.0f, 1.35f);

    auto enabled_settings = rendering_manager_->getSettings();
    EXPECT_TRUE(enabled_settings.depth_filter_enabled);
    EXPECT_TRUE(enabled_settings.point_cloud_mode);
    EXPECT_TRUE(enabled_settings.show_rings);
    EXPECT_FALSE(enabled_settings.show_center_markers);

    tool.setDepthFilterEnabled(false);

    auto disabled_settings = rendering_manager_->getSettings();
    EXPECT_FALSE(disabled_settings.depth_filter_enabled);
    EXPECT_TRUE(disabled_settings.point_cloud_mode);
    EXPECT_TRUE(disabled_settings.show_rings);
    EXPECT_FALSE(disabled_settings.show_center_markers);
}

TEST_F(SelectionOperatorModalTest, ClosedPolygonVertexDragConsumesMouseMoveUntilRelease) {
    set_initial_selection({1, 0});

    SelectionStrokeOperator op;
    OperatorProperties props;
    props.set("mode", 2);
    props.set("op", 0);
    props.set("x", 0.0);
    props.set("y", 0.0);

    EXPECT_EQ(op.invoke(*context_, props), OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                                    lfs::vis::input::ACTION_PRESS, 30.0, 0.0),
                       props),
              OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                                    lfs::vis::input::ACTION_PRESS, 0.0, 30.0),
                       props),
              OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                                    lfs::vis::input::ACTION_PRESS, 0.0, 0.0),
                       props),
              OperatorResult::RUNNING_MODAL);
    EXPECT_TRUE(service().isInteractiveSelectionClosed());

    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                                    lfs::vis::input::ACTION_PRESS, 30.0, 0.0),
                       props),
              OperatorResult::RUNNING_MODAL);
    EXPECT_TRUE(service().isInteractivePolygonVertexDragActive());
    EXPECT_EQ(dispatch(op, mouse_move(40.0, 0.0, 10.0, 0.0), props), OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                                    lfs::vis::input::ACTION_RELEASE, 40.0, 0.0),
                       props),
              OperatorResult::RUNNING_MODAL);
    EXPECT_FALSE(service().isInteractivePolygonVertexDragActive());
}

TEST_F(SelectionOperatorModalTest, DepthFilterExtentsUseIndependentScaleAxes) {
    Viewport viewport(100, 100);
    lfs::vis::ToolContext tool_context(rendering_manager_.get(), scene_manager_.get(), &viewport, nullptr);
    tool_context.updateViewportBounds(0.0f, 0.0f, 100.0f, 100.0f);

    constexpr float k_scale_x = 0.4f;
    constexpr float k_scale_y = 0.7f;
    auto settings = rendering_manager_->getSettings();
    settings.depth_filter_scale_x = k_scale_x;
    settings.depth_filter_scale_y = k_scale_y;
    rendering_manager_->updateSettings(settings);

    lfs::vis::tools::SelectionTool tool;
    ASSERT_TRUE(tool.initialize(tool_context));
    tool.setDepthFilterRange(false, 0.0f, 15.0f, 0.0f);
    tool.setEnabled(true);
    tool.setDepthFilterEnabled(true);

    const auto updated = rendering_manager_->getSettings();
    const auto [pixel_focal_x, pixel_focal_y] =
        lfs::rendering::computePixelFocalLengths(glm::ivec2(100, 100), updated.focal_length_mm);
    const float half_w_pixels = 0.5f * k_scale_x * 100.0f;
    const float half_h_pixels = 0.5f * k_scale_y * 100.0f;
    const float expected_half_x = half_w_pixels * 15.0f / pixel_focal_x;
    const float expected_half_y = half_h_pixels * 15.0f / pixel_focal_y;

    EXPECT_NEAR(updated.depth_filter_min.x, -expected_half_x, 1.0e-5f);
    EXPECT_NEAR(updated.depth_filter_max.x, expected_half_x, 1.0e-5f);
    EXPECT_NEAR(updated.depth_filter_min.y, -expected_half_y, 1.0e-5f);
    EXPECT_NEAR(updated.depth_filter_max.y, expected_half_y, 1.0e-5f);
    EXPECT_NE(expected_half_x, expected_half_y);
}
TEST_F(SelectionOperatorModalTest, ClosedPolygonShiftAddsVertexAndCtrlRemovesVertex) {
    set_initial_selection({1, 0});

    SelectionStrokeOperator op;
    OperatorProperties props;
    props.set("mode", 2);
    props.set("op", 0);
    props.set("x", 0.0);
    props.set("y", 0.0);

    EXPECT_EQ(op.invoke(*context_, props), OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                                    lfs::vis::input::ACTION_PRESS, 30.0, 0.0),
                       props),
              OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                                    lfs::vis::input::ACTION_PRESS, 0.0, 30.0),
                       props),
              OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                                    lfs::vis::input::ACTION_PRESS, 0.0, 0.0),
                       props),
              OperatorResult::RUNNING_MODAL);
    EXPECT_TRUE(service().isInteractiveSelectionClosed());

    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                                    lfs::vis::input::ACTION_PRESS,
                                    15.0,
                                    15.0,
                                    lfs::vis::input::KEYMOD_SHIFT),
                       props),
              OperatorResult::RUNNING_MODAL);
    EXPECT_TRUE(service().isInteractivePolygonVertexDragActive());
    EXPECT_EQ(dispatch(op, mouse_move(18.0, 15.0, 3.0, 0.0), props), OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                                    lfs::vis::input::ACTION_RELEASE, 18.0, 15.0),
                       props),
              OperatorResult::RUNNING_MODAL);
    EXPECT_FALSE(service().isInteractivePolygonVertexDragActive());

    service().refreshInteractivePreview();
    ASSERT_TRUE(rendering_manager_->isPolygonPreviewActive());
    const auto& inserted_points = rendering_manager_->getPolygonPoints();
    ASSERT_EQ(inserted_points.size(), 4u);
    EXPECT_FLOAT_EQ(inserted_points[2].first, 18.0f);
    EXPECT_FLOAT_EQ(inserted_points[2].second, 15.0f);

    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                                    lfs::vis::input::ACTION_PRESS,
                                    18.0,
                                    15.0,
                                    lfs::vis::input::KEYMOD_CTRL),
                       props),
              OperatorResult::RUNNING_MODAL);

    service().refreshInteractivePreview();
    const auto& reduced_points = rendering_manager_->getPolygonPoints();
    ASSERT_EQ(reduced_points.size(), 3u);
}

class DepthWindowDragLifecycleTest : public ::testing::Test {
protected:
    [[nodiscard]] virtual glm::ivec2 viewerSize() const { return {200, 200}; }

    void SetUp() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        lfs::vis::op::undoHistory().clear();

        options_.show_startup_overlay = false;
        const glm::ivec2 viewer_size = viewerSize();
        options_.width = viewer_size.x;
        options_.height = viewer_size.y;
        viewer_ = std::make_unique<lfs::vis::VisualizerImpl>(options_);
        viewer_->initializeTools();

        rendering_manager_ = viewer_->getRenderingManager();
        scene_manager_ = viewer_->getSceneManager();
        selection_tool_ = viewer_->getSelectionTool();
        ASSERT_NE(rendering_manager_, nullptr);
        ASSERT_NE(scene_manager_, nullptr);
        ASSERT_NE(selection_tool_, nullptr);

        scene_manager_->getScene().addSplat(
            "depth_window_test",
            make_test_splat({
                0.0f,
                0.0f,
                0.0f,
                1.0f,
                0.0f,
                0.0f,
            }));
        scene_manager_->initSelectionService();
        if (auto* const service = scene_manager_->getSelectionService()) {
            service->setTestingViewport({
                .x = 0.0f,
                .y = 0.0f,
                .width = static_cast<float>(options_.width),
                .height = static_cast<float>(options_.height),
                .render_width = options_.width,
                .render_height = options_.height,
            });
        }
        selection_tool_->setEnabled(true);

        auto settings = rendering_manager_->getSettings();
        settings.depth_filter_enabled = true;
        settings.depth_filter_scale_x = 0.5f;
        settings.depth_filter_scale_y = 0.5f;
        rendering_manager_->updateSettings(settings);
        selection_tool_->setDepthFilterEnabled(true);
    }

    void TearDown() override {
        lfs::vis::op::operators().cancelModalOperator();
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        viewer_.reset();
        lfs::vis::op::undoHistory().clear();
    }

    [[nodiscard]] lfs::vis::op::OperatorProperties depthDragProps(const double x, const double y) const {
        lfs::vis::op::OperatorProperties props;
        props.set("x", x);
        props.set("y", y);
        props.set("viewport_x", 0.0f);
        props.set("viewport_y", 0.0f);
        props.set("viewport_width", static_cast<float>(options_.width));
        props.set("viewport_height", static_cast<float>(options_.height));
        props.set("modifiers", lfs::vis::input::KEYMOD_SHIFT | lfs::vis::input::KEYMOD_ALT);
        return props;
    }

    void expectMidDragPreviewActive() {
        ASSERT_TRUE(startDepthDrag());
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(60.0, 60.0)),
                  OperatorResult::RUNNING_MODAL);
        EXPECT_TRUE(rendering_manager_->depthWindowDragPreview());
    }

    bool startDepthDrag(
        const double x = 20.0,
        const double y = 20.0,
        const int modifiers = lfs::vis::input::KEYMOD_SHIFT | lfs::vis::input::KEYMOD_ALT) {
        auto props = depthDragProps(x, y);
        props.set("modifiers", modifiers);
        const auto result = lfs::vis::op::operators().invoke(lfs::vis::op::BuiltinOp::DepthWindowDrag, &props);
        return result.status == lfs::vis::op::OperatorResult::RUNNING_MODAL;
    }

    [[nodiscard]] lfs::vis::op::DepthWindowRect currentDepthWindowRect() const {
        const auto settings = rendering_manager_->getSettings();
        return lfs::vis::op::depthWindowRenderRect(
            options_.width,
            options_.height,
            settings.depth_filter_scale_x,
            settings.depth_filter_scale_y,
            settings.depth_filter_offset_x,
            settings.depth_filter_offset_y);
    }

    lfs::vis::ViewerOptions options_{};
    std::unique_ptr<lfs::vis::VisualizerImpl> viewer_;
    lfs::vis::RenderingManager* rendering_manager_ = nullptr;
    lfs::vis::SceneManager* scene_manager_ = nullptr;
    lfs::vis::tools::SelectionTool* selection_tool_ = nullptr;
};

class DepthWindowDragGeometryTest : public DepthWindowDragLifecycleTest {
protected:
    [[nodiscard]] glm::ivec2 viewerSize() const override { return {320, 200}; }
};

TEST_F(DepthWindowDragLifecycleTest, ClickWithoutMovementCreatesNoWindow) {
    const auto generation = lfs::vis::app_store().depth_window_draw_generation.get();

    ASSERT_TRUE(startDepthDrag(20.0, 20.0));
    EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
    auto settings = rendering_manager_->getSettings();
    EXPECT_FLOAT_EQ(settings.depth_filter_scale_x, 0.5f);
    EXPECT_FLOAT_EQ(settings.depth_filter_scale_y, 0.5f);
    EXPECT_FLOAT_EQ(settings.depth_filter_offset_x, 0.0f);
    EXPECT_FLOAT_EQ(settings.depth_filter_offset_y, 0.0f);

    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(23.0, 22.0)),
              OperatorResult::RUNNING_MODAL);
    EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
    settings = rendering_manager_->getSettings();
    EXPECT_FLOAT_EQ(settings.depth_filter_scale_x, 0.5f);
    EXPECT_FLOAT_EQ(settings.depth_filter_scale_y, 0.5f);
    EXPECT_FLOAT_EQ(settings.depth_filter_offset_x, 0.0f);
    EXPECT_FLOAT_EQ(settings.depth_filter_offset_y, 0.0f);

    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(
                  mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                               lfs::vis::input::ACTION_RELEASE,
                               23.0,
                               22.0,
                               lfs::vis::input::KEYMOD_SHIFT | lfs::vis::input::KEYMOD_ALT)),
              OperatorResult::CANCELLED);
    settings = rendering_manager_->getSettings();
    EXPECT_FLOAT_EQ(settings.depth_filter_scale_x, 0.5f);
    EXPECT_FLOAT_EQ(settings.depth_filter_scale_y, 0.5f);
    EXPECT_FLOAT_EQ(settings.depth_filter_offset_x, 0.0f);
    EXPECT_FLOAT_EQ(settings.depth_filter_offset_y, 0.0f);
    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
    EXPECT_EQ(lfs::vis::app_store().depth_window_draw_generation.get(), generation);
    EXPECT_FALSE(lfs::vis::op::operators().hasModalOperator());
    EXPECT_TRUE(lfs::vis::op::depthWindowOverlayState().visible);
    EXPECT_FALSE(lfs::vis::op::depthWindowOverlayState().hide_handles);

    ASSERT_TRUE(startDepthDrag(20.0, 20.0));
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(
                  mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                               lfs::vis::input::ACTION_RELEASE,
                               20.0,
                               20.0)),
              OperatorResult::CANCELLED);
    EXPECT_FALSE(lfs::vis::op::depthWindowOverlayState().visible);
}

TEST_F(DepthWindowDragLifecycleTest, DrawStartsAtThreshold) {
    const auto generation = lfs::vis::app_store().depth_window_draw_generation.get();

    ASSERT_TRUE(startDepthDrag(20.0, 20.0));
    // (23,24) is exactly 5 px from the press: the draw must start AT the
    // threshold, not beyond it (locks the < vs <= boundary).
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(23.0, 24.0)),
              OperatorResult::RUNNING_MODAL);
    EXPECT_TRUE(rendering_manager_->depthWindowDragPreview());
    const auto dragged = rendering_manager_->getSettings();
    EXPECT_TRUE(dragged.depth_filter_scale_x != 0.5f ||
                dragged.depth_filter_scale_y != 0.5f ||
                dragged.depth_filter_offset_x != 0.0f ||
                dragged.depth_filter_offset_y != 0.0f);

    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(
                  mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                               lfs::vis::input::ACTION_RELEASE,
                               23.0,
                               24.0)),
              OperatorResult::FINISHED);
    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
    EXPECT_EQ(lfs::vis::app_store().depth_window_draw_generation.get(), generation + 1u);
}

TEST_F(DepthWindowDragLifecycleTest, CtrlClickWithoutMovementCreatesNoWindow) {
    const auto generation = lfs::vis::app_store().depth_window_draw_generation.get();
    const int modifiers = lfs::vis::input::KEYMOD_SHIFT | lfs::vis::input::KEYMOD_ALT |
                          lfs::vis::input::KEYMOD_CTRL;

    ASSERT_TRUE(startDepthDrag(20.0, 20.0, modifiers));
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(
                  mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                               lfs::vis::input::ACTION_RELEASE,
                               20.0,
                               20.0,
                               modifiers)),
              OperatorResult::CANCELLED);

    const auto settings = rendering_manager_->getSettings();
    EXPECT_FLOAT_EQ(settings.depth_filter_scale_x, 0.5f);
    EXPECT_FLOAT_EQ(settings.depth_filter_scale_y, 0.5f);
    EXPECT_FLOAT_EQ(settings.depth_filter_offset_x, 0.0f);
    EXPECT_FLOAT_EQ(settings.depth_filter_offset_y, 0.0f);
    EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
    EXPECT_EQ(lfs::vis::app_store().depth_window_draw_generation.get(), generation);
}

TEST_F(DepthWindowDragLifecycleTest, CtrlReleaseBeforeThresholdWritesNothing) {
    const int modifiers = lfs::vis::input::KEYMOD_SHIFT | lfs::vis::input::KEYMOD_ALT |
                          lfs::vis::input::KEYMOD_CTRL;
    ASSERT_TRUE(startDepthDrag(20.0, 20.0, modifiers));

    // Releasing Ctrl before the draw has started must neither sample the old
    // window's ratio nor write any settings.
    const ModalEvent release_ctrl{
        .type = ModalEvent::Type::KEY,
        .data = KeyEvent{
            .key = lfs::vis::input::KEY_LEFT_CONTROL,
            .scancode = 0,
            .action = lfs::vis::input::ACTION_RELEASE,
            .mods = lfs::vis::input::KEYMOD_SHIFT | lfs::vis::input::KEYMOD_ALT,
        },
    };
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(release_ctrl),
              OperatorResult::RUNNING_MODAL);
    EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
    const auto settings = rendering_manager_->getSettings();
    EXPECT_FLOAT_EQ(settings.depth_filter_scale_x, 0.5f);
    EXPECT_FLOAT_EQ(settings.depth_filter_scale_y, 0.5f);
    EXPECT_FLOAT_EQ(settings.depth_filter_offset_x, 0.0f);
    EXPECT_FLOAT_EQ(settings.depth_filter_offset_y, 0.0f);

    // The constraint really was cleared: a non-square move past the threshold
    // must produce a non-square window (a still-constrained draw would square it).
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(60.0, 40.0)),
              OperatorResult::RUNNING_MODAL);
    const auto rect = currentDepthWindowRect();
    EXPECT_NEAR(rect.size().x, 40.0f, 1.0f);
    EXPECT_NEAR(rect.size().y, 20.0f, 1.0f);
    EXPECT_GT(std::abs(rect.size().x - rect.size().y), 5.0f);

    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(
                  mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                               lfs::vis::input::ACTION_RELEASE,
                               60.0,
                               40.0,
                               lfs::vis::input::KEYMOD_SHIFT | lfs::vis::input::KEYMOD_ALT)),
              OperatorResult::FINISHED);
    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
}

TEST_F(DepthWindowDragLifecycleTest, ViewportEdgePressStillStartsDrawAfterThreshold) {
    ASSERT_TRUE(startDepthDrag(1.0, 1.0));
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(6.0, 6.0)),
              OperatorResult::RUNNING_MODAL);
    EXPECT_TRUE(rendering_manager_->depthWindowDragPreview());
    const auto settings = rendering_manager_->getSettings();
    EXPECT_TRUE(settings.depth_filter_scale_x != 0.5f ||
                settings.depth_filter_scale_y != 0.5f ||
                settings.depth_filter_offset_x != 0.0f ||
                settings.depth_filter_offset_y != 0.0f);
}

TEST_F(DepthWindowDragLifecycleTest, ReplacementBeforeThresholdLeavesNothingBehind) {
    ASSERT_TRUE(startDepthDrag(20.0, 20.0));
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(22.0, 22.0)),
              OperatorResult::RUNNING_MODAL);

    lfs::vis::op::OperatorProperties stroke_props;
    stroke_props.set("mode", 0);
    stroke_props.set("op", 0);
    stroke_props.set("x", 30.0);
    stroke_props.set("y", 30.0);
    const auto stroke = lfs::vis::op::operators().invoke(
        lfs::vis::op::BuiltinOp::SelectionStroke, &stroke_props);
    ASSERT_EQ(stroke.status, OperatorResult::RUNNING_MODAL);

    const auto settings = rendering_manager_->getSettings();
    EXPECT_FLOAT_EQ(settings.depth_filter_scale_x, 0.5f);
    EXPECT_FLOAT_EQ(settings.depth_filter_scale_y, 0.5f);
    EXPECT_FLOAT_EQ(settings.depth_filter_offset_x, 0.0f);
    EXPECT_FLOAT_EQ(settings.depth_filter_offset_y, 0.0f);
    EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
    EXPECT_FALSE(lfs::vis::op::depthWindowOverlayState().visible);
}

TEST_F(DepthWindowDragLifecycleTest, PendingSecondDepthDragPreservesIncomingBaseline) {
    ASSERT_TRUE(startDepthDrag(20.0, 20.0));
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(60.0, 60.0)),
              OperatorResult::RUNNING_MODAL);
    const auto first_drag = rendering_manager_->getSettings();

    ASSERT_TRUE(startDepthDrag(170.0, 170.0));
    auto settings = rendering_manager_->getSettings();
    EXPECT_FLOAT_EQ(settings.depth_filter_scale_x, first_drag.depth_filter_scale_x);
    EXPECT_FLOAT_EQ(settings.depth_filter_scale_y, first_drag.depth_filter_scale_y);
    EXPECT_FLOAT_EQ(settings.depth_filter_offset_x, first_drag.depth_filter_offset_x);
    EXPECT_FLOAT_EQ(settings.depth_filter_offset_y, first_drag.depth_filter_offset_y);

    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(
                  mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                               lfs::vis::input::ACTION_RELEASE,
                               170.0,
                               170.0,
                               lfs::vis::input::KEYMOD_SHIFT | lfs::vis::input::KEYMOD_ALT)),
              OperatorResult::CANCELLED);
    settings = rendering_manager_->getSettings();
    EXPECT_FLOAT_EQ(settings.depth_filter_scale_x, first_drag.depth_filter_scale_x);
    EXPECT_FLOAT_EQ(settings.depth_filter_scale_y, first_drag.depth_filter_scale_y);
    EXPECT_FLOAT_EQ(settings.depth_filter_offset_x, first_drag.depth_filter_offset_x);
    EXPECT_FLOAT_EQ(settings.depth_filter_offset_y, first_drag.depth_filter_offset_y);
    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
}

TEST_F(DepthWindowDragGeometryTest, SmallDrawKeepsAnchorAndMeetsMinimumSize) {
    ASSERT_TRUE(startDepthDrag(20.0, 20.0));
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(24.0, 24.0)),
              OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(
                  mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                               lfs::vis::input::ACTION_RELEASE,
                               24.0,
                               24.0)),
              OperatorResult::FINISHED);

    const auto rect = currentDepthWindowRect();
    EXPECT_NEAR(rect.size().x, 16.0f, 1.0f);
    EXPECT_NEAR(rect.size().y, 10.0f, 1.0f);
    // Tight anchor tolerance: the pre-fix sanitizer path lands the anchor about
    // 0.7 px off (19.2, 19.4), so 1 px would let the old behaviour pass.
    EXPECT_NEAR(rect.min.x, 20.0f, 0.25f);
    EXPECT_NEAR(rect.min.y, 20.0f, 0.25f);
}

TEST_F(DepthWindowDragGeometryTest, SmallConstrainedDrawIsSquareAtAnchor) {
    ASSERT_TRUE(startDepthDrag(
        20.0, 20.0,
        lfs::vis::input::KEYMOD_SHIFT | lfs::vis::input::KEYMOD_ALT |
            lfs::vis::input::KEYMOD_CTRL));
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(24.0, 24.0)),
              OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(
                  mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                               lfs::vis::input::ACTION_RELEASE,
                               24.0,
                               24.0)),
              OperatorResult::FINISHED);

    const auto rect = currentDepthWindowRect();
    EXPECT_NEAR(rect.size().x, 16.0f, 1.0f);
    EXPECT_NEAR(rect.size().y, 16.0f, 1.0f);
    EXPECT_NEAR(rect.size().x, rect.size().y, 1e-4f);
    EXPECT_NEAR(rect.min.x, 20.0f, 1.0f);
    EXPECT_NEAR(rect.min.y, 20.0f, 1.0f);
}

TEST_F(DepthWindowDragGeometryTest, ConstrainedDrawNearBoundFlipsAwayFromEdge) {
    ASSERT_TRUE(startDepthDrag(
        300.0, 100.0,
        lfs::vis::input::KEYMOD_SHIFT | lfs::vis::input::KEYMOD_ALT |
            lfs::vis::input::KEYMOD_CTRL));
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(304.0, 104.0)),
              OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(
                  mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                               lfs::vis::input::ACTION_RELEASE,
                               304.0,
                               104.0)),
              OperatorResult::FINISHED);

    const auto rect = currentDepthWindowRect();
    EXPECT_NEAR(rect.size().x, 16.0f, 1.0f);
    EXPECT_NEAR(rect.size().y, 16.0f, 1.0f);
    EXPECT_NEAR(rect.size().x, rect.size().y, 1e-4f);
    EXPECT_NEAR(rect.max.x, 300.0f, 1.0f);
    EXPECT_NEAR(rect.min.y, 100.0f, 1.0f);
    EXPECT_LE(rect.max.x, 314.0f);
}

TEST_F(DepthWindowDragGeometryTest, ConstrainedOverflowStillShrinksByCommonFactor) {
    ASSERT_TRUE(startDepthDrag(
        60.0, 100.0,
        lfs::vis::input::KEYMOD_SHIFT | lfs::vis::input::KEYMOD_ALT |
            lfs::vis::input::KEYMOD_CTRL));
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(314.0, 194.0)),
              OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(
                  mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                               lfs::vis::input::ACTION_RELEASE,
                               314.0,
                               194.0)),
              OperatorResult::FINISHED);

    const auto rect = currentDepthWindowRect();
    EXPECT_NEAR(rect.size().x, 94.0f, 1.0f);
    EXPECT_NEAR(rect.size().y, 94.0f, 1.0f);
    EXPECT_NEAR(rect.size().x, rect.size().y, 1e-4f);
    EXPECT_NEAR(rect.min.x, 60.0f, 1.0f);
    EXPECT_NEAR(rect.min.y, 100.0f, 1.0f);
}

TEST_F(DepthWindowDragLifecycleTest, EdgeFloorPreservesLegacyLeftEdgeBehavior) {
    auto settings = rendering_manager_->getSettings();
    settings.depth_filter_scale_x = 0.1f;
    settings.depth_filter_scale_y = 0.5f;
    settings.depth_filter_offset_x = 0.0f;
    settings.depth_filter_offset_y = 0.0f;
    rendering_manager_->updateSettings(settings);

    ASSERT_TRUE(startDepthDrag(90.0, 100.0));
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(95.0, 104.0)),
              OperatorResult::RUNNING_MODAL);
    auto rect = currentDepthWindowRect();
    EXPECT_FLOAT_EQ(rect.min.x, 95.0f);
    EXPECT_FLOAT_EQ(rect.max.x, 110.0f);

    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(104.0, 100.0)),
              OperatorResult::RUNNING_MODAL);
    rect = currentDepthWindowRect();
    EXPECT_FLOAT_EQ(rect.min.x, 104.0f);
    EXPECT_FLOAT_EQ(rect.max.x, 114.0f);
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(
                  mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                               lfs::vis::input::ACTION_RELEASE,
                               104.0,
                               100.0)),
              OperatorResult::FINISHED);
}

TEST_F(DepthWindowDragLifecycleTest, CommitClearsDepthWindowDragPreview) {
    expectMidDragPreviewActive();
    const auto release = mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                                      lfs::vis::input::ACTION_RELEASE,
                                      60.0,
                                      60.0);
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(release), lfs::vis::op::OperatorResult::FINISHED);
    EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
}

TEST_F(DepthWindowDragLifecycleTest, EscapeClearsDepthWindowDragPreview) {
    expectMidDragPreviewActive();
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(key_press(lfs::vis::input::KEY_ESCAPE)),
              lfs::vis::op::OperatorResult::CANCELLED);
    EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
}

TEST_F(DepthWindowDragLifecycleTest, ModifierReleaseClearsDepthWindowDragPreview) {
    expectMidDragPreviewActive();
    const ModalEvent release_shift{
        .type = ModalEvent::Type::KEY,
        .data = KeyEvent{
            .key = lfs::vis::input::KEY_LEFT_SHIFT,
            .scancode = 0,
            .action = lfs::vis::input::ACTION_RELEASE,
            .mods = lfs::vis::input::KEYMOD_ALT,
        },
    };
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(release_shift),
              lfs::vis::op::OperatorResult::CANCELLED);
    EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
}

TEST_F(DepthWindowDragLifecycleTest, GtToggleClearsDepthWindowDragPreview) {
    expectMidDragPreviewActive();
    lfs::core::events::cmd::ToggleGTComparison{}.emit();
    EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
    EXPECT_FALSE(lfs::vis::op::operators().hasModalOperator());
}

TEST_F(DepthWindowDragLifecycleTest, ModalReplacementRestoresDepthWindowAndPushesNoUndo) {
    ASSERT_TRUE(startDepthDrag(20.0, 20.0));
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(80.0, 80.0)),
              OperatorResult::RUNNING_MODAL);
    EXPECT_TRUE(rendering_manager_->depthWindowDragPreview());
    const auto dragged = rendering_manager_->getSettings();
    EXPECT_TRUE(dragged.depth_filter_scale_x != 0.5f ||
                dragged.depth_filter_scale_y != 0.5f);

    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_scroll()),
              OperatorResult::RUNNING_MODAL);
    const auto after_scroll = rendering_manager_->getSettings();
    EXPECT_FLOAT_EQ(after_scroll.depth_filter_scale_x, dragged.depth_filter_scale_x);
    EXPECT_FLOAT_EQ(after_scroll.depth_filter_scale_y, dragged.depth_filter_scale_y);
    EXPECT_FLOAT_EQ(after_scroll.depth_filter_offset_x, dragged.depth_filter_offset_x);
    EXPECT_FLOAT_EQ(after_scroll.depth_filter_offset_y, dragged.depth_filter_offset_y);

    lfs::vis::op::OperatorProperties stroke_props;
    stroke_props.set("mode", 0);
    stroke_props.set("op", 0);
    stroke_props.set("x", 30.0);
    stroke_props.set("y", 30.0);
    const auto stroke = lfs::vis::op::operators().invoke(lfs::vis::op::BuiltinOp::SelectionStroke, &stroke_props);
    ASSERT_EQ(stroke.status, lfs::vis::op::OperatorResult::RUNNING_MODAL);
    const auto restored = rendering_manager_->getSettings();
    EXPECT_FLOAT_EQ(restored.depth_filter_scale_x, 0.5f);
    EXPECT_FLOAT_EQ(restored.depth_filter_scale_y, 0.5f);
    EXPECT_FLOAT_EQ(restored.depth_filter_offset_x, 0.0f);
    EXPECT_FLOAT_EQ(restored.depth_filter_offset_y, 0.0f);
    EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
    // The replacing stroke's undo transaction is still open here. A
    // DepthWindowSettingsUndoEntry pushed by the replaced drag would be queued
    // there, invisible to undoCount(), but visible to transactionBytes().
    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
    EXPECT_EQ(lfs::vis::op::undoHistory().transactionBytes(), 0u);
    lfs::vis::op::operators().cancelModalOperator();
}

TEST_F(DepthWindowDragLifecycleTest, ModalReplacementByDragKeepsSecondDragState) {
    ASSERT_TRUE(startDepthDrag(20.0, 20.0));
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(80.0, 80.0)),
              OperatorResult::RUNNING_MODAL);
    const auto first_drag = rendering_manager_->getSettings();

    ASSERT_TRUE(startDepthDrag(170.0, 170.0));
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(160.0, 160.0)),
              OperatorResult::RUNNING_MODAL);
    const auto second_drag = rendering_manager_->getSettings();
    EXPECT_TRUE(second_drag.depth_filter_scale_x != first_drag.depth_filter_scale_x ||
                second_drag.depth_filter_scale_y != first_drag.depth_filter_scale_y ||
                second_drag.depth_filter_offset_x != first_drag.depth_filter_offset_x ||
                second_drag.depth_filter_offset_y != first_drag.depth_filter_offset_y);
    EXPECT_TRUE(second_drag.depth_filter_scale_x != 0.5f ||
                second_drag.depth_filter_scale_y != 0.5f ||
                second_drag.depth_filter_offset_x != 0.0f ||
                second_drag.depth_filter_offset_y != 0.0f);

    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(key_press(lfs::vis::input::KEY_ESCAPE)),
              OperatorResult::CANCELLED);
    const auto restored = rendering_manager_->getSettings();
    EXPECT_FLOAT_EQ(restored.depth_filter_scale_x, first_drag.depth_filter_scale_x);
    EXPECT_FLOAT_EQ(restored.depth_filter_scale_y, first_drag.depth_filter_scale_y);
    EXPECT_FLOAT_EQ(restored.depth_filter_offset_x, first_drag.depth_filter_offset_x);
    EXPECT_FLOAT_EQ(restored.depth_filter_offset_y, first_drag.depth_filter_offset_y);
}

TEST_F(DepthWindowDragLifecycleTest, EscapeRestoresDepthWindowAndPushesNoUndo) {
    ASSERT_TRUE(startDepthDrag(20.0, 20.0));
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(60.0, 60.0)),
              OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(key_press(lfs::vis::input::KEY_ESCAPE)),
              OperatorResult::CANCELLED);

    const auto restored = rendering_manager_->getSettings();
    EXPECT_FLOAT_EQ(restored.depth_filter_scale_x, 0.5f);
    EXPECT_FLOAT_EQ(restored.depth_filter_scale_y, 0.5f);
    EXPECT_FLOAT_EQ(restored.depth_filter_offset_x, 0.0f);
    EXPECT_FLOAT_EQ(restored.depth_filter_offset_y, 0.0f);
    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
}

TEST_F(DepthWindowDragLifecycleTest, SecondDepthDragReplacementKeepsIncomingOverlayVisible) {
    ASSERT_TRUE(startDepthDrag(20.0, 20.0));
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(60.0, 60.0)),
              OperatorResult::RUNNING_MODAL);

    ASSERT_TRUE(startDepthDrag(170.0, 170.0));
    EXPECT_TRUE(lfs::vis::op::operators().hasModalOperator());
    EXPECT_EQ(lfs::vis::op::operators().activeModalId(), "selection.depth_window_drag");
    EXPECT_TRUE(lfs::vis::op::depthWindowOverlayState().visible);
}

TEST_F(DepthWindowDragLifecycleTest, UndoOfDrawRebasesSizeReadoutReference) {
    const auto generation = []() {
        return lfs::vis::app_store().depth_window_draw_generation.get();
    };
    const auto g0 = generation();

    ASSERT_TRUE(startDepthDrag(10.0, 10.0));
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(60.0, 60.0)),
              OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(
                  mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                               lfs::vis::input::ACTION_RELEASE,
                               60.0,
                               60.0)),
              OperatorResult::FINISHED);
    const auto g1 = generation();
    EXPECT_EQ(g1, g0 + 1u);
    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);

    EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);
    EXPECT_EQ(generation(), g1 + 1u);

    EXPECT_TRUE(lfs::vis::op::undoHistory().redo().success);
    const auto g_after_redo = generation();
    EXPECT_EQ(g_after_redo, g1 + 2u);

    const auto settings = rendering_manager_->getSettings();
    const auto render_center = lfs::vis::op::depthWindowRenderRect(
                                   options_.width,
                                   options_.height,
                                   settings.depth_filter_scale_x,
                                   settings.depth_filter_scale_y,
                                   settings.depth_filter_offset_x,
                                   settings.depth_filter_offset_y)
                                   .center();
    const double center_x = static_cast<double>(render_center.x);
    const double center_y = static_cast<double>(render_center.y);

    ASSERT_TRUE(startDepthDrag(center_x, center_y));
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(center_x + 20.0, center_y)),
              OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(
                  mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                               lfs::vis::input::ACTION_RELEASE,
                               center_x + 20.0,
                               center_y)),
              OperatorResult::FINISHED);
    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 2u);
    EXPECT_EQ(generation(), g_after_redo);

    EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);
    EXPECT_EQ(generation(), g_after_redo);
}

TEST_F(DepthWindowDragLifecycleTest, DepthWindowScrollScalePreservesAspectAtLimits) {
    const auto set_scales = [&](const float scale_x, const float scale_y) {
        auto settings = rendering_manager_->getSettings();
        settings.depth_filter_scale_x = scale_x;
        settings.depth_filter_scale_y = scale_y;
        rendering_manager_->updateSettings(settings);
    };
    const auto expect_scales = [&](const float scale_x, const float scale_y) {
        const auto settings = rendering_manager_->getSettings();
        EXPECT_NEAR(settings.depth_filter_scale_x, scale_x, 1e-5f);
        EXPECT_NEAR(settings.depth_filter_scale_y, scale_y, 1e-5f);
    };

    // Upper limit: X is already at 1.0, so growing must leave BOTH axes alone.
    set_scales(1.0f, 0.5f);
    selection_tool_->adjustWindowScale(1.05f);
    expect_scales(1.0f, 0.5f);

    // Interior: both axes scale by the same factor.
    set_scales(0.8f, 0.4f);
    selection_tool_->adjustWindowScale(1.05f);
    expect_scales(0.84f, 0.42f);

    // Lower limit: Y is already on the floor, so shrinking must leave BOTH alone.
    set_scales(0.1f, 0.05f);
    selection_tool_->adjustWindowScale(0.95f);
    expect_scales(0.1f, 0.05f);

    // Extreme aspect: the common factor interval collapses to exactly 1.
    set_scales(0.05f, 1.0f);
    selection_tool_->adjustWindowScale(1.05f);
    expect_scales(0.05f, 1.0f);
    selection_tool_->adjustWindowScale(0.95f);
    expect_scales(0.05f, 1.0f);
}

TEST_F(DepthWindowDragLifecycleTest, FocusLossClearsDepthWindowDragPreview) {
    expectMidDragPreviewActive();
    // The viewer is constructed without initialize(), so no InputController is
    // wired to the WindowManager; drive the focus-loss handler on a directly
    // constructed controller (its cancelModalOperator path is what terminates
    // the drag).
    Viewport focus_viewport(options_.width, options_.height);
    lfs::vis::InputController input(nullptr, focus_viewport);
    input.onWindowFocusLost();
    EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
    EXPECT_FALSE(lfs::vis::op::operators().hasModalOperator());
}
