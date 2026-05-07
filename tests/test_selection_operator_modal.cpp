/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "input/key_codes.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_context.hpp"
#include "operator/operator_properties.hpp"
#include "operator/ops/selection_ops.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"

#include <algorithm>
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

TEST_F(SelectionOperatorModalTest, RectangleOperatorCommitsOnLeftRelease) {
    service().setTestingScreenPositions(make_screen_positions({
        10.0f,
        10.0f,
        80.0f,
        80.0f,
    }));

    SelectionStrokeOperator op;
    OperatorProperties props;
    props.set("mode", 1);
    props.set("op", 0);
    props.set("x", 0.0);
    props.set("y", 0.0);

    EXPECT_EQ(op.invoke(*context_, props), OperatorResult::RUNNING_MODAL);
    EXPECT_TRUE(service().isInteractiveSelectionActive());

    EXPECT_EQ(dispatch(op, mouse_move(30.0, 30.0), props), OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                                    lfs::vis::input::ACTION_RELEASE, 30.0, 30.0),
                       props),
              OperatorResult::FINISHED);

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
    EXPECT_FALSE(service().isInteractiveSelectionActive());
}

TEST_F(SelectionOperatorModalTest, RectangleOperatorCancelsOnBoundAction) {
    set_initial_selection({1, 0});
    service().setTestingScreenPositions(make_screen_positions({
        10.0f,
        10.0f,
        80.0f,
        80.0f,
    }));

    SelectionStrokeOperator op;
    OperatorProperties props;
    props.set("mode", 1);
    props.set("op", 0);
    props.set("x", 0.0);
    props.set("y", 0.0);

    EXPECT_EQ(op.invoke(*context_, props), OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::RIGHT),
                                    lfs::vis::input::ACTION_PRESS, 10.0, 10.0),
                       props),
              OperatorResult::PASS_THROUGH);
    EXPECT_EQ(dispatch(op,
                       modal_action(lfs::vis::input::Action::UNDO_POLYGON_VERTEX),
                       props),
              OperatorResult::CANCELLED);
    op.cancel(*context_);

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
    EXPECT_FALSE(service().isInteractiveSelectionActive());
}

TEST_F(SelectionOperatorModalTest, RectangleOperatorCommitsOnConfiguredRightRelease) {
    service().setTestingScreenPositions(make_screen_positions({
        10.0f,
        10.0f,
        80.0f,
        80.0f,
    }));

    SelectionStrokeOperator op;
    OperatorProperties props;
    props.set("mode", 1);
    props.set("op", 0);
    props.set("x", 0.0);
    props.set("y", 0.0);
    props.set("button", static_cast<int>(lfs::vis::input::AppMouseButton::RIGHT));

    EXPECT_EQ(op.invoke(*context_, props), OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(dispatch(op, mouse_move(30.0, 30.0), props), OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(dispatch(op,
                       mouse_button(static_cast<int>(lfs::vis::input::AppMouseButton::RIGHT),
                                    lfs::vis::input::ACTION_RELEASE, 30.0, 30.0),
                       props),
              OperatorResult::FINISHED);

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
    EXPECT_FALSE(service().isInteractiveSelectionActive());
}

TEST_F(SelectionOperatorModalTest, PolygonOperatorPassesThroughNavigationAndCommitsWithBoundConfirmAction) {
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
    EXPECT_TRUE(service().isInteractiveSelectionActive());

    EXPECT_EQ(dispatch(op, mouse_move(15.0, 5.0, 15.0, 5.0), props), OperatorResult::PASS_THROUGH);
    EXPECT_EQ(dispatch(op, mouse_scroll(), props), OperatorResult::PASS_THROUGH);

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

    EXPECT_EQ(dispatch(op, key_press(lfs::vis::input::KEY_ENTER, lfs::vis::input::KEYMOD_SHIFT), props),
              OperatorResult::PASS_THROUGH);
    EXPECT_EQ(dispatch(op,
                       modal_action(lfs::vis::input::Action::CONFIRM_POLYGON,
                                    lfs::vis::input::KEYMOD_SHIFT),
                       props),
              OperatorResult::FINISHED);

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 1}));
    EXPECT_FALSE(service().isInteractiveSelectionActive());
}

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
