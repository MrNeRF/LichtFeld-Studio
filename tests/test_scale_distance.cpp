/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/services.hpp"
#include "operation/undo_history.hpp"
#include "operator/ops/scale_distance_ops.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/app_store.hpp"

#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <memory>

namespace lfs::vis::op {

    // -------------------------------------------------------------------------
    // Math & Transform calculation unit tests
    // -------------------------------------------------------------------------

    TEST(ScaleDistanceTransformTest, BasicScalingAroundMidpoint) {
        const glm::vec3 p0{0.0f, 0.0f, 0.0f};
        const glm::vec3 p1{2.0f, 0.0f, 0.0f};
        const float target_distance = 4.0f;

        const auto result = computeScaleDistanceTransform(p0, p1, target_distance);
        ASSERT_TRUE(result.has_value());

        const glm::mat4& mat = *result;
        const glm::vec3 p0_scaled = glm::vec3(mat * glm::vec4(p0, 1.0f));
        const glm::vec3 p1_scaled = glm::vec3(mat * glm::vec4(p1, 1.0f));

        EXPECT_NEAR(glm::length(p1_scaled - p0_scaled), target_distance, 1e-5f);

        // Midpoint should remain anchored at (1, 0, 0)
        const glm::vec3 orig_mid = (p0 + p1) * 0.5f;
        const glm::vec3 scaled_mid = (p0_scaled + p1_scaled) * 0.5f;
        EXPECT_NEAR(scaled_mid.x, orig_mid.x, 1e-5f);
        EXPECT_NEAR(scaled_mid.y, orig_mid.y, 1e-5f);
        EXPECT_NEAR(scaled_mid.z, orig_mid.z, 1e-5f);

        // p0 expands from 0 to -1, p1 expands from 2 to 3
        EXPECT_NEAR(p0_scaled.x, -1.0f, 1e-5f);
        EXPECT_NEAR(p1_scaled.x, 3.0f, 1e-5f);
    }

    TEST(ScaleDistanceTransformTest, ScaleDownAroundMidpoint) {
        const glm::vec3 p0{-2.0f, 4.0f, 1.0f};
        const glm::vec3 p1{6.0f, 4.0f, 1.0f};
        const float original_distance = glm::length(p1 - p0);
        EXPECT_NEAR(original_distance, 8.0f, 1e-5f);

        const float target_distance = 2.0f; // Scale factor = 0.25
        const auto result = computeScaleDistanceTransform(p0, p1, target_distance);
        ASSERT_TRUE(result.has_value());

        const glm::mat4& mat = *result;
        const glm::vec3 p0_scaled = glm::vec3(mat * glm::vec4(p0, 1.0f));
        const glm::vec3 p1_scaled = glm::vec3(mat * glm::vec4(p1, 1.0f));

        EXPECT_NEAR(glm::length(p1_scaled - p0_scaled), target_distance, 1e-5f);

        const glm::vec3 orig_mid = (p0 + p1) * 0.5f;
        const glm::vec3 scaled_mid = (p0_scaled + p1_scaled) * 0.5f;
        EXPECT_NEAR(scaled_mid.x, orig_mid.x, 1e-5f);
        EXPECT_NEAR(scaled_mid.y, orig_mid.y, 1e-5f);
        EXPECT_NEAR(scaled_mid.z, orig_mid.z, 1e-5f);
    }

    TEST(ScaleDistanceTransformTest, Arbitrary3DOrientation) {
        const glm::vec3 p0{1.5f, -2.0f, 3.5f};
        const glm::vec3 p1{4.5f, 1.0f, -0.5f};
        const float target_distance = 12.345f;

        const auto result = computeScaleDistanceTransform(p0, p1, target_distance);
        ASSERT_TRUE(result.has_value());

        const glm::mat4& mat = *result;
        const glm::vec3 p0_scaled = glm::vec3(mat * glm::vec4(p0, 1.0f));
        const glm::vec3 p1_scaled = glm::vec3(mat * glm::vec4(p1, 1.0f));

        EXPECT_NEAR(glm::length(p1_scaled - p0_scaled), target_distance, 1e-5f);

        const glm::vec3 orig_mid = (p0 + p1) * 0.5f;
        const glm::vec3 scaled_mid = (p0_scaled + p1_scaled) * 0.5f;
        EXPECT_NEAR(scaled_mid.x, orig_mid.x, 1e-5f);
        EXPECT_NEAR(scaled_mid.y, orig_mid.y, 1e-5f);
        EXPECT_NEAR(scaled_mid.z, orig_mid.z, 1e-5f);
    }

    TEST(ScaleDistanceTransformTest, DegeneratePointsRejected) {
        const glm::vec3 p0{3.0f, 2.0f, 1.0f};
        const glm::vec3 p1{3.0f, 2.0f, 1.0f};
        EXPECT_FALSE(computeScaleDistanceTransform(p0, p1, 5.0f).has_value());

        // Sub-epsilon distance
        const glm::vec3 p2 = p0 + glm::vec3{1e-7f, 0.0f, 0.0f};
        EXPECT_FALSE(computeScaleDistanceTransform(p0, p2, 5.0f).has_value());
    }

    TEST(ScaleDistanceTransformTest, InvalidTargetDistanceRejected) {
        const glm::vec3 p0{0.0f, 0.0f, 0.0f};
        const glm::vec3 p1{1.0f, 0.0f, 0.0f};

        EXPECT_FALSE(computeScaleDistanceTransform(p0, p1, 0.0f).has_value());
        EXPECT_FALSE(computeScaleDistanceTransform(p0, p1, -1.0f).has_value());
        EXPECT_FALSE(computeScaleDistanceTransform(p0, p1, std::numeric_limits<float>::quiet_NaN()).has_value());
        EXPECT_FALSE(computeScaleDistanceTransform(p0, p1, std::numeric_limits<float>::infinity()).has_value());
    }

    // -------------------------------------------------------------------------
    // Services & Observable State Generation tests
    // -------------------------------------------------------------------------

    TEST(ScaleDistanceServicesTest, ObservableGenerationTracksPointsAndStatus) {
        auto& store = app_store();
        (void)store.store().drain_dirty_into_frame();
        int notifications = 0;
        auto token = store.scale_distance_state_generation.subscribe([&](const auto&) { ++notifications; });
        const auto initial_gen = store.scale_distance_state_generation.get();

        // Identical points should not bump generation
        services().scaleDistancePointPick().setPickedPoints(services().scaleDistancePointPick().getPickedPoints());
        EXPECT_EQ(store.scale_distance_state_generation.get(), initial_gen);

        // Setting picked points notifies UI
        services().scaleDistancePointPick().setPickedPoints({{0.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}});
        EXPECT_EQ(notifications, 0);
        EXPECT_TRUE(store.store().drain_dirty_into_frame());
        EXPECT_EQ(notifications, 1);
        EXPECT_EQ(services().scaleDistancePointPick().getPickedPoints().size(), 2u);

        // Setting status message notifies UI
        services().scaleDistancePointPick().setStatusMessage("Invalid target distance", 2.0);
        EXPECT_TRUE(store.store().drain_dirty_into_frame());
        EXPECT_EQ(notifications, 2);

        // Clearing picked points clears status message and notifies UI
        services().scaleDistancePointPick().clearPickedPoints();
        EXPECT_EQ(services().scaleDistancePointPick().getStatusMessage(), nullptr);
        EXPECT_TRUE(services().scaleDistancePointPick().getPickedPoints().empty());
        EXPECT_TRUE(store.store().drain_dirty_into_frame());
        EXPECT_EQ(notifications, 3);
    }

    // -------------------------------------------------------------------------
    // Operator Integration & Undo/Redo tests
    // -------------------------------------------------------------------------

    class ScaleDistanceOperatorTest : public ::testing::Test {
    protected:
        void SetUp() override {
            services().clear();
            undoHistory().clear();
            manager_ = std::make_unique<SceneManager>();
            services().set(manager_.get());
            manager_->getScene().addGroup("target");
            manager_->selectNode("target");
            context_ = std::make_unique<OperatorContext>(*manager_);
        }

        void TearDown() override {
            operation_.cancel(*context_);
            undoHistory().clear();
            services().clear();
            context_.reset();
            manager_.reset();
            lfs::event::EventBridge::instance().clear_all();
        }

        OperatorResult requestAction(PointPickState::UiAction action, float target = 0.0f) {
            services().scaleDistancePointPick().requestUiAction(action, target);
            OperatorProperties properties;
            return operation_.modal(*context_, properties);
        }

        [[nodiscard]] glm::mat4 targetTransform() const {
            return manager_->getScene().getNode("target")->transform();
        }

        void setPickedPoints(const std::vector<glm::vec3>& pts) {
            operation_.setPickedPointsForTesting(pts);
        }

        OperatorResult send(const ModalEvent& event) {
            context_->setModalEvent(event);
            OperatorProperties properties;
            return operation_.modal(*context_, properties);
        }

        OperatorResult mouse(const int action, const glm::dvec2 position, const bool over_gui = false) {
            return send({.type = ModalEvent::Type::MOUSE_BUTTON,
                         .over_gui = over_gui,
                         .data = MouseButtonEvent{.button = operation_.pick_button_, .action = action, .mods = 0, .position = position}});
        }

        OperatorResult key(const int key) {
            return send({.type = ModalEvent::Type::KEY, .data = KeyEvent{key, 0, input::ACTION_PRESS, 0}});
        }

        OperatorResult bound(const input::Action bound_action) {
            return send({.type = ModalEvent::Type::ACTION, .data = ActionEvent{.action = bound_action}});
        }

        void selectPoint(const int index) {
            operation_.selected_point_ = index;
            operation_.syncPickedPointsToServices();
        }

        void startPendingInteraction(const std::optional<int> selected, const bool dragging) {
            operation_.selected_point_ = selected;
            services().scaleDistancePointPick().setSelectedPoint(selected);
            operation_.pending_pick_ = ScaleDistancePickPointOperator::PendingPick{
                .screen_position = {10.0, 20.0},
                .point_index = selected,
            };
            operation_.press_active_ = true;
            operation_.press_button_ = operation_.pick_button_;
            operation_.press_point_index_ = selected;
            operation_.drag_active_ = dragging;
        }

        bool samplePoint(const glm::vec3& world, const SplitViewPanelId panel, std::optional<int> index = std::nullopt) {
            return operation_.applyPointSample({.world = world, .panel = panel}, index);
        }

        std::optional<SplitViewPanelId> pickPanel() const { return operation_.pick_panel_; }

        void removePoint(const bool selected) {
            if (selected) {
                operation_.removeSelectedPoint();
            } else {
                operation_.removeLastPoint();
            }
        }

        void expectInteractionCancelled() {
            EXPECT_FALSE(operation_.pending_pick_.has_value());
            EXPECT_FALSE(operation_.press_active_);
            EXPECT_EQ(operation_.press_button_, operation_.pick_button_);
            EXPECT_FALSE(operation_.press_point_index_.has_value());
            EXPECT_FALSE(operation_.drag_active_);

            context_->setModalEvent({
                .type = ModalEvent::Type::MOUSE_BUTTON,
                .data = MouseButtonEvent{
                    .button = operation_.pick_button_,
                    .action = input::ACTION_RELEASE,
                    .mods = 0,
                    .position = {10.0, 20.0},
                },
            });
            OperatorProperties properties;
            EXPECT_EQ(operation_.modal(*context_, properties), OperatorResult::PASS_THROUGH);
        }

        std::unique_ptr<SceneManager> manager_;
        std::unique_ptr<OperatorContext> context_;
        ScaleDistancePickPointOperator operation_;
    };

    TEST_F(ScaleDistanceOperatorTest, MousePressPassesThroughAndFullPointReleaseIsConsumed) {
        setPickedPoints({{0, 0, 0}, {2, 0, 0}});
        EXPECT_EQ(mouse(input::ACTION_PRESS, {10, 20}), OperatorResult::PASS_THROUGH);
        EXPECT_EQ(mouse(input::ACTION_RELEASE, {10, 20}), OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(services().scaleDistancePointPick().getPickedPoints().size(), 2u);
        EXPECT_EQ(mouse(input::ACTION_PRESS, {10, 20}, true), OperatorResult::PASS_THROUGH);
        EXPECT_EQ(mouse(input::ACTION_PRESS, {10, 20}), OperatorResult::PASS_THROUGH);
        EXPECT_EQ(mouse(input::ACTION_RELEASE, {30, 20}), OperatorResult::PASS_THROUGH);
        EXPECT_EQ(services().scaleDistancePointPick().getPickedPoints().size(), 2u);
    }

    TEST_F(ScaleDistanceOperatorTest, DragReleaseResamplesFinalPosition) {
        setPickedPoints({{0, 0, 0}, {2, 0, 0}});
        startPendingInteraction(0, true);
        // Resolve the earlier pending pick, then distinguish the final release sample.
        EXPECT_EQ(send({}), OperatorResult::RUNNING_MODAL);
        auto& state = services().scaleDistancePointPick();
        ASSERT_NE(state.getStatusMessage(), nullptr);
        state.clearStatusMessage();
        EXPECT_EQ(mouse(input::ACTION_RELEASE, {30, 20}), OperatorResult::RUNNING_MODAL);
        // With no renderer, a new no-surface status proves release sampled again.
        EXPECT_NE(state.getStatusMessage(), nullptr);
        EXPECT_EQ(services().scaleDistancePointPick().getPickedPoints().size(), 2u);
        expectInteractionCancelled();
    }

    TEST_F(ScaleDistanceOperatorTest, NewPointsSelectAndKeepFirstPanelUntilCleared) {
        auto& state = services().scaleDistancePointPick();
        ASSERT_TRUE(samplePoint({0, 0, 0}, SplitViewPanelId::Right));
        EXPECT_EQ(state.getSelectedPoint(), 0);
        ASSERT_TRUE(samplePoint({2, 0, 0}, SplitViewPanelId::Left));
        EXPECT_EQ(state.getSelectedPoint(), 1);
        EXPECT_EQ(pickPanel(), SplitViewPanelId::Right);
        ASSERT_TRUE(samplePoint({3, 0, 0}, SplitViewPanelId::Left, 1));
        EXPECT_EQ(pickPanel(), SplitViewPanelId::Right);
        EXPECT_FALSE(samplePoint({4, 0, 0}, SplitViewPanelId::Left));
        EXPECT_EQ(state.getSelectedPoint(), 1);
        EXPECT_EQ(state.getPickedPoints().size(), 2u);
        requestAction(PointPickState::UiAction::Clear);
        EXPECT_FALSE(pickPanel());
        EXPECT_FALSE(state.getSelectedPoint());
        ASSERT_TRUE(samplePoint({1, 0, 0}, SplitViewPanelId::Left));
        EXPECT_EQ(pickPanel(), SplitViewPanelId::Left);
        EXPECT_EQ(state.getSelectedPoint(), 0);
        EXPECT_EQ(bound(input::Action::REMOVE_LAST_POINT), OperatorResult::RUNNING_MODAL);
        EXPECT_FALSE(pickPanel());
        EXPECT_FALSE(state.getSelectedPoint());
    }

    TEST_F(ScaleDistanceOperatorTest, CancelReturnsImmediatelyAndCleanupRunsInCancelCallback) {
        for (const bool selected : {false, true}) {
            setPickedPoints({{0, 0, 0}, {2, 0, 0}});
            if (selected)
                selectPoint(0);
            EXPECT_EQ(bound(input::Action::CANCEL_POLYGON), OperatorResult::CANCELLED);
            EXPECT_EQ(services().scaleDistancePointPick().getPickedPoints().size(), 2u);
            EXPECT_EQ(services().scaleDistancePointPick().getSelectedPoint().has_value(), selected);
            operation_.cancel(*context_);
            EXPECT_TRUE(services().scaleDistancePointPick().getPickedPoints().empty());
            EXPECT_FALSE(services().scaleDistancePointPick().getSelectedPoint());
        }
    }

    TEST_F(ScaleDistanceOperatorTest, ApplyScalesSceneAndCreatesSingleUndoStep) {
        const auto original_transform = targetTransform();

        // 2 points placed: distance = 2.0
        const glm::vec3 p0{0.0f, 0.0f, 0.0f};
        const glm::vec3 p1{2.0f, 0.0f, 0.0f};
        setPickedPoints({p0, p1});

        EXPECT_TRUE(operation_.poll(*context_));
        EXPECT_EQ(services().scaleDistancePointPick().getPickedPoints().size(), 2u);

        // Apply scale to target distance = 6.0 (scale factor = 3.0)
        const auto result = requestAction(PointPickState::UiAction::Apply, 6.0f);
        EXPECT_EQ(result, OperatorResult::FINISHED);

        const auto scaled_transform = targetTransform();
        EXPECT_NE(scaled_transform, original_transform);

        // Midpoint should remain anchored at (1, 0, 0)
        const glm::vec3 mid = (p0 + p1) * 0.5f;
        const glm::vec3 orig_world_mid = glm::vec3(original_transform * glm::vec4(mid, 1.0f));
        const glm::vec3 scaled_world_mid = glm::vec3(scaled_transform * glm::vec4(mid, 1.0f));
        EXPECT_NEAR(scaled_world_mid.x, orig_world_mid.x, 1e-4f);
        EXPECT_NEAR(scaled_world_mid.y, orig_world_mid.y, 1e-4f);
        EXPECT_NEAR(scaled_world_mid.z, orig_world_mid.z, 1e-4f);

        // Distance between points on transformed node should match target distance (6.0)
        const glm::vec3 scaled_p0 = glm::vec3(scaled_transform * glm::vec4(p0, 1.0f));
        const glm::vec3 scaled_p1 = glm::vec3(scaled_transform * glm::vec4(p1, 1.0f));
        EXPECT_NEAR(glm::length(scaled_p1 - scaled_p0), 6.0f, 1e-4f);

        // Points in services are cleared upon successful apply
        EXPECT_TRUE(services().scaleDistancePointPick().getPickedPoints().empty());

        // Exactly 1 undo step created
        EXPECT_EQ(undoHistory().undoCount(), 1u);
        EXPECT_TRUE(undoHistory().undo().success);
        EXPECT_EQ(targetTransform(), original_transform);

        // Redo reapplies the scale
        EXPECT_TRUE(undoHistory().redo().success);
        EXPECT_EQ(targetTransform(), scaled_transform);
    }

    TEST_F(ScaleDistanceOperatorTest, ClearRemovesPickedPoints) {
        setPickedPoints({{0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 3.0f}});
        EXPECT_EQ(services().scaleDistancePointPick().getPickedPoints().size(), 2u);

        requestAction(PointPickState::UiAction::Clear);
        EXPECT_TRUE(services().scaleDistancePointPick().getPickedPoints().empty());
    }

    TEST_F(ScaleDistanceOperatorTest, RemoveLastCancelsPendingPickAndPress) {
        setPickedPoints({{0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 3.0f}});
        startPendingInteraction(std::nullopt, false);

        removePoint(false);
        expectInteractionCancelled();

        const auto& points = services().scaleDistancePointPick().getPickedPoints();
        ASSERT_EQ(points.size(), 1u);
        EXPECT_EQ(points.front(), glm::vec3(0.0f));
    }

    TEST_F(ScaleDistanceOperatorTest, RemoveSelectedCancelsPendingPickAndDrag) {
        setPickedPoints({{0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 3.0f}});
        startPendingInteraction(0, true);

        removePoint(true);
        expectInteractionCancelled();

        const auto& points = services().scaleDistancePointPick().getPickedPoints();
        ASSERT_EQ(points.size(), 1u);
        EXPECT_EQ(points.front(), glm::vec3(1.0f, 2.0f, 3.0f));
        EXPECT_FALSE(services().scaleDistancePointPick().getSelectedPoint().has_value());
    }

    TEST_F(ScaleDistanceOperatorTest, RemoveLastCancelsPendingFirstPick) {
        startPendingInteraction(std::nullopt, false);

        removePoint(false);
        expectInteractionCancelled();

        EXPECT_TRUE(services().scaleDistancePointPick().getPickedPoints().empty());
    }

    TEST_F(ScaleDistanceOperatorTest, RemoveMissingSelectionCancelsPendingInteraction) {
        startPendingInteraction(std::nullopt, true);

        removePoint(true);
        expectInteractionCancelled();

        EXPECT_TRUE(services().scaleDistancePointPick().getPickedPoints().empty());
    }

    TEST_F(ScaleDistanceOperatorTest, PollRequiresValidSelectableTarget) {
        EXPECT_TRUE(operation_.poll(*context_));

        manager_->getScene().setNodeLocked("target", true);
        EXPECT_FALSE(operation_.poll(*context_));

        manager_->getScene().setNodeLocked("target", false);
        EXPECT_TRUE(operation_.poll(*context_));

        manager_->clearSelection();
        EXPECT_FALSE(operation_.poll(*context_));
    }

    TEST_F(ScaleDistanceOperatorTest, ModalActionRoutingAndPointRemoval) {
        setPickedPoints({{0.0f, 0.0f, 0.0f}, {3.0f, 4.0f, 0.0f}});
        EXPECT_EQ(services().scaleDistancePointPick().getPickedPoints().size(), 2u);

        // Fixed keys are now passed through because keys arrive as key map actions.
        for (const int fixed_key : {input::KEY_BACKSPACE, input::KEY_DELETE, input::KEY_ESCAPE}) {
            EXPECT_EQ(key(fixed_key), OperatorResult::PASS_THROUGH) << fixed_key;
        }
        EXPECT_EQ(services().scaleDistancePointPick().getPickedPoints().size(), 2u);

        selectPoint(0);
        EXPECT_EQ(bound(input::Action::DELETE_SELECTED), OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(services().scaleDistancePointPick().getPickedPoints().size(), 1u);

        EXPECT_FALSE(services().scaleDistancePointPick().getSelectedPoint());
        EXPECT_EQ(bound(input::Action::DELETE_SELECTED), OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(services().scaleDistancePointPick().getPickedPoints().size(), 1u);
        EXPECT_EQ(bound(input::Action::REMOVE_LAST_POINT), OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(services().scaleDistancePointPick().getPickedPoints().size(), 0u);

        EXPECT_EQ(bound(input::Action::UNDO), OperatorResult::PASS_THROUGH);
        EXPECT_EQ(bound(input::Action::CANCEL_POLYGON), OperatorResult::CANCELLED);
    }

    TEST(BuiltinOpTest, RoundTripToStringAndFromString) {
        for (size_t i = 0; i < static_cast<size_t>(BuiltinOp::_Count); ++i) {
            const auto op = static_cast<BuiltinOp>(i);
            const char* str = to_string(op);
            ASSERT_NE(str, nullptr);
            EXPECT_NE(std::string_view(str), "");
            const auto parsed = builtin_op_from_string(str);
            ASSERT_TRUE(parsed.has_value());
            EXPECT_EQ(*parsed, op);
            const char* label = builtin_op_label(op);
            ASSERT_NE(label, nullptr);
            EXPECT_NE(std::string_view(label), "");
        }
    }

    TEST(BuiltinToolTest, RoundTripToStringAndFromString) {
        for (size_t i = 0; i < static_cast<size_t>(BuiltinTool::_Count); ++i) {
            const auto tool = static_cast<BuiltinTool>(i);
            const char* str = to_string(tool);
            ASSERT_NE(str, nullptr);
            EXPECT_NE(std::string_view(str), "");
            const auto parsed = builtin_tool_from_string(str);
            ASSERT_TRUE(parsed.has_value());
            EXPECT_EQ(*parsed, tool);
            const char* label = builtin_tool_label(tool);
            ASSERT_NE(label, nullptr);
            EXPECT_NE(std::string_view(label), "");
        }
    }

} // namespace lfs::vis::op
