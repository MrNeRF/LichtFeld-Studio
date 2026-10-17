/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
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
        auto token = store.distance_scale_state_generation.subscribe([&](const auto&) { ++notifications; });
        const auto initial_gen = store.distance_scale_state_generation.get();

        // Identical points should not bump generation
        services().setScaleDistancePickedPoints(services().getScaleDistancePickedPoints());
        EXPECT_EQ(store.distance_scale_state_generation.get(), initial_gen);

        // Setting picked points notifies UI
        services().setScaleDistancePickedPoints({{0.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}});
        EXPECT_EQ(notifications, 0);
        EXPECT_TRUE(store.store().drain_dirty_into_frame());
        EXPECT_EQ(notifications, 1);
        EXPECT_EQ(services().getScaleDistancePickedPoints().size(), 2u);

        // Setting status message notifies UI
        services().setScaleDistanceStatusMessage("Invalid target distance", 2.0);
        EXPECT_TRUE(store.store().drain_dirty_into_frame());
        EXPECT_EQ(notifications, 2);

        // Clearing picked points clears status message and notifies UI
        services().clearScaleDistancePickedPoints();
        EXPECT_EQ(services().getScaleDistanceStatusMessage(), nullptr);
        EXPECT_TRUE(services().getScaleDistancePickedPoints().empty());
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
            lfs::core::event::bus().clear_all();
        }

        OperatorResult requestAction(Services::ScaleDistanceUiAction action, float target = 0.0f) {
            services().requestScaleDistanceUiAction(action, target);
            OperatorProperties properties;
            return operation_.modal(*context_, properties);
        }

        [[nodiscard]] glm::mat4 targetTransform() const {
            return manager_->getScene().getNode("target")->transform();
        }

        std::unique_ptr<SceneManager> manager_;
        std::unique_ptr<OperatorContext> context_;
        ScaleDistancePickPointOperator operation_;
    };

    TEST_F(ScaleDistanceOperatorTest, ApplyScalesSceneAndCreatesSingleUndoStep) {
        const auto original_transform = targetTransform();

        // 2 points placed: distance = 2.0
        const glm::vec3 p0{0.0f, 0.0f, 0.0f};
        const glm::vec3 p1{2.0f, 0.0f, 0.0f};
        operation_.setPickedPointsForTesting({p0, p1});

        EXPECT_TRUE(operation_.poll(*context_));
        EXPECT_EQ(services().getScaleDistancePickedPoints().size(), 2u);

        // Apply scale to target distance = 6.0 (scale factor = 3.0)
        const auto result = requestAction(Services::ScaleDistanceUiAction::Apply, 6.0f);
        EXPECT_EQ(result, OperatorResult::RUNNING_MODAL);

        const auto scaled_transform = targetTransform();
        EXPECT_NE(scaled_transform, original_transform);

        // Pointers in services are transformed to new positions
        const auto& new_pts = services().getScaleDistancePickedPoints();
        ASSERT_EQ(new_pts.size(), 2u);
        EXPECT_NEAR(glm::length(new_pts[1] - new_pts[0]), 6.0f, 1e-4f);

        // Midpoint remains identical
        const glm::vec3 orig_mid = (p0 + p1) * 0.5f;
        const glm::vec3 new_mid = (new_pts[0] + new_pts[1]) * 0.5f;
        EXPECT_NEAR(new_mid.x, orig_mid.x, 1e-4f);
        EXPECT_NEAR(new_mid.y, orig_mid.y, 1e-4f);
        EXPECT_NEAR(new_mid.z, orig_mid.z, 1e-4f);

        // Exactly 1 undo step created
        EXPECT_EQ(undoHistory().undoCount(), 1u);
        EXPECT_TRUE(undoHistory().undo().success);
        EXPECT_EQ(targetTransform(), original_transform);

        // Redo reapplies the scale
        EXPECT_TRUE(undoHistory().redo().success);
        EXPECT_EQ(targetTransform(), scaled_transform);
    }

    TEST_F(ScaleDistanceOperatorTest, ClearRemovesPickedPoints) {
        operation_.setPickedPointsForTesting({{0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 3.0f}});
        EXPECT_EQ(services().getScaleDistancePickedPoints().size(), 2u);

        requestAction(Services::ScaleDistanceUiAction::Clear);
        EXPECT_TRUE(services().getScaleDistancePickedPoints().empty());
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

} // namespace lfs::vis::op

