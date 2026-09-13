/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/panel_input_utils.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rml_progress_overlay.hpp"
#include "gui/rml_viewport_overlay.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_pointer_dispatch.hpp"
#include "gui/string_keys.hpp"
#include "input/frame_input_buffer.hpp"
#include "input/key_codes.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "operator/ops/depth_window_ops.hpp"
#include "rendering/render_pass.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/rendering_types.hpp"
#include "rendering/viewport_request_builder.hpp"
#include "selection/selection_service.hpp"
#include "tools/selection_tool.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/rendering/depth_window_state.hpp"
#include "visualizer_impl.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <RmlUi/Core/EventListener.h>
#include <RmlUi/Core/Factory.h>
#include <RmlUi/Core/RenderInterface.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace lfs::vis {

    namespace {

        using lfs::vis::op::ModalEvent;
        using lfs::vis::op::OperatorProperties;
        using lfs::vis::op::OperatorResult;

        DepthWindowState makeWindow(const float depth_near_val,
                                    const float depth_far_val,
                                    const float scale_x = 0.35f,
                                    const float scale_y = 0.35f,
                                    const float offset_x = 0.0f,
                                    const float offset_y = 0.0f) {
            return {
                .near_plane = depth_near_val,
                .far_plane = depth_far_val,
                .scale_x = scale_x,
                .scale_y = scale_y,
                .offset_x = offset_x,
                .offset_y = offset_y,
            };
        }

        std::unique_ptr<lfs::core::SplatData> make_test_splat(const std::vector<float>& xyz) {
            const size_t count = xyz.size() / 3;
            auto means = lfs::core::Tensor::from_vector(xyz, {count, size_t{3}}, lfs::core::Device::CUDA)
                             .to(lfs::core::DataType::Float32);
            auto sh0 = lfs::core::Tensor::zeros({count, size_t{1}, size_t{3}}, lfs::core::Device::CUDA,
                                                lfs::core::DataType::Float32);
            auto shN = lfs::core::Tensor::zeros({count, size_t{3}, size_t{3}}, lfs::core::Device::CUDA,
                                                lfs::core::DataType::Float32);
            auto scaling = lfs::core::Tensor::zeros({count, size_t{3}}, lfs::core::Device::CUDA,
                                                    lfs::core::DataType::Float32);
            std::vector<float> rotation_data(count * 4, 0.0f);
            for (size_t i = 0; i < count; ++i) {
                rotation_data[i * 4] = 1.0f;
            }
            auto rotation = lfs::core::Tensor::from_vector(rotation_data, {count, size_t{4}}, lfs::core::Device::CUDA)
                                .to(lfs::core::DataType::Float32);
            auto opacity = lfs::core::Tensor::zeros({count, size_t{1}}, lfs::core::Device::CUDA,
                                                    lfs::core::DataType::Float32);
            return std::make_unique<lfs::core::SplatData>(
                1, std::move(means), std::move(sh0), std::move(shN), std::move(scaling), std::move(rotation),
                std::move(opacity), 1.0f);
        }

        std::shared_ptr<lfs::core::Tensor> make_screen_positions(const std::vector<float>& xy) {
            return std::make_shared<lfs::core::Tensor>(
                lfs::core::Tensor::from_vector(xy, {xy.size() / 2, size_t{2}}, lfs::core::Device::CUDA)
                    .to(lfs::core::DataType::Float32));
        }

        void arm_viewer_camera_depth_band(RenderingManager& rendering_manager,
                                          const float near_plane,
                                          const float far_plane) {
            auto settings = rendering_manager.getSettings();
            settings.depth_filter_enabled = true;
            // Camera-depth band [near_plane, far_plane]: keeps splat0 (~8.54), rejects splat1 (~9.21).
            settings.depth_filter_min = {-0.5f, -0.5f, -far_plane};
            settings.depth_filter_max = {0.5f, 0.5f, -near_plane};
            // Full-viewport window so only depth (or the stroked panel band) decides.
            settings.depth_filter_scale_x = 1.0f;
            settings.depth_filter_scale_y = 1.0f;
            settings.depth_filter_offset_x = 0.0f;
            settings.depth_filter_offset_y = 0.0f;
            rendering_manager.updateSettings(settings);
        }

        std::vector<uint8_t> selection_values(const SceneManager& scene_manager) {
            const auto mask = scene_manager.getScene().getSelectionMask();
            if (!mask || !mask->is_valid()) {
                return {};
            }
            return mask->cpu().to_vector_uint8();
        }

        DepthWindowState projectionDepthWindow(const RenderingManager& rendering_manager) {
            const auto settings = rendering_manager.getSettings();
            return {
                .near_plane = -settings.depth_filter_max.z,
                .far_plane = -settings.depth_filter_min.z,
                .scale_x = settings.depth_filter_scale_x,
                .scale_y = settings.depth_filter_scale_y,
                .offset_x = settings.depth_filter_offset_x,
                .offset_y = settings.depth_filter_offset_y,
            };
        }

        std::vector<uint8_t> finishRingStrokeAt(SceneManager& scene_manager,
                                                SelectionService& service,
                                                const float x,
                                                const float y) {
            SelectionFilterState filters{};
            filters.depth_filter = true;
            EXPECT_TRUE(service.beginInteractiveSelection(
                SelectionShape::Rings,
                SelectionMode::Replace,
                {x, y},
                0.0f,
                filters));
            const auto result = service.finishInteractiveSelection();
            EXPECT_TRUE(result.success) << result.error;
            auto values = selection_values(scene_manager);
            if (values.empty()) {
                values = {0, 0};
            }
            return values;
        }

        ModalEvent mouse_move(const double x, const double y) {
            return ModalEvent{
                .type = ModalEvent::Type::MOUSE_MOVE,
                .data = MouseMoveEvent{
                    .position = {x, y},
                    .delta = {0.0, 0.0},
                },
            };
        }

        ModalEvent mouse_release(const double x, const double y) {
            return ModalEvent{
                .type = ModalEvent::Type::MOUSE_BUTTON,
                .data = MouseButtonEvent{
                    .button = static_cast<int>(lfs::vis::input::AppMouseButton::LEFT),
                    .action = lfs::vis::input::ACTION_RELEASE,
                    .mods = lfs::vis::input::KEYMOD_SHIFT | lfs::vis::input::KEYMOD_ALT,
                    .position = {x, y},
                },
            };
        }

    } // namespace

    class DepthWindowPanelsInteractionTest : public ::testing::Test {
    protected:
        static constexpr int kViewerWidth = 400;
        static constexpr int kViewerHeight = 200;

        void SetUp() override {
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
            lfs::vis::services().clear();
            lfs::vis::op::undoHistory().clear();

            options_.show_startup_overlay = false;
            options_.width = kViewerWidth;
            options_.height = kViewerHeight;
            viewer_ = std::make_unique<VisualizerImpl>(options_);
            viewer_->initializeTools();

            rendering_manager_ = viewer_->getRenderingManager();
            selection_tool_ = viewer_->getSelectionTool();
            ASSERT_NE(rendering_manager_, nullptr);
            ASSERT_NE(selection_tool_, nullptr);

            viewer_->getSceneManager()->getScene().addSplat(
                "depth_window_interaction",
                make_test_splat({
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f,
                    0.0f,
                    0.0f,
                }));
            viewer_->getSceneManager()->initSelectionService();
            if (auto* const service = viewer_->getSceneManager()->getSelectionService()) {
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

        void enterIndependentDual() {
            lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
            ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::IndependentDual);
        }

        [[nodiscard]] OperatorProperties depthDragProps(const double x, const double y) const {
            OperatorProperties props;
            props.set("x", x);
            props.set("y", y);
            props.set("viewport_x", 0.0f);
            props.set("viewport_y", 0.0f);
            props.set("viewport_width", static_cast<float>(options_.width));
            props.set("viewport_height", static_cast<float>(options_.height));
            props.set("modifiers", lfs::vis::input::KEYMOD_SHIFT | lfs::vis::input::KEYMOD_ALT);
            return props;
        }

        bool startDepthDrag(const double x, const double y) {
            auto props = depthDragProps(x, y);
            const auto result =
                lfs::vis::op::operators().invoke(lfs::vis::op::BuiltinOp::DepthWindowDrag, &props);
            return result.status == OperatorResult::RUNNING_MODAL;
        }

        bool commitDepthDraw(const double start_x,
                             const double start_y,
                             const double end_x,
                             const double end_y) {
            if (!startDepthDrag(start_x, start_y)) {
                return false;
            }
            if (lfs::vis::op::operators().dispatchModalEvent(mouse_move(end_x, end_y)) !=
                OperatorResult::RUNNING_MODAL) {
                return false;
            }
            return lfs::vis::op::operators().dispatchModalEvent(mouse_release(end_x, end_y)) ==
                   OperatorResult::FINISHED;
        }

        ViewerOptions options_{};
        std::unique_ptr<VisualizerImpl> viewer_;
        RenderingManager* rendering_manager_ = nullptr;
        tools::SelectionTool* selection_tool_ = nullptr;
    };

    TEST_F(DepthWindowPanelsInteractionTest, StaleEpochUndoEntryNoOpsAndReportsExpiredLabel) {
        enterIndependentDual();
        ASSERT_EQ(rendering_manager_->depthWindowModeEpoch(), 1u);
        rendering_manager_->setDepthWindowSync(false);
        ASSERT_TRUE(commitDepthDraw(30.0, 30.0, 80.0, 80.0));
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);

        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        EXPECT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);
        ASSERT_EQ(rendering_manager_->depthWindowModeEpoch(), 2u);

        const auto items = lfs::vis::op::undoHistory().undoItems();
        ASSERT_FALSE(items.empty());
        EXPECT_EQ(items.front().metadata.id, "selection.depth_window_drag");
        // Compare localized keys because no locale is loaded here; English text would
        // be locale-dependent. Also reject the live key to prove the expired branch
        // ran.
        EXPECT_EQ(items.front().metadata.label,
                  LOC(lichtfeld::Strings::Selection::HISTORY_DEPTH_WINDOW_EXPIRED));
        EXPECT_NE(items.front().metadata.label,
                  LOC(lichtfeld::Strings::Selection::HISTORY_DEPTH_WINDOW_DRAG));

        const auto left_before = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right_before = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);
        const auto left_after = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right_after = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_EQ(left_before, left_after);
        EXPECT_EQ(right_before, right_after);
    }

    // Exercise real focus and sync setters in both directions with distinct values in
    // every field. An other-over-focused copy or swapped slot mapping fails the
    // Right-focused case. This pins focused-wins copying independently of the upstream
    // choice of focus at toggle time.
    TEST_F(DepthWindowPanelsInteractionTest, SyncOnCopiesFocusedOverOtherInBothDirections) {
        enterIndependentDual();

        // Focused = RIGHT: Right's window must survive in BOTH slots.
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left,
                                                   makeWindow(1.0f, 10.0f, 0.31f, 0.32f, 0.10f, 0.11f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right,
                                                   makeWindow(2.0f, 20.0f, 0.62f, 0.63f, -0.20f, -0.21f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);

        rendering_manager_->setDepthWindowSync(true);
        ASSERT_TRUE(rendering_manager_->getDepthWindowSync());
        const auto right_focused_left = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right_focused_right = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_EQ(right_focused_left, right_focused_right);
        EXPECT_FLOAT_EQ(right_focused_left.near_plane, 2.0f);
        EXPECT_FLOAT_EQ(right_focused_left.far_plane, 20.0f);
        EXPECT_FLOAT_EQ(right_focused_left.scale_x, 0.62f);
        EXPECT_FLOAT_EQ(right_focused_left.scale_y, 0.63f);
        EXPECT_FLOAT_EQ(right_focused_left.offset_x, -0.20f);
        EXPECT_FLOAT_EQ(right_focused_left.offset_y, -0.21f);

        // Focused = LEFT, same setter, opposite direction: Left's window wins.
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left,
                                                   makeWindow(3.0f, 30.0f, 0.41f, 0.42f, 0.30f, 0.31f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right,
                                                   makeWindow(4.0f, 40.0f, 0.52f, 0.53f, -0.40f, -0.41f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);

        rendering_manager_->setDepthWindowSync(true);
        ASSERT_TRUE(rendering_manager_->getDepthWindowSync());
        const auto left_focused_left = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto left_focused_right = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_EQ(left_focused_left, left_focused_right);
        EXPECT_FLOAT_EQ(left_focused_right.near_plane, 3.0f);
        EXPECT_FLOAT_EQ(left_focused_right.far_plane, 30.0f);
        EXPECT_FLOAT_EQ(left_focused_right.scale_x, 0.41f);
        EXPECT_FLOAT_EQ(left_focused_right.scale_y, 0.42f);
        EXPECT_FLOAT_EQ(left_focused_right.offset_x, 0.30f);
        EXPECT_FLOAT_EQ(left_focused_right.offset_y, 0.31f);
    }

    TEST_F(DepthWindowPanelsInteractionTest, SyncOnUndoRestoresDifferingPanelsAtomically) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, makeWindow(1.0f, 10.0f, 0.31f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.62f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);

        rendering_manager_->setDepthWindowSync(true);
        EXPECT_TRUE(rendering_manager_->getDepthWindowSync());
        EXPECT_FLOAT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right).near_plane, 1.0f);
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);

        EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());
        const auto left = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_FLOAT_EQ(left.near_plane, 1.0f);
        EXPECT_FLOAT_EQ(right.near_plane, 2.0f);
        EXPECT_FLOAT_EQ(left.scale_x, 0.31f);
        EXPECT_FLOAT_EQ(right.scale_x, 0.62f);

        EXPECT_TRUE(lfs::vis::op::undoHistory().redo().success);
        EXPECT_TRUE(rendering_manager_->getDepthWindowSync());
        EXPECT_FLOAT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left).near_plane, 1.0f);
        EXPECT_FLOAT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right).near_plane, 1.0f);
    }

    TEST_F(DepthWindowPanelsInteractionTest, UndoDrawPublishesPanelAddressedGeneration) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);

        const auto generation_before = lfs::vis::app_store().depth_window_draw_generation.get();
        ASSERT_TRUE(commitDepthDraw(260.0, 40.0, 320.0, 120.0));
        EXPECT_EQ(lfs::vis::app_store().depth_window_draw_generation.get(), generation_before + 1u);
        const auto commit_after_draw = lfs::vis::app_store().depth_window_draw_commit.get();
        EXPECT_EQ(commit_after_draw.panel, SplitViewPanelId::Right);
        EXPECT_EQ(commit_after_draw.generation, generation_before + 1u);

        EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);
        const auto commit_after_undo = lfs::vis::app_store().depth_window_draw_commit.get();
        EXPECT_EQ(commit_after_undo.panel, SplitViewPanelId::Right);
        EXPECT_EQ(lfs::vis::app_store().depth_window_draw_generation.get(), generation_before + 2u);
        EXPECT_EQ(commit_after_undo.generation, generation_before + 2u);
    }

    TEST_F(DepthWindowPanelsInteractionTest, CommittedDragFocusesDraggedPanel) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        EXPECT_EQ(rendering_manager_->getFocusedSplitPanel(), SplitViewPanelId::Left);
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_release(90.0, 90.0)),
                  OperatorResult::FINISHED);
        EXPECT_EQ(rendering_manager_->getFocusedSplitPanel(), SplitViewPanelId::Left);
    }

    TEST_F(DepthWindowPanelsInteractionTest, MidDragFocusChangeDoesNotReroutePanel) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, makeWindow(1.0f, 10.0f, 0.30f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.70f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);
        const auto right_before = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(100.0, 100.0)),
                  OperatorResult::RUNNING_MODAL);
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(120.0, 110.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_release(120.0, 110.0)),
                  OperatorResult::FINISHED);

        const auto right_after = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_FLOAT_EQ(right_before.near_plane, right_after.near_plane);
        EXPECT_FLOAT_EQ(right_before.far_plane, right_after.far_plane);
        EXPECT_FLOAT_EQ(right_before.scale_x, right_after.scale_x);
        EXPECT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left).scale_x, 0.30f);
    }

    TEST_F(DepthWindowPanelsInteractionTest, LatchReleaseLeavesOtherPanelUntouched) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, makeWindow(1.0f, 10.0f, 0.30f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.70f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);
        const auto right_before = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        const auto left_before = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);

        ASSERT_TRUE(commitDepthDraw(40.0, 40.0, 110.0, 110.0));

        const auto right_after = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_FLOAT_EQ(right_before.near_plane, right_after.near_plane);
        EXPECT_FLOAT_EQ(right_before.far_plane, right_after.far_plane);
        EXPECT_FLOAT_EQ(right_before.scale_x, right_after.scale_x);

        // The release reapply must not disturb the DRAGGED panel's depth band:
        // only its rectangle changes.
        const auto left_after = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        EXPECT_FLOAT_EQ(left_after.near_plane, left_before.near_plane);
        EXPECT_FLOAT_EQ(left_after.far_plane, left_before.far_plane);
        EXPECT_NE(left_after.scale_x, left_before.scale_x);
    }

    TEST_F(DepthWindowPanelsInteractionTest, LeaveIndependentDualWithActiveDragCancelsBeforeCollapse) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        const auto pre_drag = makeWindow(1.5f, 11.0f, 0.42f);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, pre_drag);
        rendering_manager_->setDepthWindowForPanel(
            SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.70f));
        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_TRUE(rendering_manager_->depthWindowDragPreview());
        ASSERT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left).scale_x,
                  pre_drag.scale_x);

        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();

        EXPECT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);
        EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
        EXPECT_FALSE(lfs::vis::op::operators().hasModalOperator());
        const auto restored = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        EXPECT_FLOAT_EQ(restored.near_plane, pre_drag.near_plane);
        EXPECT_FLOAT_EQ(restored.far_plane, pre_drag.far_plane);
        EXPECT_FLOAT_EQ(restored.scale_x, pre_drag.scale_x);
        EXPECT_FLOAT_EQ(restored.scale_y, pre_drag.scale_y);
    }

    TEST_F(DepthWindowPanelsInteractionTest, StrokeUsesContextPanelBandNotFocusedProjection) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        arm_viewer_camera_depth_band(*rendering_manager_, 8.0f, 8.875f);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, makeWindow(8.0f, 8.875f, 1.0f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(9.0f, 9.5f, 1.0f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);

        auto* const service = viewer_->getSceneManager()->getSelectionService();
        ASSERT_NE(service, nullptr);
        service->setTestingHoveredGaussianId(0);
        EXPECT_EQ(finishRingStrokeAt(*viewer_->getSceneManager(), *service, 50.0f, 50.0f),
                  (std::vector<uint8_t>{1, 0}));

        rendering_manager_->setDepthWindowForPanel(
            SplitViewPanelId::Left, makeWindow(99.0f, 100.0f, 0.5f));
        EXPECT_EQ(finishRingStrokeAt(*viewer_->getSceneManager(), *service, 50.0f, 50.0f),
                  (std::vector<uint8_t>{0, 0}));

        service->setTestingPanel(SplitViewPanelId::Right);
        rendering_manager_->setDepthWindowForPanel(
            SplitViewPanelId::Left, makeWindow(99.0f, 100.0f, 0.5f));
        rendering_manager_->setDepthWindowForPanel(
            SplitViewPanelId::Right, makeWindow(8.0f, 8.875f, 1.0f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);
        EXPECT_EQ(finishRingStrokeAt(*viewer_->getSceneManager(), *service, 300.0f, 50.0f),
                  (std::vector<uint8_t>{1, 0}));

        rendering_manager_->setDepthWindowForPanel(
            SplitViewPanelId::Right, makeWindow(99.0f, 100.0f, 0.5f));
        EXPECT_EQ(finishRingStrokeAt(*viewer_->getSceneManager(), *service, 300.0f, 50.0f),
                  (std::vector<uint8_t>{0, 0}));
        service->setTestingPanel(SplitViewPanelId::Left);
    }
    TEST_F(DepthWindowPanelsInteractionTest, EnteringIndependentDualWithActiveGlobalDragCancelsAndSeedsProjection) {
        auto settings = rendering_manager_->getSettings();
        settings.depth_filter_scale_x = 0.41f;
        settings.depth_filter_scale_y = 0.41f;
        settings.depth_filter_min = {-0.5f, -0.5f, -10.0f};
        settings.depth_filter_max = {0.5f, 0.5f, -1.0f};
        rendering_manager_->updateSettings(settings);
        const auto pre_drag = projectionDepthWindow(*rendering_manager_);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);

        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();

        EXPECT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::IndependentDual);
        EXPECT_FALSE(lfs::vis::op::operators().hasModalOperator());
        const auto left = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_FLOAT_EQ(left.scale_x, pre_drag.scale_x);
        EXPECT_FLOAT_EQ(left.near_plane, pre_drag.near_plane);
        EXPECT_FLOAT_EQ(left.far_plane, pre_drag.far_plane);
        EXPECT_EQ(left, right);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
    }

    TEST_F(DepthWindowPanelsInteractionTest, ReleaseAfterEpochBumpCancelsInsteadOfCommitting) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        const auto pre_drag = makeWindow(1.0f, 10.0f, 0.33f);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, pre_drag);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left).scale_x,
                  pre_drag.scale_x);

        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        EXPECT_FALSE(lfs::vis::op::operators().hasModalOperator());
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
        const auto restored = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        EXPECT_FLOAT_EQ(restored.scale_x, pre_drag.scale_x);
        EXPECT_FLOAT_EQ(restored.near_plane, pre_drag.near_plane);

        // A refused commit leaves NO trace: no undo entry, and no draw-commit or
        // draw-generation publication either.
        const auto generation_before_release =
            lfs::vis::app_store().depth_window_draw_generation.get();
        const auto commit_before_release = lfs::vis::app_store().depth_window_draw_commit.get();
        (void)lfs::vis::op::operators().dispatchModalEvent(mouse_release(90.0, 90.0));
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
        EXPECT_FLOAT_EQ(
            rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left).scale_x,
            pre_drag.scale_x);
        EXPECT_EQ(lfs::vis::app_store().depth_window_draw_generation.get(),
                  generation_before_release);
        EXPECT_EQ(lfs::vis::app_store().depth_window_draw_commit.get(), commit_before_release);
    }

    // The real registry invokes B before destroying A. B takes Right while A owns Left,
    // so A must restore Left rather than leave its uncommitted preview; B must then
    // commit normally.
    TEST_F(DepthWindowPanelsInteractionTest, CrossPanelRegistryReplacementRestoresReplacedPanel) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left,
                                                   makeWindow(1.0f, 10.0f, 0.31f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right,
                                                   makeWindow(2.0f, 20.0f, 0.62f));
        const auto left_pre = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right_pre = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);

        // Invoke the replacement on Right through the real registry. Its press is clear
        // of handles, so B is a fresh draw that has not latched when A is destroyed.
        ASSERT_TRUE(startDepthDrag(210.0, 20.0));
        EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_pre);

        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(300.0, 120.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_release(300.0, 120.0)),
                  OperatorResult::FINISHED);

        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);
        EXPECT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_pre);
        EXPECT_EQ(rendering_manager_->getFocusedSplitPanel(), SplitViewPanelId::Right);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
    }

    // On same-panel replacement, B takes Left and A cannot restore under the
    // incoming-baseline contract. The panel backup must still be pre-A state, so
    // leaving independent-dual mid-B folds neither preview.
    TEST_F(DepthWindowPanelsInteractionTest, SamePanelRegistryReplacementKeepsPreDragBackup) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left,
                                                   makeWindow(1.0f, 10.0f, 0.31f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);
        const auto left_pre = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        const auto a_preview = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        ASSERT_NE(a_preview, left_pre);

        // Same-panel replacement: A is destroyed and must NOT restore, so B
        // inherits A's preview as its own baseline. The press is clear of A's
        // preview rectangle, so B is a fresh draw of its own.
        ASSERT_TRUE(startDepthDrag(20.0, 150.0));
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), a_preview);
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(120.0, 190.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), a_preview);

        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);

        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), left_pre);
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), left_pre);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
        EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
    }

    // Project restore advances the epoch without a split transition or modal teardown.
    // Releasing the still-live drag must detect its expired epoch and leave no trace.
    TEST_F(DepthWindowPanelsInteractionTest, ReleaseOnLiveModalAfterEpochBumpPublishesNothing) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);

        const auto epoch_before = rendering_manager_->depthWindowModeEpoch();
        rendering_manager_->restoreDepthWindowStateFromProject();
        ASSERT_NE(rendering_manager_->depthWindowModeEpoch(), epoch_before);
        ASSERT_TRUE(lfs::vis::op::operators().hasModalOperator());

        const auto generation_before = lfs::vis::app_store().depth_window_draw_generation.get();
        const auto commit_before = lfs::vis::app_store().depth_window_draw_commit.get();
        EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_release(90.0, 90.0)),
                  OperatorResult::CANCELLED);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
        EXPECT_EQ(lfs::vis::app_store().depth_window_draw_generation.get(), generation_before);
        EXPECT_EQ(lfs::vis::app_store().depth_window_draw_commit.get(), commit_before);
        EXPECT_FALSE(lfs::vis::op::operators().hasModalOperator());
        EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
    }

    // Manager-level companion to the two registry tests above: brackets opened
    // directly, so BOTH panels can carry a recorded backup at once, pinning
    // that a mode transition restores every one of them before it collapses.
    TEST_F(DepthWindowPanelsInteractionTest, TransitionRestoresEveryRecordedBackupBeforeCollapse) {
        const auto run = [&](const SplitViewPanelId collapse_focus) {
            enterIndependentDual();
            rendering_manager_->setDepthWindowSync(false);
            rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left,
                                                       makeWindow(1.0f, 10.0f, 0.31f));
            rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right,
                                                       makeWindow(2.0f, 20.0f, 0.62f));
            const auto left_pre = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
            const auto right_pre =
                rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);

            ASSERT_TRUE(startDepthDrag(40.0, 40.0));
            ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                      OperatorResult::RUNNING_MODAL);
            ASSERT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);

            // Open the second drag through the operator's manager API and preview the
            // other panel via applyDepthWindowForPanelIfEpoch. The public setter would
            // be a legitimate non-drag write that supersedes the backup this case
            // needs.
            std::uint64_t right_token = 0;
            rendering_manager_->beginDepthWindowDrag(SplitViewPanelId::Right, right_token);
            ASSERT_TRUE(rendering_manager_->applyDepthWindowForPanelIfEpoch(
                SplitViewPanelId::Right, makeWindow(3.0f, 30.0f, 0.90f),
                rendering_manager_->depthWindowModeEpoch(), right_token));
            ASSERT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right),
                      right_pre);

            rendering_manager_->setFocusedSplitPanel(collapse_focus);
            lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}
                .emit();
            ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);

            const auto expected =
                collapse_focus == SplitViewPanelId::Left ? left_pre : right_pre;
            EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), expected);
            EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), expected);
            EXPECT_EQ(projectionDepthWindow(*rendering_manager_), expected);

            // Balance the manually opened bracket; the transition already dropped
            // the backups it consumed.
            rendering_manager_->endDepthWindowDrag(SplitViewPanelId::Right, right_token);
            EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
            lfs::vis::op::undoHistory().clear();
        };

        // Collapsing to Right is the load-bearing case: only the manager's
        // per-panel backup can restore a panel no operator owns.
        run(SplitViewPanelId::Right);
        run(SplitViewPanelId::Left);

        // The transition CONSUMED the backups: a later leave collapses live
        // state, never a stale pre-drag window inherited across the boundary.
        enterIndependentDual();
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left,
                                                   makeWindow(5.0f, 50.0f, 0.55f));
        const auto fresh = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);
        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), fresh);
    }

    // GT comparison suspends the depth filter entirely, so entering or leaving it
    // is a lifetime boundary: the drag is cancelled BEFORE the mode change and
    // the epoch moves on, so anything that survived the cancel race is refused.
    TEST_F(DepthWindowPanelsInteractionTest, GtComparisonIsAnEpochBoundaryAndCancelsActiveDrag) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left,
                                                   makeWindow(1.0f, 10.0f, 0.33f));
        const auto left_pre = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_TRUE(rendering_manager_->depthWindowDragPreview());
        ASSERT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);
        const auto epoch_before = rendering_manager_->depthWindowModeEpoch();

        lfs::core::events::cmd::ToggleGTComparison{}.emit();

        EXPECT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::GTComparison);
        EXPECT_FALSE(lfs::vis::op::operators().hasModalOperator());
        EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
        EXPECT_NE(rendering_manager_->depthWindowModeEpoch(), epoch_before);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);

        // A write from the drag that survived the cancel race is refused, and
        // changes nothing.
        const auto after_gt_left = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto after_gt_right =
            rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_FALSE(rendering_manager_->applyDepthWindowForPanelIfEpoch(
            SplitViewPanelId::Left, makeWindow(3.0f, 30.0f, 0.90f), epoch_before,
            /*drag_token=*/0));
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), after_gt_left);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right),
                  after_gt_right);

        lfs::core::events::cmd::ToggleGTComparison{}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);

        // Check both global/GT boundary epochs separately. Comparing only the
        // round-trip endpoints would pass even if leaving GT failed to advance its
        // epoch.
        const auto global_epoch = rendering_manager_->depthWindowModeEpoch();
        lfs::core::events::cmd::ToggleGTComparison{}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::GTComparison);
        const auto gt_epoch = rendering_manager_->depthWindowModeEpoch();
        EXPECT_NE(gt_epoch, global_epoch);
        lfs::core::events::cmd::ToggleGTComparison{}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);
        const auto back_epoch = rendering_manager_->depthWindowModeEpoch();
        EXPECT_NE(back_epoch, gt_epoch);
        EXPECT_NE(back_epoch, global_epoch);
    }

    // GT suspends filtering and preserves distinct dormant L/R windows. Start focused
    // Right: collapsing on entry or seeding from global on return would make the slots
    // equal. Exact restored values catch both failures. Also require no panel filter
    // requests, overlay or drag during GT; hiding the drawing alone must not pass.
    TEST_F(DepthWindowPanelsInteractionTest, GtComparisonKeepsBothPanelWindowsDormantAndRestoresThem) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(
            SplitViewPanelId::Left, makeWindow(1.0f, 10.0f, 0.31f, 0.32f, 0.10f, 0.11f));
        rendering_manager_->setDepthWindowForPanel(
            SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.62f, 0.63f, -0.20f, -0.21f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);

        const auto left_pre = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right_pre = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        ASSERT_NE(left_pre, right_pre);
        const auto lineage_pre = rendering_manager_->getDepthWindowCollapseRecord().generation;

        lfs::core::events::cmd::ToggleGTComparison{}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::GTComparison);

        // DURING GT the state is untouched, and the filter it describes is
        // fully suspended.
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_pre);
        {
            const auto snapshot = rendering_manager_->getDepthWindowOverlaySnapshot();
            EXPECT_FALSE(snapshot.independent_dual_active);
            EXPECT_TRUE(op::depthWindowOverlaySuppressed(rendering_manager_->isGTComparisonActive()));

            auto settings = rendering_manager_->getSettings();
            ASSERT_TRUE(settings.depth_filter_enabled);
            FrameContext ctx{
                .viewport = viewer_->getViewport(),
                .settings = settings,
                .render_size = {kViewerWidth, kViewerHeight},
                .panel_depth_windows = snapshot.panel_windows,
            };
            for (const auto panel : {SplitViewPanelId::Left, SplitViewPanelId::Right}) {
                const auto request = buildViewportRenderRequest(
                    ctx, {kViewerWidth, kViewerHeight}, &ctx.viewport, panel);
                EXPECT_FALSE(request.filters.screen_window.has_value());
                EXPECT_FALSE(request.filters.view_volume.has_value());
            }
            // No drag can start either, so no preview lane can revive it.
            EXPECT_FALSE(startDepthDrag(40.0, 40.0));
            EXPECT_FALSE(lfs::vis::op::operators().hasModalOperator());
        }

        // A GLOBAL depth write while GT is active fans out through BOTH slots
        // outside independent-dual, so the dormant pair cannot live in the
        // slots alone - this is what proves it is parked elsewhere.
        auto gt_settings = rendering_manager_->getSettings();
        gt_settings.depth_filter_max.z = -7.0f;
        gt_settings.depth_filter_min.z = -70.0f;
        gt_settings.depth_filter_scale_x = 0.77f;
        gt_settings.depth_filter_scale_y = 0.78f;
        gt_settings.depth_filter_offset_x = 0.07f;
        gt_settings.depth_filter_offset_y = 0.08f;
        rendering_manager_->updateSettings(gt_settings);

        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::IndependentDual);

        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_pre);
        // The split service resets focus to Left on entering independent-dual,
        // so the projection follows LEFT's restored window.
        EXPECT_EQ(rendering_manager_->getFocusedSplitPanel(), SplitViewPanelId::Left);
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), left_pre);
        // Nothing a per-panel reference describes was destroyed, so the
        // lineage channel stamped nothing across either leg.
        EXPECT_EQ(rendering_manager_->getDepthWindowCollapseRecord().generation, lineage_pre);
    }

    // Drag the focused Right panel so its preview occupies both slot and projection. GT
    // entry resets focus Left, but must fold Right's pre-drag window into both
    // locations. Otherwise GT -> Disabled promotes the abandoned preview to global
    // state. Assert the pre-drag Right window on both legs.
    TEST_F(DepthWindowPanelsInteractionTest, GtEntryWithFocusedRightDragLeavesNoTransientProjection) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(
            SplitViewPanelId::Left, makeWindow(1.0f, 10.0f, 0.31f, 0.32f, 0.10f, 0.11f));
        rendering_manager_->setDepthWindowForPanel(
            SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.62f, 0.63f, -0.20f, -0.21f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);

        const auto left_pre = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right_pre = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        ASSERT_NE(left_pre, right_pre);
        // Focused, so the projection tracks RIGHT.
        ASSERT_EQ(projectionDepthWindow(*rendering_manager_), right_pre);

        // A live drag on the focused RIGHT panel, through the DRAG lane so it
        // leaves a genuine pre-drag backup (a public-setter write would be a
        // legitimate non-drag write and would supersede it).
        std::uint64_t right_token = 0;
        rendering_manager_->beginDepthWindowDrag(SplitViewPanelId::Right, right_token);
        ASSERT_TRUE(rendering_manager_->applyDepthWindowForPanelIfEpoch(
            SplitViewPanelId::Right, makeWindow(3.0f, 30.0f, 0.90f),
            rendering_manager_->depthWindowModeEpoch(), right_token));
        const auto transient = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        ASSERT_NE(transient, right_pre);
        ASSERT_EQ(projectionDepthWindow(*rendering_manager_), transient);

        lfs::core::events::cmd::ToggleGTComparison{}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::GTComparison);
        ASSERT_EQ(rendering_manager_->getFocusedSplitPanel(), SplitViewPanelId::Left);

        // Parked pair intact, and the transient is gone from the projection too.
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_pre);
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), right_pre);

        // The leg that makes the leak visible to the user: GT -> Disabled
        // retains the parked pair while the projection becomes the global
        // single window.
        lfs::core::events::cmd::ToggleGTComparison{}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), right_pre);
        EXPECT_NE(projectionDepthWindow(*rendering_manager_), transient);

        // Balance the manually opened bracket; the transition consumed the backup.
        rendering_manager_->endDepthWindowDrag(SplitViewPanelId::Right, right_token);
        EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
    }

    // Sync ON must copy the focused differing window as one undo step. A parked pair
    // has neither its original focus nor an undo path into the park, so refuse the
    // toggle; changing only the flag would restore unequal slots with sync=true. The
    // global-origin control proves this refusal is limited to a dormant session.
    TEST_F(DepthWindowPanelsInteractionTest, GtTimeSyncMutationIsRefusedWhileAPairIsDormant) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(
            SplitViewPanelId::Left, makeWindow(1.0f, 10.0f, 0.31f, 0.32f, 0.10f, 0.11f));
        rendering_manager_->setDepthWindowForPanel(
            SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.62f, 0.63f, -0.20f, -0.21f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);
        const auto left_pre = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right_pre = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        ASSERT_NE(left_pre, right_pre);
        ASSERT_FALSE(rendering_manager_->getDepthWindowSync());

        lfs::core::events::cmd::ToggleGTComparison{}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::GTComparison);

        rendering_manager_->setDepthWindowSync(true);
        // REFUSED, and it left no undo step behind either.
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);

        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::IndependentDual);
        const auto left_back = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right_back = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_EQ(left_back, left_pre);
        EXPECT_EQ(right_back, right_pre);
        // THE INVARIANT: sync true never stands over an unequal pair.
        EXPECT_NE(left_back, right_back);
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());

        // A GT session entered from a GLOBAL mode parks nothing, so the sync
        // flag still moves there: the gate is the park, not GT itself.
        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);
        lfs::core::events::cmd::ToggleGTComparison{}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::GTComparison);
        rendering_manager_->setDepthWindowSync(true);
        EXPECT_TRUE(rendering_manager_->getDepthWindowSync());
    }

    // updateSettings can change mode and global projection together. On GT entry, keep
    // the incoming projection as a GT-time global write without fanning it into slots
    // or replacing it during the focused drag's backup fold. Assert all three outcomes
    // and restoration of the pre-drag parked pair.
    TEST_F(DepthWindowPanelsInteractionTest, UpdateSettingsGtEntryKeepsItsProjectionAndParksBothPanels) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(
            SplitViewPanelId::Left, makeWindow(1.0f, 10.0f, 0.31f, 0.32f, 0.10f, 0.11f));
        rendering_manager_->setDepthWindowForPanel(
            SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.62f, 0.63f, -0.20f, -0.21f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);
        const auto left_pre = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right_pre = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        ASSERT_NE(left_pre, right_pre);

        // A live drag on the FOCUSED panel, so its backup is the one the fold
        // would otherwise re-apply to the projection.
        std::uint64_t left_token = 0;
        rendering_manager_->beginDepthWindowDrag(SplitViewPanelId::Left, left_token);
        ASSERT_TRUE(rendering_manager_->applyDepthWindowForPanelIfEpoch(
            SplitViewPanelId::Left, makeWindow(3.0f, 30.0f, 0.90f),
            rendering_manager_->depthWindowModeEpoch(), left_token));
        ASSERT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);

        // ONE write: enter GT AND move the global depth projection.
        auto settings = rendering_manager_->getSettings();
        settings.split_view_mode = SplitViewMode::GTComparison;
        settings.depth_filter_scale_x = 0.77f;
        settings.depth_filter_scale_y = 0.78f;
        settings.depth_filter_offset_x = 0.0f;
        settings.depth_filter_offset_y = 0.0f;
        settings.depth_filter_min = {-0.5f, -0.5f, -70.0f};
        settings.depth_filter_max = {0.5f, 0.5f, -7.0f};
        rendering_manager_->updateSettings(settings);
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::GTComparison);

        const auto requested = makeWindow(7.0f, 70.0f, 0.77f, 0.78f, 0.0f, 0.0f);
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), requested);
        // The GT-time global write did NOT fan out into the dormant pair.
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_pre);

        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::IndependentDual);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_pre);
        EXPECT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left),
                  rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right));

        rendering_manager_->endDepthWindowDrag(SplitViewPanelId::Left, left_token);
        EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
    }

    TEST_F(DepthWindowPanelsInteractionTest, UndoSyncedDragCommitRestoresBothSlotsAfterSyncOffAndFocusMove) {
        enterIndependentDual();
        const auto before_drag = makeWindow(1.0f, 10.0f, 0.25f);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, before_drag);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, before_drag);
        rendering_manager_->setDepthWindowSync(true);
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);

        ASSERT_TRUE(commitDepthDraw(40.0, 40.0, 110.0, 110.0));
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
        const auto committed = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        EXPECT_NE(committed.scale_x, before_drag.scale_x);

        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);
        rendering_manager_->setDepthWindowForPanel(
            SplitViewPanelId::Right, makeWindow(9.0f, 90.0f, 0.90f));

        EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);
        const auto left = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_FLOAT_EQ(left.near_plane, before_drag.near_plane);
        EXPECT_FLOAT_EQ(right.near_plane, before_drag.near_plane);
        EXPECT_FLOAT_EQ(left.scale_x, before_drag.scale_x);
        EXPECT_FLOAT_EQ(right.scale_x, before_drag.scale_x);

        // The drag entry captured sync=ON, but only the dedicated sync entry
        // owns the sync flag: undoing the drag after a later sync-OFF must not
        // re-enable sync.
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());
        // Focus moved to Right before the undo, so the projection follows
        // Right's restored slot - never the projection captured at drag time.
        EXPECT_EQ(rendering_manager_->getFocusedSplitPanel(), SplitViewPanelId::Right);
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), right);
    }

    // Sync changes are IGNORED while a depth-window
    // drag is in flight (the toggle's snapshot would capture transient drag
    // geometry and an undo could resurrect it). A toggle after release works.
    TEST_F(DepthWindowPanelsInteractionTest, MidDragSyncToggleIsIgnoredUntilRelease) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(
            SplitViewPanelId::Left, makeWindow(1.0f, 10.0f, 0.30f));
        rendering_manager_->setDepthWindowForPanel(
            SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.70f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);
        const auto right_before = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        // The gate is the OWNERSHIP bracket, so the SUBTHRESHOLD window before
        // any preview exists ignores the toggle too.
        ASSERT_FALSE(rendering_manager_->depthWindowDragPreview());
        rendering_manager_->setDepthWindowSync(true);
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);

        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        rendering_manager_->setDepthWindowSync(true);
        // Ignored: sync unchanged, no undo entry, other panel untouched.
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
        EXPECT_FLOAT_EQ(
            rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right).near_plane,
            right_before.near_plane);

        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_release(90.0, 90.0)),
                  OperatorResult::FINISHED);
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);

        // After release the toggle works and takes its own undo step.
        rendering_manager_->setDepthWindowSync(true);
        EXPECT_TRUE(rendering_manager_->getDepthWindowSync());
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 2u);
        const auto items = lfs::vis::op::undoHistory().undoItems();
        ASSERT_EQ(items.size(), 2u);
        EXPECT_EQ(items[0].metadata.id, "selection.depth_window_sync");
        EXPECT_EQ(items[1].metadata.id, "selection.depth_window_drag");
    }

    TEST_F(DepthWindowPanelsInteractionTest, RedoPanelDragRepublishesCommitExpiredPublishesNothing) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);

        const auto generation_before = lfs::vis::app_store().depth_window_draw_generation.get();
        ASSERT_TRUE(commitDepthDraw(260.0, 40.0, 320.0, 120.0));
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);

        EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);
        const auto generation_after_undo = lfs::vis::app_store().depth_window_draw_generation.get();
        EXPECT_EQ(generation_after_undo, generation_before + 2u);

        const auto commit_before_redo = lfs::vis::app_store().depth_window_draw_commit.get();
        EXPECT_TRUE(lfs::vis::op::undoHistory().redo().success);
        const auto commit_after_redo = lfs::vis::app_store().depth_window_draw_commit.get();
        EXPECT_EQ(commit_after_redo.panel, SplitViewPanelId::Right);
        EXPECT_GT(commit_after_redo.generation, commit_before_redo.generation);

        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        ASSERT_EQ(rendering_manager_->depthWindowModeEpoch(), 2u);
        const auto generation_before_expired_redo =
            lfs::vis::app_store().depth_window_draw_generation.get();
        const auto commit_before_expired_redo = lfs::vis::app_store().depth_window_draw_commit.get();
        EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_TRUE(lfs::vis::op::undoHistory().redo().success);
        EXPECT_EQ(lfs::vis::app_store().depth_window_draw_generation.get(),
                  generation_before_expired_redo);
        EXPECT_EQ(lfs::vis::app_store().depth_window_draw_commit.get(), commit_before_expired_redo);
    }

    // Every drag write uses epoch-guarded apply, which must refuse a stale epoch
    // without changes. This checks the manager guarantee; the release/destructor thread
    // interleaving needs a dedicated concurrency seam to reproduce deterministically.
    TEST_F(DepthWindowPanelsInteractionTest, StaleEpochApplyRefusesAndChangesNothing) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left,
                                                   makeWindow(1.0f, 10.0f, 0.4f));
        const auto before = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto stale_epoch = rendering_manager_->depthWindowModeEpoch();

        // Leave and re-enter independent-dual: the epoch moves on.
        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        ASSERT_NE(rendering_manager_->depthWindowModeEpoch(), stale_epoch);

        const auto seeded = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto seeded_other = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        const auto seeded_projection = projectionDepthWindow(*rendering_manager_);
        const bool seeded_sync = rendering_manager_->getDepthWindowSync();
        const auto seeded_generation = rendering_manager_->depthWindowProjectionGeneration();

        // A real bracket, so the ONLY thing wrong with the write below is the
        // stale epoch: the drag genuinely owns the slot it is trying to write.
        std::uint64_t left_token = 0;
        rendering_manager_->beginDepthWindowDrag(SplitViewPanelId::Left, left_token);
        EXPECT_FALSE(rendering_manager_->applyDepthWindowForPanelIfEpoch(
            SplitViewPanelId::Left, makeWindow(3.0f, 30.0f, 0.9f), stale_epoch, left_token));
        const auto after = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        EXPECT_EQ(after, seeded);
        // A refused apply changes NOTHING: not the other slot, not the
        // projection, not the sync flag, not the projection generation.
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), seeded_other);
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), seeded_projection);
        EXPECT_EQ(rendering_manager_->getDepthWindowSync(), seeded_sync);
        EXPECT_EQ(rendering_manager_->depthWindowProjectionGeneration(), seeded_generation);

        // The same refusal holds for the absolute snapshot restore path.
        const auto stale_snapshot = rendering_manager_->depthWindowSnapshot();
        EXPECT_FALSE(rendering_manager_->restoreDepthWindowSnapshotIfEpoch(
            stale_snapshot, stale_epoch, /*restore_sync=*/false));
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), seeded);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), seeded_other);
        EXPECT_EQ(rendering_manager_->depthWindowProjectionGeneration(), seeded_generation);

        // Current epoch AND owned slot: the write lands.
        EXPECT_TRUE(rendering_manager_->applyDepthWindowForPanelIfEpoch(
            SplitViewPanelId::Left, makeWindow(3.0f, 30.0f, 0.9f),
            rendering_manager_->depthWindowModeEpoch(), left_token));
        EXPECT_FLOAT_EQ(
            rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left).near_plane, 3.0f);

        // The OWNERSHIP half of the same gate, under a current epoch: once this
        // drag's slot ownership is gone, the identical write refuses.
        rendering_manager_->endDepthWindowDrag(SplitViewPanelId::Left, left_token);
        const auto owned_write = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        EXPECT_FALSE(rendering_manager_->applyDepthWindowForPanelIfEpoch(
            SplitViewPanelId::Left, makeWindow(7.0f, 70.0f, 0.7f),
            rendering_manager_->depthWindowModeEpoch(), left_token));
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), owned_write);
        (void)before;
    }

    // Mid-drag sync toggles are refused, so the only reachable ordering is
    // drag -> release -> sync-ON. Undo twice then redo twice must land on
    // exactly the states each step recorded.
    TEST_F(DepthWindowPanelsInteractionTest, DragThenSyncOnUndoRedoConverges) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        const auto left_before = makeWindow(1.0f, 10.0f, 0.25f);
        const auto right_before = makeWindow(2.0f, 20.0f, 0.75f);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, left_before);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, right_before);
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);

        ASSERT_TRUE(commitDepthDraw(40.0, 40.0, 110.0, 110.0));
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
        const auto left_dragged = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        ASSERT_NE(left_dragged, left_before);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_before);

        rendering_manager_->setDepthWindowSync(true);
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 2u);
        ASSERT_TRUE(rendering_manager_->getDepthWindowSync());
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), left_dragged);

        // Undo 1 - the sync entry: slots diverge again, sync back OFF.
        EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_dragged);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_before);
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());

        // Undo 2 - the drag entry: absolute pre-drag state on both slots,
        // sync untouched by the drag entry.
        EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_before);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_before);
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());

        // Redo 1 - the drag entry.
        EXPECT_TRUE(lfs::vis::op::undoHistory().redo().success);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_dragged);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_before);
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());

        // Redo 2 - the sync entry re-collapses identically.
        EXPECT_TRUE(lfs::vis::op::undoHistory().redo().success);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_dragged);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), left_dragged);
        EXPECT_TRUE(rendering_manager_->getDepthWindowSync());
    }

    // Undoing a sync-ON after focus moved: the restored slots diverge again,
    // sync returns OFF, and the PROJECTION follows the panel focused NOW - the
    // restore must never resurrect the projection captured at snapshot time.
    TEST_F(DepthWindowPanelsInteractionTest, SyncUndoAfterFocusMoveProjectsFocusedSlot) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        const auto left_window = makeWindow(1.0f, 10.0f, 0.25f);
        const auto right_window = makeWindow(2.0f, 20.0f, 0.75f);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, left_window);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, right_window);
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);

        rendering_manager_->setDepthWindowSync(true);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), left_window);

        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);
        EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);

        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_window);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_window);
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), right_window);
    }

    // With sync ON and clean O, A previews P on Left; real registry replacement starts
    // B on Right, then leaves independent-dual while B is subthreshold. A's teardown
    // must restore only Left, without fan-out; B must inherit Right's O backup, not P.
    // Both slots must end at O.
    TEST_F(DepthWindowPanelsInteractionTest, SyncOnCrossPanelReplacementNeverResurrectsPreview) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(true);
        const auto clean = makeWindow(1.0f, 10.0f, 0.31f);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, clean);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), clean);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), clean);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        const auto preview = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        ASSERT_NE(preview, clean);

        // B: a fresh draw on the RIGHT panel, invoked through the registry, which
        // destroys A. B stays subthreshold - it never draws.
        ASSERT_TRUE(startDepthDrag(210.0, 20.0));

        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);

        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), clean);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), clean);
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), clean);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
    }

    // Continue unsynced cross-panel replacement through B's commit, sync ON focused
    // Right, then focus Left and leave. The legitimate sync copy must release Left's
    // stale pre-A backup so collapse keeps the copy.
    TEST_F(DepthWindowPanelsInteractionTest, SyncCopyReleasesStaleBackupBeforeCollapse) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left,
                                                   makeWindow(1.0f, 10.0f, 0.31f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right,
                                                   makeWindow(2.0f, 20.0f, 0.62f));
        const auto left_pre = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);

        // Cross-panel registry replacement, then B commits on Right.
        ASSERT_TRUE(startDepthDrag(210.0, 20.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(300.0, 120.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_release(300.0, 120.0)),
                  OperatorResult::FINISHED);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);
        const auto right_committed = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        ASSERT_EQ(rendering_manager_->getFocusedSplitPanel(), SplitViewPanelId::Right);

        rendering_manager_->setDepthWindowSync(true);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), right_committed);

        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);
        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);

        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), right_committed);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_committed);
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), right_committed);
    }

    // Sync refusal covers the entire invoke-to-destruction ownership bracket, including
    // a subthreshold press. The before_ capture must not straddle a sync change; undo
    // must preserve consistent slots and flag.
    TEST_F(DepthWindowPanelsInteractionTest, SubthresholdPressIgnoresSyncToggleAndUndoStaysConsistent) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        const auto left_pre = makeWindow(1.0f, 10.0f, 0.30f);
        const auto right_pre = makeWindow(2.0f, 20.0f, 0.70f);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, left_pre);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, right_pre);
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);

        // Subthreshold: invoked, owning Left, but not previewing yet.
        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_FALSE(rendering_manager_->depthWindowDragPreview());
        rendering_manager_->setDepthWindowSync(true);
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_pre);

        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_release(90.0, 90.0)),
                  OperatorResult::FINISHED);
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
        ASSERT_FALSE(rendering_manager_->getDepthWindowSync());

        EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_pre);
    }

    // Unsynced: A previews Left, real registry replacement invokes B on Right, B
    // commits, then undo B. Capture B's baseline at its first slot write, after A's
    // teardown restores Left, so undo cannot resurrect A's abandoned preview.
    TEST_F(DepthWindowPanelsInteractionTest, UndoOfReplacementDragNeverResurrectsReplacedPreviewUnsynced) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        const auto left_clean = makeWindow(1.0f, 10.0f, 0.31f);
        const auto right_clean = makeWindow(2.0f, 20.0f, 0.62f);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, left_clean);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, right_clean);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        const auto preview = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        ASSERT_NE(preview, left_clean);

        // B replaces A through the registry, then draws and commits on Right.
        ASSERT_TRUE(startDepthDrag(210.0, 20.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(300.0, 120.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_release(300.0, 120.0)),
                  OperatorResult::FINISHED);
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_clean);

        EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_clean);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_clean);
    }

    // Regression, verbatim sequence (sync ON): same replacement, same
    // undo. The pressed panel's clean value must survive the undo.
    TEST_F(DepthWindowPanelsInteractionTest, UndoOfReplacementDragNeverResurrectsReplacedPreviewSynced) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(true);
        const auto clean = makeWindow(1.0f, 10.0f, 0.31f);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, clean);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), clean);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        const auto preview = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        ASSERT_NE(preview, clean);

        ASSERT_TRUE(startDepthDrag(210.0, 20.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(300.0, 120.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_release(300.0, 120.0)),
                  OperatorResult::FINISHED);
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
        // Under sync ON the commit FANS OUT, so Left legitimately carries B's
        // committed window right now - what matters is what the undo restores.
        const auto committed = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), committed);
        ASSERT_NE(committed, preview);

        EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);
        // B takes both slots, so A restores neither. B's undo baseline must come from
        // each owned slot's recorded pre-A backup, not the live preview. Undo then
        // restores pre-A values in both slots.
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), clean);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), clean);
    }

    // Cancellation uses the live windows captured at B's first write, including any
    // inherited A preview, rather than the manager backups used for undo. This
    // same-panel replacement test pins that distinction alongside
    // DepthWindowDragLifecycleTest.
    TEST_F(DepthWindowPanelsInteractionTest, CancelOfReplacementDragKeepsInvokeTimeScreenStateSynced) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(true);
        const auto clean = makeWindow(1.0f, 10.0f, 0.31f);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, clean);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), clean);

        // A previews on Left; under sync ON the preview lands in BOTH slots.
        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        const auto preview = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        ASSERT_NE(preview, clean);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), preview);

        // B replaces A cross-panel: it takes over both slots, so A's teardown
        // restores neither and A's preview is still standing.
        ASSERT_TRUE(startDepthDrag(210.0, 20.0));
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), preview);
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(300.0, 120.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), preview);

        lfs::vis::op::operators().cancelModalOperator();

        // Back to the state B found on screen, on both slots: A's abandoned
        // preview, NOT the pre-A clean value. Nothing was committed, so there
        // is no undo entry - and running an undo therefore changes nothing.
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), preview);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), preview);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
        EXPECT_FALSE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), preview);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), preview);

        // After B releases ownership, C must refresh idle backups from visible pre-C
        // state P, even if pre-A O remained recorded. Undo C restores P, never O.
        ASSERT_TRUE(startDepthDrag(40.0, 200.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(130.0, 260.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_release(130.0, 260.0)),
                  OperatorResult::FINISHED);
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
        const auto committed = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        ASSERT_NE(committed, preview);

        EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), preview);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), preview);
        EXPECT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), clean);
    }

    // A mode-changing updateSettings must take the transition lock and perform the
    // event path's epoch bump, backup fold and collapse. The old drag's later writes
    // must fail the epoch guard.
    TEST_F(DepthWindowPanelsInteractionTest, UpdateSettingsModeFlipRunsTheDepthWindowTransition) {
        enterIndependentDual();
        ASSERT_EQ(rendering_manager_->depthWindowModeEpoch(), 1u);
        rendering_manager_->setDepthWindowSync(false);
        const auto left_pre = makeWindow(1.0f, 10.0f, 0.31f);
        const auto right_pre = makeWindow(2.0f, 20.0f, 0.62f);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, left_pre);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, right_pre);
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);

        // A live drag on Left with a preview already written.
        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);

        // The flip arrives through updateSettings, not through an event site.
        auto flipped = rendering_manager_->getSettings();
        flipped.split_view_mode = SplitViewMode::Disabled;
        rendering_manager_->updateSettings(flipped);

        EXPECT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);
        EXPECT_EQ(rendering_manager_->depthWindowModeEpoch(), 2u);
        // The drag's preview was folded back and the focused panel's clean
        // value collapsed over both slots.
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), left_pre);

        // The pre-flip drag's next write is refused (stale epoch).
        lfs::vis::op::operators().dispatchModalEvent(mouse_move(130.0, 130.0));
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), left_pre);
    }

    // An equal-mode updateSettings must avoid the transition lock. Toggling
    // independent-dual with an enabled tool and latched drag re-enters it via cancel
    // hook -> finishLatch -> applySelectionFilterSettings while holding that lock.
    // Reaching the coherent-state assertions proves this sequence did not deadlock.
    TEST_F(DepthWindowPanelsInteractionTest, EqualModeSettingsWriteNeverDeadlocksTheTransitionLock) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        const auto left_pre = makeWindow(1.0f, 10.0f, 0.31f);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, left_pre);
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_TRUE(rendering_manager_->depthWindowDragPreview());

        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();

        EXPECT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);
        EXPECT_EQ(rendering_manager_->depthWindowModeEpoch(), 2u);
        EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), left_pre);
    }

    // With sync ON, A previews while a same-epoch history entry is undone mid-modal; A
    // continues, then leave independent-dual focused on the other panel. A pinned both
    // backups for fan-out, so undo's idle-backup release must preserve the other pin.
    // Otherwise collapse publishes A's preview.
    TEST_F(DepthWindowPanelsInteractionTest, MidDragHistoryUndoKeepsLiveFanOutBackup) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(true);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left,
                                                   makeWindow(1.0f, 10.0f, 0.31f));

        // A committed drag puts a same-epoch depth-window entry atop the history.
        ASSERT_TRUE(commitDepthDraw(40.0, 40.0, 120.0, 120.0));
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
        const auto committed = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), committed);

        // Drag A: previews into BOTH slots, so both were backed up at its begin.
        ASSERT_TRUE(startDepthDrag(50.0, 50.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(140.0, 140.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), committed);

        // The GUI-thread history undo lands MID-MODAL and writes both slots.
        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        // A continues drawing after the playback.
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(160.0, 150.0)),
                  OperatorResult::RUNNING_MODAL);
        const auto live_preview = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);

        // Focus the panel A did NOT press, then leave: the collapse folds the
        // focused slot, which must be the surviving pre-A backup.
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);
        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);

        EXPECT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), live_preview);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), committed);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), committed);
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), committed);
    }

    // With no retained pair, Disabled/PLYComparison leaves depth state unchanged. A
    // live drag keeps its backup, pins and epoch, continues drawing, then cancels to
    // pre-drag O.
    TEST_F(DepthWindowPanelsInteractionTest, GlobalModeFlipLeavesLiveDragUndisturbed) {
        auto settings = rendering_manager_->getSettings();
        settings.depth_filter_scale_x = 0.41f;
        settings.depth_filter_scale_y = 0.41f;
        settings.depth_filter_min = {-0.5f, -0.5f, -10.0f};
        settings.depth_filter_max = {0.5f, 0.5f, -1.0f};
        rendering_manager_->updateSettings(settings);
        const auto pre_drag = projectionDepthWindow(*rendering_manager_);
        const auto epoch_before = rendering_manager_->depthWindowModeEpoch();

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        const auto preview = projectionDepthWindow(*rendering_manager_);
        ASSERT_NE(preview, pre_drag);

        // The flip arrives mid-drag, through updateSettings, and crosses no
        // depth-relevant boundary.
        auto flipped = rendering_manager_->getSettings();
        flipped.split_view_mode = SplitViewMode::PLYComparison;
        rendering_manager_->updateSettings(flipped);
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode,
                  SplitViewMode::PLYComparison);
        EXPECT_EQ(rendering_manager_->depthWindowModeEpoch(), epoch_before);

        // The drag was never expired: it keeps drawing.
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(150.0, 130.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_NE(projectionDepthWindow(*rendering_manager_), preview);

        // Cancel folds back to O - the backup survived the flip.
        lfs::vis::op::operators().cancelModalOperator();
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), pre_drag);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), pre_drag);
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), pre_drag);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);

        // And entering independent-dual seeds O, never the abandoned preview.
        enterIndependentDual();
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), pre_drag);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), pre_drag);
    }

    // After the no-op mode flip, same-panel replacement B must retain pre-A O as
    // backup. The registry destroys A after B claims its pins; A's teardown must leave
    // B's ownership intact so B can still cancel.
    TEST_F(DepthWindowPanelsInteractionTest, SamePanelDragAfterGlobalModeFlipKeepsPreDragBackupAndPins) {
        auto settings = rendering_manager_->getSettings();
        settings.depth_filter_scale_x = 0.41f;
        settings.depth_filter_scale_y = 0.41f;
        settings.depth_filter_min = {-0.5f, -0.5f, -10.0f};
        settings.depth_filter_max = {0.5f, 0.5f, -1.0f};
        rendering_manager_->updateSettings(settings);
        const auto pre_drag = projectionDepthWindow(*rendering_manager_);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);

        auto flipped = rendering_manager_->getSettings();
        flipped.split_view_mode = SplitViewMode::PLYComparison;
        rendering_manager_->updateSettings(flipped);

        // A draws on after the flip; its preview is what the slot holds when
        // the replacement lands.
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(150.0, 130.0)),
                  OperatorResult::RUNNING_MODAL);
        const auto abandoned_preview = projectionDepthWindow(*rendering_manager_);
        ASSERT_NE(abandoned_preview, pre_drag);

        // Same-panel registry replacement: B is invoked, THEN A is destroyed.
        ASSERT_TRUE(startDepthDrag(20.0, 150.0));
        ASSERT_EQ(projectionDepthWindow(*rendering_manager_), abandoned_preview);
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(120.0, 190.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_NE(projectionDepthWindow(*rendering_manager_), abandoned_preview);

        // B's pins survived A's end, so B's own cancel still folds back to the
        // baseline B inherited.
        lfs::vis::op::operators().cancelModalOperator();
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), abandoned_preview);

        // The panel's backup is still PRE-DRAG-A: entering independent-dual
        // seeds O, not A's or B's preview.
        enterIndependentDual();
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), pre_drag);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), pre_drag);
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), pre_drag);
    }

    // Independent-dual, sync ON: A previews both slots, then a settings write installs
    // a newer window and supersedes A's backups and pins. Cancelling A must restore
    // neither slot.
    TEST_F(DepthWindowPanelsInteractionTest, NewerSettingsWriteBeatsCancelTeardownOnBothSlots) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(true);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left,
                                                   makeWindow(1.0f, 10.0f, 0.31f, 0.31f));
        const auto pre_drag = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), pre_drag);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), pre_drag);

        // The legitimate GUI-thread settings write, mid-drag.
        auto written = rendering_manager_->getSettings();
        written.depth_filter_scale_x = 0.73f;
        written.depth_filter_scale_y = 0.73f;
        written.depth_filter_min = {-0.5f, -0.5f, -22.0f};
        written.depth_filter_max = {0.5f, 0.5f, -3.0f};
        rendering_manager_->updateSettings(written);
        const auto newest = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), newest);
        ASSERT_NE(newest, pre_drag);

        lfs::vis::op::operators().cancelModalOperator();

        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), newest);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), newest);
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), newest);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
    }

    // With sync ON, a settings write replaces A's preview with W and revokes both
    // slots. A's next move must be refused, and cancellation must restore nothing. W
    // must remain in both slots.
    TEST_F(DepthWindowPanelsInteractionTest, SupersededDragCannotWriteAfterNewerSettingsWrite) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(true);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left,
                                                   makeWindow(1.0f, 10.0f, 0.31f, 0.31f));
        const auto pre_drag = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), pre_drag);

        auto written = rendering_manager_->getSettings();
        written.depth_filter_scale_x = 0.73f;
        written.depth_filter_scale_y = 0.73f;
        written.depth_filter_min = {-0.5f, -0.5f, -22.0f};
        written.depth_filter_max = {0.5f, 0.5f, -3.0f};
        rendering_manager_->updateSettings(written, DirtyFlag::SELECTION);
        const auto newest = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), newest);
        ASSERT_NE(newest, pre_drag);

        // The modal still receives this move, but lost ownership must refuse its write.
        // Unlike the old epoch/aggregate-pin gate, refusal ends the drag through
        // epoch_lost_ without restoring over W.
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(150.0, 130.0)),
                  OperatorResult::CANCELLED);
        EXPECT_FALSE(lfs::vis::op::operators().hasModalOperator());

        lfs::vis::op::operators().cancelModalOperator();

        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), newest);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), newest);
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), newest);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
    }

    // A superseded drag cannot borrow a later drag's ownership for teardown. With sync
    // ON, install W over A, then replace it with B on Right, owning both slots and
    // baseline W. A has not moved since losing ownership; its destruction must preserve
    // the slots and B's baseline.
    TEST_F(DepthWindowPanelsInteractionTest, ReplacedDragTeardownNeverRestoresOverANewerDragsPins) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(true);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left,
                                                   makeWindow(1.0f, 10.0f, 0.31f, 0.31f));
        const auto pre_drag = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), pre_drag);

        auto written = rendering_manager_->getSettings();
        written.depth_filter_scale_x = 0.73f;
        written.depth_filter_scale_y = 0.73f;
        written.depth_filter_min = {-0.5f, -0.5f, -22.0f};
        written.depth_filter_max = {0.5f, 0.5f, -3.0f};
        rendering_manager_->updateSettings(written, DirtyFlag::SELECTION);
        const auto newest = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), newest);
        // Keep A live by omitting its post-write move, whose refusal would end it
        // (SupersededDragCannotWriteAfterNewerSettingsWrite). This exercises teardown
        // after ownership loss but before that refusal.
        ASSERT_TRUE(lfs::vis::op::operators().hasModalOperator());

        // Cross-panel registry replacement: B is invoked on Right (taking
        // ownership of BOTH slots under sync ON), THEN A is destroyed.
        ASSERT_TRUE(startDepthDrag(210.0, 20.0));
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), newest);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), newest);

        // B's own baseline is W, not anything A left behind: B draws, then
        // cancels, and both slots fold back to exactly W.
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(300.0, 120.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), newest);
        lfs::vis::op::operators().cancelModalOperator();

        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), newest);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), newest);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
    }

    // Independent and unsynced: A owns Left while a legitimate focused write updates
    // Right. A must still restore and release Left on cancel. The later sync copy must
    // discard its now-idle pre-A backup, so leaving independent collapses the copy, not
    // pre-A Left.
    TEST_F(DepthWindowPanelsInteractionTest, FocusedOtherPanelWriteStillLetsTheDragReleaseItsOwnSlot) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left,
                                                   makeWindow(1.0f, 10.0f, 0.31f, 0.31f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right,
                                                   makeWindow(2.0f, 20.0f, 0.62f, 0.62f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);
        const auto left_pre = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);

        ASSERT_TRUE(startDepthDrag(40.0, 40.0));
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(90.0, 90.0)),
                  OperatorResult::RUNNING_MODAL);
        ASSERT_NE(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);

        // The legitimate focused-RIGHT settings write. It supersedes Right's
        // slot only; A keeps its ownership of Left.
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);
        auto written = rendering_manager_->getSettings();
        written.depth_filter_scale_x = 0.73f;
        written.depth_filter_scale_y = 0.73f;
        written.depth_filter_min = {-0.5f, -0.5f, -22.0f};
        written.depth_filter_max = {0.5f, 0.5f, -3.0f};
        rendering_manager_->updateSettings(written, DirtyFlag::SELECTION);
        const auto right_written = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        ASSERT_NE(right_written, left_pre);

        // A cancels: it still owns Left, so Left folds back to pre-A exactly.
        lfs::vis::op::operators().cancelModalOperator();
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left_pre);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_written);
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);

        // Sync ON copies the FOCUSED (Right) slot onto Left. A's end released
        // Left's ownership, so this copy also drops Left's now-idle backup.
        rendering_manager_->setDepthWindowSync(true);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), right_written);

        // Leaving independent collapses the sync copy, never the pre-A window.
        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), right_written);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_written);
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), right_written);
    }

    // Drive the native collapse producer; a stub-fed Python test cannot detect its
    // removal. Pin the initial (source,generation), the collapse increment using
    // pre-transition focus rather than reset Left, and no increment for the other
    // operations exercised here. The toolbar uses this record to judge Size-reference
    // survival.
    TEST_F(DepthWindowPanelsInteractionTest, CollapseRecordStampsSourceAndGenerationOnLeaveOnly) {
        // Before any collapse: default Left, generation 0.
        {
            const auto initial = rendering_manager_->getDepthWindowCollapseRecord();
            EXPECT_EQ(initial.source, SplitViewPanelId::Left);
            EXPECT_EQ(initial.generation, 0u);
            EXPECT_EQ(initial.kind,
                      RenderingManager::DepthWindowLineageKind::LeaveCollapse);
        }

        // Entering independent-dual is a mode epoch boundary but NOT a collapse.
        enterIndependentDual();
        ASSERT_EQ(rendering_manager_->depthWindowModeEpoch(), 1u);
        EXPECT_EQ(rendering_manager_->getDepthWindowCollapseRecord().generation, 0u);

        rendering_manager_->setDepthWindowSync(false);
        const auto left_pre = makeWindow(1.0f, 10.0f, 0.60f);
        const auto right_pre = makeWindow(2.0f, 20.0f, 0.20f);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, left_pre);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, right_pre);
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);

        // An unrelated settings write (same mode) collapses nothing.
        {
            auto unrelated = rendering_manager_->getSettings();
            unrelated.depth_filter_scale_x = 0.44f;
            rendering_manager_->updateSettings(unrelated);
            EXPECT_EQ(rendering_manager_->getDepthWindowCollapseRecord().generation, 0u);
        }

        // Leave with Right focused. The preceding no-panel write back-routed projection
        // scale into Right, so read its current window rather than assuming right_pre
        // survived.
        const auto right_now = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        ASSERT_NE(right_now, left_pre);
        auto leaving = rendering_manager_->getSettings();
        leaving.split_view_mode = SplitViewMode::Disabled;
        rendering_manager_->updateSettings(leaving);

        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);
        const auto collapsed = rendering_manager_->getDepthWindowCollapseRecord();
        EXPECT_EQ(collapsed.source, SplitViewPanelId::Right);
        EXPECT_EQ(collapsed.generation, 1u);
        EXPECT_EQ(collapsed.kind,
                  RenderingManager::DepthWindowLineageKind::LeaveCollapse);
        // The one-value getter and the paired read agree.
        EXPECT_EQ(rendering_manager_->getDepthWindowCollapseSource(), SplitViewPanelId::Right);
        // It really was Right's window that collapsed over both slots.
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), right_now);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right_now);

        // Nothing after the collapse bumps it again on its own.
        {
            auto after = rendering_manager_->getSettings();
            after.depth_filter_scale_y = 0.33f;
            rendering_manager_->updateSettings(after);
            EXPECT_EQ(rendering_manager_->getDepthWindowCollapseRecord().generation, 1u);
        }

        // A SECOND cycle counts a second collapse - the delta, not the identity,
        // is what tells a poller it slept through one.
        enterIndependentDual();
        EXPECT_EQ(rendering_manager_->getDepthWindowCollapseRecord().generation, 1u);
        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);
        EXPECT_EQ(rendering_manager_->getDepthWindowCollapseRecord().generation, 2u);
    }

    // Pin native sync-copy and project-restore lineage producers, plus non-stamping
    // controls. Missing stamps retain references to replaced windows; spurious stamps
    // discard valid Size references. Stub-fed consumer tests cannot detect either
    // producer error.
    TEST_F(DepthWindowPanelsInteractionTest, LineageStampsCoverSyncCopyAndProjectRestore) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        EXPECT_EQ(rendering_manager_->getDepthWindowCollapseRecord().generation, 0u);

        const auto left_pre = makeWindow(1.0f, 10.0f, 0.60f);
        const auto right_pre = makeWindow(2.0f, 20.0f, 0.20f);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, left_pre);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, right_pre);
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);
        // Ordinary per-panel writes and a focus change invalidate nothing: both
        // panels' windows still exist, exactly where their references say.
        EXPECT_EQ(rendering_manager_->getDepthWindowCollapseRecord().generation, 0u);

        // (b) Sync ON copies the FOCUSED slot over the other. Left's window is
        // gone, so this is a slot-invalidating write, stamped with the panel it
        // was copied FROM.
        rendering_manager_->setDepthWindowSync(true);
        ASSERT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left),
                  right_pre);
        {
            const auto synced = rendering_manager_->getDepthWindowCollapseRecord();
            EXPECT_EQ(synced.generation, 1u);
            EXPECT_EQ(synced.source, SplitViewPanelId::Right);
            EXPECT_EQ(synced.kind, RenderingManager::DepthWindowLineageKind::SyncCopy);
        }

        // Sync OFF, then ON again with the slots ALREADY equal: nothing is
        // copied, so nothing is invalidated and nothing is stamped.
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowSync(true);
        EXPECT_EQ(rendering_manager_->getDepthWindowCollapseRecord().generation, 1u);

        // (c) A project restore seeds BOTH slots from the restored projection.
        // It can begin and end in the same mode with sync already off, so the
        // stamp is the only witness a poller gets.
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->restoreDepthWindowStateFromProject();
        {
            const auto restored = rendering_manager_->getDepthWindowCollapseRecord();
            EXPECT_EQ(restored.generation, 2u);
            EXPECT_EQ(restored.kind,
                      RenderingManager::DepthWindowLineageKind::ProjectRestore);
        }

        // (a) again, for the ordering: the leave stamps a third time, as a
        // LeaveCollapse.
        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        ASSERT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);
        {
            const auto left = rendering_manager_->getDepthWindowCollapseRecord();
            EXPECT_EQ(left.generation, 3u);
            EXPECT_EQ(left.kind, RenderingManager::DepthWindowLineageKind::LeaveCollapse);
        }

        // A mode change that means NOTHING to depth windows stamps nothing --
        // the same predicate that makes applyDepthWindowModeTransitionLocked a
        // complete no-op there.
        {
            auto flipped = rendering_manager_->getSettings();
            flipped.split_view_mode = SplitViewMode::PLYComparison;
            rendering_manager_->updateSettings(flipped);
            ASSERT_EQ(rendering_manager_->getSettings().split_view_mode,
                      SplitViewMode::PLYComparison);
            EXPECT_EQ(rendering_manager_->getDepthWindowCollapseRecord().generation, 3u);
            auto back = rendering_manager_->getSettings();
            back.split_view_mode = SplitViewMode::Disabled;
            rendering_manager_->updateSettings(back);
            EXPECT_EQ(rendering_manager_->getDepthWindowCollapseRecord().generation, 3u);
        }

        // And a plain mode ENTER seeds both slots from the one window that was
        // already displayed: nothing a reference could describe was lost.
        enterIndependentDual();
        EXPECT_EQ(rendering_manager_->getDepthWindowCollapseRecord().generation, 3u);
    }

    // Sync undo/redo restores absolute slot snapshots and must stamp lineage, even with
    // no mode/focus edge or an undo/redo pair hiding the sync edge. Drag undo shares
    // the manager restore but must stamp nothing.
    TEST_F(DepthWindowPanelsInteractionTest, SyncUndoRedoStampsLineageAndDragUndoDoesNot) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left,
                                                   makeWindow(1.0f, 10.0f, 0.60f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right,
                                                   makeWindow(2.0f, 20.0f, 0.20f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);
        ASSERT_EQ(rendering_manager_->getDepthWindowCollapseRecord().generation, 0u);

        // The sync-ON copy itself stamps once (producer (b)), and pushes the
        // sync undo entry.
        rendering_manager_->setDepthWindowSync(true);
        ASSERT_EQ(rendering_manager_->getDepthWindowCollapseRecord().generation, 1u);
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
        ASSERT_EQ(lfs::vis::op::undoHistory().undoItems()[0].metadata.id,
                  "selection.depth_window_sync");

        // Undo and redo BOTH between two polls: the flag ends where it started,
        // so a flag-edge consumer sees nothing. The generation must move twice.
        EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);
        {
            const auto undone = rendering_manager_->getDepthWindowCollapseRecord();
            EXPECT_EQ(undone.generation, 2u);
            EXPECT_EQ(undone.kind,
                      RenderingManager::DepthWindowLineageKind::ProjectRestore);
        }
        EXPECT_TRUE(lfs::vis::op::undoHistory().redo().success);
        {
            const auto redone = rendering_manager_->getDepthWindowCollapseRecord();
            EXPECT_EQ(redone.generation, 3u);
            EXPECT_EQ(redone.kind,
                      RenderingManager::DepthWindowLineageKind::ProjectRestore);
        }

        // An EXPIRED sync entry restores nothing, so it stamps nothing. Leaving
        // and re-entering independent-dual moves the epoch past the entry's.
        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        enterIndependentDual();
        const auto after_cycle = rendering_manager_->getDepthWindowCollapseRecord().generation;
        EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(rendering_manager_->getDepthWindowCollapseRecord().generation, after_cycle);

        // A DRAG undo/redo pair goes through the same epoch-guarded restore and
        // stamps NOTHING: both panels' windows still exist where a reference
        // says they do.
        lfs::vis::op::undoHistory().clear();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);
        const auto before_drag = rendering_manager_->getDepthWindowCollapseRecord().generation;
        ASSERT_TRUE(commitDepthDraw(260.0, 40.0, 320.0, 120.0));
        ASSERT_EQ(lfs::vis::op::undoHistory().undoItems()[0].metadata.id,
                  "selection.depth_window_drag");
        EXPECT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(rendering_manager_->getDepthWindowCollapseRecord().generation, before_drag);
        EXPECT_TRUE(lfs::vis::op::undoHistory().redo().success);
        EXPECT_EQ(rendering_manager_->getDepthWindowCollapseRecord().generation, before_drag);
    }

    // Pin overlay-consumed press focus as a truth table over the production helper.
    // This test exercises the decision, not a press through a window, RmlUi context and
    // complete GUI frame.
    TEST(DepthWindowOverlayPressFocusTest, ToolbarChromeNeverFocusesAndDismissAlwaysDoes) {
        using lfs::vis::gui::OverlayPressFocusInputs;
        using lfs::vis::gui::overlayPressMayFocusPanel;

        // A press on the viewport that the overlay claimed: previous behavior.
        constexpr OverlayPressFocusInputs kViewportPress{
            .left_pressed = true,
            .overlay_wants_input = true,
            .pressed_interactive_control = false,
            .press_blurred_text_input = false,
            .press_inside_viewport = true,
        };
        EXPECT_TRUE(overlayPressMayFocusPanel(kViewportPress));

        // Toolbar chrome: the overlay claims the press, and it must NOT focus --
        // otherwise the action it fires reads a focus its own press just moved.
        auto chrome = kViewportPress;
        chrome.pressed_interactive_control = true;
        EXPECT_FALSE(overlayPressMayFocusPanel(chrome));
        // Chrome wins even if the same press also dismissed a text field (a
        // click straight from a focused field onto another control).
        chrome.press_blurred_text_input = true;
        EXPECT_FALSE(overlayPressMayFocusPanel(chrome));

        // A press that dismissed an overlay text field: the field is blurred by
        // then, so the overlay no longer wants input, and this is the press that
        // used to be swallowed entirely. It focuses.
        auto dismiss = kViewportPress;
        dismiss.overlay_wants_input = false;
        dismiss.press_blurred_text_input = true;
        EXPECT_TRUE(overlayPressMayFocusPanel(dismiss));

        // Text dismissal grants focus only inside the viewport. Overlay bounds include
        // the left dock, but clicking a dock control while editing Right must not focus
        // Left; blur cannot give that press new focus permission.
        auto dock_dismiss = dismiss;
        dock_dismiss.press_inside_viewport = false;
        EXPECT_FALSE(overlayPressMayFocusPanel(dock_dismiss));

        // The containment term binds the ordinary path too: nothing outside the
        // viewport rectangle resolves to a split panel.
        auto dock_press = kViewportPress;
        dock_press.press_inside_viewport = false;
        EXPECT_FALSE(overlayPressMayFocusPanel(dock_press));

        // No press, no focus change -- hover and release frames decide nothing.
        auto no_press = kViewportPress;
        no_press.left_pressed = false;
        EXPECT_FALSE(overlayPressMayFocusPanel(no_press));
        no_press.overlay_wants_input = false;
        no_press.press_blurred_text_input = true;
        EXPECT_FALSE(overlayPressMayFocusPanel(no_press));

        // A press the overlay ignored entirely is not this block's business;
        // InputController focuses those on the viewport path.
        auto unclaimed = kViewportPress;
        unclaimed.overlay_wants_input = false;
        EXPECT_FALSE(overlayPressMayFocusPanel(unclaimed));

        // The event-time GUI verdict vetoes focus even inside viewport bounds. Half the
        // dock resize strip lies beyond the dock edge inside the viewport; containment
        // must not turn a GUI-owned resize into panel focus.
        auto resize_strip = kViewportPress;
        resize_strip.press_gui_owned = true;
        EXPECT_FALSE(overlayPressMayFocusPanel(resize_strip));

        // ...and the focused-field route, which is the one that could actually
        // reach it: editing Right, pressing the resize strip must not focus Left.
        auto resize_strip_dismiss = dismiss;
        resize_strip_dismiss.press_gui_owned = true;
        EXPECT_FALSE(overlayPressMayFocusPanel(resize_strip_dismiss));
    }

    // Containment is asked of the VIEWPORT rectangle, never of the overlay's own
    // (wider) bounds. Half-open on both axes so adjacent regions cannot both
    // claim the same pixel.
    TEST(DepthWindowOverlayPressFocusTest, ViewportContainmentExcludesTheDockAndTheEdges) {
        using lfs::vis::gui::pointInsideViewport;

        // Viewport starts at x = 320 (the left dock occupies 0..320) and is
        // 800x600; the overlay's bounds would start at x = 0.
        const glm::vec2 pos{320.0f, 40.0f};
        const glm::vec2 size{800.0f, 600.0f};

        // The half-open viewport includes its exact left boundary, but the dock resize
        // strip is centered there too. Containment alone cannot grant focus; GUI
        // ownership wins when the two disagree.
        EXPECT_TRUE(pointInsideViewport({320.0f, 40.0f}, pos, size));
        EXPECT_FALSE(lfs::vis::gui::overlayPressMayFocusPanel({
            .left_pressed = true,
            .overlay_wants_input = false,
            .pressed_interactive_control = false,
            .press_blurred_text_input = true,
            .press_inside_viewport = pointInsideViewport({320.0f, 40.0f}, pos, size),
            .press_gui_owned = true,
        })) << "the boundary passes containment, so only the GUI's event-time "
               "ownership can keep a resize-strip press from refocusing a panel";
        EXPECT_TRUE(pointInsideViewport({700.0f, 300.0f}, pos, size));
        EXPECT_TRUE(pointInsideViewport({1119.9f, 639.9f}, pos, size));

        // Inside the overlay, outside the viewport: the dock.
        EXPECT_FALSE(pointInsideViewport({120.0f, 300.0f}, pos, size));
        EXPECT_FALSE(pointInsideViewport({319.9f, 300.0f}, pos, size));
        // Chrome above/below, and past the far edges.
        EXPECT_FALSE(pointInsideViewport({700.0f, 39.9f}, pos, size));
        EXPECT_FALSE(pointInsideViewport({1120.0f, 300.0f}, pos, size));
        EXPECT_FALSE(pointInsideViewport({700.0f, 640.0f}, pos, size));
        // A degenerate viewport contains nothing.
        EXPECT_FALSE(pointInsideViewport({700.0f, 300.0f}, pos, {0.0f, 600.0f}));
    }

    // SDL batches events, so motion after a press already appears in frame-end
    // mouse_x/mouse_y. Classify each DOWN using its own recorded coordinates.
    namespace {
        // Use non-default timestamps and click counts to prove capture/copy preserves
        // them. Zero would also pass if either field were silently dropped to its
        // default.
        SDL_Event mouseDownEvent(const int sdl_button, const float x, const float y,
                                 const Uint64 timestamp = 0, const int clicks = 0) {
            SDL_Event event{};
            event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
            event.button.button = static_cast<Uint8>(sdl_button);
            event.button.x = x;
            event.button.y = y;
            event.button.timestamp = timestamp;
            event.button.clicks = static_cast<Uint8>(clicks);
            return event;
        }

        SDL_Event mouseUpEvent(const int sdl_button, const float x, const float y,
                               const Uint64 timestamp = 0, const int clicks = 0) {
            SDL_Event event{};
            event.type = SDL_EVENT_MOUSE_BUTTON_UP;
            event.button.button = static_cast<Uint8>(sdl_button);
            event.button.x = x;
            event.button.y = y;
            event.button.timestamp = timestamp;
            event.button.clicks = static_cast<Uint8>(clicks);
            return event;
        }

        SDL_Event mouseMotionEvent(const float x, const float y) {
            SDL_Event event{};
            event.type = SDL_EVENT_MOUSE_MOTION;
            event.motion.x = x;
            event.motion.y = y;
            return event;
        }

        // Everything finalize() does that matters here. finalize() itself needs
        // an SDL_Window: it samples the LIVE cursor into mouse_x/mouse_y, which
        // is precisely the position these tests must NOT be classified from.
        void settleLiveCursor(lfs::vis::FrameInputBuffer& buffer,
                              const float x, const float y) {
            buffer.mouse_x = x;
            buffer.mouse_y = y;
        }
    } // namespace

    TEST(DepthWindowOverlayPressFocusTest, PressPointSurvivesMotionCoalescedBehindThePress) {
        using lfs::vis::gui::buildPanelInputFromSDL;
        using lfs::vis::gui::pointInsideViewport;

        const glm::vec2 pos{320.0f, 40.0f};
        const glm::vec2 size{800.0f, 600.0f};

        // Use synthetic SDL through FrameInputBuffer intake and the production
        // PanelInputState copy. Hand-seeding the destination would leave both failure
        // sites untested.
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        // DOWN on the right half of the viewport...
        buffer.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 900.0f, 300.0f));
        // ...then the cursor is dragged onto the left dock inside the same
        // buffered frame. SDL delivers both to the same frame.
        buffer.processEvent(mouseMotionEvent(100.0f, 300.0f));
        settleLiveCursor(buffer, 100.0f, 300.0f);

        const auto input = buildPanelInputFromSDL(buffer);
        ASSERT_TRUE(input.mouse_clicked[0]);
        const auto* const press = input.lastPress(0);
        ASSERT_NE(press, nullptr)
            << "the button-down did not survive capture and copy";
        EXPECT_EQ(glm::vec2(press->x, press->y), glm::vec2(900.0f, 300.0f))
            << "the button-down coordinates did not survive capture and copy";
        EXPECT_TRUE(pointInsideViewport({press->x, press->y}, pos, size));
        // The latest cursor position -- what the code used to read -- would have
        // classified this press as a dock press and resolved the wrong panel.
        EXPECT_FALSE(pointInsideViewport({input.mouse_x, input.mouse_y}, pos, size));

        // The inverse: a press on the dock followed by motion into the viewport
        // must not become a viewport press. Asymmetric coordinates, so a copy
        // that swaps X and Y cannot pass either direction.
        lfs::vis::FrameInputBuffer inverse_buffer;
        inverse_buffer.beginFrame();
        inverse_buffer.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 100.0f, 500.0f));
        inverse_buffer.processEvent(mouseMotionEvent(900.0f, 300.0f));
        settleLiveCursor(inverse_buffer, 900.0f, 300.0f);
        const auto inverse = buildPanelInputFromSDL(inverse_buffer);
        const auto* const inverse_press = inverse.lastPress(0);
        ASSERT_NE(inverse_press, nullptr);
        EXPECT_EQ(glm::vec2(inverse_press->x, inverse_press->y), glm::vec2(100.0f, 500.0f));
        EXPECT_FALSE(pointInsideViewport({inverse_press->x, inverse_press->y}, pos, size));

        // One frame may contain DOWN/UP or two complete presses. Preserve every DOWN's
        // point and arrival order, apply focus per press, and let lastPress inspect the
        // last without changing the stream.
        lfs::vis::FrameInputBuffer double_press;
        double_press.beginFrame();
        double_press.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 700.0f, 300.0f));
        double_press.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 120.0f, 310.0f));
        settleLiveCursor(double_press, 120.0f, 310.0f);
        const auto doubled = buildPanelInputFromSDL(double_press);
        ASSERT_EQ(doubled.mouse_button_events.size(), 2u)
            << "the canonical stream was coalesced";
        EXPECT_FLOAT_EQ(doubled.mouse_button_events[0].x, 700.0f);
        EXPECT_FLOAT_EQ(doubled.mouse_button_events[1].x, 120.0f);
        ASSERT_NE(doubled.lastPress(0), nullptr);
        EXPECT_FLOAT_EQ(doubled.lastPress(0)->x, 120.0f);

        // Per-button, not per-frame: a right press must not answer for the left.
        lfs::vis::FrameInputBuffer right_only;
        right_only.beginFrame();
        right_only.processEvent(mouseDownEvent(SDL_BUTTON_RIGHT, 640.0f, 200.0f));
        settleLiveCursor(right_only, 700.0f, 300.0f);
        const auto right_input = buildPanelInputFromSDL(right_only);
        ASSERT_NE(right_input.lastPress(1), nullptr);
        EXPECT_EQ(glm::vec2(right_input.lastPress(1)->x, right_input.lastPress(1)->y),
                  glm::vec2(640.0f, 200.0f));
        // With no press for that button there is no press event to hand back,
        // and callers get nothing rather than another button's coordinates.
        EXPECT_EQ(right_input.lastPress(0), nullptr);
        EXPECT_EQ(right_input.lastPress(-1), nullptr);
        EXPECT_EQ(right_input.lastPress(3), nullptr);
    }

    // Record GUI ownership at every BUTTON_DOWN, including repeated same-button DOWNs.
    // Do not re-derive it from frame-time rectangles or retain only the first press's
    // verdict.
    TEST(DepthWindowOverlayPressFocusTest, PressOwnershipIsRecordedAtTheButtonDown) {
        using lfs::vis::gui::buildPanelInputFromSDL;

        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        buffer.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 320.0f, 300.0f));
        // What WindowManager does at the event, from GuiManager::hitTestMouseButton.
        buffer.notePressOwner(SDL_BUTTON_LEFT, /*gui_owned=*/true);
        // A second verdict for the SAME press is ignored: the owner and the
        // coordinates it was taken at must describe one event.
        buffer.notePressOwner(SDL_BUTTON_LEFT, /*gui_owned=*/false);
        settleLiveCursor(buffer, 320.0f, 300.0f);

        auto input = buildPanelInputFromSDL(buffer);
        ASSERT_NE(input.lastPress(0), nullptr);
        EXPECT_TRUE(input.lastPress(0)->gui_owned);
        EXPECT_EQ(input.lastPress(1), nullptr) << "ownership leaked across buttons";

        // A DOWN whose verdict was never recorded stays unowned rather than
        // inheriting the previous press's.
        buffer.beginFrame();
        buffer.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 700.0f, 300.0f));
        settleLiveCursor(buffer, 700.0f, 300.0f);
        input = buildPanelInputFromSDL(buffer);
        ASSERT_NE(input.lastPress(0), nullptr);
        EXPECT_FALSE(input.lastPress(0)->gui_owned)
            << "last frame's dock-edge verdict vetoed this frame's viewport press";

        // A stray verdict with no DOWN awaiting one owns nothing.
        buffer.beginFrame();
        buffer.notePressOwner(SDL_BUTTON_LEFT, /*gui_owned=*/true);
        EXPECT_TRUE(buildPanelInputFromSDL(buffer).mouse_button_events.empty());

        // EVERY DOWN gets its own verdict, including a second DOWN for the same
        // button in the same frame. "First DOWN of the frame wins" would have
        // given the second press the first one's owner.
        lfs::vis::FrameInputBuffer twice;
        twice.beginFrame();
        twice.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 320.0f, 300.0f));
        twice.notePressOwner(SDL_BUTTON_LEFT, /*gui_owned=*/true);
        twice.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 320.0f, 300.0f));
        twice.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 700.0f, 300.0f));
        twice.notePressOwner(SDL_BUTTON_LEFT, /*gui_owned=*/false);
        settleLiveCursor(twice, 700.0f, 300.0f);
        const auto twice_input = buildPanelInputFromSDL(twice);
        ASSERT_EQ(twice_input.mouse_button_events.size(), 3u);
        EXPECT_TRUE(twice_input.mouse_button_events[0].gui_owned);
        EXPECT_TRUE(twice_input.mouse_button_events[1].gui_owned)
            << "the UP did not follow its own DOWN's owner";
        EXPECT_FALSE(twice_input.mouse_button_events[2].gui_owned)
            << "the second press inherited the first press's owner";
        ASSERT_NE(twice_input.lastPress(0), nullptr);
        EXPECT_FALSE(twice_input.lastPress(0)->gui_owned);
    }

    // A press LIFECYCLE outlives the frame it started in: the release must find
    // its own DOWN's verdict however many frames later it arrives, and an
    // unmatched release must find nothing.
    TEST(DepthWindowOverlayPressFocusTest, ReleaseFollowsItsOwnDownAcrossFrames) {
        using lfs::vis::gui::buildPanelInputFromSDL;

        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        buffer.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 320.0f, 300.0f));
        buffer.notePressOwner(SDL_BUTTON_LEFT, /*gui_owned=*/true);
        ASSERT_EQ(buffer.mouse_button_events.size(), 1u);
        EXPECT_TRUE(buffer.mouse_button_events[0].gui_owned);

        // ...held across an idle frame...
        buffer.beginFrame();
        EXPECT_TRUE(buffer.mouse_button_events.empty());

        // ...and released in a third frame, far outside the control it started
        // on. The release still belongs to the press that opened it.
        buffer.beginFrame();
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 1100.0f, 620.0f));
        const auto released = buildPanelInputFromSDL(buffer);
        ASSERT_EQ(released.mouse_button_events.size(), 1u);
        EXPECT_FALSE(released.mouse_button_events[0].down);
        EXPECT_TRUE(released.mouse_button_events[0].gui_owned)
            << "a release outside the pressed control lost its own DOWN's owner";

        // A SECOND release with no press behind it inherits nothing.
        buffer.beginFrame();
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 1100.0f, 620.0f));
        const auto unmatched = buildPanelInputFromSDL(buffer);
        ASSERT_EQ(unmatched.mouse_button_events.size(), 1u);
        EXPECT_FALSE(unmatched.mouse_button_events[0].gui_owned)
            << "an unmatched release inherited an earlier same-button press";

        // ...and neither does a release of a button that never went down.
        lfs::vis::FrameInputBuffer other;
        other.beginFrame();
        other.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 320.0f, 300.0f));
        other.notePressOwner(SDL_BUTTON_LEFT, /*gui_owned=*/true);
        other.processEvent(mouseUpEvent(SDL_BUTTON_RIGHT, 320.0f, 300.0f));
        ASSERT_EQ(other.mouse_button_events.size(), 2u);
        EXPECT_FALSE(other.mouse_button_events[1].gui_owned)
            << "ownership leaked from the left press to the right release";
    }

    // Preserve arrival order across buttons. Reversing right-then-left invents motion
    // after the left DOWN and can start a drag. RmlPointerReplayTest observes this
    // consequence on a real RmlUi context.
    TEST(DepthWindowOverlayPressFocusTest, PressArrivalOrderIsRecordedAtTheButtonDown) {
        using lfs::vis::gui::buildPanelInputFromSDL;

        // RIGHT first, then LEFT, in one buffered frame.
        lfs::vis::FrameInputBuffer right_first;
        right_first.beginFrame();
        right_first.processEvent(mouseDownEvent(SDL_BUTTON_RIGHT, 640.0f, 200.0f));
        right_first.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 700.0f, 300.0f));
        settleLiveCursor(right_first, 700.0f, 300.0f);
        const auto right_input = buildPanelInputFromSDL(right_first);
        ASSERT_EQ(right_input.mouse_button_events.size(), 2u);
        EXPECT_EQ(right_input.mouse_button_events[0].button, 1)
            << "the right press arrived first and the stream does not say so";
        EXPECT_EQ(right_input.mouse_button_events[1].button, 0);

        // The same two buttons the other way round must read the other way
        // round: an order that merely encoded the button index would pass the
        // case above and fail here.
        lfs::vis::FrameInputBuffer left_first;
        left_first.beginFrame();
        left_first.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 700.0f, 300.0f));
        left_first.processEvent(mouseDownEvent(SDL_BUTTON_RIGHT, 640.0f, 200.0f));
        settleLiveCursor(left_first, 640.0f, 200.0f);
        const auto left_input = buildPanelInputFromSDL(left_first);
        ASSERT_EQ(left_input.mouse_button_events.size(), 2u);
        EXPECT_EQ(left_input.mouse_button_events[0].button, 0);
        EXPECT_EQ(left_input.mouse_button_events[1].button, 1);

        // A repeated press is a THIRD event at its own point, not a re-stamp of
        // the first: nothing is collapsed and nothing moves.
        lfs::vis::FrameInputBuffer repeated;
        repeated.beginFrame();
        repeated.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 700.0f, 300.0f));
        repeated.processEvent(mouseDownEvent(SDL_BUTTON_RIGHT, 640.0f, 200.0f));
        repeated.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 660.0f, 310.0f));
        settleLiveCursor(repeated, 660.0f, 310.0f);
        const auto repeated_input = buildPanelInputFromSDL(repeated);
        ASSERT_EQ(repeated_input.mouse_button_events.size(), 3u);
        EXPECT_EQ(repeated_input.mouse_button_events[0].button, 0);
        EXPECT_FLOAT_EQ(repeated_input.mouse_button_events[0].x, 700.0f);
        EXPECT_EQ(repeated_input.mouse_button_events[1].button, 1);
        EXPECT_EQ(repeated_input.mouse_button_events[2].button, 0);
        EXPECT_FLOAT_EQ(repeated_input.mouse_button_events[2].x, 660.0f);

        // Last frame's events are cleared, so no ordering can ever be compared
        // across frames.
        right_first.beginFrame();
        EXPECT_TRUE(right_first.mouse_button_events.empty());
    }

    // ------------------------------------------------------------------
    // Escape contract on real RmlUi elements. wantsTextInput excludes ordinary selects,
    // so a text-only outer gate can make select handling unreachable. Whole-file
    // searches miss this. Pin the shared cancellation decision and IME composition's
    // claim on Escape.
    // ------------------------------------------------------------------
    class OverlayEscapeContractTest : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            ASSERT_TRUE(Rml::Initialise());
        }

        static void TearDownTestSuite() {
            Rml::Shutdown();
        }

        Rml::Element* make(const Rml::String& tag) {
            auto element = document_.CreateElement(tag);
            EXPECT_TRUE(element) << "RmlUi could not instance <" << tag << ">";
            return document_.AppendChild(std::move(element));
        }

        Rml::ElementDocument document_{"body"};
    };

    TEST_F(OverlayEscapeContractTest, CancelTargetsAreTextEditablesAndSelects) {
        using lfs::vis::gui::rml_input::isEscapeCancelTarget;

        auto* const text = make("input");
        ASSERT_NE(text, nullptr);
        text->SetAttribute("type", "text");
        EXPECT_TRUE(isEscapeCancelTarget(text));

        auto* const textarea = make("textarea");
        ASSERT_NE(textarea, nullptr);
        EXPECT_TRUE(isEscapeCancelTarget(textarea));

        // The case the viewport overlay could not reach. A select is NOT a text
        // input -- wantsTextInput() rejects it -- yet Escape must still close it,
        // which is what the sidebar host does.
        auto* const select = make("select");
        ASSERT_NE(select, nullptr);
        EXPECT_FALSE(lfs::vis::gui::rml_input::wantsTextInput(select))
            << "a select must not be treated as a text input";
        EXPECT_TRUE(isEscapeCancelTarget(select));

        // Focus inside an open dropdown lands on a descendant, not the select.
        auto option = document_.CreateElement("div");
        ASSERT_TRUE(option);
        auto* const inside = select->AppendChild(std::move(option));
        ASSERT_NE(inside, nullptr);
        EXPECT_TRUE(isEscapeCancelTarget(inside));

        // Everything else keeps Escape's ordinary meaning.
        EXPECT_FALSE(isEscapeCancelTarget(make("button")));
        EXPECT_FALSE(isEscapeCancelTarget(make("div")));
        EXPECT_FALSE(isEscapeCancelTarget(nullptr));
    }

    TEST_F(OverlayEscapeContractTest, ImeCompositionKeepsEscape) {
        using lfs::vis::gui::rml_input::shouldCancelOnEscape;

        auto* const text = make("input");
        ASSERT_NE(text, nullptr);
        text->SetAttribute("type", "text");

        EXPECT_TRUE(shouldCancelOnEscape(text, /*composing=*/false));
        // While an IME composition is in flight Escape aborts the composition.
        // No host may steal it to revert the field, or the composition can never
        // be cancelled without also throwing the edit away.
        EXPECT_FALSE(shouldCancelOnEscape(text, /*composing=*/true));

        auto* const select = make("select");
        ASSERT_NE(select, nullptr);
        EXPECT_TRUE(shouldCancelOnEscape(select, /*composing=*/false));
        EXPECT_FALSE(shouldCancelOnEscape(select, /*composing=*/true));

        EXPECT_FALSE(shouldCancelOnEscape(nullptr, /*composing=*/false));
    }

    // ------------------------------------------------------------------
    // Pointer delivery through production replay and a headless Rml::Context. Exercise
    // rml_pointer_dispatch.hpp as used by processInput, observing RmlUi dispatches.
    // This does not execute a complete GUI frame.
    //
    // Keep the canonical stream intact. Delivered transitions retain their own point,
    // button and arrival order; skip unowned events without disturbing that order.
    // Ownership must not leak across buttons or repeated presses.
    // ------------------------------------------------------------------
    namespace {

        // The minimum RmlUi needs to instance a context. Nothing is drawn.
        class NullRenderInterface final : public Rml::RenderInterface {
        public:
            Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>,
                                                        Rml::Span<const int>) override {
                return Rml::CompiledGeometryHandle(1);
            }
            void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f,
                                Rml::TextureHandle) override {}
            void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
            Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions, const Rml::String&) override {
                dimensions = Rml::Vector2i(1, 1);
                return Rml::TextureHandle(1);
            }
            Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>,
                                               Rml::Vector2i) override {
                return Rml::TextureHandle(1);
            }
            void ReleaseTexture(Rml::TextureHandle) override {}
            void EnableScissorRegion(bool) override {}
            void SetScissorRegion(Rml::Rectanglei) override {}
        };

        // Record element:event order and each RmlUi button parameter (-1 when absent,
        // such as motion/hover). Ordering and counts alone would let a host replay both
        // buttons as 0; assert identity independently of loop index.
        class PointerEventRecorder final : public Rml::EventListener {
        public:
            void ProcessEvent(Rml::Event& event) override {
                auto* const target = event.GetTargetElement();
                log_.push_back((target ? target->GetId() : Rml::String("<null>")) + ":" +
                               event.GetType());
                buttons_.push_back(event.GetParameter<int>("button", -1));
            }

            void listen(Rml::Element* element) {
                // Observe dragstart: primary DOWN arms drag and later movement starts
                // it. Reversing button order can manufacture that movement and a drag.
                for (const char* type : {"mouseover", "mouseout", "mousemove", "mousedown",
                                         "mouseup", "click", "dragstart", "drag"})
                    element->AddEventListener(type, this);
            }

            void clear() {
                log_.clear();
                buttons_.clear();
            }
            const std::vector<Rml::String>& log() const { return log_; }

            // The `button` parameter of the entry at `index`, or -1 when the
            // index is out of range (so a missing event reads as "no button"
            // rather than crashing the assertion that names it).
            int buttonAt(int index) const {
                if (index < 0 || static_cast<std::size_t>(index) >= buttons_.size())
                    return -1;
                return buttons_[static_cast<std::size_t>(index)];
            }

            // How many entries equal `entry` AND were delivered with `button`.
            int countOfWithButton(const Rml::String& entry, int button) const {
                int count = 0;
                for (std::size_t i = 0; i < log_.size(); ++i) {
                    if (log_[i] == entry && buttons_[i] == button)
                        ++count;
                }
                return count;
            }

            // Index of the first entry equal to `entry`, or -1.
            int indexOf(const Rml::String& entry) const {
                for (std::size_t i = 0; i < log_.size(); ++i) {
                    if (log_[i] == entry)
                        return static_cast<int>(i);
                }
                return -1;
            }

            // How many entries equal `entry`. Distinguishes "this element was
            // pressed once" from "it was pressed once per button".
            int countOf(const Rml::String& entry) const {
                int count = 0;
                for (const auto& logged : log_) {
                    if (logged == entry)
                        ++count;
                }
                return count;
            }

            // Index of the first entry naming `id`, or -1.
            int firstIndexFor(const Rml::String& id) const {
                for (std::size_t i = 0; i < log_.size(); ++i) {
                    if (log_[i].rfind(id + ":", 0) == 0)
                        return static_cast<int>(i);
                }
                return -1;
            }

            // Diagnostic only. Button-carrying entries are suffixed "#<button>"
            // so a failure shows WHICH button was delivered, not just that one
            // was.
            Rml::String joined() const {
                Rml::String out;
                for (std::size_t i = 0; i < log_.size(); ++i) {
                    if (!out.empty())
                        out += " -> ";
                    out += log_[i];
                    if (buttons_[i] >= 0)
                        out += "#" + std::to_string(buttons_[i]);
                }
                return out;
            }

        private:
            std::vector<Rml::String> log_;
            std::vector<int> buttons_;
        };

    } // namespace

    class RmlPointerReplayTest : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            render_interface_ = new NullRenderInterface();
            ASSERT_TRUE(Rml::Initialise());
        }

        static void TearDownTestSuite() {
            Rml::Shutdown();
            delete render_interface_;
            render_interface_ = nullptr;
        }

        void SetUp() override {
            context_ = Rml::CreateContext("pointer_delivery", Rml::Vector2i(400, 300),
                                          render_interface_);
            ASSERT_NE(context_, nullptr) << "headless RmlUi context could not be created";

            // Two non-overlapping absolute boxes, far enough apart that a press
            // in one and a cursor in the other cannot be confused.
            static constexpr const char* kDocument =
                "<rml><head><style>"
                "body { width: 400px; height: 300px; }"
                "div { position: absolute; top: 0px; width: 100px; height: 100px; }"
                "#alpha { left: 0px; }"
                "#beta { left: 200px; }"
                // A DRAGGABLE third box, well clear of the other two. Kept
                // separate rather than making `alpha` draggable so the ordering
                // and ownership tests above stay free of drag semantics.
                "#dragger { left: 0px; top: 150px; drag: drag; }"
                "</style></head>"
                "<body><div id=\"alpha\"/><div id=\"beta\"/><div id=\"dragger\"/></body></rml>";

            document_ = context_->LoadDocumentFromMemory(kDocument);
            ASSERT_NE(document_, nullptr);
            document_->Show();
            context_->Update();

            alpha_ = document_->GetElementById("alpha");
            beta_ = document_->GetElementById("beta");
            dragger_ = document_->GetElementById("dragger");
            ASSERT_NE(alpha_, nullptr);
            ASSERT_NE(beta_, nullptr);
            ASSERT_NE(dragger_, nullptr);

            // Guard the fixture itself: if layout did not run, every hover
            // assertion below would pass vacuously against a null hover.
            ASSERT_EQ(context_->GetElementAtPoint(Rml::Vector2f(50.0f, 50.0f)), alpha_);
            ASSERT_EQ(context_->GetElementAtPoint(Rml::Vector2f(250.0f, 50.0f)), beta_);
            ASSERT_EQ(context_->GetElementAtPoint(Rml::Vector2f(50.0f, 200.0f)), dragger_);
            // ...and that `dragger` really is draggable: every dragstart
            // assertion below would pass vacuously against a box RmlUi refuses
            // to drag.
            ASSERT_EQ(dragger_->GetComputedValues().drag(), Rml::Style::Drag::Drag);

            recorder_.listen(alpha_);
            recorder_.listen(beta_);
            recorder_.listen(dragger_);
            recorder_.clear();
        }

        void TearDown() override {
            if (context_)
                Rml::RemoveContext(context_->GetName());
            context_ = nullptr;
            document_ = nullptr;
            alpha_ = nullptr;
            beta_ = nullptr;
            dragger_ = nullptr;
        }

        template <typename OwnsElementFn>
        bool replay(bool (&down)[3], const std::vector<FrameMouseButtonEvent>& events,
                    const OwnsElementFn& owns, const bool capture = false) {
            return gui::rml_input::replayButtonEvents(*context_, down, events, {0.f, 0.f},
                                                      {400.f, 300.f}, 0, capture, owns);
        }

        void press(const int x, const int y) {
            context_->ProcessMouseMove(x, y, 0);
            context_->ProcessMouseButtonDown(0, 0);
        }

        static inline NullRenderInterface* render_interface_ = nullptr;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::Element* alpha_ = nullptr;
        Rml::Element* beta_ = nullptr;
        Rml::Element* dragger_ = nullptr;
        PointerEventRecorder recorder_;
    };

    namespace {

        // Use the production admission question at each point. Boxes represent
        // interactive overlay chrome; nullptr outside the context is never owned.
        auto ownsOverlayBox(std::vector<Rml::Element*> owned) {
            return [owned = std::move(owned)](const Rml::Element* const element) {
                return element != nullptr &&
                       std::find(owned.begin(), owned.end(), element) != owned.end();
            };
        }

        // Everything the SDL intake would have produced, without an SDL_Window:
        // one buffered frame's worth of transitions, each with its own point and
        // its own ownership verdict.
        void pressAt(lfs::vis::FrameInputBuffer& buffer, const int sdl_button,
                     const float x, const float y, const bool gui_owned,
                     const Uint64 timestamp = 0, const int clicks = 0) {
            buffer.processEvent(mouseDownEvent(sdl_button, x, y, timestamp, clicks));
            buffer.notePressOwner(sdl_button, gui_owned);
        }

        // The host's per-button press lifecycle, fresh for one test. The
        // production owner is RmlViewportOverlay::pointer_down_delivered_.
        struct ReplayState {
            bool down_delivered[3] = {};
        };

    } // namespace

    // The host hover pass moves to frame-end position before replay. Every transition
    // must still use its own point: alpha's DOWN must reach alpha even though the frame
    // ended over beta.
    TEST_F(RmlPointerReplayTest, EachEventIsDeliveredAtItsOwnPoint) {
        // The host's own hover move to the frame-end cursor, as processInput
        // makes it before the replay.
        context_->ProcessMouseMove(250, 50, 0); // beta
        recorder_.clear();

        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 50.0f, /*gui_owned=*/false); // alpha

        const bool replayed = replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_}));
        EXPECT_TRUE(replayed);

        const auto trace = recorder_.joined();
        const int down = recorder_.indexOf("alpha:mousedown");
        ASSERT_GE(down, 0) << "the DOWN did not land on the pressed element: " << trace;
        EXPECT_EQ(recorder_.buttonAt(down), 0) << trace;
        EXPECT_EQ(context_->GetHoverElement(), alpha_) << trace;
    }

    // Skip unowned dragger while preserving later owned events. Alpha's DOWN must still
    // arrive in its original position as button 0.
    TEST_F(RmlPointerReplayTest, UnownedEventsAreSkippedWithoutDisturbingTheRest) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_RIGHT, 50.0f, 200.0f, /*gui_owned=*/false); // dragger
        pressAt(buffer, SDL_BUTTON_LEFT, 250.0f, 50.0f, /*gui_owned=*/false);  // beta

        const bool replayed = replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_}));
        EXPECT_TRUE(replayed);

        const auto trace = recorder_.joined();
        EXPECT_EQ(recorder_.countOf("dragger:mousedown"), 0)
            << "an unowned press was delivered: " << trace;
        EXPECT_EQ(recorder_.countOfWithButton("beta:mousedown", 0), 1) << trace;

        // ...and an event outside the context's rectangle is not owned either.
        recorder_.clear();
        ReplayState outside_state;
        lfs::vis::FrameInputBuffer outside;
        outside.beginFrame();
        pressAt(outside, SDL_BUTTON_LEFT, 900.0f, 900.0f, /*gui_owned=*/false);
        EXPECT_FALSE(replay(outside_state.down_delivered, outside.mouse_button_events, ownsOverlayBox({alpha_, beta_})));
        EXPECT_TRUE(recorder_.log().empty()) << recorder_.joined();
    }

    // A capture this host already owns (the VRAM HUD, the toolbar mid-drag)
    // delivers its events wherever they land -- that is what keeps a drag alive
    // once the pointer leaves the control.
    TEST_F(RmlPointerReplayTest, CaptureDeliversEventsThatLandOnNothingOfOurs) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 200.0f, /*gui_owned=*/false); // dragger

        EXPECT_TRUE(replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_}), /*capture=*/true));
        EXPECT_EQ(recorder_.countOfWithButton("dragger:mousedown", 0), 1)
            << recorder_.joined();
    }

    // Replay right-then-left in that order. Reversing them moves onto dragger after
    // primary DOWN arms drag, manufacturing dragstart without user motion.
    TEST_F(RmlPointerReplayTest, RightBeforeLeftStartsNoDragOnTheLeftTarget) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_RIGHT, 250.0f, 50.0f, /*gui_owned=*/false); // beta
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 200.0f, /*gui_owned=*/false);  // dragger

        EXPECT_TRUE(replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_, dragger_})));

        const auto trace = recorder_.joined();
        const int right_down = recorder_.indexOf("beta:mousedown");
        const int left_down = recorder_.indexOf("dragger:mousedown");
        ASSERT_GE(right_down, 0) << trace;
        ASSERT_GE(left_down, 0) << trace;
        EXPECT_LT(right_down, left_down)
            << "the recorded arrival order was not the delivery order: " << trace;
        EXPECT_EQ(recorder_.buttonAt(right_down), 1) << trace;
        EXPECT_EQ(recorder_.buttonAt(left_down), 0) << trace;
        EXPECT_EQ(recorder_.countOf("dragger:dragstart"), 0)
            << "a drag was manufactured after the left DOWN: " << trace;
    }

    // ...and the same two buttons the other way round, so an implementation
    // that hard-coded one order cannot pass both.
    TEST_F(RmlPointerReplayTest, LeftBeforeRightIsDeliveredLeftFirst) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 50.0f, /*gui_owned=*/false);   // alpha
        pressAt(buffer, SDL_BUTTON_RIGHT, 250.0f, 50.0f, /*gui_owned=*/false); // beta

        EXPECT_TRUE(replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_})));

        const auto trace = recorder_.joined();
        const int left_down = recorder_.indexOf("alpha:mousedown");
        const int right_down = recorder_.indexOf("beta:mousedown");
        ASSERT_GE(left_down, 0) << trace;
        ASSERT_GE(right_down, 0) << trace;
        EXPECT_LT(left_down, right_down) << trace;
        EXPECT_EQ(recorder_.buttonAt(left_down), 0) << trace;
        EXPECT_EQ(recorder_.buttonAt(right_down), 1) << trace;
    }

    // A matching UP follows its delivered DOWN's owner, even outside the control. Press
    // owned alpha and release over unowned dragger: hit-testing UP alone skips the
    // release needed to clear RmlUi's active/drag state.
    TEST_F(RmlPointerReplayTest, OwnedReleaseSurvivesTheCursorLeavingTheControl) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 50.0f, /*gui_owned=*/false); // alpha
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 200.0f));   // dragger

        EXPECT_TRUE(replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_})));

        const auto trace = recorder_.joined();
        EXPECT_EQ(recorder_.countOfWithButton("alpha:mousedown", 0), 1) << trace;
        EXPECT_EQ(recorder_.countOfWithButton("dragger:mouseup", 0), 1)
            << "the release outside the pressed control was withheld: " << trace;
        EXPECT_FALSE(state.down_delivered[0])
            << "the press lifecycle stayed open after its release";
    }

    // Reject an UP whose DOWN this host never delivered. It cannot borrow another
    // button's ownership, revive a refused press, or end a press merely because release
    // lands on chrome.
    TEST_F(RmlPointerReplayTest, ReleaseIsDeliveredOnlyForAMatchingDown) {
        // (a) a bare release, no press behind it at all.
        ReplayState bare_state;
        lfs::vis::FrameInputBuffer bare;
        bare.beginFrame();
        bare.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 50.0f)); // over alpha
        EXPECT_FALSE(replay(bare_state.down_delivered, bare.mouse_button_events, ownsOverlayBox({alpha_, beta_})));
        EXPECT_EQ(recorder_.countOf("alpha:mouseup"), 0) << recorder_.joined();

        // (b) a press this host REFUSED, released over chrome it owns. The
        // refusal is what makes the release unmatched.
        recorder_.clear();
        ReplayState refused_state;
        lfs::vis::FrameInputBuffer refused;
        refused.beginFrame();
        pressAt(refused, SDL_BUTTON_LEFT, 50.0f, 200.0f, /*gui_owned=*/false); // dragger
        refused.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 50.0f));     // alpha
        EXPECT_FALSE(replay(refused_state.down_delivered, refused.mouse_button_events, ownsOverlayBox({alpha_, beta_})));
        EXPECT_TRUE(recorder_.log().empty())
            << "a release with no delivered press of its own was replayed: "
            << recorder_.joined();
        EXPECT_FALSE(refused_state.down_delivered[0]);

        // (c) a stale open press cannot lend its right to a LATER press this
        // host refused: the refused DOWN clears the lifecycle.
        recorder_.clear();
        ReplayState stale_state;
        stale_state.down_delivered[0] = true; // an UP eaten by a focus loss
        lfs::vis::FrameInputBuffer stale;
        stale.beginFrame();
        pressAt(stale, SDL_BUTTON_LEFT, 50.0f, 200.0f, /*gui_owned=*/false); // dragger, refused
        stale.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 200.0f));
        EXPECT_FALSE(replay(stale_state.down_delivered, stale.mouse_button_events, ownsOverlayBox({alpha_, beta_})));
        EXPECT_TRUE(recorder_.log().empty()) << recorder_.joined();
        EXPECT_FALSE(stale_state.down_delivered[0]);
    }

    // An unowned dragger press followed by frame-end hover over owned alpha must
    // deliver nothing. A nonempty canonical vector forbids aggregate replay, which
    // would invent an alpha press at the wrong point and order. The overlay fallback
    // therefore requires an empty vector.
    TEST_F(RmlPointerReplayTest, UnownedPressThenChromeHoverDeliversNoButtonEvent) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 200.0f, /*gui_owned=*/false); // dragger
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 200.0f));
        settleLiveCursor(buffer, 50.0f, 50.0f); // ...and the cursor ends over alpha

        const auto input = lfs::vis::gui::buildPanelInputFromSDL(buffer);
        // The frame the fallback must NOT fire on: real events exist, and the
        // aggregate bits that describe them are set.
        ASSERT_FALSE(input.mouse_button_events.empty());
        ASSERT_TRUE(input.mouse_clicked[0]);
        ASSERT_TRUE(input.mouse_released[0]);

        // The host's own hover pass, as processInput makes it before the replay.
        context_->ProcessMouseMove(50, 50, 0);
        recorder_.clear();

        EXPECT_FALSE(replay(state.down_delivered, input.mouse_button_events, ownsOverlayBox({alpha_, beta_})));
        EXPECT_EQ(recorder_.countOf("alpha:mousedown"), 0)
            << "a press on something else was manufactured on chrome: "
            << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("alpha:mouseup"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("alpha:click"), 0) << recorder_.joined();
        EXPECT_FALSE(state.down_delivered[0]);
    }

    // Exercise mixed owners across buttons in both directions. A frame-wide some-owned
    // flag would pass only one ordering. The rejected button must acquire nothing while
    // the accepted lifecycle arrives intact.
    TEST_F(RmlPointerReplayTest, RejectedLeftLeavesTheAcceptedRightUntouched) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 200.0f, /*gui_owned=*/false);  // dragger
        pressAt(buffer, SDL_BUTTON_RIGHT, 250.0f, 50.0f, /*gui_owned=*/false); // beta
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 200.0f));
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_RIGHT, 250.0f, 50.0f));

        EXPECT_TRUE(replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_})));

        const auto trace = recorder_.joined();
        EXPECT_EQ(recorder_.countOf("dragger:mousedown"), 0)
            << "the rejected left press was delivered: " << trace;
        EXPECT_EQ(recorder_.countOf("dragger:mouseup"), 0) << trace;
        EXPECT_EQ(recorder_.countOfWithButton("beta:mousedown", 1), 1) << trace;
        EXPECT_EQ(recorder_.countOfWithButton("beta:mouseup", 1), 1) << trace;
        EXPECT_FALSE(state.down_delivered[0]) << "the rejected press opened a lifecycle";
        EXPECT_FALSE(state.down_delivered[1]);
    }

    TEST_F(RmlPointerReplayTest, RejectedRightTakesNoOwnershipBesideAnAcceptedLeft) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 50.0f, /*gui_owned=*/false);   // alpha
        pressAt(buffer, SDL_BUTTON_RIGHT, 50.0f, 200.0f, /*gui_owned=*/false); // dragger
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_RIGHT, 50.0f, 50.0f));     // over alpha
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 50.0f));

        EXPECT_TRUE(replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_})));

        const auto trace = recorder_.joined();
        EXPECT_EQ(recorder_.countOfWithButton("alpha:mousedown", 0), 1) << trace;
        EXPECT_EQ(recorder_.countOf("dragger:mousedown"), 0) << trace;
        // The right release lands on chrome this host owns, but its press was
        // refused, so it must not ride the left press's lifecycle in.
        EXPECT_EQ(recorder_.countOfWithButton("alpha:mouseup", 1), 0)
            << "the rejected right press took ownership from the accepted left: " << trace;
        EXPECT_EQ(recorder_.countOfWithButton("alpha:mouseup", 0), 1) << trace;
        EXPECT_FALSE(state.down_delivered[0]);
        EXPECT_FALSE(state.down_delivered[1]);
    }

    // Two independently admitted buttons run two independent lifecycles: each
    // release follows ITS OWN press, and neither closes the other's.
    TEST_F(RmlPointerReplayTest, BothAdmittedPressesReleaseTheirOwnButtons) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 50.0f, /*gui_owned=*/false);   // alpha
        pressAt(buffer, SDL_BUTTON_RIGHT, 250.0f, 50.0f, /*gui_owned=*/false); // beta

        EXPECT_TRUE(replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_})));
        EXPECT_TRUE(state.down_delivered[0]);
        EXPECT_TRUE(state.down_delivered[1]);

        // Released in a LATER frame, each outside the box it started on.
        recorder_.clear();
        buffer.beginFrame();
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_RIGHT, 50.0f, 200.0f)); // dragger
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 200.0f));  // dragger
        EXPECT_TRUE(replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_})));

        const auto trace = recorder_.joined();
        EXPECT_EQ(recorder_.countOfWithButton("dragger:mouseup", 1), 1)
            << "the right release did not follow its own press: " << trace;
        EXPECT_EQ(recorder_.countOfWithButton("dragger:mouseup", 0), 1)
            << "the left release did not follow its own press: " << trace;
        const int right_up = recorder_.indexOf("dragger:mouseup");
        ASSERT_GE(right_up, 0) << trace;
        EXPECT_EQ(recorder_.buttonAt(right_up), 1)
            << "the releases were reordered into button-index order: " << trace;
        EXPECT_FALSE(state.down_delivered[0]);
        EXPECT_FALSE(state.down_delivered[1]);
    }

    // ------------------------------------------------------------------
    // Two complete same-button presses at distinct owned/unowned points within one
    // frame, in both orders. First-DOWN or frame-end coalescing must fail.
    //
    // Derive verdicts from real elements and the replay admission predicate; fixture
    // assertions reject equivalent points. Production intake, copy and replay must
    // preserve all four events, their DOWN/UP identities and their distinct,
    // non-default coordinates, timestamps and click counts.
    //
    // Compare the full delivery sequence except replay-generated mousemove. Reject
    // extra events and ownership leakage: each UP follows its own DOWN, unowned presses
    // are not delivered, and the host lifecycle ends closed.
    // ------------------------------------------------------------------
    namespace {

        struct DoubleClickCase {
            const char* name;
            bool owned_press_first;
        };

        // One press: DOWN then UP at the same point, with its own timestamp and
        // click count, and its ownership verdict taken from `gui_owned`.
        void clickAt(lfs::vis::FrameInputBuffer& buffer, const float x, const float y,
                     const bool gui_owned, const Uint64 timestamp, const int clicks) {
            pressAt(buffer, SDL_BUTTON_LEFT, x, y, gui_owned, timestamp, clicks);
            buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, x, y, timestamp + 1, clicks));
        }

        // Remove only replay-generated mousemove from the delivered sequence: it
        // positions the pointer for each event. Keep every other event so extra
        // presses, releases, clicks and manufactured drags fail the whole-sequence
        // comparison.
        std::vector<Rml::String> withoutCursorMoves(const std::vector<Rml::String>& log) {
            std::vector<Rml::String> out;
            for (const auto& entry : log) {
                if (entry.size() >= 10 && entry.compare(entry.size() - 10, 10, ":mousemove") == 0)
                    continue;
                out.push_back(entry);
            }
            return out;
        }

    } // namespace

    class RmlPointerReplayDoubleClickTest : public RmlPointerReplayTest,
                                            public ::testing::WithParamInterface<DoubleClickCase> {
    };

    TEST_P(RmlPointerReplayDoubleClickTest, TwoPressesOneButtonKeepTheirOwnPointsAndOwners) {
        const auto& params = GetParam();

        // `alpha` is chrome this host owns; `dragger` is not in its owned set.
        // Different x AND different y, so a copy that swapped the two axes could
        // not pass either press.
        constexpr float kOwnedX = 60.0f;
        constexpr float kOwnedY = 50.0f;
        constexpr float kUnownedX = 50.0f;
        constexpr float kUnownedY = 200.0f;

        const auto owns = ownsOverlayBox({alpha_, beta_});
        const auto guiOwnsPoint = [&](const float x, const float y) {
            return owns(context_->GetElementAtPoint(Rml::Vector2f(x, y)));
        };
        // The case is only meaningful if the two points really are on opposite
        // sides of the admission question.
        ASSERT_TRUE(guiOwnsPoint(kOwnedX, kOwnedY));
        ASSERT_FALSE(guiOwnsPoint(kUnownedX, kUnownedY));

        const float first_x = params.owned_press_first ? kOwnedX : kUnownedX;
        const float first_y = params.owned_press_first ? kOwnedY : kUnownedY;
        const float second_x = params.owned_press_first ? kUnownedX : kOwnedX;
        const float second_y = params.owned_press_first ? kUnownedY : kOwnedY;
        const bool first_owned = guiOwnsPoint(first_x, first_y);
        const bool second_owned = guiOwnsPoint(second_x, second_y);
        ASSERT_NE(first_owned, second_owned);

        // Buffer two same-button presses and their releases at distinct points in one
        // frame. Non-default timestamps and click counts distinguish preserved fields
        // from silently dropped defaults.
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        clickAt(buffer, first_x, first_y, first_owned, /*timestamp=*/1100, /*clicks=*/1);
        clickAt(buffer, second_x, second_y, second_owned, /*timestamp=*/1300, /*clicks=*/2);
        settleLiveCursor(buffer, second_x, second_y);

        // --- the COMPLETE canonical stream, through the production copy ------
        const auto input = lfs::vis::gui::buildPanelInputFromSDL(buffer);
        ASSERT_EQ(input.mouse_button_events.size(), 4u)
            << "the stream was coalesced or truncated";
        const auto& events = input.mouse_button_events;

        EXPECT_TRUE(events[0].down);
        EXPECT_FALSE(events[1].down);
        EXPECT_TRUE(events[2].down);
        EXPECT_FALSE(events[3].down);
        for (const auto& event : events)
            EXPECT_EQ(event.button, 0) << "an event changed button identity";

        // EVERY canonical field of every event, in order.
        const float kExpectedX[4] = {first_x, first_x, second_x, second_x};
        const float kExpectedY[4] = {first_y, first_y, second_y, second_y};
        const std::uint64_t kExpectedTimestamp[4] = {1100u, 1101u, 1300u, 1301u};
        const int kExpectedClicks[4] = {1, 1, 2, 2};
        for (int i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(events[i].x, kExpectedX[i]) << "event " << i;
            EXPECT_FLOAT_EQ(events[i].y, kExpectedY[i]) << "event " << i;
            EXPECT_EQ(events[i].timestamp, kExpectedTimestamp[i]) << "event " << i;
            EXPECT_EQ(events[i].clicks, kExpectedClicks[i]) << "event " << i;
        }

        // --- NO OWNERSHIP LEAKAGE -------------------------------------------
        EXPECT_EQ(events[0].gui_owned, first_owned);
        EXPECT_EQ(events[1].gui_owned, first_owned)
            << "the first UP did not follow its own DOWN";
        EXPECT_EQ(events[2].gui_owned, second_owned)
            << "the second press inherited the first press's owner";
        EXPECT_EQ(events[3].gui_owned, second_owned)
            << "the second UP did not follow its own DOWN";

        // --- the COMPLETE GUI-DELIVERY SEQUENCE ------------------------------
        ReplayState state;
        EXPECT_TRUE(replay(state.down_delivered, input.mouse_button_events, owns));

        const auto trace = recorder_.joined();
        // The owned press is delivered whole, at its own point; the unowned one
        // is delivered NOWHERE, in either position in the frame. Compared as the
        // whole sequence, so an extra transition anywhere fails.
        const std::vector<Rml::String> kExpectedDelivery{
            "alpha:mouseover", "alpha:mousedown", "alpha:mouseup", "alpha:click"};
        EXPECT_EQ(withoutCursorMoves(recorder_.log()), kExpectedDelivery)
            << "the delivered sequence was not exactly the owned press: " << trace;

        // ...and the identity of the one press that was delivered.
        EXPECT_EQ(recorder_.countOfWithButton("alpha:mousedown", 0), 1) << trace;
        EXPECT_EQ(recorder_.countOfWithButton("alpha:mouseup", 0), 1) << trace;

        // --- NO STATE LEAKAGE ------------------------------------------------
        EXPECT_FALSE(state.down_delivered[0])
            << "the host still owes RmlUi an UP after both presses closed";
        EXPECT_FALSE(state.down_delivered[1]);
        EXPECT_FALSE(state.down_delivered[2]);
    }

    INSTANTIATE_TEST_SUITE_P(
        BothOwnershipOrderings, RmlPointerReplayDoubleClickTest,
        ::testing::Values(DoubleClickCase{"GuiOwnedFirst", /*owned_press_first=*/true},
                          DoubleClickCase{"ViewportOwnedFirst", /*owned_press_first=*/false}),
        [](const ::testing::TestParamInfo<DoubleClickCase>& info) {
            return std::string(info.param.name);
        });

    // Exercise the compiled overlay entry point used by GuiManager, including
    // its blockers and early returns. Only the headless context attachment is
    // fixture access; ownership, coordinates and delivery are production code.
    class RmlViewportInputRoutingTest : public RmlPointerReplayTest {
    protected:
        void seedToolbarCaptureForCancel() {
            overlay_->viewport_toolbar_position_ = "free";
            overlay_->toolbar_drag_active_ = true;
            // No preference write: this probe never moves the toolbar itself.
            overlay_->toolbar_drag_moved_ = false;
            bar_->AddEventListener(Rml::EventId::Dragend, &overlay_->toolbar_drag_listener_);
        }
        bool toolbarCaptureActive() const { return overlay_->toolbar_drag_active_; }

        void SetUp() override {
            RmlPointerReplayTest::SetUp();
            gui::guiFocusState().reset();
            static constexpr const char* kControls =
                "<rml><head><style>"
                "body { width: 400px; height: 300px; }"
                "input { position: absolute; left: 20px; top: 20px; width: 300px; height: 24px; }"
                "slidertrack { height: 24px; }"
                "sliderbar { width: 20px; height: 24px; }"
                "sliderarrowdec, sliderarrowinc { width: 0px; height: 0px; }"
                "button { position: absolute; left: 20px; top: 100px; width: 100px; height: 40px; }"
                "#text { top: 180px; }"
                "</style></head><body>"
                "<input id=\"range\" type=\"range\" min=\"0\" max=\"100\" step=\"1\" value=\"50\"/>"
                "<button id=\"action\"/><input id=\"text\" type=\"text\" value=\"keep\"/>"
                "</body></rml>";
            controls_ = context_->LoadDocumentFromMemory(kControls);
            ASSERT_NE(controls_, nullptr);
            controls_->Show();
            context_->Update();
            range_ = dynamic_cast<Rml::ElementFormControlInput*>(controls_->GetElementById("range"));
            text_ = dynamic_cast<Rml::ElementFormControlInput*>(controls_->GetElementById("text"));
            button_ = controls_->GetElementById("action");
            ASSERT_NE(range_, nullptr);
            ASSERT_NE(text_, nullptr);
            ASSERT_NE(button_, nullptr);
            bar_ = findElementByTag(range_, "sliderbar");
            ASSERT_NE(bar_, nullptr);
            bar_->SetId("range-bar");
            const auto bar_offset = bar_->GetAbsoluteOffset();
            range_point_ = {bar_offset.x + bar_->GetOffsetWidth() / 2,
                            bar_offset.y + bar_->GetOffsetHeight() / 2};
            ASSERT_EQ(context_->GetElementAtPoint({range_point_.x, range_point_.y}), bar_);
            ASSERT_EQ(context_->GetElementAtPoint({button_point_.x, button_point_.y}), button_);
            ASSERT_EQ(bar_->GetComputedValues().drag(), Rml::Style::Drag::Drag);
            recorder_.listen(bar_);
            recorder_.listen(button_);
            for (const char* event : {"dragend", "dragdrop", "mousescroll"})
                bar_->AddEventListener(event, &recorder_);
            range_->AddEventListener("change", &recorder_);
            text_->AddEventListener("keydown", &recorder_);
            overlay_ = std::make_unique<gui::RmlViewportOverlay>();
            overlay_->rml_context_ = context_;
            overlay_->document_ = controls_;
            overlay_->setViewportBounds(origin_, {400.0f, 300.0f}, {0.0f, 0.0f});
            recorder_.clear();
        }

        void TearDown() override {
            if (overlay_) {
                overlay_->rml_context_ = nullptr;
                overlay_->document_ = nullptr;
                overlay_.reset();
            }
            gui::guiFocusState().reset();
            RmlPointerReplayTest::TearDown();
        }

        static Rml::Element* findElementByTag(Rml::Element* element, const Rml::String& tag) {
            if (element->GetTagName() == tag)
                return element;
            for (int i = 0; i < element->GetNumChildren(true); ++i) {
                if (auto* found = findElementByTag(element->GetChild(i), tag))
                    return found;
            }
            return nullptr;
        }

        gui::PanelInputState inputAt(const glm::vec2 local) const {
            gui::PanelInputState input;
            input.mouse_x = local.x + origin_.x;
            input.mouse_y = local.y + origin_.y;
            return input;
        }

        FrameMouseButtonEvent transition(const bool down, const glm::vec2 local,
                                         const uint8_t button = 0) const {
            return {.button = button,
                    .down = down,
                    .x = local.x + origin_.x,
                    .y = local.y + origin_.y,
                    .timestamp = down ? 1001u : 1002u,
                    .clicks = 1,
                    .gui_owned = true};
        }

        void route(const gui::PanelInputState& input,
                   const gui::ViewportOverlayInputBlockers blockers = {}) {
            const auto original = input.mouse_button_events;
            overlay_->processInput(input, blockers);
            ASSERT_EQ(input.mouse_button_events.size(), original.size());
            for (std::size_t i = 0; i < original.size(); ++i) {
                const auto& event = input.mouse_button_events[i];
                EXPECT_EQ(event.button, original[i].button);
                EXPECT_EQ(event.down, original[i].down);
                EXPECT_FLOAT_EQ(event.x, original[i].x);
                EXPECT_FLOAT_EQ(event.y, original[i].y);
                EXPECT_EQ(event.timestamp, original[i].timestamp);
                EXPECT_EQ(event.clicks, original[i].clicks);
                EXPECT_EQ(event.gui_owned, original[i].gui_owned);
            }
        }

        void arm(const glm::vec2 point, const uint8_t button = 0) {
            auto input = inputAt(point);
            input.mouse_button_events = {transition(true, point, button)};
            input.mouse_clicked[button] = true;
            input.mouse_down[button] = true;
            route(input);
            ASSERT_TRUE(owns(button));
            recorder_.clear();
        }

        bool owns(const uint8_t button = 0) const { return overlay_->pointer_down_delivered_[button]; }
        bool hoveredInteractive() const { return overlay_->hovered_interactive_; }
        bool hasCoordinateOrigin() const { return overlay_->last_valid_input_origin_.has_value(); }

        void expectRangeInert() {
            ASSERT_FALSE(owns());
            EXPECT_FALSE(bar_->IsPseudoClassSet("active"));
            const auto value = range_->GetValue();
            recorder_.clear();
            route(inputAt(range_point_ + glm::vec2(80.0f, 0.0f)));
            EXPECT_EQ(range_->GetValue(), value);
            EXPECT_EQ(recorder_.countOf("range-bar:dragstart"), 0) << recorder_.joined();
            EXPECT_EQ(recorder_.countOf("range:change"), 0) << recorder_.joined();
        }

        std::unique_ptr<gui::RmlViewportOverlay> overlay_;
        Rml::ElementDocument* controls_ = nullptr;
        Rml::ElementFormControlInput* range_ = nullptr;
        Rml::ElementFormControlInput* text_ = nullptr;
        Rml::Element* bar_ = nullptr;
        Rml::Element* button_ = nullptr;
        const glm::vec2 origin_{40.0f, 60.0f};
        const glm::vec2 button_point_{70.0f, 120.0f};
        glm::vec2 range_point_{};
    };

    struct OverlayBlockerCase {
        const char* name;
        gui::ViewportOverlayInputBlockers blockers;
    };

    class RmlViewportInputRoutingBlockerTest : public RmlViewportInputRoutingTest,
                                               public ::testing::WithParamInterface<OverlayBlockerCase> {};

    TEST_P(RmlViewportInputRoutingBlockerTest, HeldRangeDoesNotReceiveMaskedMotionAndTrueReleaseDisarms) {
        arm(range_point_);
        ASSERT_EQ(range_->GetValue(), "50.000000");
        auto held = inputAt({390.0f, 280.0f});
        held.mouse_down[0] = true;
        held.mouse_wheel = 3.0f;
        held.keys_pressed = {SDL_SCANCODE_RIGHT};
        held.text_inputs = {"x"};
        route(held, GetParam().blockers);
        EXPECT_TRUE(owns());
        EXPECT_TRUE(bar_->IsPseudoClassSet("active"));
        EXPECT_EQ(range_->GetValue(), "50.000000");
        EXPECT_TRUE(recorder_.log().empty()) << recorder_.joined();
        EXPECT_TRUE(overlay_->leftPressClassifications().empty());

        auto released = inputAt({390.0f, 280.0f});
        released.mouse_button_events = {transition(false, range_point_)};
        route(released, GetParam().blockers);
        EXPECT_EQ(recorder_.countOf("range-bar:click"), 1) << recorder_.joined();
        EXPECT_EQ(range_->GetValue(), "50.000000");
        expectRangeInert();
    }

    TEST_P(RmlViewportInputRoutingBlockerTest, ButtonReleaseCompletesButNewBlockedPressCannotFocusOrClick) {
        arm(button_point_);
        auto released = inputAt(button_point_);
        released.mouse_button_events = {transition(false, button_point_)};
        route(released, GetParam().blockers);
        EXPECT_FALSE(owns());
        EXPECT_EQ(recorder_.countOf("action:click"), 1) << recorder_.joined();
        recorder_.clear();
        released.mouse_button_events = {transition(true, button_point_), transition(false, button_point_)};
        released.mouse_clicked[0] = true;
        released.mouse_released[0] = true;
        route(released, GetParam().blockers);
        EXPECT_FALSE(owns());
        EXPECT_TRUE(recorder_.log().empty()) << recorder_.joined();
        EXPECT_TRUE(overlay_->leftPressClassifications().empty());
    }

    INSTANTIATE_TEST_SUITE_P(
        GuiBlockers, RmlViewportInputRoutingBlockerTest,
        ::testing::Values(OverlayBlockerCase{"Startup", {.startup = true}},
                          OverlayBlockerCase{"Progress", {.progress = true}},
                          OverlayBlockerCase{"Modal", {.modal = true}},
                          OverlayBlockerCase{"PendingModal", {.pending_modal = true}},
                          OverlayBlockerCase{"ContextMenu", {.context_menu = true}},
                          OverlayBlockerCase{"MenuPointer", {.menu_pointer = true}},
                          OverlayBlockerCase{"FloatingPanel", {.floating_panel = true}}),
        [](const ::testing::TestParamInfo<OverlayBlockerCase>& info) { return info.param.name; });

    TEST_F(RmlViewportInputRoutingTest, CanonicalReleaseBypassesIdleShortcutWithoutAggregateBits) {
        arm(range_point_);
        auto released = inputAt(range_point_);
        released.mouse_button_events = {transition(false, range_point_)};
        route(released);
        EXPECT_EQ(recorder_.countOf("range-bar:click"), 1) << recorder_.joined();
        expectRangeInert();
    }

    TEST_F(RmlViewportInputRoutingTest, BlockedSameButtonInterleavingsKeepTheirOwnLifecycle) {
        arm(button_point_);
        auto input = inputAt(button_point_);
        input.mouse_button_events = {transition(false, button_point_),
                                     transition(true, button_point_),
                                     transition(false, button_point_)};
        route(input, {.modal = true});
        EXPECT_FALSE(owns());
        EXPECT_EQ(recorder_.countOf("action:click"), 1) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("action:mousedown"), 0);

        arm(button_point_);
        input.mouse_button_events = {transition(true, button_point_), transition(false, button_point_)};
        route(input, {.modal = true});
        EXPECT_FALSE(owns());
        EXPECT_EQ(recorder_.countOf("action:click"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("action:mouseup"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("action:mousedown"), 0) << recorder_.joined();
        // A replacement DOWN cancels the old press without borrowing its UP.
        EXPECT_FALSE(button_->IsPseudoClassSet("active"));
        EXPECT_TRUE(overlay_->leftPressClassifications().empty());
    }

    TEST_F(RmlViewportInputRoutingTest, BlockedReleasesRemainIndependentAcrossButtons) {
        arm(range_point_);
        arm(range_point_, 1);
        auto input = inputAt(range_point_);
        input.mouse_button_events = {transition(false, range_point_, 2), transition(false, range_point_, 1)};
        route(input, {.menu_pointer = true});
        EXPECT_TRUE(owns(0));
        EXPECT_FALSE(owns(1));
        EXPECT_FALSE(owns(2));
        EXPECT_EQ(recorder_.countOfWithButton("range-bar:mouseup", 1), 1) << recorder_.joined();
        EXPECT_EQ(recorder_.countOfWithButton("range-bar:mouseup", 2), 0);
        input.mouse_button_events = {transition(false, range_point_)};
        route(input, {.menu_pointer = true});
        expectRangeInert();
    }

    TEST_F(RmlViewportInputRoutingTest, OutsideAndMovedReleasesUseEventCoordinatesInsteadOfLatestHover) {
        arm(button_point_);
        auto input = inputAt(button_point_);
        input.mouse_button_events = {transition(false, {450.0f, 350.0f})};
        route(input, {.floating_panel = true});
        EXPECT_FALSE(owns());
        EXPECT_FALSE(button_->IsPseudoClassSet("active"));
        EXPECT_EQ(recorder_.countOf("action:click"), 0) << recorder_.joined();

        arm(range_point_);
        input = inputAt(range_point_);
        input.mouse_button_events = {transition(false, range_point_ + glm::vec2(60.0f, 0.0f))};
        route(input, {.floating_panel = true});
        EXPECT_EQ(recorder_.countOf("range-bar:dragstart"), 1) << recorder_.joined();
        EXPECT_GE(recorder_.countOf("range:change"), 1);
        EXPECT_NE(range_->GetValue(), "50.000000");
        expectRangeInert();
    }

    TEST_F(RmlViewportInputRoutingTest, CollapsedBoundsUseRetainedLiveContextOriginAndResetOnShutdown) {
        arm(button_point_);
        const auto dimensions = context_->GetDimensions();
        ASSERT_EQ(dimensions, Rml::Vector2i(400, 300));
        overlay_->setViewportBounds({900.0f, 800.0f}, {0.0f, 0.0f}, {0.0f, 0.0f});
        EXPECT_EQ(context_->GetDimensions(), dimensions);
        auto input = inputAt(button_point_);
        input.mouse_button_events = {transition(false, button_point_)};
        route(input);
        EXPECT_FALSE(owns());
        EXPECT_EQ(recorder_.countOf("action:click"), 1) << recorder_.joined();
        EXPECT_TRUE(hasCoordinateOrigin());
        overlay_->shutdown();
        EXPECT_FALSE(hasCoordinateOrigin());
        EXPECT_FALSE(owns());
        recorder_.clear();
        input.mouse_button_events = {transition(true, button_point_), transition(false, button_point_)};
        route(input);
        EXPECT_TRUE(recorder_.log().empty());
    }

    TEST_F(RmlViewportInputRoutingTest, ExternalCaptureCannotSkipOwnedReleaseOutsideOverlay) {
        arm(range_point_);
        auto outside = inputAt({-50.0f, -50.0f});
        outside.mouse_down[0] = true;
        route(outside);
        ASSERT_FALSE(hoveredInteractive());
        ASSERT_TRUE(owns());
        recorder_.clear();
        gui::guiFocusState().want_capture_mouse = true;
        outside.mouse_down[0] = false;
        outside.mouse_button_events = {transition(false, {-50.0f, -50.0f})};
        route(outside);
        expectRangeInert();
    }

    TEST_F(RmlViewportInputRoutingTest, ExternalCapturePreservesEarlierViewportTextDismissClassification) {
        const glm::vec2 bare_point{350.0f, 250.0f};
        const glm::vec2 outside_point{450.0f, 250.0f};
        auto* const bare_element = context_->GetElementAtPoint({bare_point.x, bare_point.y});
        ASSERT_NE(bare_element, nullptr);
        recorder_.listen(bare_element);
        text_->AddEventListener("blur", &recorder_);
        ASSERT_TRUE(text_->Focus());
        route(inputAt(bare_point));
        ASSERT_FALSE(hoveredInteractive());
        ASSERT_EQ(context_->GetFocusElement(), text_);
        recorder_.clear();

        // A completed viewport click retains its event-time ownership even
        // when later motion leaves the overlay for a capturing GUI panel.
        FrameInputBuffer buffer;
        buffer.beginFrame();
        const auto press_point = bare_point + origin_;
        pressAt(buffer, SDL_BUTTON_LEFT, press_point.x, press_point.y, false, 1001, 1);
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, press_point.x, press_point.y, 1002, 1));
        const auto latest_point = outside_point + origin_;
        buffer.processEvent(mouseMotionEvent(latest_point.x, latest_point.y));
        settleLiveCursor(buffer, latest_point.x, latest_point.y);
        const auto input = gui::buildPanelInputFromSDL(buffer);
        ASSERT_EQ(input.mouse_button_events.size(), 2u);
        ASSERT_FALSE(input.mouse_button_events[0].gui_owned);
        ASSERT_FALSE(input.mouse_button_events[1].gui_owned);
        ASSERT_FALSE(input.mouse_down[0]);
        gui::guiFocusState().want_capture_mouse = true;
        route(input);

        // Blur listeners commit synchronously, before GuiManager consumes this
        // classification and decides whether that original press may focus.
        EXPECT_NE(context_->GetFocusElement(), text_);
        EXPECT_EQ(recorder_.countOf("text:blur"), 1) << recorder_.joined();
        EXPECT_FALSE(owns());
        EXPECT_EQ(recorder_.countOf(bare_element->GetId() + ":mousedown"), 0) << recorder_.joined();
        ASSERT_EQ(overlay_->leftPressClassifications().size(), 1u);
        const auto& classification = overlay_->leftPressClassifications()[0];
        EXPECT_FALSE(classification.on_interactive_control);
        EXPECT_TRUE(classification.blurred_text_input);
        EXPECT_TRUE(gui::overlayPressMayFocusPanel({
            .left_pressed = true,
            .overlay_wants_input = overlay_->wantsInput(),
            .pressed_interactive_control = classification.on_interactive_control,
            .press_blurred_text_input = classification.blurred_text_input,
            .press_inside_viewport = gui::pointInsideViewport(press_point, origin_, {400.0f, 300.0f}),
            .press_gui_owned = input.mouse_button_events[0].gui_owned,
        }));
    }

    TEST_F(RmlViewportInputRoutingTest, BlockedTextInputNeitherBlursNorReceivesKeys) {
        ASSERT_TRUE(text_->Focus());
        auto input = inputAt(button_point_);
        input.mouse_button_events = {transition(true, button_point_), transition(false, button_point_)};
        input.keys_pressed = {SDL_SCANCODE_BACKSPACE};
        input.text_inputs = {"x"};
        input.text_codepoints = {U'x'};
        const auto value = text_->GetValue();
        route(input, {.pending_modal = true});
        EXPECT_EQ(context_->GetFocusElement(), text_);
        EXPECT_EQ(text_->GetValue(), value);
        EXPECT_EQ(recorder_.countOf("text:keydown"), 0);
        EXPECT_TRUE(overlay_->leftPressClassifications().empty());
        input.mouse_button_events.clear();
        input.keys_pressed.clear();
        route(input);
        EXPECT_NE(text_->GetValue(), value);
    }

} // namespace lfs::vis

namespace lfs::vis {
    TEST_F(RmlViewportInputRoutingTest, RefusedSameButtonDownMustCancelRangeWithoutHoverEdits) {
        arm(range_point_);
        gui::guiFocusState().reset();
        ASSERT_TRUE(bar_->IsPseudoClassSet("active"));
        const auto before = range_->GetValue();
        const glm::vec2 bare{350.0f, 270.0f};
        auto input = inputAt(bare);
        input.mouse_button_events = {transition(true, bare), transition(false, bare)};
        input.mouse_button_events[0].gui_owned = false;
        input.mouse_button_events[1].gui_owned = false;
        route(input);
        EXPECT_FALSE(overlay_->wantsInput());
        EXPECT_FALSE(gui::guiFocusState().want_capture_mouse);
        ASSERT_EQ(overlay_->leftPressClassifications().size(), 1);
        EXPECT_FALSE(overlay_->leftPressClassifications()[0].on_interactive_control);
        EXPECT_FALSE(owns());
        EXPECT_FALSE(bar_->IsPseudoClassSet("active"));
        recorder_.clear();
        route(inputAt({range_point_.x + 60.0f, range_point_.y}));
        EXPECT_EQ(range_->GetValue(), before);
        EXPECT_EQ(recorder_.countOf("range:change"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("range-bar:dragstart"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("range-bar:drag"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("range-bar:click"), 0) << recorder_.joined();
    }

    TEST_F(RmlViewportInputRoutingTest, StartedRangeReplacementCancelsBeforeFrameEndMove) {
        arm(range_point_);
        auto moved = inputAt(range_point_ + glm::vec2(20.0f, 0.0f));
        moved.mouse_down[0] = true;
        route(moved);
        ASSERT_GT(recorder_.countOf("range:change"), 0) << recorder_.joined();
        const auto before = range_->GetValue();
        recorder_.clear();
        const glm::vec2 bare{350.0f, 270.0f};
        auto replacement = inputAt({5.0f, 270.0f});
        replacement.mouse_button_events = {transition(true, bare), transition(false, bare)};
        for (auto& event : replacement.mouse_button_events)
            event.gui_owned = false;
        route(replacement);
        context_->ProcessMouseButtonCancel(0, 0); // Repeated cancellation must stay inert.
        EXPECT_EQ(range_->GetValue(), before);
        EXPECT_EQ(recorder_.countOf("range:change"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("range-bar:dragend"), 1) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("range-bar:dragdrop"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("range-bar:click"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("range-bar:mouseup"), 0) << recorder_.joined();
        expectRangeInert();

        context_->Update();
        const auto offset = bar_->GetAbsoluteOffset();
        const glm::vec2 current{offset.x + bar_->GetOffsetWidth() / 2, offset.y + bar_->GetOffsetHeight() / 2};
        arm(current);
        moved = inputAt(current - glm::vec2(20.0f, 0.0f));
        moved.mouse_down[0] = true;
        route(moved);
        EXPECT_NE(range_->GetValue(), before);
        moved.mouse_down[0] = false;
        moved.mouse_button_events = {transition(false, current - glm::vec2(20.0f, 0.0f))};
        route(moved);
        expectRangeInert();
    }

    TEST_F(RmlViewportInputRoutingTest, AcceptedReplacementClicksOnlyNewPress) {
        arm(button_point_);
        auto replacement = inputAt(button_point_);
        replacement.mouse_button_events = {transition(true, button_point_), transition(false, button_point_)};
        route(replacement);
        EXPECT_FALSE(owns());
        EXPECT_FALSE(button_->IsPseudoClassSet("active"));
        EXPECT_EQ(recorder_.countOf("action:mousedown"), 1) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("action:mouseup"), 1) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("action:click"), 1) << recorder_.joined();
    }

    TEST_F(RmlViewportInputRoutingTest, CancellationMustRefreshToolbarCaptureBeforeRefusingReplacement) {
        arm(range_point_);
        auto drag = inputAt({range_point_.x + 20.0f, range_point_.y});
        drag.mouse_down[0] = true;
        route(drag);
        ASSERT_TRUE(bar_->IsPseudoClassSet("active"));
        seedToolbarCaptureForCancel();
        ASSERT_TRUE(toolbarCaptureActive());
        PointerEventRecorder counts;
        auto* body = range_->GetParentNode();
        for (auto id : {Rml::EventId::Mousedown, Rml::EventId::Mouseup, Rml::EventId::Click})
            body->AddEventListener(id, &counts);
        gui::guiFocusState().reset();
        const glm::vec2 bare{350, 270};
        auto replacement = inputAt(bare);
        replacement.mouse_button_events = {transition(true, bare), transition(false, bare)};
        for (auto& event : replacement.mouse_button_events)
            event.gui_owned = false;
        route(replacement);
        // The real toolbar Dragend handler ran during cancellation and released capture.
        ASSERT_FALSE(toolbarCaptureActive());
        EXPECT_TRUE(counts.log().empty()) << counts.joined();
        EXPECT_FALSE(overlay_->wantsInput());
        EXPECT_FALSE(gui::guiFocusState().want_capture_mouse);
        EXPECT_FALSE(owns());
        for (auto id : {Rml::EventId::Mousedown, Rml::EventId::Mouseup, Rml::EventId::Click})
            body->RemoveEventListener(id, &counts);
    }
} // namespace lfs::vis

namespace lfs::vis {

    class DepthWindowReturnTest : public DepthWindowPanelsInteractionTest {
    protected:
        void parkPair() {
            if (!rendering_manager_->isIndependentSplitViewActive()) {
                enterIndependentDual();
            }
            rendering_manager_->setDepthWindowSync(false);
            rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left,
                                                       makeWindow(1.125f, 77.25f, .4137f, .5279f, .1317f, -.2813f));
            rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right,
                                                       makeWindow(2.25f, 88.5f, .3719f, .6173f, -.2391f, .1837f));
            rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);
            pair_ = rendering_manager_->depthWindowSnapshot().panels;
            lfs::core::events::cmd::ToggleGTComparison{}.emit();
            lfs::core::events::cmd::ToggleGTComparison{}.emit();
            ASSERT_EQ(rendering_manager_->getSplitViewMode(), SplitViewMode::Disabled);
            op::undoHistory().clear();
        }

        void startHandle(glm::dvec2& press, const glm::vec4 bounds = {0.f, 0.f, 400.f, 200.f},
                         const glm::dvec2 grab_offset = {}, const bool center = false) {
            const auto window = projectionDepthWindow(*rendering_manager_);
            const auto panel = rendering_manager_->resolveViewerPanel(viewer_->getViewport(),
                                                                      {bounds.x, bounds.y}, {bounds.z, bounds.w},
                                                                      glm::vec2{bounds.x + bounds.z * .25f, bounds.y + bounds.w * .5f});
            ASSERT_TRUE(panel.has_value());
            const op::DepthWindowPanelMapping mapping{panel->panel, panel->x, panel->y,
                                                      panel->width, panel->height, panel->render_width, panel->render_height};
            const auto handles = op::depthWindowHandleGeometry(op::depthWindowScreenRect(mapping,
                                                                                         window.scale_x, window.scale_y, window.offset_x, window.offset_y));
            press = glm::dvec2(center ? handles.center : handles.corners[2]) + grab_offset;
            auto props = depthDragProps(press.x, press.y);
            props.set("viewport_x", bounds.x);
            props.set("viewport_y", bounds.y);
            props.set("viewport_width", bounds.z);
            props.set("viewport_height", bounds.w);
            ASSERT_EQ(op::operators().invoke(op::BuiltinOp::DepthWindowDrag, &props).status,
                      OperatorResult::RUNNING_MODAL);
            ASSERT_EQ(op::depthWindowOverlayState().hovered_handle,
                      center ? op::DepthWindowHandle::Center : op::DepthWindowHandle::BottomRight);
        }

        std::array<DepthWindowState, 2> pair_{};
    };

    TEST_F(DepthWindowReturnTest, ReturnedHandleRestoresExactStateAndRetainedPair) {
        for (const auto offset : {glm::dvec2{}, glm::dvec2{.04, .02}}) {
            ASSERT_NO_FATAL_FAILURE(parkPair());
            const auto before = rendering_manager_->depthWindowSnapshot();
            const auto publication = app_store().depth_window_draw_generation.get();
            glm::dvec2 press;
            ASSERT_NO_FATAL_FAILURE(startHandle(press, {137.25f, 83.5f, 1234.5f, 703.25f}, offset));
            ASSERT_EQ(op::operators().dispatchModalEvent(mouse_move(press.x + 17., press.y + 13.)),
                      OperatorResult::RUNNING_MODAL);
            ASSERT_NE(rendering_manager_->depthWindowSnapshot().panels, before.panels);
            if (offset.x == 0.) {
                ASSERT_EQ(op::operators().dispatchModalEvent(mouse_move(press.x, press.y)),
                          OperatorResult::RUNNING_MODAL);
            }
            EXPECT_EQ(op::operators().dispatchModalEvent(mouse_release(press.x, press.y)),
                      OperatorResult::FINISHED);
            const auto after = rendering_manager_->depthWindowSnapshot();
            EXPECT_EQ(after.panels, before.panels);
            EXPECT_EQ(after.projection, before.projection);
            EXPECT_EQ(op::undoHistory().undoCount(), 0u);
            EXPECT_EQ(app_store().depth_window_draw_generation.get(), publication);
            EXPECT_FALSE(op::operators().hasModalOperator());
            EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
            enterIndependentDual();
            EXPECT_EQ(rendering_manager_->depthWindowSnapshot().panels, pair_);
        }
    }

    TEST_F(DepthWindowReturnTest, RadiusUsesOriginalDoubleScreenCoordinates) {
        for (const double displacement : {.00075, .00125}) {
            ASSERT_NO_FATAL_FAILURE(parkPair());
            const auto before = rendering_manager_->depthWindowSnapshot();
            glm::dvec2 press;
            ASSERT_NO_FATAL_FAILURE(startHandle(press, {8192.f, 0.f, 400.f, 200.f}, {.00001, 0.}));
            ASSERT_EQ(op::operators().dispatchModalEvent(mouse_move(press.x + 17., press.y + 13.)),
                      OperatorResult::RUNNING_MODAL);
            EXPECT_EQ(op::operators().dispatchModalEvent(mouse_release(press.x + displacement, press.y)),
                      OperatorResult::FINISHED);
            const auto after = rendering_manager_->depthWindowSnapshot();
            if (displacement < .001) {
                EXPECT_EQ(after.panels, before.panels);
                EXPECT_EQ(after.projection, before.projection);
                EXPECT_EQ(op::undoHistory().undoCount(), 0u);
            } else {
                EXPECT_NE(after.panels, before.panels);
                EXPECT_EQ(op::undoHistory().undoCount(), 1u);
            }
        }
    }

    TEST_F(DepthWindowReturnTest, FloatRepresentableReleaseOutsideRadiusRemainsEdit) {
        ASSERT_NO_FATAL_FAILURE(parkPair());
        const auto before = rendering_manager_->depthWindowSnapshot();
        glm::dvec2 press;
        ASSERT_NO_FATAL_FAILURE(startHandle(press, {0.f, 0.f, 300.75f, 200.f}));
        float release_x = static_cast<float>(press.x);
        while (static_cast<double>(release_x) - press.x <= .001) {
            release_x = std::nextafter(release_x, std::numeric_limits<float>::infinity());
        }
        ASSERT_GT(static_cast<double>(release_x) - press.x, .001);
        ASSERT_EQ(op::operators().dispatchModalEvent(mouse_move(press.x + 17., press.y + 13.)),
                  OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(op::operators().dispatchModalEvent(mouse_release(release_x, press.y)), OperatorResult::FINISHED);
        EXPECT_NE(rendering_manager_->depthWindowSnapshot().panels, before.panels);
        EXPECT_EQ(op::undoHistory().undoCount(), 1u);
    }

    TEST_F(DepthWindowReturnTest, SmallCenterMoveKeepsExactUndoRedo) {
        ASSERT_NO_FATAL_FAILURE(parkPair());
        const auto before = rendering_manager_->depthWindowSnapshot();
        glm::dvec2 press;
        ASSERT_NO_FATAL_FAILURE(startHandle(press, {0.f, 0.f, 400.f, 200.f}, {}, true));
        ASSERT_EQ(op::operators().dispatchModalEvent(mouse_release(press.x + .01, press.y)),
                  OperatorResult::FINISHED);
        const auto after = rendering_manager_->depthWindowSnapshot();
        EXPECT_NE(after.panels, before.panels);
        ASSERT_EQ(op::undoHistory().undoCount(), 1u);
        ASSERT_TRUE(op::undoHistory().undo().success);
        EXPECT_EQ(rendering_manager_->depthWindowSnapshot().panels, before.panels);
        ASSERT_TRUE(op::undoHistory().redo().success);
        EXPECT_EQ(rendering_manager_->depthWindowSnapshot().panels, after.panels);
    }

    TEST_F(DepthWindowReturnTest, SameEpochSupersessionCancelsAndRestoresOtherOwnedSlot) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(true);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left,
                                                   makeWindow(1.125f, 77.25f, .4137f, .5279f, .1317f, -.2813f));
        op::undoHistory().clear();
        const auto before = rendering_manager_->depthWindowSnapshot();
        const auto publication = app_store().depth_window_draw_generation.get();
        glm::dvec2 press;
        ASSERT_NO_FATAL_FAILURE(startHandle(press));
        ASSERT_EQ(op::operators().dispatchModalEvent(mouse_move(press.x + 17., press.y + 13.)),
                  OperatorResult::RUNNING_MODAL);
        // A same-epoch sync-history restore preserves pins while allowing a one-slot setter.
        auto unsynced = rendering_manager_->depthWindowSnapshot();
        unsynced.sync = false;
        ASSERT_TRUE(rendering_manager_->restoreDepthWindowSnapshotIfEpoch(unsynced, before.mode_epoch, true));
        const auto preview = rendering_manager_->depthWindowSnapshot();
        auto transition = rendering_manager_->acquireDepthWindowTransitionLock();
        auto release = std::async(std::launch::async, [&] {
            return op::operators().dispatchModalEvent(mouse_release(press.x, press.y));
        });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        bool final_preview_seen = false;
        while (std::chrono::steady_clock::now() < deadline) {
            if (rendering_manager_->depthWindowSnapshot().panels != preview.panels) {
                final_preview_seen = true;
                break;
            }
            std::this_thread::yield();
        }
        // Avoid fatal assertions until the lock is released and the worker is joined.
        EXPECT_TRUE(final_preview_seen);
        const auto newer = makeWindow(3.25f, 91.75f, .731f, .621f, .191f, -.331f);
        EXPECT_TRUE(rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, newer));
        transition.unlock();
        EXPECT_EQ(release.get(), OperatorResult::CANCELLED);
        const auto after = rendering_manager_->depthWindowSnapshot();
        EXPECT_EQ(after.panels[0], newer);
        EXPECT_EQ(after.panels[1], before.panels[1]);
        EXPECT_EQ(after.projection, newer);
        EXPECT_EQ(after.mode_epoch, before.mode_epoch);
        EXPECT_EQ(op::undoHistory().undoCount(), 0u);
        EXPECT_EQ(app_store().depth_window_draw_generation.get(), publication);
        EXPECT_FALSE(op::operators().hasModalOperator());
        EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
    }

} // namespace lfs::vis

namespace lfs::vis {
    class CancelPseudoElement final : public Rml::Element {
    public:
        explicit CancelPseudoElement(const Rml::String& tag) : Rml::Element(tag) {}
        std::function<void()> on_deactivate;
        void OnPseudoClassChange(const Rml::String& pseudo, const bool activate) override {
            if (pseudo == "active" && !activate && on_deactivate) {
                auto callback = std::move(on_deactivate);
                callback();
            }
        }
    };

    enum class CancelCallbackAction { Move,
                                      Up,
                                      FreshDown,
                                      Remove };
    class RmlCancelCallbackTest : public RmlPointerReplayTest,
                                  public ::testing::WithParamInterface<CancelCallbackAction> {
    protected:
        void SetUp() override {
            RmlPointerReplayTest::SetUp();
            Rml::Factory::RegisterElementInstancer("cancel-probe", &instancer_);
            auto element = Rml::Factory::InstanceElement(document_, "cancel-probe", "cancel-probe", {});
            probe_ = static_cast<CancelPseudoElement*>(element.get());
            probe_->SetId("cancel-probe");
            for (const auto& [name, value] : {std::pair{"position", "absolute"}, {"left", "200px"}, {"top", "150px"}, {"width", "100px"}, {"height", "100px"}, {"drag", "drag"}})
                probe_->SetProperty(name, value);
            document_->AppendChild(std::move(element));
            context_->Update();
            ASSERT_EQ(context_->GetElementAtPoint({250.f, 200.f}), probe_);
            recorder_.listen(probe_);
            press(250, 200);
            ASSERT_TRUE(probe_->IsPseudoClassSet("active"));
            recorder_.clear();
        }
        void TearDown() override {
            if (probe_)
                probe_->on_deactivate = {};
            RmlPointerReplayTest::TearDown();
        }
        static inline Rml::ElementInstancerGeneric<CancelPseudoElement> instancer_;
        CancelPseudoElement* probe_ = nullptr;
    };

    TEST_P(RmlCancelCallbackTest, DetachesOldPressBeforeActiveCallback) {
        bool invoked = false;
        probe_->on_deactivate = [&] {
            invoked = true;
            switch (GetParam()) {
            case CancelCallbackAction::Move: context_->ProcessMouseMove(260, 200, 0); break;
            case CancelCallbackAction::Up: context_->ProcessMouseButtonUp(0, 0); break;
            case CancelCallbackAction::FreshDown:
                press(250, 200);
                break;
            case CancelCallbackAction::Remove: {
                auto* old = probe_;
                probe_ = nullptr;
                old->GetParentNode()->RemoveChild(old).reset();
                break;
            }
            }
        };
        context_->ProcessMouseButtonCancel(0, 0);
        EXPECT_TRUE(invoked);
        EXPECT_EQ(recorder_.countOf("cancel-probe:dragstart"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("cancel-probe:drag"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("cancel-probe:click"), 0) << recorder_.joined();
        if (GetParam() == CancelCallbackAction::FreshDown) {
            ASSERT_NE(probe_, nullptr);
            EXPECT_EQ(context_->GetHoverElement(), probe_);
            EXPECT_TRUE(probe_->IsPseudoClassSet("hover"));
            EXPECT_TRUE(probe_->IsPseudoClassSet("active"));
            context_->ProcessMouseMove(260, 200, 0);
            context_->ProcessMouseButtonUp(0, 0);
            EXPECT_EQ(recorder_.countOf("cancel-probe:dragstart"), 1) << recorder_.joined();
            EXPECT_EQ(recorder_.countOf("cancel-probe:click"), 1) << recorder_.joined();
        } else {
            press(50, 50);
            context_->ProcessMouseButtonUp(0, 0);
            EXPECT_EQ(recorder_.countOf("alpha:click"), 1) << recorder_.joined();
        }
    }
    INSTANTIATE_TEST_SUITE_P(ReentrantInput, RmlCancelCallbackTest,
                             ::testing::Values(CancelCallbackAction::Move, CancelCallbackAction::Up,
                                               CancelCallbackAction::FreshDown, CancelCallbackAction::Remove));

    class CancelTestClock final : public Rml::SystemInterface {
    public:
        double GetElapsedTime() override { return now; }
        double now = 0;
    };
    class RmlDisabledCancelTest : public RmlViewportInputRoutingTest,
                                  public ::testing::WithParamInterface<bool> {
    protected:
        static void SetUpTestSuite() {
            Rml::SetSystemInterface(&clock_);
            RmlPointerReplayTest::SetUpTestSuite();
        }
        static void TearDownTestSuite() {
            RmlPointerReplayTest::TearDownTestSuite();
            Rml::SetSystemInterface(nullptr);
        }

        static inline CancelTestClock clock_;
    };
    TEST_P(RmlDisabledCancelTest, DisabledSliderTerminatesAndRemainsUsable) {
        const bool arrow_press = GetParam();
        auto* arrow = findElementByTag(range_, "sliderarrowinc");
        ASSERT_NE(arrow, nullptr);
        if (arrow_press) {
            arrow->SetId("cancel-arrow");
            recorder_.listen(arrow);
            arrow->SetProperty("width", "20px");
            arrow->SetProperty("height", "24px");
            context_->Update();
        }
        const auto offset = arrow->GetAbsoluteOffset();
        const int x = arrow_press ? int(offset.x + arrow->GetOffsetWidth() / 2) : int(range_point_.x + 65);
        const int y = arrow_press ? int(offset.y + arrow->GetOffsetHeight() / 2) : int(range_point_.y);
        press(x, y);
        if (!arrow_press) {
            context_->ProcessMouseMove(x + 20, y, 0);
            ASSERT_TRUE(bar_->IsPseudoClassSet("active"));
        }
        const auto before = range_->GetValue();
        ASSERT_NE(before, "50.000000");
        range_->SetDisabled(true);
        recorder_.clear();
        context_->ProcessMouseButtonCancel(0, 0);
        clock_.now += 1.0;
        context_->Update();
        EXPECT_EQ(range_->GetValue(), before);
        EXPECT_FALSE(bar_->IsPseudoClassSet("active"));
        EXPECT_EQ(recorder_.countOf("range:change"), 0) << recorder_.joined();
        const char* target = arrow_press ? "cancel-arrow" : "range-bar";
        EXPECT_EQ(recorder_.countOf(Rml::String(target) + ":click"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf(Rml::String(target) + ":mouseup"), 0) << recorder_.joined();
        range_->SetDisabled(false);
        clock_.now += 1.0;
        context_->Update();
        EXPECT_EQ(range_->GetValue(), before);
        if (arrow_press) {
            press(x, y);
            context_->ProcessMouseButtonUp(0, 0);
        } else {
            const auto at = bar_->GetAbsoluteOffset();
            const int bx = int(at.x + bar_->GetOffsetWidth() / 2);
            press(bx, y);
            context_->ProcessMouseMove(bx - 20, y, 0);
            context_->ProcessMouseButtonUp(0, 0);
        }
        EXPECT_NE(range_->GetValue(), before);
        EXPECT_FALSE(bar_->IsPseudoClassSet("active"));
    }
    INSTANTIATE_TEST_SUITE_P(SliderParts, RmlDisabledCancelTest, ::testing::Bool());

    TEST_F(RmlPointerReplayTest, CancellationBalancesDragTargetWithoutDrop) {
        dragger_->SetProperty("drag", "drag-drop");
        context_->Update();
        PointerEventRecorder listener;
        for (const auto id : {Rml::EventId::Dragover, Rml::EventId::Dragout, Rml::EventId::Dragdrop})
            beta_->AddEventListener(id, &listener);
        const auto start_drag = [&] {
            press(50, 200);
            context_->ProcessMouseMove(250, 50, 0);
        };
        start_drag();
        ASSERT_FALSE(listener.log().empty());
        ASSERT_EQ(listener.log().back(), "beta:dragover");
        context_->ProcessMouseButtonCancel(0, 0);
        EXPECT_EQ(listener.log(), (std::vector<Rml::String>{"beta:dragover", "beta:dragout"}));
        start_drag();
        ASSERT_FALSE(listener.log().empty());
        ASSERT_EQ(listener.log().back(), "beta:dragover");
        context_->ProcessMouseButtonUp(0, 0);
        EXPECT_EQ(listener.log(), (std::vector<Rml::String>{"beta:dragover", "beta:dragout",
                                                            "beta:dragover", "beta:dragdrop", "beta:dragout"}));
        for (const auto id : {Rml::EventId::Dragover, Rml::EventId::Dragout, Rml::EventId::Dragdrop})
            beta_->RemoveEventListener(id, &listener);
    }
} // namespace lfs::vis

namespace lfs::vis {
    class SliderCancelCallback final : public Rml::EventListener {
    public:
        std::function<void(Rml::Event&)> fn;
        void ProcessEvent(Rml::Event& event) override { fn(event); }
    };
    class RmlSharedSliderCancelTest : public RmlViewportInputRoutingTest,
                                      public ::testing::WithParamInterface<bool> {};

    TEST_P(RmlSharedSliderCancelTest, CancellationPreservesFreshInteractionOnOtherSliderPart) {
        const bool fresh_track = GetParam();
        const auto phase = fresh_track ? Rml::EventId::Dragend : Rml::EventId::Mouseout;
        const int old_x = int(range_point_.x) + (fresh_track ? 0 : 65);
        const int y = int(range_point_.y);
        press(old_x, y);
        context_->ProcessMouseMove(old_x + 20, y, 0);
        ASSERT_TRUE(bar_->IsPseudoClassSet("active"));
        SliderCancelCallback callback;
        int callbacks = 0, releases = 0, clicks = 0, drops = 0, new_x = 0;
        callback.fn = [&](Rml::Event& event) {
            releases += event.GetId() == Rml::EventId::Mouseup;
            clicks += event.GetId() == Rml::EventId::Click;
            drops += event.GetId() == Rml::EventId::Dragdrop;
            if (event.GetId() != phase || callbacks)
                return;
            ++callbacks;
            context_->Update();
            const auto at = bar_->GetAbsoluteOffset();
            new_x = fresh_track ? 290 : int(at.x + bar_->GetOffsetWidth() / 2);
            context_->ProcessMouseMove(new_x, y, 0);
            EXPECT_EQ(context_->GetHoverElement() == bar_, !fresh_track);
            context_->ProcessMouseButtonDown(0, 0);
            context_->ProcessMouseMove(new_x - 10, y, 0);
            EXPECT_TRUE(bar_->IsPseudoClassSet("active"));
        };
        for (const auto id : {phase, Rml::EventId::Mouseup, Rml::EventId::Click, Rml::EventId::Dragdrop})
            range_->AddEventListener(id, &callback);
        context_->ProcessMouseButtonCancel(0, 0);
        EXPECT_EQ(callbacks, 1);
        EXPECT_EQ(releases, 0);
        EXPECT_EQ(clicks, 0);
        EXPECT_EQ(drops, 0);
        EXPECT_TRUE(bar_->IsPseudoClassSet("active"));
        const auto before = range_->GetValue();
        context_->ProcessMouseMove(new_x - 30, y, 0);
        EXPECT_NE(range_->GetValue(), before);
        for (const auto id : {phase, Rml::EventId::Mouseup, Rml::EventId::Click, Rml::EventId::Dragdrop})
            range_->RemoveEventListener(id, &callback);
        context_->ProcessMouseButtonUp(0, 0);
        EXPECT_FALSE(bar_->IsPseudoClassSet("active"));
    }
    INSTANTIATE_TEST_SUITE_P(CleanupCallbacks, RmlSharedSliderCancelTest, ::testing::Bool());

    // Attach a headless context to the real progress overlay and use its action
    // listener, so these tests exercise the same click delivery as GuiManager.
    class RmlProgressInputRoutingTest : public RmlPointerReplayTest {
    protected:
        void SetUp() override {
            RmlPointerReplayTest::SetUp();
            document_->Hide();
            gui::guiFocusState().reset();
            controls_ = context_->LoadDocumentFromMemory(
                "<rml><head><style>body{width:400px;height:300px;}"
                "#progress-action{position:absolute;left:20px;top:100px;width:100px;height:40px;}"
                "</style></head><body><button id='progress-action'/></body></rml>");
            ASSERT_NE(controls_, nullptr);
            for (const char* id : {"progress-backdrop", "progress-dialog", "progress-title", "progress-path",
                                   "progress-row", "progress-value", "progress-text", "progress-stage",
                                   "progress-detail", "progress-error", "progress-actions"}) {
                auto element = controls_->CreateElement("div");
                element->SetId(id);
                controls_->AppendChild(std::move(element));
            }
            controls_->Show();
            context_->Update();
            ASSERT_EQ(context_->GetElementAtPoint({50, 120})->GetId(), "progress-action");
            overlay_ = std::make_unique<gui::RmlProgressOverlay>(
                &manager_, [this] { ++dismissals_; }, [this] { ++cancels_; });
            overlay_->rml_manager_ = nullptr;
            overlay_->rml_context_ = context_;
            overlay_->document_ = controls_;
            overlay_->cacheElements();
            ASSERT_TRUE(overlay_->elements_cached_);
            showVideo();
        }

        void TearDown() override {
            if (overlay_) {
                overlay_->cancelPointerInput();
                context_->UnloadDocument(controls_);
                context_->Update();
                overlay_->rml_context_ = nullptr;
                overlay_->document_ = nullptr;
                overlay_.reset();
            }
            app_store().video_export_overlay_state.set({});
            app_store().import_overlay_state.set({});
            gui::guiFocusState().reset();
            RmlPointerReplayTest::TearDown();
        }

        void showVideo() {
            AppStore::VideoExportOverlayState video;
            video.active = true;
            app_store().video_export_overlay_state.set(video);
            overlay_->presentation_ = gui::makeProgressOverlayPresentation({}, video);
        }

        void showFailedImport() {
            app_store().video_export_overlay_state.set({});
            AppStore::ImportOverlayState state;
            state.show_completion = true;
            state.error = "Invalid cameras";
            app_store().import_overlay_state.set(state);
            overlay_->presentation_ = gui::makeProgressOverlayPresentation(state, {});
        }

        gui::PanelInputState inputAt(const glm::vec2 point) const {
            gui::PanelInputState input;
            input.screen_x = 30.0f;
            input.screen_y = 40.0f;
            input.mouse_x = point.x + input.screen_x;
            input.mouse_y = point.y + input.screen_y;
            return input;
        }

        FrameMouseButtonEvent transition(const bool down, const glm::vec2 point) const {
            return {.button = 0, .down = down, .x = point.x + 30.0f, .y = point.y + 40.0f};
        }

        gui::PanelInputState clickAt(const glm::vec2 point, const glm::vec2 final_point) const {
            auto input = inputAt(final_point);
            input.mouse_clicked[0] = true;
            input.mouse_released[0] = true;
            input.mouse_button_events = {transition(true, point), transition(false, point)};
            return input;
        }

        bool ownsPress() const { return overlay_->pointer_down_delivered_[0]; }
        const glm::vec2 button_point_{50.0f, 120.0f};
        const glm::vec2 outside_point_{350.0f, 250.0f};
        gui::RmlUIManager manager_;
        std::unique_ptr<gui::RmlProgressOverlay> overlay_;
        Rml::ElementDocument* controls_ = nullptr;
        int cancels_ = 0;
        int dismissals_ = 0;
    };

    TEST_F(RmlProgressInputRoutingTest, RecordedCancelClickSurvivesMotionAway) {
        overlay_->processInput(clickAt(button_point_, outside_point_));
        EXPECT_EQ(cancels_, 1);
        EXPECT_FALSE(ownsPress());
        EXPECT_NE(context_->GetHoverElement()->GetId(), "progress-action");
    }

    TEST_F(RmlProgressInputRoutingTest, RecordedFailedImportDismissalSurvivesMotionAway) {
        showFailedImport();
        overlay_->processInput(clickAt(button_point_, outside_point_));
        EXPECT_EQ(dismissals_, 1);
        EXPECT_EQ(cancels_, 0);
    }

    TEST_F(RmlProgressInputRoutingTest, MotionOntoButtonCannotMoveAnOutsideClick) {
        overlay_->processInput(clickAt(outside_point_, button_point_));
        EXPECT_EQ(cancels_, 0);
        EXPECT_EQ(context_->GetHoverElement()->GetId(), "progress-action");
    }

    TEST_F(RmlProgressInputRoutingTest, CanonicalEventsWorkWithoutAggregateFlagsAndDoNotDuplicateClicks) {
        auto input = clickAt(button_point_, button_point_);
        input.mouse_clicked[0] = false;
        input.mouse_released[0] = false;
        overlay_->processInput(input);
        EXPECT_EQ(cancels_, 1);
        input.mouse_clicked[0] = true;
        input.mouse_released[0] = true;
        input.mouse_button_events.push_back(transition(true, button_point_));
        input.mouse_button_events.push_back(transition(false, button_point_));
        overlay_->processInput(input);
        EXPECT_EQ(cancels_, 3);
        EXPECT_FALSE(ownsPress());
    }

    TEST_F(RmlProgressInputRoutingTest, AggregateOnlyClickRemainsSupported) {
        auto input = clickAt(button_point_, button_point_);
        input.mouse_button_events.clear();
        overlay_->processInput(input);
        EXPECT_EQ(cancels_, 1);
        EXPECT_FALSE(ownsPress());
    }

    TEST_F(RmlProgressInputRoutingTest, PressSpansFramesAndOutsideReleaseDisarms) {
        auto input = inputAt(button_point_);
        input.mouse_button_events = {transition(true, button_point_)};
        overlay_->processInput(input);
        ASSERT_TRUE(ownsPress());
        input.mouse_button_events = {transition(false, button_point_)};
        overlay_->processInput(input);
        EXPECT_EQ(cancels_, 1);
        input.mouse_button_events = {transition(true, button_point_)};
        overlay_->processInput(input);
        input = inputAt({-100.0f, -100.0f});
        input.mouse_button_events = {transition(false, {-100.0f, -100.0f})};
        overlay_->processInput(input);
        EXPECT_EQ(cancels_, 1);
        EXPECT_FALSE(ownsPress());
        overlay_->processInput(clickAt(button_point_, button_point_));
        EXPECT_EQ(cancels_, 2);
    }

    TEST_F(RmlProgressInputRoutingTest, HiddenOrBlockedOverlayCannotRetainAnArmedAction) {
        for (const bool blocked : {false, true}) {
            auto input = inputAt(button_point_);
            input.mouse_button_events = {transition(true, button_point_)};
            overlay_->processInput(input);
            ASSERT_TRUE(ownsPress());
            if (!blocked)
                app_store().video_export_overlay_state.set({});
            overlay_->processInput(inputAt(button_point_), blocked);
            EXPECT_FALSE(ownsPress());
            showVideo();
            input.mouse_button_events = {transition(false, button_point_)};
            overlay_->processInput(input);
            EXPECT_EQ(cancels_, 0);
        }
        overlay_->processInput(clickAt(button_point_, button_point_));
        EXPECT_EQ(cancels_, 1);
    }

} // namespace lfs::vis
