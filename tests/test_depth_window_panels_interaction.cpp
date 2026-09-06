/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "gui/panel_input_utils.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rml_viewport_overlay.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_pointer_dispatch.hpp"
#include "gui/string_keys.hpp"
#include "input/frame_input_buffer.hpp"
#include "input/key_codes.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "operator/ops/depth_window_ops.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/rendering_types.hpp"
#include "selection/selection_service.hpp"
#include "tools/selection_tool.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/rendering/depth_window_state.hpp"
#include "visualizer_impl.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/EventListener.h>
#include <RmlUi/Core/RenderInterface.h>
#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>
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
        // The label is localized, so this asserts the KEY the expired branch
        // resolves -- an English literal would only pass in an English build,
        // and this test process loads no locale at all. Comparing against the
        // live-branch key as well is what proves the expired branch was taken.
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

    // DISCRIMINATING TEST for the reported defect "sync ON always conforms the
    // LEFT panel to the RIGHT one, whichever panel is focused". It exercises the
    // copy in BOTH focus directions through the real service focus call and the
    // real manager setter, with distinct windows in every field. If the copy
    // itself were inverted (other-over-focused) or the panel-id -> slot mapping
    // were swapped, the RIGHT-focused half would keep Left's window and this
    // would fail; a PASS proves the setter is focused-wins and puts the reported
    // bug strictly UPSTREAM of it, in whatever decides focus at toggle time.
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

    // REAL registry replacement, CROSS panel. The registry invokes the incoming
    // modal BEFORE destroying the outgoing one, so A tears down while B is
    // already live. Ownership is PER PANEL: B taking Right is not a takeover of
    // Left, so A still restores its own panel instead of leaving an
    // uncommitted preview baked there, and B goes on to commit normally.
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

        // The replacement itself: a second depth drag invoked on the RIGHT
        // panel, through the registry, which destroys A as it installs B. The
        // press is well clear of the Right window's handles, so B is a fresh
        // draw and has not latched yet.
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

    // REAL registry replacement, SAME panel. B does take Left over, so A's
    // teardown correctly refuses to restore (upstream's incoming-baseline
    // contract). What must survive the handoff is the PANEL'S BACKUP: leaving
    // independent-dual mid-B folds PRE-DRAG-A state, not A's or B's preview.
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

    // The refusal observed through a LIVE modal. A project restore is a
    // lifetime discontinuity that bumps the epoch WITHOUT being a split-mode
    // transition, so it does not destroy the drag: the release below reaches a
    // still-live drag whose epoch has moved, and must leave no trace.
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

            // The second drag's bracket, opened through the same manager API the
            // operator uses, with its own preview write on the OTHER panel. The
            // preview goes through the DRAG lane (applyDepthWindowForPanelIfEpoch),
            // not the public setter: a public-setter write is a legitimate
            // non-drag write and would supersede the very backup this case is
            // about.
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

        // A global -> GT -> global round trip is TWO boundaries, and each leg is
        // pinned separately: the enter bump against the global epoch, and the
        // leave bump against the GT epoch (comparing the far side to the near
        // side alone would pass even if the leave never bumped).
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

    // The release/destructor epoch race is closed by construction: every drag
    // write goes through the epoch-guarded apply, which refuses under a stale
    // epoch and changes nothing. This pins that guarantee at the manager level;
    // the cross-thread interleaving itself is not deterministically
    // constructible without a dedicated concurrency seam.
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

    // Regression, verbatim sequence. Sync ON, shared clean window O: A
    // previews P on Left, a fresh-draw B is invoked on Right through the REAL
    // registry (destroying A), and independent-dual is left while B is still
    // subthreshold. A's teardown restore must write ONLY Left (a restore is
    // safety machinery, never a fan-out), and B must have inherited O - not
    // A's fanned-out preview - as Right's backup. Both slots end at O.
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

    // Regression, verbatim sequence: the unsynced cross-panel scenario
    // continued through B's commit, then sync ON with Right focused, then focus
    // Left and leave. The sync copy is a legitimate NON-drag write, so it
    // releases Left's stale pre-A backup and the collapse folds the COPY.
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

    // Regression, verbatim sequence: the sync gate keys on the OWNERSHIP
    // bracket (invoke..destruction), so a sync toggle during a SUBTHRESHOLD
    // press is ignored too - a drag's before_ capture can never straddle a sync
    // change, and undo therefore leaves panels and the sync flag consistent.
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

    // Regression, verbatim sequence (sync OFF): A previews on Left, B is
    // invoked on Right through the REAL registry (destroying A), B commits, undo
    // B. B's undo baseline is captured at its FIRST SLOT WRITE - by then A's
    // teardown restore has already put Left back - so undoing B can never
    // resurrect A's abandoned preview on Left.
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
        // Residue R, now RESOLVED - by the BASELINE SOURCE, not by A's
        // teardown. B's begin takes over both slots (A therefore restores
        // nothing), but the manager's recorded backup for each of them is
        // still the pre-A value, and B's baseline is composed from the backups
        // of the slots it owns. So the undo returns both slots to the pre-A
        // value - A's abandoned preview is gone from the other slot too.
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), clean);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), clean);
    }

    // The CANCEL half, and it is a DIFFERENT question from the undo half.
    // Same take-over sequence, but B cancels instead of committing: a cancel
    // puts back what was ON SCREEN when B started - the replaced drag's
    // abandoned preview included - because the teardown target is sourced from
    // the LIVE snapshot, not from the manager's backups. This is the upstream
    // semantics DepthWindowDragLifecycleTest pins.
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

        // Regression: B's end released the slots but left the pre-A backup O
        // behind. C's claim finds them UNOWNED and therefore REFRESHES the
        // backups from the live value, so C records the VISIBLE pre-C state
        // (A's abandoned preview P) as its baseline, and undoing C
        // returns the slots to P - never to the original clean O.
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

    // STOP C's resolution, direct: updateSettings itself carries a
    // split_view_mode FLIP. It must take the transition mutex, run the same
    // transition the event sites run (epoch bump + backup fold + collapse) and
    // leave a pre-flip drag's later writes refused by the epoch guard.
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

    // STOP C's other half: an EQUAL-mode settings write must never touch the
    // transition mutex. Toggling independent-dual with the depth tool enabled
    // and a LATCHED drag runs the chain that re-enters updateSettings while
    // this same thread holds the transition lock (cancel hook -> finishLatch ->
    // applySelectionFilterSettings -> updateSettings). Reaching the assertions
    // at all IS the no-deadlock proof; the state below must also be coherent.
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

    // Regression, in its constructible form: sync ON, drag A previewing,
    // a same-epoch depth-window entry atop the history, a mid-modal
    // history.undo(), A continues, then independent-dual is left with the OTHER
    // panel focused. A's begin pinned BOTH slots' backups (its writes fan out),
    // so the mid-drag undo's idle-backup release must not drop the other slot's
    // backup - otherwise the focused-other collapse publishes A's preview.
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

    // Regression, first half: a split-mode change across a boundary that means
    // NOTHING to depth windows (Disabled <-> PLYComparison) must be a COMPLETE
    // no-op for depth-window state. A live drag keeps its backup, its pins and
    // its epoch, goes on drawing, and cancels back to the pre-drag window O.
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

    // Regression, second half (the identity hole). After the same no-op flip a
    // same-panel replacement B must record the PRE-DRAG-A window O as the
    // panel's backup - not A's abandoned preview - and A's end, which runs as
    // the registry destroys it AFTER B is already live, must not strip the pins
    // B just took: B's own cancel still restores.
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

    // Regression: a newer LEGITIMATE non-drag write beats a teardown restore.
    // Independent-dual, sync ON, A previewing into both slots; a direct
    // settings write replaces the depth window on both slots and supersedes
    // A's pre-drag backup and pins. Cancelling A must restore NEITHER slot -
    // the newest intent stands.
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

    // Regression, the FULL sequence: the superseded drag must not be able to
    // keep drawing. Sync ON, A previews; a legitimate settings write installs W
    // on both slots and takes A's ownership of them away; A's NEXT move is
    // therefore refused instead of overwriting W with a second preview, and its
    // cancel restores nothing. W stands on both slots.
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

        // THE POST-WRITE MOVE. The modal is still live and still receives the
        // event, but the slot write behind it is refused on ownership where the
        // old {epoch, aggregate pins} gate let P2 through. The refusal ends the
        // drag on the spot (the operator's existing epoch_lost_ path), and that
        // cancel restores nothing, so W stands.
        ASSERT_EQ(lfs::vis::op::operators().dispatchModalEvent(mouse_move(150.0, 130.0)),
                  OperatorResult::CANCELLED);
        EXPECT_FALSE(lfs::vis::op::operators().hasModalOperator());

        lfs::vis::op::operators().cancelModalOperator();

        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), newest);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), newest);
        EXPECT_EQ(projectionDepthWindow(*rendering_manager_), newest);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
    }

    // Regression: a superseded drag's teardown must not treat a LATER drag's
    // slot ownership as its own licence to restore. Sync ON, A previews on
    // Left; a settings write installs W and takes A's ownership; A moves
    // (refused); the real registry then invokes B on the RIGHT panel, which
    // takes ownership of both slots and records W as its baseline, and destroys
    // A. A's teardown must leave both slots - and B's baseline - untouched.
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
        // A is still LIVE here: the sequence deliberately does NOT send the
        // post-write move, because that move's refusal would end A before the
        // replacement arrives (see
        // SupersededDragCannotWriteAfterNewerSettingsWrite). This is the
        // "refused or not yet" half of the sequence - A reaches its teardown
        // with its ownership already gone.
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

    // Regression: a legitimate write to the OTHER panel must not strand this
    // drag's ownership of its own slot. Independent + unsynced, A live on Left;
    // a focused write lands on Right only. A cancels (Left folds back to
    // pre-A), and A's end releases Left, so that backup is idle again. A later
    // sync-ON copy therefore drops it, and leaving independent collapses the
    // SYNC COPY - never the pre-A Left window.
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

    // The NATIVE producer of the collapse channel, end to end. The Python
    // toolbar consumes (source, generation) to decide whether its cached
    // per-panel Size references survived a transition, and a stub-fed Python
    // test cannot tell a working producer from a removed one. This drives a
    // REAL leave and pins all three producer properties: the initial value, the
    // bump on collapse (with the pre-transition focus, not the post-transition
    // reset-to-Left), and the absence of a bump on anything else.
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

        // The real leave: Right is focused, so Right's window is what folds.
        // (That write above back-routed the projection scale into the focused
        // slot, which is exactly the documented no-panel routing - so read
        // Right's window HERE rather than assuming it is still right_pre.)
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

    // The OTHER two producers of the reference-lineage channel, plus the writes
    // that must NOT stamp it. A Python consumer reconciles on every generation
    // advance whatever endpoint it observes, so an unstamped sync copy or
    // project restore leaves it seeding from a window that no longer exists,
    // and a spurious stamp throws away a perfectly valid Size reference. Only a
    // native test can tell a working producer from a missing one.
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

    // The FOURTH stamp site: a SYNC undo/redo restore. It writes two absolute
    // window snapshots at once, so it invalidates every slot-derived reference
    // exactly like the producers above - and it can run with no mode edge, no
    // focus edge, and (across an undo+redo pair inside one poll) no sync-flag
    // edge either, so the stamp is the only witness a poller gets. A DRAG undo
    // goes through the SAME manager restore and must stamp NOTHING.
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

    // The press rule that decides whether an overlay-consumed press may move the
    // focused split panel. The press itself needs a window, an RmlUi context and
    // a GUI frame, none of which exist headlessly, so the DECISION is pinned here
    // as a truth table over the helper the focus block calls.
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

        // ...but ONLY inside the viewport. The overlay's bounds are stretched
        // over the left dock so the toolbars can hang above it, so a press on a
        // dock control also reaches this rule after blurring the field. With no
        // field focused that press focuses nothing; dismissing a field must not
        // be what earns it the power to. (Editing Right and clicking a dock
        // control must not focus Left.)
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

        // The GUI's own event-time verdict vetoes the press whatever rectangle
        // it fell in. The left-dock resize strip is the case that forced this:
        // its hitbox is CENTRED on the dock's right edge, which is exactly where
        // the viewport begins, so its right half passes containment. With no
        // field focused that press starts a dock resize (the GUI owns it) and
        // focuses nothing.
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

        // The exact left boundary IS inside the rectangle -- containment is
        // half-open, and the pixel belongs to the viewport, not the dock. What
        // it is NOT is proof that a press there may focus a panel: the dock's
        // resize strip is centred on this very boundary, so containment and
        // ownership disagree here and OWNERSHIP is the one that decides.
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

    // Event coalescing: SDL delivers a whole frame's events at once, so the
    // motion queued BEHIND a press is already applied to mouse_x/mouse_y by the
    // time the GUI frame classifies that press. Every press decision must come
    // from the down coordinates instead.
    namespace {
        // `timestamp` and `clicks` default to 0 -- the value a defaulted
        // FrameMouseButtonEvent already carries -- so a test that needs to prove
        // a field SURVIVED capture and copy must pass a distinct non-default
        // value; an assertion against the default would pass on a field that was
        // silently dropped.
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

        // The REAL path: synthetic SDL events through FrameInputBuffer's own
        // intake, then the production copy into PanelInputState. Hand-seeding
        // PanelInputState instead would leave the capture and the copy -- the
        // two places this can actually break -- untested.
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

        // A press and its release can coalesce into one frame, and so can two
        // whole presses. NEITHER is collapsed: the canonical stream keeps both
        // DOWNs, at their own coordinates, in arrival order, and the focus rule
        // is applied to each of them in turn (see
        // EachPressAppliesItsOwnFocusDecisionInOrder). `lastPress` reads the
        // last of them without disturbing either.
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

    // The other half of the same capture: WHO owned the press, recorded at the
    // BUTTON_DOWN event rather than re-derived from a rectangle in the GUI
    // frame -- and recorded for EVERY DOWN, not merely the frame's first for
    // that button (R10).
    TEST(DepthWindowOverlayPressFocusTest, PressOwnershipIsRecordedAtTheButtonDown) {
        using lfs::vis::gui::buildPanelInputFromSDL;

        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        buffer.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 320.0f, 300.0f));
        // What WindowManager does at the event, from GuiManager::pressBelongsToGui.
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

    // Cross-button ordering, retained from the per-button record it replaces:
    // the canonical vector IS the arrival order, for every transition, and it is
    // never reordered into button-index order. Replaying a real right-then-left
    // frame as left-then-right invents cursor motion after the left DOWN and can
    // manufacture a drag (RmlUi Context.cpp:625 / :1277) -- see
    // RmlPointerReplayTest for that consequence exercised against a context.
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

    // EACH LEFT PRESS APPLIES ITS OWN FOCUS DECISION, IN ARRIVAL ORDER, and a
    // refusal applies NOTHING -- it never undoes what an earlier press in the
    // same frame already did. That is the whole difference between the rule
    // being applied per press and a frame being collapsed to one governing
    // press: collapse to the LAST press and an eligible viewport press followed
    // by a press on toolbar chrome moves no focus at all.
    //
    // GuiManager's loop (gui_manager.cpp, the overlayPressMayFocusPanel block)
    // is that fold: it `continue`s on a refusal and re-points focus on an
    // admission, so the last ADMITTED press wins. A whole GUI frame cannot be
    // driven in this process, so what is executable here is the fold itself over
    // the real rule, in both orderings.
    TEST(DepthWindowOverlayPressFocusTest, EachPressAppliesItsOwnFocusDecisionInOrder) {
        using lfs::vis::gui::OverlayPressFocusInputs;
        using lfs::vis::gui::overlayPressMayFocusPanel;

        struct FramePress {
            glm::vec2 point;
            OverlayPressFocusInputs rule;
        };

        // GuiManager's loop, in the only two lines of it this can hold: an
        // admitted press re-points focus, a refused press does nothing at all.
        const auto focusAfterFrame =
            [](const std::vector<FramePress>& presses) -> std::optional<glm::vec2> {
            std::optional<glm::vec2> focused;
            for (const auto& press : presses) {
                if (!overlayPressMayFocusPanel(press.rule))
                    continue;
                focused = press.point;
            }
            return focused;
        };

        constexpr OverlayPressFocusInputs kEligible{
            .left_pressed = true,
            .overlay_wants_input = true,
            .pressed_interactive_control = false,
            .press_blurred_text_input = false,
            .press_inside_viewport = true,
            .press_gui_owned = false,
        };
        auto rejected = kEligible;
        rejected.pressed_interactive_control = true; // a press on toolbar chrome
        ASSERT_TRUE(overlayPressMayFocusPanel(kEligible));
        ASSERT_FALSE(overlayPressMayFocusPanel(rejected));

        const glm::vec2 viewport_point{700.0f, 300.0f};
        const glm::vec2 chrome_point{360.0f, 60.0f};

        // ELIGIBLE -> REJECTED. The chrome press must not erase the viewport
        // press's focus move; the frame's outcome is the viewport press.
        const auto eligible_first = focusAfterFrame({
            {viewport_point, kEligible},
            {chrome_point, rejected},
        });
        ASSERT_TRUE(eligible_first.has_value())
            << "a later chrome press erased an earlier viewport press's focus";
        EXPECT_EQ(*eligible_first, viewport_point);

        // REJECTED -> ELIGIBLE. The chrome press contributes nothing and the
        // viewport press behind it still focuses -- so the order is genuinely
        // being walked, not a hard-coded "first" or "last" event.
        const auto rejected_first = focusAfterFrame({
            {chrome_point, rejected},
            {viewport_point, kEligible},
        });
        ASSERT_TRUE(rejected_first.has_value());
        EXPECT_EQ(*rejected_first, viewport_point);

        // Two ELIGIBLE presses: the last one wins, because it is the one whose
        // focus move survives the frame.
        const glm::vec2 second_viewport_point{660.0f, 310.0f};
        const auto both_eligible = focusAfterFrame({
            {viewport_point, kEligible},
            {second_viewport_point, kEligible},
        });
        ASSERT_TRUE(both_eligible.has_value());
        EXPECT_EQ(*both_eligible, second_viewport_point);

        // Two REFUSED presses move nothing.
        EXPECT_FALSE(focusAfterFrame({{chrome_point, rejected},
                                      {viewport_point, rejected}})
                         .has_value());
    }

    // ------------------------------------------------------------------
    // Escape contract, exercised against REAL RmlUi elements.
    //
    // The viewport overlay's Escape branch advertised select handling that its
    // own outer gate could never reach (the gate published text focus, and
    // wantsTextInput excludes a normal select). A substring test over the source
    // passes on exactly that kind of dead code, so this pins the decision the
    // hosts share instead: which focused element Escape cancels, and when an IME
    // composition takes Escape back.
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
    // RmlUi pointer DELIVERY, driven against a real headless Rml::Context.
    //
    // The host buffers a whole SDL frame, so a press and the motion queued
    // behind it arrive together and the ORDER in which they reach RmlUi is a
    // decision, not an accident. These tests exercise the production replay
    // (rml_pointer_dispatch.hpp, called by RmlViewportOverlay::processInput --
    // there is no test-only API on the overlay) and they observe what RmlUi
    // itself dispatches, not what the host claims it sent.
    //
    // What must hold, per R10:
    //   (1) the canonical stream is never coalesced, reordered or truncated --
    //       every transition is delivered at ITS OWN point, in arrival order,
    //       with its own button identity;
    //   (2) an event the host does not own is SKIPPED, and skipping it changes
    //       neither the order nor the identity of the events that are
    //       delivered;
    //   (3) an ownership verdict never leaks from one press to another -- not
    //       across buttons, and not from an earlier same-button press.
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

        // Records "<element id>:<event>" in dispatch order, and -- in a parallel
        // vector -- the RmlUi `button` parameter each event was delivered with
        // (-1 for the events that carry none, e.g. mousemove/mouseover). Without
        // the button identity a host that replayed BOTH presses as button 0
        // would still satisfy every ordering and count assertion below, so the
        // identity is recorded and asserted, not inferred from the loop index.
        class PointerEventRecorder final : public Rml::EventListener {
        public:
            void ProcessEvent(Rml::Event& event) override {
                auto* const target = event.GetTargetElement();
                log_.push_back((target ? target->GetId() : Rml::String("<null>")) + ":" +
                               event.GetType());
                buttons_.push_back(event.GetParameter<int>("button", -1));
            }

            void listen(Rml::Element* element) {
                // `dragstart` is here because it is the observable consequence of
                // replaying two buttons in the wrong order: RmlUi arms `drag` on
                // the primary DOWN (Context.cpp:625) and fires Dragstart on the
                // next cursor move (Context.cpp:1277), so a move manufactured
                // AFTER the left DOWN starts a drag the user never began.
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

        static inline NullRenderInterface* render_interface_ = nullptr;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::Element* alpha_ = nullptr;
        Rml::Element* beta_ = nullptr;
        Rml::Element* dragger_ = nullptr;
        PointerEventRecorder recorder_;
    };

    namespace {

        // The overlay's own ownership predicate, as the production call site
        // supplies it: "would this host take an event that landed here?" These
        // boxes stand in for interactive overlay chrome; `nullptr` (an event
        // outside the context) is never owned.
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

    // Finding (1), restated for upstream's replay: EACH event is delivered at
    // ITS OWN point. The frame-end cursor move happens before the replay (the
    // host's hover pass), so what must be true here is that no event is
    // delivered at another event's coordinates -- alpha's DOWN lands on alpha
    // even though the frame ended over beta.
    TEST_F(RmlPointerReplayTest, EachEventIsDeliveredAtItsOwnPoint) {
        // The host's own hover move to the frame-end cursor, as processInput
        // makes it before the replay.
        context_->ProcessMouseMove(250, 50, 0); // beta
        recorder_.clear();

        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 50.0f, /*gui_owned=*/false); // alpha

        const bool replayed = gui::rml_input::replayButtonEvents(
            *context_, state.down_delivered, buffer.mouse_button_events, glm::vec2(0.0f, 0.0f),
            glm::vec2(400.0f, 300.0f), 0, /*capture_active=*/false,
            ownsOverlayBox({alpha_, beta_}));
        EXPECT_TRUE(replayed);

        const auto trace = recorder_.joined();
        const int down = recorder_.indexOf("alpha:mousedown");
        ASSERT_GE(down, 0) << "the DOWN did not land on the pressed element: " << trace;
        EXPECT_EQ(recorder_.buttonAt(down), 0) << trace;
        EXPECT_EQ(context_->GetHoverElement(), alpha_) << trace;
    }

    // Finding (2): an event the host does not own is skipped, and the events it
    // does own keep their order and identity. `dragger` is not in the owned set,
    // so its press must never reach RmlUi -- and alpha's, which arrived after
    // it, must still arrive, in that position, as button 0.
    TEST_F(RmlPointerReplayTest, UnownedEventsAreSkippedWithoutDisturbingTheRest) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_RIGHT, 50.0f, 200.0f, /*gui_owned=*/false); // dragger
        pressAt(buffer, SDL_BUTTON_LEFT, 250.0f, 50.0f, /*gui_owned=*/false);  // beta

        const bool replayed = gui::rml_input::replayButtonEvents(
            *context_, state.down_delivered, buffer.mouse_button_events, glm::vec2(0.0f, 0.0f),
            glm::vec2(400.0f, 300.0f), 0, /*capture_active=*/false,
            ownsOverlayBox({alpha_, beta_}));
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
        EXPECT_FALSE(gui::rml_input::replayButtonEvents(
            *context_, outside_state.down_delivered, outside.mouse_button_events,
            glm::vec2(0.0f, 0.0f),
            glm::vec2(400.0f, 300.0f), 0, /*capture_active=*/false,
            ownsOverlayBox({alpha_, beta_})));
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

        EXPECT_TRUE(gui::rml_input::replayButtonEvents(
            *context_, state.down_delivered, buffer.mouse_button_events, glm::vec2(0.0f, 0.0f),
            glm::vec2(400.0f, 300.0f), 0, /*capture_active=*/true,
            ownsOverlayBox({alpha_, beta_})));
        EXPECT_EQ(recorder_.countOfWithButton("dragger:mousedown", 0), 1)
            << recorder_.joined();
    }

    // Cross-button ordering, retained: a real right-then-left frame must be
    // replayed right-then-left. Replayed the other way round, the move onto
    // `dragger` would fall AFTER the left DOWN that armed `drag`
    // (Context.cpp:625) and RmlUi would fire Dragstart (Context.cpp:1277) from a
    // pointer motion the user never made.
    TEST_F(RmlPointerReplayTest, RightBeforeLeftStartsNoDragOnTheLeftTarget) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_RIGHT, 250.0f, 50.0f, /*gui_owned=*/false); // beta
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 200.0f, /*gui_owned=*/false);  // dragger

        EXPECT_TRUE(gui::rml_input::replayButtonEvents(
            *context_, state.down_delivered, buffer.mouse_button_events, glm::vec2(0.0f, 0.0f),
            glm::vec2(400.0f, 300.0f), 0, /*capture_active=*/false,
            ownsOverlayBox({alpha_, beta_, dragger_})));

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

        EXPECT_TRUE(gui::rml_input::replayButtonEvents(
            *context_, state.down_delivered, buffer.mouse_button_events, glm::vec2(0.0f, 0.0f),
            glm::vec2(400.0f, 300.0f), 0, /*capture_active=*/false,
            ownsOverlayBox({alpha_, beta_})));

        const auto trace = recorder_.joined();
        const int left_down = recorder_.indexOf("alpha:mousedown");
        const int right_down = recorder_.indexOf("beta:mousedown");
        ASSERT_GE(left_down, 0) << trace;
        ASSERT_GE(right_down, 0) << trace;
        EXPECT_LT(left_down, right_down) << trace;
        EXPECT_EQ(recorder_.buttonAt(left_down), 0) << trace;
        EXPECT_EQ(recorder_.buttonAt(right_down), 1) << trace;
    }

    // R10, matching UP: "a matching UP follows the owner of its corresponding
    // DOWN, including releases outside the original control". The press starts
    // on alpha, which this host owns, and is released over `dragger`, which it
    // does not. Judged by the hit test alone the release would be skipped and
    // RmlUi would stay pressed for good (Context.cpp:721-745 is the only thing
    // that disarms `active` and `drag`).
    TEST_F(RmlPointerReplayTest, OwnedReleaseSurvivesTheCursorLeavingTheControl) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 50.0f, /*gui_owned=*/false); // alpha
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 200.0f));   // dragger

        EXPECT_TRUE(gui::rml_input::replayButtonEvents(
            *context_, state.down_delivered, buffer.mouse_button_events, glm::vec2(0.0f, 0.0f),
            glm::vec2(400.0f, 300.0f), 0, /*capture_active=*/false,
            ownsOverlayBox({alpha_, beta_})));

        const auto trace = recorder_.joined();
        EXPECT_EQ(recorder_.countOfWithButton("alpha:mousedown", 0), 1) << trace;
        EXPECT_EQ(recorder_.countOfWithButton("dragger:mouseup", 0), 1)
            << "the release outside the pressed control was withheld: " << trace;
        EXPECT_FALSE(state.down_delivered[0])
            << "the press lifecycle stayed open after its release";
    }

    // ...and the converse: a release this host never opened is REJECTED, so it
    // cannot borrow delivery from another button or from a press that was
    // refused. Without this an UP that merely happens to land on our chrome
    // reaches RmlUi as the end of a press RmlUi never saw begin.
    TEST_F(RmlPointerReplayTest, ReleaseIsDeliveredOnlyForAMatchingDown) {
        // (a) a bare release, no press behind it at all.
        ReplayState bare_state;
        lfs::vis::FrameInputBuffer bare;
        bare.beginFrame();
        bare.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 50.0f)); // over alpha
        EXPECT_FALSE(gui::rml_input::replayButtonEvents(
            *context_, bare_state.down_delivered, bare.mouse_button_events,
            glm::vec2(0.0f, 0.0f), glm::vec2(400.0f, 300.0f), 0, /*capture_active=*/false,
            ownsOverlayBox({alpha_, beta_})));
        EXPECT_EQ(recorder_.countOf("alpha:mouseup"), 0) << recorder_.joined();

        // (b) a press this host REFUSED, released over chrome it owns. The
        // refusal is what makes the release unmatched.
        recorder_.clear();
        ReplayState refused_state;
        lfs::vis::FrameInputBuffer refused;
        refused.beginFrame();
        pressAt(refused, SDL_BUTTON_LEFT, 50.0f, 200.0f, /*gui_owned=*/false); // dragger
        refused.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 50.0f));     // alpha
        EXPECT_FALSE(gui::rml_input::replayButtonEvents(
            *context_, refused_state.down_delivered, refused.mouse_button_events,
            glm::vec2(0.0f, 0.0f), glm::vec2(400.0f, 300.0f), 0, /*capture_active=*/false,
            ownsOverlayBox({alpha_, beta_})));
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
        EXPECT_FALSE(gui::rml_input::replayButtonEvents(
            *context_, stale_state.down_delivered, stale.mouse_button_events,
            glm::vec2(0.0f, 0.0f), glm::vec2(400.0f, 300.0f), 0, /*capture_active=*/false,
            ownsOverlayBox({alpha_, beta_})));
        EXPECT_TRUE(recorder_.log().empty()) << recorder_.joined();
        EXPECT_FALSE(stale_state.down_delivered[0]);
    }

    // A press this host does not own, followed by the cursor coming to rest over
    // chrome it DOES own. The canonical vector is not empty, so nothing may be
    // delivered: the aggregate button bits would put a press the user made on
    // `dragger` onto `alpha`, at the frame-end cursor, with neither its own
    // coordinates nor its own place in the order. R10 lets a consumer skip what
    // it does not own; it does not let it invent what it does. (The overlay's
    // aggregate fallback is gated on the vector being EMPTY for this reason --
    // rml_viewport_overlay.cpp.)
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

        EXPECT_FALSE(gui::rml_input::replayButtonEvents(
            *context_, state.down_delivered, input.mouse_button_events, glm::vec2(0.0f, 0.0f),
            glm::vec2(400.0f, 300.0f), 0, /*capture_active=*/false,
            ownsOverlayBox({alpha_, beta_})));
        EXPECT_EQ(recorder_.countOf("alpha:mousedown"), 0)
            << "a press on something else was manufactured on chrome: "
            << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("alpha:mouseup"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("alpha:click"), 0) << recorder_.joined();
        EXPECT_FALSE(state.down_delivered[0]);
    }

    // CROSS-BUTTON, MIXED OWNERS -- both directions, because a host that keyed
    // delivery off "some press this frame was owned" would pass one and fail the
    // other. The rejected button must take no ownership beside the accepted one,
    // and the accepted one must arrive whole.
    TEST_F(RmlPointerReplayTest, RejectedLeftLeavesTheAcceptedRightUntouched) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 200.0f, /*gui_owned=*/false);  // dragger
        pressAt(buffer, SDL_BUTTON_RIGHT, 250.0f, 50.0f, /*gui_owned=*/false); // beta
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 200.0f));
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_RIGHT, 250.0f, 50.0f));

        EXPECT_TRUE(gui::rml_input::replayButtonEvents(
            *context_, state.down_delivered, buffer.mouse_button_events, glm::vec2(0.0f, 0.0f),
            glm::vec2(400.0f, 300.0f), 0, /*capture_active=*/false,
            ownsOverlayBox({alpha_, beta_})));

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

        EXPECT_TRUE(gui::rml_input::replayButtonEvents(
            *context_, state.down_delivered, buffer.mouse_button_events, glm::vec2(0.0f, 0.0f),
            glm::vec2(400.0f, 300.0f), 0, /*capture_active=*/false,
            ownsOverlayBox({alpha_, beta_})));

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

        EXPECT_TRUE(gui::rml_input::replayButtonEvents(
            *context_, state.down_delivered, buffer.mouse_button_events, glm::vec2(0.0f, 0.0f),
            glm::vec2(400.0f, 300.0f), 0, /*capture_active=*/false,
            ownsOverlayBox({alpha_, beta_})));
        EXPECT_TRUE(state.down_delivered[0]);
        EXPECT_TRUE(state.down_delivered[1]);

        // Released in a LATER frame, each outside the box it started on.
        recorder_.clear();
        buffer.beginFrame();
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_RIGHT, 50.0f, 200.0f)); // dragger
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 200.0f));  // dragger
        EXPECT_TRUE(gui::rml_input::replayButtonEvents(
            *context_, state.down_delivered, buffer.mouse_button_events, glm::vec2(0.0f, 0.0f),
            glm::vec2(400.0f, 300.0f), 0, /*capture_active=*/false,
            ownsOverlayBox({alpha_, beta_})));

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
    // R10 INTEGRATION: DOWN -> UP -> DOWN -> UP on ONE button, at two distinct
    // points, ONE of which this host actually owns and one of which it actually
    // does not -- run in BOTH orderings (owned press first, then unowned first).
    //
    // This is the case the replaced design could not express. "First DOWN of
    // the frame wins" recorded one press, one point and one verdict per button,
    // so the second press was delivered at the first's coordinates and under
    // the first's owner; and a coalesced same-button double click was collapsed
    // onto the frame-end cursor.
    //
    // THE TWO VERDICTS ARE NOT ASSIGNED BY HAND. Each press's `gui_owned` is
    // taken from the SAME admission question the replay asks of that point --
    // "would this host take an event that landed here?" -- against the real
    // element under it. So the two presses differ in ownership because they
    // landed on genuinely different things, and a change that made the two
    // points equivalent would fail the fixture assertions rather than quietly
    // turn the case into two identical presses.
    //
    // Asserted here, end to end through the production intake, copy and replay:
    //   - the COMPLETE canonical stream: four events, in arrival order, each
    //     with its own DOWN/UP identity and its own x, y, timestamp and click
    //     count, all distinct and none of them the defaulted value;
    //   - the COMPLETE GUI-delivery sequence RmlUi observes, compared whole
    //     rather than filtered, so an extra event cannot hide inside it;
    //   - NO ownership leakage and NO state leakage: neither press takes the
    //     other's verdict, each UP takes its own DOWN's, the unowned press is
    //     delivered NOWHERE, and the host's press lifecycle ends closed.
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

        // The delivered sequence, with the host's own cursor moves dropped:
        // `mousemove` is the replay's "put the pointer where this event
        // happened" step, not a delivered transition. NOTHING else is removed,
        // so an extra press, an extra release, a stray click or a manufactured
        // drag all show up in the comparison.
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

        // Two presses of the SAME button, at two clearly different points,
        // inside ONE buffered frame -- and their releases, so the frame carries
        // two complete press lifecycles. Distinct, non-default timestamps and
        // click counts, so a field that was dropped cannot read as "correct".
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
        EXPECT_TRUE(gui::rml_input::replayButtonEvents(
            *context_, state.down_delivered, input.mouse_button_events, glm::vec2(0.0f, 0.0f),
            glm::vec2(400.0f, 300.0f), 0, /*capture_active=*/false, owns));

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

} // namespace lfs::vis
