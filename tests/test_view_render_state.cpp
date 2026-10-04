/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/services.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/screen_host.hpp"
#include "input/input_controller.hpp"
#include "input/key_codes.hpp"
#include "ipc/view_context.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "python/python_runtime.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "screen/screen_service.hpp"
#include "visualizer/app_store.hpp"
#include <array>
#include <gtest/gtest.h>

namespace lfs::vis {
    namespace {
        class ViewRenderStateTest : public ::testing::Test {
        protected:
            screen::ScreenService source;
            RenderingManager rendering{source};
            ViewId first = source.activeView();
            ViewId second = source.screen().split(screen::AreaId{first}, screen::SplitAxis::Columns, 0.5f).value;

            void clearDirty() {
                rendering.viewState(first).dirty_mask_.store(0);
                rendering.viewState(second).dirty_mask_.store(0);
            }
            void renderEmpty(ViewId id) {
                auto& camera = source.view3D(id)->camera;
                camera.frameBufferSize = {640, 480};
                const auto settings = rendering.settingsForView(id);
                rendering.renderFrame({.view = id, .viewport = camera, .settings = settings, .logical_screen_size = {640, 480}});
            }
        };

        TEST_F(ViewRenderStateTest, QuadLayoutRequestsTheRetainedViewsNewExtentOnce) {
            gui::ScreenHost host(source);
            host.layout({0.0f, 0.0f, 1920.0f, 1080.0f}, 1.0f);
            const auto request_sizes = [&] {
                std::vector<ViewId> visible;
                for (const auto id : source.screen().views())
                    visible.push_back(id.value);
                rendering.retainVisibleViews(visible);
                for (const auto id : visible) {
                    const auto rect = host.viewContent(screen::AreaId{id});
                    ASSERT_TRUE(rect.has_value());
                    rendering.requestViewportResize(id, {static_cast<int>(rect->w), static_cast<int>(rect->h)});
                }
            };
            auto& ledger = rendering.frameDemandLedger();
            request_sizes();
            static_cast<void>(ledger.plan(FrameClock::now()));
            clearDirty();
            const auto old_extent = rendering.viewState(first).requested_viewport_size_;

            // The command's frame can render before ScreenHost commits the layout.
            ASSERT_TRUE(source.screen().toggleQuadView(screen::AreaId{first}, 500.0f));
            rendering.requestViewportResize(first, old_extent);
            static_cast<void>(ledger.plan(FrameClock::now()));
            host.refreshLayout();
            request_sizes();
            const auto plan = ledger.plan(FrameClock::now());
            EXPECT_NE(rendering.viewState(first).requested_viewport_size_, old_extent);
            EXPECT_NE(plan.render_views & rendering.viewMask(first), 0u);
            EXPECT_EQ(plan.render_views & rendering.viewMask(second), 0u);
            EXPECT_TRUE(plan.reasons.test(static_cast<size_t>(FrameReason::ViewportResize)));

            ledger.countViewRendered(plan.render_views, plan);
            const auto counts = ledger.snapshot().views_rendered;
            for (int i = 0; i < 4; ++i) {
                request_sizes();
                EXPECT_TRUE(ledger.plan(FrameClock::now()).empty());
            }
            EXPECT_EQ(ledger.snapshot().views_rendered, counts);
        }

        TEST_F(ViewRenderStateTest, PreviewTopologyCacheInvalidationIsNotAnUnrequestedSceneChange) {
            SceneManager scene;
            auto& camera = source.view3D(first)->camera;
            camera.frameBufferSize = {640, 480};
            const auto before = rendering.viewInputFingerprint(camera, &scene, first);
            const auto cache_generation = scene.getScene().renderGeneration();
            // Refinement publishes topology for the next scheduled training preview.
            scene.getScene().syncTrainingModelTopology(128);
            ASSERT_NE(scene.getScene().renderGeneration(), cache_generation);
            EXPECT_EQ(rendering.viewInputFingerprint(camera, &scene, first), before);
        }

        TEST_F(ViewRenderStateTest, FingerprintStillDetectsCameraAndPublishedSceneAndSelectionChanges) {
            auto& camera = source.view3D(first)->camera;
            camera.frameBufferSize = {640, 480};
            const auto fingerprint = [&] { return rendering.viewInputFingerprint(camera, nullptr, first); };
            auto before = fingerprint();
            camera.setViewMatrix(glm::mat3(1.0f), {1.0f, 2.0f, 3.0f});
            EXPECT_NE(fingerprint(), before);
            before = fingerprint();
            ++camera.windowSize.x;
            EXPECT_NE(fingerprint(), before);
            before = fingerprint();
            const auto scene_generation = app_store().scene_generation.get();
            app_store().scene_generation.set(scene_generation + 1);
            EXPECT_NE(fingerprint(), before);
            app_store().scene_generation.set(scene_generation);
            before = fingerprint();
            const auto selection_generation = app_store().selection_generation.get();
            app_store().selection_generation.set(selection_generation + 1);
            EXPECT_NE(fingerprint(), before);
            app_store().selection_generation.set(selection_generation);
        }

        TEST_F(ViewRenderStateTest, CameraAndViewSettingsDirtyOnlyTheirView) {
            clearDirty();
            rendering.markCameraPoseChanged(second);
            EXPECT_EQ(rendering.viewState(first).dirty_mask_.load(), 0u);
            EXPECT_EQ(rendering.viewState(second).dirty_mask_.load(), DirtyFlag::CAMERA);
            clearDirty();
            source.screen().setActiveView(screen::AreaId{first});
            auto settings = rendering.getSettings();
            settings.point_cloud_mode = true;
            rendering.updateSettings(settings);
            EXPECT_NE(rendering.viewState(first).dirty_mask_.load(), 0u);
            EXPECT_EQ(rendering.viewState(second).dirty_mask_.load(), 0u);
            EXPECT_FALSE(rendering.settingsForView(second).point_cloud_mode);
        }

        TEST_F(ViewRenderStateTest, SceneMutationsAndSceneSettingsDirtyEveryView) {
            clearDirty();
            rendering.markDirty(DirtyFlag::SPLATS, lfs::vis::FrameReason::SceneChange);
            EXPECT_EQ(rendering.viewState(first).dirty_mask_.load(), DirtyFlag::SPLATS);
            EXPECT_EQ(rendering.viewState(second).dirty_mask_.load(), DirtyFlag::SPLATS);
            clearDirty();
            auto settings = rendering.getSettings();
            settings.background_color = {0.25f, 0.5f, 0.75f};
            rendering.updateSettings(settings);
            EXPECT_NE(rendering.viewState(first).dirty_mask_.load(), 0u);
            EXPECT_NE(rendering.viewState(second).dirty_mask_.load(), 0u);
        }

        TEST_F(ViewRenderStateTest, TargetsAreLazyDistinctAndNotReusedAfterRetirement) {
            EXPECT_FALSE(rendering.viewState(first).main_render_target_.valid());
            renderEmpty(first);
            renderEmpty(second);
            const auto first_target = rendering.viewState(first).main_render_target_;
            const auto second_target = rendering.viewState(second).main_render_target_;
            EXPECT_TRUE(first_target.valid());
            EXPECT_TRUE(second_target.valid());
            EXPECT_NE(first_target, second_target);
            EXPECT_FALSE(rendering.viewState(first).split_left_render_target_.valid());
            rendering.retainVisibleViews({first, second});
            rendering.retainVisibleViews({first});
            EXPECT_TRUE(rendering.hasViewState(second));
            rendering.viewState(second).last_visible -= std::chrono::seconds(1);
            rendering.retainVisibleViews({first});
            EXPECT_FALSE(rendering.hasViewState(second));
            EXPECT_EQ(rendering.viewState(first).main_render_target_, first_target);
            renderEmpty(second);
            EXPECT_NE(rendering.viewState(second).main_render_target_, second_target);
        }

        TEST_F(ViewRenderStateTest, ScreenReplacementDropsFramesWithReusedAreaIds) {
            rendering.retainVisibleViews({first, second});
            renderEmpty(first);
            const auto old_target = rendering.viewState(first).main_render_target_;
            source.resetToDefault();
            rendering.retainVisibleViews({source.activeView()});
            renderEmpty(source.activeView());
            EXPECT_NE(rendering.viewState(source.activeView()).main_render_target_, old_target);
            EXPECT_FALSE(rendering.hasViewState(second));
        }

        TEST_F(ViewRenderStateTest, DepthUndoRestoresItsViewAfterActiveAreaChanges) {
            source.screen().setActiveView(screen::AreaId{first});
            const auto before = rendering.depthWindowSnapshot(rendering.activeViewId());
            auto window = before.window;
            window.far_plane = 8.0f;
            rendering.setDepthWindow(window);
            const auto after = rendering.depthWindowSnapshot(rendering.activeViewId());
            op::DepthWindowSettingsUndoEntry undo(rendering, before, after, false);
            source.screen().setActiveView(screen::AreaId{second});
            const auto other = rendering.getDepthWindow();
            undo.undo();
            EXPECT_EQ(rendering.getDepthWindow(), other);
            source.screen().setActiveView(screen::AreaId{first});
            EXPECT_EQ(rendering.getDepthWindow(), before.window);
            source.screen().setActiveView(screen::AreaId{second});
            undo.redo();
            EXPECT_EQ(rendering.getDepthWindow(), other);
            source.screen().setActiveView(screen::AreaId{first});
            EXPECT_EQ(rendering.getDepthWindow(), after.window);
        }

        TEST_F(ViewRenderStateTest, HistoryPlaybackDirtiesOnlyTheDepthWindowsOwner) {
            struct Cleanup {
                ~Cleanup() {
                    op::undoHistory().clear();
                    services().clear();
                }
            } cleanup;
            op::undoHistory().clear();
            services().set(&rendering);
            const auto before = rendering.depthWindowSnapshot(first);
            auto window = before.window;
            window.far_plane = 11.0f;
            rendering.setDepthWindow(window);
            const auto after = rendering.depthWindowSnapshot(first);
            op::undoHistory().push(std::make_unique<op::DepthWindowSettingsUndoEntry>(rendering, before, after, false));
            source.screen().setActiveView(screen::AreaId{second});
            clearDirty();
            ASSERT_TRUE(op::undoHistory().undo().success);
            EXPECT_NE(rendering.viewState(first).dirty_mask_.load(), 0u);
            EXPECT_EQ(rendering.viewState(second).dirty_mask_.load(), 0u);
            clearDirty();
            ASSERT_TRUE(op::undoHistory().redo().success);
            EXPECT_NE(rendering.viewState(first).dirty_mask_.load(), 0u);
            EXPECT_EQ(rendering.viewState(second).dirty_mask_.load(), 0u);
            op::undoHistory().clear();
            op::undoHistory().beginTransaction("Adjust depth window");
            op::undoHistory().push(std::make_unique<op::DepthWindowSettingsUndoEntry>(rendering, before, after, false));
            op::undoHistory().push(std::make_unique<op::DepthWindowSettingsUndoEntry>(rendering, before, after, false));
            op::undoHistory().commitTransaction();
            clearDirty();
            ASSERT_TRUE(op::undoHistory().undo().success);
            EXPECT_NE(rendering.viewState(first).dirty_mask_.load(), 0u);
            EXPECT_EQ(rendering.viewState(second).dirty_mask_.load(), 0u);
        }

        TEST_F(ViewRenderStateTest, ComparisonTargetsBelongOnlyToTheComparingView) {
#ifndef LFS_GRAPHICS_VULKAN
            GTEST_SKIP() << "Split-view render targets are an explicit Phase 3 capability on Metal";
#endif
            source.editViewSettings(first, [](ViewSettings& settings) { settings.split_view_mode = SplitViewMode::PLYComparison; });
            renderEmpty(first);
            renderEmpty(second);
            EXPECT_TRUE(rendering.viewState(first).split_left_render_target_.valid());
            EXPECT_TRUE(rendering.viewState(first).split_right_render_target_.valid());
            EXPECT_FALSE(rendering.viewState(second).split_left_render_target_.valid());
            source.editViewSettings(second, [](ViewSettings& settings) { settings.split_view_mode = SplitViewMode::PLYComparison; });
            renderEmpty(first);
            renderEmpty(second);
            EXPECT_FALSE(rendering.viewState(first).split_left_render_target_.valid());
            EXPECT_FALSE(rendering.viewState(first).split_right_render_target_.valid());
            EXPECT_TRUE(rendering.viewState(second).split_left_render_target_.valid());
            EXPECT_TRUE(rendering.viewState(second).split_right_render_target_.valid());
        }

        TEST_F(ViewRenderStateTest, ComparisonDividerUsesItsViewWhenAnotherIsActive) {
            source.editViewSettings(first, [](ViewSettings& settings) {
                settings.split_view_mode = SplitViewMode::PLYComparison;
                settings.split_position = .25f;
            });
            source.screen().setActiveView(screen::AreaId{second});
            const auto divider = rendering.getSplitDividerScreenX(first, {100, 200}, {800, 600});
            ASSERT_TRUE(divider);
            EXPECT_FLOAT_EQ(*divider, 300.0f);
            EXPECT_FALSE(rendering.getSplitDividerScreenX(second, {100, 200}, {800, 600}));
        }

        TEST_F(ViewRenderStateTest, DepthUndoSurvivesRetirementButExpiresOnProjectReplacement) {
            source.screen().setActiveView(screen::AreaId{second});
            rendering.restoreDepthWindowStateFromProject();
            const auto before = rendering.depthWindowSnapshot(rendering.activeViewId());
            auto window = before.window;
            window.far_plane = 7.0f;
            rendering.setDepthWindow(window);
            const auto after = rendering.depthWindowSnapshot(rendering.activeViewId());
            op::DepthWindowSettingsUndoEntry undo(rendering, before, after, false);
            rendering.retainVisibleViews({first, second});
            source.screen().setActiveView(screen::AreaId{first});
            rendering.viewState(second).last_visible -= std::chrono::seconds(1);
            rendering.retainVisibleViews({first});
            ASSERT_FALSE(rendering.hasViewState(second));
            undo.undo();
            EXPECT_FLOAT_EQ(-source.viewSettings(second)->depth_filter_min.z, before.window.far_plane);
            undo.redo();
            EXPECT_FLOAT_EQ(-source.viewSettings(second)->depth_filter_min.z, after.window.far_plane);
            rendering.dropViewStates();
            undo.undo();
            EXPECT_FLOAT_EQ(-source.viewSettings(second)->depth_filter_min.z, after.window.far_plane);
            EXPECT_FALSE(rendering.hasViewState(second));
        }

        TEST_F(ViewRenderStateTest, HiddenEditorKeepsDepthUndoAndSettingsOwnership) {
            source.screen().setActiveView(screen::AreaId{second});
            const auto before = rendering.depthWindowSnapshot(rendering.activeViewId());
            auto window = before.window;
            window.far_plane = 9.0f;
            rendering.setDepthWindow(window);
            const auto after = rendering.depthWindowSnapshot(rendering.activeViewId());
            op::DepthWindowSettingsUndoEntry undo(rendering, before, after, false);
            ASSERT_TRUE(source.screen().setEditor(screen::AreaId{second}, "console"));
            source.screen().setActiveView(screen::AreaId{first});
            undo.undo();
            ASSERT_TRUE(source.viewSettings(second));
            EXPECT_FLOAT_EQ(-source.viewSettings(second)->depth_filter_min.z, before.window.far_plane);
            EXPECT_TRUE(source.screen().setEditor(screen::AreaId{second}, "view3d"));
            undo.redo();
            EXPECT_FLOAT_EQ(-source.viewSettings(second)->depth_filter_min.z, after.window.far_plane);
        }

        TEST_F(ViewRenderStateTest, DepthDragKeepsItsOwnerWhenAnotherViewBecomesActive) {
            const auto original = rendering.depthWindowSnapshot(first);
            uint64_t token = 0;
            rendering.beginDepthWindowDrag(first, token);
            rendering.beginDepthWindowPreview(first);
            source.screen().setActiveView(screen::AreaId{second});
            const auto other = rendering.getDepthWindow();
            auto edited = original.window;
            edited.scale_x = .7f;
            clearDirty();
            ASSERT_TRUE(rendering.applyDepthWindowIfEpoch(first, edited, original.mode_epoch, token));
            EXPECT_EQ(rendering.getDepthWindow(), other);
            EXPECT_EQ(rendering.viewState(second).dirty_mask_.load(), 0u);
            EXPECT_FALSE(rendering.depthWindowDragPreview());
            op::DepthWindowModeSnapshot committed;
            ASSERT_TRUE(rendering.commitDepthWindowIfEpoch(first, edited, original.mode_epoch, token, committed));
            EXPECT_EQ(committed.view, first);
            EXPECT_EQ(committed.window, edited);
            rendering.endDepthWindowPreview(first);
            rendering.endDepthWindowDrag(first, token);
            EXPECT_EQ(rendering.viewState(first).depth_window_preview_count_, 0);
        }

        TEST_F(ViewRenderStateTest, ClosingOrReplacingAViewInvalidatesCameraDrags) {
            struct Targets : ViewTargets {
                screen::ScreenService& source;
                explicit Targets(screen::ScreenService& source) : source(source) {}
                uint64_t viewEpoch() const override { return source.screenEpoch(); }
                ViewTarget activeView() override { return findView(source.activeView()); }
                ViewTarget viewAt(float x, float y) override { return activeView().contains(x, y) ? activeView() : ViewTarget{}; }
                ViewTarget findView(ViewId id) override {
                    auto* view = source.view3D(id);
                    return view ? ViewTarget{id, &view->camera, {0, 0}, {200, 200}} : ViewTarget{};
                }
                ViewId viewId(const Viewport& camera) const override {
                    for (auto id : source.screen().views())
                        if (&source.view3D(id.value)->camera == &camera)
                            return id.value;
                    return kNoView;
                }
                void activateView(ViewId id) override { source.screen().setActiveView(screen::AreaId{id}); }
                bool runViewCommand(ViewId, std::string_view) override { return false; }
            } targets(source);
            gui::guiFocusState().reset();
            InputController controller(nullptr, targets);
            controller.updateViewportBounds(0, 0, 200, 200);
            const auto press = [&] {
                controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE), input::ACTION_PRESS, 100, 100);
            };
            const auto release = [&] {
                controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE), input::ACTION_RELEASE, 125, 110);
            };
            press();
            ASSERT_TRUE(controller.isContinuousInputActive());
            ASSERT_TRUE(source.screen().close(screen::AreaId{first}));
            const auto rotation = source.activeView3D().camera.camera.R;
            EXPECT_FALSE(controller.isContinuousInputActive());
            controller.handleMouseMove(125, 110);
            release();
            EXPECT_EQ(source.activeView3D().camera.camera.R, rotation);
            press();
            ASSERT_TRUE(controller.isContinuousInputActive());
            source.resetToDefault();
            EXPECT_FALSE(controller.isContinuousInputActive());
            controller.handleMouseMove(125, 110);
            release();
        }

        TEST(ViewOverlayContextTest, NestedDrawsRestoreTheirCameraAndRectangle) {
            ViewInfo outer{};
            outer.width = 640;
            outer.height = 480;
            outer.translation = {1, 2, 3};
            const auto original = get_current_view_info();
            {
                python::ScopedOverlayDrawContext context({.viewport_bounds = std::array<float, 4>{10, 20, 640, 480}, .camera = &outer});
                EXPECT_EQ(get_current_view_info()->translation, outer.translation);
                {
                    auto inner = outer;
                    inner.translation = {4, 5, 6};
                    python::ScopedOverlayDrawContext nested_context({.viewport_bounds = std::array<float, 4>{660, 20, 320, 480}, .camera = &inner});
                    EXPECT_EQ(get_current_view_info()->translation, inner.translation);
                    float x, y, w, h;
                    python::get_viewport_bounds(x, y, w, h);
                    EXPECT_FLOAT_EQ(x, 660);
                    EXPECT_FLOAT_EQ(w, 320);
                }
                EXPECT_EQ(get_current_view_info()->translation, outer.translation);
                float x, y, w, h;
                python::get_viewport_bounds(x, y, w, h);
                EXPECT_FLOAT_EQ(x, 10);
                EXPECT_FLOAT_EQ(w, 640);
            }
            EXPECT_EQ(get_current_view_info().has_value(), original.has_value());
        }
    } // namespace
} // namespace lfs::vis
