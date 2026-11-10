/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "gui/gui_manager.hpp"
#include "licht_test_support.hpp"
#include "project/session_state.hpp"
#include "python/python_runtime.hpp"
#include "rendering/rendering_manager.hpp"
#include "visualizer/visualizer_impl.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <glm/vec2.hpp>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <utility>

namespace lfs::vis {
    void installSessionTestInputController(VisualizerImpl& viewer);
    class UiVisibilityTransitionTest : public ::testing::Test {
    protected:
        float previous_dpi_ = 1.0f;
        void SetUp() override {
            previous_dpi_ = lfs::python::get_shared_dpi_scale();
            lfs::python::set_shared_dpi_scale(1.0f);
            if (const auto* previous = std::getenv("LFS_HOME"))
                previous_home_ = previous;
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", temporary_.path.string().c_str());
#else
            (void)setenv("LFS_HOME", temporary_.path.string().c_str(), 1);
#endif
            lfs::event::EventBridge::instance().clear_all();
        }

        void TearDown() override {
            lfs::python::set_shared_dpi_scale(previous_dpi_);
            lfs::event::EventBridge::instance().clear_all();
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", previous_home_ ? previous_home_->c_str() : "");
#else
            if (previous_home_)
                (void)setenv("LFS_HOME", previous_home_->c_str(), 1);
            else
                (void)unsetenv("LFS_HOME");
#endif
        }

        ViewerOptions projectOptions() const {
            ViewerOptions options;
            options.show_startup_overlay = false;
            options.project_lifecycle_settings_path = temporary_.path / "lifecycle.json";
            return options;
        }

        lfs::test::licht::TemporaryDirectory temporary_{"lfs-ui-visibility"};
        std::optional<std::string> previous_home_;
    };

    TEST_F(UiVisibilityTransitionTest,
           UiVisibilityWaitsForMatchingFrame) {
        VisualizerImpl viewer(projectOptions());
        auto* const gui = viewer.getGuiManager();
        ASSERT_NE(gui, nullptr);

        gui->ui_hidden_ = false;
        gui->viewport_layout_.pos = {20.0f, 30.0f};
        gui->viewport_layout_.size = {640.0f, 480.0f};
        gui->ui_visibility_resize_active_ = true;
        gui->ui_visibility_layout_committed_ = false;
        gui->ui_visibility_target_ready_ = true;
        gui->ui_visibility_target_hidden_ = true;
        gui->ui_target_left_collapsed_ = true;
        gui->ui_visibility_target_layout_.pos = {0.0f, 0.0f};
        gui->ui_visibility_target_layout_.size = {960.0f, 540.0f};

        gui->commitUiVisibilityTransitionIfFrameReady(false);
        EXPECT_FALSE(gui->ui_hidden_);
        EXPECT_TRUE(gui->panelAnimationVisibility().right_panel_visible);
        EXPECT_FALSE(gui->panel_layout_.isLeftDockCollapsed());
        EXPECT_EQ(gui->viewport_layout_.pos, glm::vec2(20.0f, 30.0f));
        EXPECT_EQ(gui->viewport_layout_.size, glm::vec2(640.0f, 480.0f));
        EXPECT_TRUE(gui->ui_visibility_target_ready_);
        EXPECT_FALSE(gui->ui_visibility_layout_committed_);

        gui->commitUiVisibilityTransitionIfFrameReady(true);
        EXPECT_TRUE(gui->ui_hidden_);
        EXPECT_FALSE(gui->panelAnimationVisibility().right_panel_visible);
        EXPECT_TRUE(gui->panel_layout_.isLeftDockCollapsed());
        EXPECT_FALSE(gui->panel_layout_.isRightDockCollapsed());
        EXPECT_EQ(gui->viewport_layout_.pos, glm::vec2(0.0f, 0.0f));
        EXPECT_EQ(gui->viewport_layout_.size, glm::vec2(960.0f, 540.0f));
        EXPECT_FALSE(gui->ui_visibility_target_ready_);
        EXPECT_TRUE(gui->ui_visibility_layout_committed_);

        gui->ui_visibility_resize_active_ = true;
        gui->ui_visibility_layout_committed_ = false;
        gui->ui_visibility_target_ready_ = true;
        gui->ui_visibility_target_hidden_ = false;
        gui->ui_visibility_target_layout_.pos = {20.0f, 30.0f};
        gui->ui_visibility_target_layout_.size = {640.0f, 480.0f};

        gui->commitUiVisibilityTransitionIfFrameReady(false);
        EXPECT_TRUE(gui->ui_hidden_);
        EXPECT_EQ(gui->viewport_layout_.pos, glm::vec2(0.0f, 0.0f));
        EXPECT_EQ(gui->viewport_layout_.size, glm::vec2(960.0f, 540.0f));

        gui->commitUiVisibilityTransitionIfFrameReady(true);
        EXPECT_FALSE(gui->ui_hidden_);
        EXPECT_TRUE(gui->panelAnimationVisibility().right_panel_visible);
        EXPECT_TRUE(gui->panel_layout_.isLeftDockCollapsed());
        EXPECT_EQ(gui->viewport_layout_.pos, glm::vec2(20.0f, 30.0f));
        EXPECT_EQ(gui->viewport_layout_.size, glm::vec2(640.0f, 480.0f));
    }

    TEST_F(UiVisibilityTransitionTest,
           UiVisibilityTimeoutCommitsRequestedLayout) {
        VisualizerImpl viewer(projectOptions());
        auto* const gui = viewer.getGuiManager();
        ASSERT_NE(gui, nullptr);
        auto* const rendering = viewer.getRenderingManager();
        ASSERT_NE(rendering, nullptr);

        gui->ui_hidden_ = false;
        gui->viewport_layout_.pos = {25.0f, 35.0f};
        gui->viewport_layout_.size = {640.0f, 480.0f};
        gui->ui_visibility_resize_active_ = true;
        gui->ui_visibility_layout_committed_ = false;
        gui->ui_visibility_target_ready_ = true;
        gui->ui_visibility_target_hidden_ = true;
        gui->ui_visibility_target_layout_.pos = {0.0f, 0.0f};
        gui->ui_visibility_target_layout_.size = {960.0f, 540.0f};
        gui->ui_visibility_deadline_ =
            std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
        rendering->setViewportResizeActive(
            true, ViewportResizeRenderPolicy::FullResolution);

        gui->updateInteractiveTransitionGuard();

        EXPECT_TRUE(gui->ui_hidden_);
        EXPECT_EQ(gui->viewport_layout_.pos, glm::vec2(0.0f, 0.0f));
        EXPECT_EQ(gui->viewport_layout_.size, glm::vec2(960.0f, 540.0f));
        EXPECT_FALSE(gui->ui_visibility_target_ready_);
        EXPECT_FALSE(gui->ui_visibility_resize_active_);
        EXPECT_FALSE(gui->ui_visibility_layout_committed_);
        // Ending an active resize deliberately enters the short settle/debounce
        // phase before the full viewport refresh is considered complete.
        EXPECT_TRUE(rendering->isViewportResizeDeferring());
        EXPECT_TRUE(rendering->hasPendingViewportResizeSettle());
    }

    TEST_F(UiVisibilityTransitionTest, SideDockCollapseRoundTripsWithoutReplacingPreferredWidths) {
        using namespace lfs::vis::project;
        using lfs::test::licht::json_root;
        using lfs::test::licht::require_result;
        VisualizerImpl viewer(projectOptions());
        installSessionTestInputController(viewer);
        auto& layout = viewer.getGuiManager()->panelLayout();
        auto state = layout.captureProjectState();
        state.left_dock_width = 290.0f;
        state.right_panel_width = 370.0f;
        state.left_dock_collapsed = state.right_dock_collapsed = true;
        layout.applyProjectState(state);
        std::vector<CameraBookmarkProjectState> bookmarks;
        auto captured = require_result(captureGuiSession(viewer, {}, bookmarks));
        auto gui = json_root(captured.gui_layout.dom());
        auto& fixed = gui["layouts"][0]["areas"][0]["spaces"][0]["opaque_payload"];
        EXPECT_EQ(fixed["left_dock_collapsed"], true);
        EXPECT_EQ(fixed["right_dock_collapsed"], true);
        EXPECT_EQ(fixed["left_dock_width"], 290.0f);
        EXPECT_EQ(fixed["right_panel_width"], 370.0f);
        layout.applyProjectState({});
        applyGuiSession(viewer, require_result(prepareGuiSessionRestore(captured)), bookmarks);
        EXPECT_TRUE(layout.isLeftDockCollapsed());
        EXPECT_TRUE(layout.isRightDockCollapsed());
        EXPECT_FLOAT_EQ(layout.captureProjectState().left_dock_width, 290.0f);
        EXPECT_FLOAT_EQ(layout.captureProjectState().right_panel_width, 370.0f);
        fixed.erase("left_dock_collapsed");
        fixed.erase("right_dock_collapsed");
        captured.gui_layout = require_result(lfs::io::project::GuiLayoutChapter::parse(gui.dump()));
        applyGuiSession(viewer, require_result(prepareGuiSessionRestore(captured)), bookmarks);
        EXPECT_FALSE(layout.isLeftDockCollapsed());
        EXPECT_FALSE(layout.isRightDockCollapsed());
    }

    TEST_F(UiVisibilityTransitionTest, UiVisibilityFinishesWithoutWaitingForFullscreenGuard) {
        VisualizerImpl viewer(projectOptions());
        auto* const gui = viewer.getGuiManager();
        ASSERT_NE(gui, nullptr);

        gui->interactive_transition_guard_until_ =
            std::chrono::steady_clock::now() + std::chrono::seconds(3);
        const auto fullscreen_deadline = gui->interactive_transition_guard_until_;
        gui->ui_visibility_deadline_ =
            std::chrono::steady_clock::now() + std::chrono::seconds(1);
        gui->ui_visibility_resize_active_ = true;
        gui->ui_visibility_layout_committed_ = true;

        gui->updateInteractiveTransitionGuard();

        EXPECT_FALSE(gui->ui_visibility_resize_active_);
        EXPECT_FALSE(gui->ui_visibility_layout_committed_);
        EXPECT_EQ(gui->interactive_transition_guard_until_, fullscreen_deadline);
    }

    TEST_F(UiVisibilityTransitionTest, UiVisibilityRequestsFreshFrameWithoutCooldown) {
        VisualizerImpl viewer(projectOptions());
        auto* const gui = viewer.getGuiManager();
        auto* const rendering = viewer.getRenderingManager();
        ASSERT_NE(gui, nullptr);
        ASSERT_NE(rendering, nullptr);

        gui->last_ui_layout_work_pos_ = {0.0f, 30.0f};
        gui->last_ui_layout_work_size_ = {960.0f, 510.0f};
        gui->requestUiVisibilityToggle();
        gui->updateUiVisibilityTransition();

        EXPECT_FALSE(gui->ui_toggle_pending_);
        EXPECT_TRUE(gui->ui_visibility_target_ready_);
        EXPECT_TRUE(gui->ui_visibility_resize_active_);
        EXPECT_EQ(gui->interactive_transition_guard_until_,
                  std::chrono::steady_clock::time_point{});
        EXPECT_NE(rendering->pendingDirtyMask() & DirtyFlag::VIEWPORT, 0);
        const auto plan = rendering->frameDemandLedger().plan(std::chrono::steady_clock::now());
        EXPECT_NE(plan.render_views, 0);
        EXPECT_NE(std::find(plan.details.begin(), plan.details.end(), "ui_visibility"), plan.details.end());

        // A second request must run as soon as the matching frame completes,
        // rather than waiting 750 ms (or three seconds during training).
        gui->commitUiVisibilityTransitionIfFrameReady(true);
        gui->updateInteractiveTransitionGuard();
        gui->requestUiVisibilityToggle();
        gui->updateUiVisibilityTransition();
        EXPECT_FALSE(gui->ui_toggle_pending_);
        EXPECT_FALSE(gui->ui_visibility_target_hidden_);
        EXPECT_TRUE(gui->ui_visibility_target_ready_);
        // Reopening a dock must commit the same console width used by its prepared frame.
        gui->ui_visibility_resize_active_ = false;
        gui->ui_visibility_layout_committed_ = false;
        gui->ui_visibility_target_ready_ = false;
        const gui::ScreenState screen{.work_size = {900.0f, 600.0f}};
        gui->ui_hidden_ = false;
        gui->window_states_["python_console"] = true;
        gui->last_ui_layout_work_pos_ = screen.work_pos;
        gui->last_ui_layout_work_size_ = screen.work_size;
        gui->panel_layout_.setSideDockCollapsed(lfs::core::SideDock::Left, true);
        gui->panel_layout_.setRightPanelWidth(360.0f, screen);
        gui->panel_layout_.setSideDockCollapsed(lfs::core::SideDock::Right, true);
        gui->panel_layout_.setPythonConsoleWidth(450.0f);
        gui->side_dock_toggle_pending_.set(std::to_underlying(lfs::core::SideDock::Right));
        gui->updateUiVisibilityTransition();
        ASSERT_TRUE(gui->ui_visibility_target_ready_);
        const auto prepared = gui->ui_visibility_target_layout_;
        EXPECT_FLOAT_EQ(prepared.size.x, 198.0f);
        EXPECT_FLOAT_EQ(gui->panel_layout_.getPythonConsoleWidth(), 450.0f);
        gui->commitUiVisibilityTransitionIfFrameReady(true);
        EXPECT_FLOAT_EQ(gui->panel_layout_.getPythonConsoleWidth(), 338.0f);
        gui->panel_layout_.enforceWidthConstraints(true, false, screen);
        EXPECT_EQ(gui->panel_layout_.computeViewportLayout(true, false, true, screen).size, prepared.size);
    }

} // namespace lfs::vis
