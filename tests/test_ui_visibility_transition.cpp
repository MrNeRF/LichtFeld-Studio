/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "gui/gui_manager.hpp"
#include "licht_test_support.hpp"
#include "rendering/rendering_manager.hpp"
#include "visualizer/visualizer_impl.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <glm/vec2.hpp>
#include <gtest/gtest.h>
#include <optional>
#include <string>

namespace lfs::vis {
    class UiVisibilityTransitionTest : public ::testing::Test {
    protected:
        void SetUp() override {
            if (const auto* previous = std::getenv("LFS_HOME"))
                previous_home_ = previous;
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", temporary_.path.string().c_str());
#else
            (void)setenv("LFS_HOME", temporary_.path.string().c_str(), 1);
#endif
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
        }

        void TearDown() override {
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
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
        gui->ui_visibility_target_layout_.pos = {0.0f, 0.0f};
        gui->ui_visibility_target_layout_.size = {960.0f, 540.0f};

        gui->commitUiVisibilityTransitionIfFrameReady(false);
        EXPECT_FALSE(gui->ui_hidden_);
        EXPECT_EQ(gui->viewport_layout_.pos, glm::vec2(20.0f, 30.0f));
        EXPECT_EQ(gui->viewport_layout_.size, glm::vec2(640.0f, 480.0f));
        EXPECT_TRUE(gui->ui_visibility_target_ready_);
        EXPECT_FALSE(gui->ui_visibility_layout_committed_);

        gui->commitUiVisibilityTransitionIfFrameReady(true);
        EXPECT_TRUE(gui->ui_hidden_);
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
    }

} // namespace lfs::vis
