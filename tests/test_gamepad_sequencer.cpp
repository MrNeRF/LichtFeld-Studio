/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/services.hpp"
#include "gui/gui_manager.hpp"
#include "licht_test_support.hpp"
#include "operation/undo_history.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "visualizer_impl.hpp"
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <optional>
#include <string>

namespace {
    using lfs::sequencer::Keyframe;
    using lfs::vis::LoopMode;
    using lfs::vis::SequencerController;
    Keyframe makeKeyframe(float time) {
        Keyframe key;
        key.time = time;
        return key;
    }

    class GamepadSequencerTest : public ::testing::Test {
    protected:
        void SetUp() override {
            if (const char* previous = std::getenv("LFS_HOME"))
                previous_home_ = previous;
            setHome(temporary_.path.string().c_str());
            lfs::event::EventBridge::instance().clear_all();
            lfs::vis::services().clear();
            lfs::vis::op::undoHistory().clear();
            ASSERT_TRUE(lfs::event::LocalizationManager::getInstance().initialize(
                (std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/resources/locales").string()));
        }

        void TearDown() override {
            lfs::vis::op::undoHistory().clear();
            lfs::vis::services().clear();
            lfs::event::EventBridge::instance().clear_all();
            lfs::event::LocalizationManager::getInstance().reset();
            setHome(previous_home_ ? previous_home_->c_str() : nullptr);
        }

        static void setHome(const char* value) {
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", value ? value : "");
#else
            if (value)
                (void)setenv("LFS_HOME", value, 1);
            else
                (void)unsetenv("LFS_HOME");
#endif
        }

        lfs::vis::ViewerOptions options() const {
            lfs::vis::ViewerOptions result;
            result.show_startup_overlay = false;
            result.project_lifecycle_settings_path = temporary_.path / "lifecycle.json";
            return result;
        }

        lfs::test::licht::TemporaryDirectory temporary_{"gamepad-sequencer"};
        std::optional<std::string> previous_home_;
    };

    TEST(GamepadSequencer, AdjacentKeyframeUsesSelectionAndStopsAtEndpoints) {
        SequencerController controller;
        EXPECT_FALSE(controller.adjacentKeyframeIndex(1));
        const auto first = controller.addKeyframe(makeKeyframe(0));
        const auto middle = controller.addKeyframe(makeKeyframe(2));
        const auto last = controller.addKeyframe(makeKeyframe(5));
        controller.setLoopMode(LoopMode::LOOP);
        controller.seek(1);
        const auto next = controller.adjacentKeyframeIndex(1);
        ASSERT_TRUE(next);
        EXPECT_EQ(controller.timeline().getKeyframe(*next)->id, middle);
        const auto previous = controller.adjacentKeyframeIndex(-1);
        ASSERT_TRUE(previous);
        EXPECT_EQ(controller.timeline().getKeyframe(*previous)->id, first);
        ASSERT_TRUE(controller.selectKeyframeById(first));
        EXPECT_FALSE(controller.adjacentKeyframeIndex(-1));
        ASSERT_TRUE(controller.selectKeyframeById(middle));
        EXPECT_EQ(controller.timeline().getKeyframe(*controller.adjacentKeyframeIndex(1))->id, last);
        ASSERT_TRUE(controller.selectKeyframeById(last));
        EXPECT_FALSE(controller.adjacentKeyframeIndex(1)); // synthetic loop keys are skipped
        EXPECT_FALSE(controller.adjacentKeyframeIndex(0));
        EXPECT_FALSE(controller.adjacentKeyframeIndex(2));
    }

    TEST_F(GamepadSequencerTest, ControllerCaptureUsesAssignedViewAndSharesUndoAndReplacement) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& ui = viewer.getGuiManager()->sequencerUI();
        auto& controller = ui.controller();
        auto& history = lfs::vis::op::undoHistory();
        Viewport secondary(200, 200);
        secondary.camera.t = {3, 5, 7};
        controller.seek(2);
        ui.addKeyframeFromView(secondary);
        ASSERT_EQ(controller.timeline().realKeyframeCount(), 1u);
        auto original = *controller.timeline().getKeyframe(0);
        EXPECT_EQ(original.position, secondary.camera.t);
        secondary.camera.t = {9, 8, 7};
        ui.addKeyframeFromView(secondary);
        ASSERT_EQ(controller.timeline().realKeyframeCount(), 1u);
        EXPECT_EQ(controller.timeline().getKeyframe(0)->position, secondary.camera.t);
        ASSERT_TRUE(history.undo().success);
        EXPECT_EQ(controller.timeline().getKeyframe(0)->position, original.position);
        ASSERT_TRUE(history.undo().success);
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 0u);
        ASSERT_TRUE(history.redo().success);
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 1u);
    }

    TEST_F(GamepadSequencerTest, AdjacentNavigationAndCaptureShareTheAssignedViewport) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& ui = viewer.getGuiManager()->sequencerUI();
        auto& controller = ui.controller();
        Viewport secondary(200, 200);
        auto first = makeKeyframe(0);
        auto second = makeKeyframe(2);
        first.position = {3, 5, 7};
        second.position = {9, 8, 7};
        const auto first_id = controller.addKeyframe(first);
        controller.addKeyframe(second);
        ASSERT_TRUE(controller.selectKeyframeById(first_id));
        const auto primary_before = viewer.getViewport().camera.t;
        ASSERT_TRUE(ui.selectAdjacentKeyframe(1, &secondary));
        EXPECT_EQ(secondary.camera.t, second.position);
        EXPECT_EQ(viewer.getViewport().camera.t, primary_before);
        ui.addKeyframeFromView(secondary);
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 2u);
        EXPECT_EQ(controller.timeline().getKeyframe(1)->position, second.position);
    }

    TEST_F(GamepadSequencerTest, ControllerDeleteRequiresActualSelectionAndCanBeUndone) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& ui = viewer.getGuiManager()->sequencerUI();
        auto& controller = ui.controller();
        auto& history = lfs::vis::op::undoHistory();
        const auto first = controller.addKeyframe(makeKeyframe(0));
        const auto second = controller.addKeyframe(makeKeyframe(2));
        history.clear();
        controller.seek(2);
        EXPECT_FALSE(ui.removeSelectedKeyframe());
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 2u);
        EXPECT_FALSE(history.canUndo());
        ASSERT_TRUE(ui.selectAdjacentKeyframe(-1));
        ASSERT_TRUE(controller.selectedKeyframeId());
        EXPECT_EQ(*controller.selectedKeyframeId(), first);
        EXPECT_FALSE(ui.removeSelectedKeyframe());
        EXPECT_FALSE(history.canUndo());
        ASSERT_TRUE(ui.selectAdjacentKeyframe(1));
        EXPECT_EQ(*controller.selectedKeyframeId(), second);
        EXPECT_FLOAT_EQ(controller.playhead(), 2);
        EXPECT_FALSE(ui.selectAdjacentKeyframe(1));
        ASSERT_TRUE(ui.removeSelectedKeyframe());
        EXPECT_EQ(controller.timeline().getKeyframeById(second), nullptr);
        EXPECT_FALSE(ui.removeSelectedKeyframe());
        ASSERT_TRUE(history.undo().success);
        EXPECT_NE(controller.timeline().getKeyframeById(second), nullptr);
        EXPECT_NE(controller.timeline().getKeyframeById(first), nullptr);
    }

} // namespace
