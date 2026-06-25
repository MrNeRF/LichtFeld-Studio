/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/editor_context.hpp"
#include "core/event_bridge/scoped_handler.hpp"
#include "core/events.hpp"
#include "core/services.hpp"
#include "gui/gui_focus_state.hpp"
#include "input/input_controller.hpp"
#include "input/input_router.hpp"
#include "input/key_codes.hpp"
#include "internal/viewport.hpp"
#include "python/python_runtime.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering_manager.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <glm/gtc/constants.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <variant>
#include <imgui.h>

namespace lfs::vis {

    namespace {
        class InputControllerFocusTest : public ::testing::Test {
        protected:
            void SetUp() override {
                isolateInputProfileHome();
                services().clear();
                gui::guiFocusState().reset();

                IMGUI_CHECKVERSION();
                ImGui::CreateContext();
            }

            void TearDown() override {
                ImGui::DestroyContext();

                gui::guiFocusState().reset();
                services().clear();
                restoreHome();
            }

        private:
            std::optional<std::string> old_home_;
            std::filesystem::path temp_home_;

            void isolateInputProfileHome() {
#ifndef _WIN32
                if (const char* home = std::getenv("HOME")) {
                    old_home_ = home;
                }
                temp_home_ = std::filesystem::temp_directory_path() /
                             ("lfs_input_focus_home_" +
                              std::to_string(reinterpret_cast<std::uintptr_t>(this)));
                std::filesystem::create_directories(temp_home_);
                setenv("HOME", temp_home_.string().c_str(), 1);
#endif
            }

            void restoreHome() {
#ifndef _WIN32
                if (old_home_) {
                    setenv("HOME", old_home_->c_str(), 1);
                } else {
                    unsetenv("HOME");
                }
                if (!temp_home_.empty()) {
                    std::error_code ec;
                    std::filesystem::remove_all(temp_home_, ec);
                }
#endif
            }
        };
    } // namespace

    TEST_F(InputControllerFocusTest, CameraViewHotkeysDoNotBypassGuiKeyboardCapture) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        lfs::event::ScopedHandler handlers;
        int goto_cam_view_count = 0;
        handlers.subscribe<core::events::cmd::GoToCamView>(
            [&](const auto&) { ++goto_cam_view_count; });

        auto& focus = gui::guiFocusState();
        focus.want_capture_keyboard = true;

        controller.handleKey(input::KEY_RIGHT, input::ACTION_PRESS, input::KEYMOD_NONE);

        EXPECT_EQ(goto_cam_view_count, 0);
    }

    TEST_F(InputControllerFocusTest, RebindingKeyCaptureBypassesPythonKeyboardCapture) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);

        controller.getBindings().startCapture(input::ToolMode::GLOBAL,
                                              input::Action::TOOL_ALIGN);
        lfs::python::request_keyboard_capture("input-controller-focus-test");
        controller.handleKey(input::KEY_B, input::ACTION_PRESS, input::KEYMOD_NONE);
        lfs::python::release_keyboard_capture("input-controller-focus-test");

        const auto captured = controller.getBindings().getAndClearCaptured();
        ASSERT_TRUE(captured.has_value());

        const auto* key_trigger = std::get_if<input::KeyTrigger>(&*captured);
        ASSERT_NE(key_trigger, nullptr);
        EXPECT_EQ(key_trigger->key, input::KEY_B);
        EXPECT_EQ(key_trigger->modifiers, input::MODIFIER_NONE);
        EXPECT_FALSE(controller.getBindings().isCapturing());
    }

    TEST_F(InputControllerFocusTest, ViewportViewHotkeysDoNotBypassGuiKeyboardFocus) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        lfs::event::ScopedHandler handlers;
        int toggle_gt_count = 0;
        int toggle_split_count = 0;
        handlers.subscribe<core::events::cmd::ToggleGTComparison>(
            [&](const auto&) { ++toggle_gt_count; });
        handlers.subscribe<core::events::cmd::ToggleSplitView>(
            [&](const auto&) { ++toggle_split_count; });

        auto& focus = gui::guiFocusState();
        focus.want_capture_keyboard = true;
        focus.any_item_active = true;

        controller.handleKey(input::KEY_G, input::ACTION_PRESS, input::KEYMOD_NONE);
        controller.handleKey(input::KEY_V, input::ACTION_PRESS, input::KEYMOD_NONE);

        EXPECT_EQ(toggle_gt_count, 0);
        EXPECT_EQ(toggle_split_count, 0);
    }

    TEST_F(InputControllerFocusTest, ViewportViewHotkeysWorkAfterViewportFocus) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        lfs::event::ScopedHandler handlers;
        int toggle_gt_count = 0;
        int toggle_split_count = 0;
        handlers.subscribe<core::events::cmd::ToggleGTComparison>(
            [&](const auto&) { ++toggle_gt_count; });
        handlers.subscribe<core::events::cmd::ToggleSplitView>(
            [&](const auto&) { ++toggle_split_count; });

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);
        router.endMouseButton(input::ACTION_RELEASE);

        controller.handleKey(input::KEY_G, input::ACTION_PRESS, input::KEYMOD_NONE);
        controller.handleKey(input::KEY_V, input::ACTION_PRESS, input::KEYMOD_NONE);

        EXPECT_EQ(toggle_gt_count, 1);
        EXPECT_EQ(toggle_split_count, 1);
    }

    TEST_F(InputControllerFocusTest, ProgrammaticViewportFocusAllowsViewportHotkeys) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        lfs::event::ScopedHandler handlers;
        int toggle_gt_count = 0;
        handlers.subscribe<core::events::cmd::ToggleGTComparison>(
            [&](const auto&) { ++toggle_gt_count; });

        auto& focus = gui::guiFocusState();
        focus.want_capture_keyboard = true;
        focus.any_item_active = true;

        router.focusViewportKeyboard();
        controller.handleKey(input::KEY_G, input::ACTION_PRESS, input::KEYMOD_NONE);

        EXPECT_EQ(toggle_gt_count, 1);
    }

    TEST_F(InputControllerFocusTest, ViewportViewHotkeysStayBlockedDuringTextEntry) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        lfs::event::ScopedHandler handlers;
        int toggle_gt_count = 0;
        int toggle_split_count = 0;
        handlers.subscribe<core::events::cmd::ToggleGTComparison>(
            [&](const auto&) { ++toggle_gt_count; });
        handlers.subscribe<core::events::cmd::ToggleSplitView>(
            [&](const auto&) { ++toggle_split_count; });

        auto& focus = gui::guiFocusState();
        focus.want_capture_keyboard = true;
        focus.want_text_input = true;
        focus.any_item_active = true;

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);
        router.endMouseButton(input::ACTION_RELEASE);

        controller.handleKey(input::KEY_G, input::ACTION_PRESS, input::KEYMOD_NONE);
        controller.handleKey(input::KEY_V, input::ACTION_PRESS, input::KEYMOD_NONE);

        EXPECT_EQ(toggle_gt_count, 0);
        EXPECT_EQ(toggle_split_count, 0);
    }

    TEST_F(InputControllerFocusTest, GlobalShortcutsUseLogicalKeyWhileMovementUsesPhysicalKey) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        controller.initialize();
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        lfs::event::ScopedHandler handlers;
        int undo_count = 0;
        handlers.subscribe<core::events::cmd::Undo>(
            [&](const auto&) { ++undo_count; });

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);
        router.endMouseButton(input::ACTION_RELEASE);

        controller.handleKey(input::KEY_W, input::KEY_Z, 0, input::ACTION_PRESS, input::KEYMOD_CTRL);

        EXPECT_EQ(undo_count, 1);
        EXPECT_FALSE(controller.isContinuousInputActive());

        controller.handleKey(input::KEY_W, input::KEY_Z, 0, input::ACTION_PRESS, input::KEYMOD_NONE);

        EXPECT_TRUE(controller.isContinuousInputActive());
    }

    TEST_F(InputControllerFocusTest, RedoAliasUsesLogicalKeyAndDoesNotTriggerMovement) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        controller.initialize();
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        lfs::event::ScopedHandler handlers;
        int redo_count = 0;
        handlers.subscribe<core::events::cmd::Redo>(
            [&](const auto&) { ++redo_count; });

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);
        router.endMouseButton(input::ACTION_RELEASE);

        controller.handleKey(input::KEY_W, input::KEY_Z, 0, input::ACTION_PRESS,
                             input::KEYMOD_CTRL | input::KEYMOD_SHIFT);

        EXPECT_EQ(redo_count, 1);
        EXPECT_FALSE(controller.isContinuousInputActive());
    }

    TEST_F(InputControllerFocusTest, CameraDragBindingsIgnoreExtraShiftModifier) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);

        EXPECT_EQ(controller.getBindings().getActionForDrag(
                      input::ToolMode::GLOBAL, input::MouseButton::MIDDLE, input::KEYMOD_SHIFT),
                  input::Action::CAMERA_ORBIT);
        EXPECT_EQ(controller.getBindings().getActionForDrag(
                      input::ToolMode::GLOBAL, input::MouseButton::RIGHT, input::KEYMOD_SHIFT),
                  input::Action::CAMERA_PAN);
    }

    TEST_F(InputControllerFocusTest, GlobalCameraBindingsFallBackToToolModes) {
        input::InputBindings bindings;

        EXPECT_EQ(bindings.getActionForDrag(
                      input::ToolMode::SELECTION, input::MouseButton::RIGHT, input::KEYMOD_NONE),
                  input::Action::CAMERA_PAN);

        bindings.setBinding(input::ToolMode::SELECTION,
                            input::Action::CAMERA_ORBIT,
                            input::MouseDragTrigger{input::MouseButton::RIGHT, input::MODIFIER_NONE});

        EXPECT_EQ(bindings.getActionForDrag(
                      input::ToolMode::SELECTION, input::MouseButton::RIGHT, input::KEYMOD_NONE),
                  input::Action::CAMERA_ORBIT);
        EXPECT_EQ(bindings.getActionForDrag(
                      input::ToolMode::ALIGN, input::MouseButton::RIGHT, input::KEYMOD_NONE),
                  input::Action::CAMERA_PAN);
    }

    TEST_F(InputControllerFocusTest, GlobalSetPivotDoubleClickWorksInEveryToolMode) {
        input::InputBindings bindings;

        for (const auto mode : input::kAllToolModes) {
            EXPECT_EQ(bindings.getActionForMouseButton(
                          mode, input::MouseButton::RIGHT, input::MODIFIER_NONE, true),
                      input::Action::CAMERA_SET_PIVOT)
                << "tool mode " << static_cast<int>(mode);
        }

        // Selection deliberately owns an ordinary right-click for polygon undo;
        // that single-click action must not shadow the global double-click.
        EXPECT_EQ(bindings.getActionForMouseButton(
                      input::ToolMode::SELECTION,
                      input::MouseButton::RIGHT,
                      input::MODIFIER_NONE,
                      false),
                  input::Action::UNDO_POLYGON_VERTEX);
    }

    TEST_F(InputControllerFocusTest, LocalDoubleClickOverridesGlobalSetPivot) {
        input::InputBindings bindings;
        bindings.setBinding(
            input::ToolMode::SELECTION,
            input::Action::CAMERA_ORBIT,
            input::MouseButtonTrigger{input::MouseButton::RIGHT, input::MODIFIER_NONE, true});

        EXPECT_EQ(bindings.getActionForMouseButton(
                      input::ToolMode::SELECTION,
                      input::MouseButton::RIGHT,
                      input::MODIFIER_NONE,
                      true),
                  input::Action::CAMERA_ORBIT);
    }

    TEST_F(InputControllerFocusTest, BindingConflictChecksInheritedGlobalBindings) {
        input::InputBindings bindings;
        const input::MouseDragTrigger right_drag{
            input::MouseButton::RIGHT,
            input::MODIFIER_NONE,
        };

        bindings.setBinding(input::ToolMode::TRANSLATE,
                            input::Action::NODE_RECT_SELECT,
                            right_drag);

        const auto conflict = bindings.findConflict(input::ToolMode::TRANSLATE,
                                                    right_drag,
                                                    input::Action::NODE_RECT_SELECT);
        ASSERT_TRUE(conflict.has_value());
        EXPECT_EQ(conflict->other_action, input::Action::CAMERA_PAN);
        EXPECT_EQ(conflict->other_mode, input::ToolMode::GLOBAL);
    }

    TEST_F(InputControllerFocusTest, BindingConflictPrefersSameModeOverInheritedGlobal) {
        input::InputBindings bindings;
        const input::MouseDragTrigger right_drag{
            input::MouseButton::RIGHT,
            input::MODIFIER_NONE,
        };

        bindings.setBinding(input::ToolMode::TRANSLATE,
                            input::Action::NODE_RECT_SELECT,
                            right_drag);
        bindings.setBinding(input::ToolMode::TRANSLATE,
                            input::Action::CAMERA_ORBIT,
                            right_drag);

        const auto conflict = bindings.findConflict(input::ToolMode::TRANSLATE,
                                                    right_drag,
                                                    input::Action::NODE_RECT_SELECT);
        ASSERT_TRUE(conflict.has_value());
        EXPECT_EQ(conflict->other_action, input::Action::CAMERA_ORBIT);
        EXPECT_EQ(conflict->other_mode, input::ToolMode::TRANSLATE);
    }

    TEST_F(InputControllerFocusTest, TransformModeLeftDragUsesNodeSelectionNotSelectionStroke) {
        input::InputBindings bindings;

        EXPECT_EQ(bindings.getActionForDrag(
                      input::ToolMode::TRANSLATE, input::MouseButton::LEFT, input::KEYMOD_NONE),
                  input::Action::NODE_RECT_SELECT);
    }

    TEST_F(InputControllerFocusTest, StaleMouseCaptureDoesNotRequireSecondViewportClick) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        gui::guiFocusState().want_capture_mouse = true;

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 40.0, 50.0);
        router.endMouseButton(input::ACTION_PRESS);

        EXPECT_TRUE(controller.hasViewportKeyboardFocus());
        EXPECT_TRUE(controller.isContinuousInputActive());
    }

    TEST_F(InputControllerFocusTest, MissedMouseReleaseClearsPointerCapture) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);

        EXPECT_EQ(router.state().pointer_capture, input::InputTarget::Viewport);
        EXPECT_EQ(router.pointerTarget(2500.0, 2500.0), input::InputTarget::Viewport);

        router.syncPressedMouseButtons(false);

        EXPECT_EQ(router.state().pointer_capture, input::InputTarget::None);
        EXPECT_EQ(router.pointerTarget(2500.0, 2500.0), input::InputTarget::None);
    }

    TEST_F(InputControllerFocusTest, HoverTargetIgnoresPointerCapture) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);

        EXPECT_EQ(router.pointerTarget(2500.0, 2500.0), input::InputTarget::Viewport);
        EXPECT_EQ(router.hoverTarget(2500.0, 2500.0), input::InputTarget::None);
    }

    TEST_F(InputControllerFocusTest, SplitToggleClearsActiveCameraDrag) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 40.0, 50.0);

        ASSERT_TRUE(controller.isContinuousInputActive());

        core::events::cmd::ToggleSplitView{}.emit();

        EXPECT_FALSE(controller.isContinuousInputActive());
    }

    TEST_F(InputControllerFocusTest, FpvModeUsesInPlaceLookForPrimaryCameraDrag) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = glm::mat3(1.0f);
        controller.setCameraNavigationMode(InputController::CameraNavigationMode::FPV);

        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 40.0, 50.0);
        controller.handleMouseMove(40.0, 0.0);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 40.0, 0.0);

        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        const glm::vec3 to_pivot = glm::normalize(viewport.camera.getPivot() - viewport.camera.t);
        EXPECT_NEAR(viewport.camera.t.x, 0.0f, 1e-5f);
        EXPECT_NEAR(viewport.camera.t.y, 0.0f, 1e-5f);
        EXPECT_NEAR(viewport.camera.t.z, 5.0f, 1e-5f);
        EXPECT_GT(forward.y, 0.0f);
        EXPECT_NEAR(glm::dot(forward, to_pivot), 1.0f, 1e-4f);
    }

    TEST_F(InputControllerFocusTest, TrackballModeAllowsPerfectTopView) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
            viewport.camera.t, viewport.camera.getPivot());
        controller.setCameraNavigationMode(InputController::CameraNavigationMode::Trackball);

        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 40.0, 50.0);
        controller.handleMouseMove(40.0, 50.0 - glm::half_pi<float>() / 0.002f);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 40.0, 50.0 - glm::half_pi<float>() / 0.002f);

        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        EXPECT_NEAR(glm::length(viewport.camera.getPivot() - viewport.camera.t), 5.0f, 1e-3f);
        EXPECT_NEAR(std::abs(forward.y), 1.0f, 1e-3f);
    }

    TEST_F(InputControllerFocusTest, TrackballModeCrossesTopViewWithoutLocking) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
            viewport.camera.t, viewport.camera.getPivot());
        controller.setCameraNavigationMode(InputController::CameraNavigationMode::Trackball);

        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 40.0, 50.0);
        controller.handleMouseMove(40.0, 50.0 - glm::half_pi<float>() / 0.002f);
        controller.handleMouseMove(40.0, 50.0 - (glm::half_pi<float>() + glm::quarter_pi<float>()) / 0.002f);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 40.0,
                                     50.0 - (glm::half_pi<float>() + glm::quarter_pi<float>()) / 0.002f);

        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        EXPECT_NEAR(glm::length(viewport.camera.getPivot() - viewport.camera.t), 5.0f, 1e-3f);
        EXPECT_LT(std::abs(forward.y), 0.9f);
        EXPECT_GT(std::abs(forward.z), 0.5f);
        EXPECT_NEAR(glm::dot(forward, glm::normalize(viewport.camera.getPivot() - viewport.camera.t)), 1.0f, 1e-5f);
    }

    TEST_F(InputControllerFocusTest, TrackballModeOrbitsAwayFromTopViewWithoutRollSnapping) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 5.0f, 0.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
            viewport.camera.t, viewport.camera.getPivot());
        controller.setCameraNavigationMode(InputController::CameraNavigationMode::Trackball);

        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 100.0, 100.0);

        glm::vec3 prev_right = lfs::rendering::cameraRight(viewport.camera.R);
        float min_right_continuity = 1.0f;
        constexpr int kSteps = 200;
        constexpr double kStepPixels = 10.0;
        for (int step = 1; step <= kSteps; ++step) {
            controller.handleMouseMove(100.0 + step * kStepPixels, 100.0);
            const glm::vec3 right = lfs::rendering::cameraRight(viewport.camera.R);
            min_right_continuity = std::min(min_right_continuity, glm::dot(prev_right, right));
            prev_right = right;
        }
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 100.0 + kSteps * kStepPixels, 100.0);

        constexpr glm::vec3 world_up(0.0f, 1.0f, 0.0f);
        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        const glm::vec3 level_right = glm::normalize(glm::cross(forward, world_up));
        const glm::vec3 level_up = glm::normalize(glm::cross(-forward, level_right));
        const glm::vec3 actual_up = lfs::rendering::cameraUp(viewport.camera.R);

        EXPECT_NEAR(glm::length(viewport.camera.getPivot() - viewport.camera.t), 5.0f, 1e-3f);
        EXPECT_GT(min_right_continuity, 0.99f);
        EXPECT_LT(std::abs(forward.y), 0.9f);
        EXPECT_GT(std::abs(glm::dot(actual_up, level_up)), 0.999f);
        EXPECT_NEAR(glm::dot(forward, glm::normalize(viewport.camera.getPivot() - viewport.camera.t)), 1.0f, 1e-5f);
    }

    TEST_F(InputControllerFocusTest, TrackballModeDoesNotBankOnDiagonalOrbitDrag) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
            viewport.camera.t, viewport.camera.getPivot());
        controller.setCameraNavigationMode(InputController::CameraNavigationMode::Trackball);

        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 100.0, 100.0);
        controller.handleMouseMove(300.0, 220.0);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 300.0, 220.0);

        constexpr glm::vec3 world_up(0.0f, 1.0f, 0.0f);
        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        const glm::vec3 right = glm::normalize(glm::cross(forward, world_up));
        const glm::vec3 expected_up = glm::normalize(glm::cross(-forward, right));
        const glm::vec3 actual_up = lfs::rendering::cameraUp(viewport.camera.R);

        EXPECT_NEAR(glm::length(viewport.camera.getPivot() - viewport.camera.t), 5.0f, 1e-4f);
        EXPECT_NEAR(glm::dot(forward, glm::normalize(viewport.camera.getPivot() - viewport.camera.t)), 1.0f, 1e-5f);
        EXPECT_GT(glm::dot(actual_up, expected_up), 0.9999f);
    }

    TEST_F(InputControllerFocusTest, TrackballSnapAlignsToNearestAxisView) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
            viewport.camera.t, viewport.camera.getPivot());
        controller.setCameraNavigationMode(InputController::CameraNavigationMode::Trackball);
        controller.setCameraViewSnapEnabled(true);

        const double snap_target_y = 50.0 - glm::radians(85.0f) / 0.002f;
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 40.0, 50.0);
        controller.handleMouseMove(40.0, snap_target_y);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 40.0, snap_target_y);

        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        EXPECT_NEAR(std::abs(forward.y), 1.0f, 1e-4f);
        EXPECT_NEAR(std::abs(forward.x), 0.0f, 1e-4f);
        EXPECT_NEAR(std::abs(forward.z), 0.0f, 1e-4f);
    }

    TEST_F(InputControllerFocusTest, FpvModePitchClampPreventsPoleFlip) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = glm::mat3(1.0f);
        controller.setCameraNavigationMode(InputController::CameraNavigationMode::FPV);

        const double y0 = 900.0;
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 40.0, y0);
        controller.handleMouseMove(40.0, y0 - glm::radians(140.0f) / 0.001f);
        controller.handleMouseMove(40.0, y0 - glm::radians(200.0f) / 0.001f);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 40.0, y0 - glm::radians(200.0f) / 0.001f);

        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        const glm::vec3 up = lfs::rendering::cameraUp(viewport.camera.R);
        ASSERT_TRUE(std::isfinite(forward.x) && std::isfinite(forward.y) && std::isfinite(forward.z));
        EXPECT_GT(forward.y, 0.99f);
        EXPECT_LE(forward.y, 0.9999f);
        EXPECT_GT(up.y, 0.0f);
    }

    TEST_F(InputControllerFocusTest, OrbitModeFirstDragFromTopViewDoesNotJump) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.setAxisAlignedView(1, false);
        const glm::vec3 forward_before = lfs::rendering::cameraForward(viewport.camera.R);

        const glm::vec3 right_before = lfs::rendering::cameraRight(viewport.camera.R);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 100.0, 100.0);
        controller.handleMouseMove(100.0, 90.0);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 100.0, 90.0);

        const glm::vec3 forward_after = lfs::rendering::cameraForward(viewport.camera.R);
        const float angle = std::acos(glm::clamp(glm::dot(forward_before, forward_after), -1.0f, 1.0f));
        EXPECT_GT(angle, glm::radians(0.5f));
        EXPECT_LT(angle, glm::radians(3.0f));
        EXPECT_GT(glm::dot(right_before, lfs::rendering::cameraRight(viewport.camera.R)), 0.9f);
    }

    TEST_F(InputControllerFocusTest, OrbitModePitchIntoLimitDoesNotFlip) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
            viewport.camera.t, viewport.camera.getPivot());

        const double y0 = 900.0;
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 100.0, y0);

        glm::vec3 prev_right = lfs::rendering::cameraRight(viewport.camera.R);
        float min_right_continuity = 1.0f;
        constexpr int kSteps = 60;
        constexpr double kStepPixels = 30.0;
        for (int step = 1; step <= kSteps; ++step) {
            controller.handleMouseMove(100.0, y0 - step * kStepPixels);
            const glm::vec3 right = lfs::rendering::cameraRight(viewport.camera.R);
            min_right_continuity = std::min(min_right_continuity, glm::dot(prev_right, right));
            prev_right = right;
        }
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 100.0, y0 - kSteps * kStepPixels);

        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        const glm::vec3 up = lfs::rendering::cameraUp(viewport.camera.R);
        EXPECT_GT(min_right_continuity, 0.9f);
        EXPECT_GT(std::abs(forward.y), 0.999f);
        EXPECT_LE(std::abs(forward.y), 0.99995f);
        EXPECT_GE(up.y, 0.0f);
    }

    TEST_F(InputControllerFocusTest, OrbitModeReachesNearTopDownView) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
            viewport.camera.t, viewport.camera.getPivot());

        const double y0 = 900.0;
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 100.0, y0);
        controller.handleMouseMove(100.0, y0 - glm::radians(85.0f) / 0.002f);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 100.0, y0 - glm::radians(85.0f) / 0.002f);

        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        EXPECT_GT(std::abs(forward.y), 0.99f);
        EXPECT_LE(std::abs(forward.y), 0.99995f);
    }

    TEST_F(InputControllerFocusTest, ZoomInPushesPivotInsteadOfDeadlocking) {
        Viewport viewport(200, 200);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
            viewport.camera.t, viewport.camera.getPivot());

        const glm::vec3 start = viewport.camera.t;
        float min_pivot_distance = std::numeric_limits<float>::max();
        for (int i = 0; i < 500; ++i) {
            viewport.camera.zoom(1.0f);
            const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
            min_pivot_distance = std::min(
                min_pivot_distance,
                glm::dot(viewport.camera.getPivot() - viewport.camera.t, forward));
        }
        EXPECT_GE(min_pivot_distance, 0.0999f);
        EXPECT_GT(glm::distance(start, viewport.camera.t), 5.0f);
    }

    TEST_F(InputControllerFocusTest, ZoomWithCarryPivotKeepsDistance) {
        Viewport viewport(200, 200);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
            viewport.camera.t, viewport.camera.getPivot());

        const glm::vec3 start = viewport.camera.t;
        for (int i = 0; i < 100; ++i) {
            viewport.camera.zoom(1.0f, true);
        }
        EXPECT_NEAR(glm::length(viewport.camera.getPivot() - viewport.camera.t), 5.0f, 1e-3f);
        EXPECT_GT(glm::distance(start, viewport.camera.t), 20.0f);
    }

    TEST_F(InputControllerFocusTest, CameraRollDirectionMatchesRotationSign) {
        Viewport viewport(200, 200);
        viewport.camera.R = glm::mat3(1.0f);
        viewport.camera.rotate_roll(10.0f);

        const glm::vec3 right = viewport.camera.R[0];
        EXPECT_NEAR(right.x, std::cos(0.1f), 1e-4f);
        EXPECT_NEAR(right.y, std::sin(0.1f), 1e-4f);
    }

    TEST_F(InputControllerFocusTest, TrackballOrbitPreservesDeliberateRoll) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
            viewport.camera.t, viewport.camera.getPivot());
        controller.setCameraNavigationMode(InputController::CameraNavigationMode::Trackball);

        viewport.camera.rotate_roll(50.0f);

        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 100.0, 100.0);
        for (int step = 1; step <= 10; ++step) {
            controller.handleMouseMove(100.0 + step * 20.0, 100.0);
        }
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 300.0, 100.0);

        constexpr glm::vec3 world_up(0.0f, 1.0f, 0.0f);
        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        const glm::vec3 level_right = glm::normalize(glm::cross(forward, world_up));
        EXPECT_NEAR(std::abs(glm::dot(viewport.camera.R[0], level_right)), std::cos(0.5f), 0.03f);
    }

    TEST_F(InputControllerFocusTest, NavigationModeSwitchPreservesPivot) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
            viewport.camera.t, glm::vec3(0.0f));
        viewport.camera.setPivot(glm::vec3(1.0f, 2.0f, 3.0f));

        controller.setCameraNavigationMode(InputController::CameraNavigationMode::Trackball);

        const glm::vec3 pivot = viewport.camera.getPivot();
        EXPECT_NEAR(pivot.x, 1.0f, 1e-6f);
        EXPECT_NEAR(pivot.y, 2.0f, 1e-6f);
        EXPECT_NEAR(pivot.z, 3.0f, 1e-6f);
    }

    TEST_F(InputControllerFocusTest, GlideEasesToTargetAndFinishSnaps) {
        Viewport viewport(200, 200);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);

        const glm::vec3 target(3.0f, 0.0f, 5.0f);
        viewport.camera.startGlide(target);
        ASSERT_TRUE(viewport.camera.isGliding());
        viewport.camera.updateGlide(1.0f / 60.0f);
        EXPECT_GT(glm::distance(viewport.camera.t, target), 1e-3f);
        for (int i = 0; i < 60; ++i) {
            viewport.camera.updateGlide(1.0f / 60.0f);
        }
        EXPECT_FALSE(viewport.camera.isGliding());
        EXPECT_NEAR(glm::distance(viewport.camera.t, target), 0.0f, 1e-4f);

        viewport.camera.startGlide(glm::vec3(0.0f));
        viewport.camera.finishGlide();
        EXPECT_FALSE(viewport.camera.isGliding());
        EXPECT_NEAR(glm::length(viewport.camera.t), 0.0f, 1e-6f);
    }

    TEST_F(InputControllerFocusTest, SetPivotOnBackgroundKeepsOrbitRadius) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 7.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
            viewport.camera.t, viewport.camera.getPivot());

        const glm::vec3 t_before = viewport.camera.t;
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::RIGHT),
                                     input::ACTION_PRESS, 100.0, 100.0);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::RIGHT),
                                     input::ACTION_RELEASE, 100.0, 100.0);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::RIGHT),
                                     input::ACTION_PRESS, 100.0, 100.0);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::RIGHT),
                                     input::ACTION_RELEASE, 100.0, 100.0);

        EXPECT_NEAR(glm::length(viewport.camera.getPivot() - viewport.camera.t), 7.0f, 1e-3f);
        controller.update(1.0f);
        EXPECT_NEAR(glm::distance(viewport.camera.t, t_before), 0.0f, 1e-3f);
    }

    TEST_F(InputControllerFocusTest, CameraSpeedAdjustIsMultiplicative) {
        Viewport viewport(200, 200);
        const float wasd_before = viewport.camera.getWasdSpeed();
        viewport.camera.increaseWasdSpeed();
        EXPECT_NEAR(viewport.camera.getWasdSpeed(), wasd_before * 1.2f, 1e-4f);
        viewport.camera.decreaseWasdSpeed();
        EXPECT_NEAR(viewport.camera.getWasdSpeed(), wasd_before, 1e-4f);

        const float zoom_before = viewport.camera.getZoomSpeed();
        viewport.camera.increaseZoomSpeed();
        EXPECT_NEAR(viewport.camera.getZoomSpeed(), zoom_before * 1.2f, 1e-4f);
        viewport.camera.decreaseZoomSpeed();
        EXPECT_NEAR(viewport.camera.getZoomSpeed(), zoom_before, 1e-4f);
    }

    TEST_F(InputControllerFocusTest, AxisAlignedViewPreservesPivotAndDistance) {
        Viewport viewport(200, 200);
        const glm::vec3 pivot(12.0f, 1.5f, -7.0f);
        viewport.camera.t = glm::vec3(9.0f, 4.0f, -2.0f);
        viewport.camera.setPivot(pivot);
        viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
            viewport.camera.t, viewport.camera.getPivot());

        const float initial_distance = glm::length(viewport.camera.getPivot() - viewport.camera.t);
        viewport.camera.setAxisAlignedView(1, false);

        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        EXPECT_NEAR(glm::length(viewport.camera.getPivot() - viewport.camera.t), initial_distance, 1e-4f);
        EXPECT_NEAR(glm::distance(viewport.camera.getPivot(), pivot), 0.0f, 1e-6f);
        EXPECT_NEAR(std::abs(forward.y), 1.0f, 1e-4f);
        EXPECT_NEAR(std::abs(forward.x), 0.0f, 1e-4f);
        EXPECT_NEAR(std::abs(forward.z), 0.0f, 1e-4f);
    }

    TEST_F(InputControllerFocusTest, PointerTargetsExposeHoverAndCapturedTargets) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);

        const auto targets = router.pointerTargets(2500.0, 2500.0);
        EXPECT_EQ(targets.pointer_target, input::InputTarget::Viewport);
        EXPECT_EQ(targets.hover_target, input::InputTarget::None);
    }

    TEST_F(InputControllerFocusTest, NavigationMouseCaptureFinalizesToSingleClickBinding) {
        input::InputBindings bindings;
        bindings.startCapture(input::ToolMode::GLOBAL, input::Action::CAMERA_ORBIT);
        bindings.captureMouseButton(static_cast<int>(input::MouseButton::RIGHT), input::MODIFIER_NONE);

        auto& capture_state = const_cast<input::CaptureState&>(bindings.getCaptureState());
        capture_state.first_click_time -= std::chrono::milliseconds(500);
        bindings.updateCapture();

        const auto captured = bindings.getAndClearCaptured();
        ASSERT_TRUE(captured.has_value());

        const auto* mouse_trigger = std::get_if<input::MouseButtonTrigger>(&*captured);
        ASSERT_NE(mouse_trigger, nullptr);
        EXPECT_EQ(mouse_trigger->button, input::MouseButton::RIGHT);
        EXPECT_EQ(mouse_trigger->modifiers, input::MODIFIER_NONE);
        EXPECT_FALSE(mouse_trigger->double_click);
        EXPECT_FALSE(bindings.isCapturing());
    }

    TEST_F(InputControllerFocusTest, NavigationMouseCaptureDragMovementFinalizesToDragBinding) {
        input::InputBindings bindings;
        bindings.startCapture(input::ToolMode::GLOBAL, input::Action::CAMERA_ORBIT);
        bindings.captureMouseButton(static_cast<int>(input::MouseButton::RIGHT),
                                    input::MODIFIER_NONE,
                                    40.0,
                                    50.0);

        bindings.captureMouseMove(40.0 + input::CaptureState::DRAG_CAPTURE_THRESHOLD_PX + 1.0, 50.0);

        const auto captured = bindings.getAndClearCaptured();
        ASSERT_TRUE(captured.has_value());

        const auto* drag_trigger = std::get_if<input::MouseDragTrigger>(&*captured);
        ASSERT_NE(drag_trigger, nullptr);
        EXPECT_EQ(drag_trigger->button, input::MouseButton::RIGHT);
        EXPECT_EQ(drag_trigger->modifiers, input::MODIFIER_NONE);
        EXPECT_FALSE(bindings.isCapturing());
    }

    TEST_F(InputControllerFocusTest, NavigationMouseCaptureHeldButtonWaitsForDragBeforeSingleClick) {
        input::InputBindings bindings;
        bindings.startCapture(input::ToolMode::GLOBAL, input::Action::CAMERA_ORBIT);
        bindings.captureMouseButton(static_cast<int>(input::MouseButton::RIGHT),
                                    input::MODIFIER_NONE,
                                    40.0,
                                    50.0);

        auto& capture_state = const_cast<input::CaptureState&>(bindings.getCaptureState());
        capture_state.first_click_time -= std::chrono::milliseconds(500);
        bindings.updateCapture();
        EXPECT_TRUE(bindings.isCapturing());
        EXPECT_FALSE(bindings.getAndClearCaptured().has_value());

        bindings.captureMouseMove(40.0 + input::CaptureState::DRAG_CAPTURE_THRESHOLD_PX + 1.0, 50.0);

        const auto captured = bindings.getAndClearCaptured();
        ASSERT_TRUE(captured.has_value());
        EXPECT_NE(std::get_if<input::MouseDragTrigger>(&*captured), nullptr);
        EXPECT_FALSE(bindings.isCapturing());
    }

    TEST_F(InputControllerFocusTest, NavigationMouseCaptureReleaseBeforeMoveKeepsSingleClickBinding) {
        input::InputBindings bindings;
        bindings.startCapture(input::ToolMode::GLOBAL, input::Action::CAMERA_ORBIT);
        bindings.captureMouseButton(static_cast<int>(input::MouseButton::RIGHT),
                                    input::MODIFIER_NONE,
                                    40.0,
                                    50.0);
        bindings.captureMouseButtonRelease(static_cast<int>(input::MouseButton::RIGHT));
        bindings.captureMouseMove(80.0, 50.0);

        auto& capture_state = const_cast<input::CaptureState&>(bindings.getCaptureState());
        capture_state.first_click_time -= std::chrono::milliseconds(500);
        bindings.updateCapture();

        const auto captured = bindings.getAndClearCaptured();
        ASSERT_TRUE(captured.has_value());

        const auto* mouse_trigger = std::get_if<input::MouseButtonTrigger>(&*captured);
        ASSERT_NE(mouse_trigger, nullptr);
        EXPECT_EQ(mouse_trigger->button, input::MouseButton::RIGHT);
        EXPECT_FALSE(mouse_trigger->double_click);
        EXPECT_FALSE(bindings.isCapturing());
    }

    TEST_F(InputControllerFocusTest, SelectionMouseCaptureStillFinalizesToDragBinding) {
        input::InputBindings bindings;
        bindings.startCapture(input::ToolMode::SELECTION, input::Action::SELECTION_REPLACE);
        bindings.captureMouseButton(static_cast<int>(input::MouseButton::LEFT), input::MODIFIER_NONE);

        auto& capture_state = const_cast<input::CaptureState&>(bindings.getCaptureState());
        capture_state.first_click_time -= std::chrono::milliseconds(500);
        bindings.updateCapture();

        const auto captured = bindings.getAndClearCaptured();
        ASSERT_TRUE(captured.has_value());

        const auto* drag_trigger = std::get_if<input::MouseDragTrigger>(&*captured);
        ASSERT_NE(drag_trigger, nullptr);
        EXPECT_EQ(drag_trigger->button, input::MouseButton::LEFT);
        EXPECT_EQ(drag_trigger->modifiers, input::MODIFIER_NONE);
        EXPECT_FALSE(bindings.isCapturing());
    }

    TEST_F(InputControllerFocusTest, ReloadingCurrentProfileDoesNotRestoreClearedZoomBinding) {
        const auto profile_path = std::filesystem::temp_directory_path() / "lfs_input_bindings_no_zoom.json";
        std::filesystem::remove(profile_path);

        input::InputBindings bindings;
        bindings.clearBinding(input::ToolMode::GLOBAL, input::Action::CAMERA_ZOOM);
        ASSERT_TRUE(bindings.saveProfileToFile(profile_path));

        input::InputBindings loaded;
        ASSERT_TRUE(loaded.loadProfileFromFile(profile_path));

        EXPECT_EQ(loaded.getActionForScroll(input::ToolMode::GLOBAL, input::MODIFIER_NONE),
                  input::Action::NONE);

        std::filesystem::remove(profile_path);
    }

    TEST_F(InputControllerFocusTest, HistogramZoomMarkedDefaultsToCtrlScroll) {
        input::InputBindings bindings;

        EXPECT_EQ(bindings.getActionForScroll(input::ToolMode::GLOBAL, input::MODIFIER_CTRL),
                  input::Action::HISTOGRAM_ZOOM_MARKED);

        const auto trigger = bindings.getTriggerForAction(input::Action::HISTOGRAM_ZOOM_MARKED,
                                                          input::ToolMode::GLOBAL);
        ASSERT_TRUE(trigger.has_value());
        const auto* scroll_trigger = std::get_if<input::MouseScrollTrigger>(&*trigger);
        ASSERT_NE(scroll_trigger, nullptr);
        EXPECT_EQ(scroll_trigger->modifiers, input::MODIFIER_CTRL);
        EXPECT_FALSE(scroll_trigger->chord_key.has_value());
        EXPECT_TRUE(input::describe(input::Action::HISTOGRAM_ZOOM_MARKED).allowed_kinds &
                    input::TRIGGER_KIND_MOUSE_SCROLL);
    }

    TEST_F(InputControllerFocusTest, CameraFrustumsDefaultToAltCAndToggleRenderSetting) {
        RenderingManager rendering_manager;
        services().set(&rendering_manager);

        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);
        router.focusViewportKeyboard();

        EXPECT_EQ(controller.getBindings().getActionForKey(input::ToolMode::GLOBAL,
                                                           input::KEY_C,
                                                           input::MODIFIER_ALT),
                  input::Action::TOGGLE_CAMERA_FRUSTUMS);
        EXPECT_EQ(input::shortcutScopeForAction(input::Action::TOGGLE_CAMERA_FRUSTUMS),
                  input::ShortcutScope::Viewport);

        EXPECT_FALSE(rendering_manager.getSettings().show_camera_frustums);
        controller.handleKey(input::KEY_C, input::ACTION_PRESS, input::KEYMOD_ALT);
        EXPECT_TRUE(rendering_manager.getSettings().show_camera_frustums);
        controller.handleKey(input::KEY_C, input::ACTION_PRESS, input::KEYMOD_ALT);
        EXPECT_FALSE(rendering_manager.getSettings().show_camera_frustums);
    }

    TEST_F(InputControllerFocusTest, VersionFifteenProfileMigratesCameraFrustumShortcutWhenFree) {
        const auto profile_path = std::filesystem::temp_directory_path() / "lfs_input_bindings_legacy_v15.json";
        std::filesystem::remove(profile_path);
        {
            std::ofstream file(profile_path);
            ASSERT_TRUE(file.is_open());
            file << R"({
  "name": "LegacyV15",
  "version": 15,
  "bindings": []
})";
        }

        input::InputBindings loaded;
        ASSERT_TRUE(loaded.loadProfileFromFile(profile_path));
        EXPECT_EQ(loaded.getActionForKey(input::ToolMode::GLOBAL,
                                         input::KEY_C,
                                         input::MODIFIER_ALT),
                  input::Action::TOGGLE_CAMERA_FRUSTUMS);

        std::filesystem::remove(profile_path);
    }

    TEST_F(InputControllerFocusTest, VersionFifteenProfilePreservesOccupiedAltC) {
        const auto profile_path = std::filesystem::temp_directory_path() / "lfs_input_bindings_legacy_v15_alt_c.json";
        std::filesystem::remove(profile_path);
        {
            std::ofstream file(profile_path);
            ASSERT_TRUE(file.is_open());
            file << R"({
  "name": "LegacyV15AltC",
  "version": 15,
  "bindings": [
    {
      "mode": 0,
      "action": 25,
      "description": "Cycle PLY",
      "trigger_type": "key",
      "key": 67,
      "modifiers": 4
    }
  ]
})";
        }

        input::InputBindings loaded;
        ASSERT_TRUE(loaded.loadProfileFromFile(profile_path));
        EXPECT_EQ(loaded.getActionForKey(input::ToolMode::GLOBAL,
                                         input::KEY_C,
                                         input::MODIFIER_ALT),
                  input::Action::CYCLE_PLY);
        EXPECT_FALSE(loaded.getTriggerForAction(input::Action::TOGGLE_CAMERA_FRUSTUMS,
                                                input::ToolMode::GLOBAL)
                         .has_value());

        std::filesystem::remove(profile_path);
    }

    TEST_F(InputControllerFocusTest, CropApplyDefaultsToEnterAndNumEnter) {
        input::InputBindings bindings;

        EXPECT_EQ(bindings.getActionForKey(input::ToolMode::CROP_BOX,
                                           input::KEY_ENTER,
                                           input::MODIFIER_NONE),
                  input::Action::APPLY_CROP_BOX);
        EXPECT_EQ(bindings.getActionForKey(input::ToolMode::CROP_BOX,
                                           input::KEY_KP_ENTER,
                                           input::MODIFIER_NONE),
                  input::Action::APPLY_CROP_BOX);
    }

    TEST_F(InputControllerFocusTest, VersionFourteenProfileMigratesCropApplyEnterBindings) {
        const auto profile_path = std::filesystem::temp_directory_path() / "lfs_input_bindings_legacy_v14.json";
        std::filesystem::remove(profile_path);
        {
            std::ofstream file(profile_path);
            ASSERT_TRUE(file.is_open());
            file << R"({
  "name": "LegacyV14",
  "version": 14,
  "bindings": [
    {
      "mode": 0,
      "action": 71,
      "description": "Zoom Histogram at Cursor",
      "trigger_type": "scroll",
      "modifiers": 2
    }
  ]
})";
        }

        input::InputBindings loaded;
        ASSERT_TRUE(loaded.loadProfileFromFile(profile_path));

        EXPECT_EQ(loaded.getActionForKey(input::ToolMode::CROP_BOX,
                                         input::KEY_ENTER,
                                         input::MODIFIER_NONE),
                  input::Action::APPLY_CROP_BOX);
        EXPECT_EQ(loaded.getActionForKey(input::ToolMode::CROP_BOX,
                                         input::KEY_KP_ENTER,
                                         input::MODIFIER_NONE),
                  input::Action::APPLY_CROP_BOX);

        std::filesystem::remove(profile_path);
    }

    TEST_F(InputControllerFocusTest, ToolControlActivationShortcutsResolveAcrossModesAtRuntime) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        controller.getBindings().setBinding(input::ToolMode::SELECTION,
                                            input::Action::TOOL_MIRROR,
                                            input::KeyTrigger{input::KEY_M, input::MODIFIER_CTRL});

        lfs::event::ScopedHandler handlers;
        int tool_mode = -1;
        handlers.subscribe<core::events::tools::SetToolbarTool>(
            [&](const auto& event) { tool_mode = event.tool_mode; });

        controller.handleKey(input::KEY_M, input::ACTION_PRESS, input::MODIFIER_CTRL);

        EXPECT_EQ(tool_mode, static_cast<int>(ToolType::Mirror));
        EXPECT_TRUE(controller.getBindings()
                        .getTriggerForAction(input::Action::TOOL_MIRROR,
                                             input::ToolMode::SELECTION)
                        .has_value());
        EXPECT_FALSE(controller.getBindings()
                         .getTriggerForAction(input::Action::TOOL_MIRROR,
                                              input::ToolMode::GLOBAL)
                         .has_value());
    }

    TEST_F(InputControllerFocusTest, ToolLocalOperationalShortcutsDoNotResolveAcrossModes) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        controller.getBindings().setBinding(input::ToolMode::SELECTION,
                                            input::Action::DELETE_SELECTED,
                                            input::KeyTrigger{input::KEY_B, input::MODIFIER_CTRL});

        lfs::event::ScopedHandler handlers;
        int delete_count = 0;
        handlers.subscribe<core::events::cmd::DeleteSelected>(
            [&](const auto&) { ++delete_count; });

        controller.handleKey(input::KEY_B, input::ACTION_PRESS, input::MODIFIER_CTRL);

        EXPECT_EQ(delete_count, 0);
    }

    TEST_F(InputControllerFocusTest, CutSelectionDefaultShortcutDispatchesCommand) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        EXPECT_EQ(controller.getBindings().getActionForKey(
                      input::ToolMode::GLOBAL,
                      input::KEY_X,
                      input::MODIFIER_CTRL),
                  input::Action::CUT_SELECTION);

        lfs::event::ScopedHandler handlers;
        int cut_count = 0;
        handlers.subscribe<core::events::cmd::CutSelection>(
            [&](const auto&) { ++cut_count; });

        controller.handleKey(input::KEY_X, input::ACTION_PRESS, input::MODIFIER_CTRL);

        EXPECT_EQ(cut_count, 1);
    }

    TEST_F(InputControllerFocusTest, LegacyProfileMigrationAddsOnlyVersionedModalDefaults) {
        const auto profile_path = std::filesystem::temp_directory_path() / "lfs_input_bindings_legacy_v5.json";
        std::filesystem::remove(profile_path);
        {
            std::ofstream file(profile_path);
            ASSERT_TRUE(file.is_open());
            file << R"({
  "name": "Legacy",
  "version": 5,
  "bindings": [
    {
      "mode": 0,
      "action": 3,
      "description": "Zoom",
      "trigger_type": "scroll",
      "modifiers": 0
    }
  ]
})";
        }

        input::InputBindings loaded;
        ASSERT_TRUE(loaded.loadProfileFromFile(profile_path));

        EXPECT_EQ(loaded.getActionForScroll(input::ToolMode::GLOBAL, input::MODIFIER_NONE,
                                            std::vector<int>{input::KEY_R}),
                  input::Action::CAMERA_ROLL);
        EXPECT_EQ(loaded.getActionForMouseButton(input::ToolMode::GLOBAL,
                                                 input::MouseButton::RIGHT,
                                                 input::MODIFIER_NONE),
                  input::Action::NONE);
        EXPECT_EQ(loaded.getActionForMouseButton(input::ToolMode::SELECTION,
                                                 input::MouseButton::RIGHT,
                                                 input::MODIFIER_NONE),
                  input::Action::UNDO_POLYGON_VERTEX);

        std::filesystem::remove(profile_path);
    }

    TEST_F(InputControllerFocusTest, VersionSevenProfileCollapsesCopiedGlobalDefaults) {
        const auto profile_path = std::filesystem::temp_directory_path() / "lfs_input_bindings_legacy_v7.json";
        std::filesystem::remove(profile_path);
        {
            std::ofstream file(profile_path);
            ASSERT_TRUE(file.is_open());
            file << R"({
  "name": "CopiedDefaults",
  "version": 7,
  "bindings": [
    {
      "mode": 0,
      "action": 2,
      "description": "Custom global pan",
      "trigger_type": "drag",
      "button": 2,
      "modifiers": 0
    },
    {
      "mode": 1,
      "action": 2,
      "description": "Stale copied pan",
      "trigger_type": "drag",
      "button": 1,
      "modifiers": 0
    },
    {
      "mode": 3,
      "action": 44,
      "description": "Stale transform selection",
      "trigger_type": "drag",
      "button": 0,
      "modifiers": 0
    },
    {
      "mode": 1,
      "action": 44,
      "description": "Selection",
      "trigger_type": "drag",
      "button": 0,
      "modifiers": 0
    }
  ]
})";
        }

        input::InputBindings loaded;
        ASSERT_TRUE(loaded.loadProfileFromFile(profile_path));

        EXPECT_FALSE(loaded.getTriggerForAction(
                               input::Action::CAMERA_PAN, input::ToolMode::SELECTION)
                         .has_value());
        EXPECT_EQ(loaded.getActionForDrag(input::ToolMode::SELECTION,
                                          input::MouseButton::MIDDLE,
                                          input::MODIFIER_NONE),
                  input::Action::CAMERA_PAN);
        EXPECT_NE(loaded.getActionForDrag(input::ToolMode::TRANSLATE,
                                          input::MouseButton::LEFT,
                                          input::MODIFIER_NONE),
                  input::Action::SELECTION_REPLACE);

        std::filesystem::remove(profile_path);
    }

    TEST_F(InputControllerFocusTest, PolygonConfirmBindingAcceptsSelectionModifiers) {
        input::InputBindings bindings;

        EXPECT_EQ(bindings.getActionForKey(input::ToolMode::SELECTION,
                                           input::KEY_ENTER,
                                           input::MODIFIER_NONE),
                  input::Action::CONFIRM_POLYGON);
        EXPECT_EQ(bindings.getActionForKey(input::ToolMode::SELECTION,
                                           input::KEY_ENTER,
                                           input::MODIFIER_SHIFT),
                  input::Action::CONFIRM_POLYGON);
        EXPECT_EQ(bindings.getActionForKey(input::ToolMode::SELECTION,
                                           input::KEY_ENTER,
                                           input::MODIFIER_CTRL),
                  input::Action::CONFIRM_POLYGON);
        EXPECT_EQ(bindings.getActionForMouseButton(input::ToolMode::SELECTION,
                                                   input::MouseButton::RIGHT,
                                                   input::MODIFIER_NONE),
                  input::Action::UNDO_POLYGON_VERTEX);
    }

    TEST_F(InputControllerFocusTest, ClearedZoomBindingStopsViewportScrollZoom) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        controller.getBindings().clearBinding(input::ToolMode::GLOBAL, input::Action::CAMERA_ZOOM);

        const glm::vec3 start_t = viewport.camera.t;
        const glm::mat3 start_r = viewport.camera.R;

        controller.handleScroll(0.0, 1.0);

        EXPECT_NEAR(glm::distance(viewport.camera.t, start_t), 0.0f, 1e-6f);
        for (int col = 0; col < 3; ++col) {
            EXPECT_NEAR(glm::distance(viewport.camera.R[col], start_r[col]), 0.0f, 1e-6f);
        }
    }

} // namespace lfs::vis
