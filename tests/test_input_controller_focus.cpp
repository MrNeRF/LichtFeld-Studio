/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/editor_context.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/scoped_handler.hpp"
#include "core/events.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/user_paths.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/rml_menu_bar.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "input/camera_animation_cadence.hpp"
#include "input/frame_input_buffer.hpp"
#include "input/input_controller.hpp"
#include "input/input_router.hpp"
#include "input/key_codes.hpp"
#include "input/sdl_key_mapping.hpp"
#include "internal/viewport.hpp"
#include "keymap_history_helpers.hpp"
#include "licht_test_support.hpp"
#include "operator/operator_registry.hpp"
#include "python/python_runtime.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "tools/tool_base.hpp"
#include "visualizer/visualizer.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#ifdef __linux__
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <glm/gtc/constants.hpp>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace lfs::vis {

    namespace {
        class ScopedEnvironmentVariable {
        public:
            ScopedEnvironmentVariable(const char* name,
                                      const std::optional<std::string>& value)
                : name_(name) {
                if (const char* previous = std::getenv(name))
                    previous_ = previous;
                set(value);
            }

            ~ScopedEnvironmentVariable() { set(previous_); }

        private:
            void set(const std::optional<std::string>& value) const {
#ifdef _WIN32
                (void)_putenv_s(name_.c_str(), value ? value->c_str() : "");
#else
                if (value)
                    (void)setenv(name_.c_str(), value->c_str(), 1);
                else
                    (void)unsetenv(name_.c_str());
#endif
            }

            std::string name_;
            std::optional<std::string> previous_;
        };

        int python_console_toggle_calls = 0;

        void recordPythonConsoleToggle() {
            ++python_console_toggle_calls;
        }

        class RestorePythonConsoleCallback {
        public:
            RestorePythonConsoleCallback() {
                python_console_toggle_calls = 0;
                lfs::python::set_show_python_console_callback(&recordPythonConsoleToggle);
            }
            ~RestorePythonConsoleCallback() {
                lfs::python::set_show_python_console_callback(nullptr);
            }
        };

        class InputControllerFocusTest : public ::testing::Test {
        protected:
            void SetUp() override {
                isolateInputProfileHome();
                input::InputBindings::setPersistenceEnabled(false);
                lfs::event::EventBridge::instance().clear_all();
                services().clear();
                gui::guiFocusState().reset();
            }

            void TearDown() override {
                setRuntimeServiceControls({});
                gui::guiFocusState().reset();
                services().clear();
                lfs::event::EventBridge::instance().clear_all();
                input::InputBindings::setPersistenceEnabled(true);
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

        // Two Gaussians at x +/- 1 around `x`, so framing has a nonzero box.
        std::unique_ptr<lfs::core::SplatData> makeTwoGaussianSplat(const float x) {
            using lfs::core::DataType;
            using lfs::core::Device;
            using lfs::core::Tensor;
            return std::make_unique<lfs::core::SplatData>(
                0,
                Tensor::from_vector({x - 1.0f, 0.0f, 0.0f, x + 1.0f, 0.0f, 0.0f}, {size_t{2}, size_t{3}}, Device::CPU),
                Tensor::zeros({size_t{2}, size_t{1}, size_t{3}}, Device::CPU),
                Tensor::zeros({size_t{2}, size_t{0}, size_t{3}}, Device::CPU, DataType::Float32),
                Tensor::zeros({size_t{2}, size_t{3}}, Device::CPU),
                Tensor::from_vector({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f}, {size_t{2}, size_t{4}}, Device::CPU),
                Tensor::zeros({size_t{2}, size_t{1}}, Device::CPU),
                1.0f);
        }
    } // namespace

    TEST_F(InputControllerFocusTest, WalkRejectionDoesNotEscapeNativeDispatch) {
        SceneManager manager;
        services().set(&manager);
        auto& scene = manager.getScene();
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        controller.initialize();
        controller.setCameraNavigationMode(InputController::CameraNavigationMode::Orbit);
        gui::RmlMenuBar bar;
        bar.rebuildToolbarButtons();
        EXPECT_FALSE(bar.camera_buttons_.back().enabled);
        EXPECT_NO_THROW(bar.dispatchToolbarAction("set_camera_navigation_mode", "walk"));
        EXPECT_EQ(controller.walk_error, LOC("collision.walk_no_collision"));
        const auto floor = scene.addCollision("Floor", collision_test::box({-8, -1, -8}, {16, 1, 16}));
        scene.setNodeTransform(floor, lfs::rendering::VISUALIZER_TO_DATA_WORLD_AXES_4);
        scene.setNodeVisibility(floor, false);
        bar.rebuildToolbarButtons();
        EXPECT_TRUE(bar.camera_buttons_.back().enabled);
        viewport.camera.t = {0, 20, 0};
        EXPECT_NO_THROW(bar.dispatchToolbarAction("set_camera_navigation_mode", "walk"));
        EXPECT_EQ(controller.walk_error, LOC("collision.walk_no_support"));
        EXPECT_EQ(controller.cameraNavigationMode(), InputController::CameraNavigationMode::Orbit);
        EXPECT_EQ(viewport.camera.t, glm::vec3(0, 20, 0));
    }

    TEST_F(InputControllerFocusTest, WalkIdleAndSettingsKeepThePhysicsOwnerAcrossSplitFocusChanges) {
        SceneManager manager;
        services().set(&manager);
        auto& scene = manager.getScene();
        const auto floor = scene.addCollision("Floor", collision_test::box({-8, -1, -8}, {16, 1, 16}));
        scene.setNodeTransform(floor, lfs::rendering::VISUALIZER_TO_DATA_WORLD_AXES_4);
        Viewport primary(200, 200);
        InputController controller(nullptr, primary);
        RenderingManager rendering;
        services().set(&rendering);
        rendering.restoreSplitViewMode(SplitViewMode::IndependentDual, primary);
        auto& secondary = rendering.resolvePanelViewport(primary, SplitViewPanelId::Right);
        primary.camera.t = {-4, 1.7f, 0};
        secondary.camera.t = {4, 1.7f, 0};
        rendering.setFocusedSplitPanel(SplitViewPanelId::Right);
        ASSERT_TRUE(controller.setCameraNavigationMode(InputController::CameraNavigationMode::Walk));
        const auto primary_before = primary.camera.t;
        rendering.setFocusedSplitPanel(SplitViewPanelId::Left);
        controller.update(.016f);
        EXPECT_EQ(primary.camera.t, primary_before);
        EXPECT_NEAR(secondary.camera.t.x, 4, 1e-5f);
        auto settings = controller.walk_settings;
        settings.height = 1.9f;
        ASSERT_NO_THROW(controller.setWalkSettings(settings));
        EXPECT_EQ(primary.camera.t, primary_before);
        EXPECT_NEAR(secondary.camera.t.y, 1.9f, .02f);
        services().clear();
    }

    TEST_F(InputControllerFocusTest, NavigationModeChangesRequestGuiRedraw) {
        using Mode = InputController::CameraNavigationMode;
        const Mode modes[] = {Mode::Orbit, Mode::Trackball, Mode::FPV, Mode::Drone};
        for (const auto from : modes) {
            for (const auto to : modes) {
                SCOPED_TRACE(std::string(InputController::cameraNavigationModeName(from)) + " -> " +
                             InputController::cameraNavigationModeName(to));
                Viewport viewport(200, 200);
                InputController controller(nullptr, viewport);
                viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
                viewport.camera.R = glm::mat3(1.0f);
                viewport.camera.setPivot(glm::vec3(0.0f));
                controller.setCameraNavigationMode(from);
                const auto before = python::redraw_request_generation();
                const auto position = viewport.camera.t;
                const auto rotation = viewport.camera.R;
                const auto pivot = viewport.camera.getPivot();

                controller.setCameraNavigationMode(to);

                EXPECT_EQ(controller.cameraNavigationMode(), to);
                EXPECT_EQ(python::redraw_request_generation(), before + (from != to ? 1 : 0));
                EXPECT_EQ(viewport.camera.t, position);
                EXPECT_EQ(viewport.camera.R, rotation);
                EXPECT_EQ(viewport.camera.getPivot(), pivot);
            }
        }
    }

    TEST_F(InputControllerFocusTest, InitializationRestoresRememberedCameraControls) {
        const auto root = std::filesystem::temp_directory_path() / "lfs_InitializationRestoresRememberedCameraControls";
        std::filesystem::remove_all(root);
        const ScopedEnvironmentVariable home("LFS_HOME", root.string());
        const auto paths = core::UserPaths::resolve();
        ASSERT_TRUE(paths);
        ASSERT_TRUE(paths->ensureDirectories());
        for (const auto* mode : {"orbit", "trackball", "fpv", "drone"}) {
            for (const bool snap : {false, true}) {
                SCOPED_TRACE(std::string(mode) + (snap ? "/snap" : "/free"));
                const ScopedEnvironmentVariable case_home("LFS_HOME", (paths->rootDir() / (std::string(mode) + (snap ? "_snap" : "_free"))).string());
                const auto case_paths = core::UserPaths::resolve();
                ASSERT_TRUE(case_paths);
                ASSERT_TRUE(case_paths->ensureDirectories());
                const nlohmann::json preferences = {
                    {"schema_version", 1},
                    {"project_location", ""},
                    {"remember_camera_navigation", true},
                    {"camera_navigation_mode", mode},
                    {"remember_camera_view_snap", true},
                    {"camera_view_snap", snap}};
                {
                    std::ofstream output(case_paths->preferencesFile());
                    output << preferences;
                }
                Viewport viewport(200, 200);
                InputController controller(nullptr, viewport);
                controller.initialize();
                EXPECT_EQ(controller.cameraNavigationMode(),
                          *InputController::cameraNavigationModeFromName(mode));
                EXPECT_EQ(controller.cameraViewSnapEnabled(), snap);
                std::ifstream input(case_paths->preferencesFile());
                EXPECT_EQ(nlohmann::json::parse(input), preferences);
            }
        }
    }

    TEST_F(InputControllerFocusTest, InitializationHonorsIndependentRememberFlags) {
        const auto root = std::filesystem::temp_directory_path() / "lfs_InitializationHonorsIndependentRememberFlags";
        std::filesystem::remove_all(root);
        const ScopedEnvironmentVariable home("LFS_HOME", root.string());
        const auto paths = core::UserPaths::resolve();
        ASSERT_TRUE(paths);
        ASSERT_TRUE(paths->ensureDirectories());
        for (const bool remember_navigation : {false, true}) {
            for (const bool remember_snap : {false, true}) {
                const ScopedEnvironmentVariable case_home("LFS_HOME", (paths->rootDir() / (std::to_string(remember_navigation) + std::to_string(remember_snap))).string());
                const auto case_paths = core::UserPaths::resolve();
                ASSERT_TRUE(case_paths);
                ASSERT_TRUE(case_paths->ensureDirectories());
                const nlohmann::json preferences = {
                    {"schema_version", 1},
                    {"project_location", ""},
                    {"remember_camera_navigation", remember_navigation},
                    {"camera_navigation_mode", "fpv"},
                    {"remember_camera_view_snap", remember_snap},
                    {"camera_view_snap", true}};
                {
                    std::ofstream output(case_paths->preferencesFile());
                    output << preferences;
                }
                Viewport viewport(200, 200);
                InputController controller(nullptr, viewport);
                controller.initialize();
                EXPECT_EQ(controller.cameraNavigationMode(), remember_navigation
                                                                 ? InputController::CameraNavigationMode::FPV
                                                                 : InputController::CameraNavigationMode::Orbit);
                EXPECT_EQ(controller.cameraViewSnapEnabled(), remember_snap);
                std::ifstream input(case_paths->preferencesFile());
                EXPECT_EQ(nlohmann::json::parse(input), preferences);
            }
        }
    }

    TEST_F(InputControllerFocusTest, InitializationKeepsDefaultControlsWithoutPreferences) {
        const auto root = std::filesystem::temp_directory_path() / "lfs_InitializationKeepsDefaultControlsWithoutPreferences";
        std::filesystem::remove_all(root);
        const ScopedEnvironmentVariable home("LFS_HOME", root.string());
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        controller.initialize();
        EXPECT_EQ(controller.cameraNavigationMode(), InputController::CameraNavigationMode::Orbit);
        EXPECT_FALSE(controller.cameraViewSnapEnabled());
    }

    TEST_F(InputControllerFocusTest, HeldOrbitRequestsFramesOnlyWhenTheCameraChanges) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 100.0, 100.0);
        ASSERT_TRUE(controller.isCameraDragging());
        EXPECT_TRUE(controller.isContinuousInputActive());
        EXPECT_FALSE(controller.needsCameraAnimationFrame());

        const auto before = viewport.camera.R;
        controller.handleMouseMove(120.0, 110.0);
        EXPECT_NE(viewport.camera.R, before);
        ASSERT_TRUE(viewport.camera.hasOrbitMomentum());
        EXPECT_FALSE(controller.needsCameraAnimationFrame());

        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 120.0, 110.0);
        EXPECT_FALSE(controller.isCameraDragging());
        EXPECT_TRUE(controller.needsCameraAnimationFrame());
        for (int i = 0; i < 300; ++i)
            controller.update(1.0f / 60.0f);
        EXPECT_FALSE(controller.needsCameraAnimationFrame());
    }

    TEST_F(InputControllerFocusTest, HeldPanVelocityIsNotAnAnimationUntilRelease) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::RIGHT),
                                     input::ACTION_PRESS, 100.0, 100.0);
        ASSERT_TRUE(controller.isCameraDragging());
        EXPECT_FALSE(controller.needsCameraAnimationFrame());
        controller.handleMouseMove(125.0, 105.0);
        ASSERT_TRUE(viewport.camera.hasPanMomentum());
        EXPECT_FALSE(controller.needsCameraAnimationFrame());
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::RIGHT),
                                     input::ACTION_RELEASE, 125.0, 105.0);
        EXPECT_TRUE(controller.needsCameraAnimationFrame());
    }

    TEST_F(InputControllerFocusTest, HeldLookIsEventDrivenButKeyboardMovementAnimates) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        controller.initialize();
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);
        router.focusViewportKeyboard();
        controller.setCameraNavigationMode(InputController::CameraNavigationMode::FPV);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 100.0, 100.0);
        ASSERT_TRUE(controller.isCameraDragging());
        controller.handleMouseMove(120.0, 110.0);
        EXPECT_FALSE(controller.needsCameraAnimationFrame());
        controller.handleKey(input::KEY_W, input::ACTION_PRESS, input::KEYMOD_NONE);
        EXPECT_TRUE(controller.needsCameraAnimationFrame());
    }

    TEST_F(InputControllerFocusTest, OrthographicWasdMatchesPerspectiveThroughUpdate) {
        RenderingManager rendering;
        services().set(&rendering);
        auto settings = rendering.getSettings();
        settings.focal_length_mm = 50.0f;
        settings.orthographic = true;
        settings.ortho_scale = 70.0f;
        rendering.updateSettings(settings);
        constexpr float dt = 1.0f / 60.0f;
        constexpr int frames = 30;
        const auto hold = [&](InputController& controller, const int key) {
            controller.handleKey(key, input::ACTION_PRESS, input::KEYMOD_NONE);
            for (int i = 0; i < frames; ++i)
                controller.update(dt);
            controller.handleKey(key, input::ACTION_RELEASE, input::KEYMOD_NONE);
        };
        Viewport ortho(200, 200);
        InputController ortho_controller(nullptr, ortho);
        input::InputRouter ortho_router;
        ortho_controller.setInputRouter(&ortho_router);
        ortho_controller.initialize();
        ortho_router.focusViewportKeyboard();
        const auto eye = ortho.camera.t;
        const auto pivot = ortho.camera.pivot;
        const float distance_to_pivot = glm::length(pivot - eye);
        hold(ortho_controller, input::KEY_A);
        const float ortho_pixels = glm::length(ortho.camera.t - eye) * settings.ortho_scale;
        settings.ortho_scale = 140.0f;
        rendering.updateSettings(settings);
        ortho.camera.clearWasdMomentum();
        ortho.camera.t = eye;
        ortho.camera.pivot = pivot;
        hold(ortho_controller, input::KEY_A);
        const float wide_ortho_pixels = glm::length(ortho.camera.t - eye) * settings.ortho_scale;
        settings.ortho_scale = 70.0f;
        rendering.updateSettings(settings);
        Viewport forward_ortho(200, 200);
        forward_ortho.camera.t = eye;
        forward_ortho.camera.pivot = pivot;
        forward_ortho.camera.pivot = pivot;
        InputController forward_controller(nullptr, forward_ortho);
        input::InputRouter forward_router;
        forward_controller.setInputRouter(&forward_router);
        forward_controller.initialize();
        forward_router.focusViewportKeyboard();
        hold(forward_controller, input::KEY_W);
        EXPECT_GT(rendering.getSettings().ortho_scale, settings.ortho_scale);
        const glm::vec3 forward = lfs::rendering::cameraForward(forward_ortho.camera.R);
        const float moved_distance = glm::dot(forward_ortho.camera.t - eye, forward);
        EXPECT_GT(moved_distance, 0.0f);
        const float next_distance = glm::length(forward_ortho.camera.pivot - forward_ortho.camera.t);
        EXPECT_LT(glm::length(forward_ortho.camera.pivot - pivot), 1e-5f);
        EXPECT_NEAR(next_distance * rendering.getSettings().ortho_scale,
                    distance_to_pivot * settings.ortho_scale, 1e-3f);
        settings.orthographic = false;
        rendering.updateSettings(settings);
        Viewport perspective(200, 200);
        perspective.camera.t = eye;
        perspective.camera.pivot = pivot;
        InputController perspective_controller(nullptr, perspective);
        input::InputRouter perspective_router;
        perspective_controller.setInputRouter(&perspective_router);
        perspective_controller.initialize();
        perspective_router.focusViewportKeyboard();
        auto expected = perspective.camera;
        perspective_controller.handleKey(input::KEY_A, input::ACTION_PRESS, input::KEYMOD_NONE);
        for (int i = 0; i < frames; ++i) {
            perspective_controller.update(dt);
            expected.advanceWasd(dt, false, false, true, false, false, false);
        }
        perspective_controller.handleKey(input::KEY_A, input::ACTION_RELEASE, input::KEYMOD_NONE);
        EXPECT_EQ(perspective.camera.t, expected.t);
        EXPECT_EQ(perspective.camera.pivot, expected.pivot);
        const float perspective_pixels = glm::length(perspective.camera.t - eye) *
                                         (perspective.windowSize.y /
                                          (2.0f * distance_to_pivot *
                                           std::tan(glm::radians(rendering.getFovDegrees()) * 0.5f)));
        EXPECT_NEAR(ortho_pixels, perspective_pixels, 0.75f);
        EXPECT_NEAR(wide_ortho_pixels, perspective_pixels, 0.75f);
    }

    TEST_F(InputControllerFocusTest, PausedOrbitDecaysWithoutFrameTicks) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 100.0, 100.0);
        controller.handleMouseMove(120.0, 110.0);
        ASSERT_TRUE(viewport.camera.hasOrbitMomentum());
        const auto paused_rotation = viewport.camera.R;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 120.0, 110.0);
        EXPECT_FALSE(controller.needsCameraAnimationFrame());
        EXPECT_EQ(viewport.camera.R, paused_rotation);
    }

    TEST(CameraAnimationCadenceTest, ContinuousTicksUseDisplayOpportunitiesWithoutACatchUpBurst) {
        CameraAnimationCadence cadence;
        const CameraAnimationCadence::Clock::time_point start{};
        int frames = 0;
        int last_visible_input = -1;
        for (int input = 0; input < 1000; ++input) {
            const auto now = start + std::chrono::milliseconds(input);
            if (cadence.secondsUntilReady(now, 0.01) == 0.0) {
                cadence.startFrame(now);
                last_visible_input = input;
                ++frames;
            }
        }
        EXPECT_EQ(frames, 100);
        EXPECT_EQ(last_visible_input, 990);
        // No accumulated render debt after a slow frame or an idle period.
        const auto later = start + std::chrono::seconds(2);
        EXPECT_EQ(cadence.secondsUntilReady(later, 0.01), 0.0);
        cadence.startFrame(later);
        EXPECT_GT(cadence.secondsUntilReady(later, 0.01), 0.0);
    }

    TEST(CameraAnimationCadenceTest, FirstUpdateIsImmediateAndDisplayChangesAreRespected) {
        CameraAnimationCadence cadence;
        const CameraAnimationCadence::Clock::time_point start{};
        EXPECT_EQ(cadence.secondsUntilReady(start, 1.0 / 60.0), 0.0);
        cadence.startFrame(start);
        const auto next = start + std::chrono::milliseconds(5);
        EXPECT_GT(cadence.secondsUntilReady(next, 1.0 / 60.0), 0.0);
        EXPECT_EQ(cadence.secondsUntilReady(next, 1.0 / 240.0), 0.0);
    }

    // Fails if a wake without input (a viewer-thread user event) counts as input, which would let
    // demand-driven frames skip the display pacing, or if any kind of user input does not bypass it.
    TEST(FrameInputBufferTest, OnlyUserInputBypassesDisplayPacing) {
        FrameInputBuffer input;
        input.had_event = true;
        input.user_event = true;
        EXPECT_FALSE(input.hasUserInput());

        const std::vector<std::function<void(FrameInputBuffer&)>> inputs{
            [](FrameInputBuffer& b) { b.mouse_moved = true; },
            [](FrameInputBuffer& b) { b.window_event = true; },
            [](FrameInputBuffer& b) { b.mouse_wheel = 1.0f; },
            [](FrameInputBuffer& b) { b.mouse_wheel_x = -1.0f; },
            [](FrameInputBuffer& b) { b.mouse_button_events.emplace_back(); },
            [](FrameInputBuffer& b) { b.input_events.emplace_back(); },
            [](FrameInputBuffer& b) { b.keys_pressed.push_back(SDL_SCANCODE_W); },
        };
        for (size_t i = 0; i < inputs.size(); ++i) {
            FrameInputBuffer buffer;
            inputs[i](buffer);
            EXPECT_TRUE(buffer.hasUserInput()) << "input kind " << i;
        }
    }

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

    TEST_F(InputControllerFocusTest, CameraViewKeysUseSceneUidsWithoutTrainerAndPreserveComparison) {
        // core::Camera keeps its transforms on the device; the CPU-only CI runners have none.
        if (int devices = 0; cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            GTEST_SKIP() << "CUDA device unavailable";
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);
        router.focusViewportKeyboard();
        SceneManager scene_manager;
        RenderingManager rendering_manager;
        services().set(&scene_manager);
        services().set(&rendering_manager);
        auto& scene = scene_manager.getScene();
        scene_manager.changeContentType(SceneManager::ContentType::Dataset);
        const auto group = scene.addCameraGroup("Cameras", scene.addGroup("Dataset"), 3);
        for (const int uid : {4, 17, 42}) {
            auto camera = std::make_shared<core::Camera>(
                core::Tensor::eye(3, core::Device::CPU),
                core::Tensor::zeros({3}, core::Device::CPU),
                100.0f, 100.0f, 32.0f, 32.0f,
                core::Tensor(), core::Tensor(), core::CameraModelType::PINHOLE,
                std::to_string(uid), std::filesystem::path("source.png"), std::filesystem::path{},
                64, 64, uid);
            scene.addCamera(std::to_string(uid), group, std::move(camera));
        }
        ASSERT_EQ(services().trainerOrNull(), nullptr);
        controller.handleKey(input::KEY_G, input::ACTION_PRESS, input::KEYMOD_NONE);
        ASSERT_TRUE(rendering_manager.isGTComparisonActive());
        const auto press = [&](const int key, const int expected_uid) {
            controller.handleKey(key, input::ACTION_PRESS, input::KEYMOD_NONE);
            // Browsing repeats on its own timer; an OS repeat must not add a step.
            controller.handleKey(key, input::ACTION_REPEAT, input::KEYMOD_NONE);
            controller.handleKey(key, input::ACTION_RELEASE, input::KEYMOD_NONE);
            controller.update(0.016f);
            EXPECT_EQ(rendering_manager.getCurrentCameraId(), expected_uid);
            EXPECT_TRUE(rendering_manager.isGTComparisonActive());
        };
        press(input::KEY_LEFT, 42);
        press(input::KEY_RIGHT, 4);
        press(input::KEY_RIGHT, 17);
        press(input::KEY_LEFT, 4);
        core::events::cmd::GoToCamView{.cam_id = 17}.emit();
        press(input::KEY_RIGHT, 42);
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

    TEST_F(InputControllerFocusTest, ViewportViewHotkeysBypassGuiKeyboardFocusWhenNotTextEditing) {
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
        EXPECT_EQ(toggle_split_count, 1);
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

        EXPECT_EQ(toggle_gt_count, 0);
        EXPECT_EQ(toggle_split_count, 1);
    }

    TEST_F(InputControllerFocusTest, GUsesNodeCompareForViewerAndKeepsVBindings) {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);
        router.focusViewportKeyboard();

        SceneManager scene_manager;
        RenderingManager rendering_manager;
        services().set(&scene_manager);
        services().set(&rendering_manager);
        auto& scene = scene_manager.getScene();
        const auto make_splat = [](const float x) {
            return std::make_unique<lfs::core::SplatData>(
                0,
                Tensor::from_vector({x, 0.0f, 2.0f}, {size_t{1}, size_t{3}}, Device::CPU),
                Tensor::from_vector({1.0f, 1.0f, 1.0f}, {size_t{1}, size_t{1}, size_t{3}}, Device::CPU),
                Tensor::zeros({size_t{1}, size_t{0}, size_t{3}}, Device::CPU, DataType::Float32),
                Tensor::zeros({size_t{1}, size_t{3}}, Device::CPU),
                Tensor::from_vector({1.0f, 0.0f, 0.0f, 0.0f}, {size_t{1}, size_t{4}}, Device::CPU),
                Tensor::from_vector({8.0f}, {size_t{1}, size_t{1}}, Device::CPU),
                1.0f);
        };
        int split_toggles = 0;
        int gt_toggles = 0;
        lfs::event::ScopedHandler handlers;
        handlers.subscribe<core::events::cmd::ToggleSplitView>([&](const auto&) { ++split_toggles; });
        handlers.subscribe<core::events::cmd::ToggleGTComparison>([&](const auto&) { ++gt_toggles; });

        controller.handleKey(input::KEY_G, input::ACTION_PRESS, input::KEYMOD_NONE);
        EXPECT_EQ(split_toggles, 0);
        EXPECT_EQ(gt_toggles, 0);
        EXPECT_FALSE(rendering_manager.isGTComparisonActive());

        scene.addSplat("first", make_splat(0.0f));
        controller.handleKey(input::KEY_G, input::ACTION_PRESS, input::KEYMOD_NONE);
        EXPECT_EQ(split_toggles, 0);
        EXPECT_EQ(gt_toggles, 0);
        EXPECT_FALSE(rendering_manager.isGTComparisonActive());

        scene.addSplat("second", make_splat(1.0f));
        controller.handleKey(input::KEY_G, input::ACTION_PRESS, input::KEYMOD_NONE);
        EXPECT_EQ(split_toggles, 1);
        EXPECT_EQ(gt_toggles, 0);
        EXPECT_FALSE(rendering_manager.isGTComparisonActive());

        controller.handleKey(input::KEY_V, input::ACTION_PRESS, input::KEYMOD_NONE);
        EXPECT_EQ(split_toggles, 2);
        EXPECT_EQ(gt_toggles, 0);

        controller.handleKey(input::KEY_V, input::ACTION_PRESS, input::KEYMOD_SHIFT);
        EXPECT_EQ(split_toggles, 2);
        EXPECT_EQ(gt_toggles, 0);
        EXPECT_TRUE(rendering_manager.isIndependentSplitViewActive());
    }

    TEST_F(InputControllerFocusTest, EscapeWithScenePanelFocusStaysWithGui) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        auto& focus = gui::guiFocusState();
        focus.want_capture_keyboard = true;
        focus.any_item_active = true;

        ASSERT_EQ(router.keyboardFocus(), input::InputTarget::Gui);
        controller.handleKey(input::KEY_ESCAPE, input::ACTION_PRESS, input::KEYMOD_NONE);
        EXPECT_EQ(router.keyboardFocus(), input::InputTarget::Gui);
        EXPECT_FALSE(router.isViewportKeyboardFocused());
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

        EXPECT_EQ(toggle_gt_count, 0);
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

    TEST_F(InputControllerFocusTest, DeleteNodeShortcutDoesNotFireDuringTextEntry) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        SceneManager scene_manager;
        scene_manager.getScene().addGroup("delete_me");
        scene_manager.selectNode("delete_me");
        ASSERT_EQ(scene_manager.getSelectedNodeNames(), std::vector<std::string>{"delete_me"});

        ToolContext tool_context(nullptr, &scene_manager, &viewport, nullptr);
        controller.setToolContext(&tool_context);

        lfs::event::ScopedHandler handlers;
        int remove_ply_count = 0;
        handlers.subscribe<core::events::cmd::RemovePLY>(
            [&](const auto&) { ++remove_ply_count; });

        auto& focus = gui::guiFocusState();
        focus.want_capture_keyboard = true;
        focus.want_text_input = true;
        focus.any_item_active = true;

        router.focusViewportKeyboard();
        controller.handleKey(input::KEY_DELETE, input::ACTION_PRESS, input::KEYMOD_NONE);

        EXPECT_EQ(remove_ply_count, 0);
        EXPECT_EQ(input::shortcutScopeForAction(input::Action::DELETE_NODE),
                  input::ShortcutScope::GlobalWhenNotTextEditing);
    }

    TEST_F(InputControllerFocusTest, EditorFocusProtectsKeysUntilBlur) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);
        SceneManager scene;
        scene.getScene().addGroup("model");
        scene.selectNode("model");
        ToolContext context(nullptr, &scene, &viewport, nullptr);
        controller.setToolContext(&context);
        lfs::event::ScopedHandler handlers;
        int deletes = 0;
        handlers.subscribe<core::events::cmd::RemovePLY>([&](const auto&) { ++deletes; });

        router.focusViewportKeyboard();
        router.beginMouseButton(input::ACTION_PRESS, 10, 10);
        router.endMouseButton(input::ACTION_RELEASE);
        gui::guiFocusState().want_text_input = true;
        EXPECT_TRUE(router.isTextInputActive());
        controller.handleKey(input::KEY_DELETE, input::ACTION_PRESS, 0);
        EXPECT_EQ(deletes, 0);

        gui::guiFocusState().want_text_input = true;
        controller.handleKey(input::KEY_DELETE, input::ACTION_PRESS, 0);
        EXPECT_EQ(deletes, 0);
        gui::guiFocusState().reset();
        router.focusViewportKeyboard();
        controller.handleKey(input::KEY_DELETE, input::ACTION_PRESS, 0);
        EXPECT_EQ(deletes, 1);
        router.onWindowFocusLost();
        EXPECT_FALSE(router.isTextInputActive());
    }

    TEST_F(InputControllerFocusTest, EditorLettersDoNotBecomeCameraScrollChords) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        const glm::mat3 original = viewport.camera.R;
        gui::guiFocusState().want_text_input = true;
        controller.handleKey(input::KEY_R, input::ACTION_PRESS, 0);
        controller.handleScroll(0.0, 1.0);
        for (int column = 0; column < 3; ++column)
            EXPECT_LT(glm::distance(viewport.camera.R[column], original[column]), 1e-6f);
        controller.handleKey(input::KEY_R, input::ACTION_RELEASE, 0);
        gui::guiFocusState().reset();
        controller.handleKey(input::KEY_R, input::ACTION_PRESS, 0);
        controller.handleScroll(0.0, 1.0);
        EXPECT_GT(glm::distance(viewport.camera.R[0], original[0]), 1e-3f);
        controller.handleKey(input::KEY_R, input::ACTION_RELEASE, 0);
    }

    TEST_F(InputControllerFocusTest, EditorKeysNeverReachModalOperatorsEvenOverViewport) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);
        router.focusViewportKeyboard();
        int received = 0;
        constexpr auto id = "test.editor.key_isolation";
        op::operators().registerCallbackOperator(
            {.python_class_id = id, .label = "Key isolation"},
            {.invoke = [](op::OperatorProperties&) { return op::OperatorResult::RUNNING_MODAL; },
             .modal = [&](const op::ModalEvent&, op::OperatorProperties&) {
                 ++received;
                 return op::OperatorResult::RUNNING_MODAL; }});
        ASSERT_EQ(op::operators().invoke(id).status, op::OperatorResult::RUNNING_MODAL);
        gui::guiFocusState().want_text_input = true;
        for (const auto key : {input::KEY_DELETE, input::KEY_BACKSPACE, input::KEY_Z,
                               input::KEY_Y, input::KEY_A, input::KEY_C, input::KEY_X,
                               input::KEY_V, input::KEY_HOME, input::KEY_END,
                               input::KEY_G, input::KEY_R, input::KEY_W, input::KEY_D}) {
            for (const auto mods : {0, input::KEYMOD_CTRL, input::KEYMOD_CTRL | input::KEYMOD_SHIFT}) {
                controller.handleKey(key, input::ACTION_PRESS, mods);
                controller.handleKey(key, input::ACTION_REPEAT, mods);
                controller.handleKey(key, input::ACTION_RELEASE, mods);
            }
        }
        EXPECT_EQ(received, 0);
        gui::guiFocusState().reset();
        controller.handleKey(input::KEY_G, input::ACTION_PRESS, 0);
        EXPECT_EQ(received, 1);
        op::operators().cancelModalOperator();
        op::operators().unregisterOperator(id);
    }

    TEST_F(InputControllerFocusTest, OwnedModifierReleaseReachesModalAfterTextFocus) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        gui::RmlUIManager dispatcher;
        std::vector<int> actions;
        constexpr auto id = "test.input.press_owner";
        op::operators().registerCallbackOperator(
            {.python_class_id = id, .label = "Press owner"},
            {.invoke = [](op::OperatorProperties&) { return op::OperatorResult::RUNNING_MODAL; },
             .modal = [&](const op::ModalEvent& event, op::OperatorProperties&) {
                 if (const auto* key = event.as<KeyEvent>())
                     actions.push_back(key->action);
                 return op::OperatorResult::RUNNING_MODAL; }});
        ASSERT_EQ(op::operators().invoke(id).status, op::OperatorResult::RUNNING_MODAL);
        SDL_Event down{};
        down.type = SDL_EVENT_KEY_DOWN;
        down.key.scancode = SDL_SCANCODE_LALT;
        down.key.down = true;
        ASSERT_FALSE(dispatcher.dispatchInputEvent(down).consumed);
        controller.handleKey(input::KEY_LEFT_ALT, input::ACTION_PRESS, input::KEYMOD_ALT);
        SDL_Event up = down;
        up.type = SDL_EVENT_KEY_UP;
        up.key.down = false;
        const auto release = dispatcher.dispatchInputEvent(up);
        ASSERT_TRUE(release.owned_release);
        gui::guiFocusState().want_text_input = true;
        controller.handleKey(input::KEY_LEFT_ALT, input::KEY_LEFT_ALT, SDL_SCANCODE_LALT,
                             input::ACTION_RELEASE, 0, release.owned_release);
        EXPECT_EQ(actions, (std::vector<int>{input::ACTION_PRESS, input::ACTION_RELEASE}));
        op::operators().cancelModalOperator();
        op::operators().unregisterOperator(id);
    }

    TEST_F(InputControllerFocusTest, FocusFramesOnlyVisibleModels) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        SceneManager scene_manager;
        ToolContext tool_context(nullptr, &scene_manager, &viewport, nullptr);
        controller.setToolContext(&tool_context);

        auto& scene = scene_manager.getScene();
        const auto group = scene.addGroup("group");
        scene.addSplat("near", makeTwoGaussianSplat(0.0f), group);
        scene.setNodeVisibility(scene.addSplat("far", makeTwoGaussianSplat(100.0f), group), false);

        const auto expect_framed_near = [&](const char* selection) {
            SCOPED_TRACE(selection);
            viewport.camera.pivot = glm::vec3(7.0f);
            ASSERT_TRUE(controller.focusSelection());
            EXPECT_LT(glm::length(viewport.camera.getPivot()), 1e-3f);
        };
        expect_framed_near("nothing");
        scene_manager.selectNode("group");
        expect_framed_near("group with a hidden child");
        scene_manager.selectNode("far");
        expect_framed_near("hidden node");
    }

    // Fails if one far floater makes F park the camera beyond the zoom-out limit.
    TEST_F(InputControllerFocusTest, FocusFramesThroughFarFloatersAndCanZoomOut) {
        using lfs::core::Device;
        using lfs::core::Tensor;
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        SceneManager scene_manager;
        ToolContext tool_context(nullptr, &scene_manager, &viewport, nullptr);
        controller.setToolContext(&tool_context);

        constexpr size_t count = 500;
        std::vector<float> means(count * 3, 0.0f), rotations(count * 4, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            means[i * 3] = static_cast<float>(i % 10) * 0.2f - 1.0f;
            means[i * 3 + 1] = static_cast<float>((i / 10) % 10) * 0.2f - 1.0f;
            rotations[i * 4] = 1.0f;
        }
        means[0] = 90000.0f; // one floater
        scene_manager.getScene().addSplat(
            "model",
            std::make_unique<lfs::core::SplatData>(
                0, Tensor::from_vector(means, {count, size_t{3}}, Device::CPU),
                Tensor::zeros({count, size_t{1}, size_t{3}}, Device::CPU),
                Tensor::zeros({count, size_t{0}, size_t{3}}, Device::CPU),
                Tensor::zeros({count, size_t{3}}, Device::CPU),
                Tensor::from_vector(rotations, {count, size_t{4}}, Device::CPU),
                Tensor::zeros({count, size_t{1}}, Device::CPU), 1.0f));

        ASSERT_TRUE(controller.focusSelection());
        const float framed = glm::length(viewport.camera.pivot - viewport.camera.t);
        EXPECT_LT(framed, 100.0f);
        viewport.camera.zoom(-1.0f);
        EXPECT_GT(glm::length(viewport.camera.pivot - viewport.camera.t), framed);
    }

    TEST_F(InputControllerFocusTest, FrameAndHomeKeysWorkWhileAPanelHasKeyboardFocus) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);
        SceneManager scene_manager;
        ToolContext tool_context(nullptr, &scene_manager, &viewport, nullptr);
        controller.setToolContext(&tool_context);
        scene_manager.getScene().addSplat("model", makeTwoGaussianSplat(10.0f));

        // A row clicked in the scene panel holds keyboard focus.
        auto& focus = gui::guiFocusState();
        focus.want_capture_keyboard = true;
        focus.any_item_active = true;
        ASSERT_FALSE(router.isViewportKeyboardFocused());

        // A key the panel handled itself stays there (the image preview uses F).
        controller.handleKey(input::KEY_F, input::KEY_F, 0, input::ACTION_PRESS, input::KEYMOD_NONE, false, true);
        EXPECT_LT(glm::length(viewport.camera.getPivot()), 1e-3f);

        controller.handleKey(input::KEY_F, input::ACTION_PRESS, input::KEYMOD_NONE);
        const glm::vec3 framed = viewport.camera.getPivot();
        EXPECT_NEAR(glm::length(framed), 10.0f, 1e-3f);

        controller.handleKey(input::KEY_H, input::ACTION_PRESS, input::KEYMOD_SHIFT);
        viewport.camera.pivot = glm::vec3(7.0f);
        controller.handleKey(input::KEY_H, input::ACTION_PRESS, input::KEYMOD_NONE);
        EXPECT_EQ(viewport.camera.getPivot(), framed);

        const decltype(viewport.camera) defaults{};
        controller.resetHome();
        EXPECT_EQ(viewport.camera.home_t, defaults.home_t);
        EXPECT_EQ(viewport.camera.getPivot(), defaults.home_pivot);

        focus.want_text_input = true;
        controller.handleKey(input::KEY_F, input::ACTION_PRESS, input::KEYMOD_NONE);
        EXPECT_EQ(viewport.camera.getPivot(), defaults.home_pivot);
    }

    // Fails if Ctrl+H does not restore the default home, as the Reset Home button does.
    TEST_F(InputControllerFocusTest, CtrlHRestoresTheDefaultHome) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);
        const decltype(viewport.camera) defaults{};

        viewport.camera.pivot = glm::vec3(7.0f);
        controller.handleKey(input::KEY_H, input::ACTION_PRESS, input::KEYMOD_SHIFT);
        ASSERT_NE(viewport.camera.home_pivot, defaults.home_pivot);

        controller.handleKey(input::KEY_H, input::ACTION_PRESS, input::KEYMOD_CTRL);
        EXPECT_EQ(viewport.camera.home_t, defaults.home_t);
        EXPECT_EQ(viewport.camera.home_pivot, defaults.home_pivot);
        EXPECT_EQ(viewport.camera.getPivot(), defaults.home_pivot);
    }

    // Fails if WASD, Shift speed-up or the speed keys stop while a panel row holds keyboard focus.
    TEST_F(InputControllerFocusTest, NavigationKeysWorkWhileAPanelHasKeyboardFocus) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        controller.initialize();
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);
        auto& focus = gui::guiFocusState();
        focus.want_capture_keyboard = true;
        focus.any_item_active = true;
        ASSERT_FALSE(router.isViewportKeyboardFocused());

        // Hold W for half a second. Without a window the release cannot be read
        // back from SDL, so focus loss clears the held key as it does in the app.
        const auto travel = [&](const int mods) {
            SDL_SetModState(mods & input::KEYMOD_SHIFT ? SDL_KMOD_LSHIFT : SDL_KMOD_NONE);
            const glm::vec3 start = viewport.camera.t;
            controller.handleKey(input::KEY_W, input::ACTION_PRESS, mods);
            for (int i = 0; i < 30; ++i)
                controller.update(1.0f / 60.0f);
            const float distance = glm::length(viewport.camera.t - start);
            controller.onWindowFocusLost();
            viewport.camera.clearWasdMomentum();
            SDL_SetModState(SDL_KMOD_NONE);
            return distance;
        };
        const float plain = travel(input::KEYMOD_NONE);
        EXPECT_GT(plain, 0.0f);
        EXPECT_GT(travel(input::KEYMOD_SHIFT), plain * 1.5f);

        const float speed = viewport.camera.getWasdSpeed();
        controller.handleKey(input::KEY_EQUAL, input::ACTION_PRESS, input::KEYMOD_CTRL);
        EXPECT_GT(viewport.camera.getWasdSpeed(), speed);

        // A focused text field keeps its keys.
        focus.want_text_input = true;
        EXPECT_EQ(travel(input::KEYMOD_NONE), 0.0f);
    }

    TEST_F(InputControllerFocusTest, DeleteNodeShortcutFiresWhenViewportFocused) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        SceneManager scene_manager;
        scene_manager.getScene().addGroup("delete_me");
        scene_manager.selectNode("delete_me");
        ASSERT_EQ(scene_manager.getSelectedNodeNames(), std::vector<std::string>{"delete_me"});

        ToolContext tool_context(nullptr, &scene_manager, &viewport, nullptr);
        controller.setToolContext(&tool_context);

        lfs::event::ScopedHandler handlers;
        int remove_ply_count = 0;
        handlers.subscribe<core::events::cmd::RemovePLY>(
            [&](const auto& cmd) {
                ++remove_ply_count;
                EXPECT_EQ(cmd.name, "delete_me");
            });

        router.focusViewportKeyboard();
        controller.handleKey(input::KEY_DELETE, input::ACTION_PRESS, input::KEYMOD_NONE);

        EXPECT_EQ(remove_ply_count, 1);
    }

    TEST_F(InputControllerFocusTest, TransformToolShortcutDoesNotFireDuringTextEntry) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        lfs::event::ScopedHandler handlers;
        int toolbar_tool_count = 0;
        handlers.subscribe<core::events::tools::SetToolbarTool>(
            [&](const auto&) { ++toolbar_tool_count; });

        auto& focus = gui::guiFocusState();
        focus.want_capture_keyboard = true;
        focus.want_text_input = true;
        focus.any_item_active = true;

        router.focusViewportKeyboard();
        controller.handleKey(input::KEY_2, input::ACTION_PRESS, input::KEYMOD_NONE);

        EXPECT_EQ(toolbar_tool_count, 0);
        EXPECT_EQ(input::shortcutScopeForAction(input::Action::TOOL_TRANSLATE),
                  input::ShortcutScope::GlobalWhenNotTextEditing);
        EXPECT_EQ(input::shortcutScopeForAction(input::Action::CYCLE_PLY),
                  input::ShortcutScope::GlobalWhenNotTextEditing);
    }

    TEST_F(InputControllerFocusTest, ProjectSaveShortcutRemainsGlobalDuringTextEntry) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        lfs::event::ScopedHandler handlers;
        int project_save_count = 0;
        handlers.subscribe<core::events::cmd::ProjectSave>(
            [&](const auto&) { ++project_save_count; });

        auto& focus = gui::guiFocusState();
        focus.want_capture_keyboard = true;
        focus.want_text_input = true;
        focus.any_item_active = true;

        controller.handleKey(
            input::KEY_S, input::ACTION_PRESS,
            input::KEYMOD_CTRL);

        EXPECT_EQ(project_save_count, 1);
    }

    TEST_F(InputControllerFocusTest, ViewportClickDuringTextEntryDoesNotStartCameraGesture) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        auto& focus = gui::guiFocusState();
        focus.want_capture_keyboard = true;
        focus.want_text_input = true;
        focus.any_item_active = true;

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 40.0, 50.0);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 40.0, 50.0);
        router.endMouseButton(input::ACTION_RELEASE);

        EXPECT_FALSE(controller.isContinuousInputActive());
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

    TEST_F(InputControllerFocusTest, UnmatchedViewportReleaseRestoresKeyboardFocusAfterWindowHide) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);
        ASSERT_EQ(router.keyboardFocus(), input::InputTarget::Viewport);
        router.onWindowFocusLost();
        ASSERT_EQ(router.keyboardFocus(), input::InputTarget::None);

        router.endMouseButton(input::ACTION_RELEASE, 40.0, 50.0);

        EXPECT_EQ(router.keyboardFocus(), input::InputTarget::Viewport);
        EXPECT_EQ(router.state().pointer_capture, input::InputTarget::None);
    }

    TEST_F(InputControllerFocusTest, UnmatchedReleasePreservesOtherFocusTargets) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        router.endMouseButton(input::ACTION_RELEASE, 2500.0, 2500.0);
        EXPECT_EQ(router.keyboardFocus(), input::InputTarget::None);

        auto& focus = gui::guiFocusState();
        focus.want_capture_keyboard = true;
        focus.any_item_active = true;
        router.endMouseButton(input::ACTION_RELEASE, 40.0, 50.0);
        EXPECT_EQ(router.keyboardFocus(), input::InputTarget::Gui);

        focus.reset();
        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);
        router.endMouseButton(input::ACTION_RELEASE, 40.0, 50.0);
        EXPECT_EQ(router.keyboardFocus(), input::InputTarget::Viewport);
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

    // Fails if mouse look stops while the locked pointer rests at the window edge.
    TEST_F(InputControllerFocusTest, FpvLookKeepsTurningWhileThePointerRestsAtTheEdge) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = glm::mat3(1.0f);
        controller.setCameraNavigationMode(InputController::CameraNavigationMode::FPV);
        const glm::vec3 start = lfs::rendering::cameraForward(viewport.camera.R);

        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 40.0, 50.0);
        controller.handleMouseMove(0.0, 50.0, glm::dvec2(-40.0, 0.0));
        const glm::vec3 first = lfs::rendering::cameraForward(viewport.camera.R);
        controller.handleMouseMove(0.0, 50.0, glm::dvec2(-40.0, 0.0));
        const glm::vec3 second = lfs::rendering::cameraForward(viewport.camera.R);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 0.0, 50.0);

        const float first_turn = std::acos(glm::clamp(glm::dot(start, first), -1.0f, 1.0f));
        const float second_turn = std::acos(glm::clamp(glm::dot(first, second), -1.0f, 1.0f));
        EXPECT_GT(first_turn, 0.01f);
        EXPECT_NEAR(second_turn, first_turn, 1e-3f);
        EXPECT_NEAR(viewport.camera.t.z, 5.0f, 1e-5f);
    }

    TEST_F(InputControllerFocusTest, VrViewingBlocksRelativeMouseLookAndRestoresItOnExit) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = glm::mat3(1.0f);
        controller.setCameraNavigationMode(InputController::CameraNavigationMode::FPV);

        const auto button = static_cast<int>(input::AppMouseButton::MIDDLE);
        controller.handleMouseButton(button, input::ACTION_PRESS, 40.0, 50.0);
        ASSERT_TRUE(controller.isCameraDragging());
        controller.setVrViewing(true);
        EXPECT_FALSE(controller.isCameraDragging());
        const auto rotation = viewport.camera.R;
        const auto position = viewport.camera.t;
        const auto pivot = viewport.camera.getPivot();

        controller.handleMouseButton(button, input::ACTION_PRESS, 40.0, 50.0);
        controller.handleMouseMove(0.0, 50.0, glm::dvec2(-40.0, 0.0));
        controller.handleMouseMove(40.0, 0.0);
        EXPECT_EQ(viewport.camera.R, rotation);
        EXPECT_EQ(viewport.camera.t, position);
        EXPECT_EQ(viewport.camera.getPivot(), pivot);
        EXPECT_FALSE(controller.isCameraDragging());

        controller.setVrViewing(false);
        controller.handleMouseButton(button, input::ACTION_PRESS, 40.0, 50.0);
        controller.handleMouseMove(0.0, 50.0, glm::dvec2(-40.0, 0.0));
        EXPECT_GT(glm::distance(lfs::rendering::cameraForward(viewport.camera.R),
                                lfs::rendering::cameraForward(rotation)),
                  0.01f);
        EXPECT_EQ(viewport.camera.t, position);
        controller.handleMouseButton(button, input::ACTION_RELEASE, 0.0, 50.0);
        EXPECT_FALSE(controller.isCameraDragging());
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

    TEST_F(InputControllerFocusTest, PreferencesDefaultsToRemappableCtrlComma) {
        input::InputBindings bindings;

        EXPECT_EQ(static_cast<int>(input::Action::OPEN_PREFERENCES), 78);
        EXPECT_EQ(bindings.getActionForKey(input::ToolMode::GLOBAL,
                                           input::KEY_COMMA,
                                           input::MODIFIER_CTRL),
                  input::Action::OPEN_PREFERENCES);

        const auto trigger = bindings.getTriggerForAction(input::Action::OPEN_PREFERENCES,
                                                          input::ToolMode::GLOBAL);
        ASSERT_TRUE(trigger.has_value());
        const auto* key_trigger = std::get_if<input::KeyTrigger>(&*trigger);
        ASSERT_NE(key_trigger, nullptr);
        EXPECT_EQ(key_trigger->key, input::KEY_COMMA);
        EXPECT_EQ(key_trigger->modifiers, input::MODIFIER_CTRL);
        EXPECT_EQ(input::describe(input::Action::OPEN_PREFERENCES).ui_section,
                  input::ActionSection::UI);
    }

    TEST_F(InputControllerFocusTest, SelectAllSceneNodesDefaultsToRemappableCtrlShiftA) {
        input::InputBindings bindings;

        EXPECT_EQ(bindings.getActionForKey(input::ToolMode::GLOBAL,
                                           input::KEY_A,
                                           input::MODIFIER_CTRL | input::MODIFIER_SHIFT),
                  input::Action::SELECT_ALL_SCENE_NODES);

        const auto trigger = bindings.getTriggerForAction(
            input::Action::SELECT_ALL_SCENE_NODES, input::ToolMode::GLOBAL);
        ASSERT_TRUE(trigger.has_value());
        const auto* key_trigger = std::get_if<input::KeyTrigger>(&*trigger);
        ASSERT_NE(key_trigger, nullptr);
        EXPECT_EQ(key_trigger->key, input::KEY_A);
        EXPECT_EQ(key_trigger->modifiers,
                  input::MODIFIER_CTRL | input::MODIFIER_SHIFT);
        EXPECT_EQ(input::describe(input::Action::SELECT_ALL_SCENE_NODES).ui_section,
                  input::ActionSection::UI);
    }

    TEST_F(InputControllerFocusTest, SceneGraphActionsUseRemappableNonConflictingDefaults) {
        input::InputBindings bindings;

        EXPECT_EQ(bindings.getActionForKey(input::ToolMode::GLOBAL,
                                           input::KEY_SPACE,
                                           input::MODIFIER_CTRL),
                  input::Action::NONE);
        EXPECT_EQ(bindings.getActionForKey(input::ToolMode::GLOBAL,
                                           input::KEY_H,
                                           input::MODIFIER_CTRL | input::MODIFIER_SHIFT),
                  input::Action::TOGGLE_SCENE_SELECTION_VISIBILITY);
        EXPECT_EQ(bindings.getActionForKey(input::ToolMode::GLOBAL,
                                           input::KEY_T,
                                           input::MODIFIER_CTRL | input::MODIFIER_SHIFT),
                  input::Action::TOGGLE_SCENE_SELECTION_TRAINING);
        EXPECT_EQ(input::describe(input::Action::TOGGLE_SCENE_SELECTION_VISIBILITY).ui_section,
                  input::ActionSection::UI);
        EXPECT_EQ(input::describe(input::Action::TOGGLE_SCENE_SELECTION_TRAINING).ui_section,
                  input::ActionSection::UI);
    }

    TEST_F(InputControllerFocusTest, VersionTwentyFourSceneGraphBindingsMigrateToFinalDefaults) {
        const auto profile_path = std::filesystem::temp_directory_path() /
                                  "lfs_input_bindings_transient_v24.json";
        std::filesystem::remove(profile_path);
        {
            std::ofstream file(profile_path);
            ASSERT_TRUE(file.is_open());
            file << R"({
  "name": "TransientV24",
  "version": 24,
  "bindings": [
    {"mode":0,"action":82,"description":"Select Scene Hierarchy","trigger_type":"key","key":65,"modifiers":3},
    {"mode":0,"action":83,"description":"Toggle Scene Cursor Selection","trigger_type":"key","key":32,"modifiers":2},
    {"mode":0,"action":84,"description":"Toggle Scene Selection Visibility","trigger_type":"key","key":72,"modifiers":4},
    {"mode":0,"action":85,"description":"Toggle Scene Selection Training","trigger_type":"key","key":84,"modifiers":4}
  ]
})";
        }

        input::InputBindings loaded;
        ASSERT_TRUE(loaded.loadProfileFromFile(profile_path));
        EXPECT_EQ(loaded.getActionForKey(input::ToolMode::GLOBAL, input::KEY_SPACE,
                                         input::MODIFIER_CTRL),
                  input::Action::NONE);
        EXPECT_EQ(loaded.getActionForKey(input::ToolMode::GLOBAL, input::KEY_H,
                                         input::MODIFIER_CTRL | input::MODIFIER_SHIFT),
                  input::Action::TOGGLE_SCENE_SELECTION_VISIBILITY);
        EXPECT_EQ(loaded.getActionForKey(input::ToolMode::GLOBAL, input::KEY_T,
                                         input::MODIFIER_CTRL | input::MODIFIER_SHIFT),
                  input::Action::TOGGLE_SCENE_SELECTION_TRAINING);

        std::filesystem::remove(profile_path);
    }

    TEST_F(InputControllerFocusTest, VersionTwentyProfileMigratesPreferencesShortcut) {
        const auto profile_path = std::filesystem::temp_directory_path() /
                                  "lfs_input_bindings_legacy_v20.json";
        std::filesystem::remove(profile_path);
        {
            std::ofstream file(profile_path);
            ASSERT_TRUE(file.is_open());
            file << R"({
  "name": "LegacyV20",
  "version": 20,
  "bindings": []
})";
        }

        input::InputBindings loaded;
        ASSERT_TRUE(loaded.loadProfileFromFile(profile_path));
        EXPECT_EQ(loaded.getActionForKey(input::ToolMode::GLOBAL,
                                         input::KEY_COMMA,
                                         input::MODIFIER_CTRL),
                  input::Action::OPEN_PREFERENCES);

        std::filesystem::remove(profile_path);
    }

    TEST_F(InputControllerFocusTest, McpRuntimeShortcutsAreStableAndRemappable) {
        input::InputBindings bindings;

        EXPECT_EQ(static_cast<int>(input::Action::TOGGLE_MCP_SERVER), 79);
        EXPECT_EQ(static_cast<int>(input::Action::TOGGLE_MCP_BINDING), 80);
        EXPECT_EQ(static_cast<int>(input::Action::TOGGLE_GRID), 81);
        EXPECT_EQ(bindings.getActionForKey(input::ToolMode::GLOBAL,
                                           input::KEY_M,
                                           input::MODIFIER_CTRL | input::MODIFIER_SHIFT),
                  input::Action::TOGGLE_MCP_SERVER);
        EXPECT_EQ(bindings.getActionForKey(input::ToolMode::GLOBAL,
                                           input::KEY_N,
                                           input::MODIFIER_CTRL | input::MODIFIER_SHIFT),
                  input::Action::TOGGLE_MCP_BINDING);
        EXPECT_EQ(input::shortcutScopeForAction(input::Action::TOGGLE_MCP_SERVER),
                  input::ShortcutScope::Global);
        EXPECT_EQ(input::shortcutScopeForAction(input::Action::TOGGLE_MCP_BINDING),
                  input::ShortcutScope::Global);
    }

    TEST_F(InputControllerFocusTest, VersionTwentyOneProfileMigratesBothMcpShortcuts) {
        const auto profile_path = std::filesystem::temp_directory_path() /
                                  "lfs_input_bindings_legacy_v21.json";
        std::filesystem::remove(profile_path);
        {
            std::ofstream file(profile_path);
            ASSERT_TRUE(file.is_open());
            file << R"({
  "name": "LegacyV21",
  "version": 21,
  "bindings": []
})";
        }

        input::InputBindings loaded;
        ASSERT_TRUE(loaded.loadProfileFromFile(profile_path));
        EXPECT_EQ(loaded.getActionForKey(input::ToolMode::GLOBAL,
                                         input::KEY_M,
                                         input::MODIFIER_CTRL | input::MODIFIER_SHIFT),
                  input::Action::TOGGLE_MCP_SERVER);
        EXPECT_EQ(loaded.getActionForKey(input::ToolMode::GLOBAL,
                                         input::KEY_N,
                                         input::MODIFIER_CTRL | input::MODIFIER_SHIFT),
                  input::Action::TOGGLE_MCP_BINDING);

        std::filesystem::remove(profile_path);
    }

    TEST_F(InputControllerFocusTest, DefaultProfileMigrationClosesSourceBeforeAtomicPersistence) {
        const auto root = std::filesystem::temp_directory_path() /
                          "lfs_input_bindings_default_migration";
        std::error_code filesystem_error;
        std::filesystem::remove_all(root, filesystem_error);
        const ScopedEnvironmentVariable home("LFS_HOME", root.string());
        const auto paths = core::UserPaths::resolve();
        ASSERT_TRUE(paths);
        ASSERT_TRUE(paths->ensureDirectories());

        const auto profile_path = paths->keymapDir() / "Default.json";
        std::filesystem::create_directories(profile_path.parent_path());
        {
            std::ofstream file(profile_path);
            ASSERT_TRUE(file.is_open());
            file << R"({
  "name": "Default",
  "version": 21,
  "bindings": []
})";
        }

        // Construct before persistence is on so this load is the migration
        // that replaces Default.json. The callback runs after that save and
        // before load returns. An unclosed source stream still owns the
        // original inode there; Windows then refuses the atomic replace.
        input::InputBindings::setPersistenceEnabled(false);
        input::InputBindings loaded;
        bool published = false;
        bool source_open_at_publish = false;
#ifdef __linux__
        struct stat source_identity {};
        ASSERT_EQ(::stat(profile_path.c_str(), &source_identity), 0);
#endif
        loaded.setOnBindingsChanged([&] {
            published = true;
#ifdef __linux__
            std::error_code scan_error;
            const std::filesystem::directory_iterator end;
            for (std::filesystem::directory_iterator entry("/proc/self/fd", scan_error);
                 !scan_error && entry != end; entry.increment(scan_error)) {
                const auto fd_text = entry->path().filename().string();
                char* parsed_end = nullptr;
                const auto fd = std::strtol(fd_text.c_str(), &parsed_end, 10);
                if (parsed_end == fd_text.c_str() || fd < 0)
                    continue;
                struct stat open_identity {};
                if (::fstat(static_cast<int>(fd), &open_identity) != 0)
                    continue;
                if (open_identity.st_dev == source_identity.st_dev &&
                    open_identity.st_ino == source_identity.st_ino)
                    source_open_at_publish = true;
            }
#endif
        });
        input::InputBindings::setPersistenceEnabled(true);
        ASSERT_TRUE(loaded.loadProfileFromFile(profile_path));
        input::InputBindings::setPersistenceEnabled(false);
        EXPECT_TRUE(published);
        EXPECT_FALSE(source_open_at_publish);

        std::ifstream persisted(profile_path);
        ASSERT_TRUE(persisted.is_open());
        const std::string contents((std::istreambuf_iterator<char>(persisted)), {});
        EXPECT_NE(contents.find("\"version\": " + std::to_string(input::test::currentProfileVersion())),
                  std::string::npos);
        EXPECT_NE(contents.find("Gallery Primary Action"), std::string::npos);
        EXPECT_NE(contents.find("Copy Gallery Link"), std::string::npos);
        EXPECT_NE(contents.find("Refresh Assets"), std::string::npos);
        EXPECT_NE(contents.find("Toggle MCP Server"), std::string::npos);
        EXPECT_NE(contents.find("Toggle MCP Local/Network Binding"), std::string::npos);

        persisted.close();
        std::filesystem::remove_all(root, filesystem_error);
    }

    TEST_F(InputControllerFocusTest, WindowProfileMigratesWithoutReusingWindowActionIds) {
        const auto path = std::filesystem::temp_directory_path() / "lfs_keymap_v28.json";
        {
            std::ofstream file(path);
            ASSERT_TRUE(file.is_open());
            file << R"({"name":"Legacy","version":28,"bindings":[
                {"mode":1,"action":85,"trigger_type":"scroll","modifiers":5},
                {"mode":1,"action":86,"trigger_type":"drag","button":0,"modifiers":5},
                {"mode":0,"action":87,"trigger_type":"key","key":71,"modifiers":2},
                {"mode":0,"action":88,"trigger_type":"key","key":71,"modifiers":3},
                {"mode":0,"action":81,"trigger_type":"key","key":294,"modifiers":0}
            ]})";
        }
        using namespace input;
        InputBindings bindings;
        ASSERT_TRUE(bindings.loadProfileFromFile(path));
        EXPECT_TRUE(bindings.getBindingsForMode(ToolMode::SELECTION).empty());
        EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_G, MODIFIER_CTRL), Action::GROUP_SELECTED_SCENE_NODES);
        EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_G, MODIFIER_CTRL | MODIFIER_SHIFT), Action::UNGROUP_SELECTED_SCENE_NODE);
        EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_F5, MODIFIER_NONE), Action::TOGGLE_GRID);
        EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_ENTER, MODIFIER_CTRL), Action::ASSET_GALLERY_PRIMARY);
        EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_C, MODIFIER_CTRL | MODIFIER_SHIFT), Action::ASSET_GALLERY_COPY_LINK);
        bindings.clearBinding(ToolMode::GLOBAL, Action::ASSET_GALLERY_PRIMARY);
        ASSERT_TRUE(bindings.saveProfileToFile(path));
        ASSERT_TRUE(bindings.loadProfileFromFile(path));
        EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_ENTER, MODIFIER_CTRL), Action::NONE);
        std::filesystem::remove(path);
    }

    TEST_F(InputControllerFocusTest, VersionTwentySevenDistinguishesWindowAndGalleryProfiles) {
        const auto path = std::filesystem::temp_directory_path() / "lfs_keymap_v27.json";
        using namespace input;
        InputBindings bindings;
        {
            std::ofstream file(path);
            file << R"({"name":"Legacy","version":27,"bindings":[
                {"mode":1,"action":85,"description":"Window size","trigger_type":"scroll","modifiers":5},
                {"mode":1,"action":86,"description":"Window drag","trigger_type":"drag","button":0,"modifiers":5}
            ]})";
        }
        ASSERT_TRUE(bindings.loadProfileFromFile(path));
        EXPECT_TRUE(bindings.getBindingsForMode(ToolMode::SELECTION).empty());
        EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_ENTER, MODIFIER_CTRL), Action::ASSET_GALLERY_PRIMARY);
        {
            std::ofstream file(path);
            file << R"({"name":"Unbound","version":27,"bindings":[]})";
        }
        ASSERT_TRUE(bindings.loadProfileFromFile(path));
        EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_ENTER, MODIFIER_CTRL), Action::NONE);
        EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_F5, MODIFIER_NONE), Action::NONE);
        std::filesystem::remove(path);
    }

    TEST_F(InputControllerFocusTest, MenuAcceleratorsResolveAndDispatchThroughInputBindings) {
        using namespace input;
        RestorePythonConsoleCallback restore_console_callback;
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        const auto& bindings = controller.getBindings();

        struct MenuAccelerator {
            std::string source;
            Action action = Action::NONE;
            bool save_project = false;
        };
        const auto menu_root = std::filesystem::path(PROJECT_ROOT_PATH) / "src/python/lfs_plugins";
        std::vector<MenuAccelerator> menu_accelerators;
        for (const auto& entry : std::filesystem::directory_iterator(menu_root)) {
            if (entry.path().extension() != ".py" ||
                !entry.path().filename().string().ends_with("_menu.py"))
                continue;
            std::ifstream input(entry.path());
            ASSERT_TRUE(input.good()) << entry.path();
            std::string line;
            while (std::getline(input, line)) {
                const auto start = line.find("shortcut=");
                if (start == std::string::npos)
                    continue;
                if (line.find("\"Ctrl+S\"", start) != std::string::npos) {
                    menu_accelerators.push_back(
                        {.source = entry.path().filename().string(),
                         .save_project = true});
                    continue;
                }
                auto action_start = line.find("Action.", start);
                if (action_start == std::string::npos)
                    action_start = line.find("action.", start);
                ASSERT_NE(action_start, std::string::npos)
                    << entry.path().filename().string() << ": " << line;
                action_start += 7;
                auto action_end = action_start;
                while (action_end < line.size() &&
                       (std::isalnum(static_cast<unsigned char>(line[action_end])) ||
                        line[action_end] == '_'))
                    ++action_end;
                const auto action = actionFromName(
                    std::string_view(line).substr(action_start, action_end - action_start));
                ASSERT_TRUE(action)
                    << entry.path().filename().string() << ": " << line;
                menu_accelerators.push_back(
                    {.source = entry.path().filename().string(), .action = *action});
            }
        }
        ASSERT_FALSE(menu_accelerators.empty());

        lfs::event::ScopedHandler handlers;
        std::map<Action, int> dispatched;
        const auto record = [&](const Action action) {
            return [&, action](const auto&) { ++dispatched[action]; };
        };
        handlers.subscribe<core::events::cmd::Undo>(record(Action::UNDO));
        handlers.subscribe<core::events::cmd::Redo>(record(Action::REDO));
        handlers.subscribe<core::events::cmd::CopySelection>(record(Action::COPY_SELECTION));
        handlers.subscribe<core::events::cmd::CutSelection>(record(Action::CUT_SELECTION));
        handlers.subscribe<core::events::cmd::PasteSelection>(record(Action::PASTE_SELECTION));
        handlers.subscribe<core::events::cmd::InvertSelection>(record(Action::INVERT_SELECTION));
        handlers.subscribe<core::events::cmd::SelectAll>(record(Action::SELECT_ALL));
        handlers.subscribe<core::events::cmd::DeselectAll>(record(Action::DESELECT_ALL));
        handlers.subscribe<core::events::ui::ToggleVramHud>(record(Action::TOGGLE_PERFORMANCE_HUD));
        handlers.subscribe<core::events::ui::ToggleUI>(record(Action::TOGGLE_UI));
        handlers.subscribe<core::events::cmd::ProjectSave>(record(Action::NONE));

        const auto default_profile = InputBindings::createDefaultProfile();
        for (const auto& accelerator : menu_accelerators) {
            SCOPED_TRACE(accelerator.source);
            if (accelerator.save_project) {
                const auto before = dispatched[Action::NONE];
                const int physical_key = sdlScancodeToAppKey(SDL_SCANCODE_S);
                const int logical_key = sdlShortcutKeyToAppKey(
                    SDL_SCANCODE_S,
                    SDL_GetKeyFromScancode(SDL_SCANCODE_S, SDL_KMOD_NONE, false));
                EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, logical_key,
                                                   MODIFIER_CTRL),
                          Action::NONE);
                controller.handleKey(physical_key, logical_key, SDL_SCANCODE_S,
                                     ACTION_PRESS, MODIFIER_CTRL);
                EXPECT_EQ(dispatched[Action::NONE], before + 1);
                continue;
            }

            const auto binding = std::ranges::find_if(
                default_profile.bindings, [&](const auto& candidate) {
                    return candidate.mode == ToolMode::GLOBAL &&
                           candidate.action == accelerator.action;
                });
            ASSERT_NE(binding, default_profile.bindings.end())
                << getActionName(accelerator.action);
            const auto* trigger = std::get_if<KeyTrigger>(&binding->trigger);
            ASSERT_NE(trigger, nullptr) << getActionName(accelerator.action);
            const auto scancode = appKeyToSdlScancode(trigger->key);
            ASSERT_NE(scancode, SDL_SCANCODE_UNKNOWN)
                << getActionName(accelerator.action);
            const int physical_key = sdlScancodeToAppKey(scancode);
            const int logical_key = sdlShortcutKeyToAppKey(
                scancode, SDL_GetKeyFromScancode(scancode, SDL_KMOD_NONE, false));
            EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, logical_key,
                                               trigger->modifiers),
                      accelerator.action);
            const auto before = dispatched[accelerator.action];
            const int console_before = python_console_toggle_calls;
            // The home keys move the viewport camera instead of publishing a command.
            const bool home_action = accelerator.action == Action::CAMERA_RESET_HOME ||
                                     accelerator.action == Action::CAMERA_SET_HOME ||
                                     accelerator.action == Action::CAMERA_RESTORE_HOME;
            if (home_action)
                viewport.camera.pivot = glm::vec3(3.0f);
            const auto camera_before = std::pair{viewport.camera.getPivot(), viewport.camera.home_pivot};
            controller.handleKey(physical_key, logical_key, scancode,
                                 ACTION_PRESS, trigger->modifiers);
            controller.handleKey(physical_key, logical_key, scancode,
                                 ACTION_RELEASE, trigger->modifiers);
            if (home_action)
                EXPECT_NE(std::pair(viewport.camera.getPivot(), viewport.camera.home_pivot), camera_before)
                    << getActionName(accelerator.action);
            else if (accelerator.action == Action::TOGGLE_PYTHON_CONSOLE)
                EXPECT_EQ(python_console_toggle_calls, console_before + 1);
            else if (accelerator.action != Action::OPEN_PREFERENCES)
                EXPECT_EQ(dispatched[accelerator.action], before + 1)
                    << getActionName(accelerator.action);
        }
    }

    TEST_F(InputControllerFocusTest, GalleryActionsHaveRebindableNativeDefaults) {
        using namespace input;
        InputBindings bindings;
        EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_ENTER, MODIFIER_CTRL), Action::ASSET_GALLERY_PRIMARY);
        EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_C, MODIFIER_CTRL | MODIFIER_SHIFT), Action::ASSET_GALLERY_COPY_LINK);
        EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_F5, MODIFIER_NONE), Action::ASSET_REFRESH);
        EXPECT_EQ(actionFromName("asset_refresh"), Action::ASSET_REFRESH);
        bindings.setBinding(ToolMode::GLOBAL, Action::ASSET_REFRESH, KeyTrigger{KEY_F6, MODIFIER_CTRL});
        EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_F6, MODIFIER_CTRL), Action::ASSET_REFRESH);
        EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_F5, MODIFIER_NONE), Action::NONE);
        bindings.clearBinding(ToolMode::GLOBAL, Action::ASSET_REFRESH);
        EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_F6, MODIFIER_CTRL), Action::NONE);
    }

    TEST_F(InputControllerFocusTest, McpRuntimeShortcutsDispatchDuringPythonCapture) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);

        int server_calls = 0;
        int binding_calls = 0;
        setRuntimeServiceControls({
            .toggle_mcp_enabled = [&] {
                ++server_calls;
                return true; },
            .toggle_mcp_binding = [&] {
                ++binding_calls;
                return true; },
        });

        lfs::python::request_keyboard_capture("mcp-shortcut-test");
        controller.handleKey(input::KEY_M, input::ACTION_PRESS,
                             input::MODIFIER_CTRL | input::MODIFIER_SHIFT);
        controller.handleKey(input::KEY_N, input::ACTION_PRESS,
                             input::MODIFIER_CTRL | input::MODIFIER_SHIFT);
        lfs::python::release_keyboard_capture("mcp-shortcut-test");

        EXPECT_EQ(server_calls, 1);
        EXPECT_EQ(binding_calls, 1);
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
                  input::ShortcutScope::GlobalWhenNotTextEditing);

        EXPECT_FALSE(rendering_manager.getSettings().show_camera_frustums);
        controller.handleKey(input::KEY_C, input::ACTION_PRESS, input::KEYMOD_ALT);
        EXPECT_TRUE(rendering_manager.getSettings().show_camera_frustums);
        controller.handleKey(input::KEY_C, input::ACTION_PRESS, input::KEYMOD_ALT);
        EXPECT_FALSE(rendering_manager.getSettings().show_camera_frustums);
    }

    TEST_F(InputControllerFocusTest, GridDefaultToAltGAndToggleRenderSetting) {
        RenderingManager rendering_manager;
        services().set(&rendering_manager);

        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);
        router.focusViewportKeyboard();

        EXPECT_EQ(controller.getBindings().getActionForKey(input::ToolMode::GLOBAL,
                                                           input::KEY_G,
                                                           input::MODIFIER_ALT),
                  input::Action::TOGGLE_GRID);
        EXPECT_EQ(input::shortcutScopeForAction(input::Action::TOGGLE_GRID),
                  input::ShortcutScope::GlobalWhenNotTextEditing);

        EXPECT_TRUE(rendering_manager.getSettings().show_grid);
        controller.handleKey(input::KEY_G, input::ACTION_PRESS, input::KEYMOD_ALT);
        EXPECT_FALSE(rendering_manager.getSettings().show_grid);
        controller.handleKey(input::KEY_G, input::ACTION_PRESS, input::KEYMOD_ALT);
        EXPECT_TRUE(rendering_manager.getSettings().show_grid);
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

    TEST_F(InputControllerFocusTest, VersionTwentyTwoProfileMigratesGridShortcutWhenFree) {
        const auto profile_path = std::filesystem::temp_directory_path() / "lfs_input_bindings_legacy_v22.json";
        std::filesystem::remove(profile_path);
        {
            std::ofstream file(profile_path);
            ASSERT_TRUE(file.is_open());
            file << R"({
  "name": "LegacyV22",
  "version": 22,
  "bindings": []
})";
        }

        input::InputBindings loaded;
        ASSERT_TRUE(loaded.loadProfileFromFile(profile_path));
        EXPECT_EQ(loaded.getActionForKey(input::ToolMode::GLOBAL,
                                         input::KEY_G,
                                         input::MODIFIER_ALT),
                  input::Action::TOGGLE_GRID);

        std::filesystem::remove(profile_path);
    }

    TEST_F(InputControllerFocusTest, VersionTwentyTwoProfilePreservesOccupiedAltG) {
        const auto profile_path = std::filesystem::temp_directory_path() / "lfs_input_bindings_legacy_v22_alt_g.json";
        std::filesystem::remove(profile_path);
        {
            std::ofstream file(profile_path);
            ASSERT_TRUE(file.is_open());
            file << R"({
  "name": "LegacyV22AltG",
  "version": 22,
  "bindings": [
    {
      "mode": 0,
      "action": 25,
      "description": "Cycle PLY",
      "trigger_type": "key",
      "key": 71,
      "modifiers": 4
    }
  ]
})";
        }

        input::InputBindings loaded;
        ASSERT_TRUE(loaded.loadProfileFromFile(profile_path));
        EXPECT_EQ(loaded.getActionForKey(input::ToolMode::GLOBAL,
                                         input::KEY_G,
                                         input::MODIFIER_ALT),
                  input::Action::CYCLE_PLY);
        EXPECT_FALSE(loaded.getTriggerForAction(input::Action::TOGGLE_GRID,
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
        EXPECT_TRUE(controller.getBindings()
                        .getTriggerForAction(input::Action::TOOL_MIRROR,
                                             input::ToolMode::GLOBAL)
                        .has_value());
    }

    TEST_F(InputControllerFocusTest, NumberRowToolShortcutsFollowPhysicalKeysAcrossLayouts) {
        struct Layout {
            const char* name;
            std::array<SDL_Keycode, 6> keycodes;
        };
        const std::array layouts = {
            Layout{"us", {SDLK_1, SDLK_2, SDLK_3, SDLK_4, SDLK_5, SDLK_6}},
            Layout{"be", {SDLK_AMPERSAND, static_cast<SDL_Keycode>(0xe9), SDLK_DBLAPOSTROPHE, SDLK_APOSTROPHE, SDLK_LEFTPAREN, static_cast<SDL_Keycode>(0xa7)}},
            Layout{"fr", {SDLK_AMPERSAND, static_cast<SDL_Keycode>(0xe9), SDLK_DBLAPOSTROPHE, SDLK_APOSTROPHE, SDLK_LEFTPAREN, SDLK_MINUS}},
        };
        const std::array expected_tools = {
            ToolType::Selection,
            ToolType::Translate,
            ToolType::Rotate,
            ToolType::Scale,
            ToolType::Mirror,
            ToolType::Align,
        };
        const std::array selection_modes = {
            input::Action::SELECT_MODE_CENTERS,
            input::Action::SELECT_MODE_RECTANGLE,
            input::Action::SELECT_MODE_POLYGON,
            input::Action::SELECT_MODE_LASSO,
            input::Action::SELECT_MODE_RINGS,
            input::Action::SELECT_MODE_COLOR,
        };

        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        lfs::event::ScopedHandler handlers;
        int tool_mode = -1;
        handlers.subscribe<core::events::tools::SetToolbarTool>(
            [&](const auto& event) { tool_mode = event.tool_mode; });

        for (const auto& layout : layouts) {
            for (size_t i = 0; i < expected_tools.size(); ++i) {
                SCOPED_TRACE(std::string(layout.name) + " key " + std::to_string(i + 1));
                const auto scancode = static_cast<SDL_Scancode>(SDL_SCANCODE_1 + i);
                const int physical_key = input::sdlScancodeToAppKey(scancode);
                const int logical_key = input::sdlShortcutKeyToAppKey(scancode, layout.keycodes[i]);
                EXPECT_EQ(logical_key, input::KEY_1 + static_cast<int>(i));

                tool_mode = -1;
                controller.handleKey(physical_key, logical_key, scancode,
                                     input::ACTION_PRESS, input::KEYMOD_NONE);
                EXPECT_EQ(tool_mode, static_cast<int>(expected_tools[i]));
                controller.handleKey(physical_key, logical_key, scancode,
                                     input::ACTION_RELEASE, input::KEYMOD_NONE);

                EXPECT_EQ(controller.getBindings().getActionForKey(input::ToolMode::GLOBAL,
                                                                   logical_key, input::MODIFIER_CTRL),
                          selection_modes[i]);
            }
        }
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
                  input::Action::ASSET_GALLERY_PRIMARY);
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

    TEST_F(InputControllerFocusTest, SetPivotCentersSharedComparisonCamera) {
        for (const auto mode : {SplitViewMode::Disabled, SplitViewMode::PLYComparison,
                                SplitViewMode::IndependentDual}) {
            for (const double click_x : {60.0, 160.0}) {
                SCOPED_TRACE(static_cast<int>(mode));
                SCOPED_TRACE(click_x);
                Viewport primary(200, 200);
                InputController controller(nullptr, primary);
                RenderingManager rendering;
                services().set(&rendering);
                controller.updateViewportBounds(0, 0, 200, 200);
                rendering.restoreSplitViewMode(mode, primary);
                auto& target = rendering.resolvePanelViewport(
                    primary, mode == SplitViewMode::IndependentDual && click_x > 100
                                 ? SplitViewPanelId::Right
                                 : SplitViewPanelId::Left);
                target.camera.R = glm::mat3(1.0f);
                target.camera.t = glm::vec3(0.0f, 0.0f, -5.0f);
                target.camera.pivot = glm::vec3(0.0f);
                // Exercise the normal right-button double click binding.
                controller.handleMouseButton(static_cast<int>(input::MouseButton::RIGHT),
                                             input::ACTION_PRESS, click_x, 80.0);
                controller.handleMouseButton(static_cast<int>(input::MouseButton::RIGHT),
                                             input::ACTION_RELEASE, click_x, 80.0);
                controller.handleMouseButton(static_cast<int>(input::MouseButton::RIGHT),
                                             input::ACTION_PRESS, click_x, 80.0);
                ASSERT_TRUE(target.camera.isGliding());
                target.camera.finishGlide();
                // Centering a perspective orbit pivot puts it on the camera's
                // forward axis, regardless of which side of the wipe was clicked.
                const auto direction = glm::transpose(target.camera.R) *
                                       (target.camera.pivot - target.camera.t);
                EXPECT_NEAR(direction.x, 0.0f, 1e-5f);
                EXPECT_NEAR(direction.y, 0.0f, 1e-5f);
                EXPECT_NEAR(glm::length(direction), 5.0f, 1e-5f);
                controller.handleMouseButton(static_cast<int>(input::MouseButton::RIGHT),
                                             input::ACTION_RELEASE, click_x, 80.0);
                services().clear();
            }
        }
    }

    TEST_F(InputControllerFocusTest, JoypadAssignmentSurvivesSidebarFocusAndApplicationFocusLoss) {
        Viewport primary(200, 200);
        InputController controller(nullptr, primary);
        RenderingManager rendering;
        services().set(&rendering);
        controller.updateViewportBounds(0, 0, 200, 200);
        rendering.restoreSplitViewMode(SplitViewMode::IndependentDual, primary);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);
        const auto click = [&](double x, double y) {
            router.beginMouseButton(input::ACTION_PRESS, x, y);
            controller.handleMouseButton(static_cast<int>(input::AppMouseButton::LEFT), input::ACTION_PRESS, x, y);
            controller.handleMouseButton(static_cast<int>(input::AppMouseButton::LEFT), input::ACTION_RELEASE, x, y);
            router.endMouseButton(input::ACTION_RELEASE, x, y);
        };
        click(160, 80);
        ASSERT_EQ(controller.gamepadNavigationPanel(), SplitViewPanelId::Right);
        click(250, 80); // panel outside the viewport
        EXPECT_FALSE(controller.hasViewportKeyboardFocus());
        EXPECT_EQ(controller.gamepadNavigationPanel(), SplitViewPanelId::Right);
        controller.onWindowFocusLost();
        EXPECT_EQ(controller.gamepadNavigationPanel(), SplitViewPanelId::Right);
        rendering.setFocusedSplitPanel(SplitViewPanelId::Left);
        EXPECT_EQ(controller.gamepadNavigationPanel(), SplitViewPanelId::Right);
        click(40, 80);
        EXPECT_EQ(controller.gamepadNavigationPanel(), SplitViewPanelId::Left);
        services().clear();
    }

    TEST_F(InputControllerFocusTest, JoypadSplitAssignmentIsRevokedOnSceneChange) {
        Viewport primary(200, 200);
        InputController controller(nullptr, primary);
        RenderingManager rendering;
        services().set(&rendering);
        controller.updateViewportBounds(0, 0, 200, 200);
        rendering.restoreSplitViewMode(SplitViewMode::IndependentDual, primary);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);
        router.beginMouseButton(input::ACTION_PRESS, 160, 80);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::LEFT), input::ACTION_PRESS, 160, 80);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::LEFT), input::ACTION_RELEASE, 160, 80);
        router.endMouseButton(input::ACTION_RELEASE, 160, 80);
        ASSERT_EQ(controller.gamepadNavigationPanel(), SplitViewPanelId::Right);
        core::events::state::SceneCleared{}.emit();
        EXPECT_EQ(controller.gamepadNavigationPanel(), SplitViewPanelId::Left);
        services().clear();
    }

} // namespace lfs::vis
