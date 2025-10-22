/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_controller.hpp"
#include "core/logger.hpp"
#include "rendering/rendering_manager.hpp"
#include "tools/tool_base.hpp"
#include "tools/translation_gizmo_tool.hpp"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <format>
#include <imgui.h>

namespace gs::visualizer {
    InputController* InputController::instance_ = nullptr;

    InputController::InputController(GLFWwindow* window, Viewport& viewport)
        : window_(window),
          viewport_(viewport) {
        // Subscribe to GoToCamView events
        events::cmd::GoToCamView::when([this](const auto& e) {
            handleGoToCamView(e);
        });
        // Subscribe to WindowFocusLost to reset states
        events::internal::WindowFocusLost::when([this](const auto&) {
            drag_mode_ = DragMode::None;
            std::fill(std::begin(keys_wasd_), std::end(keys_wasd_), false);
            hovered_camera_id_ = -1;
        });
        // Subscribe to GimbalLock events
        events::cmd::ToggleGimbalLock::when([this](const events::cmd::ToggleGimbalLock& e) {
            gimbal_locked = e.locked;
        });

        LOG_DEBUG("InputController created");
    }

    InputController::~InputController() {
        if (instance_ == this) {
            instance_ = nullptr;
        }

        // Clean up cursor resources
        if (resize_cursor_) {
            glfwDestroyCursor(resize_cursor_);
            resize_cursor_ = nullptr;
        }
        if (hand_cursor_) {
            glfwDestroyCursor(hand_cursor_);
            hand_cursor_ = nullptr;
        }

        // Reset cursor to default before destruction
        if (window_ && current_cursor_ != CursorType::Default) {
            glfwSetCursor(window_, nullptr);
        }
    }

    void InputController::initialize() {
        // CRITICAL: This must be called AFTER ImGui_ImplGlfw_InitForOpenGL
        instance_ = this;

        // Store ImGui's callbacks so we can chain to them
        imgui_callbacks_.mouse_button = glfwSetMouseButtonCallback(window_, mouseButtonCallback);
        imgui_callbacks_.cursor_pos = glfwSetCursorPosCallback(window_, cursorPosCallback);
        imgui_callbacks_.scroll = glfwSetScrollCallback(window_, scrollCallback);
        imgui_callbacks_.key = glfwSetKeyCallback(window_, keyCallback);
        imgui_callbacks_.drop = glfwSetDropCallback(window_, dropCallback);
        imgui_callbacks_.focus = glfwSetWindowFocusCallback(window_, windowFocusCallback);

        // Get initial mouse position
        double x, y;
        glfwGetCursorPos(window_, &x, &y);
        last_mouse_pos_ = {x, y};

        // Initialize frame timer
        last_frame_time_ = std::chrono::high_resolution_clock::now();

        // Create the cursors once at initialization
        resize_cursor_ = glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR);
        hand_cursor_ = glfwCreateStandardCursor(GLFW_HAND_CURSOR);

        LOG_DEBUG("InputController initialized - callbacks set");
    }

    // Static callbacks - chain to ImGui then handle ourselves
    void InputController::mouseButtonCallback(GLFWwindow* w, int button, int action, int mods) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.mouse_button) {
            instance_->imgui_callbacks_.mouse_button(w, button, action, mods);
        }

        // Then handle for camera
        if (instance_) {
            double x, y;
            glfwGetCursorPos(w, &x, &y);
            instance_->handleMouseButton(button, action, x, y);
        }
    }

    void InputController::cursorPosCallback(GLFWwindow* w, double x, double y) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.cursor_pos) {
            instance_->imgui_callbacks_.cursor_pos(w, x, y);
        }

        // Then handle for camera
        if (instance_) {
            instance_->handleMouseMove(x, y);
        }
    }

    void InputController::scrollCallback(GLFWwindow* w, double xoff, double yoff) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.scroll) {
            instance_->imgui_callbacks_.scroll(w, xoff, yoff);
        }

        // Then handle for camera
        if (instance_) {
            instance_->handleScroll(xoff, yoff);
        }
    }

    void InputController::keyCallback(GLFWwindow* w, int key, int scancode, int action, int mods) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.key) {
            instance_->imgui_callbacks_.key(w, key, scancode, action, mods);
        }

        // Then handle for camera
        if (instance_) {
            instance_->handleKey(key, action, mods);
        }
    }

    void InputController::dropCallback(GLFWwindow* w, int count, const char** paths) {
        // Let ImGui handle first (though it probably doesn't use this)
        if (instance_ && instance_->imgui_callbacks_.drop) {
            instance_->imgui_callbacks_.drop(w, count, paths);
        }

        // Then handle file drops
        if (instance_) {
            std::vector<std::string> files(paths, paths + count);
            instance_->handleFileDrop(files);
        }
    }

    void InputController::windowFocusCallback(GLFWwindow* w, int focused) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.focus) {
            instance_->imgui_callbacks_.focus(w, focused);
        }

        // Reset states on focus loss
        if (!focused) {
            if (instance_) {
                instance_->drag_mode_ = DragMode::None;
                std::fill(std::begin(instance_->keys_wasd_),
                          std::end(instance_->keys_wasd_), false);
                instance_->hovered_camera_id_ = -1; // Reset hovered camera

                // Reset cursor to default when losing focus
                if (instance_->current_cursor_ != CursorType::Default) {
                    glfwSetCursor(instance_->window_, nullptr);
                    instance_->current_cursor_ = CursorType::Default;
                }
            }
            events::internal::WindowFocusLost{}.emit();
            LOG_DEBUG("Window lost focus - input states reset");
        } else {
            LOG_DEBUG("Window gained focus");
        }
    }

    bool InputController::isNearSplitter(double x) const {
        if (!rendering_manager_ || rendering_manager_->getSettings().split_view_mode == SplitViewMode::Disabled) {
            return false;
        }

        float split_pos = rendering_manager_->getSettings().split_position;
        float split_x = viewport_bounds_.x + viewport_bounds_.width * split_pos;

        // Increase the hit area to 10 pixels for easier grabbing
        return std::abs(x - split_x) < 10.0;
    }

    // Core handlers
    void InputController::handleMouseButton(int button, int action, double x, double y) {
        // Check for splitter drag FIRST
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            // Check for double-click on camera frustum
            auto now = std::chrono::steady_clock::now();
            auto time_since_last = std::chrono::duration<double>(now - last_click_time_).count();
            double dist = glm::length(glm::dvec2(x, y) - last_click_pos_);

            constexpr double DOUBLE_CLICK_TIME = 0.5;
            constexpr double DOUBLE_CLICK_DISTANCE = 10.0;

            bool is_double_click = (time_since_last < DOUBLE_CLICK_TIME &&
                                    dist < DOUBLE_CLICK_DISTANCE);

            // If we have a hovered camera, check for double-click
            if (hovered_camera_id_ >= 0) {
                if (is_double_click && hovered_camera_id_ == last_clicked_camera_id_) {
                    LOG_INFO("Double-clicked on camera ID: {}", hovered_camera_id_);
                    events::cmd::GoToCamView{.cam_id = hovered_camera_id_}.emit();

                    // Reset click tracking to prevent triple-click
                    last_click_time_ = std::chrono::steady_clock::time_point();
                    last_click_pos_ = {-1000, -1000}; // Far away position
                    last_clicked_camera_id_ = -1;
                    return;
                }
                // First click on a camera - record it
                last_click_time_ = now;
                last_click_pos_ = {x, y};
                last_clicked_camera_id_ = hovered_camera_id_; // Remember which camera was clicked
                LOG_DEBUG("First click on camera ID: {} (time for double-click: {:.1f}s)",
                          hovered_camera_id_, DOUBLE_CLICK_TIME);
            } else {
                last_click_time_ = std::chrono::steady_clock::time_point();
                last_click_pos_ = {-1000, -1000};
                last_clicked_camera_id_ = -1;
            }

            // Check for splitter drag
            if (isNearSplitter(x) && rendering_manager_) {
                drag_mode_ = DragMode::Splitter;
                splitter_start_pos_ = rendering_manager_->getSettings().split_position;
                splitter_start_x_ = x;
                glfwSetCursor(window_, resize_cursor_);
                LOG_TRACE("Started splitter drag");
                return;
            }
        }

        if (action == GLFW_RELEASE && drag_mode_ == DragMode::Splitter) {
            drag_mode_ = DragMode::None;
            glfwSetCursor(window_, nullptr); // Reset cursor
            LOG_TRACE("Ended splitter drag");
            return;
        }

        // CHECK GIZMO NEXT - before any other input handling
        if (translation_gizmo_ && translation_gizmo_->isEnabled() && tool_context_) {
            if (translation_gizmo_->handleMouseButton(button, action, x, y, *tool_context_)) {
                // Gizmo consumed the event
                if (action == GLFW_PRESS) {
                    drag_mode_ = DragMode::Gizmo;
                    LOG_TRACE("Started gizmo drag");
                } else if (action == GLFW_RELEASE && drag_mode_ == DragMode::Gizmo) {
                    drag_mode_ = DragMode::None;
                    LOG_TRACE("Ended gizmo drag");
                }
                return; // Don't process camera controls
            }
        }

        if (action == GLFW_PRESS) {
            // Special handling for point cloud mode
            bool imgui_wants_mouse = ImGui::GetIO().WantCaptureMouse;
            if (point_cloud_mode_ && isInViewport(x, y) &&
                !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
                imgui_wants_mouse = false; // Override for point cloud mode
                LOG_TRACE("Point cloud mode - overriding ImGui mouse capture");
            }

            // Check if we should handle this
            if (imgui_wants_mouse || !isInViewport(x, y)) {
                return;
            }

            // Start camera interaction
            viewport_.camera.initScreenPos(glm::vec2(x, y));

            if (button == GLFW_MOUSE_BUTTON_LEFT) {
                drag_mode_ = DragMode::Pan;
                LOG_TRACE("Started camera pan");
            } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                drag_mode_ = DragMode::Rotate;
                LOG_TRACE("Started camera rotate");
            } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
                drag_mode_ = DragMode::Orbit;
                float current_time = static_cast<float>(glfwGetTime());
                viewport_.camera.startRotateAroundCenter(glm::vec2(x, y), current_time);
                LOG_TRACE("Started camera orbit");
            }
        } else if (action == GLFW_RELEASE) {
            // Always handle our own releases if we were dragging
            bool was_dragging = false;
            if (button == GLFW_MOUSE_BUTTON_LEFT && drag_mode_ == DragMode::Pan) {
                drag_mode_ = DragMode::None;
                was_dragging = true;
                LOG_TRACE("Ended camera pan");
            } else if (button == GLFW_MOUSE_BUTTON_RIGHT && drag_mode_ == DragMode::Rotate) {
                drag_mode_ = DragMode::None;
                was_dragging = true;
                LOG_TRACE("Ended camera rotate");
            } else if (button == GLFW_MOUSE_BUTTON_MIDDLE && drag_mode_ == DragMode::Orbit) {
                viewport_.camera.endRotateAroundCenter();
                drag_mode_ = DragMode::None;
                was_dragging = true;
                LOG_TRACE("Ended camera orbit");
            }

            // Force publish on mouse release
            if (was_dragging) {
                events::ui::CameraMove{
                    .rotation = viewport_.getRotationMatrix(),
                    .translation = viewport_.getTranslation()}
                    .emit();
            }
        }
    }

    void InputController::handleMouseMove(double x, double y) {
        // Track if we moved significantly
        glm::dvec2 current_pos{x, y};

        // Handle splitter dragging
        if (drag_mode_ == DragMode::Splitter && rendering_manager_) {
            double delta = x - splitter_start_x_;
            float new_pos = splitter_start_pos_ + static_cast<float>(delta / viewport_bounds_.width);

            // FIX: Allow dragging all the way to the edges - no margins!
            new_pos = std::clamp(new_pos, 0.0f, 1.0f);

            events::ui::SplitPositionChanged{.position = new_pos}.emit();
            last_mouse_pos_ = {x, y};
            return;
        }

        // Camera frustum hover detection with improved throttling
        if (rendering_manager_ &&
            rendering_manager_->getSettings().show_camera_frustums &&
            isInViewport(x, y) &&
            drag_mode_ == DragMode::None &&
            !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {

            // Additional throttling based on movement distance
            static glm::dvec2 last_pick_pos{-1, -1};
            static constexpr double MIN_PICK_DISTANCE = 3.0; // pixels

            bool should_pick = false;

            // Check if we moved enough from last pick position
            if (last_pick_pos.x < 0) {
                // First pick
                should_pick = true;
                last_pick_pos = current_pos;
            } else {
                double pick_distance = glm::length(current_pos - last_pick_pos);
                if (pick_distance >= MIN_PICK_DISTANCE) {
                    should_pick = true;
                    last_pick_pos = current_pos;
                }
            }

            if (should_pick) {
                auto result = rendering_manager_->pickCameraFrustum(glm::vec2(x, y));
                if (result >= 0) {
                    const int cam_id = result;
                    if (cam_id != hovered_camera_id_) {
                        hovered_camera_id_ = cam_id;
                        LOG_TRACE("Hovering over camera ID: {}", cam_id);

                        // Change cursor to hand
                        if (current_cursor_ != CursorType::Hand) {
                            glfwSetCursor(window_, hand_cursor_);
                            current_cursor_ = CursorType::Hand;
                        }
                    }
                } else {
                    // No camera under cursor
                    if (hovered_camera_id_ != -1) {
                        hovered_camera_id_ = -1;
                        LOG_TRACE("No longer hovering over camera");
                        if (current_cursor_ == CursorType::Hand) {
                            glfwSetCursor(window_, nullptr);
                            current_cursor_ = CursorType::Default;
                        }
                    }
                }
            }
        } else {
            // Not in conditions for camera picking
            if (hovered_camera_id_ != -1) {
                hovered_camera_id_ = -1;
                if (current_cursor_ == CursorType::Hand) {
                    glfwSetCursor(window_, nullptr);
                    current_cursor_ = CursorType::Default;
                }
            }
        }

        // Determine if we should show resize cursor for splitter
        bool should_show_resize = false;
        if (rendering_manager_ && rendering_manager_->getSettings().split_view_mode != SplitViewMode::Disabled) {
            should_show_resize = (drag_mode_ == DragMode::None &&
                                  isInViewport(x, y) &&
                                  isNearSplitter(x) &&
                                  !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow));
        }

        // Only call glfwSetCursor when state actually changes
        if (should_show_resize && current_cursor_ != CursorType::Resize) {
            glfwSetCursor(window_, resize_cursor_);
            current_cursor_ = CursorType::Resize;
        } else if (!should_show_resize && current_cursor_ == CursorType::Resize) {
            glfwSetCursor(window_, nullptr);
            current_cursor_ = CursorType::Default;
        }

        // Check gizmo if not already in gizmo drag mode
        if (translation_gizmo_ && translation_gizmo_->isEnabled() && tool_context_) {
            if (drag_mode_ == DragMode::Gizmo ||
                translation_gizmo_->handleMouseMove(x, y, *tool_context_)) {
                last_mouse_pos_ = {x, y};
                return;
            }
        }

        glm::vec2 pos(x, y);
        last_mouse_pos_ = current_pos;

        // Handle camera dragging
        if (drag_mode_ != DragMode::None &&
            drag_mode_ != DragMode::Gizmo &&
            drag_mode_ != DragMode::Splitter) {

            switch (drag_mode_) {
            case DragMode::Pan:
                viewport_.camera.translate(pos);
                break;
            case DragMode::Rotate:
                viewport_.camera.rotate(pos, gimbal_locked);
                break;
            case DragMode::Orbit: {
                float current_time = static_cast<float>(glfwGetTime());
                viewport_.camera.updateRotateAroundCenter(pos, current_time);
                break;
            }
            default:
                break;
            }
            publishCameraMove();
        }
    }

    void InputController::handleScroll([[maybe_unused]] double xoff, double yoff) {
        // Don't scroll if gizmo or splitter is active
        if (drag_mode_ == DragMode::Gizmo || drag_mode_ == DragMode::Splitter) {
            return;
        }

        if (!shouldCameraHandleInput())
            return;

        float delta = static_cast<float>(yoff);
        if (std::abs(delta) < 0.01f)
            return;

        if (key_r_pressed_) {
            viewport_.camera.rotate_roll(delta);
            LOG_TRACE("Camera roll: {}", delta);
        } else {
            viewport_.camera.zoom(delta);
            LOG_TRACE("Camera zoom: {}", delta);
        }

        publishCameraMove();
    }

    void InputController::handleKey(int key, int action, [[maybe_unused]] int mods) {
        // Track modifier keys (always, even if GUI has focus)
        if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL) {
            key_ctrl_pressed_ = (action != GLFW_RELEASE);
        }
        if (key == GLFW_KEY_R) {
            key_r_pressed_ = (action != GLFW_RELEASE);
            // Reset gizmo position if R is pressed and gizmo is enabled
            if (action == GLFW_PRESS && translation_gizmo_ && translation_gizmo_->isEnabled()) {
                LOG_DEBUG("Reset key pressed with gizmo enabled");
                // This would need to be exposed by the gizmo tool
                // For now, let the gizmo handle it internally
            }
        }

        if (key == GLFW_KEY_T && action == GLFW_PRESS && !ImGui::GetIO().WantCaptureKeyboard) {
            events::cmd::CyclePLY{}.emit();
            return;
        }

        if (key == GLFW_KEY_V && action == GLFW_PRESS && !ImGui::GetIO().WantCaptureKeyboard) {
            events::cmd::ToggleSplitView{}.emit();
            return;
        }

        if (key == GLFW_KEY_G && action == GLFW_PRESS && !ImGui::GetIO().WantCaptureKeyboard) {
            events::cmd::ToggleGTComparison{}.emit();
            LOG_DEBUG("Toggled GT comparison mode");
            return;
        }

        if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS && !ImGui::GetIO().WantCaptureKeyboard) {
            if (!training_manager_) {
                LOG_WARN("Training manager is not set; cannot cycle camera view.");
                return;
            }
            int num_cams = training_manager_->getCamList().size();
            if (num_cams == 0) {
                return;
            }

            last_camview++;
            if (last_camview >= num_cams) {
                last_camview = 0; // Wrap to beginning
            }

            events::cmd::GoToCamView{
                .cam_id = last_camview}
                .emit();
            return;
        }

        if (key == GLFW_KEY_LEFT && action == GLFW_PRESS && !ImGui::GetIO().WantCaptureKeyboard) {
            if (!training_manager_) {
                LOG_WARN("Training manager is not set; cannot cycle camera view.");
                return;
            }
            int num_cams = training_manager_->getCamList().size();
            if (num_cams == 0) {
                return;
            }

            last_camview--;
            if (last_camview < 0) {
                last_camview = num_cams - 1; // Wrap to end
            }

            events::cmd::GoToCamView{
                .cam_id = last_camview}
                .emit();
            return;
        }

        // Speed control works even when GUI has focus
        if (key_ctrl_pressed_ && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) {
                updateCameraSpeed(true);
                return;
            }
            if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) {
                updateCameraSpeed(false);
                return;
            }
        }

        // WASD only works when viewport has focus and gizmo isn't active
        if (!shouldCameraHandleInput() || drag_mode_ == DragMode::Gizmo || drag_mode_ == DragMode::Splitter)
            return;

        bool pressed = (action != GLFW_RELEASE);
        bool changed = false;

        switch (key) {
        case GLFW_KEY_W:
            keys_wasd_[0] = pressed;
            changed = true;
            break;
        case GLFW_KEY_A:
            keys_wasd_[1] = pressed;
            changed = true;
            break;
        case GLFW_KEY_S:
            keys_wasd_[2] = pressed;
            changed = true;
            break;
        case GLFW_KEY_D:
            keys_wasd_[3] = pressed;
            changed = true;
            break;
        case GLFW_KEY_Q:
            keys_wasd_[4] = pressed;
            changed = true;
            break;
        case GLFW_KEY_E:
            keys_wasd_[5] = pressed;
            changed = true;
            break;
        }

        if (changed) {
            LOG_TRACE("WASD state changed - W:{} A:{} S:{} D:{} Q:{} E:{}",
                      keys_wasd_[0], keys_wasd_[1], keys_wasd_[2], keys_wasd_[3], keys_wasd_[4], keys_wasd_[5]);
        }
    }

    void InputController::update(float delta_time) {
        // This catches cases where mouse release events are missed (e.g., outside window)
        if (drag_mode_ == DragMode::Orbit &&
            glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_MIDDLE) != GLFW_PRESS) {
            viewport_.camera.endRotateAroundCenter();
            drag_mode_ = DragMode::None;
            LOG_TRACE("Orbit stopped - button released outside window");
        }

        if (drag_mode_ == DragMode::Rotate &&
            glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS) {
            drag_mode_ = DragMode::None;
            LOG_TRACE("Rotate stopped - button released outside window");
        }

        if (drag_mode_ == DragMode::Pan &&
            glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
            drag_mode_ = DragMode::None;
            LOG_TRACE("Pan stopped - button released outside window");
        }

        if (drag_mode_ == DragMode::Splitter &&
            glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
            drag_mode_ = DragMode::None;
            glfwSetCursor(window_, nullptr); // Reset cursor
            LOG_TRACE("Splitter drag stopped - button released outside window");
        }

        // Handle continuous WASD movement
        if (shouldCameraHandleInput() && drag_mode_ != DragMode::Gizmo && drag_mode_ != DragMode::Splitter) {
            if (keys_wasd_[0]) {
                viewport_.camera.advance_forward(delta_time);
            }
            if (keys_wasd_[1]) {
                viewport_.camera.advance_left(delta_time);
            }
            if (keys_wasd_[2]) {
                viewport_.camera.advance_backward(delta_time);
            }
            if (keys_wasd_[3]) {
                viewport_.camera.advance_right(delta_time);
            }
            if (keys_wasd_[4]) {
                viewport_.camera.advance_up(delta_time);
            }
            if (keys_wasd_[5]) {
                viewport_.camera.advance_down(delta_time);
            }
        }

        // Publish if moving (removed inertia check)
        bool moving = keys_wasd_[0] || keys_wasd_[1] || keys_wasd_[2] || keys_wasd_[3] || keys_wasd_[4] || keys_wasd_[5];
        if (moving) {
            publishCameraMove();
        }
    }

    void InputController::handleFileDrop(const std::vector<std::string>& paths) {
        LOG_DEBUG("Handling file drop with {} files", paths.size());

        std::vector<std::filesystem::path> splat_files;
        std::optional<std::filesystem::path> dataset_path;

        for (const auto& path_str : paths) {
            std::filesystem::path filepath(path_str);
            LOG_TRACE("Processing dropped file: {}", filepath.string());

            auto ext = filepath.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".ply" || ext == ".sog") {
                splat_files.push_back(filepath);
            } else if (!dataset_path && std::filesystem::is_directory(filepath)) {
                // Check for dataset markers
                LOG_TRACE("Checking directory for dataset markers: {}", filepath.string());
                if (gs::loader::Loader::isDatasetPath(filepath)) {
                    dataset_path = filepath;
                    LOG_DEBUG("Dataset detected in dropped directory");
                }
            }
        }

        // Load splat files (PLY or SOG)
        for (const auto& splat : splat_files) {
            events::cmd::LoadFile{.path = splat, .is_dataset = false}.emit();
            LOG_INFO("Loading {} via drag-and-drop: {}",
                     splat.extension().string(), splat.filename().string());
        }

        // Load dataset if found
        if (dataset_path) {
            events::cmd::LoadFile{.path = *dataset_path, .is_dataset = true}.emit();
            LOG_INFO("Loading dataset via drag-and-drop: {}", dataset_path->filename().string());
        }

        if (paths.size() == 1) {
            auto project_path = std::filesystem::path(paths[0]);
            if (project_path.extension() == gs::management::Project::EXTENSION) {
                events::cmd::LoadProject{.path = project_path}.emit();
                LOG_INFO("Loading LS Project via drag-and-drop: {}", project_path.filename().string());
            }
        }
    }

    void InputController::handleGoToCamView(const events::cmd::GoToCamView& event) {
        LOG_TIMER_TRACE("HandleGoToCamView");

        if (!training_manager_) {
            LOG_ERROR("GoToCamView: trainer_manager_ not initialized");
            return;
        }

        auto cam_data = training_manager_->getCamById(event.cam_id);
        if (!cam_data) {
            LOG_ERROR("Camera ID {} not found", event.cam_id);
            return;
        }

        LOG_DEBUG("Moving camera to view ID: {} ({})", event.cam_id, cam_data->image_name());

        // Transform from WorldToCam to CamToWorld
        glm::mat3 world_to_cam_R;
        auto R_accessor = cam_data->R().accessor<float, 2>();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                world_to_cam_R[j][i] = R_accessor[i][j];
            }
        }

        auto T_accessor = cam_data->T().accessor<float, 1>();
        glm::vec3 world_to_cam_T(T_accessor[0], T_accessor[1], T_accessor[2]);

        glm::mat3 cam_to_world_R = glm::transpose(world_to_cam_R);
        glm::vec3 cam_to_world_T = -cam_to_world_R * world_to_cam_T;

        // Apply world transform to sync camera with transformed gaussian splat
        if (rendering_manager_) {
            auto settings = rendering_manager_->getSettings();
            if (!settings.world_transform.isIdentity()) {
                glm::mat4 world_transform = settings.world_transform.toMat4();

                // Build camera matrix using GLM constructor
                glm::mat4 cam_mat(
                    glm::vec4(cam_to_world_R[0], 0),
                    glm::vec4(cam_to_world_R[1], 0),
                    glm::vec4(cam_to_world_R[2], 0),
                    glm::vec4(cam_to_world_T, 1)
                );

                // Apply world transform
                glm::mat4 transformed_cam = world_transform * cam_mat;

                // Extract transformed rotation and translation
                cam_to_world_R = glm::mat3(transformed_cam);
                cam_to_world_T = glm::vec3(transformed_cam[3]);

                LOG_DEBUG("Applied world transform to camera view for 'go to image'");
            }
        }

        viewport_.camera.R = cam_to_world_R;
        viewport_.camera.t = cam_to_world_T;

        // Get camera intrinsics using the proper method
        auto [focal_x, focal_y, center_x, center_y] = cam_data->get_intrinsics();
        const float width = static_cast<float>(cam_data->image_width());
        const float height = static_cast<float>(cam_data->image_height());

        // Calculate vertical FOV using the actual focal length
        const float fov_y_rad = 2.0f * std::atan(height / (2.0f * focal_y));
        const float fov_y_deg = glm::degrees(fov_y_rad);

        LOG_DEBUG("Camera params - focal: ({:.1f}, {:.1f}), center: ({:.1f}, {:.1f}), image: {}x{}, FOV: {:.2f}°",
                  focal_x, focal_y, center_x, center_y, width, height, fov_y_deg);

        // Check for principal point offset (should be near center)
        const float cx_expected = width / 2.0f;
        const float cy_expected = height / 2.0f;

        if (std::abs(center_x - cx_expected) > 1.0f || std::abs(center_y - cy_expected) > 1.0f) {
            LOG_WARN("Camera has non-centered principal point: ({:.1f}, {:.1f}) vs expected ({:.1f}, {:.1f})",
                     center_x, center_y, cx_expected, cy_expected);
        }

        // Set the FOV
        events::ui::RenderSettingsChanged{
            .sh_degree = std::nullopt,
            .fov = fov_y_deg,
            .scaling_modifier = std::nullopt,
            .antialiasing = std::nullopt,
            .background_color = std::nullopt,
            .equirectangular = std::nullopt}
            .emit();

        // Force immediate camera update
        events::ui::CameraMove{
            .rotation = viewport_.getRotationMatrix(),
            .translation = viewport_.getTranslation()}
            .emit();

        // Set this as the current camera for GT comparison
        if (rendering_manager_) {
            rendering_manager_->setCurrentCameraId(event.cam_id);
        }

        last_camview = event.cam_id;
    }

    // Helpers
    bool InputController::isInViewport(double x, double y) const {
        return x >= viewport_bounds_.x &&
               x < viewport_bounds_.x + viewport_bounds_.width &&
               y >= viewport_bounds_.y &&
               y < viewport_bounds_.y + viewport_bounds_.height;
    }

    bool InputController::shouldCameraHandleInput() const {
        // Don't handle if gizmo or splitter is active
        if (drag_mode_ == DragMode::Gizmo || drag_mode_ == DragMode::Splitter) {
            return false;
        }

        // Special handling for point cloud mode
        if (point_cloud_mode_) {
            double x, y;
            glfwGetCursorPos(window_, &x, &y);
            if (isInViewport(x, y) && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
                return true; // Force handle in point cloud mode
            }
        }

        return !ImGui::GetIO().WantCaptureKeyboard &&
               !ImGui::GetIO().WantCaptureMouse;
    }

    void InputController::updateCameraSpeed(bool increase) {
        if (increase) {
            viewport_.camera.increaseWasdSpeed();
        } else {
            viewport_.camera.decreaseWasdSpeed();
        }

        const float new_speed = viewport_.camera.getWasdSpeed();
        const float max_speed = viewport_.camera.getMaxWasdSpeed();

        LOG_DEBUG("Camera speed changed to: {:.3f} (max: {:.3f})", new_speed, max_speed);

        events::ui::SpeedChanged{
            .current_speed = new_speed,
            .max_speed = max_speed}
            .emit();
    }

    void InputController::publishCameraMove() {
        auto now = std::chrono::steady_clock::now();
        if (now - last_camera_publish_ >= camera_publish_interval_) {
            events::ui::CameraMove{
                .rotation = viewport_.getRotationMatrix(),
                .translation = viewport_.getTranslation()}
                .emit();
            last_camera_publish_ = now;
        }
    }
} // namespace gs::visualizer